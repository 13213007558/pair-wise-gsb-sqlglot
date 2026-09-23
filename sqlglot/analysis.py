"""Offline analysis for SQL scripts that mutate a catalog."""

from __future__ import annotations

import copy
import typing as t
from dataclasses import dataclass

from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import ErrorLevel, OptimizeError, SchemaError, UnsupportedError
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.qualify import qualify
from sqlglot.schema import MappingSchema, Schema, flatten_schema, nested_get, nested_set

if t.TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class TableName:
    catalog: str
    schema: str
    name: str
    temporary: bool = False


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: exp.DataType


@dataclass(frozen=True)
class CatalogChange:
    operation: str
    table: TableName
    columns: tuple[ColumnInfo, ...] = ()
    from_name: str = ""
    to_name: str = ""


@dataclass(frozen=True)
class CatalogSnapshot:
    persistent: dict[str, dict[str, dict[str, dict[str, exp.DataType]]]]
    temporary: dict[str, dict[str, dict[str, exp.DataType]]]

    @classmethod
    def _copy(
        cls,
        persistent: dict[str, dict[str, dict[str, dict[str, exp.DataType]]]],
        temporary: dict[str, dict[str, dict[str, exp.DataType]]],
    ) -> CatalogSnapshot:
        return cls(copy.deepcopy(persistent), copy.deepcopy(temporary))

    def tables(self) -> dict[TableName, tuple[ColumnInfo, ...]]:
        result: dict[TableName, tuple[ColumnInfo, ...]] = {}
        self._collect_tables(result, self.persistent, False)
        self._collect_tables(result, self.temporary, True)
        return result

    def _collect_tables(
        self,
        result: dict[TableName, tuple[ColumnInfo, ...]],
        catalogs: dict[str, t.Any],
        temporary: bool,
    ) -> None:
        for catalog_name, schemas in catalogs.items():
            for schema_name, tables in schemas.items():
                for table_name, columns in tables.items():
                    key = TableName(catalog_name, schema_name, table_name, temporary)
                    result[key] = tuple(
                        ColumnInfo(name, copy.deepcopy(dtype)) for name, dtype in columns.items()
                    )


@dataclass(frozen=True)
class ScriptDiagnostic:
    index: int
    error: Exception
    sql: str | None = None


@dataclass(frozen=True)
class ScriptStatement:
    index: int
    expression: exp.Expr
    output_columns: tuple[ColumnInfo, ...]
    changes: tuple[CatalogChange, ...]
    catalog_before: CatalogSnapshot
    catalog_after: CatalogSnapshot
    error: ScriptDiagnostic | None = None
    sql: str | None = None


@dataclass(frozen=True)
class ScriptAnalysis:
    statements: tuple[ScriptStatement, ...]
    diagnostics: tuple[ScriptDiagnostic, ...]
    catalog: CatalogSnapshot


PersistentCatalog = dict[str, dict[str, dict[str, dict[str, exp.DataType]]]]
TemporaryCatalog = dict[str, dict[str, dict[str, exp.DataType]]]


class OfflineCatalog(Schema):
    TEMP_SCHEMA = "pg_temp"

    def __init__(
        self,
        dialect: Dialect,
        persistent: PersistentCatalog,
        temporary: TemporaryCatalog,
        search_path: Sequence[str],
        default_catalog: str,
    ) -> None:
        self._dialect = dialect
        self.persistent = persistent
        self.temporary = temporary
        self.search_path = tuple(search_path)
        self.default_catalog = default_catalog

    @property
    def dialect(self) -> Dialect:
        return self._dialect

    @property
    def empty(self) -> bool:
        return not self.persistent and not self.temporary

    @property
    def supported_table_args(self) -> tuple[str, ...]:
        return ("this", "db", "catalog")

    @property
    def snapshot(self) -> CatalogSnapshot:
        return CatalogSnapshot._copy(self.persistent, self.temporary)

    def copy_state(self) -> tuple[PersistentCatalog, TemporaryCatalog]:
        return copy.deepcopy(self.persistent), copy.deepcopy(self.temporary)

    def with_state(
        self, persistent: PersistentCatalog, temporary: TemporaryCatalog
    ) -> OfflineCatalog:
        return OfflineCatalog(
            self.dialect,
            persistent,
            temporary,
            self.search_path,
            self.default_catalog,
        )

    def add_table(
        self,
        table: exp.Table | str,
        column_mapping: dict[str, t.Any] | str | list[str] | None = None,
        dialect: DialectType = None,
        normalize: bool | None = None,
        match_depth: bool = True,
    ) -> None:
        raise UnsupportedError("Use ScriptAnalyzer to mutate the offline catalog.")

    def normalize_identifier(self, identifier: str | exp.Identifier) -> str:
        if isinstance(identifier, str):
            identifier = exp.parse_identifier(identifier, dialect=self.dialect)
        return self.dialect.normalize_identifier(identifier.copy()).name

    def resolve(
        self,
        table: exp.Table,
        temporary: bool = False,
        include_temporary: bool = True,
        create: bool = False,
    ) -> TableName:
        name = self.normalize_identifier(table.this)
        schema_name = table.db and self.normalize_identifier(table.db)
        catalog_name = table.catalog and self.normalize_identifier(table.catalog)

        if temporary:
            if schema_name and schema_name != self.TEMP_SCHEMA:
                raise UnsupportedError("Temporary tables must use the pg_temp schema.")
            return TableName("", schema_name or self.TEMP_SCHEMA, name, True)
        if schema_name == self.TEMP_SCHEMA:
            return TableName(catalog_name or "", self.TEMP_SCHEMA, name, True)
        if schema_name:
            return TableName(catalog_name or self.default_catalog, schema_name, name)

        if create:
            if not self.search_path:
                raise SchemaError(
                    "A schema search path is required to resolve an unqualified table."
                )
            return TableName(self.default_catalog, self.search_path[0], name)

        if include_temporary:
            temporary_key = TableName("", self.TEMP_SCHEMA, name, True)
            if self.get_columns(temporary_key) is not None:
                return temporary_key

        for schema in self.search_path:
            key = TableName(self.default_catalog, schema, name)
            if self.get_columns(key) is not None:
                return key

        if not self.search_path:
            raise SchemaError("A schema search path is required to resolve an unqualified table.")
        return TableName(self.default_catalog, self.search_path[0], name)

    def column_names(
        self,
        table: exp.Table | str,
        only_visible: bool = False,
        dialect: DialectType = None,
        normalize: bool | None = None,
    ) -> list[str]:
        key = self.resolve(self._as_table(table))
        columns = self.get_columns(key)
        return list(columns) if columns is not None else []

    def get_column_type(
        self,
        table: exp.Table | str,
        column: exp.Column | str,
        dialect: DialectType = None,
        normalize: bool | None = None,
    ) -> exp.DataType:
        key = self.resolve(self._as_table(table))
        columns = self.get_columns(key) or {}
        column_name = column if isinstance(column, str) else column.name
        return copy.deepcopy(columns.get(column_name, exp.DType.UNKNOWN.into_expr()))

    def find(
        self, table: exp.Table, raise_on_missing: bool = True, ensure_data_types: bool = False
    ) -> dict[str, exp.DataType] | None:
        columns = self.get_columns(self.resolve(table))
        if columns is None and raise_on_missing:
            raise SchemaError(f"Table {table.sql(dialect=self.dialect)} was not found.")
        return copy.deepcopy(columns) if columns is not None else None

    def get_columns(self, key: TableName) -> dict[str, exp.DataType] | None:
        catalogs = self.temporary if key.temporary else self.persistent
        value: t.Any = catalogs.get(key.catalog, {}).get(key.schema, {}).get(key.name)
        return value

    def set_columns(
        self, catalogs: dict[str, t.Any], key: TableName, columns: dict[str, exp.DataType]
    ) -> None:
        catalogs.setdefault(key.catalog, {}).setdefault(key.schema, {})[key.name] = columns

    def remove_table(self, catalogs: dict[str, t.Any], key: TableName) -> None:
        catalogs.get(key.catalog, {}).get(key.schema, {}).pop(key.name, None)

    def _as_table(self, table: exp.Table | str) -> exp.Table:
        if isinstance(table, exp.Table):
            return table
        return exp.maybe_parse(table, into=exp.Table, dialect=self.dialect)


class ScriptAnalyzer:
    def __init__(
        self,
        schema: dict[str, object] | Schema | None = None,
        dialect: DialectType = "postgres",
        search_path: str | Iterable[str] = ("public",),
        default_catalog: str = "",
        error_level: ErrorLevel = ErrorLevel.IMMEDIATE,
    ) -> None:
        self.dialect = Dialect.get_or_raise(dialect)
        if self.dialect.__class__.__name__ != "Postgres":
            raise UnsupportedError("Only PostgreSQL script analysis is currently supported.")
        if error_level not in (ErrorLevel.IMMEDIATE, ErrorLevel.IGNORE):
            raise UnsupportedError("Only IMMEDIATE and IGNORE error levels are supported.")

        self.error_level = error_level
        self.search_path = self._normalize_search_path(search_path)
        self.default_catalog = self._normalize_name(default_catalog)
        persistent, temporary = self._initial_catalog(schema)
        self.catalog = OfflineCatalog(
            self.dialect,
            persistent,
            temporary,
            self.search_path,
            self.default_catalog,
        )

    def analyze(self, sql: str | Sequence[exp.Expr]) -> ScriptAnalysis:
        statements = (
            list(sql)
            if not isinstance(sql, str)
            else self.dialect.parse(sql, error_level=ErrorLevel.IGNORE)
        )

        results: list[ScriptStatement] = []
        diagnostics: list[ScriptDiagnostic] = []

        for index, statement in enumerate(statements):
            if statement is None:
                continue

            before = self.catalog.snapshot
            statement_sql = statement.sql(dialect=self.dialect)
            try:
                columns, changes, analyzed = self._analyze_statement(statement)
            except Exception as error:
                diagnostic = ScriptDiagnostic(index, error, statement_sql)
                diagnostics.append(diagnostic)
                results.append(
                    ScriptStatement(
                        index=index,
                        expression=statement,
                        output_columns=(),
                        changes=(),
                        catalog_before=before,
                        catalog_after=before,
                        error=diagnostic,
                        sql=statement_sql,
                    )
                )
                if self.error_level == ErrorLevel.IMMEDIATE:
                    raise
                continue

            results.append(
                ScriptStatement(
                    index=index,
                    expression=analyzed,
                    output_columns=tuple(columns),
                    changes=tuple(changes),
                    catalog_before=before,
                    catalog_after=self.catalog.snapshot,
                    sql=statement_sql,
                )
            )

        return ScriptAnalysis(tuple(results), tuple(diagnostics), self.catalog.snapshot)

    def _analyze_statement(
        self, statement: exp.Expr
    ) -> tuple[list[ColumnInfo], list[CatalogChange], exp.Expr]:
        if isinstance(statement, exp.Query):
            analyzed = self._qualify_query(statement)
            return self._output_columns(analyzed), [], analyzed
        if isinstance(statement, exp.Create) and statement.kind == "TABLE":
            return self._create_table(statement)
        if isinstance(statement, exp.Alter) and statement.kind == "TABLE":
            return self._alter_table(statement)
        if isinstance(statement, exp.Drop) and statement.kind == "TABLE":
            return self._drop_table(statement)

        raise UnsupportedError(f"Unsupported statement: {statement.sql(dialect=self.dialect)}")

    def _create_table(
        self, create: exp.Create
    ) -> tuple[list[ColumnInfo], list[CatalogChange], exp.Expr]:
        target = create.this
        target_table = target.this if isinstance(target, exp.Schema) else target
        if not isinstance(target_table, exp.Table):
            raise UnsupportedError("CREATE TABLE requires a table target.")

        temporary = self._is_temporary(create)
        key = self.catalog.resolve(
            target_table,
            temporary=temporary,
            include_temporary=temporary,
            create=not temporary,
        )
        existing = self.catalog.get_columns(key)

        if existing is not None and create.args.get("exists") and not create.args.get("replace"):
            return self._columns(key), [], create.copy()
        if existing is not None and not create.args.get("replace"):
            raise OptimizeError(f"Table {self._table_sql(key)} already exists.")

        query = create.expression
        analyzed = create.copy()

        if isinstance(query, exp.Query):
            analyzed.set("expression", self._qualify_query(query))
            columns = self._output_columns(analyzed.expression)
        elif isinstance(target, exp.Schema):
            columns = self._defined_columns(target.expressions)
        else:
            raise UnsupportedError("CREATE TABLE requires a column list or query.")

        if not columns:
            raise OptimizeError("Tables must have at least one output column.")

        persistent, temporary_catalog = self.catalog.copy_state()
        destination = temporary_catalog if key.temporary else persistent
        self.catalog.set_columns(
            destination,
            key,
            {column.name: copy.deepcopy(column.type) for column in columns},
        )
        self.catalog = self.catalog.with_state(persistent, temporary_catalog)

        operation = "REPLACE" if existing is not None else "CREATE"
        return columns, [CatalogChange(operation, key, tuple(columns))], analyzed

    def _alter_table(
        self, alter: exp.Alter
    ) -> tuple[list[ColumnInfo], list[CatalogChange], exp.Expr]:
        table = alter.this
        if not isinstance(table, exp.Table):
            raise UnsupportedError("ALTER TABLE requires a table target.")

        key = self.catalog.resolve(table)
        existing = self.catalog.get_columns(key)
        if existing is None:
            if alter.args.get("exists"):
                return [], [], alter.copy()
            raise OptimizeError(f"Table {self._table_sql(key)} was not found.")

        persistent, temporary = self.catalog.copy_state()
        columns: dict[str, exp.DataType] = copy.deepcopy(existing)
        changes: list[CatalogChange] = []

        for action in alter.actions:
            if isinstance(action, exp.ColumnDef):
                column = self._defined_columns([action])[0]
                if column.name in columns:
                    if action.args.get("exists"):
                        continue
                    raise OptimizeError(f"Column {column.name} already exists.")
                columns[column.name] = copy.deepcopy(column.type)
                changes.append(CatalogChange("ADD_COLUMN", key, (column,)))
            elif isinstance(action, exp.Drop) and action.kind == "COLUMN":
                for dropped in action.args.get("tables") or []:
                    if not isinstance(dropped, exp.Column):
                        raise UnsupportedError("Unsupported DROP COLUMN target.")
                    column_name = dropped.name
                    column_type = columns.get(column_name)
                    if column_type is None:
                        if action.args.get("exists"):
                            continue
                        raise OptimizeError(f"Column {column_name} was not found.")
                    columns.pop(column_name)
                    changes.append(
                        CatalogChange(
                            "DROP_COLUMN",
                            key,
                            (ColumnInfo(column_name, copy.deepcopy(column_type)),),
                        )
                    )
            elif isinstance(action, exp.RenameColumn):
                old_name = action.this.name
                new_column = action.args["to"]
                new_name = self._normalize_name(new_column.this)
                column_type = columns.get(old_name)
                if column_type is None:
                    if action.args.get("exists"):
                        continue
                    raise OptimizeError(f"Column {old_name} was not found.")
                if new_name in columns:
                    raise OptimizeError(f"Column {new_name} already exists.")
                columns.pop(old_name)
                columns[new_name] = copy.deepcopy(column_type)
                changes.append(
                    CatalogChange(
                        "RENAME_COLUMN",
                        key,
                        (ColumnInfo(new_name, copy.deepcopy(column_type)),),
                        old_name,
                        new_name,
                    )
                )
            else:
                raise UnsupportedError(
                    f"Unsupported ALTER TABLE action: {action.sql(dialect=self.dialect)}"
                )

        if not columns:
            raise OptimizeError("Tables must retain at least one column.")

        destination = temporary if key.temporary else persistent
        self.catalog.set_columns(destination, key, columns)
        self.catalog = self.catalog.with_state(persistent, temporary)
        return self._columns(key), changes, alter.copy()

    def _drop_table(self, drop: exp.Drop) -> tuple[list[ColumnInfo], list[CatalogChange], exp.Expr]:
        planned: list[tuple[TableName, list[ColumnInfo]]] = []

        for table in drop.args.get("tables") or []:
            if not isinstance(table, exp.Table):
                raise UnsupportedError("Unsupported DROP TABLE target.")

            key = self.catalog.resolve(table, temporary=bool(drop.args.get("temporary")))
            if self.catalog.get_columns(key) is None:
                if drop.args.get("exists"):
                    continue
                raise OptimizeError(f"Table {self._table_sql(key)} was not found.")
            planned.append((key, self._columns(key)))

        persistent, temporary = self.catalog.copy_state()
        changes: list[CatalogChange] = []
        output_columns: list[ColumnInfo] = []

        for key, columns in planned:
            destination = temporary if key.temporary else persistent
            self.catalog.remove_table(destination, key)
            changes.append(CatalogChange("DROP", key, tuple(columns)))
            output_columns.extend(columns)

        self.catalog = self.catalog.with_state(persistent, temporary)
        return output_columns, changes, drop.copy()

    def _qualify_query(self, query: exp.Query) -> exp.Query:
        qualified = qualify(
            query.copy(),
            dialect=self.dialect,
            schema=self.catalog,
            infer_schema=False,
        )
        return annotate_types(qualified, schema=self.catalog, dialect=self.dialect)

    def _output_columns(self, query: exp.Query) -> list[ColumnInfo]:
        result = []
        for selection in query.selects:
            name = selection.output_name
            if not name:
                raise OptimizeError("Failed to infer an output column name.")
            result.append(
                ColumnInfo(name, copy.deepcopy(selection.type or exp.DType.UNKNOWN.into_expr()))
            )
        return result

    def _defined_columns(self, definitions: list[exp.Expr]) -> list[ColumnInfo]:
        result = []
        names: set[str] = set()

        for definition in definitions:
            if not isinstance(definition, exp.ColumnDef):
                raise UnsupportedError(
                    f"Unsupported table element: {definition.sql(dialect=self.dialect)}"
                )

            dtype = definition.args.get("kind")
            if not isinstance(dtype, exp.DataType):
                raise UnsupportedError(
                    f"Column {definition.name} requires a data type: {definition.sql(self.dialect)}"
                )

            name = self._normalize_name(definition.this)
            if name in names:
                raise OptimizeError(f"Column {name} is defined more than once.")
            names.add(name)
            result.append(ColumnInfo(name, copy.deepcopy(dtype)))

        return result

    def _columns(self, key: TableName) -> list[ColumnInfo]:
        columns = self.catalog.get_columns(key)
        if columns is None:
            raise OptimizeError(f"Table {self._table_sql(key)} was not found.")
        return [ColumnInfo(name, copy.deepcopy(dtype)) for name, dtype in columns.items()]

    def _is_temporary(self, create: exp.Create) -> bool:
        properties = create.args.get("properties")
        return bool(properties and properties.find(exp.TemporaryProperty))

    def _table_sql(self, key: TableName) -> str:
        parts = [key.name]
        if key.schema:
            parts.append(key.schema)
        if key.catalog:
            parts.append(key.catalog)
        return ".".join(reversed(parts))

    def _initial_catalog(
        self, schema: dict[str, object] | Schema | None
    ) -> tuple[PersistentCatalog, TemporaryCatalog]:
        if isinstance(schema, OfflineCatalog):
            return schema.copy_state()
        if schema is not None and not isinstance(schema, (dict, MappingSchema)):
            raise UnsupportedError("Initial schema must be a mapping or MappingSchema.")

        mapping_schema = (
            schema
            if isinstance(schema, MappingSchema)
            else MappingSchema(schema or {}, dialect=self.dialect)
        )
        persistent: PersistentCatalog = {}
        depth = mapping_schema.depth()
        if depth > 3:
            raise SchemaError("Schema mappings may contain up to three table qualifiers.")

        for keys in flatten_schema(mapping_schema.mapping, depth=depth):
            initial_columns = nested_get(mapping_schema.mapping, *zip(keys, keys))
            if not isinstance(initial_columns, dict):
                raise SchemaError("Invalid schema mapping.")

            key: tuple[str, ...]
            if depth == 1:
                if not self.search_path:
                    raise SchemaError(
                        "A schema search path is required to resolve an unqualified table."
                    )
                key = (self.default_catalog, self.search_path[0], keys[0])
            elif depth == 2:
                key = (self.default_catalog, keys[0], keys[1])
            else:
                key = tuple(keys)

            columns = {
                column_name: self._data_type(column_type)
                for column_name, column_type in initial_columns.items()
            }
            nested_set(persistent, key, columns)

        return persistent, {}

    def _normalize_search_path(self, search_path: str | Iterable[str]) -> tuple[str, ...]:
        if isinstance(search_path, str):
            parts = search_path.split(",")
        else:
            parts = list(search_path)

        result = []
        for part in parts:
            name = part.strip()
            if name == "$user":
                continue
            if name:
                result.append(self._normalize_name(name))
        return tuple(result)

    def _normalize_name(self, name: str | exp.Identifier) -> str:
        return self.dialect.normalize_identifier(exp.to_identifier(name).copy()).name

    def _data_type(self, value: t.Any) -> exp.DataType:
        if isinstance(value, exp.DataType):
            return copy.deepcopy(value)
        if value is None:
            return exp.DType.UNKNOWN.into_expr()
        return exp.DataType.from_str(str(value), dialect=self.dialect)


def analyze_script(
    sql: str | Sequence[exp.Expr],
    schema: dict[str, object] | Schema | None = None,
    dialect: DialectType = "postgres",
    search_path: str | Iterable[str] = ("public",),
    default_catalog: str = "",
    error_level: ErrorLevel = ErrorLevel.IMMEDIATE,
) -> ScriptAnalysis:
    return ScriptAnalyzer(
        schema=schema,
        dialect=dialect,
        search_path=search_path,
        default_catalog=default_catalog,
        error_level=error_level,
    ).analyze(sql)
