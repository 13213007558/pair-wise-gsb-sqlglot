"""
Offline analysis of multi-statement SQL scripts.

This module provides a ScriptSession that tracks a session catalog
(persistent tables plus session-local temporary tables) while walking
through the statements of a script one by one. For every statement it
reports the output columns with their inferred types as well as the
catalog changes the statement causes, without connecting to a database.

The supported statement set is PostgreSQL-flavored:

- CREATE TABLE (with explicit column definitions)
- CREATE [TEMP] TABLE ... AS <query>
- ALTER TABLE ... ADD/DROP/RENAME COLUMN
- DROP TABLE
- queries (SELECT, WITH, set operations, ...)

Anything else is reported as "other" and leaves the catalog untouched.
"""

from __future__ import annotations

import copy
import typing as t
from dataclasses import dataclass, field

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.helper import dict_depth
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import MappingSchema, Schema

TEMP_SCHEMA = "pg_temp"
DEFAULT_SEARCH_PATH = ("public",)

UNKNOWN_TYPE = exp.DataType(this=exp.DataType.Type.UNKNOWN)


class _StatementError(Exception):
    """A statement failed to analyze; the catalog is left unchanged."""


@dataclass
class ColumnInfo:
    """An output or stored column together with its inferred type."""

    name: str
    type: exp.DataType


@dataclass
class CatalogChange:
    """A single catalog mutation caused by a statement."""

    action: str  # create_table | drop_table | add_column | drop_column | rename_column
    table: str
    temp: bool = False
    column: str | None = None
    new_name: str | None = None
    column_type: exp.DataType | None = None
    columns: list[ColumnInfo] = field(default_factory=list)


@dataclass
class CatalogSnapshot:
    """An immutable-ish view of the catalog at a point in the script."""

    tables: dict[str, dict[str, exp.DataType]] = field(default_factory=dict)
    temp_tables: dict[str, dict[str, exp.DataType]] = field(default_factory=dict)


@dataclass
class StatementResult:
    """The analysis outcome of a single statement."""

    index: int
    kind: str  # query | create_table | create_table_as | alter_table | drop_table | other
    sql: str
    ok: bool
    error: str | None
    output_columns: list[ColumnInfo]
    changes: list[CatalogChange]
    diagnostics: list[str]
    catalog_after: CatalogSnapshot
    statement: exp.Expr | None = None


@dataclass
class ScriptResult:
    """The analysis outcome of a whole script."""

    statements: list[StatementResult]
    catalog: CatalogSnapshot

    @property
    def ok(self) -> bool:
        return all(statement.ok for statement in self.statements)


class _Catalog:
    """Session catalog: persistent tables shadowed by session temp tables."""

    def __init__(self, search_path: tuple[str, ...]) -> None:
        self.persistent: dict[tuple[str, str], dict[str, exp.DataType]] = {}
        self.temp: dict[str, dict[str, exp.DataType]] = {}
        self.search_path = search_path

    def copy(self) -> _Catalog:
        clone = _Catalog(self.search_path)
        clone.persistent = copy.deepcopy(self.persistent)
        clone.temp = copy.deepcopy(self.temp)
        return clone

    def resolve(self, parts: list[str]) -> tuple[str, str, bool] | None:
        """
        Resolve normalized table name parts to (schema, name, is_temp).

        Unqualified names hit session temp tables first and then follow the
        search path; qualified names are looked up directly ("pg_temp" refers
        to the session's temp tables).
        """
        if len(parts) >= 2:
            schema, name = parts[-2], parts[-1]
            if schema == TEMP_SCHEMA and name in self.temp:
                return TEMP_SCHEMA, name, True
            if (schema, name) in self.persistent:
                return schema, name, False
            return None

        name = parts[-1]
        if name in self.temp:
            return TEMP_SCHEMA, name, True
        for schema in self.search_path:
            if (schema, name) in self.persistent:
                return schema, name, False
        return None

    def create_schema(self, parts: list[str]) -> str:
        """The schema a CREATE statement targets for the given name parts."""
        return parts[-2] if len(parts) >= 2 else self.search_path[0]

    def snapshot(self) -> CatalogSnapshot:
        return CatalogSnapshot(
            tables={
                f"{schema}.{name}": dict(cols) for (schema, name), cols in self.persistent.items()
            },
            temp_tables={name: dict(cols) for name, cols in self.temp.items()},
        )


def _display_name(schema: str, name: str) -> str:
    return f"{schema}.{name}"


class ScriptSession:
    """
    A stateful, offline script-analysis session.

    The session owns a private copy of the initial schema, so two sessions
    created from the same schema mapping evolve independently.

    Args:
        schema: the initial catalog, either a nested mapping in one of the
            MappingSchema forms ({table: {col: type}}, {db: ...},
            {catalog: {db: ...}}) or a MappingSchema instance.
        dialect: the SQL dialect used for parsing, identifier normalization
            (including quoted identifiers) and type inference.
        search_path: ordered schema names used to resolve unqualified table
            references; the first entry is the default CREATE target schema.
        on_error: "stop" halts the script at the first failed statement,
            "continue" records the diagnostic and keeps analyzing. A failed
            statement never modifies the catalog.
    """

    def __init__(
        self,
        schema: dict[str, object] | Schema | None = None,
        *,
        dialect: DialectType = "postgres",
        search_path: t.Sequence[str] | None = None,
        on_error: str = "stop",
    ) -> None:
        if on_error not in ("stop", "continue"):
            raise ValueError(f"Invalid on_error mode: {on_error!r}. Expected 'stop' or 'continue'.")

        self.dialect = Dialect.get_or_raise(dialect)
        self.on_error = on_error
        self.search_path = tuple(
            self._normalize_name(s) for s in (search_path or DEFAULT_SEARCH_PATH)
        )
        self._catalog = _Catalog(self.search_path)

        if schema is not None:
            self._load_schema(schema)

    def analyze(self, sql: str, *, on_error: str | None = None) -> ScriptResult:
        """
        Parse and analyze every statement in sql, advancing the session
        catalog. Returns per-statement results plus the final catalog.
        """
        mode = on_error or self.on_error
        if mode not in ("stop", "continue"):
            raise ValueError(f"Invalid on_error mode: {mode!r}. Expected 'stop' or 'continue'.")

        results = []
        statements = [s for s in self.dialect.parse(sql) if s is not None]

        for index, statement in enumerate(statements):
            result = self.analyze_statement(statement, index=index)
            results.append(result)
            if not result.ok and mode == "stop":
                break

        return ScriptResult(statements=results, catalog=self._catalog.snapshot())

    def analyze_statement(self, statement: exp.Expr, index: int = 0) -> StatementResult:
        """Analyze a single parsed statement against the session catalog."""
        sql = statement.sql(dialect=self.dialect)
        kind = "other"
        changes: list[CatalogChange] = []
        diagnostics: list[str] = []
        output_columns: list[ColumnInfo] = []
        error = None

        normalize_identifiers(statement, dialect=self.dialect)
        working = self._catalog.copy()

        try:
            if isinstance(statement, exp.Query):
                kind = "query"
                output_columns = self._analyze_query(statement, working)
            elif isinstance(statement, exp.Create) and statement.kind == "TABLE":
                kind = "create_table"
                kind = self._apply_create(statement, working, changes, diagnostics)
            elif isinstance(statement, exp.Alter) and statement.kind == "TABLE":
                kind = "alter_table"
                self._apply_alter(statement, working, changes, diagnostics)
            elif isinstance(statement, exp.Drop) and statement.kind == "TABLE":
                kind = "drop_table"
                self._apply_drop(statement, working, changes, diagnostics)
            else:
                kind = "other"
                diagnostics.append("Unsupported statement; catalog left unchanged.")
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            diagnostics.append(f"Statement failed: {error}")
            changes.clear()
        else:
            self._catalog = working

        return StatementResult(
            index=index,
            kind=kind,
            sql=sql,
            ok=error is None,
            error=error,
            output_columns=output_columns,
            changes=changes,
            diagnostics=diagnostics,
            catalog_after=self._catalog.snapshot(),
            statement=statement,
        )

    @property
    def catalog(self) -> CatalogSnapshot:
        """The current catalog of the session."""
        return self._catalog.snapshot()

    def _normalize_name(self, name: str) -> str:
        return normalize_identifiers(name, dialect=self.dialect).name

    def _load_schema(self, schema: dict[str, object] | Schema) -> None:
        if isinstance(schema, Schema):
            if not isinstance(schema, MappingSchema):
                raise ValueError(f"Unsupported schema type: {type(schema).__name__}")
            mapping = copy.deepcopy(schema.mapping)
            normalize = False
        else:
            mapping = copy.deepcopy(schema)
            normalize = True

        depth = dict_depth(mapping)
        if depth <= 1:
            return
        if depth > 4:
            raise ValueError(f"Invalid schema mapping shape. Depth: {depth}")

        def norm(name: str) -> str:
            return self._normalize_name(name) if normalize else name

        def columns_of(col_mapping: object) -> dict[str, exp.DataType]:
            if not isinstance(col_mapping, dict):
                raise ValueError(f"Invalid column mapping: {col_mapping!r}")
            return {norm(c): self._to_data_type(t) for c, t in col_mapping.items()}

        for level_one, value in mapping.items():
            if depth == 2:
                self._catalog.persistent[(self.search_path[0], norm(level_one))] = columns_of(value)
            elif depth == 3:
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid schema mapping: {value!r}")
                for table, cols in value.items():
                    self._catalog.persistent[(norm(level_one), norm(table))] = columns_of(cols)
            else:
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid schema mapping: {value!r}")
                for db, tables in value.items():
                    if not isinstance(tables, dict):
                        raise ValueError(f"Invalid schema mapping: {tables!r}")
                    for table, cols in tables.items():
                        self._catalog.persistent[(norm(db), norm(table))] = columns_of(cols)

    def _to_data_type(self, dtype: object) -> exp.DataType:
        if isinstance(dtype, exp.DataType):
            return dtype
        if isinstance(dtype, str):
            return exp.DataType.build(dtype, dialect=self.dialect)
        return UNKNOWN_TYPE

    def _mapping_schema(self, catalog: _Catalog) -> MappingSchema:
        mapping: dict[str, dict[str, dict[str, exp.DataType]]] = {}
        for (schema, table), columns in catalog.persistent.items():
            mapping.setdefault(schema, {})[table] = copy.deepcopy(columns)
        for name, columns in catalog.temp.items():
            mapping.setdefault(TEMP_SCHEMA, {})[name] = copy.deepcopy(columns)
        return MappingSchema(
            t.cast(dict[str, object], mapping), dialect=self.dialect, normalize=False
        )

    def _resolve_tables(self, query: exp.Expr, catalog: _Catalog) -> None:
        """
        Qualify unqualified table references against the catalog, honoring
        temp-table shadowing and the search path. CTE references are left
        alone so a CTE is never mistaken for a catalog table.
        """
        for scope in traverse_scope(query):
            for table in scope.tables:
                if not isinstance(table.this, exp.Identifier):
                    continue
                name = table.name
                if name in scope.cte_sources or isinstance(scope.sources.get(name), Scope):
                    continue
                parts = [part.name for part in table.parts]
                resolved = catalog.resolve(parts)
                if resolved is None:
                    raise _StatementError(f"Table {'.'.join(parts)} does not exist.")
                if not table.args.get("db"):
                    table.set("db", exp.to_identifier(resolved[0], quoted=True))

    def _analyze_query(self, query: exp.Query, catalog: _Catalog) -> list[ColumnInfo]:
        self._resolve_tables(query, catalog)
        schema = self._mapping_schema(catalog)
        qualified = qualify(query, dialect=self.dialect, schema=schema)
        annotated = annotate_types(qualified, schema=schema, dialect=self.dialect)

        return [
            ColumnInfo(
                name=projection.output_name or "*",
                type=projection.type or UNKNOWN_TYPE,
            )
            for projection in annotated.selects
        ]

    def _apply_create(
        self,
        statement: exp.Create,
        catalog: _Catalog,
        changes: list[CatalogChange],
        diagnostics: list[str],
    ) -> str:
        this = statement.this
        column_defs: list[exp.ColumnDef] = []

        if isinstance(this, exp.Schema):
            table = this.this
            column_defs = [e for e in this.expressions if isinstance(e, exp.ColumnDef)]
        elif isinstance(this, exp.Table):
            table = this
        else:
            raise _StatementError(
                f"Unsupported CREATE TABLE target: {this.sql(dialect=self.dialect)}"
            )

        if not isinstance(table, exp.Table):
            raise _StatementError("Cannot determine the target table of CREATE TABLE.")

        parts = [part.name for part in table.parts]
        name = parts[-1]
        query = statement.expression

        if query is not None:
            if not isinstance(query, exp.Query):
                raise _StatementError("Unsupported CREATE TABLE AS source.")
            kind = "create_table_as"
            columns = self._analyze_query(query, catalog)
            if column_defs:
                if len(column_defs) != len(columns):
                    raise _StatementError(
                        "CREATE TABLE AS column list does not match the query's output columns."
                    )
                columns = [
                    ColumnInfo(name=column_def.name, type=column.type)
                    for column_def, column in zip(column_defs, columns)
                ]
        else:
            kind = "create_table"
            if not column_defs:
                raise _StatementError("Cannot infer the columns of CREATE TABLE.")
            columns = [
                ColumnInfo(
                    name=column_def.name,
                    type=column_def.args.get("kind") or UNKNOWN_TYPE,
                )
                for column_def in column_defs
            ]

        column_names = [column.name for column in columns]
        if len(set(column_names)) != len(column_names):
            raise _StatementError(f"Duplicate column names in CREATE TABLE {name}.")

        temporary = self._is_temporary(statement)
        if temporary:
            exists = name in catalog.temp
            display = _display_name(TEMP_SCHEMA, name)
        else:
            schema = catalog.create_schema(parts)
            exists = (schema, name) in catalog.persistent
            display = _display_name(schema, name)

        if exists:
            if statement.args.get("replace"):
                diagnostics.append(f"Table {display} already exists; replacing.")
            elif statement.args.get("exists"):
                diagnostics.append(f"Table {display} already exists; statement skipped.")
                return kind
            else:
                raise _StatementError(f"Table {display} already exists.")

        column_mapping = {column.name: column.type for column in columns}
        if temporary:
            catalog.temp[name] = column_mapping
        else:
            catalog.persistent[(catalog.create_schema(parts), name)] = column_mapping

        changes.append(
            CatalogChange(action="create_table", table=display, temp=temporary, columns=columns)
        )
        return kind

    def _apply_alter(
        self,
        statement: exp.Alter,
        catalog: _Catalog,
        changes: list[CatalogChange],
        diagnostics: list[str],
    ) -> None:
        table = statement.this
        if not isinstance(table, exp.Table):
            raise _StatementError("Unsupported ALTER TABLE target.")

        parts = [part.name for part in table.parts]
        resolved = catalog.resolve(parts)
        display = ".".join(parts)

        if resolved is None:
            if statement.args.get("exists"):
                diagnostics.append(f"Table {display} does not exist; statement skipped.")
                return
            raise _StatementError(f"Table {display} does not exist.")

        schema, name, temporary = resolved
        display = _display_name(schema, name)
        current = catalog.temp[name] if temporary else catalog.persistent[(schema, name)]
        updated = dict(current)

        for action in statement.actions:
            if isinstance(action, exp.ColumnDef):
                column = action.name
                if column in updated:
                    if action.args.get("exists"):
                        diagnostics.append(f"Column {column} already exists on {display}; skipped.")
                        continue
                    raise _StatementError(f"Column {column} already exists on {display}.")
                column_type = action.args.get("kind") or UNKNOWN_TYPE
                updated[column] = column_type
                changes.append(
                    CatalogChange(
                        action="add_column",
                        table=display,
                        temp=temporary,
                        column=column,
                        column_type=column_type,
                    )
                )
            elif isinstance(action, exp.Drop) and action.kind == "COLUMN":
                for column_expr in action.args.get("tables") or []:
                    column = column_expr.name
                    if column not in updated:
                        if action.args.get("exists"):
                            diagnostics.append(
                                f"Column {column} does not exist on {display}; skipped."
                            )
                            continue
                        raise _StatementError(f"Column {column} does not exist on {display}.")
                    del updated[column]
                    changes.append(
                        CatalogChange(
                            action="drop_column", table=display, temp=temporary, column=column
                        )
                    )
            elif isinstance(action, exp.RenameColumn):
                old_name = action.this.name
                new_name = action.args["to"].name
                if old_name not in updated:
                    if action.args.get("exists"):
                        diagnostics.append(
                            f"Column {old_name} does not exist on {display}; skipped."
                        )
                        continue
                    raise _StatementError(f"Column {old_name} does not exist on {display}.")
                if new_name in updated:
                    raise _StatementError(f"Column {new_name} already exists on {display}.")
                updated = {new_name if c == old_name else c: t for c, t in updated.items()}
                changes.append(
                    CatalogChange(
                        action="rename_column",
                        table=display,
                        temp=temporary,
                        column=old_name,
                        new_name=new_name,
                    )
                )
            else:
                raise _StatementError(
                    f"Unsupported ALTER TABLE action: {action.sql(dialect=self.dialect)}"
                )

        if temporary:
            catalog.temp[name] = updated
        else:
            catalog.persistent[(schema, name)] = updated

    def _apply_drop(
        self,
        statement: exp.Drop,
        catalog: _Catalog,
        changes: list[CatalogChange],
        diagnostics: list[str],
    ) -> None:
        resolved_tables = []
        seen = set()

        for table in statement.args.get("tables") or []:
            parts = [part.name for part in table.parts]
            display = ".".join(parts)
            resolved = catalog.resolve(parts)

            if resolved is None:
                if statement.args.get("exists"):
                    diagnostics.append(f"Table {display} does not exist; skipped.")
                    continue
                raise _StatementError(f"Table {display} does not exist.")

            key = (resolved[0], resolved[1], resolved[2])
            if key not in seen:
                seen.add(key)
                resolved_tables.append(resolved)

        for schema, name, temporary in resolved_tables:
            display = _display_name(schema, name)
            if temporary:
                del catalog.temp[name]
            else:
                del catalog.persistent[(schema, name)]
            changes.append(CatalogChange(action="drop_table", table=display, temp=temporary))

    @staticmethod
    def _is_temporary(statement: exp.Create) -> bool:
        properties = statement.args.get("properties")
        if not isinstance(properties, exp.Properties):
            return False
        return any(isinstance(prop, exp.TemporaryProperty) for prop in properties.expressions)


def analyze_script(
    sql: str,
    schema: dict[str, object] | Schema | None = None,
    *,
    dialect: DialectType = "postgres",
    search_path: t.Sequence[str] | None = None,
    on_error: str = "stop",
) -> ScriptResult:
    """
    Analyze a SQL script offline, statement by statement.

    Args:
        sql: the script to analyze.
        schema: the initial catalog (see ScriptSession).
        dialect: the SQL dialect used for parsing and type inference.
        search_path: ordered schema names for unqualified table resolution.
        on_error: "stop" halts at the first failed statement, "continue"
            records the diagnostic and keeps going.

    Returns:
        A ScriptResult with one StatementResult per analyzed statement.
    """
    session = ScriptSession(schema, dialect=dialect, search_path=search_path, on_error=on_error)
    return session.analyze(sql)
