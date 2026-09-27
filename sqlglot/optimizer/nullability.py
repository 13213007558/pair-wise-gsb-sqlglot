from __future__ import annotations

import typing as t
from dataclasses import dataclass
from enum import Enum

from sqlglot import exp, parse_one
from sqlglot.optimizer.resolver import Resolver
from sqlglot.optimizer.scope import Scope, build_scope
from sqlglot.schema import MappingSchema

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType


class Nullability(Enum):
    """Guaranteed nullability of an output column, in increasing order of certainty."""

    NOT_NULL = "not_null"
    NULLABLE = "nullable"
    UNKNOWN = "unknown"


@dataclass
class ColumnNullability:
    """
    Nullability of a single projected column.

    Attributes:
        name: Output name of the column.
        nullability: NOT_NULL if NULL is impossible, NULLABLE if NULL is
            possible and UNKNOWN if there isn't enough information to decide.
        reason: Short human-readable explanation of how nullability was derived.
    """

    name: str
    nullability: Nullability
    reason: str


ColumnMapping = dict[str, bool | str | Nullability]
NullabilityMapping = dict[str, ColumnMapping]
FunctionContracts = dict[str, bool | str | Nullability]
Result = tuple[Nullability, str]


def _coerce_nullability(value: bool | str | Nullability) -> Nullability:
    if isinstance(value, Nullability):
        return value
    if isinstance(value, bool):
        return Nullability.NULLABLE if value else Nullability.NOT_NULL

    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in ("nullable", "null", "can_be_null"):
        return Nullability.NULLABLE
    if normalized in ("not_null", "non_null", "not_nullable"):
        return Nullability.NOT_NULL
    return Nullability.UNKNOWN


def _combine(results: t.Iterable[Result]) -> Result:
    """Combine per-branch/per-operand nullability: NULLABLE wins, then UNKNOWN."""
    results = list(results)
    nullable = [reason for status, reason in results if status is Nullability.NULLABLE]
    if nullable:
        return Nullability.NULLABLE, nullable[0]
    unknown = [reason for status, reason in results if status is Nullability.UNKNOWN]
    if unknown:
        return Nullability.UNKNOWN, unknown[0]
    return Nullability.NOT_NULL, results[0][1] if results else "NOT NULL"


def infer_nullability(
    expression: str | exp.Expr,
    nullability: NullabilityMapping | None = None,
    function_contracts: FunctionContracts | None = None,
    dialect: DialectType = None,
) -> list[ColumnNullability]:
    """
    Infers the nullability of each projected column in a query.

    This is intentionally independent of annotate_types: it never reads the
    type annotations attached to expressions.

    Args:
        expression: The query to analyze, as a string or parsed expression.
        nullability: Caller-provided nullability of base table columns, mapping
            {table: {column: ...}}. Values may be False/"not_null" (NOT NULL),
            True/"nullable" or "unknown". Missing tables or columns default to
            UNKNOWN rather than being assumed NOT NULL.
        function_contracts: Explicit nullability contracts for user functions,
            mapping an upper-case function name to the same accepted values as
            column nullability. Unlisted functions are reported as UNKNOWN.
        dialect: The dialect to use when parsing/generating identifiers.

    Returns:
        One ColumnNullability per output column, in output order. Star and
        qualified star (t.*) are expanded using the available sources.

    Example:
        >>> from sqlglot.optimizer.nullability import Nullability, infer_nullability
        >>> nullability = {"users": {"id": False, "name": True}}
        >>> results = infer_nullability("SELECT id, name FROM users", nullability)
        >>> [(r.name, r.nullability.value) for r in results]
        [('id', 'not_null'), ('name', 'nullable')]
    """
    if isinstance(expression, str):
        expression = parse_one(expression, dialect=dialect)

    if not isinstance(expression, (exp.Select, exp.SetOperation)):
        raise ValueError(f"Cannot infer output nullability from: {expression}")

    return _NullabilityInferencer(nullability or {}, function_contracts or {}, dialect).analyze(
        expression
    )


@dataclass
class _ScopeContext:
    scope: Scope
    null_supplied: set[str]
    where_not_null: set[tuple[str, str]]
    inferencer: _NullabilityInferencer


class _NullabilityInferencer:
    def __init__(
        self,
        column_nullability: NullabilityMapping,
        function_contracts: FunctionContracts,
        dialect: DialectType,
    ) -> None:
        self.column_nullability = {
            table: {column: _coerce_nullability(value) for column, value in columns.items()}
            for table, columns in column_nullability.items()
        }
        self.function_contracts = {
            name.upper(): _coerce_nullability(value) for name, value in function_contracts.items()
        }
        self.dialect = dialect
        # Only used to disambiguate unqualified columns; the dummy types carry no meaning.
        self.schema = MappingSchema(
            {
                table: {column: "INT" for column in columns}
                for table, columns in column_nullability.items()
            },
            dialect=dialect,
        )
        self._cache: dict[int, list[ColumnNullability]] = {}

    def analyze(self, root: exp.Expr) -> list[ColumnNullability]:
        scope = build_scope(root)
        if scope is None:
            raise ValueError(f"Cannot build scope for: {root}")
        return self._query_columns(scope)

    # ------------------------------------------------------------------
    # Scope-level analysis
    # ------------------------------------------------------------------

    def _query_columns(self, scope: Scope) -> list[ColumnNullability]:
        cached = self._cache.get(id(scope))
        if cached is not None:
            return cached

        query = scope.expression
        if isinstance(query, exp.SetOperation):
            result = self._set_operation_columns(scope)
        elif isinstance(query, exp.Select):
            result = self._select_columns(scope)
        else:
            raise ValueError(f"Cannot infer output nullability from: {query}")

        if scope.outer_columns and len(scope.outer_columns) == len(result):
            result = [
                ColumnNullability(name, column.nullability, column.reason)
                for name, column in zip(scope.outer_columns, result)
            ]

        self._cache[id(scope)] = result
        return result

    def _set_operation_columns(self, scope: Scope) -> list[ColumnNullability]:
        branches = [self._query_columns(branch) for branch in scope.set_operation_scopes]
        if not branches:
            return []

        if not isinstance(scope.expression, exp.Union):
            operation = type(scope.expression).__name__.upper()
            return [
                ColumnNullability(
                    column.name,
                    Nullability.UNKNOWN,
                    f"unsupported {operation} result, no nullability proof",
                )
                for column in branches[0]
            ]

        suffix = "UNION ALL" if scope.expression.args.get("all") else "UNION"
        width = max((len(branch) for branch in branches), default=0)
        result: list[ColumnNullability] = []
        for index in range(width):
            columns = [branch[index] for branch in branches if index < len(branch)]
            if len(columns) < len(branches) or not columns:
                result.append(
                    ColumnNullability(
                        branches[0][index].name if index < len(branches[0]) else f"col{index}",
                        Nullability.UNKNOWN,
                        f"{suffix} branches disagree on column count",
                    )
                )
                continue

            statuses = [(column.nullability, column.reason) for column in columns]
            status, _ = _combine(statuses)
            if status is Nullability.NOT_NULL:
                reason = f"all {suffix} branches are NOT NULL"
            else:
                nullable_branch = next(
                    i for i, (column_status, _) in enumerate(statuses, 1) if column_status is status
                )
                reason = f"{suffix} branch {nullable_branch} is {status.value}"
            result.append(ColumnNullability(columns[0].name, status, reason))
        return result

    def _select_columns(self, scope: Scope) -> list[ColumnNullability]:
        context = _ScopeContext(
            scope,
            self._null_supplied_sources(scope),
            self._where_not_null(scope),
            self,
        )

        result: list[ColumnNullability] = []
        select = t.cast(exp.Select, scope.expression)
        for projection in select.selects:
            result.extend(self._projection_columns(t.cast(exp.Expression, projection), context))
        return result

    def _null_supplied_sources(self, scope: Scope) -> set[str]:
        null_supplied: set[str] = set()
        from_ = scope.expression.args.get("from_")
        if not from_ or not from_.this:
            return null_supplied

        seen = {from_.this.alias_or_name}
        for join in scope.expression.args.get("joins") or []:
            joined = join.this.alias_or_name
            side = (join.side or "").upper()
            if side == "LEFT":
                null_supplied.add(joined)
            elif side == "RIGHT":
                null_supplied |= seen
            elif side == "FULL":
                null_supplied |= seen | {joined}
            seen.add(joined)
        return null_supplied

    def _where_not_null(self, scope: Scope) -> set[tuple[str, str]]:
        where = scope.expression.args.get("where")
        if not where:
            return set()

        predicates = where.this.flatten() if isinstance(where.this, exp.And) else [where.this]
        proven: set[tuple[str, str]] = set()
        resolver = Resolver(scope, self.schema)
        for predicate in predicates:
            if (
                isinstance(predicate, exp.Not)
                and isinstance(predicate.this, exp.Is)
                and isinstance(predicate.this.expression, exp.Null)
                and isinstance(predicate.this.this, exp.Column)
            ):
                column = predicate.this.this
                source = self._resolve_source(column, scope, resolver)
                if source:
                    proven.add((source, column.name))
        return proven

    def _projection_columns(
        self, projection: exp.Expression, context: _ScopeContext
    ) -> list[ColumnNullability]:
        scope = context.scope

        if isinstance(projection, exp.Star):
            return self._star_columns(scope, None, context)
        if isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
            source = projection.table or None
            return self._star_columns(scope, source, context)

        if isinstance(projection, exp.Alias):
            status, reason = self._infer(projection.this, context)
            return [ColumnNullability(projection.alias, status, reason)]

        status, reason = self._infer(projection, context)
        name = projection.name if isinstance(projection, exp.Column) else projection.alias_or_name
        if not name or name == "*":
            name = projection.sql()
        return [ColumnNullability(name, status, reason)]

    def _star_columns(
        self, scope: Scope, only_source: str | None, context: _ScopeContext
    ) -> list[ColumnNullability]:
        source_names = [only_source] if only_source else list(scope.selected_sources)
        result: list[ColumnNullability] = []
        for source_name in source_names:
            if source_name not in scope.sources:
                result.append(
                    ColumnNullability(
                        "*",
                        Nullability.UNKNOWN,
                        f"cannot resolve source {source_name} for star expansion",
                    )
                )
                continue
            for column_name in self._source_column_names(scope, source_name):
                status, reason = self._source_column_nullability(
                    scope, source_name, column_name, context
                )
                result.append(ColumnNullability(column_name, status, reason))
        return result

    # ------------------------------------------------------------------
    # Source (table/derived table/CTE) column resolution
    # ------------------------------------------------------------------

    def _source_column_names(self, scope: Scope, source_name: str) -> list[str]:
        source = scope.sources[source_name]
        if isinstance(source, Scope):
            return [column.name for column in self._query_columns(source)]
        return self._table_columns(source)

    def _table_columns(self, table: exp.Table) -> list[str]:
        full_name = ".".join(part.name for part in table.parts)
        for key in (full_name, table.name):
            if key in self.column_nullability:
                return list(self.column_nullability[key])
        return []

    def _resolve_source(
        self,
        column: exp.Column,
        scope: Scope,
        resolver: Resolver | None = None,
    ) -> str | None:
        if column.table:
            return column.table if column.table in scope.sources else None

        candidates = [
            source_name
            for source_name in scope.sources
            if column.name in self._source_column_names(scope, source_name)
        ]
        if len(candidates) == 1:
            return candidates[0]

        resolver = resolver or Resolver(scope, self.schema)
        table = resolver.get_table(column)
        if table is None or table.name not in scope.sources:
            return None
        return table.name

    def _source_column_nullability(
        self,
        scope: Scope,
        source_name: str,
        column_name: str,
        context: _ScopeContext,
    ) -> Result:
        if (source_name, column_name) in context.where_not_null:
            return Nullability.NOT_NULL, f"filtered by WHERE {column_name} IS NOT NULL"

        source = scope.sources[source_name]
        if isinstance(source, Scope):
            for column in self._query_columns(source):
                if column.name == column_name:
                    status, reason = column.nullability, column.reason
                    break
            else:
                status, reason = Nullability.UNKNOWN, f"unknown column {source_name}.{column_name}"
        else:
            table_columns = self.column_nullability.get(
                ".".join(part.name for part in source.parts),
                self.column_nullability.get(source.name),
            )
            if table_columns is None or column_name not in table_columns:
                status, reason = (
                    Nullability.UNKNOWN,
                    f"no nullability info for {source_name}.{column_name}",
                )
            else:
                status = table_columns[column_name]
                reason = (
                    f"column {column_name} is NOT NULL"
                    if status is Nullability.NOT_NULL
                    else f"column {column_name} is nullable"
                )

        if status is Nullability.NOT_NULL and source_name in context.null_supplied:
            status, reason = (
                Nullability.NULLABLE,
                f"column {column_name} is from null-supplying side of an outer join",
            )
        return status, reason

    # ------------------------------------------------------------------
    # Expression-level inference
    # ------------------------------------------------------------------

    def _infer(self, expression: exp.Expression, context: _ScopeContext) -> Result:
        scope = context.scope

        if isinstance(expression, exp.Column):
            source = self._resolve_source(expression, scope)
            if source is None:
                return (
                    Nullability.UNKNOWN,
                    f"cannot resolve column {expression.name} to a source",
                )
            return self._source_column_nullability(scope, source, expression.name, context)

        if isinstance(expression, exp.Paren):
            return self._infer(expression.this, context)

        if isinstance(expression, (exp.Cast, exp.TryCast)):
            return self._infer(expression.this, context)

        if isinstance(expression, exp.Case):
            return self._case_nullability(expression, context)

        if isinstance(expression, exp.Coalesce):
            arguments = [expression.this, *expression.expressions]
            results = [self._infer(argument, context) for argument in arguments if argument]
            if any(status is Nullability.NOT_NULL for status, _ in results):
                return Nullability.NOT_NULL, "COALESCE has a NOT NULL argument"
            if results and all(status is Nullability.NULLABLE for status, _ in results):
                return Nullability.NULLABLE, "all COALESCE arguments are nullable"
            return Nullability.UNKNOWN, "COALESCE arguments have unknown nullability"

        if isinstance(expression, exp.Nullif):
            return Nullability.NULLABLE, "NULLIF returns NULL when its arguments are equal"

        if isinstance(expression, exp.Count):
            return Nullability.NOT_NULL, "COUNT returns 0 for empty input"

        if isinstance(expression, exp.Sum):
            return self._sum_nullability(expression, context)

        if isinstance(expression, exp.Null):
            return Nullability.NULLABLE, "NULL literal"

        if isinstance(expression, (exp.Literal, exp.Boolean)):
            return Nullability.NOT_NULL, "non-NULL literal"

        if isinstance(expression, exp.Neg):
            return self._infer(expression.this, context)

        if isinstance(expression, exp.Binary):
            operands = (expression.this, expression.expression)
            return _combine(
                self._infer(operand, context)
                for operand in operands
                if isinstance(operand, exp.Expression)
            )

        if isinstance(expression, exp.Func):
            name = (
                expression.name.upper()
                if isinstance(expression, exp.Anonymous)
                else expression.sql_name().upper()
            )
            contract = self.function_contracts.get(name)
            if contract is not None:
                return contract, f"function contract declares {name} as {contract.value}"
            return Nullability.UNKNOWN, f"no nullability contract for function {name}"

        if isinstance(expression, exp.Subquery):
            return Nullability.UNKNOWN, "scalar subquery result has no nullability proof"

        return Nullability.UNKNOWN, f"unsupported expression {type(expression).__name__}"

    def _case_nullability(self, expression: exp.Case, context: _ScopeContext) -> Result:
        branches = [if_.args["true"] for if_ in expression.args.get("ifs") or []]
        default = expression.args.get("default")
        if default is None:
            return Nullability.NULLABLE, "CASE has no ELSE, unmatched rows yield NULL"

        results = [self._infer(branch, context) for branch in branches]
        results.append(self._infer(default, context))
        return _combine(results)

    def _sum_nullability(self, expression: exp.Sum, context: _ScopeContext) -> Result:
        argument = expression.this
        if isinstance(argument, exp.Star):
            argument_status = Nullability.NOT_NULL
        else:
            argument_status, _ = self._infer(argument, context)

        if argument_status is Nullability.NULLABLE:
            return Nullability.NULLABLE, "SUM argument is nullable"
        if argument_status is Nullability.UNKNOWN:
            return Nullability.UNKNOWN, "SUM argument has unknown nullability"

        if context.scope.expression.args.get("group"):
            return Nullability.NOT_NULL, "SUM of NOT NULL values within a non-empty GROUP BY group"
        return Nullability.NULLABLE, "SUM without GROUP BY yields NULL for empty input"
