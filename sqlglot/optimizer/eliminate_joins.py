from __future__ import annotations

import typing as t

from sqlglot import expressions as exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import Scope, traverse_scope

if t.TYPE_CHECKING:
    from sqlglot._typing import E


def eliminate_joins(
    expression: E,
    constraints: dict[str, dict[str, t.Any]] | None = None,
    trusted_tables: t.Collection[str] | None = None,
) -> E:
    """
    Remove unused joins from an expression.

    This only removes joins when we know that the join condition doesn't produce duplicate rows.

    Example:
        >>> import sqlglot
        >>> sql = "SELECT x.a FROM x LEFT JOIN (SELECT DISTINCT y.b FROM y) AS y ON x.b = y.b"
        >>> expression = sqlglot.parse_one(sql)
        >>> eliminate_joins(expression).sql()
        'SELECT x.a FROM x'

    Joins on base tables can also be removed when the caller supplies constraint
    metadata and explicitly marks the involved tables as trusted, i.e. the
    declared constraints are enforced by the database:

        - A LEFT JOIN is removed when the joined table's equi-join key covers a
          trusted primary or unique key, so each outer row matches at most one row.
        - An INNER JOIN is additionally removed only when every outer row is
          guaranteed to match: the outer key must be a trusted, non-nullable
          foreign key referencing the joined table, and the ON clause must not
          contain any additional predicates.

    The metadata is only used for the duration of this call; it is never stored
    on the expression or the schema, so it cannot leak into subsequent calls.

    Args:
        expression: expression to optimize
        constraints: optional mapping of table name to constraint metadata, e.g.
            {"y": {"primary_key": ["id"], "unique_keys": [["a", "b"]],
                   "foreign_keys": [{"columns": ["x_id"], "ref_table": "x",
                                     "ref_columns": ["id"]}],
                   "not_null": ["id", "x_id"]}}
            Names should match the normalized identifiers produced by qualify.
        trusted_tables: tables whose declared constraints are enforced by the
            database. Constraints declared for any other table are ignored.

    Returns:
        The optimized expression
    """
    constraints = constraints or {}
    trusted_tables = set(trusted_tables or ())

    for scope in traverse_scope(expression):
        joins: list[exp.Join] = scope.expression.args.get("joins", [])
        if not joins:
            continue

        # If any columns in this scope aren't qualified, it's hard to determine if a join isn't used.
        # It's probably possible to infer this from the outputs of derived tables.
        # But for now, let's just skip this rule.
        if scope.unqualified_columns:
            continue

        # Reverse the joins so we can remove chains of unused joins
        for join in reversed(joins):
            if join.is_semi_or_anti_join:
                continue

            alias = join.alias_or_name
            if _should_eliminate_join(scope, join, alias, constraints, trusted_tables):
                join.pop()
                scope.remove_source(alias)

    return expression


def _should_eliminate_join(
    scope: Scope,
    join: exp.Join,
    alias: str,
    constraints: dict[str, dict[str, t.Any]],
    trusted_tables: set[str],
) -> bool:
    inner_source = scope.sources.get(alias)
    if _join_is_used(scope, join, alias):
        return False

    if isinstance(inner_source, Scope):
        return (join.side == "LEFT" and _is_joined_on_all_unique_outputs(inner_source, join)) or (
            not join.args.get("on") and _has_single_output_row(inner_source)
        )

    if isinstance(inner_source, exp.Table):
        return _should_eliminate_table_join(scope, join, inner_source, constraints, trusted_tables)

    return False


def _should_eliminate_table_join(
    scope: Scope,
    join: exp.Join,
    table: exp.Table,
    constraints: dict[str, dict[str, t.Any]],
    trusted_tables: set[str],
) -> bool:
    if table.name not in trusted_tables:
        return False

    table_constraints = constraints.get(table.name) or {}
    unique_keys = [
        table_constraints.get("primary_key") or (),
        *(table_constraints.get("unique_keys") or ()),
    ]
    unique_keys = [frozenset(key) for key in unique_keys if key]
    if not unique_keys:
        return False

    source_keys, join_keys, remaining = join_condition(join)

    # Map each joined column to the source column it's equated to
    joined_columns = {
        join_key.name: source_key
        for source_key, join_key in zip(source_keys, join_keys)
        if isinstance(source_key, exp.Column) and isinstance(join_key, exp.Column)
    }

    # The equi-join key must cover a trusted unique key, so each outer row
    # matches at most one joined row
    if not any(key <= joined_columns.keys() for key in unique_keys):
        return False

    if join.side == "LEFT":
        return True

    # INNER JOINs are only removed when every outer row is guaranteed to match,
    # so the ON clause must not carry any additional predicates
    if join.side or join.kind not in ("", "INNER") or not _is_true(remaining):
        return False

    return _every_outer_row_matches(scope, table, constraints, trusted_tables, joined_columns)


def _every_outer_row_matches(
    scope: Scope,
    table: exp.Table,
    constraints: dict[str, dict[str, t.Any]],
    trusted_tables: set[str],
    joined_columns: dict[str, exp.Column],
) -> bool:
    pairs = {(source.table, source.name): joined for joined, source in joined_columns.items()}

    for alias, source in scope.sources.items():
        if not isinstance(source, exp.Table) or source.name not in trusted_tables:
            continue

        source_constraints = constraints.get(source.name) or {}
        not_null = set(source_constraints.get("not_null") or ())

        for foreign_key in source_constraints.get("foreign_keys") or ():
            fk_columns = foreign_key.get("columns") or ()
            ref_columns = foreign_key.get("ref_columns") or ()

            # A trusted foreign key with non-nullable columns guarantees that
            # every outer row matches a row in the referenced table
            if (
                foreign_key.get("ref_table") == table.name
                and fk_columns
                and len(fk_columns) == len(ref_columns)
                and not_null >= set(fk_columns)
                and all(
                    pairs.get((alias, fk_column)) == ref_column
                    for fk_column, ref_column in zip(fk_columns, ref_columns)
                )
            ):
                return True

    return False


def _is_true(expression: exp.Expr) -> bool:
    if isinstance(expression, exp.And):
        return all(_is_true(operand) for operand in expression.flatten())
    return isinstance(expression, exp.Boolean) and expression.this


def _join_is_used(scope: Scope, join: exp.Join, alias: str) -> bool:
    # We need to find all columns that reference this join.
    # But columns in the ON clause shouldn't count.
    on: exp.Expr | None = join.args.get("on")
    if on is not None:
        on_clause_columns: set[int] = {id(column) for column in on.find_all(exp.Column)}
    else:
        on_clause_columns = set()
    return any(
        column for column in scope.source_columns(alias) if id(column) not in on_clause_columns
    )


def _is_joined_on_all_unique_outputs(scope: Scope, join: exp.Join) -> bool:
    unique_outputs = _unique_outputs(scope)
    if not unique_outputs:
        return False

    _, join_keys, _ = join_condition(join)
    remaining_unique_outputs = unique_outputs - {c.name for c in join_keys}
    return not remaining_unique_outputs


def _unique_outputs(scope: Scope) -> set[str]:
    """Determine output columns of `scope` that must have a unique combination per row"""
    expr = t.cast(exp.Query, scope.expression)
    if expr.args.get("distinct") is not None:
        return set(expr.named_selects)

    group: exp.Group | None = expr.args.get("group")
    if group is not None:
        grouped_expressions: set[exp.Expr] = set(group.expressions)
        grouped_outputs: set[exp.Expr] = set()

        unique_outputs: set[str] = set()
        for select in expr.selects:
            output = select.unalias()
            if output in grouped_expressions:
                grouped_outputs.add(output)
                unique_outputs.add(select.alias_or_name)

        # All the grouped expressions must be in the output
        if not grouped_expressions.difference(grouped_outputs):
            return unique_outputs
        else:
            return set()

    if _has_single_output_row(scope):
        return set(expr.named_selects)

    return set()


def _has_single_output_row(scope: Scope) -> bool:
    return isinstance(scope.expression, exp.Select) and (
        all(isinstance(e.unalias(), exp.AggFunc) for e in scope.expression.selects)
        or _is_limit_1(scope)
        or not scope.expression.args.get("from_")
    )


def _is_limit_1(scope: Scope) -> bool:
    limit: exp.Limit | None = scope.expression.args.get("limit")
    return limit is not None and limit.expression.this == "1"


def join_condition(join: exp.Join) -> tuple[list[exp.Expr], list[exp.Expr], exp.Expr]:
    """
    Extract the join condition from a join expression.

    Args:
        join (exp.Join)
    Returns:
        tuple[list[str], list[str], exp.Expr]:
            Tuple of (source key, join key, remaining predicate)
    """
    name = join.alias_or_name
    on: exp.Expr = (join.args.get("on") or exp.true()).copy()
    source_key: list[exp.Expr] = []
    join_key: list[exp.Expr] = []

    def extract_condition(condition: exp.Expr) -> None:
        left, right = condition.unnest_operands()
        left_tables = exp.column_table_names(left)
        right_tables = exp.column_table_names(right)

        if name in left_tables and name not in right_tables:
            join_key.append(left)
            source_key.append(right)
            condition.replace(exp.true())
        elif name in right_tables and name not in left_tables:
            join_key.append(right)
            source_key.append(left)
            condition.replace(exp.true())

    # find the join keys
    # SELECT
    # FROM x
    # JOIN y
    #   ON x.a = y.b AND y.b > 1
    #
    # should pull y.b as the join key and x.a as the source key
    if normalized(on):
        on = on if isinstance(on, exp.And) else exp.and_(on, exp.true(), copy=False)

        for condition in on.flatten():
            if isinstance(condition, exp.EQ):
                extract_condition(condition)
    elif normalized(on, dnf=True):
        conditions: list[exp.EQ] = []

        for condition in on.flatten():
            parts = [part for part in condition.flatten() if isinstance(part, exp.EQ)]
            if not conditions:
                conditions = parts
            else:
                temp: list[exp.EQ] = []
                for p in parts:
                    cs = [c for c in conditions if p == c]

                    if cs:
                        temp.append(p)
                        temp.extend(cs)
                conditions = temp

        for condition in conditions:
            extract_condition(condition)

    return source_key, join_key, on
