from __future__ import annotations

import typing as t

from sqlglot import exp
from sqlglot.errors import UnsupportedError
from sqlglot.optimizer.scope import traverse_scope

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType


def apply_table_filters(
    expression: exp.Expr,
    filters: dict[str, str | exp.Expr],
    dialect: DialectType = None,
    copy: bool = True,
) -> exp.Expr:
    """
    Rewrites a query so that every scan of a filtered base table applies a caller-supplied predicate.

    Each matching table reference is wrapped in a derived table of the form
    (SELECT * FROM <table> WHERE <condition>) AS <alias>. Applying predicates per scan
    (instead of injecting them into an outer WHERE) preserves the unmatched side of
    LEFT/RIGHT/FULL joins. CTEs that shadow a base table name are resolved by scope,
    and correlated subqueries keep working because aliases are preserved.

    Args:
        expression: the query to rewrite. Only read queries (SELECT, UNION, CTEs, subqueries)
            are supported.
        filters: mapping of table identity (table, db.table or catalog.db.table) to a
            boolean condition, either as SQL text or an expression. Unqualified columns in a
            condition bind to the filtered table. Bind parameters (e.g. :cutoff or ?) are
            preserved as placeholders. A filter matches a table when its parts equal the
            trailing parts of the table identity; the most specific match wins.
        dialect: the dialect used to parse string conditions.
        copy: whether to copy the input expression before rewriting it.

    Returns:
        The rewritten expression.

    Raises:
        UnsupportedError: if the query contains DML statements, recursive CTEs or table
            functions, which cannot be rewritten safely.
    """
    if copy:
        expression = expression.copy()

    if not filters:
        return expression

    for dml in expression.find_all(exp.Insert, exp.Update, exp.Delete, exp.Merge):
        raise UnsupportedError(
            f"Cannot apply table filters to {type(dml).__name__.upper()} statements"
        )

    if any(with_.args.get("recursive") for with_ in expression.find_all(exp.With)):
        raise UnsupportedError("Cannot apply table filters to queries with recursive CTEs")

    parsed_filters: list[tuple[tuple[str, ...], exp.Expr]] = []
    for name, condition in filters.items():
        parsed = (
            exp.condition(condition, dialect=dialect) if isinstance(condition, str) else condition
        )
        if not isinstance(parsed, exp.Expr):
            raise TypeError(f"Invalid filter condition for table {name!r}: {condition!r}")
        parsed_filters.append((tuple(part.lower() for part in name.split(".")), parsed))

    parsed_filters.sort(key=lambda item: -len(item[0]))

    matched: list[tuple[exp.Table, exp.Expr]] = []
    for scope in traverse_scope(expression):
        if scope.udtf_scopes:
            raise UnsupportedError("Cannot apply table filters to queries with table functions")
        for source in scope.sources.values():
            if not isinstance(source, exp.Table):
                continue
            if not isinstance(source.this, exp.Identifier):
                raise UnsupportedError(
                    f"Cannot apply table filters to table function {source.sql(dialect=dialect)}"
                )
            parts = tuple(part.lower() for part in (source.catalog, source.db, source.name) if part)
            for key, condition in parsed_filters:
                if len(key) <= len(parts) and parts[-len(key) :] == key:
                    matched.append((source, condition))
                    break

    for table, condition in matched:
        _wrap_table(table, condition.copy())

    return expression


def _wrap_table(table: exp.Table, condition: exp.Expr) -> None:
    alias = table.args.get("alias") or exp.TableAlias(
        this=exp.to_identifier(table.name, quoted=table.this.quoted)
    )
    pivots = table.args.get("pivots")

    inner = table.copy()
    inner.set("alias", None)
    inner.set("pivots", None)

    replacement = exp.Subquery(this=exp.select("*").from_(inner).where(condition), alias=alias)
    if pivots:
        replacement.set("pivots", pivots)

    table.replace(replacement)
