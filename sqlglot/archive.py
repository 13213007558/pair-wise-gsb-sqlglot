"""
Utilities for rewriting queries against archived tables.

Historical-data query services often need every read of an archived table to
carry an extra caller-supplied predicate (e.g. ``archived_at <= :cutoff``).
This module provides a public AST transformation entry point that applies such
predicates to every scan of a base table, keyed by the table's full identity.
"""

from __future__ import annotations

import typing as t

from sqlglot import exp
from sqlglot.errors import UnsupportedError
from sqlglot.optimizer.scope import traverse_scope

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType


class ArchiveFilterError(UnsupportedError):
    """Raised when archive filters cannot be applied to a query safely."""


def apply_archive_filters(
    expression: exp.Expr,
    filters: t.Mapping[str | exp.Table, exp.Expr | str],
    dialect: DialectType = None,
    copy: bool = True,
) -> exp.Expr:
    """
    Filters every scan of the given base tables by attaching a predicate to each scan.

    Each filter is applied by wrapping the matching base table scan in a derived
    table, e.g. ``FROM db.t x`` becomes ``FROM (SELECT * FROM db.t WHERE <condition>) AS x``.
    Because the predicate is applied at the scan site, outer joins keep their
    unmatched rows (unlike naively adding the predicate to the outermost WHERE),
    and correlated subqueries, CTEs, set operations and self-joins under
    different aliases are all handled per scan. A CTE or derived table that
    shadows a base table's name is left untouched, per the usual scoping rules.

    Filter keys are matched against the full identity of each table reference as
    written in the query, so ``"db.t"`` matches ``db.t`` but not ``t`` or
    ``other.t``. Columns in a condition bind to the filtered base table; bind
    parameters (e.g. ``:cutoff``) are preserved as-is.

    Example:
        >>> import sqlglot
        >>> sqlglot.apply_archive_filters(
        ...     sqlglot.parse_one("SELECT * FROM db.events e LEFT JOIN db.users u ON u.id = e.uid"),
        ...     {"db.events": "archived_at <= :cutoff"},
        ... ).sql()
        'SELECT * FROM (SELECT * FROM db.events WHERE archived_at <= :cutoff) AS e LEFT JOIN db.users AS u ON u.id = e.uid'

    Args:
        expression: the query to rewrite.
        filters: a mapping of fully-qualified table name to the condition that
            should filter every scan of that table. Conditions can be given as
            SQL strings or expressions.
        dialect: the dialect used to parse string table names and conditions.
        copy: whether to copy the input expression, leaving it unchanged.

    Returns:
        The rewritten query.

    Raises:
        ArchiveFilterError: if the query cannot be rewritten safely, i.e. it is
            a DML statement, contains a recursive CTE, or scans an unknown table
            function.
    """
    if expression.find(exp.DML):
        raise ArchiveFilterError("Archive filters cannot be applied to DML statements.")
    if not isinstance(expression, exp.Query):
        raise ArchiveFilterError(
            f"Archive filters can only be applied to queries, got {type(expression).__name__}."
        )
    if any(with_.args.get("recursive") for with_ in expression.find_all(exp.With)):
        raise ArchiveFilterError("Archive filters cannot be applied to recursive CTEs.")

    conditions: dict[tuple[str, ...], exp.Expr] = {}
    for table, condition in filters.items():
        table_expr = exp.to_table(table, dialect=dialect) if isinstance(table, str) else table
        condition_expr = (
            exp.condition(condition, dialect=dialect) if isinstance(condition, str) else condition
        )
        conditions[_table_key(table_expr)] = condition_expr

    if copy:
        expression = expression.copy()

    rewrites = []
    for scope in traverse_scope(expression):
        for source in scope.sources.values():
            if not isinstance(source, exp.Table):
                continue
            this = source.this
            if isinstance(this, exp.Anonymous):
                raise ArchiveFilterError(
                    f"Archive filters cannot be applied to unknown table function '{this.this}'."
                )
            if not isinstance(this, exp.Identifier):
                continue
            matched = conditions.get(_table_key(source))
            if matched is not None:
                rewrites.append((source, matched))

    for source, matched in rewrites:
        source.replace(_filtered_scan(source, matched))

    return expression


def _table_key(table: exp.Table) -> tuple[str, ...]:
    return tuple(
        identifier.name if identifier.quoted else identifier.name.lower()
        for identifier in (table.args.get(arg) for arg in ("catalog", "db", "this"))
        if isinstance(identifier, exp.Identifier)
    )


def _filtered_scan(table: exp.Table, condition: exp.Expr) -> exp.Subquery:
    scan = table.copy()
    alias = scan.args.pop("alias", None)

    if alias is None:
        identifier = scan.this
        quoted = identifier.quoted if isinstance(identifier, exp.Identifier) else None
        alias = exp.TableAlias(this=exp.to_identifier(scan.name, quoted=quoted))

    return exp.Subquery(
        this=exp.select("*").from_(scan).where(condition.copy()),
        alias=alias,
    )
