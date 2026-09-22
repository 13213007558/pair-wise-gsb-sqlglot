from __future__ import annotations

import typing as t

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.scope import traverse_scope

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType

# Table args that survive wrapping a table reference in a policy-filtering derived
# table. Anything else (hints, pivots, sampling, ...) can't be preserved, so the
# corresponding reference is rejected instead of being silently rewritten.
_SUPPORTED_TABLE_ARGS = {"this", "db", "catalog", "alias"}


def apply_row_policies(
    expression: str | exp.Expression,
    policies: dict[str | exp.Table, str | exp.Expression],
    dialect: DialectType = None,
    copy: bool = True,
) -> exp.Expression:
    """
    Rewrite a SELECT query so that row-level policies are enforced on physical tables.

    Each policy is a predicate, keyed by the full (e.g. catalog/database qualified)
    name of a physical table. Every reference to that table is wrapped in a filtering
    derived table, e.g.:

        SELECT * FROM orders  ->  SELECT * FROM (SELECT * FROM orders WHERE <policy>) AS orders

    Wrapping each reference - instead of adding predicates to an outer WHERE clause -
    preserves the semantics of outer joins: rows on the preserved side of a LEFT JOIN
    are kept, with NULLs for the filtered side's columns. References are rewritten
    wherever they appear: self-joins, CTE bodies, (correlated) subqueries and set
    operation branches. A CTE that shadows a physical table name is not treated as a
    table reference, so it isn't wrapped. Column names and order are preserved, so
    SELECT * and JOIN ... USING behave the same before and after the rewrite.

    Columns in a policy expression are resolved against the corresponding table,
    since each policy is evaluated inside a derived table that only selects from that
    table. Parameters (e.g. @tenant_id, :tenant_id, ?) are left in place for the
    caller to bind. This function only rewrites the AST; it doesn't connect to a
    database or an account system.

    Example:
        >>> import sqlglot
        >>> sqlglot.optimizer.apply_row_policies(
        ...     "SELECT * FROM customers c LEFT JOIN orders o ON c.id = o.customer_id",
        ...     {"orders": "tenant_id = @tenant_id"},
        ... ).sql()
        'SELECT * FROM customers AS c LEFT JOIN (SELECT * FROM orders WHERE tenant_id = @tenant_id) AS o ON c.id = o.customer_id'

    Args:
        expression: the SELECT query (or set operation) to rewrite, as SQL or an AST.
        policies: mapping of full physical table name to the policy predicate for
            that table, as SQL or an expression.
        dialect: the dialect used to parse SQL strings and normalize table names.
        copy: if False, the input expression is modified in place; otherwise the
            input and the returned trees are independent.

    Returns:
        The rewritten query.

    Raises:
        OptimizeError: if the input is not a SELECT query, if scope binding fails,
            or if a matched table reference uses a construct (e.g. pivots, hints,
            sampling) that can't be wrapped safely.
    """
    parsed_policies: dict[str, exp.Expression] = {}

    for table, policy in policies.items():
        name = exp.normalize_table_name(table, dialect=dialect)
        parsed_policies[name] = exp.maybe_parse(
            policy, into=exp.Condition, dialect=dialect, copy=True
        )

    query = exp.maybe_parse(expression, dialect=dialect, copy=copy)

    if not isinstance(query, (exp.Select, exp.SetOperation)):
        raise OptimizeError(
            f"Row policies can only be applied to SELECT queries, got: {type(query).__name__}"
        )

    if not parsed_policies:
        return query

    targets: list[tuple[exp.Table, exp.Expression]] = []

    for scope in traverse_scope(query):
        for _, source in scope.selected_sources.values():
            # CTE and derived table references resolve to scopes; only physical
            # tables resolve to exp.Table nodes
            if isinstance(source, exp.Table):
                matched = parsed_policies.get(exp.normalize_table_name(source, dialect=dialect))
                if matched is not None:
                    targets.append((source, matched))

    for table, policy in targets:
        _validate_table_reference(table)

        alias = table.args.get("alias")
        if alias is None:
            this = table.this
            identifier = (
                this.copy() if isinstance(this, exp.Identifier) else exp.to_identifier(table.name)
            )
            alias = exp.TableAlias(this=identifier)

        inner = table.copy()
        inner.set("alias", None)

        table.replace(
            exp.select("*")
            .from_(inner, copy=False)
            .where(policy.copy(), copy=False)
            .subquery(alias, copy=False)
        )

    return query


def _validate_table_reference(table: exp.Table) -> None:
    unsupported = [
        key for key, value in table.args.items() if value and key not in _SUPPORTED_TABLE_ARGS
    ]

    if unsupported:
        raise OptimizeError(
            f"Cannot apply a row policy to table '{exp.table_name(table)}': "
            f"unsupported table construct(s): {', '.join(sorted(unsupported))}"
        )

    if isinstance(table.parent, exp.Final):
        raise OptimizeError(
            f"Cannot apply a row policy to table '{exp.table_name(table)}': FINAL is not supported"
        )
