from __future__ import annotations

from sqlglot import exp
from sqlglot.dialects.dialect import DialectType
from sqlglot.errors import UnsupportedError
from sqlglot.optimizer.scope import traverse_scope


def apply_row_policies(
    query: str | exp.Query,
    policies: dict[str, str | exp.Expr],
    dialect: DialectType = None,
    copy: bool = True,
) -> exp.Query:
    """
    Applies row-level security policies to every physical table reference in a SELECT query.

    Each policy is enforced by wrapping the matching table in a derived table, e.g.
    `orders` becomes `(SELECT * FROM orders WHERE <policy>) AS orders`. Because the
    predicate is applied inside the derived table, outer joins keep their
    null-extended rows and the output column names and order are preserved, including
    for `SELECT *` and `USING` merges. Every reference is covered: self-joins, CTE
    bodies, correlated subqueries and set operation branches. CTEs or derived tables
    that shadow a physical table name are left untouched.

    Examples:
        >>> import sqlglot
        >>> sqlglot.apply_row_policies(
        ...     "SELECT * FROM customers c LEFT JOIN orders o ON c.id = o.customer_id",
        ...     {"orders": "tenant_id = 1"},
        ... ).sql()
        'SELECT * FROM customers AS c LEFT JOIN (SELECT * FROM orders WHERE tenant_id = 1) AS o ON c.id = o.customer_id'

    Args:
        query: the SELECT query to rewrite, either as a SQL string or an expression.
        policies: a mapping of fully qualified physical table names to the predicate
            to enforce on that table. Predicates can be SQL strings or expressions;
            their columns resolve against the corresponding table and any parameters
            or placeholders are left for the caller to bind.
        dialect: the dialect used to parse `query` and string policies.
        copy: whether to copy the input query, so the returned tree shares no nodes
            with the original one.

    Returns:
        The rewritten query.
    """
    expression: exp.Expr
    if isinstance(query, str):
        expression = exp.maybe_parse(query, dialect=dialect)
    elif copy:
        expression = query.copy()
    else:
        expression = query

    if not isinstance(expression, exp.Query):
        raise UnsupportedError(
            f"Row policies can only be applied to SELECT queries, got: {type(expression).__name__}"
        )

    parsed_policies: dict[tuple[str, ...], exp.Expr] = {}
    for table_name, policy in policies.items():
        key = tuple(part.name for part in exp.to_table(table_name, dialect=dialect).parts)
        predicate = exp.maybe_parse(policy, dialect=dialect)
        parsed_policies[key] = predicate.this if isinstance(predicate, exp.Where) else predicate

    if not parsed_policies:
        return expression

    matched: list[tuple[exp.Table, exp.Expr]] = []
    seen: set[int] = set()

    for scope in traverse_scope(expression):
        parent = scope.parent

        for name, source in scope.sources.items():
            # Sources that aren't Tables are CTEs, derived tables or UDTFs, none of
            # which are physical tables. Empty names correspond to sources such as
            # table functions. Sources propagated from an enclosing scope (laterals)
            # are processed in the scope that owns them.
            if (
                not name
                or not isinstance(source, exp.Table)
                or (parent and parent.sources.get(name) is source)
                or id(source) in seen
            ):
                continue

            seen.add(id(source))

            parts = source.parts
            if not all(isinstance(part, exp.Identifier) for part in parts):
                continue

            policy_expr = parsed_policies.get(tuple(part.name for part in parts))
            if policy_expr is None:
                continue

            if source.args.get("pivots"):
                raise UnsupportedError(
                    f"Cannot apply a row policy to pivoted table: {source.sql(dialect=dialect)}"
                )

            matched.append((source, policy_expr))

    for table, predicate in matched:
        # The inner reference drops its alias so that policy columns qualified with
        # the physical table name still resolve; the original alias is moved to the
        # wrapping derived table, keeping outer references and output intact.
        inner_table = table.copy()
        inner_table.set("alias", None)

        inner = exp.select("*").from_(inner_table, copy=False).where(predicate.copy(), copy=False)

        replacement = inner.subquery(copy=False)
        alias = table.args.get("alias")
        replacement.set("alias", alias.copy() if alias else exp.TableAlias(this=table.this.copy()))

        table.replace(replacement)

    return expression
