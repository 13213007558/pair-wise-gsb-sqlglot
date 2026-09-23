from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass, field

from sqlglot import Schema, exp, maybe_parse
from sqlglot.errors import SqlglotError
from sqlglot.optimizer import Scope, build_scope, find_all_in_scope, normalize_identifiers, qualify
from sqlglot.optimizer.scope import ScopeType
from sqlglot.schema import ensure_schema

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType
    from collections.abc import Iterator, Mapping, Sequence
    from sqlglot._typing import GraphHTMLArgs
    from typing_extensions import Unpack

logger = logging.getLogger("sqlglot")


@dataclass(frozen=True)
class Node:
    name: str
    expression: exp.Expr
    source: exp.Expr
    downstream: list[Node] = field(default_factory=list)
    source_name: str = ""
    reference_node_name: str = ""

    # Caller-injected per-node data, populated via the `on_node` hook on lineage()
    payload: dict[str, t.Any] = field(default_factory=dict)

    # The kind of value a DML write-lineage node represents, one of WriteKind.
    # Empty for nodes built by select-lineage
    kind: str = ""

    def walk(self) -> Iterator[Node]:
        visited: set[int] = set()
        queue = [self]
        while queue:
            node = queue.pop()
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)
            yield node
            queue.extend(reversed(node.downstream))

    def to_html(self, dialect: DialectType = None, **opts: Unpack[GraphHTMLArgs]) -> GraphHTML:
        nodes = {}
        edges = []

        for node in self.walk():
            if isinstance(node.expression, exp.Table):
                label = f"FROM {node.expression.this}"
                title = f"<pre>SELECT {node.name} FROM {node.expression.this}</pre>"
                group = 1
            else:
                label = node.expression.sql(pretty=True, dialect=dialect)
                source = node.source.transform(
                    lambda n: (
                        exp.Tag(this=n, prefix="<b>", postfix="</b>") if n is node.expression else n
                    ),
                    copy=False,
                ).sql(pretty=True, dialect=dialect)
                title = f"<pre>{source}</pre>"
                group = 0

            node_id = id(node)

            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "title": title,
                "group": group,
            }

            for d in node.downstream:
                edges.append({"from": node_id, "to": id(d)})
        return GraphHTML(nodes, edges, **opts)


class WriteKind:
    """The origin of the value written to a target column by a DML statement."""

    # Derived from source tables / expressions; its inputs are traced downstream
    EXPRESSION = "expression"

    # Read from a row of an INSERT ... VALUES statement; rows are downstream
    VALUES = "values"

    # A literal constant
    CONSTANT = "constant"

    # The DEFAULT keyword (or INSERT ... DEFAULT VALUES)
    DEFAULT = "default"

    # The target column is not written, so its pre-update value is kept
    RETAINED = "retained"


@dataclass
class WriteBranch:
    """A single write action of a DML statement.

    Attributes:
        action: The action performed by this branch: "INSERT", "UPDATE" or "DELETE".
        columns: Mapping of target column name to the lineage Node of the written value.
        condition: The branch's own control condition (e.g. a MERGE WHEN clause's
            condition or an UPDATE's WHERE clause). It is a control dependency only:
            it never appears in the value lineage of columns.
        matched: For MERGE branches, whether this is a MATCHED or NOT MATCHED branch.
            None for plain INSERT / UPDATE statements.
        source: For MERGE branches, whether this is a NOT MATCHED BY SOURCE branch.
    """

    action: str
    columns: dict[str, Node] = field(default_factory=dict)
    condition: exp.Expr | None = None
    matched: bool | None = None
    source: bool = False


@dataclass
class DMLLineage:
    """The write lineage of an INSERT, UPDATE or MERGE statement.

    Attributes:
        target: The name of the table being written to.
        branches: The statement's write actions. Plain INSERT / UPDATE statements have
            exactly one branch; MERGE has one per WHEN clause, in order.
        on: The MERGE statement's ON condition, a control dependency shared by all
            branches. None for other statements.
    """

    target: str
    branches: list[WriteBranch] = field(default_factory=list)
    on: exp.Expr | None = None

    @property
    def columns(self) -> dict[str, Node]:
        """All written target columns, merged across branches (first branch wins)."""
        merged: dict[str, Node] = {}
        for branch in self.branches:
            for name, node in branch.columns.items():
                merged.setdefault(name, node)
        return merged

    def walk(self) -> Iterator[Node]:
        """Yield every node of every branch's column lineage graph."""
        seen: set[int] = set()
        for branch in self.branches:
            for root in branch.columns.values():
                for node in root.walk():
                    if id(node) not in seen:
                        seen.add(id(node))
                        yield node


@t.overload
def lineage(column: str | exp.Column, sql: str | exp.Expr, **kwargs: t.Any) -> Node: ...


@t.overload
def lineage(column: None, sql: str | exp.Expr, **kwargs: t.Any) -> dict[str, Node]: ...


def lineage(
    column: str | exp.Column | None,
    sql: str | exp.Expr,
    schema: dict | Schema | None = None,
    sources: Mapping[str, str | exp.Query] | None = None,
    dialect: DialectType = None,
    scope: Scope | None = None,
    trim_selects: bool = True,
    copy: bool = True,
    on_node: t.Callable[[Node], None] | None = None,
    **kwargs,
) -> Node | dict[str, Node]:
    """Build the lineage graph for a SQL query.

    If `column` is given, returns the lineage Node for that single output column.
    If `column` is None, returns a dict mapping every top-level output column name
    to its lineage Node (with a shared cache so cross-column work is deduplicated).

    Args:
        column: The column to build the lineage for. Pass None to get all output columns.
        sql: The SQL string or expression.
        schema: The schema of tables.
        sources: A mapping of queries which will be used to continue building lineage.
        dialect: The dialect of input SQL.
        scope: A pre-created scope to use instead.
        trim_selects: Whether to clean up selects by trimming to only relevant columns.
        copy: Whether to copy the Expr arguments.
        on_node: Optional callback invoked for every Node created during the walk,
            after the Node's downstream is populated. Useful for injecting
            caller-managed data into Node.payload during the walk.
        **kwargs: Qualification optimizer kwargs.

    Returns:
        A Node when `column` is provided, or a dict[str, Node] when `column` is None.
    """
    expression = maybe_parse(sql, copy=copy, dialect=dialect)

    if sources:
        expression = exp.expand(
            expression,
            {
                k: t.cast(exp.Query, maybe_parse(v, copy=copy, dialect=dialect))
                for k, v in sources.items()
            },
            dialect=dialect,
            copy=copy,
        )

    schema = ensure_schema(schema, dialect=dialect)

    if not scope:
        expression = qualify.qualify(
            expression,
            dialect=dialect,
            schema=schema,
            **{"validate_qualify_columns": False, "identify": False, **kwargs},  # type: ignore
        )
        scope = build_scope(expression)

    if not scope:
        raise SqlglotError("Cannot build lineage, sql must be SELECT")

    selectable = scope.expression
    if not isinstance(selectable, exp.Selectable):
        raise SqlglotError("Cannot build lineage, sql must be a query")

    cache: dict[tuple, Node] = {}
    scope_meta: dict[int, tuple[bool, dict[str, exp.Expr]]] = {}

    if column is not None:
        column_name = normalize_identifiers.normalize_identifiers(column, dialect=dialect).name
        if not any(select.alias_or_name == column_name for select in selectable.selects):
            raise SqlglotError(f"Cannot find column '{column_name}' in query.")

        return to_node(
            column_name,
            scope,
            dialect,
            trim_selects=trim_selects,
            schema=schema,
            _cache=cache,
            _scope_meta=scope_meta,
            on_node=on_node,
        )

    result: dict[str, Node] = {}
    for sel in selectable.selects:
        name = sel.alias_or_name
        if not name:
            raise SqlglotError(
                f"Cannot fetch lineage for unnamed projection: {sel.sql(dialect=dialect)}."
            )

        result[name] = to_node(
            name,
            scope,
            dialect,
            trim_selects=trim_selects,
            schema=schema,
            _cache=cache,
            _scope_meta=scope_meta,
            on_node=on_node,
        )

    return result


def dml_lineage(
    sql: str | exp.Expr,
    schema: dict | Schema | None = None,
    sources: Mapping[str, str | exp.Query] | None = None,
    dialect: DialectType = None,
    copy: bool = True,
    on_node: t.Callable[[Node], None] | None = None,
    **kwargs,
) -> DMLLineage:
    """Build the write-lineage graph of an INSERT, UPDATE or MERGE statement.

    For every column of the target table that the statement writes, this traces
    which sources the written value comes from. VALUES rows, constants, DEFAULT,
    retained pre-update values and expressions read from source tables are
    distinguished through Node.kind (see WriteKind).

    UPDATE assignments read the pre-update row: a SET clause never sources a later
    SET clause. MERGE WHEN conditions are kept on their branch as control
    dependencies and are not mixed into the column value sources.

    Args:
        sql: The SQL string or expression. Must be an INSERT, UPDATE or MERGE.
        schema: The schema of tables. Required to resolve target columns when the
            statement has no explicit column list, and to expand stars.
        sources: A mapping of queries which will be used to continue building lineage.
        dialect: The dialect of input SQL.
        copy: Whether to copy the Expr arguments.
        on_node: Optional callback invoked for every Node created during the walk.
        **kwargs: Qualification optimizer kwargs.

    Returns:
        The DMLLineage of the statement.
    """
    expression = maybe_parse(sql, copy=copy, dialect=dialect)

    if sources:
        expression = exp.expand(
            expression,
            {
                k: t.cast(exp.Query, maybe_parse(v, copy=copy, dialect=dialect))
                for k, v in sources.items()
            },
            dialect=dialect,
            copy=copy,
        )

    if not isinstance(expression, (exp.Insert, exp.Update, exp.Merge)):
        raise SqlglotError(
            f"Cannot build DML lineage for {type(expression).__name__}, "
            "sql must be an INSERT, UPDATE or MERGE statement"
        )

    schema = ensure_schema(schema, dialect=dialect)
    qualify_kwargs = {"validate_qualify_columns": False, "identify": False, **kwargs}
    expression = qualify.qualify(
        expression,
        dialect=dialect,
        schema=schema,
        **qualify_kwargs,  # type: ignore
    )

    if isinstance(expression, exp.Insert):
        return _insert_lineage(expression, schema, dialect, on_node, qualify_kwargs)
    if isinstance(expression, exp.Update):
        return _update_lineage(expression, schema, dialect, on_node, qualify_kwargs)
    return _merge_lineage(t.cast(exp.Merge, expression), schema, dialect, on_node, qualify_kwargs)


def to_node(
    column: str | int,
    scope: Scope,
    dialect: DialectType,
    scope_name: str | None = None,
    upstream: Node | None = None,
    source_name: str | None = None,
    reference_node_name: str | None = None,
    trim_selects: bool = True,
    schema: Schema | None = None,
    _cache: dict[tuple, Node] | None = None,
    _scope_meta: dict[int, tuple[bool, dict[str, exp.Expr]]] | None = None,
    on_node: t.Callable[[Node], None] | None = None,
) -> Node:
    cache_key = (column, id(scope), scope_name, source_name, reference_node_name)

    if _cache is not None and cache_key in _cache:
        cached_node = _cache[cache_key]
        if upstream:
            upstream.downstream.append(cached_node)
        return cached_node

    # Find the specific select clause that is the source of the column we want.
    # This can either be a specific, named select or a generic `*` clause.
    selectable = t.cast(exp.Selectable, scope.expression)
    if isinstance(column, int):
        if column >= len(selectable.selects):
            raise SqlglotError(
                f"Cannot find column's source with index {column} in query: {selectable.sql(dialect=dialect)}"
            )
        select = selectable.selects[column]
    else:
        # Resolving a column to its select scans selectable.selects on every call;
        # memoize a per-scope {name: select} map and is_star bit instead.
        if _scope_meta is None:
            select = next(
                (s for s in selectable.selects if s.alias_or_name == column),
                exp.Star() if selectable.is_star else scope.expression,
            )
        else:
            scope_id = id(scope)
            meta = _scope_meta.get(scope_id)
            if meta is None:
                select_by_name: dict[str, exp.Expr] = {}
                for sel in selectable.selects:
                    select_by_name.setdefault(sel.alias_or_name, sel)
                meta = (selectable.is_star, select_by_name)
                _scope_meta[scope_id] = meta
            is_star, select_by_name = meta
            select = select_by_name.get(column, exp.Star() if is_star else scope.expression)

    if isinstance(scope.expression, exp.Subquery):
        for inner_scope in scope.subquery_scopes:
            result = to_node(
                column,
                scope=inner_scope,
                dialect=dialect,
                upstream=upstream,
                source_name=source_name,
                reference_node_name=reference_node_name,
                trim_selects=trim_selects,
                schema=schema,
                _cache=_cache,
                _scope_meta=_scope_meta,
                on_node=on_node,
            )
            # Skip caching a passed-in upstream returned by an inner SetOp:
            # a sibling call at the same key with that node as its upstream
            # would otherwise self-loop on the cache hit.
            if _cache is not None and result is not upstream:
                _cache[cache_key] = result
            return result
    if isinstance(scope.expression, exp.SetOperation):
        name = type(scope.expression).__name__.upper()
        created_setop = upstream is None
        upstream = upstream or Node(name=name, source=scope.expression, expression=select)

        index = (
            column
            if isinstance(column, int)
            else next(
                (
                    i
                    for i, select in enumerate(selectable.selects)
                    if select.alias_or_name == column or select.is_star
                ),
                -1,  # mypy will not allow a None here, but a negative index should never be returned
            )
        )

        if index == -1:
            raise ValueError(f"Could not find {column} in {scope.expression}")

        for s in scope.set_operation_scopes:
            to_node(
                index,
                scope=s,
                dialect=dialect,
                upstream=upstream,
                source_name=source_name,
                reference_node_name=reference_node_name,
                trim_selects=trim_selects,
                schema=schema,
                _cache=_cache,
                _scope_meta=_scope_meta,
                on_node=on_node,
            )

        if _cache is not None and created_setop:
            _cache[cache_key] = upstream
        if created_setop and on_node:
            on_node(upstream)
        return upstream

    if trim_selects and isinstance(scope.expression, exp.Select):
        # For better ergonomics in our node labels, replace the full select with
        # a version that has only the column we care about.
        #   "x", SELECT x, y FROM foo
        #     => "x", SELECT x FROM foo
        source: exp.Expr = scope.expression.select(select, append=False)
    else:
        source = scope.expression

    # Create the node for this step in the lineage chain, and attach it to the previous one.
    node = Node(
        name=f"{scope_name}.{column}" if scope_name else str(column),
        source=source,
        expression=select,
        source_name=source_name or "",
        reference_node_name=reference_node_name or "",
    )

    if upstream:
        upstream.downstream.append(node)

    subquery_scopes = {
        id(subquery_scope.expression): subquery_scope for subquery_scope in scope.subquery_scopes
    }

    for subquery in find_all_in_scope(select, *exp.UNWRAPPED_QUERIES):
        subquery_scope: Scope | None = subquery_scopes.get(id(subquery))
        if not subquery_scope:
            logger.warning(f"Unknown subquery scope: {subquery.sql(dialect=dialect)}")
            continue

        for name in subquery.named_selects:
            to_node(
                name,
                scope=subquery_scope,
                dialect=dialect,
                upstream=node,
                trim_selects=trim_selects,
                schema=schema,
                _cache=_cache,
                _scope_meta=_scope_meta,
                on_node=on_node,
            )

    # if the select is a star add all scope sources as downstreams
    if isinstance(select, exp.Star):
        for src in scope.sources.values():
            src_expr = src.expression if isinstance(src, Scope) else src
            star_node = Node(name=select.sql(comments=False), source=src_expr, expression=src_expr)
            node.downstream.append(star_node)
            if on_node:
                on_node(star_node)

    # Find all columns that went into creating this one to list their lineage nodes.
    source_columns = set(find_all_in_scope(select, exp.Column))

    # If the source is a UDTF find columns used in the UDTF to generate the table
    if isinstance(source, exp.UDTF):
        source_columns |= set(source.find_all(exp.Column))
        derived_tables: Sequence[exp.Expr] = [
            src.expression.parent
            for src in scope.sources.values()
            if isinstance(src, Scope) and src.is_derived_table and src.expression.parent
        ]
    else:
        derived_tables = scope.derived_tables

    source_names = {
        dt.alias: dt.comments[0].split()[1]
        for dt in derived_tables
        if dt.comments and dt.comments[0].startswith("source: ")
    }

    pivots = scope.pivots
    if pivots and pivots[0].parent is not pivots[-1].parent:
        # The scope's pivots only form a chain when they all hang off of the same source;
        # otherwise they can't be folded, so their columns degrade to unresolved leaves
        pivots = []

    pivot_renames: dict[str, str] = {}
    pivot_column_mapping: dict[str, list[exp.Column]] = {}

    if pivots:
        pivot_renames, pivot_column_mapping = _pivot_chain_mapping(pivots, scope, schema)

    for c in source_columns:
        table = c.table
        col_source: exp.Table | Scope | None = scope.sources.get(table)

        if isinstance(col_source, Scope):
            reference_node_name = None
            if col_source.scope_type == ScopeType.DERIVED_TABLE and table not in source_names:
                reference_node_name = table
            elif col_source.scope_type == ScopeType.CTE:
                selected_node, _ = scope.selected_sources.get(table, (None, None))
                reference_node_name = selected_node.name if selected_node else None

            # The table itself came from a more specific scope. Recurse into that one using the unaliased column name.
            to_node(
                c.name,
                scope=col_source,
                dialect=dialect,
                scope_name=table,
                upstream=node,
                source_name=source_names.get(table) or source_name,
                reference_node_name=reference_node_name,
                trim_selects=trim_selects,
                schema=schema,
                _cache=_cache,
                _scope_meta=_scope_meta,
                on_node=on_node,
            )
        elif pivots and pivots[-1].alias_or_name == c.table:
            # Only the last operator in a chain names the resulting source
            pivot_parent = pivots[-1].parent
            downstream_columns = []

            column_name = c.name
            if column_name in pivot_column_mapping:
                downstream_columns.extend(pivot_column_mapping[column_name])
            else:
                # The column is not in the pivot, so it must be an implicit column of the
                # pivoted source -- adapt column to be from the implicit pivoted source.
                downstream_columns.append(
                    exp.column(
                        pivot_renames.get(c.name, c.this),
                        table=pivot_parent.alias_or_name if pivot_parent else None,
                    )
                )

            for downstream_column in downstream_columns:
                if not downstream_column.table:
                    # Some dialects (e.g. bigquery) don't qualify the IN-list columns,
                    # but they can only come from the pivoted source
                    downstream_column = exp.column(
                        downstream_column.this,
                        table=pivot_parent.alias_or_name if pivot_parent else None,
                    )

                table = downstream_column.table
                col_source = scope.sources.get(table)
                if isinstance(col_source, exp.Table) and not col_source.db:
                    # A pivoted CTE reference maps to the raw table in `scope.sources`,
                    # so recover the CTE's scope to keep tracing through it
                    col_source = scope.cte_sources.get(col_source.name, col_source)
                if isinstance(col_source, Scope):
                    to_node(
                        downstream_column.name,
                        scope=col_source,
                        scope_name=table,
                        dialect=dialect,
                        upstream=node,
                        source_name=source_names.get(table) or source_name,
                        reference_node_name=reference_node_name,
                        trim_selects=trim_selects,
                        schema=schema,
                        _cache=_cache,
                        _scope_meta=_scope_meta,
                        on_node=on_node,
                    )
                else:
                    col_expr = col_source or exp.Placeholder()
                    pivot_leaf = Node(
                        name=downstream_column.sql(comments=False),
                        source=col_expr,
                        expression=col_expr,
                    )
                    node.downstream.append(pivot_leaf)
                    if on_node:
                        on_node(pivot_leaf)
        else:
            # The source is not a scope and the column is not in any pivot - we've reached the end
            # of the line. At this point, if a source is not found it means this column's lineage
            # is unknown. This can happen if the definition of a source used in a query is not
            # passed into the `sources` map.
            col_expr = col_source or exp.Placeholder()
            leaf = Node(name=c.sql(comments=False), source=col_expr, expression=col_expr)
            node.downstream.append(leaf)
            if on_node:
                on_node(leaf)

    if _cache is not None:
        _cache[cache_key] = node

    if on_node:
        on_node(node)

    return node


def _is_default(expression: exp.Expr) -> bool:
    """Whether an expression is the DEFAULT keyword in a DML value position."""
    if isinstance(expression, exp.Var) and expression.name.upper() == "DEFAULT":
        return True
    return (
        isinstance(expression, exp.Column)
        and not expression.table
        and expression.name.upper() == "DEFAULT"
    )


def _classify_value(expression: exp.Expr) -> str:
    expression = expression.unalias()
    if isinstance(expression, (exp.Literal, exp.Boolean, exp.Null)):
        return WriteKind.CONSTANT
    if _is_default(expression):
        return WriteKind.DEFAULT
    return WriteKind.EXPRESSION


def _write_node(
    target: str,
    column: str,
    source: exp.Expr,
    kind: str,
    expression: exp.Expr,
    on_node: t.Callable[[Node], None] | None,
) -> Node:
    node = Node(name=f"{target}.{column}", expression=expression, source=source, kind=kind)
    if on_node:
        on_node(node)
    return node


def _resolve_target_columns(
    table: exp.Table, explicit: list[str] | None, schema: Schema, context: str
) -> list[str]:
    if explicit:
        return explicit

    columns = list(schema.column_names(table))
    if not columns:
        raise SqlglotError(
            f"Cannot determine the target columns of {context}: provide an explicit "
            f"column list or include table '{table.name}' in the schema"
        )
    return columns


def _extract_assignments(
    expressions: list[exp.Expr], context: str, dialect: DialectType
) -> list[tuple[str, exp.Expr]]:
    assignments = []
    for expression in expressions:
        if not isinstance(expression, exp.EQ) or not isinstance(expression.this, exp.Column):
            raise SqlglotError(
                f"Unsupported SET expression in {context}: {expression.sql(dialect=dialect)}"
            )
        assignments.append((expression.this.name, expression.expression))
    return assignments


def _assignment_scope(
    assignments: list[tuple[str, exp.Expr]],
    from_relations: list[exp.Expr],
    conditions: list[exp.Expr],
    with_: exp.With | None,
    schema: Schema,
    dialect: DialectType,
    qualify_kwargs: dict[str, t.Any],
) -> Scope:
    """Build the scope of a synthetic SELECT over the assignment's value expressions.

    Projecting the SET / VALUES expressions from the target table and the statement's
    source relations resolves them exactly like the DML statement would: unqualified
    columns bind to the target table (reading the pre-update row) or to a source
    relation, and conditions stay out of the projections.
    """
    synthetic = exp.Select(
        expressions=[exp.alias_(rhs, column) for column, rhs in assignments],
        from_=exp.From(this=from_relations[0].copy()),
    )
    if len(from_relations) > 1:
        synthetic.set("joins", [exp.Join(this=relation.copy()) for relation in from_relations[1:]])
    if conditions:
        synthetic.set("where", exp.Where(this=exp.and_(*[c.copy() for c in conditions])))
    if with_:
        synthetic.set("with_", with_.copy())

    qualified = qualify.qualify(
        synthetic,
        dialect=dialect,
        schema=schema,
        **qualify_kwargs,  # type: ignore
    )
    scope = build_scope(qualified)
    if not scope:
        raise SqlglotError("Cannot build lineage for the DML statement's value expressions")
    return scope


def _select_key(selects: list[exp.Expr], index: int) -> str | int:
    """The key to trace a projection by: its alias, unless another projection
    shares it, in which case fall back to its position."""
    name = selects[index].alias_or_name
    return name if name and sum(s.alias_or_name == name for s in selects) == 1 else index


def _assignment_branch(
    action: str,
    table: exp.Table,
    assignments: list[tuple[str, exp.Expr]],
    from_relations: list[exp.Expr],
    conditions: list[exp.Expr],
    with_: exp.With | None,
    condition: exp.Expr | None,
    matched: bool | None,
    source: bool,
    statement: exp.Expr,
    schema: Schema,
    dialect: DialectType,
    on_node: t.Callable[[Node], None] | None,
    qualify_kwargs: dict[str, t.Any],
) -> WriteBranch:
    branch = WriteBranch(action=action, condition=condition, matched=matched, source=source)
    table_name = table.name

    classified = [(column, rhs, _classify_value(rhs)) for column, rhs in assignments]
    traced = [(column, rhs) for column, rhs, kind in classified if kind == WriteKind.EXPRESSION]

    graphs: list[Node] = []
    if traced:
        scope = _assignment_scope(
            traced, from_relations, conditions, with_, schema, dialect, qualify_kwargs
        )
        selects = t.cast(exp.Select, scope.expression).selects
        cache: dict[tuple, Node] = {}
        scope_meta: dict[int, tuple[bool, dict[str, exp.Expr]]] = {}
        for index in range(len(traced)):
            graphs.append(
                to_node(
                    _select_key(selects, index),
                    scope,
                    dialect,
                    trim_selects=True,
                    schema=schema,
                    _cache=cache,
                    _scope_meta=scope_meta,
                    on_node=on_node,
                )
            )

    graphs_iter = iter(graphs)
    for column, rhs, kind in classified:
        root = _write_node(table_name, column, statement, kind, rhs, on_node)
        if kind == WriteKind.EXPRESSION:
            root.downstream.append(next(graphs_iter))
        branch.columns[column] = root

    # Target columns that are not assigned keep their pre-update values
    normalize = normalize_identifiers.normalize_identifiers
    assigned = {normalize(column, dialect=dialect).name for column in branch.columns}
    for column in schema.column_names(table):
        if normalize(column, dialect=dialect).name not in assigned:
            retained = Node(
                name=f"{table_name}.{column}",
                expression=exp.column(column, table=table.alias_or_name),
                source=table,
                kind=WriteKind.RETAINED,
            )
            if on_node:
                on_node(retained)
            branch.columns[column] = retained

    return branch


def _trace_free_expression(
    expression: exp.Expr,
    node: Node,
    dialect: DialectType,
    on_node: t.Callable[[Node], None] | None,
) -> None:
    """Trace the inputs of a value expression that has no enclosing query scope,
    e.g. an entry of an INSERT ... VALUES row."""
    for subquery in expression.find_all(exp.Subquery):
        subquery_scope = build_scope(subquery)
        if subquery_scope and isinstance(subquery_scope.expression, exp.Selectable):
            for name in subquery_scope.expression.named_selects:
                to_node(
                    name,
                    scope=subquery_scope,
                    dialect=dialect,
                    upstream=node,
                    trim_selects=True,
                    _cache={},
                    _scope_meta={},
                    on_node=on_node,
                )

    for column in find_all_in_scope(expression, exp.Column):
        leaf = Node(
            name=column.sql(comments=False),
            source=exp.Placeholder(),
            expression=exp.Placeholder(),
        )
        node.downstream.append(leaf)
        if on_node:
            on_node(leaf)


def _insert_lineage(
    insert: exp.Insert,
    schema: Schema,
    dialect: DialectType,
    on_node: t.Callable[[Node], None] | None,
    qualify_kwargs: dict[str, t.Any],
) -> DMLLineage:
    this = insert.this
    if isinstance(this, exp.Schema) and isinstance(this.this, exp.Table):
        table = this.this
        explicit = [column.name for column in this.expressions]
    elif isinstance(this, exp.Table):
        table = this
        explicit = None
    else:
        raise SqlglotError(f"Unsupported INSERT target: {this.sql(dialect=dialect)}")

    context = f"INSERT into '{table.name}'"
    columns = _resolve_target_columns(table, explicit, schema, context)
    branch = WriteBranch(action="INSERT")
    result = DMLLineage(target=table.name, branches=[branch])

    if insert.args.get("by_name"):
        raise SqlglotError(f"Cannot build lineage for {context}: BY NAME is not supported")

    if insert.args.get("default"):
        # INSERT ... DEFAULT VALUES
        for column in columns:
            branch.columns[column] = _write_node(
                table.name, column, insert, WriteKind.DEFAULT, exp.var("DEFAULT"), on_node
            )
        return result

    source = insert.expression
    if isinstance(source, exp.Values):
        rows = source.expressions
        for row in rows:
            if len(row.expressions) != len(columns):
                raise SqlglotError(
                    f"Column count mismatch in {context}: {len(columns)} target "
                    f"column(s) but a VALUES row has {len(row.expressions)} value(s)"
                )

        for index, column in enumerate(columns):
            root = _write_node(table.name, column, insert, WriteKind.VALUES, source, on_node)
            for row in rows:
                entry = row.expressions[index]
                kind = _classify_value(entry)
                value_node = Node(
                    name=entry.sql(comments=False), expression=entry, source=row, kind=kind
                )
                if on_node:
                    on_node(value_node)
                if kind == WriteKind.EXPRESSION:
                    _trace_free_expression(entry, value_node, dialect, on_node)
                root.downstream.append(value_node)
            branch.columns[column] = root
    elif isinstance(source, exp.Query):
        scope = build_scope(insert)
        if not scope or not isinstance(scope.expression, exp.Selectable):
            raise SqlglotError(f"Cannot build lineage for the source query of {context}")

        selects = scope.expression.selects
        if any(
            isinstance(select, exp.Star) or (isinstance(select, exp.Column) and select.is_star)
            for select in selects
        ):
            raise SqlglotError(
                f"Cannot map the source columns of {context}: the source query "
                "contains an unexpanded '*'; provide a schema"
            )
        if len(selects) != len(columns):
            raise SqlglotError(
                f"Column count mismatch in {context}: {len(columns)} target "
                f"column(s) but the source query produces {len(selects)}"
            )

        cache: dict[tuple, Node] = {}
        scope_meta: dict[int, tuple[bool, dict[str, exp.Expr]]] = {}
        for index, column in enumerate(columns):
            projection = selects[index]
            kind = _classify_value(projection)
            root = _write_node(table.name, column, insert, kind, projection, on_node)
            if kind == WriteKind.EXPRESSION:
                root.downstream.append(
                    to_node(
                        _select_key(selects, index),
                        scope,
                        dialect,
                        trim_selects=True,
                        schema=schema,
                        _cache=cache,
                        _scope_meta=scope_meta,
                        on_node=on_node,
                    )
                )
            branch.columns[column] = root
    else:
        raise SqlglotError(
            f"Unsupported INSERT source in {context}: "
            f"{source.sql(dialect=dialect) if source else None}"
        )

    return result


def _update_lineage(
    update: exp.Update,
    schema: Schema,
    dialect: DialectType,
    on_node: t.Callable[[Node], None] | None,
    qualify_kwargs: dict[str, t.Any],
) -> DMLLineage:
    table = update.this
    if not isinstance(table, exp.Table):
        raise SqlglotError(f"Unsupported UPDATE target: {table.sql(dialect=dialect)}")

    relations: list[exp.Expr] = []
    from_ = update.args.get("from_")
    if isinstance(from_, exp.From):
        relations.append(from_.this)
        relations.extend(from_.expressions)

    where = update.args.get("where")
    branch = _assignment_branch(
        "UPDATE",
        table,
        _extract_assignments(update.expressions, f"UPDATE '{table.name}'", dialect),
        [table, *relations],
        [where.this] if where else [],
        update.args.get("with_"),
        condition=where.this if where else None,
        matched=None,
        source=False,
        statement=update,
        schema=schema,
        dialect=dialect,
        on_node=on_node,
        qualify_kwargs=qualify_kwargs,
    )
    return DMLLineage(target=table.name, branches=[branch])


def _merge_lineage(
    merge: exp.Merge,
    schema: Schema,
    dialect: DialectType,
    on_node: t.Callable[[Node], None] | None,
    qualify_kwargs: dict[str, t.Any],
) -> DMLLineage:
    table = merge.this
    if not isinstance(table, exp.Table):
        raise SqlglotError(f"Unsupported MERGE target: {table.sql(dialect=dialect)}")

    context = f"MERGE into '{table.name}'"
    using = merge.args.get("using")
    on = merge.args.get("on")
    with_ = merge.args.get("with_")
    result = DMLLineage(target=table.name, on=on)

    for when in merge.args["whens"].expressions:
        matched = when.args["matched"]
        by_source = when.args.get("source") or False
        condition = when.args.get("condition")
        then = when.args["then"]
        control = [c for c in (on, condition) if c is not None]

        if isinstance(then, exp.Update):
            branch = _assignment_branch(
                "UPDATE",
                table,
                _extract_assignments(then.expressions, context, dialect),
                [table, *([using] if isinstance(using, exp.Expr) else [])],
                control,
                with_,
                condition=condition,
                matched=matched,
                source=by_source,
                statement=then,
                schema=schema,
                dialect=dialect,
                on_node=on_node,
                qualify_kwargs=qualify_kwargs,
            )
        elif isinstance(then, exp.Insert):
            branch = WriteBranch(
                action="INSERT", condition=condition, matched=matched, source=by_source
            )
            columns_arg = then.this
            explicit = (
                [column.name for column in columns_arg.expressions]
                if isinstance(columns_arg, exp.Tuple)
                else None
            )
            columns = _resolve_target_columns(table, explicit, schema, context)
            values = then.expression
            if not isinstance(values, exp.Tuple) or len(values.expressions) != len(columns):
                count = len(values.expressions) if isinstance(values, exp.Tuple) else 0
                raise SqlglotError(
                    f"Column count mismatch in {context}: {len(columns)} target "
                    f"column(s) but the INSERT clause has {count} value(s)"
                )

            classified = [
                (column, value, _classify_value(value))
                for column, value in zip(columns, values.expressions)
            ]
            traced = [
                (column, value)
                for column, value, kind in classified
                if kind == WriteKind.EXPRESSION
            ]

            graphs: list[Node] = []
            if traced:
                if not isinstance(using, exp.Expr):
                    raise SqlglotError(
                        f"Cannot trace the INSERT values of {context} without a USING source"
                    )
                scope = _assignment_scope(
                    traced, [using], [], with_, schema, dialect, qualify_kwargs
                )
                merge_selects = t.cast(exp.Select, scope.expression).selects
                cache: dict[tuple, Node] = {}
                scope_meta: dict[int, tuple[bool, dict[str, exp.Expr]]] = {}
                for index in range(len(traced)):
                    graphs.append(
                        to_node(
                            _select_key(merge_selects, index),
                            scope,
                            dialect,
                            trim_selects=True,
                            schema=schema,
                            _cache=cache,
                            _scope_meta=scope_meta,
                            on_node=on_node,
                        )
                    )

            graphs_iter = iter(graphs)
            for column, value, kind in classified:
                root = _write_node(table.name, column, then, kind, value, on_node)
                if kind == WriteKind.EXPRESSION:
                    root.downstream.append(next(graphs_iter))
                branch.columns[column] = root
        elif isinstance(then, exp.Delete) or (
            isinstance(then, exp.Var) and then.name.upper() == "DELETE"
        ):
            branch = WriteBranch(
                action="DELETE", condition=condition, matched=matched, source=by_source
            )
        else:
            raise SqlglotError(
                f"Unsupported MERGE action in {context}: {then.sql(dialect=dialect)}"
            )

        result.branches.append(branch)

    return result


def _pre_pivot_columns(pivot: exp.Pivot, scope: Scope, schema: Schema | None = None) -> list[str]:
    """
    The columns the first operator of a chain sees, taken from the projections of a
    derived table or CTE source, or from the schema for a physical table. Returns an
    empty list when they can't be determined (e.g. an unexpanded star), since anything
    positional over them would silently shift.
    """
    parent = pivot.parent
    columns: list[str] = []
    if isinstance(parent, exp.DerivedTable) and isinstance(parent.this, exp.Query):
        columns = parent.this.named_selects
    elif isinstance(parent, exp.Table):
        cte_source = scope.cte_sources.get(parent.name) if not parent.db else None
        if isinstance(cte_source, Scope) and isinstance(cte_source.expression, exp.Query):
            columns = cte_source.expression.named_selects
        elif schema is not None:
            columns = list(schema.column_names(parent, only_visible=True))

    return [] if "*" in columns else columns


def _pivot_chain_mapping(
    pivots: list[exp.Pivot], scope: Scope, schema: Schema | None = None
) -> tuple[dict[str, str], dict[str, list[exp.Column]]]:
    """
    Fold a chain of (UN)PIVOT operators into a single view of its output, since each one
    consumes the previous one's columns rather than the pivoted source's.

    Returns the composed output-name -> pre-chain-name renames (from alias column lists),
    and the composed output-name -> source columns it derives from.
    """
    available = _pre_pivot_columns(pivots[0], scope, schema)
    renames: dict[str, str] = {}
    mapping: dict[str, list[exp.Column]] = {}

    for pivot in pivots:
        # Renames are positional over the operator's full output, so they can only be
        # applied when the columns going into it are known
        step_renames = (
            pivot.output_columns(available) if pivot.alias_column_names and available else {}
        )
        step_mapping = _pivot_column_mapping(pivot)
        if step_renames:
            step_mapping = {
                post: step_mapping[pre] for post, pre in step_renames.items() if pre in step_mapping
            }

        # Columns this operator consumed may have been produced by an earlier one, or be
        # alias-list renames of passthroughs; resolve through what we've folded so far
        def resolve(col: exp.Column) -> list[exp.Column]:
            if col.name in mapping:
                return mapping[col.name]
            if col.name in renames:
                return [exp.column(renames[col.name])]
            return [col]

        composed = {
            out: [resolved for col in cols for resolved in resolve(col)]
            for out, cols in step_mapping.items()
        }

        # Whatever an earlier operator produced and this one didn't consume passes through,
        # under whatever name this operator's alias column list gives it
        consumed = {col.name for cols in step_mapping.values() for col in cols}
        pre_to_post = {pre: post for post, pre in step_renames.items()}
        for out, cols in mapping.items():
            if out not in consumed:
                composed.setdefault(pre_to_post.get(out, out), cols)

        renames = (
            {post: renames.get(pre, pre) for post, pre in step_renames.items()}
            if step_renames
            else renames
        )
        mapping = composed
        available = list(pivot.output_columns(available)) if available else []

    return renames, mapping


def _pivot_column_mapping(pivot: exp.Pivot) -> dict[str, list[exp.Column]]:
    """Map each (UN)PIVOT output column name to the source columns it's derived from."""
    mapping: dict[str, list[exp.Column]] = {}

    if pivot.unpivot:
        # UNPIVOT((v1, v2) FOR name IN ((a1, a2), (b1, b2))): each value column is derived
        # positionally from the IN-list entries, and the name column from all of them
        value_columns = [
            identifier for e in pivot.expressions for identifier in e.find_all(exp.Identifier)
        ]
        for value_column in value_columns:
            mapping[value_column.name] = []

        for field in pivot.fields:
            if not isinstance(field, exp.In):
                continue

            name_columns = mapping.setdefault(field.this.name, [])
            for entry in field.expressions:
                entry_columns = list(entry.find_all(exp.Column))
                name_columns.extend(entry_columns)

                if len(entry_columns) == len(value_columns):
                    for value_column, column in zip(value_columns, entry_columns):
                        mapping[value_column.name].append(column)
                else:
                    for value_column in value_columns:
                        mapping[value_column.name].extend(entry_columns)

        return mapping

    # For each aggregation function, the pivot creates a new column for each field in category
    # combined with the aggfunc. So the columns parsed have this order: cat_a_value_sum, cat_a,
    # b_value_sum, b. Because of this step wise manner the aggfunc 'sum(value) as value_sum'
    # belongs to the column indices 0, 2, and the aggfunc 'max(price)' without an alias belongs
    # to the column indices 1, 3. Here, only the columns used in the aggregations are of interest
    # in the lineage, so lookup the pivot column name by index and map that with the columns used
    # in the aggregation.
    #
    # Example: PIVOT (SUM(value) AS value_sum, MAX(price)) FOR category IN ('a' AS cat_a, 'b')
    pivot_columns = pivot.args["columns"]
    pivot_aggs_count = len(pivot.expressions)

    mapping = {}
    for i, agg in enumerate(pivot.expressions):
        agg_cols = list(agg.find_all(exp.Column))
        for col_index in range(i, len(pivot_columns), pivot_aggs_count):
            mapping[pivot_columns[col_index].name] = agg_cols
    return mapping


class GraphHTML:
    """Node to HTML generator using vis.js.

    https://visjs.github.io/vis-network/docs/network/
    """

    def __init__(
        self,
        nodes: dict,
        edges: list,
        imports: bool = True,
        options: Mapping[str, object] | None = None,
    ):
        self.imports = imports

        self.options = {
            "height": "500px",
            "width": "100%",
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "nodeSpacing": 200,
                    "sortMethod": "directed",
                },
            },
            "interaction": {
                "dragNodes": False,
                "selectable": False,
            },
            "physics": {
                "enabled": False,
            },
            "edges": {
                "arrows": "to",
            },
            "nodes": {
                "font": "20px monaco",
                "shape": "box",
                "widthConstraint": {
                    "maximum": 300,
                },
            },
            **(options or {}),
        }

        self.nodes = nodes
        self.edges = edges

    def __str__(self):
        nodes = json.dumps(list(self.nodes.values()))
        edges = json.dumps(self.edges)
        options = json.dumps(self.options)
        imports = (
            """<script type="text/javascript" src="https://unpkg.com/vis-data@latest/peer/umd/vis-data.min.js"></script>
  <script type="text/javascript" src="https://unpkg.com/vis-network@latest/peer/umd/vis-network.min.js"></script>
  <link rel="stylesheet" type="text/css" href="https://unpkg.com/vis-network/styles/vis-network.min.css" />"""
            if self.imports
            else ""
        )

        return f"""<div>
  <div id="sqlglot-lineage"></div>
  {imports}
  <script type="text/javascript">
    var nodes = new vis.DataSet({nodes})
    nodes.forEach(row => row["title"] = new DOMParser().parseFromString(row["title"], "text/html").body.childNodes[0])

    new vis.Network(
        document.getElementById("sqlglot-lineage"),
        {{
            nodes: nodes,
            edges: new vis.DataSet({edges})
        }},
        {options},
    )
  </script>
</div>"""

    def _repr_html_(self) -> str:
        return self.__str__()
