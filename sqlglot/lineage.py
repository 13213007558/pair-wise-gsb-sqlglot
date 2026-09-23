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


@dataclass(frozen=True)
class WriteBranch:
    """A single write action of a DML statement.

    Attributes:
        kind: The action performed by this branch: "INSERT", "UPDATE", or "DELETE".
        columns: A mapping of target column name to the lineage Node of the value
            written into it. Empty for DELETE branches. Each Node's payload["kind"]
            classifies the written value: "source" (traced from a source table or
            query), "constant", "default", "values" (an INSERT ... VALUES row), or
            "kept" (the target column is not assigned, so its old value is kept).
        condition: The control dependency of this branch (the WHEN ... AND condition
            of a MERGE branch, or the WHERE clause of an UPDATE). Columns referenced
            here only decide which rows are written; they are not value sources and
            are therefore kept out of "columns".
        action: The parsed action expression of this branch.
    """

    kind: str
    columns: dict[str, Node] = field(default_factory=dict)
    condition: exp.Expr | None = None
    action: exp.Expr | None = None


@dataclass(frozen=True)
class WriteLineage:
    """The write lineage of an INSERT, UPDATE, or MERGE statement.

    Attributes:
        statement: The parsed DML statement.
        target: The table being written to.
        branches: The write actions of the statement, in order of appearance.
        on: The ON clause of a MERGE statement. Like WriteBranch.condition, this
            is a control dependency and is not part of any column's value lineage.
    """

    statement: exp.Expr
    target: exp.Table
    branches: list[WriteBranch] = field(default_factory=list)
    on: exp.Expr | None = None

    @property
    def target_columns(self) -> list[str]:
        """The names of all target columns written by the statement, in write order."""
        columns: list[str] = []
        for branch in self.branches:
            for name in branch.columns:
                if name not in columns:
                    columns.append(name)
        return columns


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

    if isinstance(expression, (exp.Update, exp.Merge)):
        raise SqlglotError("Cannot build lineage for a DML statement, use write_lineage instead.")

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


def write_lineage(
    sql: str | exp.Expr,
    schema: dict | Schema | None = None,
    sources: Mapping[str, str | exp.Query] | None = None,
    dialect: DialectType = None,
    copy: bool = True,
    trim_selects: bool = True,
    on_node: t.Callable[[Node], None] | None = None,
    **kwargs,
) -> WriteLineage:
    """Build the write lineage of an INSERT, UPDATE, or MERGE statement.

    For every column of the target table written by the statement, this traces the
    written value back to its sources: a source query (INSERT ... SELECT), a VALUES
    row, a constant, DEFAULT, a source table expression (UPDATE ... FROM, MERGE), or
    the target row's own previous value for columns the statement doesn't assign.

    UPDATE assignments all read the pre-update row, so e.g. swapping two columns
    with SET a = b, b = a correctly traces each target column to the other column's
    old value. MERGE keeps every WHEN branch's condition and action separate, and
    control dependencies (ON / WHEN / WHERE clauses) are never mixed into the value
    lineage of a target column.

    Args:
        sql: The INSERT, UPDATE, or MERGE statement, as a SQL string or expression.
        schema: The schema of tables. Required to resolve target columns when an
            INSERT omits its column list, and to report columns whose old value is
            kept by an UPDATE.
        sources: A mapping of queries which will be used to continue building lineage.
        dialect: The dialect of input SQL.
        copy: Whether to copy the Expr arguments.
        trim_selects: Whether to clean up selects by trimming to only relevant columns.
        on_node: Optional callback invoked for every Node created during the walk,
            after the Node's downstream is populated.
        **kwargs: Qualification optimizer kwargs.

    Returns:
        The WriteLineage of the statement.
    """
    statement = maybe_parse(sql, copy=copy, dialect=dialect)

    if sources:
        statement = exp.expand(
            statement,
            {
                k: t.cast(exp.Query, maybe_parse(v, copy=copy, dialect=dialect))
                for k, v in sources.items()
            },
            dialect=dialect,
            copy=copy,
        )

    tracer = _WriteTracer(
        schema=ensure_schema(schema, dialect=dialect),
        dialect=dialect,
        sql=statement.sql(dialect=dialect),
        trim_selects=trim_selects,
        on_node=on_node,
        qualify_kwargs=kwargs,
    )

    if isinstance(statement, exp.Insert):
        return tracer.insert(statement)
    if isinstance(statement, exp.Update):
        return tracer.update(statement)
    if isinstance(statement, exp.Merge):
        return tracer.merge(statement)

    raise SqlglotError("Cannot build write lineage, sql must be INSERT, UPDATE, or MERGE")


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


class _WriteTracer:
    """Traces the values written by a DML statement back to their sources.

    Value expressions (UPDATE SET right-hand sides, MERGE ... VALUES, etc.) are
    traced by assembling a synthetic SELECT over the statement's sources and
    reusing to_node, so CTEs, subqueries, aliases, stars, and set operations
    resolve exactly as they do for SELECT lineage. Every synthetic SELECT reads
    the raw source tables, which is what makes UPDATE assignments see the
    pre-update row rather than each other's results.
    """

    def __init__(
        self,
        schema: Schema,
        dialect: DialectType,
        sql: str,
        trim_selects: bool,
        on_node: t.Callable[[Node], None] | None,
        qualify_kwargs: dict[str, t.Any],
    ):
        self.schema = schema
        self.dialect = dialect
        self.sql = sql
        self.trim_selects = trim_selects
        self.on_node = on_node
        self.qualify_kwargs = qualify_kwargs
        self.cache: dict[tuple, Node] = {}
        self.scope_meta: dict[int, tuple[bool, dict[str, exp.Expr]]] = {}

    def insert(self, insert: exp.Insert) -> WriteLineage:
        target, columns = self._insert_target(insert)
        query = insert.expression

        if query is None:
            raise SqlglotError(
                f"Cannot build write lineage for {self.sql}: INSERT has no source query"
            )

        if isinstance(query, exp.Values):
            branch_columns = self._values_columns(insert, query, columns)
        elif isinstance(query, exp.Query):
            with_ = insert.args.get("with_")
            if with_ is not None and query.args.get("with_") is None:
                query.set("with_", with_)

            scope = self._build_scope(query)
            selects = query.selects

            if any(isinstance(select, exp.Star) for select in selects):
                raise SqlglotError(
                    f"Cannot expand * in {query.sql(dialect=self.dialect)}: "
                    "provide a schema for the source tables"
                )
            if len(selects) != len(columns):
                raise SqlglotError(
                    f"Column count mismatch in {self.sql}: "
                    f"{len(columns)} target columns but the source query produces {len(selects)}"
                )

            branch_columns = {}
            for index, name in enumerate(columns):
                node = self._target_node(name, insert, selects[index])
                branch_columns[name] = node
                self._to_node(index, scope, upstream=node)
                self._emit(node)
        else:
            raise SqlglotError(
                f"Cannot build write lineage for {self.sql}: "
                f"unsupported source {query.sql(dialect=self.dialect)}"
            )

        branch = WriteBranch(kind="INSERT", columns=branch_columns, action=insert)
        return WriteLineage(statement=insert, target=target, branches=[branch])

    def update(self, update: exp.Update) -> WriteLineage:
        target = update.this
        if not isinstance(target, exp.Table):
            raise SqlglotError(
                f"Cannot build write lineage for {self.sql}: UPDATE must target a table"
            )

        sources: list[exp.Expr] = [target]
        from_ = update.args.get("from_")
        if from_:
            sources.extend(_from_sources(from_))

        columns = self._set_columns(update, sources, update)
        self._add_kept_columns(columns, target)

        branch = WriteBranch(
            kind="UPDATE",
            columns=columns,
            condition=update.args.get("where"),
            action=update,
        )
        return WriteLineage(statement=update, target=target, branches=[branch])

    def merge(self, merge: exp.Merge) -> WriteLineage:
        target = merge.this
        using = merge.args.get("using")
        whens = merge.args.get("whens")

        if not isinstance(target, exp.Table) or using is None or whens is None:
            raise SqlglotError(
                f"Cannot build write lineage for {self.sql}: "
                "MERGE must target a table with a USING source and WHEN clauses"
            )

        branches = []
        for when in whens.expressions:
            matched = when.args["matched"]
            by_source = bool(when.args.get("source"))
            then = when.args["then"]

            # NOT MATCHED [BY TARGET] branches only see the source row, NOT MATCHED
            # BY SOURCE branches only see the target row, MATCHED branches see both
            sources: list[exp.Expr] = []
            if matched or by_source:
                sources.append(target)
            if matched or not by_source:
                sources.append(using)

            if isinstance(then, exp.Update):
                kind = "UPDATE"
                columns = self._set_columns(then, sources, merge)
                self._add_kept_columns(columns, target)
            elif isinstance(then, exp.Insert):
                kind = "INSERT"
                columns = self._merge_insert_columns(merge, then, target, sources)
            else:
                kind = "DELETE"
                columns = {}

            branches.append(
                WriteBranch(
                    kind=kind,
                    columns=columns,
                    condition=when.args.get("condition"),
                    action=then,
                )
            )

        return WriteLineage(
            statement=merge,
            target=target,
            branches=branches,
            on=merge.args.get("on"),
        )

    def _insert_target(self, insert: exp.Insert) -> tuple[exp.Table, list[str]]:
        this = insert.this
        if isinstance(this, exp.Schema) and isinstance(this.this, exp.Table):
            return this.this, [column.name for column in this.expressions]

        if isinstance(this, exp.Table):
            columns = list(self.schema.column_names(this, only_visible=True))
            if not columns:
                raise SqlglotError(
                    f"Cannot determine the target columns of {self.sql}: "
                    f"add an explicit column list or provide a schema for table "
                    f"'{this.sql(dialect=self.dialect)}'"
                )
            return this, columns

        raise SqlglotError(f"Cannot build write lineage for {self.sql}: INSERT must target a table")

    def _values_columns(
        self, insert: exp.Insert, values: exp.Values, columns: list[str]
    ) -> dict[str, Node]:
        rows = values.expressions
        for row in rows:
            if not isinstance(row, exp.Tuple) or len(row.expressions) != len(columns):
                raise SqlglotError(
                    f"Column count mismatch in {self.sql}: "
                    f"{len(columns)} target columns but row {row.sql(dialect=self.dialect)} "
                    f"produces {len(row.expressions) if isinstance(row, exp.Tuple) else 1}"
                )

        result = {}
        for index, name in enumerate(columns):
            node = Node(
                name=name,
                source=insert,
                expression=exp.column(name),
                payload={"kind": "values"},
            )
            result[name] = node
            for row in t.cast(list[exp.Tuple], rows):
                value = row.expressions[index]
                leaf = Node(
                    name=value.sql(comments=False),
                    source=values,
                    expression=value,
                    payload={"kind": _value_kind(value)},
                )
                node.downstream.append(leaf)
                self._emit(leaf)
            self._emit(node)

        return result

    def _set_columns(
        self, update: exp.Update, sources: list[exp.Expr], statement: exp.Expr
    ) -> dict[str, Node]:
        columns = []
        values = []
        for assignment in update.expressions:
            if not isinstance(assignment, exp.EQ) or not isinstance(assignment.this, exp.Column):
                raise SqlglotError(
                    f"Cannot build write lineage for {self.sql}: "
                    f"unsupported SET assignment {assignment.sql(dialect=self.dialect)}"
                )
            columns.append(assignment.this.name)
            values.append(assignment.expression)

        if len(set(columns)) != len(columns):
            raise SqlglotError(
                f"Cannot build write lineage for {self.sql}: "
                "a target column is assigned more than once"
            )

        return self._trace_values(columns, values, sources, statement)

    def _merge_insert_columns(
        self, merge: exp.Merge, insert: exp.Insert, target: exp.Table, sources: list[exp.Expr]
    ) -> dict[str, Node]:
        column_list = insert.this
        if isinstance(column_list, exp.Tuple):
            columns = [column.name for column in column_list.expressions]
        elif column_list is None:
            columns = list(self.schema.column_names(target, only_visible=True))
            if not columns:
                raise SqlglotError(
                    f"Cannot determine the target columns of {self.sql}: "
                    "add an explicit column list to the INSERT branch or provide a "
                    f"schema for table '{target.sql(dialect=self.dialect)}'"
                )
        else:
            raise SqlglotError(
                f"Cannot build write lineage for {self.sql}: "
                f"unsupported INSERT branch {self.sql}"
            )

        row = insert.expression
        if not isinstance(row, exp.Tuple):
            raise SqlglotError(
                f"Cannot build write lineage for {self.sql}: "
                "INSERT branch must have a VALUES row"
            )

        return self._trace_values(columns, row.expressions, sources, merge)

    def _trace_values(
        self,
        columns: list[str],
        values: t.Sequence[exp.Expr],
        sources: list[exp.Expr],
        statement: exp.Expr,
    ) -> dict[str, Node]:
        if len(columns) != len(values):
            raise SqlglotError(
                f"Column count mismatch in {self.sql}: "
                f"{len(columns)} target columns but {len(values)} source expressions"
            )

        select = exp.Select(expressions=[value.copy() for value in values])
        select.set("from_", exp.From(this=sources[0].copy()))
        for source in sources[1:]:
            source = source.copy()
            select.append(
                "joins", source if isinstance(source, exp.Join) else exp.Join(this=source)
            )

        with_ = statement.args.get("with_")
        if with_ is not None:
            select.set("with_", with_.copy())

        scope = self._build_scope(select)

        result = {}
        for index, name in enumerate(columns):
            node = self._target_node(name, statement, values[index])
            result[name] = node
            self._to_node(index, scope, upstream=node)
            self._emit(node)

        return result

    def _add_kept_columns(self, columns: dict[str, Node], target: exp.Table) -> None:
        for name in self.schema.column_names(target, only_visible=True):
            if name not in columns:
                node = Node(
                    name=name,
                    source=target,
                    expression=exp.column(name, table=target.alias_or_name or None),
                    payload={"kind": "kept"},
                )
                columns[name] = node
                self._emit(node)

    def _target_node(self, name: str, statement: exp.Expr, value: exp.Expr) -> Node:
        return Node(
            name=name,
            source=statement,
            expression=exp.column(name),
            payload={"kind": _value_kind(value)},
        )

    def _build_scope(self, query: exp.Query) -> Scope:
        query = qualify.qualify(
            query,
            dialect=self.dialect,
            schema=self.schema,
            **{"validate_qualify_columns": False, "identify": False, **self.qualify_kwargs},  # type: ignore
        )
        scope = build_scope(query)
        if not scope:
            raise SqlglotError(
                f"Cannot build write lineage, could not scope {query.sql(dialect=self.dialect)}"
            )
        return scope

    def _to_node(self, index: int, scope: Scope, upstream: Node) -> None:
        if isinstance(scope.expression, exp.SetOperation):
            # to_node only creates a node for the set operation when there is no
            # upstream; create it here so UNIONs stay visible in the graph
            set_operation = Node(
                name=type(scope.expression).__name__.upper(),
                source=scope.expression,
                expression=scope.expression,
            )
            upstream.downstream.append(set_operation)
            self._emit(set_operation)
            upstream = set_operation

        to_node(
            index,
            scope=scope,
            dialect=self.dialect,
            upstream=upstream,
            trim_selects=self.trim_selects,
            schema=self.schema,
            _cache=self.cache,
            _scope_meta=self.scope_meta,
            on_node=self.on_node,
        )

    def _emit(self, node: Node) -> None:
        if self.on_node:
            self.on_node(node)


def _from_sources(from_: exp.From) -> list[exp.Expr]:
    # In an UPDATE ... FROM, the parser hangs comma-joined and explicit joins off
    # the first source, so detach them to rebuild a flat source list
    sources = []
    first = from_.this
    nested = first.args.get("joins") if isinstance(first, (exp.Table, exp.Subquery)) else None
    if nested:
        first = first.copy()
        first.args.pop("joins")
        sources.append(first)
        sources.extend(nested)
    else:
        sources.append(first)
    sources.extend(from_.args.get("expressions") or [])
    return sources


def _value_kind(expression: exp.Expr) -> str:
    expression = expression.unalias()
    if (
        isinstance(expression, exp.Var)
        or (isinstance(expression, exp.Column) and not expression.table)
    ) and expression.name.upper() == "DEFAULT":
        return "default"
    if expression.find(exp.Column, exp.Subquery, exp.Star):
        return "source"
    return "constant"


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
