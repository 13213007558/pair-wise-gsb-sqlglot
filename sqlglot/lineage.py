from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import asdict, dataclass, field

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


class DependencyKind:
    """Impact categories recorded in a DependencyReport."""

    # The column feeds a computed output value (projections, CASE conditions,
    # window definitions, aggregate inputs).
    VALUE = "value"
    # The column determines which rows appear (JOIN/WHERE/HAVING/QUALIFY,
    # GROUP BY, DISTINCT, ORDER BY ... LIMIT).
    ROWS = "rows"
    # The column only determines output ordering (ORDER BY without LIMIT).
    ORDER = "order"


@dataclass(frozen=True)
class Dependency:
    """A classified dependency on a physical table column or relation.

    Attributes:
        kind: One of DependencyKind.VALUE / ROWS / ORDER.
        table: The physical table the dependency bottoms out at.
        column: The column name, or "*" when the dependency is on the
            relation's rows (e.g. COUNT(*)).
        reason: The syntactic construct that introduced the dependency,
            e.g. "projection", "where", "join", "group_by", "having",
            "distinct", "window_partition", "window_order", "case_condition",
            "order_by" or "order_by_limit".
        scope: Stable identifier of the query level owning the construct
            ("q0", "q1", ...), see DependencyReport.scopes.
        select: The output column whose value this feeds (value dependencies).
    """

    kind: str
    table: str
    column: str
    reason: str
    scope: str
    select: str = ""


@dataclass(frozen=True)
class Unknown:
    """A dependency that could not be determined, with the reason why."""

    reason: str
    detail: str
    scope: str


@dataclass
class DependencyReport:
    """Classified dependencies of a query, resolved to physical table columns.

    Unlike the lineage() graph, which only follows output values, a report
    also records which columns determine the row set and the output ordering,
    so impact analysis can tell "changing this column changes values, the row
    set, or the order" apart.
    """

    dependencies: list[Dependency]
    unknowns: list[Unknown]
    # Stable scope id -> the SQL of that query level
    scopes: dict[str, str]

    @property
    def value(self) -> list[Dependency]:
        return [d for d in self.dependencies if d.kind == DependencyKind.VALUE]

    @property
    def rows(self) -> list[Dependency]:
        return [d for d in self.dependencies if d.kind == DependencyKind.ROWS]

    @property
    def order(self) -> list[Dependency]:
        return [d for d in self.dependencies if d.kind == DependencyKind.ORDER]

    def by_column(self) -> dict[str, list[str]]:
        """Map each physical "table.column" to the impact categories it has."""
        impact: dict[str, set[str]] = {}
        for d in self.dependencies:
            impact.setdefault(f"{d.table}.{d.column}", set()).add(d.kind)
        return {column: sorted(kinds) for column, kinds in sorted(impact.items())}

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "dependencies": [asdict(d) for d in self.dependencies],
            "unknowns": [asdict(u) for u in self.unknowns],
            "scopes": dict(self.scopes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def dependency_report(
    sql: str | exp.Expr,
    schema: dict | Schema | None = None,
    sources: Mapping[str, str | exp.Query] | None = None,
    dialect: DialectType = None,
    copy: bool = True,
    **kwargs,
) -> DependencyReport:
    """Build a classified dependency report for a SQL query.

    This is an explicit alternative to lineage(): every dependency is traced
    through CTEs, subqueries and set operations down to physical table columns
    (stars are expanded when a schema is available) and classified by impact
    kind - "value", "rows" or "order" - together with the query level that
    introduced it. Anything that cannot be determined (e.g. unexpanded stars,
    recursive constructs, UDTFs) is reported as an Unknown with a reason
    instead of being silently dropped.

    Args:
        sql: The SQL string or expression.
        schema: The schema of tables, used to expand stars and resolve
            unqualified columns.
        sources: A mapping of queries which will be used to continue resolving.
        dialect: The dialect of input SQL.
        copy: Whether to copy the Expr arguments.
        **kwargs: Qualification optimizer kwargs.

    Returns:
        A DependencyReport with stable, serializable node identities.

    Example:
        >>> report = dependency_report(
        ...     "SELECT salary FROM staff WHERE region = 'east' ORDER BY salary LIMIT 3"
        ... )
        >>> report.by_column()
        {'staff.region': ['rows'], 'staff.salary': ['rows', 'value']}
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

    expression = qualify.qualify(
        expression,
        dialect=dialect,
        schema=schema,
        **{"validate_qualify_columns": False, "identify": False, **kwargs},  # type: ignore
    )
    scope = build_scope(expression)

    if not scope:
        raise SqlglotError("Cannot build dependency report, sql must be SELECT")

    return _DependencyCollector(scope, schema, dialect).run()


class _DependencyCollector:
    """Walks every scope of a qualified query, classifying column dependencies."""

    def __init__(self, root: Scope, schema: Schema, dialect: DialectType) -> None:
        self.schema = schema
        self.dialect = dialect

        # Stable, deterministic query-level identities; the root is always q0
        self.ordered_scopes = [root] + [s for s in root.traverse() if s is not root]
        self.scope_ids = {id(s): f"q{i}" for i, s in enumerate(self.ordered_scopes)}
        self.scope_sql = {
            self.scope_ids[id(s)]: s.expression.sql(dialect=dialect) for s in self.ordered_scopes
        }

        self.dependencies: set[Dependency] = set()
        self.unknowns: set[Unknown] = set()

        self._column_cache: dict[tuple, frozenset] = {}
        self._select_cache: dict[tuple, frozenset] = {}
        self._select_maps: dict[int, dict[str, exp.Expr]] = {}
        self._resolving: set[tuple] = set()

    def run(self) -> DependencyReport:
        for scope in self.ordered_scopes:
            expression = scope.expression
            if isinstance(expression, exp.Select):
                self._collect_select(scope)
            elif isinstance(expression, exp.SetOperation):
                # Branches are visited as their own scopes; only the set
                # operation's own modifiers are collected here
                self._collect_order(scope, expression)

        return DependencyReport(
            dependencies=sorted(
                self.dependencies,
                key=lambda d: (d.kind, d.table, d.column, d.select, d.reason, d.scope),
            ),
            unknowns=sorted(self.unknowns, key=lambda u: (u.reason, u.detail, u.scope)),
            scopes=self.scope_sql,
        )

    # Clause collection

    def _collect_select(self, scope: Scope) -> None:
        select = t.cast(exp.Select, scope.expression)
        sid = self.scope_ids[id(scope)]

        for projection in select.selects:
            self._collect_projection(scope, projection, sid)

        for key in ("where", "having", "qualify"):
            clause = select.args.get(key)
            if clause:
                self._collect_columns(scope, clause, DependencyKind.ROWS, key, sid)

        for join in select.args.get("joins") or []:
            on = join.args.get("on")
            if on:
                self._collect_columns(scope, on, DependencyKind.ROWS, "join", sid)
            for using in join.args.get("using") or []:
                self._add_column_deps(
                    scope, exp.column(using.name), DependencyKind.ROWS, "join", sid
                )

        group = select.args.get("group")
        if group:
            self._collect_columns(scope, group, DependencyKind.ROWS, "group_by", sid)

        if select.args.get("distinct"):
            for projection in select.selects:
                self._collect_columns(scope, projection, DependencyKind.ROWS, "distinct", sid)

        self._collect_order(scope, select)

    def _collect_order(self, scope: Scope, query: exp.Expr) -> None:
        order = query.args.get("order")
        if not order:
            return

        sid = self.scope_ids[id(scope)]
        if query.args.get("limit") or query.args.get("fetch"):
            # With a row cap, ordering decides which rows survive
            self._collect_columns(scope, order, DependencyKind.ROWS, "order_by_limit", sid)
        else:
            self._collect_columns(scope, order, DependencyKind.ORDER, "order_by", sid)

    def _collect_projection(self, scope: Scope, projection: exp.Expr, sid: str) -> None:
        name = projection.alias_or_name or projection.sql(comments=False, dialect=self.dialect)

        if isinstance(projection, exp.Star) or (
            isinstance(projection, exp.Column) and projection.is_star
        ):
            table = projection.table if isinstance(projection, exp.Column) else None
            for table_name, column in self._resolve_star(scope, table or None):
                self.dependencies.add(
                    Dependency(DependencyKind.VALUE, table_name, column, "projection", sid, name)
                )
            return

        # Columns in CASE conditions and window definitions are value
        # dependencies too, but with a more specific reason
        skip: set[int] = set()
        for window in find_all_in_scope(projection, exp.Window):
            for partition in window.args.get("partition_by") or []:
                self._collect_columns(
                    scope, partition, DependencyKind.VALUE, "window_partition", sid, name
                )
                skip.update(id(c) for c in find_all_in_scope(partition, exp.Column))
            order = window.args.get("order")
            if order:
                self._collect_columns(scope, order, DependencyKind.VALUE, "window_order", sid, name)
                skip.update(id(c) for c in find_all_in_scope(order, exp.Column))

        for case_if in find_all_in_scope(projection, exp.If):
            self._collect_columns(
                scope, case_if.this, DependencyKind.VALUE, "case_condition", sid, name
            )
            skip.update(id(c) for c in find_all_in_scope(case_if.this, exp.Column))

        self._collect_columns(
            scope, projection, DependencyKind.VALUE, "projection", sid, name, skip
        )

    def _collect_columns(
        self,
        scope: Scope,
        expression: exp.Expr,
        kind: str,
        reason: str,
        sid: str,
        select_name: str = "",
        skip: t.AbstractSet[int] = frozenset(),
    ) -> None:
        for column in find_all_in_scope(expression, exp.Column):
            if id(column) not in skip:
                self._add_column_deps(scope, column, kind, reason, sid, select_name)

        # COUNT(*) / COUNT(1) have no column value source but still depend on
        # the input rows of every relation in this scope
        for count in find_all_in_scope(expression, exp.Count):
            if not any(find_all_in_scope(count, exp.Column)):
                self._collect_relation_rows(scope, kind, reason, sid, select_name)

        subquery_scopes = {id(s.expression): s for s in scope.subquery_scopes}
        for subquery in find_all_in_scope(expression, *exp.UNWRAPPED_QUERIES):
            subquery_scope = subquery_scopes.get(id(subquery))
            if subquery_scope is None:
                self._unknown(
                    "unknown_subquery", subquery.sql(comments=False, dialect=self.dialect), sid
                )
                continue
            for name in subquery.named_selects:
                for table, column in self._resolve_named_select(subquery_scope, name):
                    self.dependencies.add(
                        Dependency(
                            kind,
                            table,
                            column,
                            reason,
                            self.scope_ids[id(subquery_scope)],
                            select_name,
                        )
                    )

    def _collect_relation_rows(
        self,
        scope: Scope,
        kind: str,
        reason: str,
        sid: str,
        select_name: str,
        seen: set[int] | None = None,
    ) -> None:
        seen = seen or set()
        for name, source in scope.sources.items():
            if isinstance(source, exp.Table):
                self.dependencies.add(
                    Dependency(kind, self._table_name(source), "*", reason, sid, select_name)
                )
            elif isinstance(source, Scope):
                if source.scope_type == ScopeType.UDTF:
                    self._unknown("udtf", name, sid)
                elif id(source) not in self.scope_ids:
                    self._unknown("recursive_reference", name, sid)
                elif id(source) in seen:
                    self._unknown("recursive_reference", name, sid)
                else:
                    seen.add(id(source))
                    self._collect_relation_rows(source, kind, reason, sid, select_name, seen)
            else:
                self._unknown("unknown_source", name, sid)

    # Dependency recording

    def _add_column_deps(
        self,
        scope: Scope,
        column: exp.Column,
        kind: str,
        reason: str,
        sid: str,
        select_name: str = "",
    ) -> None:
        for table, name in self._resolve_column(scope, column):
            self.dependencies.add(Dependency(kind, table, name, reason, sid, select_name))

    def _unknown(self, reason: str, detail: str, sid: str) -> None:
        self.unknowns.add(Unknown(reason=reason, detail=detail, scope=sid))

    # Resolution of column references down to physical (table, column) leaves

    def _resolve_column(self, scope: Scope, column: exp.Column) -> frozenset:
        key = ("column", id(scope), column.table, column.name)
        if key in self._column_cache:
            return self._column_cache[key]
        if key in self._resolving:
            self._unknown(
                "recursive_reference",
                column.sql(comments=False, dialect=self.dialect),
                self.scope_ids[id(scope)],
            )
            return frozenset()

        self._resolving.add(key)
        try:
            leaves = self._resolve_column_inner(scope, column)
        finally:
            self._resolving.discard(key)

        self._column_cache[key] = leaves
        return leaves

    def _resolve_column_inner(self, scope: Scope, column: exp.Column) -> frozenset:
        table = column.table

        if not table:
            # An unqualified reference is usually an output alias used in
            # GROUP BY / ORDER BY / HAVING
            projection = self._select_map(scope).get(column.name)
            if projection is not None:
                return self._resolve_expression(scope, projection)
            return self._resolve_unqualified(scope, column)

        source = scope.sources.get(table)
        if source is None:
            # Correlated column belonging to an outer query level
            parent = scope.parent
            while parent is not None:
                if table in parent.sources:
                    return self._resolve_column(parent, column)
                parent = parent.parent
            self._unknown("unknown_source", f"{table}.{column.name}", self.scope_ids[id(scope)])
            return frozenset()

        return self._resolve_source(scope, source, column.name)

    def _resolve_unqualified(self, scope: Scope, column: exp.Column) -> frozenset:
        sources = scope.sources
        if len(sources) == 1:
            return self._resolve_source(scope, next(iter(sources.values())), column.name)

        matches = [
            source
            for source in sources.values()
            if isinstance(source, exp.Table)
            and column.name in self.schema.column_names(source, only_visible=True)
        ]
        if len(matches) == 1:
            return frozenset({(self._table_name(matches[0]), column.name)})

        self._unknown(
            "unqualified_column",
            column.sql(comments=False, dialect=self.dialect),
            self.scope_ids[id(scope)],
        )
        return frozenset()

    def _resolve_source(self, scope: Scope, source: exp.Expr | Scope, name: str) -> frozenset:
        if isinstance(source, exp.Table):
            return frozenset({(self._table_name(source), name)})
        if isinstance(source, Scope):
            if source.scope_type == ScopeType.UDTF:
                self._unknown("udtf", name, self.scope_ids[id(scope)])
                return frozenset()
            if id(source) not in self.scope_ids:
                # A source scope outside the traversed tree is the synthetic
                # base case of a recursive CTE; the recursive step cannot be
                # resolved statically
                self._unknown("recursive_reference", name, self.scope_ids[id(scope)])
                return frozenset()
            return self._resolve_named_select(source, name)

        self._unknown("unknown_source", name, self.scope_ids[id(scope)])
        return frozenset()

    def _resolve_named_select(self, scope: Scope, name: str) -> frozenset:
        key = ("select", id(scope), name)
        if key in self._select_cache:
            return self._select_cache[key]
        if key in self._resolving:
            self._unknown("recursive_reference", name, self.scope_ids[id(scope)])
            return frozenset()

        self._resolving.add(key)
        try:
            expression = scope.expression
            if isinstance(expression, exp.Subquery):
                leaves = self._union(
                    self._resolve_named_select(s, name) for s in scope.subquery_scopes
                )
            elif isinstance(expression, exp.SetOperation):
                named = expression.named_selects
                if name in named:
                    index = named.index(name)
                    leaves = self._union(
                        self._resolve_select_index(s, index) for s in scope.set_operation_scopes
                    )
                else:
                    self._unknown("column_not_found", name, self.scope_ids[id(scope)])
                    leaves = frozenset()
            elif isinstance(expression, exp.Select):
                projection = self._select_map(scope).get(name)
                if projection is not None:
                    leaves = self._resolve_expression(scope, projection)
                elif expression.is_star:
                    leaves = self._resolve_unqualified(scope, exp.column(name))
                else:
                    self._unknown("column_not_found", name, self.scope_ids[id(scope)])
                    leaves = frozenset()
            else:
                self._unknown(
                    "unsupported_scope", type(expression).__name__, self.scope_ids[id(scope)]
                )
                leaves = frozenset()
        finally:
            self._resolving.discard(key)

        self._select_cache[key] = leaves
        return leaves

    def _resolve_select_index(self, scope: Scope, index: int) -> frozenset:
        expression = scope.expression
        if isinstance(expression, exp.SetOperation):
            return self._union(
                self._resolve_select_index(s, index) for s in scope.set_operation_scopes
            )
        if isinstance(expression, exp.Subquery):
            return self._union(self._resolve_select_index(s, index) for s in scope.subquery_scopes)

        selects = t.cast(exp.Select, expression).selects
        if index >= len(selects):
            self._unknown("column_not_found", str(index), self.scope_ids[id(scope)])
            return frozenset()
        return self._resolve_expression(scope, selects[index])

    def _resolve_expression(self, scope: Scope, expression: exp.Expr) -> frozenset:
        if isinstance(expression, exp.Star):
            return self._resolve_star(scope, None)
        if isinstance(expression, exp.Column) and expression.is_star:
            return self._resolve_star(scope, expression.table or None)

        leaves: set[tuple[str, str]] = set()
        for column in find_all_in_scope(expression, exp.Column):
            leaves |= self._resolve_column(scope, column)

        subquery_scopes = {id(s.expression): s for s in scope.subquery_scopes}
        for subquery in find_all_in_scope(expression, *exp.UNWRAPPED_QUERIES):
            subquery_scope = subquery_scopes.get(id(subquery))
            if subquery_scope is None:
                self._unknown(
                    "unknown_subquery",
                    subquery.sql(comments=False, dialect=self.dialect),
                    self.scope_ids[id(scope)],
                )
                continue
            for name in subquery.named_selects:
                leaves |= self._resolve_named_select(subquery_scope, name)

        return frozenset(leaves)

    def _resolve_star(self, scope: Scope, table: str | None) -> frozenset:
        sid = self.scope_ids[id(scope)]
        sources = {table: scope.sources.get(table)} if table else scope.sources

        leaves: set[tuple[str, str]] = set()
        for name, source in sources.items():
            if isinstance(source, exp.Table):
                columns = self.schema.column_names(source, only_visible=True)
                if columns:
                    leaves.update((self._table_name(source), c) for c in columns)
                else:
                    self._unknown("unexpanded_star", f"{name}.*", sid)
                    leaves.add((self._table_name(source), "*"))
            elif isinstance(source, Scope):
                if source.scope_type == ScopeType.UDTF:
                    self._unknown("udtf", name, sid)
                    continue
                expression = source.expression
                named = expression.named_selects if isinstance(expression, exp.Query) else []
                for select_name in named:
                    if select_name == "*":
                        leaves |= self._resolve_star(source, None)
                    else:
                        leaves |= self._resolve_named_select(source, select_name)
            else:
                self._unknown("unknown_source", str(name), sid)

        return frozenset(leaves)

    # Helpers

    def _select_map(self, scope: Scope) -> dict[str, exp.Expr]:
        key = id(scope)
        select_map = self._select_maps.get(key)
        if select_map is None:
            select_map = {}
            expression = scope.expression
            if isinstance(expression, exp.Select):
                for projection in expression.selects:
                    select_map.setdefault(projection.alias_or_name, projection)
            self._select_maps[key] = select_map
        return select_map

    def _table_name(self, table: exp.Table) -> str:
        name = table.name
        if table.db:
            name = f"{table.db}.{name}"
        if table.catalog:
            name = f"{table.catalog}.{name}"
        return name

    @staticmethod
    def _union(sets: t.Iterable[frozenset]) -> frozenset:
        return frozenset().union(*sets)


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
