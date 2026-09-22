from __future__ import annotations

import re
import typing as t

from dataclasses import dataclass, field
from decimal import Decimal

from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import UnsupportedError

if t.TYPE_CHECKING:
    from typing_extensions import Unpack

    from sqlglot._typing import GeneratorNoDialectArgs

# Dialects whose DB-API drivers expect "?" placeholders
QMARK_DIALECTS = {"sqlite"}

# Dialects whose DB-API drivers expect numbered "$n" placeholders
NUMBERED_DIALECTS = {"postgres"}

# Marks the position of each extracted parameter in the generated SQL, so that the
# parameter list can be ordered by its appearance in the final SQL text
_MARKER_PATTERN = re.compile("\x00(\\d+)\x01")


@dataclass
class SkippedLiteral:
    """A literal that was intentionally left in the SQL text instead of becoming a parameter."""

    value: t.Any
    reason: str


@dataclass
class ParameterizedQuery:
    """The result of extracting bind parameters from a SQL statement."""

    expression: exp.Expr
    """A new AST with placeholders in place of the extracted literals."""

    sql: str
    """The generated SQL, ready to be handed to a DB-API cursor."""

    params: list[t.Any] = field(default_factory=list)
    """The parameter values, ordered by their appearance in `sql`."""

    skipped: list[SkippedLiteral] = field(default_factory=list)
    """Literals that were kept in the SQL text because their position cannot be parameterized."""


def parameterize(
    sql: str | exp.Expr,
    dialect: DialectType = None,
    **opts: Unpack[GeneratorNoDialectArgs],
) -> ParameterizedQuery:
    """
    Extracts literal values from a SELECT statement into an ordered parameter list.

    Literals in projections, WHERE clauses and other bindable positions are replaced by
    placeholders ("?" for SQLite, "$n" for PostgreSQL), while literals in positions that
    cannot be parameterized (e.g. ORDER BY ordinals, type lengths/precisions, interval
    values, window frame bounds) are preserved and reported in the result. The input
    AST is never mutated.

    Args:
        sql: the SQL code string or expression to parameterize.
        dialect: the target dialect, which determines parsing and the placeholder style.
            Currently only "sqlite" and "postgres" are supported.
        **opts: other `sqlglot.generator.Generator` options.

    Returns:
        A ParameterizedQuery containing the new AST, the generated SQL, the ordered
        parameter values and the literals that were intentionally kept in place.

    Raises:
        UnsupportedError: if the dialect is not supported, if the input is not a single
            SELECT statement, or if it already contains placeholders.
    """
    target = Dialect.get_or_raise(dialect)
    name = target.__class__.__name__.lower()

    if name in QMARK_DIALECTS:
        numbered = False
    elif name in NUMBERED_DIALECTS:
        numbered = True
    else:
        supported = sorted(QMARK_DIALECTS | NUMBERED_DIALECTS)
        raise UnsupportedError(
            f"parameterize() does not support dialect {name!r}, supported dialects: {supported}"
        )

    if isinstance(sql, str):
        statements = target.parse(sql)
        if len(statements) != 1 or statements[0] is None:
            raise UnsupportedError("parameterize() expects a single SQL statement")
        tree = statements[0]
    else:
        tree = sql

    if tree.find(exp.Placeholder, exp.Parameter):
        raise UnsupportedError(
            "parameterize() does not accept SQL that already contains placeholders or parameters"
        )

    if not isinstance(tree, exp.Query):
        raise UnsupportedError("parameterize() only supports SELECT statements")

    tree = tree.copy()

    replacements: list[tuple[exp.Expr, t.Any]] = []
    skipped: list[SkippedLiteral] = []
    _collect(tree, None, None, replacements, skipped)

    values: list[t.Any] = []
    slots: dict[int, int] = {}
    for node, value in replacements:
        placeholder = exp.Placeholder()
        if node.comments:
            placeholder.comments = list(node.comments)
        node.replace(placeholder)
        slots[id(placeholder)] = len(values)
        values.append(value)

    generator_class: t.Any = target.generator_class

    class ParameterizingGenerator(generator_class):
        def placeholder_sql(self, expression: exp.Placeholder) -> str:
            return f"\x00{slots[id(expression)]}\x01"

    marked = ParameterizingGenerator(dialect=target, **opts).generate(tree, copy=False)

    params: list[t.Any] = []
    parts: list[str] = []
    cursor = 0
    for match in _MARKER_PATTERN.finditer(marked):
        parts.append(marked[cursor : match.start()])
        params.append(values[int(match.group(1))])
        parts.append(f"${len(params)}" if numbered else "?")
        cursor = match.end()
    parts.append(marked[cursor:])

    if len(params) != len(values):
        raise UnsupportedError("Could not determine the parameter order of the generated SQL")

    return ParameterizedQuery(expression=tree, sql="".join(parts), params=params, skipped=skipped)


def _collect(
    node: exp.Expr,
    parent: exp.Expr | None,
    context: str | None,
    replacements: list[tuple[exp.Expr, t.Any]],
    skipped: list[SkippedLiteral],
) -> None:
    if isinstance(node, exp.Literal):
        _bind_or_skip(node, node.to_py(), parent, context, replacements, skipped)
        return

    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal) and node.this.is_number:
        value = node.this.to_py()
        if isinstance(value, (int, Decimal)):
            _bind_or_skip(node, -value, parent, context, replacements, skipped)
        return

    if isinstance(node, exp.DataType):
        context = "data type length/precision cannot be parameterized"
    elif isinstance(node, exp.Interval):
        context = "interval value cannot be parameterized"
    elif isinstance(node, exp.WindowSpec):
        context = "window frame bound cannot be parameterized"

    for child in node.iter_expressions():
        _collect(child, node, context, replacements, skipped)


def _bind_or_skip(
    node: exp.Expr,
    value: t.Any,
    parent: exp.Expr | None,
    context: str | None,
    replacements: list[tuple[exp.Expr, t.Any]],
    skipped: list[SkippedLiteral],
) -> None:
    value = _param_value(value)

    if context:
        skipped.append(SkippedLiteral(value=value, reason=context))
    elif _is_ordinal(node, parent):
        skipped.append(
            SkippedLiteral(
                value=value,
                reason="positional reference in ORDER BY / GROUP BY cannot be parameterized",
            )
        )
    else:
        replacements.append((node, value))


def _is_ordinal(node: exp.Expr, parent: exp.Expr | None) -> bool:
    return isinstance(parent, (exp.Ordered, exp.Group)) and _is_integer(node)


def _is_integer(node: exp.Expr) -> bool:
    literal = node.this if isinstance(node, exp.Neg) else node
    return (
        isinstance(literal, exp.Literal) and literal.is_number and isinstance(literal.to_py(), int)
    )


def _param_value(value: t.Any) -> t.Any:
    # DB-API drivers such as sqlite3 cannot bind Decimal values
    return float(value) if isinstance(value, Decimal) else value
