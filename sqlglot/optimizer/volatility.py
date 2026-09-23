from __future__ import annotations

import typing as t

from sqlglot import exp

# Volatility levels for function calls, mirroring PostgreSQL's function volatility categories:
#
# - IMMUTABLE: the function cannot modify the database and always returns the same result for
#   the same arguments. It can be constant-folded, deduplicated, moved and copied freely.
# - STABLE: the function cannot modify the database and returns the same result for the same
#   arguments within a single statement. It can be shared/deduplicated within a statement, but
#   it can never be treated as a cross-statement constant.
# - VOLATILE: the function can return a different value on every call or have side effects. It
#   cannot be common-subexpression-eliminated, constant-folded, deleted, or copied onto more
#   rows than it appeared on originally.
IMMUTABLE = "immutable"
STABLE = "stable"
VOLATILE = "volatile"

LEVELS = (IMMUTABLE, STABLE, VOLATILE)

# Functions built into sqlglot that are known to be volatile
BUILTIN_VOLATILE = (exp.Rand, exp.Randn)

# Expression types representing calls to functions sqlglot doesn't know. Their volatility is
# unknown, so they are treated conservatively as volatile unless configured otherwise.
UNKNOWN_FUNCTIONS = (exp.Anonymous, exp.AnonymousAggFunc)

# A mapping of (case-insensitive) function names to volatility levels. A name can be qualified
# with a schema (e.g. "my_schema.my_func"), in which case it only matches calls qualified the
# same way; an unqualified name only matches unqualified calls.
FunctionVolatility = dict[str, str]


def normalize_function_volatility(
    config: t.Mapping[str, str] | None,
) -> FunctionVolatility:
    """
    Validate and normalize a user-provided function volatility config.

    Args:
        config: a mapping of function names to "immutable", "stable" or "volatile".

    Returns:
        The normalized mapping, with uppercased names and lowercased levels.
    """
    normalized = {}

    for name, level in (config or {}).items():
        level = level.lower()
        if level not in LEVELS:
            raise ValueError(
                f"Invalid volatility {level!r} for function {name!r}. "
                f"Expected one of: {', '.join(LEVELS)}."
            )
        normalized[name.upper()] = level

    return normalized


def _qualified_name(func: exp.Func) -> str | None:
    """The dot-qualified name of a function call, e.g. MY_SCHEMA.MY_FUNC, if it is qualified."""
    node: exp.Expr = func
    while (
        node.parent is not None
        and isinstance(node.parent, exp.Dot)
        and node.parent.expression is node
    ):
        node = node.parent

    if node is func:
        return None

    parts = []
    while isinstance(node, exp.Dot):
        parts.append(node.expression.name)
        node = node.this
    parts.append(node.name)

    return ".".join(reversed(parts)).upper()


def function_volatility(func: exp.Func, config: t.Mapping[str, str] | None = None) -> str:
    """
    Resolve the volatility of a single function call.

    A user-provided config takes precedence. Otherwise, known volatile functions (e.g. RAND)
    and unknown functions (conservatively) are volatile, and every other known function is
    treated as immutable, matching sqlglot's historical behavior.
    """
    if config:
        qualified = _qualified_name(func)
        if qualified:
            level = config.get(qualified)
        else:
            bare = func.name if isinstance(func, UNKNOWN_FUNCTIONS) else func.sql_name()
            level = config.get(bare.upper())
        if level:
            return level

    if isinstance(func, (*BUILTIN_VOLATILE, *UNKNOWN_FUNCTIONS)):
        return VOLATILE

    return IMMUTABLE


def expression_volatility(expression: exp.Expr, config: t.Mapping[str, str] | None = None) -> str:
    """
    Derive the volatility of an expression from the function calls it contains, including
    calls nested in subqueries and CTEs. The result is the strictest level found.
    """
    volatility = IMMUTABLE

    for node in expression.walk():
        if isinstance(node, exp.Func):
            level = function_volatility(node, config)
            if level == VOLATILE:
                return VOLATILE
            if level == STABLE:
                volatility = STABLE

    return volatility


def is_volatile(expression: exp.Expr, config: t.Mapping[str, str] | None = None) -> bool:
    """Whether an expression contains any volatile function calls."""
    return expression_volatility(expression, config) == VOLATILE
