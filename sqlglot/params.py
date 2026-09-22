"""
Transpile SQL statements together with their bound parameter values.

Services that accept SQL with bind parameters in one dialect often need to
execute it through a driver for another dialect. sqlglot.transpile only
returns the target SQL, which is not enough: the target driver may expect a
different placeholder style, and transpilation may duplicate or reorder the
expressions the parameters belong to. The functions in this module return the
target SQL and the target bind parameters together, so values are always
passed to the driver separately from the SQL text.

Three parameter styles are supported, mirroring the DB-API paramstyles:

- "named":   :name, values are given as a mapping from name to value.
- "qmark":   ?, values are given as a sequence in order of appearance.
- "numeric": :1, values are given as a sequence indexed by number.
"""

from __future__ import annotations

import re
import typing as t

from collections.abc import Mapping, Sequence

from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import ErrorLevel, ParamError, ParseError
from sqlglot.tokens import Token, TokenType

__all__ = ["parse_params", "generate_params", "transpile_params"]

ParamStyle = t.Literal["named", "qmark", "numeric"]

PARAM_STYLES: tuple[str, ...] = ("named", "qmark", "numeric")

_PARAM_PREFIX = "__sqlglot_param_"

# Placeholders inside brackets or braces are ambiguous with slice and
# struct/map literal syntax (e.g. arr[1:2], {'a': 1}), so colons are only
# treated as parameter markers at container depth zero.
_CONTAINER_START = (TokenType.L_BRACKET, TokenType.L_BRACE)
_CONTAINER_END = (TokenType.R_BRACKET, TokenType.R_BRACE)

_FOREIGN_LABELS = {
    "named": "named (:name)",
    "qmark": "qmark (?)",
    "numeric": "numeric (:N)",
    "parameter": "parameter (@name)",
}


def parse_params(
    sql: str,
    read: DialectType = None,
    input_style: ParamStyle = "named",
    error_level: ErrorLevel | None = None,
) -> tuple[exp.Expr, list[t.Any]]:
    """
    Parses a SQL statement that contains bind parameters.

    The returned expression contains a uniquely named placeholder for every
    parameter occurrence and can be transpiled to any number of targets with
    generate_params. The returned keys identify the source parameter of each
    occurrence: the name for "named" parameters, the 1-based position for
    "qmark" parameters and the number for "numeric" parameters.

    Args:
        sql: the SQL statement to parse.
        read: the source dialect.
        input_style: the placeholder style used by the SQL statement.
        error_level: the desired error level of the parser.

    Returns:
        A tuple of the parsed expression and the parameter keys, in order of
        occurrence.
    """
    _check_style(input_style)
    if _PARAM_PREFIX in sql:
        raise ParamError(f"SQL contains the reserved parameter prefix '{_PARAM_PREFIX}'")

    dialect = Dialect.get_or_raise(read)
    occurrences, foreign = _scan_params(dialect.tokenize(sql), input_style)
    if foreign:
        styles = ", ".join(_FOREIGN_LABELS[style] for style in sorted(set(foreign)))
        raise ParamError(
            f"Mixed parameter styles are not supported: found {styles} "
            f"but input_style='{input_style}'"
        )

    if occurrences:
        parts = []
        pos = 0
        for i, (start, end, _) in enumerate(occurrences):
            parts.append(sql[pos:start])
            parts.append(f":{_PARAM_PREFIX}{i}")
            pos = end
        parts.append(sql[pos:])
        sql = "".join(parts)

    opts: dict[str, t.Any] = {"error_level": error_level} if error_level else {}
    expressions = dialect.parse(sql, **opts)
    if not expressions or expressions[0] is None:
        raise ParseError(f"No expression was parsed from '{sql}'")
    if len(expressions) > 1:
        raise ParamError("Expected a single SQL statement")

    expression = expressions[0]
    _validate_expression(expression, len(occurrences))
    return expression, [key for _, _, key in occurrences]


def generate_params(
    expression: exp.Expr,
    keys: t.Sequence[t.Any],
    params: Mapping[str, t.Any] | Sequence[t.Any] | None = None,
    write: DialectType = None,
    input_style: ParamStyle = "named",
    output_style: ParamStyle | None = None,
    **opts: t.Any,
) -> tuple[str, list[t.Any] | dict[str, t.Any]]:
    """
    Generates SQL and bind parameters for an expression parsed by parse_params.

    The expression is not modified, so it can be reused to target any number of
    dialects. Every placeholder position of the generated SQL receives the
    value of its source parameter: parameters that the transpilation duplicated
    appear once per position, and distinct source parameters are never merged,
    even if their values are equal.

    Args:
        expression: an expression returned by parse_params.
        keys: the parameter keys returned by parse_params.
        params: the bound values; a mapping for "named" input, a sequence for
            "qmark" and "numeric" input.
        write: the target dialect.
        input_style: the placeholder style the expression was parsed with.
        output_style: the placeholder style to generate, defaults to
            input_style.
        **opts: other sqlglot.generator.Generator options.

    Returns:
        A tuple of the generated SQL and its bind parameters; a mapping for
        "named" output, a list of values in placeholder order otherwise.
    """
    _check_style(input_style)
    output_style = input_style if output_style is None else output_style
    _check_style(output_style)

    values = _resolve_values(keys, params, input_style)
    dialect = Dialect.get_or_raise(write)
    _validate_expression(expression, len(keys))
    sql = dialect.generate(expression, copy=True, **opts)

    if not keys:
        return sql, {} if output_style == "named" else []

    renderings = {
        dialect.generate(exp.Placeholder(this=f"{_PARAM_PREFIX}{i}"), copy=False, **opts): i
        for i in range(len(keys))
    }
    pattern = re.compile("|".join(re.escape(marker) for marker in renderings))

    out_params: list[t.Any] | dict[str, t.Any] = {} if output_style == "named" else []
    parts = []
    pos = 0
    ordinal = 0
    found = set()

    for match in pattern.finditer(sql):
        source = renderings[match.group(0)]
        found.add(source)
        parts.append(sql[pos : match.start()])
        if output_style == "qmark":
            parts.append("?")
            t.cast(list, out_params).append(values[source])
        elif output_style == "numeric":
            ordinal += 1
            parts.append(f":{ordinal}")
            t.cast(list, out_params).append(values[source])
        else:
            name = keys[source] if input_style == "named" else f"p{source + 1}"
            parts.append(f":{name}")
            t.cast(dict, out_params)[name] = values[source]
        pos = match.end()

    if len(found) != len(keys):
        raise ParamError(
            "Transpilation dropped bind parameter(s), cannot map values to the generated SQL"
        )

    parts.append(sql[pos:])
    return "".join(parts), out_params


def transpile_params(
    sql: str,
    read: DialectType = None,
    write: DialectType = None,
    params: Mapping[str, t.Any] | Sequence[t.Any] | None = None,
    input_style: ParamStyle = "named",
    output_style: ParamStyle | None = None,
    error_level: ErrorLevel | None = None,
    **opts: t.Any,
) -> tuple[str, list[t.Any] | dict[str, t.Any]]:
    """
    Transpiles a SQL statement with bind parameters into the target dialect.

    Args:
        sql: the SQL statement to transpile.
        read: the source dialect.
        write: the target dialect, defaults to the source dialect.
        params: the bound values; a mapping for "named" input, a sequence for
            "qmark" and "numeric" input.
        input_style: the placeholder style used by the SQL statement.
        output_style: the placeholder style to generate, defaults to
            input_style.
        error_level: the desired error level of the parser.
        **opts: other sqlglot.generator.Generator options.

    Returns:
        A tuple of the transpiled SQL and its bind parameters; a mapping for
        "named" output, a list of values in placeholder order otherwise.
    """
    expression, keys = parse_params(
        sql, read=read, input_style=input_style, error_level=error_level
    )
    return generate_params(
        expression,
        keys,
        params,
        write=read if write is None else write,
        input_style=input_style,
        output_style=output_style,
        **opts,
    )


def _check_style(style: str) -> None:
    if style not in PARAM_STYLES:
        raise ParamError(f"Unknown parameter style '{style}', expected one of: {PARAM_STYLES}")


def _scan_params(tokens: list[Token], style: str) -> tuple[list[tuple[int, int, t.Any]], list[str]]:
    """
    Locates bind parameters of the given style in a token stream.

    Returns the (start, end, key) character span of every occurrence, plus the
    styles of any other parameter-like tokens, so mixed input can be rejected.
    Strings, comments and quoted identifiers are single tokens, so their
    contents are never mistaken for parameters.
    """
    occurrences: list[tuple[int, int, t.Any]] = []
    foreign: list[str] = []
    depth = 0
    position = 0
    size = len(tokens)

    for i, token in enumerate(tokens):
        token_type = token.token_type
        if token_type in _CONTAINER_START:
            depth += 1
        elif token_type in _CONTAINER_END:
            depth = max(depth - 1, 0)
        elif token_type is TokenType.PLACEHOLDER:
            if style == "qmark":
                position += 1
                occurrences.append((token.start, token.end + 1, position))
            else:
                foreign.append("qmark")
        elif token_type is TokenType.PARAMETER:
            foreign.append("parameter")
        elif token_type is TokenType.COLON and not depth:
            following = tokens[i + 1] if i + 1 < size else None
            if following and following.start == token.end + 1:
                if following.token_type is TokenType.VAR:
                    if style == "named":
                        occurrences.append((token.start, following.end + 1, following.text))
                    else:
                        foreign.append("named")
                elif following.token_type is TokenType.NUMBER:
                    if style == "numeric":
                        if not following.text.isdigit():
                            raise ParamError(f"Invalid numeric parameter ':{following.text}'")
                        occurrences.append((token.start, following.end + 1, int(following.text)))
                    else:
                        foreign.append("numeric")

    return occurrences, foreign


def _validate_expression(expression: exp.Expr, count: int) -> None:
    if expression.find(exp.Command):
        raise ParamError(
            "Cannot transpile bind parameters for Command expressions, their text is "
            "preserved as-is so parameter order cannot be verified"
        )
    if expression.find(exp.Parameter):
        raise ParamError(
            "Unsupported parameter syntax, use named (:name), qmark (?) or numeric (:N) "
            "parameters instead"
        )

    expected = {f"{_PARAM_PREFIX}{i}" for i in range(count)}
    found = {placeholder.text("this") for placeholder in expression.find_all(exp.Placeholder)}
    unexpected = found - expected
    if unexpected:
        raise ParamError(
            "Mixed parameter styles are not supported, found unexpected "
            f"placeholder(s): {', '.join(sorted(unexpected))}"
        )
    if expected - found:
        raise ParamError("Parameter(s) could not be parsed in this position")


def _resolve_values(
    keys: t.Sequence[t.Any],
    params: Mapping[str, t.Any] | Sequence[t.Any] | None,
    style: str,
) -> list[t.Any]:
    if not keys:
        if params:
            raise ParamError("Bind parameters were provided but the SQL contains no placeholders")
        return []

    if style == "named":
        if not isinstance(params, Mapping):
            raise ParamError("Named parameters require a mapping of names to values")
        expected = set(keys)
        missing = expected - set(params)
        if missing:
            raise ParamError(f"Missing value(s) for parameter(s): {', '.join(sorted(missing))}")
        extra = set(params) - expected
        if extra:
            raise ParamError(f"Unused value(s) for parameter(s): {', '.join(sorted(extra))}")
        return [params[key] for key in keys]

    if isinstance(params, (str, bytes)) or not isinstance(params, Sequence):
        raise ParamError(f"'{style}' parameters require a sequence of values")

    if style == "qmark":
        if len(params) != len(keys):
            raise ParamError(f"Expected {len(keys)} parameter value(s), got {len(params)}")
        return list(params)

    used = set(keys)
    missing_numbers = [key for key in sorted(used) if key < 1 or key > len(params)]
    if missing_numbers:
        raise ParamError(
            f"Missing value(s) for parameter(s): "
            f"{', '.join(f':{key}' for key in missing_numbers)}"
        )
    unused = set(range(1, len(params) + 1)) - used
    if unused:
        raise ParamError(
            f"Unused value(s) for parameter(s): {', '.join(f':{key}' for key in sorted(unused))}"
        )
    return [params[key - 1] for key in keys]
