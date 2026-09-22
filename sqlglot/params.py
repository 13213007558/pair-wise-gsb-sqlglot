"""
Transpile SQL statements together with their bound parameters.

Drivers differ in how they mark bind parameters in SQL text (DB-API "paramstyles"):

- "named":   :name placeholders, values supplied as a mapping.
- "qmark":   ? placeholders, values supplied as a sequence, in order of appearance.
- "numeric": :1, :2, ... placeholders, values supplied as a sequence (1-based).

transpile_params converts a statement from one dialect to another while keeping every
value out of the SQL text: the returned SQL only contains placeholders in the requested
output style and the accompanying parameter collection lines up with them, one entry per
placeholder position, even when transpilation duplicates or reorders expressions.
"""

from __future__ import annotations

import typing as t
from collections.abc import Mapping, Sequence

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import ErrorLevel, UnsupportedError
from sqlglot.tokens import Tokenizer, TokenType

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType
    from sqlglot.parser import Parser

PARAM_STYLES = ("named", "qmark", "numeric")

_POSITION_META_KEY = "sqlglot_param_position"
_CANONICAL_PREFIX = "__sqlglot_param_"

# Cache of parser classes wrapped with placeholder position tracking, keyed by the
# original parser class. There is one parser class per dialect, so this stays small.
_TRACKING_PARSER_CACHE: dict[type, type] = {}


class TranspiledQuery(t.NamedTuple):
    """A transpiled statement and the bind parameters matching its placeholders."""

    sql: str
    params: list[t.Any] | dict[str, t.Any]


def _tracking_parser_class(parser_class: type[Parser]) -> type[Parser]:
    """
    Subclass parser_class so that every parsed exp.Placeholder records the offset of
    its introducing token ("?", ":", ...) in the source SQL. The parser alone decides
    what is a placeholder, so strings, comments, quoted identifiers and slice colons
    are never mistaken for parameters.
    """
    cached = _TRACKING_PARSER_CACHE.get(parser_class)
    if cached is not None:
        return cached

    def track(parse: t.Callable) -> t.Callable:
        def parse_and_track(self: Parser) -> t.Any:
            # _parse_placeholder matches the introducing token before dispatching,
            # so _prev is the token that starts this placeholder occurrence.
            start = self._prev.start
            result = parse(self)
            if isinstance(result, exp.Placeholder):
                result.meta[_POSITION_META_KEY] = start
            return result

        return parse_and_track

    placeholder_parsers = dict(parser_class.PLACEHOLDER_PARSERS)

    # The base parser only accepts identifiers after ":", extend it to also accept
    # numbers so that numeric (":1") placeholders are recognized.
    def parse_colon_placeholder(self: Parser) -> t.Any:
        if self._match_set(self.COLON_PLACEHOLDER_TOKENS) or self._match(TokenType.NUMBER):
            return self.expression(exp.Placeholder(this=self._prev.text))
        return None

    placeholder_parsers[TokenType.COLON] = parse_colon_placeholder

    tracking_class = type(
        f"Tracking{parser_class.__name__}",
        (parser_class,),
        {
            "PLACEHOLDER_PARSERS": {
                token: track(parse) for token, parse in placeholder_parsers.items()
            }
        },
    )
    _TRACKING_PARSER_CACHE[parser_class] = tracking_class
    return tracking_class


def _parse_statements(
    sql: str, dialect: Dialect, error_level: ErrorLevel | None
) -> list[exp.Expr | None]:
    parser_class = _tracking_parser_class(dialect.parser_class)
    opts: dict[str, t.Any] = {"dialect": dialect}
    if error_level is not None:
        opts["error_level"] = error_level
    return parser_class(**opts).parse(dialect.tokenize(sql), sql)


def _placeholder_style(placeholder: exp.Placeholder) -> str:
    name = placeholder.text("this")
    if not name:
        return "qmark"
    return "numeric" if name.isdigit() else "named"


def _collect_occurrences(
    statements: list[exp.Expr | None],
) -> list[list[exp.Placeholder]]:
    """Placeholders per statement, sorted by source text position."""
    occurrences = []
    for statement in statements:
        placeholders: dict[int, exp.Placeholder] = {}
        if statement is not None:
            if statement.find(exp.Command) is not None:
                raise UnsupportedError(
                    "Cannot transpile bind parameters for a statement that was preserved "
                    "as a Command, because its parameter positions cannot be verified."
                )
            for placeholder in statement.find_all(exp.Placeholder):
                position = placeholder.meta.get(_POSITION_META_KEY)
                if position is None:
                    raise UnsupportedError(
                        "Cannot determine the position of a placeholder in the source SQL."
                    )
                placeholders[id(placeholder)] = placeholder
        occurrences.append(sorted(placeholders.values(), key=lambda p: p.meta[_POSITION_META_KEY]))
    return occurrences


def _bind(occurrences: list[exp.Placeholder], style: str, params: t.Any) -> list[t.Any]:
    """Resolve the source value of every occurrence, in source text order."""
    if style == "named":
        if not isinstance(params, Mapping):
            raise ValueError(
                "Named placeholders require a mapping of parameter values, "
                f"got: {type(params).__name__}"
            )
        values = []
        for placeholder in occurrences:
            name = placeholder.text("this")
            if name not in params:
                raise ValueError(f"No value was provided for parameter :{name}")
            values.append(params[name])
        return values

    if (
        isinstance(params, Mapping)
        or not isinstance(params, Sequence)
        or isinstance(params, (str, bytes))
    ):
        raise ValueError(
            f"{style.capitalize()} placeholders require a sequence of parameter values, "
            f"got: {type(params).__name__}"
        )

    if style == "qmark":
        if len(params) != len(occurrences):
            raise ValueError(f"Expected {len(occurrences)} parameter values, got {len(params)}")
        return list(params)

    values = []
    for placeholder in occurrences:
        number = int(placeholder.text("this"))
        if number < 1 or number > len(params):
            raise ValueError(f"No value was provided for parameter :{number}")
        values.append(params[number - 1])
    return values


def _canonical_generator_class(dialect: Dialect) -> type:
    """
    Subclass the dialect's generator so placeholders render as :<canonical name>
    regardless of the dialect's native placeholder syntax. The canonical names make
    every placeholder occurrence in the generated SQL traceable back to its source
    parameter; they are replaced with the requested output style afterwards.
    """

    def placeholder_sql(self: t.Any, expression: exp.Placeholder) -> str:
        return f":{expression.name}"

    return type(
        f"Canonical{dialect.generator_class.__name__}",
        (dialect.generator_class,),
        {"NAMED_PLACEHOLDER_TOKEN": ":", "placeholder_sql": placeholder_sql},
    )


def _canonical_occurrences(sql: str) -> t.Iterator[tuple[int, int, str]]:
    """Yield (start, end, canonical name) for each canonical placeholder in sql."""
    tokens = Tokenizer().tokenize(sql)
    for token, following in zip(tokens, tokens[1:]):
        if (
            token.token_type == TokenType.COLON
            and following.token_type == TokenType.VAR
            and following.start == token.end + 1
            and following.text.startswith(_CANONICAL_PREFIX)
        ):
            yield token.start, following.end + 1, following.text


def transpile_params(
    sql: str | exp.Expression,
    params: Mapping[str, t.Any] | Sequence[t.Any] | None = None,
    read: DialectType = None,
    write: DialectType = None,
    input_style: str | None = None,
    output_style: str = "qmark",
    error_level: ErrorLevel | None = None,
    **opts: t.Any,
) -> list[TranspiledQuery]:
    """
    Transpile SQL with bound parameters, returning target SQL and target bind parameters.

    Values are never inlined into the generated SQL: every parameter stays behind a
    placeholder, and the returned parameter collection lines up with the placeholders
    of the returned SQL. If transpilation duplicates or reorders an expression, each
    resulting placeholder position is assigned the value of its source parameter.

    Args:
        sql: the SQL code to transpile, or an already parsed expression. Expressions
            are never mutated, so the same AST can be transpiled to multiple targets.
        params: the bound values. A mapping for named placeholders, a sequence for
            qmark/numeric placeholders. Never mutated.
        read: the source dialect used to parse the input (eg. "spark", "hive", "presto").
        write: the target dialect into which the input should be transformed.
        input_style: optional expected input style ("named", "qmark" or "numeric").
            Styles are detected automatically; passing this validates the detection.
        output_style: the placeholder style of the returned SQL and parameters:
            "named" (:name + dict), "qmark" (? + list) or "numeric" (:1 ... + list).
        error_level: the desired error level of the parser.
        **opts: other sqlglot.generator.Generator options.

    Returns:
        A list of TranspiledQuery (sql, params), one per parsed statement.

    Raises:
        ValueError: if placeholder styles are mixed, if input_style/output_style
            is invalid, or if bindings are missing or of the wrong kind.
        UnsupportedError: if a statement was preserved as a Command, in which case
            parameter positions cannot be verified.
    """
    if input_style is not None and input_style not in PARAM_STYLES:
        raise ValueError(f"Invalid input_style: {input_style!r}, expected one of {PARAM_STYLES}")
    if output_style not in PARAM_STYLES:
        raise ValueError(f"Invalid output_style: {output_style!r}, expected one of {PARAM_STYLES}")

    read_dialect = Dialect.get_or_raise(read)
    write_dialect = Dialect.get_or_raise(write) if write is not None else read_dialect

    if isinstance(sql, exp.Expression):
        # Re-render the expression so placeholder positions can be tracked exactly.
        # generate() copies by default, leaving the caller's AST untouched.
        sql = read_dialect.generate(sql)

    statements = _parse_statements(sql, read_dialect, error_level)
    occurrences = _collect_occurrences(statements)
    all_occurrences = [placeholder for statement in occurrences for placeholder in statement]

    values: list[t.Any] = []
    style = input_style
    if all_occurrences:
        styles = {_placeholder_style(placeholder) for placeholder in all_occurrences}
        if len(styles) > 1:
            raise ValueError(
                f"Mixing placeholder styles ({', '.join(sorted(styles))}) is not supported"
            )
        style = styles.pop()
        if input_style is not None and style != input_style:
            raise ValueError(f"Expected {input_style} placeholders, but found {style} placeholders")
        values = _bind(all_occurrences, style, params)
    elif params:
        raise ValueError("Parameter values were provided, but the SQL has no placeholders")

    # Give every placeholder occurrence a unique canonical name and remember its value.
    # Occurrences of the same source parameter keep distinct canonical names but share
    # their value; distinct source parameters are never merged, even if their values
    # happen to be equal.
    canonical_values: dict[str, t.Any] = {}
    canonical_names: dict[str, str] = {}
    for index, (placeholder, value) in enumerate(zip(all_occurrences, values)):
        canonical = f"{_CANONICAL_PREFIX}{index}__"
        source_name = placeholder.text("this")
        if output_style == "named":
            name = source_name if style == "named" else f"p{source_name or index + 1}"
            canonical_names[canonical] = name
        canonical_values[canonical] = value
        placeholder.set("this", canonical)

    generator_class = _canonical_generator_class(write_dialect)
    results = []
    for statement in statements:
        if statement is None:
            results.append(TranspiledQuery("", {} if output_style == "named" else []))
            continue

        intermediate = generator_class(dialect=write_dialect, **opts).generate(
            statement, copy=False
        )

        parts: list[str] = []
        positional_params: list[t.Any] = []
        named_params: dict[str, t.Any] = {}
        last = 0
        for number, (start, end, canonical) in enumerate(
            _canonical_occurrences(intermediate), start=1
        ):
            value = canonical_values[canonical]
            parts.append(intermediate[last:start])
            if output_style == "named":
                name = canonical_names[canonical]
                parts.append(f":{name}")
                named_params[name] = value
            elif output_style == "numeric":
                parts.append(f":{number}")
                positional_params.append(value)
            else:
                parts.append("?")
                positional_params.append(value)
            last = end
        parts.append(intermediate[last:])

        results.append(
            TranspiledQuery(
                "".join(parts),
                named_params if output_style == "named" else positional_params,
            )
        )

    return results
