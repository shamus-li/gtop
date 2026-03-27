from __future__ import annotations

from typing import Any, List, Optional, Set

from rich.text import Text


def _tokenize_constraint(expr: str) -> List[str]:
    tokens: List[str] = []
    index = 0

    while index < len(expr):
        char = expr[index]
        if char.isspace():
            index += 1
            continue
        if char in "()&|":
            tokens.append(char)
            index += 1
            continue
        if char == "[":
            tokens.append("(")
            index += 1
            continue
        if char == "]":
            tokens.append(")")
            index += 1
            continue

        start = index
        while index < len(expr) and expr[index] not in "()&|[] " and not expr[
            index
        ].isspace():
            index += 1
        tokens.append(expr[start:index])

    return tokens


def matches_constraint(
    features: Set[str],
    constraint: str,
    stderr_console: Optional[Any] = None,
) -> bool:
    if not constraint:
        return True

    tokens = _tokenize_constraint(constraint.lower())
    if not tokens:
        return True

    position = 0

    def parse_term() -> bool:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("Unexpected end of constraint expression")

        token = tokens[position]
        if token == "(":
            position += 1
            value = parse_expression()
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("Unmatched parenthesis in constraint expression")
            position += 1
            return value

        if token in {"&", "|", ")"}:
            raise ValueError(f"Unexpected token '{token}' in constraint expression")

        position += 1
        feature_name = token.split("*", 1)[0].strip()
        return feature_name in features

    def parse_expression() -> bool:
        nonlocal position
        value = parse_term()
        while position < len(tokens) and tokens[position] in {"&", "|"}:
            operator = tokens[position]
            position += 1
            rhs = parse_term()
            if operator == "&":
                value = value and rhs
            else:
                value = value or rhs
        return value

    try:
        result = parse_expression()
        if position != len(tokens):
            raise ValueError("Trailing tokens in constraint expression")
        return result
    except ValueError as error:
        if stderr_console is not None:
            stderr_console.print(
                Text(
                    f"Failed to evaluate constraint '{constraint}': {error}",
                    style="red",
                )
            )
        return False
