from __future__ import annotations

from typing import Optional

from .runner import Command


def set_value_option(
    command: Command,
    option_names: tuple[str, ...],
    value: Optional[str],
) -> Command:
    filtered: list[str] = []
    skip_next = False
    for token in command:
        if skip_next:
            skip_next = False
            continue
        if token in option_names:
            skip_next = True
            continue
        if any(token.startswith(f"{name}=") for name in option_names):
            continue
        filtered.append(token)
    if value is not None:
        filtered.extend((option_names[0], value))
    return tuple(filtered)
