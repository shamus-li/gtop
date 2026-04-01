from __future__ import annotations

import shlex
from typing import Optional, Sequence


def command_option_value(command: str, option_names: Sequence[str]) -> Optional[str]:
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token in option_names and index + 1 < len(tokens):
            return tokens[index + 1]
        for name in option_names:
            prefix = f"{name}="
            if token.startswith(prefix):
                return token[len(prefix) :]
    return None


def remove_command_options(command: str, option_names: Sequence[str]) -> str:
    tokens = shlex.split(command)
    filtered: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in option_names:
            skip_next = True
            continue
        if any(token.startswith(f"{name}=") for name in option_names):
            continue
        filtered.append(token)
    return shlex.join(filtered)


def override_command_option(command: str, option_names: Sequence[str], value: str) -> str:
    existing = command_option_value(command, option_names)
    if existing == value:
        return command
    filtered = shlex.split(remove_command_options(command, option_names))
    filtered.extend([option_names[0], value])
    return shlex.join(filtered)
