from __future__ import annotations

import re

_REQUESTED_FEATURE_PATTERN = re.compile(r"[A-Za-z0-9_.:+-]+")


def requested_constraint_features(value: str) -> frozenset[str]:
    return frozenset(
        match.group(0).lower()
        for match in _REQUESTED_FEATURE_PATTERN.finditer(value)
        if not match.group(0).isdigit()
    )
