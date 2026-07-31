from __future__ import annotations

import re
from typing import AbstractSet, Callable

ConstraintMatcher = Callable[[AbstractSet[str]], bool]
_FEATURE_PATTERN = re.compile(r"^[A-Za-z0-9_.:+-]+$")
_REQUESTED_FEATURE_PATTERN = re.compile(r"[A-Za-z0-9_.:+-]+")


class ConstraintSyntaxError(ValueError):
    pass


def normalize_constraint_feature(value: str) -> str:
    feature = value.strip().lower()
    if not feature:
        raise ConstraintSyntaxError("constraint feature must not be empty")
    if not _FEATURE_PATTERN.fullmatch(feature):
        raise ConstraintSyntaxError(
            "constraint features must be exact names without boolean operators"
        )
    return feature


def compile_constraint(constraint: str) -> ConstraintMatcher:
    required = frozenset(
        normalize_constraint_feature(value)
        for value in constraint.split(",")
    )

    def matches(features: AbstractSet[str]) -> bool:
        return required.issubset(feature.lower() for feature in features)

    return matches


def requested_constraint_features(value: str) -> frozenset[str]:
    return frozenset(
        match.group(0).lower()
        for match in _REQUESTED_FEATURE_PATTERN.finditer(value)
        if not match.group(0).isdigit()
    )
