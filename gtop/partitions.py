from __future__ import annotations

from typing import Iterable

SEMANTIC_PARTITIONS = ("priority", "gpu", "default")


def partition_bucket(name: str) -> str | None:
    lower_name = name.lower()
    if "default" in lower_name:
        return "default"
    if "gpu" in lower_name:
        return "gpu"
    if "priority" in lower_name:
        return "priority"
    return None


def uses_semantic_partitions(names: Iterable[str]) -> bool:
    normalized = {name for name in names if name}
    return bool(normalized) and all(partition_bucket(name) is not None for name in normalized)

