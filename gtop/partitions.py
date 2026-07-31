from __future__ import annotations

SEMANTIC_PARTITIONS = ("priority", "gpu", "default")


def normalize_partition_name(value: str) -> str:
    return value.strip().rstrip("*")


def partition_names(value: str) -> tuple[str, ...]:
    return tuple(
        partition
        for partition in (
            normalize_partition_name(name)
            for name in value.split(",")
        )
        if partition
    )


def partition_bucket(name: str) -> str | None:
    lower_name = name.lower()
    if lower_name == "other":
        return None
    if "default" in lower_name:
        return "default"
    if "gpu" in lower_name:
        return "gpu"
    return "priority"
