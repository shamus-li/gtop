from __future__ import annotations

import math
import re
from typing import List

from .models import CpuInfo, GpuInfo, JobUsage, MemoryInfo


def _split_outside_parens(value: str, delimiter: str = ",") -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0

    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(value[start:index])
            start = index + 1

    parts.append(value[start:])
    return parts


def _split_gres_components(item: str) -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0

    for index, char in enumerate(item):
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == ":" and depth == 0:
            parts.append(item[start:index])
            start = index + 1

    parts.append(item[start:])
    return [part.strip() for part in parts if part.strip()]


def _extract_count(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)(?:\([^)]*\))?\s*", value)
    if not match:
        raise ValueError(f"Invalid GRES count '{value}'")
    return int(match.group(1))


def _used_shard_devices(value: str) -> int:
    match = re.search(r"\(([^()]*)\)\s*$", value)
    if not match:
        return 0
    return sum(
        1
        for item in match.group(1).split(",")
        if _extract_count(item.split("/", 1)[0]) > 0
    )


def parse_gpu(gres: str, gres_used: str = "") -> GpuInfo:
    no_capacity = not gres or gres.strip().lower() in {"(null)", "null", "none"}
    no_usage = not gres_used or gres_used.strip().lower() in {"(null)", "null", "none"}
    if no_capacity and no_usage:
        return GpuInfo()
    if no_capacity:
        gres = ""

    gpu_types: List[str] = []
    total = 0
    shards = 0
    used_gpus = 0
    used_shards = 0
    shard_gpus_used = 0

    for item in _split_outside_parens(gres):
        item = item.strip()
        if not item:
            continue

        parts = _split_gres_components(item)
        if not parts:
            continue

        if parts[0] == "gpu" and len(parts) < 2:
            raise ValueError(f"Invalid GPU GRES '{item}'")
        if len(parts) >= 2 and parts[0] == "gpu":
            if len(parts) >= 3:
                gpu_type = ":".join(parts[1:-1])
                count = _extract_count(parts[-1])
                total += count
                if gpu_type:
                    gpu_types.append(gpu_type)
            elif len(parts) == 2:
                total += _extract_count(parts[1])
        elif parts[0] == "shard":
            if len(parts) < 2:
                raise ValueError(f"Invalid shard GRES '{item}'")
            gpu_type = ":".join(parts[1:-1]) or "gpu"
            shard_count = _extract_count(parts[-1])
            if gpu_type != "gpu" and f"{gpu_type}_shard" not in gpu_types:
                gpu_types.append(f"{gpu_type}_shard")
            shards += shard_count

    if not no_usage:
        for item in _split_outside_parens(gres_used):
            item = item.strip()
            if not item:
                continue

            parts = _split_gres_components(item)
            if not parts:
                continue

            if parts[0] == "gpu":
                if len(parts) < 2:
                    raise ValueError(f"Invalid used GPU GRES '{item}'")
                used_gpus += _extract_count(parts[-1])
            elif parts[0] == "shard":
                if len(parts) < 2:
                    raise ValueError(f"Invalid used shard GRES '{item}'")
                used_shards += _extract_count(parts[-1])
                shard_gpus_used += _used_shard_devices(parts[-1])

    if total <= 0 and shards <= 0 and (used_gpus > 0 or used_shards > 0):
        raise ValueError("Observed GPU or shard usage exceeds capacity 0")
    if total <= 0 and shards <= 0:
        return GpuInfo()
    if shards > 0 and total <= 0:
        raise ValueError("Shard capacity requires GPU capacity")
    if used_gpus > total:
        raise ValueError(f"Observed GPU usage {used_gpus} exceeds capacity {total}")
    if used_shards > shards:
        raise ValueError(
            f"Observed shard usage {used_shards} exceeds capacity {shards}"
        )
    if shard_gpus_used > total:
        raise ValueError(
            f"Observed shard device usage {shard_gpus_used} exceeds GPU capacity {total}"
        )
    if shards > 0:
        shards_per_gpu = shards / total
        occupied_shards = used_shards + used_gpus * shards_per_gpu
        if occupied_shards > shards:
            raise ValueError(
                f"Observed shard usage {occupied_shards:g} exceeds capacity {shards}"
            )
        if used_gpus + shard_gpus_used > total:
            raise ValueError(
                f"Observed GPU and shard device usage exceeds GPU capacity {total}"
            )

    if shards > 0 or any(item.endswith("_shard") for item in gpu_types):
        shard_types = [item for item in gpu_types if item.endswith("_shard")]
        if shard_types:
            display_type = (
                "Shard("
                + "|".join(
                    dict.fromkeys(item.replace("_shard", "") for item in shard_types)
                )
                + ")"
            )
        else:
            display_type = "Shard(gpu)"
    elif len(gpu_types) > 1:
        display_type = "(" + "|".join(dict.fromkeys(gpu_types)) + ")"
    elif gpu_types:
        display_type = gpu_types[0]
    else:
        display_type = "gpu"

    return GpuInfo(
        type=display_type,
        num=total,
        shards=shards,
        used=used_gpus,
        used_shards=used_shards,
        shard_gpus_used=shard_gpus_used,
    )


def parse_cpu(cpu_state: str) -> CpuInfo:
    parts = cpu_state.split("/")
    if len(parts) != 4 or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid CPU state '{cpu_state}'")
    allocated, idle, other, total = (int(part) for part in parts)
    if allocated + idle + other > total:
        raise ValueError(f"CPU state '{cpu_state}' exceeds total {total}")
    return CpuInfo(idle=idle, total=total, other=other)


def _parse_numeric(value: str) -> float:
    if not value:
        return 0.0

    lowered = value.strip().lower()
    if lowered in {"(null)", "none"}:
        return 0.0

    try:
        number = float(lowered)
    except ValueError:
        raise ValueError(f"Invalid numeric value '{value}'") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Value '{value}' must be a non-negative finite number")
    return number


def _parse_tres_value(value: str, default_unit: str = "") -> float:
    stripped = value.strip()
    if not stripped:
        return 0.0

    unit = default_unit.upper()
    if stripped[-1].isalpha():
        unit = stripped[-1].upper()
        stripped = stripped[:-1]

    try:
        number = float(stripped)
    except ValueError:
        raise ValueError(f"Invalid resource value '{value}'") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(
            f"Resource value '{value}' must be a non-negative finite number"
        )

    if unit in {"", "G"}:
        return number
    if unit == "M":
        return number / 1024.0
    if unit == "K":
        return number / (1024.0 * 1024.0)
    if unit == "T":
        return number * 1024.0
    if unit == "P":
        return number * 1024.0 * 1024.0
    raise ValueError(f"Unsupported resource unit '{unit}' in '{value}'")


def parse_mem(alloc_mem: str, total_mem: str) -> MemoryInfo:
    alloc = _parse_numeric(alloc_mem)
    total = _parse_numeric(total_mem)
    if alloc > total:
        raise ValueError(f"Allocated memory {alloc:g} exceeds total {total:g}")
    return MemoryInfo(idle=total - alloc, total=total)


def parse_usage(alloc_tres: str) -> JobUsage:
    if not alloc_tres:
        return JobUsage()

    cpu = 0.0
    mem = 0.0
    typed_gpu = 0.0
    typed_shard = 0.0
    generic_gpu = None
    generic_shard = None
    for part in alloc_tres.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "cpu":
            cpu = _parse_tres_value(value)
        elif key == "mem":
            mem = _parse_tres_value(value, default_unit="G")
        elif key == "gres/gpu":
            generic_gpu = _parse_tres_value(value)
        elif key.startswith("gres/gpu:"):
            typed_gpu += _parse_tres_value(value)
        elif key == "gres/shard":
            generic_shard = _parse_tres_value(value)
        elif key.startswith("gres/shard:"):
            typed_shard += _parse_tres_value(value)
    return JobUsage(
        cpu=cpu,
        gpu=typed_gpu if generic_gpu is None else generic_gpu,
        mem=mem,
        shard=typed_shard if generic_shard is None else generic_shard,
    )
