from __future__ import annotations

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
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else 0


def parse_gpu(gres: str, gres_used: str = "") -> GpuInfo:
    if not gres or gres.strip().lower() in {"(null)", "null", "none"}:
        return GpuInfo()

    gpu_types: List[str] = []
    total = 0
    shards = 0
    used_gpus = 0
    used_shards = 0

    for item in _split_outside_parens(gres):
        item = item.strip()
        if not item:
            continue

        parts = _split_gres_components(item)
        if not parts:
            continue

        if len(parts) >= 2 and parts[0] == "gpu":
            if len(parts) >= 3:
                gpu_type = ":".join(parts[1:-1])
                count = _extract_count(parts[-1])
                total += count
                if gpu_type:
                    gpu_types.append(gpu_type)
            elif len(parts) == 2:
                total += _extract_count(parts[1])
        elif parts[0] == "shard" and len(parts) >= 3:
            gpu_type = ":".join(parts[1:-1]) or "gpu"
            shard_count = _extract_count(parts[-1])
            if gpu_type != "gpu" and f"{gpu_type}_shard" not in gpu_types:
                gpu_types.append(f"{gpu_type}_shard")
            shards = shard_count

    if gres_used and gres_used.strip().lower() not in {"(null)", "null", "none"}:
        for item in _split_outside_parens(gres_used):
            item = item.strip()
            if not item:
                continue

            parts = _split_gres_components(item)
            if not parts:
                continue

            if parts[0] == "gpu":
                used_gpus += _extract_count(parts[-1])
            elif parts[0] == "shard":
                used_shards += _extract_count(parts[-1])

    if shards > 0 or any(item.endswith("_shard") for item in gpu_types):
        shard_types = [item for item in gpu_types if item.endswith("_shard")]
        if shard_types:
            display_type = "Shard(" + "|".join(
                dict.fromkeys(item.replace("_shard", "") for item in shard_types)
            ) + ")"
        else:
            display_type = f"Shard({shards})"
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
    )


def parse_cpu(cpu_state: str) -> CpuInfo:
    parts = cpu_state.split("/")
    idle = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return CpuInfo(idle=idle)


def _parse_numeric(value: str) -> float:
    if not value:
        return 0.0

    lowered = value.strip().lower()
    if lowered in {"(null)", "none"}:
        return 0.0

    match = re.search(r"\d+(?:\.\d+)?", lowered)
    return float(match.group(0)) if match else 0.0


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
        return 0.0

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
    return 0.0


def parse_mem(alloc_mem: str, total_mem: str) -> MemoryInfo:
    alloc = _parse_numeric(alloc_mem)
    total = _parse_numeric(total_mem)
    return MemoryInfo(idle=max(total - alloc, 0.0), total=total)


def parse_usage(alloc_tres: str) -> JobUsage:
    usage = JobUsage()
    if not alloc_tres:
        return usage

    seen_generic_gpu = False
    seen_generic_shard = False
    for part in alloc_tres.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key == "cpu":
            usage.cpu = _parse_tres_value(value)
        elif key == "mem":
            usage.mem = _parse_tres_value(value, default_unit="G")
        elif key == "gres/gpu":
            usage.gpu = _parse_tres_value(value)
            seen_generic_gpu = True
        elif key.startswith("gres/gpu:") and not seen_generic_gpu:
            usage.gpu += _parse_tres_value(value)
        elif key == "gres/shard":
            usage.shard = _parse_tres_value(value)
            seen_generic_shard = True
        elif key.startswith("gres/shard:") and not seen_generic_shard:
            usage.shard += _parse_tres_value(value)
    return usage
