from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

from .constants import SINFO_FIELD_WIDTHS
from .constraints import requested_constraint_features
from .models import JobRecord, ServerState
from .resources import parse_cpu, parse_gpu, parse_mem, parse_usage


def expand_range(value: str) -> List[str]:
    if "-" not in value:
        if not value.isdigit():
            raise ValueError(f"Invalid host range '{value}'")
        return [value]

    try:
        start_str, end_str = value.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ValueError
        width = max(len(start_str), len(end_str))
        return [str(number).zfill(width) for number in range(start, end + 1)]
    except ValueError:
        raise ValueError(f"Invalid host range '{value}'") from None


def _expand_host(value: str) -> List[str]:
    open_bracket = value.find("[")
    if open_bracket < 0:
        if "]" in value:
            raise ValueError(f"Invalid hostlist '{value}'")
        return [value]

    close_bracket = value.find("]", open_bracket + 1)
    if close_bracket < 0 or "[" in value[open_bracket + 1 : close_bracket]:
        raise ValueError(f"Invalid hostlist '{value}'")

    prefix = value[:open_bracket]
    ranges = value[open_bracket + 1 : close_bracket]
    suffix = value[close_bracket + 1 :]
    if not ranges:
        raise ValueError(f"Invalid hostlist '{value}'")

    expanded_suffixes = _expand_host(suffix)
    return [
        f"{prefix}{number}{expanded_suffix}"
        for item in ranges.split(",")
        for number in expand_range(item)
        for expanded_suffix in expanded_suffixes
    ]


def parse_nodelist(nodelist: str) -> List[str]:
    nodelist = nodelist.strip()
    if nodelist.upper() in {"", "N/A", "(NULL)", "NONE", "NONE ASSIGNED"}:
        return []

    nodes: List[str] = []
    bracket_depth = 0
    start = 0

    for index, char in enumerate(nodelist):
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "," and bracket_depth == 0:
            nodes.append(nodelist[start:index])
            start = index + 1
    nodes.append(nodelist[start:])

    if bracket_depth != 0 or any(not node for node in nodes):
        raise ValueError(f"Invalid hostlist '{nodelist}'")

    return [host for node in nodes for host in _expand_host(node)]


def parse_features_field(features: str) -> set[str]:
    if not features:
        return set()

    parsed = []
    for raw in features.replace("|", ",").split(","):
        cleaned = raw.strip().lower()
        if not cleaned or cleaned == "(null)":
            continue
        base_feature = cleaned.split("*", 1)[0].strip()
        if base_feature:
            parsed.append(base_feature)
    return set(parsed)


def _split_job_line(line: str) -> Optional[List[str]]:
    if not line:
        return None
    if "|" in line:
        parts = [segment.strip() for segment in line.split("|")]
    else:
        parts = line.split()
    return parts if len(parts) >= 6 else None


def _canonical_job_state(state: str) -> str:
    words = state.split(maxsplit=1)
    return words[0].removesuffix("+") if words else ""


def _valid_job_fields(
    user: str,
    job_id: str,
    state: str,
    usage: str,
) -> bool:
    state_code = state.replace("_", "")
    return bool(
        user
        and job_id
        and job_id[0].isdigit()
        and state_code.isalpha()
        and state_code.isupper()
        and (not usage or "=" in usage)
    )


def parse_jobs(output: str) -> List[JobRecord]:
    jobs: List[JobRecord] = []
    if not output.strip():
        return jobs

    for line_number, line in enumerate(output.strip().splitlines(), start=1):
        parts = _split_job_line(line.strip())
        if not parts:
            raise ValueError(f"Malformed sacct record on line {line_number}")
        if len(parts) >= 10 and parts[-1] == "" and (not parts[-3] or "=" in parts[-3]):
            parts.pop()
        has_constraints_column = len(parts) >= 9 and (not parts[-2] or "=" in parts[-2])
        if has_constraints_column:
            user, job_id, job_name, state, partition, nodelist = parts[:6]
            constraints_str = "|".join(parts[6:-2])
            usage_str, time_limit = parts[-2:]
        elif len(parts) >= 8:
            (
                user,
                job_id,
                job_name,
                state,
                partition,
                nodelist,
                usage_str,
                time_limit,
            ) = parts[:8]
            constraints_str = ""
        else:
            user, partition, nodelist, state, usage_str, job_id = parts[:6]
            job_name = ""
            time_limit = ""
            constraints_str = ""
        state = _canonical_job_state(state)
        if not _valid_job_fields(user, job_id, state, usage_str):
            raise ValueError(f"Malformed sacct record on line {line_number}")
        try:
            usage = parse_usage(usage_str)
        except ValueError as error:
            raise ValueError(
                f"Malformed sacct record on line {line_number}: {error}"
            ) from error
        jobs.append(
            JobRecord(
                user=user,
                job_id=job_id,
                job_name=job_name,
                state=state,
                partition=partition,
                nodelist=nodelist,
                usage=usage,
                time_limit=time_limit,
                constraints=requested_constraint_features(constraints_str),
            )
        )
    return jobs


def _split_sinfo_line(line: str) -> Optional[List[str]]:
    if not line:
        return None
    if "|" in line:
        parts = [segment.strip() for segment in line.split("|")]
        return parts if len(parts) == 7 else None

    fields: List[str] = []
    start = 0
    for width in SINFO_FIELD_WIDTHS:
        end = start + width
        fields.append(line[start:end].strip())
        start = end
    return fields if not line[start:].strip() else None


def parse_sinfo(output: str) -> Dict[str, ServerState]:
    servers: Dict[str, ServerState] = {}
    for line_number, raw_line in enumerate(output.strip().splitlines(), start=1):
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        parts = _split_sinfo_line(line)
        if not parts:
            raise ValueError(f"Malformed sinfo record on line {line_number}")

        node_name, features_raw, gres, gres_used, cpu_state, alloc_mem, total_mem = (
            parts[:7]
        )
        if not node_name or not cpu_state or not total_mem:
            raise ValueError(f"Malformed sinfo record on line {line_number}")

        try:
            gpu = parse_gpu(gres, gres_used)
            cpu = parse_cpu(cpu_state)
            mem = parse_mem(alloc_mem, total_mem)
            server_names = parse_nodelist(node_name)
        except ValueError as error:
            raise ValueError(
                f"Malformed sinfo record on line {line_number}: {error}"
            ) from error
        if not server_names:
            raise ValueError(f"Malformed sinfo record on line {line_number}")
        for server_name in server_names:
            servers[server_name] = ServerState(
                name=server_name,
                features=parse_features_field(features_raw),
                gpu=replace(gpu),
                cpu=replace(cpu),
                mem=replace(mem),
            )
    return servers
