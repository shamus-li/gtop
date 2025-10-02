import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from subprocess import PIPE, Popen
from typing import Any, Dict, List, Optional, Set, Tuple

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

GTOP_COMMAND = "sacct -X --format=User%10,partition%40,NodeList%40,State,AllocTRES%120,Jobid -a --units=G | grep RUNNING"
SINFO_COMMAND = (
    "sinfo -O "
    "nodehost:100,features:200,gres:256,gresused:256,cpusstate:100,allocmem:100,memory:100 "
    "-h"
)

SINFO_FIELD_WIDTHS = [100, 200, 256, 256, 100, 100, 100]

RESOURCES = ["cpu", "gpu", "mem"]
PARTITIONS = ["priority", "default"]

console = Console()
stderr_console = Console(stderr=True)


def exec_cmd(cmd: str) -> str:
    """Execute shell command with improved error handling and timeout."""
    try:
        process = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate(timeout=120)  # 120 second timeout
        if process.returncode != 0:
            stderr_console.print(f"[red]Command failed: {cmd}[/]")
            stderr_console.print(f"[red]Error: {stderr.decode('utf-8')}[/]")
            return ""
        return stdout.decode("utf-8")
    except Exception as e:
        stderr_console.print(f"[red]Error executing command '{cmd}': {e}[/]")
        return ""


def exec_commands_parallel(commands: List[Tuple[str, str]]) -> Dict[str, str]:
    """Execute multiple commands in parallel for improved performance."""
    results = {}

    with ThreadPoolExecutor(max_workers=min(len(commands), 4)) as executor:
        # Submit all commands
        future_to_name = {
            executor.submit(exec_cmd, cmd): name for name, cmd in commands
        }

        # Collect results as they complete
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                stderr_console.print(f"[red]Error executing {name}: {e}[/]")
                results[name] = ""

    return results


def is_priority(name: str) -> bool:
    return not (("default" in name) or ("gpu" in name))


def expand_range(s: str) -> List[str]:
    """Expand numeric ranges in node names with improved efficiency."""
    if "-" not in s:
        return [s]

    try:
        start, end = map(int, s.split("-", 1))
        return [f"{x:02d}" for x in range(start, end + 1)]
    except ValueError:
        # If not numeric, return as is
        return [s]


def parse_nodelist(nodelist: str) -> List[str]:
    nodes = []
    bracket_depth = 0
    start = 0

    for i, char in enumerate(nodelist):
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "," and bracket_depth == 0:
            nodes.append(nodelist[start:i])
            start = i + 1
    nodes.append(nodelist[start:])

    result = []
    for node in nodes:
        if "[" in node and "]" in node:
            base = node[: node.find("[")]
            ranges = node[node.find("[") + 1 : node.find("]")]
            for r in ranges.split(","):
                for suffix in expand_range(r):
                    result.append(base + suffix)
        else:
            result.append(node)
    return result


def parse_features_field(features: str) -> Set[str]:
    """Turn the sinfo features column into a normalized set."""
    if not features:
        return set()

    feature_list = []
    for raw in features.replace("|", ",").split(","):
        cleaned = raw.strip().lower()
        if not cleaned or cleaned == "(null)":
            continue

        base_feature = cleaned.split("*", 1)[0].strip()
        if base_feature:
            feature_list.append(base_feature)
    return set(feature_list)


def _tokenize_constraint(expr: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    length = len(expr)

    while i < length:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()&|":
            tokens.append(ch)
            i += 1
            continue
        if ch == "[":
            tokens.append("(")
            i += 1
            continue
        if ch == "]":
            tokens.append(")")
            i += 1
            continue

        start = i
        while i < length and expr[i] not in "()&|[] " and not expr[i].isspace():
            i += 1
        tokens.append(expr[start:i])

    return tokens


def matches_constraint(features: Set[str], constraint: str) -> bool:
    """Evaluate a SLURM-style constraint expression against node features.

    This implements left-to-right evaluation for `&` and `|`, matching SLURM's
    documented behaviour when parentheses are not provided. Square brackets are
    treated as parentheses for Filtering purposes. Multipliers (e.g., foo*2) are
    ignored for single-node checks.
    """

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
        feature_name = token.split("*", 1)[0]
        feature_name = feature_name.strip()
        return feature_name in features

    def parse_expression() -> bool:
        nonlocal position
        value = parse_term()
        while position < len(tokens) and tokens[position] in {"&", "|"}:
            op = tokens[position]
            position += 1
            rhs = parse_term()
            if op == "&":
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
        stderr_console.print(
            f"[red]Failed to evaluate constraint '{constraint}': {error}[/]"
        )
        return False


def parse_gpu(gres: str, gres_used: str = "") -> Dict[str, Any]:
    """Parse GPU information including MIG and sharded GPU support."""
    if not gres or gres.strip().lower() in {"(null)", "null", "none"}:
        return {"type": "null", "num": 0, "mig_instances": [], "shards": 0, "used": 0}

    types, total, mig_instances, shards = [], 0, [], 0
    used_gpus = 0

    # Parse available GRES
    for item in gres.split(","):
        item = item.strip()
        # Handle parentheses carefully to avoid splitting on colons inside them
        # First split on all colons, then rejoin parts that belong to parentheses
        temp_parts = item.split(":")
        if len(temp_parts) > 3:
            # Rejoin parts 3 and onwards (likely parentheses content)
            parts = temp_parts[:2] + [":".join(temp_parts[2:])]
        else:
            parts = temp_parts

        if len(parts) >= 2 and parts[0] == "gpu":
            if len(parts) >= 3:
                gpu_type = parts[1]
                count_part = parts[2]

                # Extract count from complex formats like "4(S:0-1)" or "1g.5gb:2"
                if "(" in count_part:
                    # Handle sharded format: "4(S:0-1)" - count is the actual GPU hardware count
                    count_str = count_part.split("(")[0]
                    # All (S:...) patterns are just SLURM organization, not true sharing
                    # Only nodes with explicit shard: entries are actually sharded
                    if "(S:" in count_part:
                        count = int("".join(c for c in count_str if c.isdigit()) or 0)
                        types.append(gpu_type)
                        total += count
                    else:
                        # Other parenthetical format
                        count = int("".join(c for c in count_str if c.isdigit()) or 0)
                        types.append(gpu_type)
                        total += count
                elif len(parts) >= 4:
                    # Handle MIG format: "gpu:a100:1g.5gb:2"
                    count_str = parts[3]
                    count = int("".join(c for c in count_str if c.isdigit()) or 0)
                    if "g." in gpu_type:
                        mig_instances.append({"type": gpu_type, "count": count})
                    else:
                        types.append(gpu_type)
                    total += count
                else:
                    count_str = count_part
                    count = int("".join(c for c in count_str if c.isdigit()) or 0)
                    types.append(gpu_type)
                    total += count
            elif len(parts) == 2:
                # Handle basic gpu entries
                count_str = parts[1]
                total += int("".join(c for c in count_str if c.isdigit()) or 0)

        # Handle shard entries (e.g., shard:nvidia_h100_nvl:48(S:12-23))
        elif parts[0] == "shard" and len(parts) >= 3:
            gpu_type = parts[1] if len(parts) >= 2 else "gpu"
            count_part = parts[2] if len(parts) >= 3 else "0"

            # Extract shard count from formats like "48(S:12-23)"
            if "(" in count_part:
                shard_count_str = count_part.split("(")[0]
            else:
                shard_count_str = count_part

            shard_count = int("".join(c for c in shard_count_str if c.isdigit()) or 0)

            # Store the GPU type being sharded (but don't add to types if already present)
            if gpu_type != "gpu" and f"{gpu_type}_shard" not in types:
                types.append(f"{gpu_type}_shard")

            # Only add to shards, don't add to total GPU count
            shards = (
                shard_count  # Use assignment, not addition to avoid double counting
            )

    # Parse used GRES if provided
    if gres_used and gres_used.strip().lower() not in {"(null)", "null", "none"}:
        for item in gres_used.split(","):
            parts = item.split(":")
            if len(parts) >= 2 and parts[0].startswith("gpu"):
                if len(parts) >= 3:
                    count_str = parts[2]
                    used_gpus += int("".join(c for c in count_str if c.isdigit()) or 0)
                elif len(parts) == 2:
                    count_str = parts[1]
                    used_gpus += int("".join(c for c in count_str if c.isdigit()) or 0)

    # Determine display type
    if mig_instances:
        mig_types = [inst["type"] for inst in mig_instances]
        display_type = f"MIG({'|'.join(set(mig_types))})"
    elif shards > 0 or any(t.endswith("_shard") for t in types):
        # Check if we have shard-specific GPU types
        shard_types = [t for t in types if t.endswith("_shard")]
        if shard_types:
            # Extract the GPU type from shard type (remove '_shard' suffix)
            gpu_types = [t.replace("_shard", "") for t in shard_types]
            display_type = f"Shard({'|'.join(set(gpu_types))})"
        else:
            display_type = f"Shard({shards})"
    elif len(types) > 1:
        display_type = f"({'|'.join(types)})"
    elif types:
        display_type = types[0]
    else:
        display_type = "gpu"

    return {
        "type": display_type,
        "num": total,
        "mig_instances": mig_instances,
        "shards": shards,
        "used": used_gpus,
    }


def parse_cpu(cpu_state: str) -> Dict[str, int]:
    parts = cpu_state.split("/")
    return {"idle": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0}


def _parse_numeric(value: str) -> float:
    if not value:
        return 0.0

    lowered = value.strip().lower()
    if lowered in {"(null)", "none"}:
        return 0.0

    cleaned = "".join(c for c in value if c.isdigit() or c == ".")
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def parse_mem(alloc_mem: str, total_mem: str) -> Dict[str, float]:
    alloc = _parse_numeric(alloc_mem)
    total = _parse_numeric(total_mem)
    idle = max(total - alloc, 0.0)
    return {"idle": idle, "total": total}


def parse_usage(alloc_tres: str) -> Dict[str, float]:
    """Parse TRES allocation string with improved efficiency."""
    usage = {r: 0.0 for r in RESOURCES}

    if not alloc_tres:
        return usage

    # Split once and process all resources in single pass
    parts = alloc_tres.split(",")
    for part in parts:
        if "=" not in part:
            continue

        key, val = part.split("=", 1)

        # Handle SLURM TRES format mapping
        if key == "cpu":
            usage["cpu"] = float(val.rstrip("G"))
        elif key == "mem":
            usage["mem"] = float(val.rstrip("G"))
        elif key == "gres/gpu":
            # Only count the generic gres/gpu=N to avoid double counting
            # (SLURM outputs both gres/gpu:type=N and gres/gpu=N for the same GPU)
            try:
                usage["gpu"] += float(val.rstrip("G"))
            except ValueError:
                continue

    return usage


def _split_sinfo_line(line: str) -> Optional[List[str]]:
    if not line:
        return None

    if "|" in line:
        parts = [segment.strip() for segment in line.split("|")]
        return parts if len(parts) >= 7 else None

    fields: List[str] = []
    start = 0
    for width in SINFO_FIELD_WIDTHS:
        end = start + width
        fields.append(line[start:end].strip())
        start = end

    return fields if len(fields) >= 7 else None


def parse_sinfo(output: str, gpu_only: bool) -> Dict[str, Dict]:
    """Parse sinfo output with enhanced GPU support including MIG and sharded GPUs."""
    servers = {}
    for raw_line in output.strip().split("\n"):
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        parts = _split_sinfo_line(line)
        if not parts:
            continue

        node_name, features_raw, gres, gres_used, cpu_state, alloc_mem, total_mem = (
            parts[:7]
        )

        features = parse_features_field(features_raw)

        gpu = parse_gpu(gres, gres_used)
        if gpu_only and gpu["type"] == "null":
            continue

        cpu = parse_cpu(cpu_state)
        mem = parse_mem(alloc_mem, total_mem)

        for server in parse_nodelist(node_name):
            servers[server] = {
                "features": features,
                "gpu": gpu,
                "cpu": cpu,
                "mem": mem,
                "usage": {r: {"priority": 0, "default": 0} for r in RESOURCES},
                "users": {},
            }
    return servers


def process_jobs(gtop_output: str, servers: Dict[str, Dict]) -> None:
    """Process job information and update server usage data."""
    if not gtop_output.strip():
        return

    processed_jobs = 0
    for line_num, line in enumerate(gtop_output.strip().split("\n"), 1):
        line = line.strip()
        if not line:
            continue

        # More flexible parsing - handle whitespace better
        parts = line.split()
        if len(parts) < 6:
            stderr_console.print(
                f"[yellow]Warning: Line {line_num} has insufficient columns: {len(parts)} < 6[/]"
            )
            continue

        try:
            user, partition, nodelist, state, alloc_tres, job_id = parts[:6]

            # Skip if not actually running
            if state != "RUNNING":
                continue

            # Parse usage and nodes
            usage = parse_usage(alloc_tres)
            nodes = parse_nodelist(nodelist)

            if not nodes:
                stderr_console.print(
                    f"[yellow]Warning: No nodes found for job {job_id}[/]"
                )
                continue

            partition_type = "priority" if is_priority(partition) else "default"

            # Debug: Print job processing info (only if debug enabled)
            if hasattr(process_jobs, "debug_enabled") and process_jobs.debug_enabled:
                stderr_console.print(
                    f"[dim]Processing job {job_id}: {user}@{partition}, nodes: {len(nodes)}, usage: {usage}[/]"
                )

            # Update server data - only process nodes that exist in our server list
            matched_nodes = [node for node in nodes if node in servers]
            if matched_nodes:
                for node in matched_nodes:
                    servers[node]["users"][job_id] = {
                        "netid": user,
                        "partition": partition,
                        **{r: usage[r] / len(matched_nodes) for r in RESOURCES},
                    }
                    for r in RESOURCES:
                        servers[node]["usage"][r][partition_type] += usage[r] / len(
                            matched_nodes
                        )
                processed_jobs += 1
            elif hasattr(process_jobs, "debug_enabled") and process_jobs.debug_enabled:
                # Only show debug warning if debug is enabled
                stderr_console.print(
                    f"[dim]Skipping job {job_id}: nodes {nodes} not in server list (likely CPU-only)[/]"
                )

        except (ValueError, IndexError) as e:
            stderr_console.print(f"[red]Error processing job line {line_num}: {e}[/]")
            stderr_console.print(f"[red]Line content: {line}[/]")
            continue

    if hasattr(process_jobs, "debug_enabled") and process_jobs.debug_enabled:
        stderr_console.print(f"[dim]Processed {processed_jobs} job allocations[/]")


def format_resource(info: Dict, res: str, disp_shard: bool = False) -> str:
    """Format resource usage with improved handling for MIG and sharded GPUs."""
    priority = info["usage"][res].get("priority", 0)
    default = info["usage"][res].get("default", 0)

    if res == "cpu":
        idle = info[res]["idle"]
        return f"{int(priority):2d}/{int(default):2d}/{int(idle):2d}"
    elif res == "gpu":
        gpu_info = info[res]
        total = gpu_info["num"]

        # For MIG instances, show more detailed breakdown
        if gpu_info.get("mig_instances"):
            used = gpu_info.get("used", 0)
            idle = total - used
            return f"{used}/{idle} (MIG)"
        # For sharded GPUs, show shard allocation only if --disp-shard is enabled
        elif disp_shard and gpu_info.get("shards", 0) > 0:
            used = priority + default
            idle = gpu_info["shards"] - used
            return f"{int(priority)}/{int(default)}/{int(idle)}"
        else:
            idle = total - priority - default
            return f"{int(priority)}/{int(default)}/{int(idle)}"
    else:  # mem
        idle = info[res]["idle"] / 1024.0
        return f"{int(priority):3d}/{int(default):3d}/{int(idle):3d}"


def sort_server_names(servers: Dict[str, Dict], sort_by: str) -> List[str]:
    """Return server names sorted according to the requested strategy."""

    if sort_by == "feature":

        def sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[str, str]:
            name, info = item
            features = info.get("features") or set()
            feature_repr = ",".join(sorted(features)) if features else ""
            return (feature_repr, name)

        return [name for name, _ in sorted(servers.items(), key=sort_key)]

    return sorted(servers.keys())


def create_main_table(
    servers: Dict[str, Dict],
    target_users: Optional[Set[str]],
    disp_users: bool,
    sort_by: str,
    disp_shard: bool = False,
) -> Table:
    table = Table(box=box.ROUNDED)
    table.add_column("Server", style="bright_cyan", no_wrap=True)
    table.add_column("GPU", style="yellow", overflow="fold")
    table.add_column("CPU (P/D/I)", style="green", justify="center")
    table.add_column("GPU (P/D/I)", style="magenta", justify="center")
    table.add_column("Memory GB (P/D/I)", style="blue", justify="center")

    for server in sort_server_names(servers, sort_by):
        info = servers[server]

        if target_users:
            has_target_user = any(
                u["netid"] in target_users for u in info["users"].values()
            )
            if not has_target_user:
                continue

        # Enhanced GPU info display for MIG and sharded GPUs
        gpu_data = info["gpu"]
        if gpu_data.get("mig_instances"):
            mig_details = []
            for inst in gpu_data["mig_instances"]:
                mig_details.append(f"{inst['count']}x{inst['type']}")
            gpu_info = f"MIG: {', '.join(mig_details)}"
        elif disp_shard and gpu_data.get("shards", 0) > 0:
            # Show the underlying GPU type being sharded only if --disp-shard is enabled
            gpu_type = gpu_data["type"]
            if (
                gpu_type.startswith("Shard(")
                and gpu_type != f"Shard({gpu_data['shards']})"
            ):
                # Extract GPU type from Shard(gpu_type) format
                base_type = gpu_type[6:-1]  # Remove 'Shard(' and ')'
                gpu_info = f"{gpu_data['shards']} x {base_type} (Sharded)"
            else:
                gpu_info = f"{gpu_data['shards']} shards ({gpu_data['num']} total)"
        else:
            # Default: show GPU count regardless of sharding
            gpu_type = gpu_data["type"]
            # Extract base GPU type if it's in Shard() format
            if gpu_type.startswith("Shard(") and gpu_type.endswith(")"):
                base_type = gpu_type[6:-1]  # Remove 'Shard(' and ')'
                gpu_info = f"{gpu_data['num']} x {base_type}"
            else:
                gpu_info = f"{gpu_data['num']} x {gpu_data['type']}"

        # Calculate idle GPUs to determine row color
        priority = info["usage"]["gpu"].get("priority", 0)
        default = info["usage"]["gpu"].get("default", 0)
        total_gpus = (
            gpu_data.get("shards", 0)
            if disp_shard and gpu_data.get("shards", 0) > 0
            else gpu_data["num"]
        )
        idle_gpus = total_gpus - priority - default

        # Determine row style based on idle GPU count
        if idle_gpus >= 1:
            row_style = "green"
        elif idle_gpus <= 0:
            row_style = "red"
        else:
            row_style = None

        row_data = [
            server,
            Text(gpu_info, overflow="fold"),
            format_resource(info, "cpu", disp_shard),
            format_resource(info, "gpu", disp_shard),
            format_resource(info, "mem", disp_shard),
        ]

        table.add_row(*row_data, style=row_style)

        if disp_users and info["users"]:
            jobs_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
            jobs_table.add_column("Job ID", style="dim")
            jobs_table.add_column("User", style="bright_white")
            jobs_table.add_column("Partition")
            jobs_table.add_column("CPU", justify="right")
            jobs_table.add_column("GPU", justify="right")
            jobs_table.add_column("Mem(GB)", justify="right")

            for job_id, job in info["users"].items():
                style = None
                if target_users and job["netid"] in target_users:
                    style = "bright_cyan bold"
                elif not is_priority(job["partition"]):
                    style = "green"
                else:
                    style = "red"

                jobs_table.add_row(
                    job_id,
                    job["netid"],
                    job["partition"],
                    str(int(job.get("cpu", 0))),
                    str(int(job.get("gpu", 0))),
                    f"{job.get('mem', 0):.1f}",
                    style=style,
                )

            table.add_row("", jobs_table, "", "", "")

    return table


def print_summary(
    servers: Dict[str, Dict], jobs: List[Dict], target_users: Optional[Set[str]]
) -> None:
    total_gpus = sum(s["gpu"]["num"] for s in servers.values())
    used_gpus = sum(
        s["usage"]["gpu"].get("priority", 0) + s["usage"]["gpu"].get("default", 0)
        for s in servers.values()
    )

    if total_gpus > 0:
        pct = (used_gpus / total_gpus) * 100
        console.print(
            f"[bold yellow]Cluster GPU Overview: {int(used_gpus)}/{total_gpus} GPUs Used ({pct:.1f}%)[/]"
        )
    else:
        console.print("[bold yellow]Cluster GPU Overview: No GPUs detected[/]")

    if target_users:
        user_stats = {
            u: {"nodes": set(), "gpus_by_partition": {}} for u in target_users
        }

        for job in jobs:
            user = job["user"]
            if user in target_users:
                job_nodes = parse_nodelist(job["nodelist"])
                matched_nodes = [node for node in job_nodes if node in servers]
                if not matched_nodes:
                    continue

                usage = parse_usage(job["usage_str"])
                gpus = int(usage.get("gpu", 0))
                partition = job["partition"]

                for node in matched_nodes:
                    user_stats[user]["nodes"].add(node)

                current = user_stats[user]["gpus_by_partition"].get(partition, 0)
                user_stats[user]["gpus_by_partition"][partition] = current + gpus

        console.print("\n" + "=" * 80, style="bright_white")
        console.print("[bold]Summary of Resources Used by Specified Users[/]")
        console.print("=" * 80, style="bright_white")

        total_by_partition = {}
        for user in sorted(target_users):
            stats = user_stats[user]
            node_count = len(stats["nodes"])
            if node_count > 0:
                total_gpus = sum(stats["gpus_by_partition"].values())
                console.print(
                    f"[cyan]• {user:<15}[/] using [bold]{node_count}[/] node(s), [bold]{total_gpus}[/] GPU(s):"
                )
                for partition, count in sorted(stats["gpus_by_partition"].items()):
                    console.print(f"  - {partition:<20}: [bold]{count:>3}[/] GPU(s)")
                    total_by_partition[partition] = (
                        total_by_partition.get(partition, 0) + count
                    )

        console.print("-" * 45, style="dim")
        total_nodes = len(set().union(*[s["nodes"] for s in user_stats.values()]))
        total_gpus = sum(total_by_partition.values())
        console.print(f"[bold]Total:[/] {total_nodes} nodes, {total_gpus} GPUs")
        for partition, count in sorted(total_by_partition.items()):
            console.print(f"  {partition:<20}: [bold]{count:>3}[/] GPU(s)")
        console.print("=" * 80, style="bright_white")


def print_mig_summary(servers: Dict[str, Dict]) -> None:
    """Print detailed MIG instance summary."""
    mig_servers = {k: v for k, v in servers.items() if v["gpu"].get("mig_instances")}

    if not mig_servers:
        console.print("[yellow]No MIG-enabled servers found in the cluster.[/]")
        return

    console.print("\n" + "=" * 80, style="bright_cyan")
    console.print("[bold cyan]MIG Instance Details[/]")
    console.print("=" * 80, style="bright_cyan")

    mig_table = Table(box=box.ROUNDED)
    mig_table.add_column("Server", style="bright_cyan", no_wrap=True)
    mig_table.add_column("MIG Instance", style="yellow")
    mig_table.add_column("Count", style="green", justify="center")
    mig_table.add_column("Used/Total", style="magenta", justify="center")

    for server, info in sorted(mig_servers.items()):
        for instance in info["gpu"]["mig_instances"]:
            used = info["gpu"].get("used", 0)
            total = instance["count"]
            mig_table.add_row(
                server, instance["type"], str(instance["count"]), f"{used}/{total}"
            )

    console.print(mig_table)
    console.print("=" * 80, style="bright_cyan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display SLURM cluster usage with detailed node and resource information"
    )
    parser.add_argument("--gpu", action="store_true", help="Only show nodes with GPUs")
    parser.add_argument(
        "--disp-users", action="store_true", help="Display detailed per-job usage"
    )
    parser.add_argument("--users", nargs="+", help="Filter by user netids")
    parser.add_argument(
        "--constraint",
        help="Filter nodes by features matching the constraint expression (e.g., gpu-high)",
    )
    parser.add_argument(
        "--mig-info",
        action="store_true",
        help="Display detailed MIG instance information",
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel command execution"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output for troubleshooting"
    )
    parser.add_argument(
        "--sort",
        choices=["name", "feature"],
        default="feature",
        help="Sort the node list by the specified attribute",
    )
    parser.add_argument(
        "--shard",
        action="store_true",
        help="Display sharded GPU information instead of total GPU count",
    )

    args = parser.parse_args()
    target_users = set(args.users) if args.users else None

    # Enable debug mode if requested
    if args.debug:
        process_jobs.debug_enabled = True

    # Execute SLURM commands in parallel for better performance
    if not args.no_parallel:
        commands = [("sinfo", SINFO_COMMAND), ("gtop", GTOP_COMMAND)]

        results = exec_commands_parallel(commands)
        sinfo_output = results.get("sinfo", "")
        gtop_output = results.get("gtop", "")
    else:
        # Sequential execution for debugging or compatibility
        sinfo_output = exec_cmd(SINFO_COMMAND)
        gtop_output = exec_cmd(GTOP_COMMAND)

    if not sinfo_output:
        console.print(
            "[red]Failed to get cluster node information. Check SLURM access.[/]"
        )
        return

    # Debug output
    if args.debug:
        stderr_console.print(
            f"[dim]sinfo returned {len(sinfo_output.splitlines())} lines[/]"
        )
        stderr_console.print(
            f"[dim]gtop returned {len(gtop_output.splitlines()) if gtop_output else 0} lines[/]"
        )

    servers = parse_sinfo(sinfo_output, args.gpu)

    if args.constraint:
        constraint_expr = args.constraint.strip()
        filtered_servers = {
            name: info
            for name, info in servers.items()
            if matches_constraint(info.get("features", set()), constraint_expr)
        }

        if not filtered_servers:
            console.print(
                f"[yellow]No servers found matching constraint '{constraint_expr}'.[/]"
            )
            return

        servers = filtered_servers

    if not servers:
        console.print("[yellow]No servers found matching the criteria.[/]")
        return

    # Parse job information
    jobs = []
    if gtop_output:
        for line in gtop_output.strip().split("\n"):
            parts = [p for p in line.split(" ") if p]
            if len(parts) >= 6:
                jobs.append(
                    {
                        "user": parts[0],
                        "partition": parts[1],
                        "nodelist": parts[2],
                        "usage_str": parts[4],
                        "job_id": parts[5],
                    }
                )

        process_jobs(gtop_output, servers)

    print_summary(servers, jobs, target_users)

    # Display MIG details if requested
    if args.mig_info:
        print_mig_summary(servers)

    table = create_main_table(
        servers, target_users, args.disp_users, args.sort, args.shard
    )
    console.print(table)


if __name__ == "__main__":
    main()
