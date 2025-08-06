import argparse
from subprocess import Popen, PIPE
from typing import Dict, List, Set, Optional, Any
from rich.console import Console
from rich.table import Table
from rich import box

GTOP = "sacct -X --format=User%10,partition%30,NodeList%30,State,AllocTRES%80,Jobid -a --units=G | grep RUNNING | grep billing"
SINFO = 'sinfo -O nodehost:100,gres:100,cpusstate,allocmem,memory -h -e'

RESOURCES = ["cpu", "gpu", "mem"]
PARTITIONS = ["priority", "default"]

console = Console()


def exec_cmd(cmd: str) -> str:
    return Popen(cmd, shell=True, stdout=PIPE).stdout.read().decode("utf-8")


def is_priority(name: str) -> bool:
    return not (("default" in name) or ("gpu" in name))


def expand_range(s: str) -> List[str]:
    if "-" in s:
        start, end = map(int, s.split("-"))
        return [f"{x:02d}" for x in range(start, end + 1)]
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
            base = node[:node.find("[")]
            ranges = node[node.find("[") + 1:node.find("]")]
            for r in ranges.split(","):
                for suffix in expand_range(r):
                    result.append(base + suffix)
        else:
            result.append(node)
    return result


def parse_gpu(gres: str) -> Dict[str, Any]:
    if "null" in gres:
        return {"type": "null", "num": 0}
    
    types, total = [], 0
    for item in gres.split(","):
        parts = item.split(":")
        if len(parts) >= 3 and parts[0].startswith("gpu"):
            types.append(parts[1])
            num_str = parts[2]
            total += int(''.join(c for c in num_str if c.isdigit()) or 0)
    
    return {
        "type": f'({"|".join(types)})' if len(types) > 1 else (types[0] if types else "null"),
        "num": total
    }


def parse_cpu(cpu_state: str) -> Dict[str, int]:
    parts = cpu_state.split("/")
    return {"idle": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0}


def parse_mem(alloc_mem: str, total_mem: str) -> Dict[str, int]:
    alloc = int(alloc_mem) if alloc_mem.isdigit() else 0
    total = int(total_mem) if total_mem.isdigit() else 0
    return {"idle": total - alloc, "total": total}


def parse_usage(alloc_tres: str) -> Dict[str, float]:
    usage = {r: 0.0 for r in RESOURCES}
    for res in RESOURCES:
        if res in alloc_tres:
            start = alloc_tres.find(f"{res}=")
            if start != -1:
                val_str = alloc_tres[start + len(res) + 1:]
                end = val_str.find(",")
                val_str = val_str[:end] if end != -1 else val_str
                usage[res] = float(val_str.rstrip("G"))
    return usage


def parse_sinfo(output: str, gpu_only: bool) -> Dict[str, Dict]:
    servers = {}
    for line in output.strip().split("\n"):
        parts = line.split()
        if not parts:
            continue
            
        gpu = parse_gpu(parts[1])
        if gpu_only and gpu["type"] == "null":
            continue
            
        cpu = parse_cpu(parts[2])
        mem = parse_mem(parts[3], parts[4])
        
        for server in parse_nodelist(parts[0]):
            servers[server] = {
                "gpu": gpu,
                "cpu": cpu,
                "mem": mem,
                "usage": {r: {"priority": 0, "default": 0} for r in RESOURCES},
                "users": {}
            }
    return servers


def process_jobs(gtop_output: str, servers: Dict[str, Dict]) -> None:
    for line in gtop_output.strip().split("\n"):
        parts = [p for p in line.split(" ") if p]
        if len(parts) < 6:
            continue
            
        user, partition, nodelist, _, alloc_tres, job_id = parts[:6]
        usage = parse_usage(alloc_tres)
        nodes = parse_nodelist(nodelist)
        num_nodes = len(nodes)
        partition_type = "priority" if is_priority(partition) else "default"
        
        for node in nodes:
            if node in servers:
                servers[node]["users"][job_id] = {
                    'netid': user,
                    'partition': partition,
                    **{r: usage[r] / num_nodes for r in RESOURCES}
                }
                for r in RESOURCES:
                    servers[node]["usage"][r][partition_type] += usage[r] / num_nodes


def format_resource(info: Dict, res: str) -> str:
    priority = info["usage"][res].get('priority', 0)
    default = info["usage"][res].get('default', 0)
    
    if res == "cpu":
        idle = info[res]["idle"]
        return f"{int(priority):2d}/{int(default):2d}/{int(idle):2d}"
    elif res == "gpu":
        idle = info[res]["num"] - priority - default
        return f"{int(priority)}/{int(default)}/{int(idle)}"
    else:  # mem
        idle = info[res]["idle"] / 1024.0
        return f"{priority:5.1f}/{default:5.1f}/{idle:5.1f}"


def create_main_table(servers: Dict[str, Dict], target_users: Optional[Set[str]], 
                      disp_users: bool) -> Table:
    table = Table(box=box.ROUNDED)
    table.add_column("Server", style="bright_cyan", no_wrap=True)
    table.add_column("GPU", style="yellow")
    table.add_column("CPU (P/D/I)", style="green", justify="center")
    table.add_column("GPU (P/D/I)", style="magenta", justify="center")
    table.add_column("Memory GB (P/D/I)", style="blue", justify="center")
    
    for server in sorted(servers.keys()):
        info = servers[server]
        
        if target_users:
            has_target_user = any(u['netid'] in target_users for u in info['users'].values())
            if not has_target_user:
                continue
        
        gpu_info = f"{info['gpu']['num']} x {info['gpu']['type']}"
        row_data = [
            server,
            gpu_info,
            format_resource(info, "cpu"),
            format_resource(info, "gpu"),
            format_resource(info, "mem")
        ]
        
        table.add_row(*row_data)
        
        if disp_users and info['users']:
            jobs_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
            jobs_table.add_column("Job ID", style="dim")
            jobs_table.add_column("User", style="bright_white")
            jobs_table.add_column("Partition")
            jobs_table.add_column("CPU", justify="right")
            jobs_table.add_column("GPU", justify="right")
            jobs_table.add_column("Mem(GB)", justify="right")
            
            for job_id, job in info['users'].items():
                style = None
                if target_users and job['netid'] in target_users:
                    style = "bright_cyan bold"
                elif not is_priority(job['partition']):
                    style = "green"
                else:
                    style = "red"
                
                jobs_table.add_row(
                    job_id,
                    job['netid'],
                    job['partition'],
                    str(int(job.get('cpu', 0))),
                    str(int(job.get('gpu', 0))),
                    f"{job.get('mem', 0):.1f}",
                    style=style
                )
            
            table.add_row("", jobs_table, "", "", "")
    
    return table


def print_summary(servers: Dict[str, Dict], jobs: List[Dict], target_users: Optional[Set[str]]) -> None:
    total_gpus = sum(s['gpu']['num'] for s in servers.values())
    used_gpus = sum(
        s['usage']['gpu'].get('priority', 0) + s['usage']['gpu'].get('default', 0)
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
        user_stats = {u: {"nodes": set(), "gpus_by_partition": {}} for u in target_users}
        
        for job in jobs:
            user = job['user']
            if user in target_users:
                usage = parse_usage(job['usage_str'])
                gpus = int(usage.get('gpu', 0))
                partition = job['partition']
                
                for node in parse_nodelist(job['nodelist']):
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
                console.print(f"[cyan]• {user:<15}[/] using [bold]{node_count}[/] node(s), [bold]{total_gpus}[/] GPU(s):")
                for partition, count in sorted(stats["gpus_by_partition"].items()):
                    console.print(f"  - {partition:<20}: [bold]{count:>3}[/] GPU(s)")
                    total_by_partition[partition] = total_by_partition.get(partition, 0) + count
        
        console.print("-" * 45, style="dim")
        total_nodes = len(set().union(*[s["nodes"] for s in user_stats.values()]))
        total_gpus = sum(total_by_partition.values())
        console.print(f"[bold]Total:[/] {total_nodes} nodes, {total_gpus} GPUs")
        for partition, count in sorted(total_by_partition.items()):
            console.print(f"  {partition:<20}: [bold]{count:>3}[/] GPU(s)")
        console.print("=" * 80, style="bright_white")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display SLURM cluster usage with detailed node and resource information"
    )
    parser.add_argument('--gpu-only', action="store_true", help="Only show nodes with GPUs")
    parser.add_argument('--disp-users', action="store_true", help="Display detailed per-job usage")
    parser.add_argument('--users', nargs='+', help='Filter by user netids')
    
    args = parser.parse_args()
    target_users = set(args.users) if args.users else None
    
    servers = parse_sinfo(exec_cmd(SINFO), args.gpu_only)
    
    gtop_output = exec_cmd(GTOP)
    jobs = []
    for line in gtop_output.strip().split("\n"):
        parts = [p for p in line.split(" ") if p]
        if len(parts) >= 6:
            jobs.append({
                'user': parts[0],
                'partition': parts[1],
                'nodelist': parts[2],
                'usage_str': parts[4],
                'job_id': parts[5]
            })
    
    process_jobs(gtop_output, servers)
    print_summary(servers, jobs, target_users)
    
    table = create_main_table(servers, target_users, args.disp_users)
    console.print(table)


if __name__ == "__main__":
    main()