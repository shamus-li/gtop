import io
from dataclasses import replace
from typing import Optional
from unittest.mock import patch

import pytest
from rich.console import Console

from gtop.accounting import process_jobs
from gtop.models import (
    CpuInfo,
    GpuInfo,
    JobRecord,
    JobUsage,
    MemoryInfo,
    ResourceUsageSplit,
    ServerState,
    UserUsage,
)
from gtop.render import (
    SEMANTIC_PALETTE,
    _build_bar,
    _detail_policy,
    _display_gpu_type,
    _partition_segments,
    help_legend,
    print_filtered_users,
    print_top_users,
)
from gtop.partitions import partition_bucket
from gtop.render_cluster import _resource_numbers, _resource_row, render_table
from gtop.render_jobs import render_jobs_view

WIDTHS = (80, 100, 160)
NODE_WIDTHS = (40, *WIDTHS)
JOB_WIDTHS = (60, *WIDTHS)
GPU_TYPE = "Pro 6000 Blackwell Server Edition Extra Identifier"
NODE_NAME = "research-gpu-node-with-stable-id-001"
USER = "researcher_with_stable_identifier"


def _console(width: int) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    return Console(file=stream, width=width, force_terminal=False), stream


def _assert_bounded(output: str, width: int) -> None:
    assert max((len(line) for line in output.splitlines()), default=0) <= width


def _server() -> ServerState:
    server = ServerState(
        name=NODE_NAME,
        features={"gpu"},
        gpu=GpuInfo(type=GPU_TYPE, num=4, used=1),
        cpu=CpuInfo(idle=32, total=40),
        mem=MemoryInfo(idle=32768, total=65536),
    )
    server.usage["gpu"] = ResourceUsageSplit(partitions={"priority": 1})
    server.usage["cpu"] = ResourceUsageSplit(partitions={"priority": 8})
    server.usage["mem"] = ResourceUsageSplit(partitions={"priority": 32})
    return server


def _job() -> JobRecord:
    return JobRecord(
        user=USER,
        job_id="12345678_901",
        job_name="training_job_with_realistically_long_optional_detail",
        state="RUNNING",
        partition="priority_partition_with_detail",
        nodelist=NODE_NAME,
        usage=JobUsage(cpu=8, gpu=1, mem=32),
        time_limit="7-00:00:00",
    )


def test_canonical_palette_and_legend_are_shared_by_help():
    legend = help_legend()
    bar = _build_bar(
        {"priority": 1, "gpu": 1, "default": 1, "other": 1},
        free=0,
        unavailable=1,
        width=5,
    )

    assert SEMANTIC_PALETTE == {
        "priority": "#c764f4",
        "gpu": "#4fd3a1",
        "default": "#88b4ff",
    }
    assert legend.plain.count("████") == 4
    assert "violet" not in legend.plain
    assert "free" not in legend.plain
    assert not bar.style
    assert [span.style for span in legend.spans] == [
        span.style for span in bar.spans[1:-1]
    ]


def test_filtered_users_heading_is_plural_for_multiple_users():
    console, stream = _console(100)

    print_filtered_users(
        {
            "alice": UserUsage(user="alice", usage_by_partition={"gpu": 1}),
            "bob": UserUsage(user="bob", usage_by_partition={"gpu": 1}),
        },
        unit="GPU",
        console=console,
    )

    assert stream.getvalue().splitlines()[0] == "Users"


@pytest.mark.parametrize(
    ("partition", "bucket"),
    [
        ("default_partition", "default"),
        ("gpu-interactive", "gpu"),
        ("monakhova", "priority"),
        ("kilian-interactive", "priority"),
        ("other", None),
    ],
)
def test_every_actual_partition_uses_a_semantic_bucket(
    partition: str,
    bucket: Optional[str],
):
    assert partition_bucket(partition) == bucket


def test_capacity_bar_keeps_small_positive_segments_visible():
    assert _build_bar({"gpu": 10}, free=150, width=8).plain == "[█·······]"


def test_capacity_bar_has_one_combined_other_segment():
    segments = _partition_segments({"priority": 1, "gpu": 1, "default": 1, "other": 1})

    assert [segment[0] for segment in segments].count("other") == 1


def test_narrow_detail_policy_drops_split_before_bar():
    assert _detail_policy(
        30,
        required_widths=[10],
        bar_widths=[14],
        split_widths=[20],
    ) == (True, False)


def test_mixed_gpu_types_use_readable_names():
    server = _server()
    server.gpu.type = "(nvidia_rtx_a6000|nvidia_a100-pcie-40gb)"

    assert _display_gpu_type(server) == "A6000 + A100 PCIe 40GB"


def test_filtered_bar_only_shows_selected_usage():
    server = _server()
    server.gpu.num = 8
    server.gpu.used = 6
    server.usage["gpu"] = ResourceUsageSplit(partitions={"monakhova": 2})
    values = {
        "gpu": _resource_numbers(
            server,
            "gpu",
            show_used=True,
        )
    }

    cells = _resource_row(
        values,
        resources=("gpu",),
        show_used=True,
        show_total=True,
        include_bar=True,
        include_split=False,
    )

    bar = cells[1]
    assert bar.plain == "[██······]"
    assert all(str(span.style) != "dim #a4b0be" for span in bar.spans)


def test_resource_headers_name_free_and_used_counts():
    console, free_stream = _console(160)
    console.print(render_table([_server()], width=160, verbose=True))
    console, used_stream = _console(160)
    console.print(
        render_table(
            [_server()],
            width=160,
            show_used=True,
            verbose=True,
        )
    )

    assert "GPU free" in free_stream.getvalue()
    assert "CPU free" in free_stream.getvalue()
    assert "Memory free" in free_stream.getvalue()
    assert "GPU used" in used_stream.getvalue()
    assert "CPU used" in used_stream.getvalue()
    assert "Memory used" in used_stream.getvalue()


@pytest.mark.parametrize("width", WIDTHS)
def test_summary_preserves_gpu_identifier_and_counts(width: int):
    console, stream = _console(width)

    console.print(
        render_table(
            [_server()],
            width=width,
        )
    )

    output = stream.getvalue()
    _assert_bounded(output, width)
    assert GPU_TYPE in output
    assert "3/4" in output
    assert "1" in output


@pytest.mark.parametrize("width", NODE_WIDTHS)
def test_nodes_preserve_node_identifier_and_primary_counts(width: int):
    console, stream = _console(width)

    console.print(
        render_table(
            [_server()],
            width=width,
            verbose=True,
        )
    )

    output = stream.getvalue()
    _assert_bounded(output, width)
    assert NODE_NAME in output
    assert "3/4" in output
    assert "32/40" in output
    assert "32/64" in output


@pytest.mark.parametrize("width", WIDTHS)
def test_top_users_preserve_netid_and_primary_counts(width: int):
    users = [
        UserUsage(
            user=USER,
            nodes={NODE_NAME},
            usage_by_partition={"priority": 12},
        )
    ]
    console, stream = _console(width)

    with patch(
        "gtop.render._lookup_full_name",
        return_value="A Very Long Full Name That Is Optional Detail",
    ):
        print_top_users(
            users,
            unit="GPU",
            console=console,
        )

    output = stream.getvalue()
    _assert_bounded(output, width)
    assert USER in output
    assert "12" in output
    assert "1" in output
    if width == 80:
        assert "A Very Long Full Name" not in output
    if width == 160:
        assert "A Very Long Full Name" in output


@pytest.mark.parametrize("width", JOB_WIDTHS)
def test_jobs_preserve_identifiers_and_primary_counts(width: int):
    job = _job()
    server = _server()
    process_jobs([job], {NODE_NAME: server})
    console, stream = _console(width)

    console.print(
        render_jobs_view(
            [job],
            servers={NODE_NAME: server},
            width=width,
        )
    )

    output = stream.getvalue()
    _assert_bounded(output, width)
    assert NODE_NAME in output
    assert GPU_TYPE in output
    if width in {60, 160}:
        assert USER in output
    else:
        assert USER.split("_", 1)[0] in output
    assert job.job_id in output
    assert "1/4 GPUs used" in output
    if width == 160:
        assert "·" in next(line for line in output.splitlines() if "GPUs used" in line)
    assert "8" in output
    assert "32G" in output
    if width == 80:
        header = next(line for line in output.splitlines() if line.startswith("ID "))
        assert header.split() == [
            "ID",
            "State",
            "User",
            "Partition",
            "GPU",
            "CPU",
            "MEM",
            "Time",
            "Name",
        ]
        assert "ID:" not in output
        assert "GPU/CPU/MEM:" not in output
    if width >= 80:
        assert "…" not in output


def test_job_header_uses_only_scoped_node_allocations():
    job = replace(
        _job(),
        nodelist="node-[1-2]",
        usage=JobUsage(cpu=8, gpu=2, mem=32),
    )
    servers = {
        name: ServerState(
            name=name,
            features={"gpu"},
            gpu=GpuInfo(type="a100", num=4),
            cpu=CpuInfo(total=8),
            mem=MemoryInfo(total=64),
        )
        for name in ("node-1", "node-2")
    }
    process_jobs([job], servers)
    console, stream = _console(120)

    console.print(
        render_jobs_view(
            [job],
            servers={"node-1": servers["node-1"]},
            width=120,
        )
    )

    assert "1/4 GPUs used" in stream.getvalue()


def test_job_header_converts_each_sharded_node_at_its_own_rate():
    job = replace(
        _job(),
        nodelist="node-1,node-2",
        usage=JobUsage(gpu=2),
    )
    servers = {
        "node-1": ServerState(
            name="node-1",
            features={"gpu"},
            gpu=GpuInfo(type="Shard(a)", num=2, shards=200),
            cpu=CpuInfo(),
            mem=MemoryInfo(),
        ),
        "node-2": ServerState(
            name="node-2",
            features={"gpu"},
            gpu=GpuInfo(type="Shard(b)", num=2, shards=400),
            cpu=CpuInfo(),
            mem=MemoryInfo(),
        ),
    }
    process_jobs([job], servers)
    console, stream = _console(120)

    console.print(render_jobs_view([job], servers=servers, width=120))

    assert "300/600 shards used" in stream.getvalue()


def test_free_gpus_need_4_cpus_and_16g_ram_on_the_node():
    from gtop.render_ready import build_availability
    from gtop.scheduling import parse_partitions
    from gtop.slurm import parse_sinfo

    servers = parse_sinfo(
        "\n".join(
            [
                # One job can take all 6 GPUs with 4 CPUs and 16G.
                "roomy|gpu-high|gpu:a100:6(S:0)|gpu:a100:0(IDX:N/A)|60/4/0/64|495616|512000",
                # Idle GPUs, but only 3 CPUs free.
                "cpu-bound|gpu-high|gpu:a100:2(S:0)|gpu:a100:0(IDX:N/A)|61/3/0/64|0|512000",
                # Idle GPUs, but only 10G RAM free.
                "mem-bound|gpu-high|gpu:h100:4(S:0)|gpu:h100:0(IDX:N/A)|0/64/0/64|501760|512000",
            ]
        )
    )
    partitions = parse_partitions(
        "PartitionName=gpu AllowGroups=ALL PriorityTier=5 "
        "Nodes=roomy,cpu-bound,mem-bound"
    )

    (result,) = build_availability(servers, partitions, [partitions["gpu"]])

    assert {row.gpu_type: (row.free, row.short) for row in result.rows} == {
        "A100": (6, 2),
        "H100": (0, 4),
    }
