#!/usr/bin/env python3
"""
Test suite for gtop --disp-shard functionality
"""

from gtop import CpuInfo, MemoryInfo, ResourceUsageSplit, ServerState, format_resource
from gtop.resources import parse_gpu


def make_server(
    gpu_gres: str,
    gpu_used: str,
    *,
    gpu_priority: float = 0,
    gpu_default: float = 0,
    shard_priority: float = 0,
    shard_default: float = 0,
) -> ServerState:
    server = ServerState(
        name="demo-node",
        features=set(),
        gpu=parse_gpu(gpu_gres, gpu_used),
        cpu=CpuInfo(),
        mem=MemoryInfo(),
    )
    server.usage["gpu"] = ResourceUsageSplit(priority=gpu_priority, default=gpu_default)
    server.usage["shard"] = ResourceUsageSplit(
        priority=shard_priority,
        default=shard_default,
    )
    return server


def test_default_shows_gpu_count_not_shards():
    """Test that by default, GPU count is shown, not shard info"""

    # Sharded GPU example (dutta-compute-01)
    gres = "gpu:nvidia_h100_nvl:2(S:12-23),shard:nvidia_h100_nvl:48(S:12-23)"
    gres_used = "gpu:nvidia_h100_nvl:1(IDX:0),shard:nvidia_h100_nvl:0(0/24,0/24)"
    server = make_server(gres, gres_used)

    # Should have 2 GPUs and 48 shards
    assert server.gpu.num == 2
    assert server.gpu.shards == 48

    # Default behavior (disp_shard=False): should show GPU count
    output_default = format_resource(server, "gpu", show_shards=False)
    # With 2 GPUs total, 0 priority, 0 gpu, 0 default -> 0/0/0/2
    assert output_default == "0/0/0/2"

    # With --disp-shard: should show shard count
    output_shard = format_resource(server, "gpu", show_shards=True)
    # With 48 shards, 0 priority, 0 gpu, 0 default -> 0/0/0/48
    assert output_shard == "0/0/0/48"


def test_disp_shard_with_usage():
    """Test shard display with actual usage"""

    gres = "gpu:nvidia_a40:2(S:1),shard:nvidia_a40:400(S:1)"
    gres_used = "gpu:nvidia_a40:1(IDX:0),shard:nvidia_a40:100(0/200)"
    server = make_server(
        gres,
        gres_used,
        gpu_priority=1,
        shard_priority=1,
    )

    # Default: show GPU count (2 total, 1 used)
    output_default = format_resource(server, "gpu", show_shards=False)
    assert output_default == "1/0/0/1"  # 1 priority, 0 gpu, 0 default, 1 idle

    # With --disp-shard: show shard count (400 total, 1 used)
    output_shard = format_resource(server, "gpu", show_shards=True)
    assert output_shard == "1/0/0/399"  # 1 priority, 0 gpu, 0 default, 399 idle


def test_non_sharded_gpu_unchanged():
    """Test that non-sharded GPUs behave the same regardless of disp_shard flag"""

    # Regular A6000 GPU
    gres = "gpu:nvidia_rtx_a6000:8(S:0)"
    gres_used = "gpu:nvidia_rtx_a6000:2(IDX:0-1)"
    server = make_server(
        gres,
        gres_used,
        gpu_priority=1,
        gpu_default=1,
    )

    # Should have 8 GPUs, 0 shards
    assert server.gpu.num == 8
    assert server.gpu.shards == 0

    # Both should show the same output for non-sharded GPUs
    output_default = format_resource(server, "gpu", show_shards=False)
    output_shard = format_resource(server, "gpu", show_shards=True)

    # 1 priority, 0 gpu, 1 default, 6 idle
    assert output_default == "1/0/1/6"
    assert output_shard == "1/0/1/6"  # Same output


def test_regular_gpu_type_display():
    """Test that GPU type is correctly extracted from Shard() format for display"""

    # Sharded GPU with specific type
    gres = "gpu:nvidia_h100_nvl:2(S:12-23),shard:nvidia_h100_nvl:48(S:12-23)"
    result = parse_gpu(gres)

    # Type should be in Shard(gpu_type) format
    assert result.type.startswith("Shard(")
    assert "nvidia_h100_nvl" in result.type

    # When disp_shard=False, the display should show GPU count with base type
    # (tested via integration, as display logic is in create_main_table)


def test_integration_default_vs_disp_shard():
    """Integration test showing difference between default and --disp-shard behavior"""

    # Parse a sharded GPU node
    gres = "gpu:nvidia_a40:2(S:1),shard:nvidia_a40:400(S:1)"
    gres_used = "none"
    server = make_server(
        gres,
        gres_used,
        gpu_default=1,
        shard_default=1,
    )

    # Verify parsing is correct
    assert server.gpu.num == 2  # 2 physical GPUs
    assert server.gpu.shards == 400  # 400 shards available

    # Default behavior: show GPU count
    # With 2 GPUs and 1 in use: priority=0, default=1, idle=1
    default_output = format_resource(server, "gpu", show_shards=False)
    assert default_output == "0/0/1/1"

    # With --disp-shard: show shard count
    # With 400 shards and 1 in use: priority=0, default=1, idle=399
    shard_output = format_resource(server, "gpu", show_shards=True)
    assert shard_output == "0/0/1/399"
