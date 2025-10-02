#!/usr/bin/env python3
"""
Test suite for gtop --disp-shard functionality
"""

from gtop import parse_gpu, format_resource


def test_default_shows_gpu_count_not_shards():
    """Test that by default, GPU count is shown, not shard info"""

    # Sharded GPU example (dutta-compute-01)
    gres = "gpu:nvidia_h100_nvl:2(S:12-23),shard:nvidia_h100_nvl:48(S:12-23)"
    gres_used = "gpu:nvidia_h100_nvl:1(IDX:0),shard:nvidia_h100_nvl:0(0/24,0/24)"
    result = parse_gpu(gres, gres_used)

    # Should have 2 GPUs and 48 shards
    assert result["num"] == 2
    assert result["shards"] == 48

    # Mock server info structure
    info = {
        "gpu": result,
        "usage": {
            "gpu": {"priority": 0, "default": 0}
        }
    }

    # Default behavior (disp_shard=False): should show GPU count
    output_default = format_resource(info, "gpu", disp_shard=False)
    # With 2 GPUs total, 0 priority, 0 default -> 0/0/2
    assert output_default == "0/0/2"

    # With --disp-shard: should show shard count
    output_shard = format_resource(info, "gpu", disp_shard=True)
    # With 48 shards, 0 priority, 0 default -> 0/0/48
    assert output_shard == "0/0/48"


def test_disp_shard_with_usage():
    """Test shard display with actual usage"""

    gres = "gpu:nvidia_a40:2(S:1),shard:nvidia_a40:400(S:1)"
    gres_used = "gpu:nvidia_a40:1(IDX:0),shard:nvidia_a40:100(0/200)"
    result = parse_gpu(gres, gres_used)

    # Mock server info with some usage
    info = {
        "gpu": result,
        "usage": {
            "gpu": {"priority": 1, "default": 0}
        }
    }

    # Default: show GPU count (2 total, 1 used)
    output_default = format_resource(info, "gpu", disp_shard=False)
    assert output_default == "1/0/1"  # 1 priority, 0 default, 1 idle

    # With --disp-shard: show shard count (400 total, 1 used)
    output_shard = format_resource(info, "gpu", disp_shard=True)
    assert output_shard == "1/0/399"  # 1 priority, 0 default, 399 idle


def test_non_sharded_gpu_unchanged():
    """Test that non-sharded GPUs behave the same regardless of disp_shard flag"""

    # Regular A6000 GPU
    gres = "gpu:nvidia_rtx_a6000:8(S:0)"
    gres_used = "gpu:nvidia_rtx_a6000:2(IDX:0-1)"
    result = parse_gpu(gres, gres_used)

    # Should have 8 GPUs, 0 shards
    assert result["num"] == 8
    assert result["shards"] == 0

    info = {
        "gpu": result,
        "usage": {
            "gpu": {"priority": 1, "default": 1}
        }
    }

    # Both should show the same output for non-sharded GPUs
    output_default = format_resource(info, "gpu", disp_shard=False)
    output_shard = format_resource(info, "gpu", disp_shard=True)

    # 1 priority, 1 default, 6 idle
    assert output_default == "1/1/6"
    assert output_shard == "1/1/6"  # Same output


def test_regular_gpu_type_display():
    """Test that GPU type is correctly extracted from Shard() format for display"""

    # Sharded GPU with specific type
    gres = "gpu:nvidia_h100_nvl:2(S:12-23),shard:nvidia_h100_nvl:48(S:12-23)"
    result = parse_gpu(gres)

    # Type should be in Shard(gpu_type) format
    assert result["type"].startswith("Shard(")
    assert "nvidia_h100_nvl" in result["type"]

    # When disp_shard=False, the display should show GPU count with base type
    # (tested via integration, as display logic is in create_main_table)


def test_integration_default_vs_disp_shard():
    """Integration test showing difference between default and --disp-shard behavior"""

    # Parse a sharded GPU node
    gres = "gpu:nvidia_a40:2(S:1),shard:nvidia_a40:400(S:1)"
    gres_used = "none"
    result = parse_gpu(gres, gres_used)

    # Verify parsing is correct
    assert result["num"] == 2  # 2 physical GPUs
    assert result["shards"] == 400  # 400 shards available

    # Create mock server info
    info = {
        "gpu": result,
        "usage": {
            "gpu": {"priority": 0, "default": 1}  # 1 default job using resources
        }
    }

    # Default behavior: show GPU count
    # With 2 GPUs and 1 in use: priority=0, default=1, idle=1
    default_output = format_resource(info, "gpu", disp_shard=False)
    assert default_output == "0/1/1"

    # With --disp-shard: show shard count
    # With 400 shards and 1 in use: priority=0, default=1, idle=399
    shard_output = format_resource(info, "gpu", disp_shard=True)
    assert shard_output == "0/1/399"
