# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.moe.routing_stats import (  # noqa: E402
    MoERoutingStatsRecorder,
    summarize_routing_stats,
)


def test_summarize_routing_stats_reports_occupancy_and_rank_skew():
    aggregate = [
        [
            2,
            1024,
            256,
            20,
            12,
            140,
            80,
            116,
            50,
            24,
            30,
            128,
        ]
    ]
    histogram = [[0] * 34]
    histogram[0][2] = 128
    histogram[0][6] = 128

    report = summarize_routing_stats(
        aggregate,
        histogram,
        layer_ids=(7,),
        num_experts=896,
        num_ranks=8,
        num_routes=512,
        block_size=64,
    )

    overall = report["overall"]
    assert report["decode_iterations_min"] == 2
    assert report["decode_iterations_max"] == 2
    assert overall["average_nonempty_block_size"] == 4
    assert overall["average_nonempty_block_utilization"] == 0.0625
    assert overall["largest_block_mean"] == 10
    assert overall["largest_block_max"] == 12
    assert overall["busiest_rank_excess_over_uniform_mean"] == 6
    assert overall["busiest_rank_excess_over_uniform_max"] == 16
    assert overall["rank_load_rms_error_from_uniform"] == pytest.approx(math.sqrt(8))
    assert overall["block_occupancy_quantiles"] == {
        "p50": 2,
        "p90": 6,
        "p95": 6,
        "p99": 6,
    }
    assert report["layers"][0]["layer"] == 7


def test_summarize_routing_stats_combines_layers():
    aggregate = [
        [1, 512, 128, 8, 8, 70, 70, 58, 58, 12, 12, 64],
        [1, 512, 64, 16, 16, 80, 80, 48, 48, 32, 32, 512],
    ]
    histogram = [[0] * 34 for _ in aggregate]
    histogram[0][4] = 128
    histogram[1][8] = 64

    report = summarize_routing_stats(
        aggregate,
        histogram,
        layer_ids=(1, 2),
        num_experts=896,
        num_ranks=8,
        num_routes=512,
        block_size=64,
    )

    overall = report["overall"]
    assert overall["observations"] == 2
    assert overall["route_conservation_ok"]
    assert overall["block_occupancy_conservation_ok"]
    assert overall["average_nonempty_block_size"] == pytest.approx(16 / 3)
    assert overall["largest_block_mean"] == 12
    assert overall["largest_block_max"] == 16
    assert overall["busiest_rank_routes_mean"] == 75
    assert overall["busiest_rank_excess_over_uniform_mean"] == 11
    assert overall["lightest_rank_routes_min"] == 48
    assert overall["rank_spread_max"] == 32


def test_routing_stats_reject_int32_accumulator_overflow():
    with pytest.raises(ValueError, match="int32 accumulator limit"):
        MoERoutingStatsRecorder(
            output_dir="unused",
            rank=0,
            layer_ids=(1,),
            num_experts=896,
            top_k=16,
            ep_size=8,
            batch_size=32,
            block_size=64,
            max_replays=9363,
            device="cpu",
        )
