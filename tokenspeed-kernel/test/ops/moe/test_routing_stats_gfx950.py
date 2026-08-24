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

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

if not current_platform().is_cdna4:
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.moe.routing_stats import (  # noqa: E402
    ROUTING_STATS_FIELDS,
    accumulate_routing_stats,
    record_routing_counts,
)


def _expected_fields(counts: torch.Tensor, num_ranks: int) -> list[int]:
    counts = counts.cpu()
    num_experts = counts.numel()
    rank_loads = counts.reshape(num_ranks, num_experts // num_ranks).sum(dim=1)
    ideal = int(counts.sum().item()) // num_ranks
    return [
        1,
        int(counts.sum().item()),
        int((counts > 0).sum().item()),
        int(counts.max().item()),
        int(counts.max().item()),
        int(rank_loads.max().item()),
        int(rank_loads.max().item()),
        int(rank_loads.min().item()),
        int(rank_loads.min().item()),
        int((rank_loads.max() - rank_loads.min()).item()),
        int((rank_loads.max() - rank_loads.min()).item()),
        int(((rank_loads - ideal) ** 2).sum().item()),
    ]


def _expected_histogram(counts: torch.Tensor, max_occupancy: int) -> list[int]:
    histogram = [0] * (max_occupancy + 2)
    for value in counts.cpu().tolist():
        if value > max_occupancy:
            histogram[-1] += 1
        elif value > 0:
            histogram[value] += 1
    return histogram


def test_routing_stats_match_exact_counts_and_online_aggregation():
    batch_size = 32
    top_k = 16
    num_experts = 896
    num_ranks = 8
    layer_ids = torch.empty((2, batch_size, top_k), dtype=torch.int32, device="cuda")
    layer_ids[0] = torch.arange(top_k, device="cuda", dtype=torch.int32).repeat(
        batch_size, 1
    )
    layer_ids[1] = torch.arange(
        batch_size * top_k, device="cuda", dtype=torch.int32
    ).reshape(batch_size, top_k)

    counts = torch.empty((2, num_experts), dtype=torch.int32, device="cuda")
    expert_totals = torch.zeros_like(counts)
    aggregate = torch.zeros(
        (2, len(ROUTING_STATS_FIELDS)), dtype=torch.int32, device="cuda"
    )
    aggregate[:, ROUTING_STATS_FIELDS.index("lightest_rank_min")].fill_(
        batch_size * top_k
    )
    histogram = torch.zeros((2, batch_size + 2), dtype=torch.int32, device="cuda")
    enabled = torch.ones(1, dtype=torch.int32, device="cuda")

    for row in range(2):
        record_routing_counts(layer_ids[row], counts[row])
    accumulate_routing_stats(
        counts,
        expert_totals,
        aggregate,
        histogram,
        enabled,
        num_routes=batch_size * top_k,
        num_ranks=num_ranks,
        max_occupancy=batch_size,
    )
    torch.cuda.synchronize()

    reference_counts = [
        torch.bincount(layer_ids[row].flatten().cpu(), minlength=num_experts)
        for row in range(2)
    ]
    torch.testing.assert_close(
        counts.cpu(), torch.stack(reference_counts).to(torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        expert_totals.cpu(),
        torch.stack(reference_counts).to(torch.int32),
        rtol=0,
        atol=0,
    )
    assert aggregate.cpu().tolist() == [
        _expected_fields(row, num_ranks) for row in reference_counts
    ]
    assert histogram.cpu().tolist() == [
        _expected_histogram(row, batch_size) for row in reference_counts
    ]


def test_routing_stats_are_graph_capturable_and_runtime_gated():
    batch_size = 32
    top_k = 16
    num_experts = 896
    ids = torch.arange(batch_size * top_k, dtype=torch.int32, device="cuda").reshape(
        batch_size, top_k
    )
    counts = torch.empty((1, num_experts), dtype=torch.int32, device="cuda")
    expert_totals = torch.zeros_like(counts)
    aggregate = torch.zeros(
        (1, len(ROUTING_STATS_FIELDS)), dtype=torch.int32, device="cuda"
    )
    aggregate[:, ROUTING_STATS_FIELDS.index("lightest_rank_min")].fill_(
        batch_size * top_k
    )
    histogram = torch.zeros((1, batch_size + 2), dtype=torch.int32, device="cuda")
    enabled = torch.zeros(1, dtype=torch.int32, device="cuda")

    record_routing_counts(ids, counts[0])
    accumulate_routing_stats(
        counts,
        expert_totals,
        aggregate,
        histogram,
        enabled,
        num_routes=batch_size * top_k,
        num_ranks=8,
        max_occupancy=batch_size,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        record_routing_counts(ids, counts[0])
        accumulate_routing_stats(
            counts,
            expert_totals,
            aggregate,
            histogram,
            enabled,
            num_routes=batch_size * top_k,
            num_ranks=8,
            max_occupancy=batch_size,
        )

    torch.cuda.synchronize()
    assert aggregate[0, 0].item() == 0
    enabled.fill_(1)
    graph.replay()
    torch.cuda.synchronize()

    reference_counts = torch.bincount(ids.flatten().cpu(), minlength=num_experts)
    torch.testing.assert_close(
        expert_totals.cpu()[0], reference_counts.to(torch.int32), rtol=0, atol=0
    )
    assert aggregate.cpu()[0].tolist() == _expected_fields(reference_counts, 8)
    assert histogram.cpu()[0].tolist() == _expected_histogram(
        reference_counts, batch_size
    )
