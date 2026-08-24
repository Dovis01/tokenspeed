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

"""On-device expert-routing statistics for gfx950 decode workloads."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

ROUTING_STATS_FIELDS = (
    "observations",
    "routes",
    "nonempty_blocks",
    "largest_block_sum",
    "largest_block_max",
    "busiest_rank_sum",
    "busiest_rank_max",
    "lightest_rank_sum",
    "lightest_rank_min",
    "rank_spread_sum",
    "rank_spread_max",
    "rank_squared_error_sum",
)


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


@gluon.jit
def _record_routing_counts_kernel(
    topk_ids_ptr,
    counts_ptr,
    NUM_ROUTES: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    ROUTE_BLOCK: gl.constexpr,
    EXPERT_BLOCK: gl.constexpr,
):
    route_layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    expert_layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    counter_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[EXPERT_BLOCK, 1]], [EXPERT_BLOCK], [0]
    )
    counters = gl.allocate_shared_memory(gl.int32, [EXPERT_BLOCK], counter_layout)

    expert = gl.arange(0, EXPERT_BLOCK, layout=expert_layout)
    counters.store(gl.zeros([EXPERT_BLOCK], gl.int32, layout=expert_layout))
    gl.barrier()

    route = gl.arange(0, ROUTE_BLOCK, layout=route_layout)
    route_mask = route < NUM_ROUTES
    selected_expert = gl.load(topk_ids_ptr + route, mask=route_mask, other=-1).to(
        gl.int32
    )
    valid = route_mask & (selected_expert >= 0) & (selected_expert < NUM_EXPERTS)
    safe_expert = gl.where(valid, selected_expert, 0)
    counters.atomic_scatter_add(
        gl.full([ROUTE_BLOCK], 1, gl.int32, layout=route_layout),
        safe_expert,
        axis=0,
        mask=valid,
    )
    gl.barrier()

    counts = counters.load(expert_layout)
    gl.store(counts_ptr + expert, counts, mask=expert < NUM_EXPERTS)


@gluon.jit
def _accumulate_routing_stats_kernel(
    counts_ptr,
    expert_totals_ptr,
    aggregate_ptr,
    occupancy_histogram_ptr,
    enabled_ptr,
    NUM_LAYERS: gl.constexpr,
    NUM_ROUTES: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    EXPERT_BLOCK: gl.constexpr,
    NUM_RANKS: gl.constexpr,
    EXPERTS_PER_RANK: gl.constexpr,
    MAX_OCCUPANCY: gl.constexpr,
    NUM_FIELDS: gl.constexpr,
):
    expert_layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    layer = gl.program_id(0)
    expert = gl.arange(0, EXPERT_BLOCK, layout=expert_layout)
    expert_mask = expert < NUM_EXPERTS
    counts = gl.load(
        counts_ptr + layer * NUM_EXPERTS + expert,
        mask=expert_mask,
        other=0,
    ).to(gl.int32)
    enabled = gl.load(enabled_ptr) != 0

    previous_total = gl.load(
        expert_totals_ptr + layer * NUM_EXPERTS + expert,
        mask=expert_mask,
        other=0,
    )
    gl.store(
        expert_totals_ptr + layer * NUM_EXPERTS + expert,
        previous_total + counts,
        mask=enabled & expert_mask,
    )

    routes = gl.sum(counts, axis=0)
    nonempty_blocks = gl.sum((counts > 0).to(gl.int32), axis=0)
    largest_block = gl.max(counts, axis=0)

    busiest_rank = 0
    lightest_rank = NUM_ROUTES
    rank_squared_error = 0
    ideal_rank_routes: gl.constexpr = NUM_ROUTES // NUM_RANKS
    for rank in gl.static_range(NUM_RANKS):
        rank_routes = gl.sum(
            gl.where(
                expert_mask
                & (expert >= rank * EXPERTS_PER_RANK)
                & (expert < (rank + 1) * EXPERTS_PER_RANK),
                counts,
                0,
            ),
            axis=0,
        )
        busiest_rank = gl.maximum(busiest_rank, rank_routes)
        lightest_rank = gl.minimum(lightest_rank, rank_routes)
        rank_delta = rank_routes - ideal_rank_routes
        rank_squared_error += rank_delta * rank_delta

    rank_spread = busiest_rank - lightest_rank
    aggregate_base = layer * NUM_FIELDS

    old = gl.load(aggregate_ptr + aggregate_base + 0)
    gl.store(aggregate_ptr + aggregate_base + 0, old + 1, mask=enabled)
    old = gl.load(aggregate_ptr + aggregate_base + 1)
    gl.store(aggregate_ptr + aggregate_base + 1, old + routes, mask=enabled)
    old = gl.load(aggregate_ptr + aggregate_base + 2)
    gl.store(
        aggregate_ptr + aggregate_base + 2,
        old + nonempty_blocks,
        mask=enabled,
    )
    old = gl.load(aggregate_ptr + aggregate_base + 3)
    gl.store(
        aggregate_ptr + aggregate_base + 3,
        old + largest_block,
        mask=enabled,
    )
    old = gl.load(aggregate_ptr + aggregate_base + 4)
    gl.store(
        aggregate_ptr + aggregate_base + 4,
        gl.maximum(old, largest_block),
        mask=enabled,
    )
    old = gl.load(aggregate_ptr + aggregate_base + 5)
    gl.store(
        aggregate_ptr + aggregate_base + 5,
        old + busiest_rank,
        mask=enabled,
    )
    old = gl.load(aggregate_ptr + aggregate_base + 6)
    gl.store(
        aggregate_ptr + aggregate_base + 6,
        gl.maximum(old, busiest_rank),
        mask=enabled,
    )
    old = gl.load(aggregate_ptr + aggregate_base + 7)
    gl.store(
        aggregate_ptr + aggregate_base + 7,
        old + lightest_rank,
        mask=enabled,
    )
    old = gl.load(aggregate_ptr + aggregate_base + 8)
    gl.store(
        aggregate_ptr + aggregate_base + 8,
        gl.minimum(old, lightest_rank),
        mask=enabled,
    )
    old = gl.load(aggregate_ptr + aggregate_base + 9)
    gl.store(
        aggregate_ptr + aggregate_base + 9,
        old + rank_spread,
        mask=enabled,
    )
    old = gl.load(aggregate_ptr + aggregate_base + 10)
    gl.store(
        aggregate_ptr + aggregate_base + 10,
        gl.maximum(old, rank_spread),
        mask=enabled,
    )
    old = gl.load(aggregate_ptr + aggregate_base + 11)
    gl.store(
        aggregate_ptr + aggregate_base + 11,
        old + rank_squared_error,
        mask=enabled,
    )

    histogram_base = layer * (MAX_OCCUPANCY + 2)
    for occupancy in gl.static_range(1, MAX_OCCUPANCY + 1):
        frequency = gl.sum((expert_mask & (counts == occupancy)).to(gl.int32), axis=0)
        old = gl.load(occupancy_histogram_ptr + histogram_base + occupancy)
        gl.store(
            occupancy_histogram_ptr + histogram_base + occupancy,
            old + frequency,
            mask=enabled,
        )
    overflow = gl.sum((expert_mask & (counts > MAX_OCCUPANCY)).to(gl.int32), axis=0)
    old = gl.load(occupancy_histogram_ptr + histogram_base + MAX_OCCUPANCY + 1)
    gl.store(
        occupancy_histogram_ptr + histogram_base + MAX_OCCUPANCY + 1,
        old + overflow,
        mask=enabled,
    )


def record_routing_counts(topk_ids: torch.Tensor, counts: torch.Tensor) -> None:
    """Record one layer's exact per-expert route counts in one workgroup."""
    if topk_ids.device != counts.device:
        raise ValueError("topk_ids and counts must be on the same device")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must be shaped [tokens, top_k]")
    if counts.ndim != 1 or counts.dtype != torch.int32:
        raise ValueError("counts must be a one-dimensional int32 tensor")

    num_routes = topk_ids.numel()
    num_experts = counts.numel()
    route_block = _next_power_of_two(num_routes)
    expert_block = _next_power_of_two(num_experts)
    _record_routing_counts_kernel[(1,)](
        topk_ids,
        counts,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        ROUTE_BLOCK=route_block,
        EXPERT_BLOCK=expert_block,
        num_warps=8,
    )


def accumulate_routing_stats(
    counts: torch.Tensor,
    expert_totals: torch.Tensor,
    aggregate: torch.Tensor,
    occupancy_histogram: torch.Tensor,
    enabled: torch.Tensor,
    *,
    num_routes: int,
    num_ranks: int,
    max_occupancy: int,
) -> None:
    """Fold one completed decode iteration into persistent per-layer totals."""
    if counts.ndim != 2 or counts.dtype != torch.int32:
        raise ValueError("counts must be a two-dimensional int32 tensor")
    num_layers, num_experts = counts.shape
    if expert_totals.shape != counts.shape or expert_totals.dtype != torch.int32:
        raise ValueError("expert_totals must match counts as an int32 tensor")
    if num_experts % num_ranks:
        raise ValueError("num_experts must be divisible by num_ranks")
    if aggregate.shape != (num_layers, len(ROUTING_STATS_FIELDS)):
        raise ValueError("aggregate has the wrong shape")
    if aggregate.dtype != torch.int32:
        raise ValueError("aggregate must use int32 storage")
    if occupancy_histogram.shape != (num_layers, max_occupancy + 2):
        raise ValueError("occupancy_histogram has the wrong shape")
    if occupancy_histogram.dtype != torch.int32:
        raise ValueError("occupancy_histogram must use int32 storage")
    if enabled.shape != (1,) or enabled.dtype != torch.int32:
        raise ValueError("enabled must be a one-element int32 tensor")
    devices = {
        counts.device,
        expert_totals.device,
        aggregate.device,
        occupancy_histogram.device,
        enabled.device,
    }
    if len(devices) != 1:
        raise ValueError("routing-stat tensors must be on the same device")

    _accumulate_routing_stats_kernel[(num_layers,)](
        counts,
        expert_totals,
        aggregate,
        occupancy_histogram,
        enabled,
        NUM_LAYERS=num_layers,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        EXPERT_BLOCK=_next_power_of_two(num_experts),
        NUM_RANKS=num_ranks,
        EXPERTS_PER_RANK=num_experts // num_ranks,
        MAX_OCCUPANCY=max_occupancy,
        NUM_FIELDS=len(ROUTING_STATS_FIELDS),
        num_warps=8,
    )


__all__ = [
    "ROUTING_STATS_FIELDS",
    "accumulate_routing_stats",
    "record_routing_counts",
]
