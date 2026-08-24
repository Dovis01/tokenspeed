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

"""Fused small-M MoE block-align (decode): single-kernel, pure Gluon.

Both paths use one workgroup and preserve the same block-alignment contract:

  * in-kernel sentinel/zero init of the output + ``gl.barrier`` (folds away the
    separate init-kernel launch),
  * EP localization from global expert IDs plus per-expert route counts (remote
    routes are masked; EP = num_experts, no dump bin),
  * the quadratic path emits one block per hit expert when ``M <= block_m``;
    the LDS path emits ``ceil(count[e] / block_m)`` blocks and supports up to
    two blocks per expert,
  * ``gl.gather`` of the per-expert block offset + scatter each slot to
    ``block_off[e]*block_m + rank``.

At up to 128 routes, a [G, G] comparison tile computes stable per-expert ranks.
Above that, explicit LDS counters compute the same ranks in O(G + E) work. The
atomic route order is unspecified, as in the scalable multi-kernel align path.

Outputs retain static capacities for graph capture; ``num_valid`` (real padded
extent) stays on-device and the GEMM stages early-out on the unused tail.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

FUSED_ALIGN_QUADRATIC_MAX_ROUTES = 128
FUSED_ALIGN_MAX_ROUTES = 1024


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1)).bit_length()


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _fused_align_kernel(
    ids_ptr,  # [G] int32  flat topk_ids
    wts_ptr,  # [G] fp32   flat topk_weights
    sti_ptr,  # [EM_MAX] int32  out (packed slot<<24|token)
    sw_ptr,  # [EM_MAX] fp32   out (routed weight)
    sei_ptr,  # [NB_MAX] int32  out (expert per block, -1 pad)
    nv_ptr,  # [1] int32       out (EM)
    G,
    num_experts,
    block_m,
    sentinel,
    TOPK: gl.constexpr,
    GP: gl.constexpr,  # next_pow2(G)  (>= NB_MAX == G)
    EP: gl.constexpr,  # next_pow2(num_experts)
    EXPERT_START: gl.constexpr,
    NB_MAX: gl.constexpr,  # == G (max blocks) == sei length
    EM_MAX: gl.constexpr,  # == NB_MAX * block_m
    INIT_TILE: gl.constexpr,
):
    LG: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # [GP]
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # [EP]
    LR: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # init tile
    LT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])  # 2D rank

    # ---- in-kernel init of the full output (folds away a separate launch) ----
    for r0 in gl.static_range(0, EM_MAX, INIT_TILE):
        r = r0 + gl.arange(0, INIT_TILE, layout=LR)
        rm = r < EM_MAX
        gl.store(
            sti_ptr + r, gl.full([INIT_TILE], sentinel, gl.int32, layout=LR), mask=rm
        )
        gl.store(sw_ptr + r, gl.full([INIT_TILE], 0.0, gl.float32, layout=LR), mask=rm)
    jb = gl.arange(0, GP, layout=LG)
    gl.store(sei_ptr + jb, gl.full([GP], -1, gl.int32, layout=LG), mask=jb < NB_MAX)
    gl.barrier()  # order init stores before the (overlapping) scatter below

    g = gl.arange(0, GP, layout=LG)
    gmask = g < G
    global_idx = gl.load(ids_ptr + g, mask=gmask, other=EXPERT_START)
    idx = global_idx - EXPERT_START
    route_mask = gmask & (idx >= 0) & (idx < num_experts)
    safe_idx = gl.where(route_mask, idx, 0)
    vals = gl.load(wts_ptr + g, mask=route_mask, other=0.0)
    tok = g // TOPK
    slot = g % TOPK
    packed = ((slot << 24) | tok).to(gl.int32)

    # ---- per-expert counts (masked histogram -> masked lanes excluded) ----
    counts = gl.histogram(safe_idx, EP, mask=route_mask, layout=LE)
    e = gl.arange(0, EP, layout=LE)
    valid_e = e < num_experts
    # single-block collapse: 1 block per hit expert (count<=M<=block_m)
    hit = valid_e & (counts > 0)
    blocks_pe = hit.to(gl.int32)
    block_off = gl.associative_scan(blocks_pe, 0, _add) - blocks_pe  # exclusive
    num_blocks = gl.sum(blocks_pe, 0)
    gl.store(nv_ptr, num_blocks * block_m)  # EM

    # ---- sorted_expert_ids: scatter e -> sei[block_off[e]] (O(E)) ----
    gl.store(sei_ptr + block_off, e.to(gl.int32), mask=hit)

    # ---- stable per-expert rank via [G, G] compare tile ----
    idx_row = gl.expand_dims(
        gl.convert_layout(safe_idx, gl.SliceLayout(1, LT)),
        1,
    )
    idx_col = gl.expand_dims(
        gl.convert_layout(safe_idx, gl.SliceLayout(0, LT)),
        0,
    )
    valid_row = gl.expand_dims(
        gl.convert_layout(route_mask, gl.SliceLayout(1, LT)),
        1,
    )
    valid_col = gl.expand_dims(
        gl.convert_layout(route_mask, gl.SliceLayout(0, LT)),
        0,
    )
    g_row = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(1, LT)), 1)
    g_col = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(0, LT)), 0)
    match = ((idx_row == idx_col) & (g_col < g_row) & valid_row & valid_col).to(
        gl.int32
    )
    rank = gl.convert_layout(gl.sum(match, axis=1), LG)  # [GP]

    # ---- dest = block_off[expert]*block_m + rank, then scatter ----
    dest = gl.gather(block_off, safe_idx, axis=0) * block_m + rank
    gl.store(sti_ptr + dest, packed, mask=route_mask)
    gl.store(sw_ptr + dest, vals, mask=route_mask)


@gluon.jit
def _fused_align_lds_kernel(
    ids_ptr,  # [G] int32 flat topk_ids
    wts_ptr,  # [G] fp32 flat topk_weights
    sti_ptr,  # [EM_MAX] int32 out (packed slot<<24|token)
    sw_ptr,  # [EM_MAX] fp32 out (routed weight)
    sei_ptr,  # [NB_MAX] int32 out (expert per block, -1 pad)
    nv_ptr,  # [1] int32 out (padded active extent)
    G,
    num_experts,
    sentinel,
    TOPK: gl.constexpr,
    GP: gl.constexpr,
    EP: gl.constexpr,
    EXPERT_START: gl.constexpr,
    BLOCK_M: gl.constexpr,
    MAX_BLOCKS_PER_EXPERT: gl.constexpr,
    NB_MAX: gl.constexpr,
    EM_MAX: gl.constexpr,
    INIT_TILE: gl.constexpr,
):
    route_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    expert_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    init_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    counter_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[EP, 1]], [EP], [0]
    )
    counters = gl.allocate_shared_memory(gl.int32, [EP], counter_shared_layout)
    counter_zeros = gl.zeros([EP], gl.int32, layout=expert_layout)

    expert = gl.arange(0, EP, layout=expert_layout)
    for block_start in gl.static_range(0, NB_MAX, INIT_TILE):
        block = block_start + gl.arange(0, INIT_TILE, layout=init_layout)
        gl.store(
            sei_ptr + block,
            gl.full([INIT_TILE], -1, gl.int32, layout=init_layout),
            mask=block < NB_MAX,
        )
    counters.store(counter_zeros)
    gl.barrier()

    route = gl.arange(0, GP, layout=route_layout)
    route_in_bounds = route < G
    global_expert = gl.load(
        ids_ptr + route,
        mask=route_in_bounds,
        other=EXPERT_START + num_experts,
    )
    local_expert = global_expert - EXPERT_START
    local_route = route_in_bounds & (local_expert >= 0) & (local_expert < num_experts)
    safe_expert = gl.where(local_route, local_expert, 0).to(gl.int32)
    counters.atomic_scatter_add(
        gl.full([GP], 1, gl.int32, layout=route_layout),
        safe_expert,
        axis=0,
        mask=local_route,
    )
    gl.barrier()

    counts = counters.load(expert_layout)
    valid_expert = expert < num_experts
    hit = valid_expert & (counts > 0)
    blocks_per_expert = gl.where(hit, gl.cdiv(counts, BLOCK_M), 0)
    block_offset = gl.associative_scan(blocks_per_expert, 0, _add) - blocks_per_expert
    num_blocks = gl.sum(blocks_per_expert, 0)
    gl.store(nv_ptr, num_blocks * BLOCK_M)
    for block in gl.static_range(0, MAX_BLOCKS_PER_EXPERT):
        gl.store(
            sei_ptr + block_offset + block,
            expert.to(gl.int32),
            mask=hit & (block < blocks_per_expert),
        )

    # Only the padded active prefix is consumed downstream. Initialize that
    # dynamic prefix rather than the larger graph-capture allocation bound.
    active_rows = num_blocks * BLOCK_M
    for row_start in gl.static_range(0, EM_MAX, INIT_TILE):
        row = row_start + gl.arange(0, INIT_TILE, layout=init_layout)
        row_mask = row < active_rows
        gl.store(
            sti_ptr + row,
            gl.full([INIT_TILE], sentinel, gl.int32, layout=init_layout),
            mask=row_mask,
        )
        gl.store(
            sw_ptr + row,
            gl.full([INIT_TILE], 0.0, gl.float32, layout=init_layout),
            mask=row_mask,
        )

    # Reuse the histogram as per-expert reservation counters. Each real route
    # receives one unique row in its expert's block; route order is immaterial.
    counters.store(counter_zeros)
    gl.barrier()
    global_expert = gl.load(
        ids_ptr + route,
        mask=route_in_bounds,
        other=EXPERT_START + num_experts,
    )
    local_expert = global_expert - EXPERT_START
    local_route = route_in_bounds & (local_expert >= 0) & (local_expert < num_experts)
    safe_expert = gl.where(local_route, local_expert, 0).to(gl.int32)
    expert_row = counters.atomic_scatter_add(
        gl.full([GP], 1, gl.int32, layout=route_layout),
        safe_expert,
        axis=0,
        mask=local_route,
    )
    first_row = gl.gather(block_offset, safe_expert, axis=0) * BLOCK_M
    destination = first_row + expert_row
    token = route // TOPK
    slot = route % TOPK
    packed = ((slot << 24) | token).to(gl.int32)
    weight = gl.load(wts_ptr + route, mask=local_route, other=0.0)
    gl.store(sti_ptr + destination, packed, mask=local_route)
    gl.store(sw_ptr + destination, weight, mask=local_route)


def moe_align_block_size_fused(
    topk_ids: torch.Tensor,  # [M, topk] int
    topk_weights: torch.Tensor,  # [M, topk] float
    num_experts: int,
    block_m: int,
    *,
    expert_start: int = 0,
):
    """Single-kernel sync-free decode block-align (pure Gluon). Same return
    contract as ``moe_align_block_size``.

    ``topk_ids`` may use global expert IDs. ``expert_start`` identifies the
    first expert owned by this rank; routes outside the contiguous local range
    are ignored without a separate localization kernel.
    """
    assert topk_ids.shape == topk_weights.shape
    if expert_start < 0:
        raise ValueError("expert_start must be non-negative")
    device = topk_ids.device
    M, topk = topk_ids.shape
    G = M * topk
    sentinel = M
    max_blocks_per_expert = triton.cdiv(M, block_m)
    assert max_blocks_per_expert <= 2, (
        "fused small-M align supports at most two blocks per expert; "
        f"got M={M}, block_m={block_m}"
    )

    # Preserve the original generic fallback outside the bounded LDS path;
    # Kimi's grouped caller only selects this function through MAX_ROUTES.
    use_quadratic_rank = (
        G <= FUSED_ALIGN_QUADRATIC_MAX_ROUTES and max_blocks_per_expert == 1
    )
    NB_MAX = G if use_quadratic_rank else min(G, num_experts * max_blocks_per_expert)
    EM_MAX = NB_MAX * block_m
    GP = _next_pow2(G)
    EP = _next_pow2(num_experts)

    ids = topk_ids.reshape(-1).to(torch.int32).contiguous()
    wts = topk_weights.reshape(-1).to(torch.float32).contiguous()
    sti = torch.empty(EM_MAX, dtype=torch.int32, device=device)
    sw = torch.empty(EM_MAX, dtype=torch.float32, device=device)
    sei = torch.empty(NB_MAX, dtype=torch.int32, device=device)
    nv = torch.empty(1, dtype=torch.int32, device=device)

    if use_quadratic_rank:
        _fused_align_kernel[(1,)](
            ids,
            wts,
            sti,
            sw,
            sei,
            nv,
            G,
            num_experts,
            block_m,
            sentinel,
            TOPK=topk,
            GP=GP,
            EP=EP,
            EXPERT_START=expert_start,
            NB_MAX=NB_MAX,
            EM_MAX=EM_MAX,
            INIT_TILE=_next_pow2(min(1024, EM_MAX)),
            num_warps=4,
        )
    else:
        _fused_align_lds_kernel[(1,)](
            ids,
            wts,
            sti,
            sw,
            sei,
            nv,
            G,
            num_experts,
            sentinel,
            TOPK=topk,
            GP=GP,
            EP=EP,
            EXPERT_START=expert_start,
            BLOCK_M=block_m,
            MAX_BLOCKS_PER_EXPERT=max_blocks_per_expert,
            NB_MAX=NB_MAX,
            EM_MAX=EM_MAX,
            INIT_TILE=_next_pow2(min(1024, EM_MAX)),
            num_warps=4,
        )
    return sti, sei, sw, nv
