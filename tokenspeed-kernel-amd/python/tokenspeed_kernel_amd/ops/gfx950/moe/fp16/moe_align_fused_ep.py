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

"""Expert-parallel fused small-M MoE block-align (decode): one kernel, no atomics.

Motivation
----------
``moe_align_block_size_fused`` already collapses the whole align preamble into a
single kernel, but its launcher is gated on ``num_routes = M * topk <= 128``
because two of its internals scale with ``num_routes``:

  * ``NB_MAX = G`` (one block per *route*), so the in-kernel output init is
    ``O(G * block_m)`` work done by a **single** workgroup, and
  * the stable rank uses an ``O(G^2)`` ``[GP, GP]`` compare tile held in the
    registers of that one workgroup.

At the decode shapes this stack actually runs (``M = 32``, ``topk = 16``,
``G = 512``) both bounds are violated, so decode falls back to the five-launch
``moe_align_block_size_device`` path (``_init`` + ``torch.zeros`` memset +
``_count`` + single-CTA ``_offsets`` + ``_scatter``).

This module keeps the single-block collapse -- which depends only on
``M <= block_m``, *not* on ``num_routes`` -- and removes both bounds:

  * ``NB_MAX = min(G, num_experts)``: under the collapse each *hit expert* owns
    exactly one block, so the number of blocks can never exceed the number of
    experts. At ``G = 512, E = 112`` this shrinks the output from 32768 slots to
    7168.
  * The rank is computed **per expert, one workgroup per expert**, so the
    ``O(G^2)`` compare tile becomes an ``O(G)`` exclusive scan of a boolean
    "is this route mine" mask. ``num_experts`` workgroups replace one.

Each workgroup ``e``:
  1. loads the flat ``[G]`` route ids/weights and localizes them against
     ``expert_start`` (remote routes masked out),
  2. recomputes the cheap ``gl.histogram`` over all experts -- redundant across
     workgroups but ``O(G)`` and L2-resident -- to derive ``hit = count > 0``
     and its own ``blocks_before = popcount(hit[:e])``,
  3. initialises **its own** ``block_m`` output slots to the sentinel/zero pad
     (so no separate init launch and no ``torch.zeros`` memset), plus one tail
     block ``num_blocks + e`` when that index is still inside ``NB_MAX`` -- the
     two together cover every block index exactly once,
  4. scatters its routes to ``blocks_before * block_m + rank``.

No device atomics, no cross-workgroup dependency, no host sync: ``num_valid``
(the real EM) is written on-device by workgroup 0 and the GEMM stages early-out
on the padded tail exactly as before.

Return contract is identical to ``moe_align_block_size_device`` /
``moe_align_block_size_fused``: ``(sorted_ids, sorted_expert_ids,
sorted_weights, num_valid)`` with ``sorted_ids`` packed as
``(slot << 24) | token`` and the pad sentinel equal to ``M``.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1)).bit_length()


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _fused_align_ep_kernel(
    ids_ptr,  # [G] int32  flat topk_ids (may hold global expert ids)
    wts_ptr,  # [G] fp32   flat topk_weights
    sti_ptr,  # [EM_MAX] int32  out (packed slot<<24|token)
    sw_ptr,  # [EM_MAX] fp32   out (routed weight)
    sei_ptr,  # [NB_MAX] int32  out (expert per block, -1 pad)
    nv_ptr,  # [1] int32       out (EM)
    G,
    num_experts,
    sentinel,
    TOPK: gl.constexpr,
    GP: gl.constexpr,  # next_pow2(G)
    EP: gl.constexpr,  # next_pow2(num_experts)
    EXPERT_START: gl.constexpr,
    BM: gl.constexpr,  # block_m (power of two)
    NB_MAX: gl.constexpr,  # min(G, num_experts)
    SPT_G: gl.constexpr,  # elements per thread for the [GP] tile
):
    LG: gl.constexpr = gl.BlockedLayout([SPT_G], [64], [4], [0])  # [GP]
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # [EP]
    LB: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # [BM]

    pid = gl.program_id(0)

    g = gl.arange(0, GP, layout=LG)
    gmask = g < G
    global_idx = gl.load(ids_ptr + g, mask=gmask, other=EXPERT_START)
    idx = global_idx - EXPERT_START
    route_mask = gmask & (idx >= 0) & (idx < num_experts)
    safe_idx = gl.where(route_mask, idx, 0)

    # ---- per-expert counts (masked histogram -> masked lanes excluded) ----
    counts = gl.histogram(safe_idx, EP, mask=route_mask, layout=LE)
    e = gl.arange(0, EP, layout=LE)
    valid_e = e < num_experts
    # single-block collapse: top-k holds each expert at most once per token, so
    # at M <= block_m every hit expert is exactly one block.
    hit = valid_e & (counts > 0)
    num_blocks = gl.sum(hit.to(gl.int32), 0)

    if pid == 0:
        gl.store(nv_ptr, num_blocks * BM)

    # ---- pad the tail block this workgroup is responsible for ----
    # Block indices [num_blocks, NB_MAX) are covered by pids 0 .. NB_MAX-1-
    # num_blocks; the hit experts below cover [0, num_blocks). Every block index
    # is therefore written exactly once, without a separate init launch.
    tail_b = num_blocks + pid
    if tail_b < NB_MAX:
        tb = gl.arange(0, BM, layout=LB)
        tbase = tail_b * BM
        gl.store(sti_ptr + tbase + tb, gl.full([BM], sentinel, gl.int32, layout=LB))
        gl.store(sw_ptr + tbase + tb, gl.full([BM], 0.0, gl.float32, layout=LB))
        gl.store(sei_ptr + tail_b, -1)

    mine = route_mask & (safe_idx == pid)
    my_count = gl.sum(mine.to(gl.int32), 0)
    if my_count > 0:
        blocks_before = gl.sum((hit & (e < pid)).to(gl.int32), 0)
        base = blocks_before * BM

        b = gl.arange(0, BM, layout=LB)
        gl.store(sti_ptr + base + b, gl.full([BM], sentinel, gl.int32, layout=LB))
        gl.store(sw_ptr + base + b, gl.full([BM], 0.0, gl.float32, layout=LB))
        gl.store(sei_ptr + blocks_before, pid)
        gl.barrier()  # order the pad stores before the overlapping scatter

        m_i32 = mine.to(gl.int32)
        rank = gl.associative_scan(m_i32, 0, _add) - m_i32  # exclusive
        tok = g // TOPK
        slot = g % TOPK
        packed = ((slot << 24) | tok).to(gl.int32)
        vals = gl.load(wts_ptr + g, mask=mine, other=0.0)
        dest = base + rank
        gl.store(sti_ptr + dest, packed, mask=mine)
        gl.store(sw_ptr + dest, vals, mask=mine)


def moe_align_block_size_fused_ep(
    topk_ids: torch.Tensor,  # [M, topk] int
    topk_weights: torch.Tensor,  # [M, topk] float
    num_experts: int,
    block_m: int,
    *,
    expert_start: int = 0,
):
    """Single-kernel, expert-parallel, sync-free decode block-align.

    Args:
        topk_ids: ``[M, topk]`` integer routes. May carry global expert IDs;
            ``expert_start`` localizes them and out-of-range routes are dropped.
        topk_weights: ``[M, topk]`` routing weights, same shape as ``topk_ids``.
        num_experts: number of experts owned by this rank.
        block_m: GEMM block size along M. Must be a power of two and satisfy
            ``M <= block_m`` (the single-block collapse precondition).
        expert_start: first expert ID owned by this rank.

    Returns:
        ``(sorted_ids, sorted_expert_ids, sorted_weights, num_valid)`` -- the
        same contract as :func:`moe_align_block_size_device`.
    """
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("top-k ids and weights must have the same shape")
    if expert_start < 0:
        raise ValueError("expert_start must be non-negative")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    device = topk_ids.device
    M, topk = topk_ids.shape
    G = M * topk
    if G <= 0:
        raise ValueError("expert-parallel fused align needs a non-empty route set")
    if M > block_m:
        raise ValueError(
            f"expert-parallel fused align needs M ({M}) <= block_m ({block_m})"
        )
    if block_m & (block_m - 1):
        raise ValueError("block_m must be a power of two")
    sentinel = M

    NB_MAX = min(G, num_experts)
    EM_MAX = NB_MAX * block_m
    GP = _next_pow2(G)
    EP = _next_pow2(num_experts)
    spt_g = max(1, GP // 256)

    ids = topk_ids.reshape(-1).to(torch.int32).contiguous()
    wts = topk_weights.reshape(-1).to(torch.float32).contiguous()
    sti = torch.empty(EM_MAX, dtype=torch.int32, device=device)
    sw = torch.empty(EM_MAX, dtype=torch.float32, device=device)
    sei = torch.empty(NB_MAX, dtype=torch.int32, device=device)
    nv = torch.empty(1, dtype=torch.int32, device=device)

    _fused_align_ep_kernel[(num_experts,)](
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
        BM=block_m,
        NB_MAX=NB_MAX,
        SPT_G=spt_g,
        num_warps=4,
    )
    return sti, sei, sw, nv
