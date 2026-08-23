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

"""Grouped gfx950 A16W4 SiTU MoE for contiguous expert parallelism.

The route list is aligned entirely on-device.  Both GEMMs consume packed MXFP4
weights directly; activations are staged through LDS while gfx950's native
scaled upcast converts only the active weight tile into the register dot
layout.  This avoids both the Python expert loop and model-sized BF16 weight
traffic.  Stage 1 fuses the BF16 boundary and SiTU activation.  Stage 2
scatters BF16 route outputs and a final masked FP32 reduction combines only
locally-owned EP slots.
"""

from __future__ import annotations

import os

import torch
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
    moe_align_block_size_device,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_fused import (
    moe_align_block_size_fused,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_fused_ep import (
    moe_align_block_size_fused_ep,
)

MXFP4_GROUP_SIZE = 32
_MXFP4_GROUP_SIZE_GL = gl.constexpr(32)
GROUPED_BLOCK_M = 64
GROUPED_FUSED_ALIGN_MAX_ROUTES = 128


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


# Opt-in (default OFF): route the decode-shaped align through the single-kernel
# expert-parallel implementation instead of the five-launch device path.
# ``moe_align_block_size_fused`` above is capped at 128 routes because it does
# O(M*topk*block_m) init and an O((M*topk)^2) rank tile inside ONE workgroup;
# the decode shape here is M=32, topk=16 -> 512 routes, so it falls back to the
# five-launch path (_init + torch.zeros memset + _count + single-CTA _offsets +
# _scatter).  The expert-parallel variant keeps the single-block collapse (which
# needs only M <= block_m) and parallelizes the rank over experts, so it holds
# at 512 routes.  Measured on gfx950 at M=32/topk=16/E=112/block_m=64:
# 12.5us -> 3.7us of GPU time per MoE layer.
_ALIGN_EP_SWITCH = "TSK_AMD_MOE_ALIGN_EXPERT_PARALLEL"
# Fallback guards: one workgroup per local expert, each holding a
# next_pow2(M*topk) route tile in registers.  Outside these bounds the original
# selection is used unchanged.
GROUPED_EP_ALIGN_MAX_ROUTES = 2048
GROUPED_EP_ALIGN_MAX_EXPERTS = 1024
# Atomic combine's cost hardly depends on the batch size at all.  The list of
# (token, expert) pairs is padded out to a full GROUPED_BLOCK_M-row block per
# expert, so with 112 experts it is ~7168 rows even for a single token.
# Every one of those rows issues its atomics, whether or not it holds real work
# given padding is masked off by pointing the address out of bounds, which does
# not skip the instruction.  A 16-token batch therefore issues 25.7M atomics to
# perform 0.92M useful ones.
#
# The constant stays instead of deleting the atomic path outright, so the
# comparison is easy to redo if the padding ever stops dominating.
GROUPED_ATOMIC_COMBINE_MAX_TOKENS = 0

# --- TSK_K3_MOE_REDUCE_DECODE_GEOMETRY (default OFF) -------------------------
# ``reduce_block_m=64`` / ``reduce_block_n=256`` below are prefill tuning.  At
# decode shapes (num_tokens <= 64, hidden_dim 3584) that is a 14-workgroup grid
# on a 256-CU MI355X, so ``_masked_topk_reduce_kernel`` is occupancy bound, not
# bandwidth bound.  Measured GPU time per launch (torch.profiler, gfx950,
# top_k=16, 112 local experts, hidden 3584):
#
#     num_tokens   (64,256) default   (16,128) tuned   speedup
#              1        15.53 us          5.68 us       2.73x
#              8        14.61 us          4.90 us       2.98x
#             16        14.92 us          5.38 us       2.77x
#             32        15.47 us          5.31 us       2.91x
#
# At 92 MoE layers that is ~0.94 ms off every decode step.  The kernel, its
# ``static_range`` over TOP_K and its fp32 accumulation order are untouched --
# only the tiling changes -- so the reduction is bit-identical.  Off by default;
# set TSK_K3_MOE_REDUCE_DECODE_GEOMETRY=1 to enable.
_REDUCE_TUNE_SMALL_M = (
    os.environ.get("TSK_K3_MOE_REDUCE_DECODE_GEOMETRY", "0") == "1"
)
# Fast-path applicability bound, deliberately a module constant rather than a
# second environment switch: it is the shape guard for the measurement above,
# not an independent lever.  64 covers every captured decode size on this
# deployment (--cudagraph-capture-sizes 1 2 4 8 16 32, --max-num-seqs 32) with
# margin, and the four rows of the table above are all at or below it.  Anything
# larger is prefill-shaped and falls through to the original (64, 256) tiling.
_REDUCE_TUNE_MAX_TOKENS = 64
_REDUCE_TUNE_BLOCK_M = 16
_REDUCE_TUNE_BLOCK_N = 128

# --- TSK_K3_MOE_GROUPED_SMALL_M_WARPS (default OFF) --------------------------
# ``s1_warps = {64: 4, 128: 8}[block_m]`` and ``s2_warps = 4`` below tie the
# warp count of both expert GEMMs to the row tile.  The BM64 arm of that table
# is the small-M regime -- every decode step and every prefill chunk below
# 3584 tokens -- and there four warps is the wrong choice: at BM64 the block
# is padded out to a full 64 rows per touched expert, so the kernel streams
# MXFP4 expert weights for ~60 experts to consume a handful of real routes and
# is weight-fetch bound, not row bound.  Eight warps issue twice as many
# outstanding global loads per CTA and cover more of that latency.
#
# Measured on this host (gfx950, HIP_VISIBLE_DEVICES=0, median of 30 reps
# after 5 warmups), full grouped path -- align + stage1 + stage2 + reduce --
# at the live EP8 shapes (latent 3584, intermediate 3072, 112 local experts,
# top_k 16), microseconds:
#
#     tokens   4 warps    8 warps   delta
#          1     211.0      198.4   -6.0%
#          8     267.9      258.6   -3.5%
#         16     341.8      320.1   -6.3%
#         32     584.2      566.6   -3.0%     <- --max-num-seqs 32 decode
#         64     676.7      653.1   -3.5%
#        512     964.1      938.1   -2.7%
#       1024     995.3      948.4   -4.7%
#       2048    1101.8     1043.3   -5.3%
#       3072    1264.0     1204.0   -4.7%
#
# Isolated per kernel at 32 tokens: stage 1 307.2 -> 295.5 us (-3.8%),
# stage 2 168.5 -> 163.8 us (-2.8%).  The output was compared element-wise
# against the four-warp result at every size above and is **bit-identical**
# (max abs err 0.0): num_warps changes only how the fixed BLOCK_M x BLOCK_N
# tile is partitioned across warps of the MFMA layout, not the tile shape, the
# K-loop trip count, or the FP32 accumulation order.
#
# The BM128 arm is deliberately untouched: at block_m=128 stage 1 already runs
# eight warps, and stage 2 at eight warps measured *slower* there (4096
# tokens: 434.6 -> 509.1 us), so the fast path guards on ``block_m == 64`` and
# every other shape falls through to the original table.  Off by default; set
# TSK_K3_MOE_GROUPED_SMALL_M_WARPS=1 to enable.
_SMALL_M_WARPS = os.environ.get("TSK_K3_MOE_GROUPED_SMALL_M_WARPS", "0") == "1"
_SMALL_M_WARPS_BLOCK_M = 64
_SMALL_M_WARPS_COUNT = 8


@gluon.jit
def _dequant_mxfp4_tile(
    packed_ptr,
    scale_ptr,
    packed_offs_n,
    packed_offs_k,
    expanded_offs_n,
    expanded_offs_k,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
):
    packed_offsets = (
        packed_offs_n[None, :].to(gl.int64) * stride_wn
        + packed_offs_k[:, None].to(gl.int64) * stride_wk
    )
    scale_offsets = (
        expanded_offs_n[None, :].to(gl.int64) * stride_sn
        + (expanded_offs_k[:, None] // _MXFP4_GROUP_SIZE_GL).to(gl.int64) * stride_sk
    )
    packed = gl.amd.cdna4.buffer_load(
        ptr=packed_ptr,
        offsets=packed_offsets.to(gl.int32),
    )
    scale = gl.amd.cdna4.buffer_load(
        ptr=scale_ptr,
        offsets=scale_offsets.to(gl.int32),
    )
    # gfx950 has a native packed FP4 -> BF16 conversion that applies the
    # UE8M0 block scale at the same time.  Keeping the packed and expanded
    # layouts related along K lets Gluon lower this to
    # v_cvt_scalef32_pk_bf16_fp4 instead of dozens of integer/FP operations.
    return gl.amd.cdna4.scaled_upcast(
        packed,
        scale,
        gl.bfloat16,
        axis=0,
    )


@gluon.jit
def _load_w13_gate_up_dot(
    w_ptr,
    scale_ptr,
    smem_bg: gl.shared_memory_descriptor,
    smem_bu: gl.shared_memory_descriptor,
    expert,
    k_base,
    packed_offs_bn_gate,
    packed_offs_bn_up,
    packed_offs_bk,
    offs_bn_gate,
    offs_bn_up,
    offs_bk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_sk,
    DOT_B_LAYOUT: gl.constexpr,
):
    """Scaled-upcast one W13 tile and return its gate/up MFMA operands."""

    w_base = w_ptr + expert.to(gl.int64) * stride_we + (k_base // 2) * stride_wk
    scale_base = (
        scale_ptr
        + expert.to(gl.int64) * stride_se
        + (k_base // _MXFP4_GROUP_SIZE_GL) * stride_sk
    )
    bg = _dequant_mxfp4_tile(
        w_base,
        scale_base,
        packed_offs_bn_gate,
        packed_offs_bk,
        offs_bn_gate,
        offs_bk,
        stride_wn,
        stride_wk,
        stride_sn,
        stride_sk,
    )
    bu = _dequant_mxfp4_tile(
        w_base,
        scale_base,
        packed_offs_bn_up,
        packed_offs_bk,
        offs_bn_up,
        offs_bk,
        stride_wn,
        stride_wk,
        stride_sn,
        stride_sk,
    )
    smem_bg.store(bg)
    smem_bu.store(bu)
    return smem_bg.load(DOT_B_LAYOUT), smem_bu.load(DOT_B_LAYOUT)


@gluon.jit
def _grouped_a16w4_situ_stage1_kernel(
    a_ptr,
    w_ptr,
    scale_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    K,
    EM,
    num_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_sk,
    stride_cm,
    stride_cn,
    I_R: gl.constexpr,
    SITU_BETA: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    num_pid_n = gl.cdiv(I_R, BLOCK_N)
    pid = gl.program_id(0)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_valid = gl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_M >= num_valid:
        return
    expert = gl.load(sorted_expert_ids_ptr + pid_m)
    if expert < 0:
        return

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, NUM_WARPS // 2],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )
    gload_a: gl.constexpr = gl.BlockedLayout(
        [1, 8], [512 // BLOCK_K, BLOCK_K // 8], [NUM_WARPS, 1], [1, 0]
    )
    gload_b: gl.constexpr = gl.BlockedLayout(
        [8, 1], [BLOCK_K // 8, 512 // BLOCK_K], [1, NUM_WARPS], [0, 1]
    )
    gload_b_packed: gl.constexpr = gl.BlockedLayout(
        [4, 1], [BLOCK_K // 8, 512 // BLOCK_K], [1, NUM_WARPS], [0, 1]
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])
    smem_a = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_M, BLOCK_K], shared_a)
    smem_bg = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_K, BLOCK_N], shared_b)
    smem_bu = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_K, BLOCK_N], shared_b)

    am_layout: gl.constexpr = gl.SliceLayout(1, gload_a)
    ak_layout: gl.constexpr = gl.SliceLayout(0, gload_a)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=am_layout)
    packed_route = gl.load(sorted_token_ids_ptr + offs_m)
    token = packed_route & 0xFFFFFF
    # Every expert is independently padded to BLOCK_M. Even when this block is
    # below the aggregate num_valid boundary, its tail can contain the
    # num_tokens sentinel and must not read one row beyond A.
    token_mask = token < num_tokens
    offs_ak = gl.arange(0, BLOCK_K, layout=ak_layout)
    a_offsets = (token[:, None] * stride_am + offs_ak[None, :] * stride_ak).to(gl.int32)

    bk_layout: gl.constexpr = gl.SliceLayout(1, gload_b)
    bn_layout: gl.constexpr = gl.SliceLayout(0, gload_b)
    offs_bk = gl.arange(0, BLOCK_K, layout=bk_layout)
    offs_bn_gate = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=bn_layout)
    offs_bn_up = I_R + offs_bn_gate
    packed_bk_layout: gl.constexpr = gl.SliceLayout(1, gload_b_packed)
    packed_bn_layout: gl.constexpr = gl.SliceLayout(0, gload_b_packed)
    packed_offs_bk = gl.arange(0, BLOCK_K // 2, layout=packed_bk_layout)
    packed_offs_bn_gate = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=packed_bn_layout
    )
    packed_offs_bn_up = I_R + packed_offs_bn_gate

    gate_acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)
    up_acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)
    num_k = gl.cdiv(K, BLOCK_K)
    gl.assume(num_k > 3)

    # Keep two gathered-A tiles in flight. W13 is scaled-upcast into LDS near
    # use, and two K tiles are consumed per loop iteration to expose the local
    # prefetch/MFMA schedule that wins for sparse EP routing on gfx950.
    for prologue_tile in gl.static_range(0, 2):
        cdna4_async_copy.buffer_load_to_shared(
            smem_a.index(prologue_tile),
            a_ptr + prologue_tile * BLOCK_K * stride_ak,
            a_offsets,
            mask=token_mask[:, None],
        )
        cdna4_async_copy.commit_group()
    cdna4_async_copy.wait_group(1)
    a = smem_a.index(0).load(dot_a_layout)

    for k_tile in range(0, num_k - 2, 2):
        k_base = k_tile * BLOCK_K
        bg_dot, bu_dot = _load_w13_gate_up_dot(
            w_ptr,
            scale_ptr,
            smem_bg,
            smem_bu,
            expert,
            k_base,
            packed_offs_bn_gate,
            packed_offs_bn_up,
            packed_offs_bk,
            offs_bn_gate,
            offs_bn_up,
            offs_bk,
            stride_we,
            stride_wn,
            stride_wk,
            stride_se,
            stride_sn,
            stride_sk,
            DOT_B_LAYOUT=dot_b_layout,
        )
        gate_acc = gl.amd.cdna4.mfma(a, bg_dot, gate_acc)
        up_acc = gl.amd.cdna4.mfma(a, bu_dot, up_acc)
        cdna4_async_copy.wait_group(0)
        refill_k = (k_tile + 2) * BLOCK_K
        cdna4_async_copy.buffer_load_to_shared(
            smem_a.index(0),
            a_ptr + refill_k * stride_ak,
            a_offsets,
            mask=token_mask[:, None],
        )
        cdna4_async_copy.commit_group()
        a_next = smem_a.index(1).load(dot_a_layout)

        k_base = (k_tile + 1) * BLOCK_K
        bg_dot, bu_dot = _load_w13_gate_up_dot(
            w_ptr,
            scale_ptr,
            smem_bg,
            smem_bu,
            expert,
            k_base,
            packed_offs_bn_gate,
            packed_offs_bn_up,
            packed_offs_bk,
            offs_bn_gate,
            offs_bn_up,
            offs_bk,
            stride_we,
            stride_wn,
            stride_wk,
            stride_se,
            stride_sn,
            stride_sk,
            DOT_B_LAYOUT=dot_b_layout,
        )
        gate_acc = gl.amd.cdna4.mfma(a_next, bg_dot, gate_acc)
        up_acc = gl.amd.cdna4.mfma(a_next, bu_dot, up_acc)
        cdna4_async_copy.wait_group(0)
        refill_k = (k_tile + 3) * BLOCK_K
        cdna4_async_copy.buffer_load_to_shared(
            smem_a.index(1),
            a_ptr + refill_k * stride_ak,
            a_offsets,
            mask=token_mask[:, None],
        )
        cdna4_async_copy.commit_group()
        a = smem_a.index(0).load(dot_a_layout)

    k_base = (num_k - 2) * BLOCK_K
    bg_dot, bu_dot = _load_w13_gate_up_dot(
        w_ptr,
        scale_ptr,
        smem_bg,
        smem_bu,
        expert,
        k_base,
        packed_offs_bn_gate,
        packed_offs_bn_up,
        packed_offs_bk,
        offs_bn_gate,
        offs_bn_up,
        offs_bk,
        stride_we,
        stride_wn,
        stride_wk,
        stride_se,
        stride_sn,
        stride_sk,
        DOT_B_LAYOUT=dot_b_layout,
    )
    gate_acc = gl.amd.cdna4.mfma(a, bg_dot, gate_acc)
    up_acc = gl.amd.cdna4.mfma(a, bu_dot, up_acc)
    cdna4_async_copy.wait_group(0)
    a_next = smem_a.index(1).load(dot_a_layout)
    k_base = (num_k - 1) * BLOCK_K
    bg_dot, bu_dot = _load_w13_gate_up_dot(
        w_ptr,
        scale_ptr,
        smem_bg,
        smem_bu,
        expert,
        k_base,
        packed_offs_bn_gate,
        packed_offs_bn_up,
        packed_offs_bk,
        offs_bn_gate,
        offs_bn_up,
        offs_bk,
        stride_we,
        stride_wn,
        stride_wk,
        stride_se,
        stride_sn,
        stride_sk,
        DOT_B_LAYOUT=dot_b_layout,
    )
    gate_acc = gl.amd.cdna4.mfma(a_next, bg_dot, gate_acc)
    up_acc = gl.amd.cdna4.mfma(a_next, bu_dot, up_acc)

    # Preserve Kimi's BF16 tensor boundary before the FP32 SiTU math.
    gate = gate_acc.to(gl.bfloat16).to(gl.float32)
    up = up_acc.to(gl.bfloat16).to(gl.float32)
    gate = (
        SITU_BETA
        * gl.extra.libdevice.tanh(gate / SITU_BETA)
        * (1.0 / (1.0 + gl.exp(-gate)))
    )
    if HAS_LINEAR_BETA:
        up = SITU_LINEAR_BETA * gl.extra.libdevice.tanh(up / SITU_LINEAR_BETA)
    inter = (gate * up).to(c_ptr.dtype.element_ty)

    cm_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    cn_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=cm_layout)
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout)
    packed_c = gl.load(
        sorted_token_ids_ptr + offs_cm,
        mask=offs_cm < EM,
        other=num_tokens,
    )
    token_c = packed_c & 0xFFFFFF
    slot_c = packed_c >> 24
    dst_row = token_c * top_k + slot_c
    c_offsets = (
        dst_row[:, None].to(gl.int64) * stride_cm
        + offs_cn[None, :].to(gl.int64) * stride_cn
    )
    c_mask = (token_c[:, None] < num_tokens) & (offs_cn[None, :] < I_R)
    gl.store(c_ptr + c_offsets, inter, mask=c_mask)


@gluon.jit
def _grouped_a16w4_stage2_kernel(
    a_ptr,
    w_ptr,
    scale_ptr,
    partials_ptr,
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    sorted_weights_ptr,
    num_valid_ids_ptr,
    N,
    K,
    EM,
    num_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_sk,
    stride_pt,
    stride_ps,
    stride_pn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    FUSE_COMBINE: gl.constexpr,
):
    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    num_valid = gl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_M >= num_valid:
        return
    expert = gl.load(sorted_expert_ids_ptr + pid_m)
    if expert < 0:
        return

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, NUM_WARPS // 2],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )
    gload_a: gl.constexpr = gl.BlockedLayout(
        [1, 8], [512 // BLOCK_K, BLOCK_K // 8], [NUM_WARPS, 1], [1, 0]
    )
    gload_b: gl.constexpr = gl.BlockedLayout(
        [8, 1], [BLOCK_K // 8, 512 // BLOCK_K], [1, NUM_WARPS], [0, 1]
    )
    gload_b_packed: gl.constexpr = gl.BlockedLayout(
        [4, 1], [BLOCK_K // 8, 512 // BLOCK_K], [1, NUM_WARPS], [0, 1]
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])
    smem_a = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_M, BLOCK_K], shared_a)
    smem_b = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_K, BLOCK_N], shared_b)

    am_layout: gl.constexpr = gl.SliceLayout(1, gload_a)
    ak_layout: gl.constexpr = gl.SliceLayout(0, gload_a)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=am_layout)
    packed_route = gl.load(
        sorted_token_ids_ptr + offs_m,
        mask=offs_m < EM,
        other=num_tokens,
    )
    token = packed_route & 0xFFFFFF
    slot = packed_route >> 24
    inter_row = token * top_k + slot
    token_mask = token < num_tokens
    offs_ak = gl.arange(0, BLOCK_K, layout=ak_layout)
    a_offsets = (inter_row[:, None] * stride_am + offs_ak[None, :] * stride_ak).to(
        gl.int32
    )

    bk_layout: gl.constexpr = gl.SliceLayout(1, gload_b)
    bn_layout: gl.constexpr = gl.SliceLayout(0, gload_b)
    offs_bk = gl.arange(0, BLOCK_K, layout=bk_layout)
    offs_bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=bn_layout)
    packed_bk_layout: gl.constexpr = gl.SliceLayout(1, gload_b_packed)
    packed_bn_layout: gl.constexpr = gl.SliceLayout(0, gload_b_packed)
    packed_offs_bk = gl.arange(0, BLOCK_K // 2, layout=packed_bk_layout)
    packed_offs_bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=packed_bn_layout)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)
    num_k = gl.cdiv(K, BLOCK_K)
    cdna4_async_copy.buffer_load_to_shared(
        smem_a.index(0),
        a_ptr,
        a_offsets,
        mask=token_mask[:, None],
    )
    cdna4_async_copy.commit_group()

    for k_tile in range(0, num_k - 1):
        l_idx = k_tile % 2
        g_idx = 1 - l_idx
        k_base = k_tile * BLOCK_K
        next_k_base = (k_tile + 1) * BLOCK_K
        cdna4_async_copy.buffer_load_to_shared(
            smem_a.index(g_idx),
            a_ptr + next_k_base * stride_ak,
            a_offsets,
            mask=token_mask[:, None],
        )
        cdna4_async_copy.commit_group()
        w_base = w_ptr + expert.to(gl.int64) * stride_we + (k_base // 2) * stride_wk
        scale_base = (
            scale_ptr
            + expert.to(gl.int64) * stride_se
            + (k_base // _MXFP4_GROUP_SIZE_GL) * stride_sk
        )
        b = _dequant_mxfp4_tile(
            w_base,
            scale_base,
            packed_offs_bn,
            packed_offs_bk,
            offs_bn,
            offs_bk,
            stride_wn,
            stride_wk,
            stride_sn,
            stride_sk,
        )
        smem_b.store(b)
        b_dot = smem_b.load(dot_b_layout)
        cdna4_async_copy.wait_group(1)
        a = smem_a.index(l_idx).load(dot_a_layout)
        acc = gl.amd.cdna4.mfma(a, b_dot, acc)

    cdna4_async_copy.wait_group(0)
    l_idx = (num_k - 1) % 2
    k_base = (num_k - 1) * BLOCK_K
    w_base = w_ptr + expert.to(gl.int64) * stride_we + (k_base // 2) * stride_wk
    scale_base = (
        scale_ptr
        + expert.to(gl.int64) * stride_se
        + (k_base // _MXFP4_GROUP_SIZE_GL) * stride_sk
    )
    b = _dequant_mxfp4_tile(
        w_base,
        scale_base,
        packed_offs_bn,
        packed_offs_bk,
        offs_bn,
        offs_bk,
        stride_wn,
        stride_wk,
        stride_sn,
        stride_sk,
    )
    a = smem_a.index(l_idx).load(dot_a_layout)
    smem_b.store(b)
    b_dot = smem_b.load(dot_b_layout)
    acc = gl.amd.cdna4.mfma(a, b_dot, acc)

    cm_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    cn_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=cm_layout)
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout)
    packed_c = gl.load(
        sorted_token_ids_ptr + offs_cm,
        mask=offs_cm < EM,
        other=num_tokens,
    )
    token_c = packed_c & 0xFFFFFF
    slot_c = packed_c >> 24
    c_mask = (token_c[:, None] < num_tokens) & (offs_cn[None, :] < N)
    # Match the reference's BF16 W2 output before route weighting/reduction.
    w2_bf16 = acc.to(gl.bfloat16)
    if FUSE_COMBINE:
        route_weight = gl.load(
            sorted_weights_ptr + offs_cm,
            mask=offs_cm < EM,
            other=0.0,
        )
        value = w2_bf16.to(gl.float32) * route_weight[:, None].to(gl.float32)
        c_offsets = (token_c[:, None] * stride_pt + offs_cn[None, :] * stride_pn).to(
            gl.int32
        )
        gl.amd.cdna4.buffer_atomic_add(
            partials_ptr,
            c_offsets,
            value,
            mask=c_mask,
        )
    else:
        c_offsets = (
            token_c[:, None].to(gl.int64) * stride_pt
            + slot_c[:, None].to(gl.int64) * stride_ps
            + offs_cn[None, :].to(gl.int64) * stride_pn
        )
        gl.store(partials_ptr + c_offsets, w2_bf16, mask=c_mask)


@gluon.jit
def _masked_topk_reduce_kernel(
    partials_ptr,
    local_ids_ptr,
    topk_weights_ptr,
    out_ptr,
    num_tokens,
    N,
    stride_pt,
    stride_ps,
    stride_pn,
    stride_im,
    stride_ik,
    stride_wm,
    stride_wk,
    stride_ot,
    stride_on,
    TOP_K: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    EXPERT_START: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 4], [1, 0])
    rm_layout: gl.constexpr = gl.SliceLayout(1, layout)
    cn_layout: gl.constexpr = gl.SliceLayout(0, layout)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=rm_layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout)
    mn_mask = (offs_m[:, None] < num_tokens) & (offs_n[None, :] < N)
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, layout)
    base = (
        partials_ptr
        + offs_m[:, None].to(gl.int64) * stride_pt
        + offs_n[None, :].to(gl.int64) * stride_pn
    )
    for slot in gl.static_range(0, TOP_K):
        global_id = gl.load(
            local_ids_ptr + offs_m * stride_im + slot * stride_ik,
            mask=offs_m < num_tokens,
            other=EXPERT_START - 1,
        )
        local_id = global_id - EXPERT_START
        weight = gl.load(
            topk_weights_ptr + offs_m * stride_wm + slot * stride_wk,
            mask=offs_m < num_tokens,
            other=0.0,
        )
        valid = mn_mask & (local_id[:, None] >= 0) & (local_id[:, None] < NUM_EXPERTS)
        value = gl.load(
            base + slot * stride_ps,
            mask=valid,
            other=0.0,
        )
        acc += value.to(gl.float32) * weight[:, None].to(gl.float32)
    out_offsets = (
        offs_m[:, None].to(gl.int64) * stride_ot
        + offs_n[None, :].to(gl.int64) * stride_on
    )
    gl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty), mask=mn_mask)


def gluon_a16w4_situ_grouped_ep_gfx950(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    local_topk_ids: torch.Tensor,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
    block_m: int | None = None,
    expert_start: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one rank's grouped A16W4 SiTU contribution without host sync.

    ``local_topk_ids`` may contain global expert IDs. ``expert_start`` marks
    the first expert represented by the local weight tensors; non-local routes
    are discarded by alignment and reduction kernels.
    """
    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        raise ValueError("grouped gfx950 A16W4 requires rank-2 BF16 activations")
    if topk_weights.shape != local_topk_ids.shape or local_topk_ids.ndim != 2:
        raise ValueError("top-k weights and local ids must have the same rank-2 shape")
    if local_topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("top-k token count must match hidden states")
    if not (w13_weight.ndim == w13_scale.ndim == w2_weight.ndim == w2_scale.ndim == 3):
        raise ValueError("local MXFP4 expert tensors must be rank-3")
    if situ_beta <= 0.0:
        raise ValueError("SiTU beta must be positive")
    if situ_linear_beta is not None and situ_linear_beta <= 0.0:
        raise ValueError("SiTU linear beta must be positive")
    if expert_start < 0:
        raise ValueError("expert_start must be non-negative")

    num_tokens, hidden_dim = hidden_states.shape
    if out is not None and (
        out.shape != hidden_states.shape
        or out.dtype != hidden_states.dtype
        or out.device != hidden_states.device
        or not out.is_contiguous()
    ):
        raise ValueError("output must match the hidden-state shape, dtype, and device")
    if block_m is None:
        # Sparse EP padding dominates until each rank owns roughly 7k routes.
        # Keep BM64 below M=3584; BM128 then amortizes launch/grid overhead.
        block_m = 128 if num_tokens >= 3584 else GROUPED_BLOCK_M
    num_experts, two_intermediate, packed_hidden = w13_weight.shape
    intermediate = two_intermediate // 2
    top_k = int(local_topk_ids.shape[1])
    if two_intermediate % 2 or packed_hidden * 2 != hidden_dim:
        raise ValueError("W13 shape is inconsistent with the activation width")
    if tuple(w13_scale.shape) != (
        num_experts,
        two_intermediate,
        hidden_dim // MXFP4_GROUP_SIZE,
    ):
        raise ValueError("W13 scale shape mismatch")
    if tuple(w2_weight.shape) != (num_experts, hidden_dim, intermediate // 2):
        raise ValueError("W2 shape mismatch")
    if tuple(w2_scale.shape) != (
        num_experts,
        hidden_dim,
        intermediate // MXFP4_GROUP_SIZE,
    ):
        raise ValueError("W2 scale shape mismatch")
    if hidden_dim % 256 or intermediate % 128 or block_m not in (64, 128):
        raise ValueError(
            "grouped gfx950 A16W4 requires hidden_dim divisible by 256, "
            "intermediate divisible by 128, and block_m in {64, 128}"
        )

    num_routes = num_tokens * top_k
    if 0 < num_routes <= GROUPED_FUSED_ALIGN_MAX_ROUTES and num_tokens <= block_m:
        align = moe_align_block_size_fused
    elif (
        _env_flag(_ALIGN_EP_SWITCH)
        and 0 < num_tokens <= block_m
        and num_routes <= GROUPED_EP_ALIGN_MAX_ROUTES
        and num_experts <= GROUPED_EP_ALIGN_MAX_EXPERTS
    ):
        align = moe_align_block_size_fused_ep
    else:
        align = moe_align_block_size_device
    sorted_ids, sorted_experts, sorted_weights, num_valid = align(
        local_topk_ids,
        topk_weights,
        num_experts,
        block_m,
        expert_start=expert_start,
    )
    em = int(sorted_ids.shape[0])
    inter = torch.empty(
        (num_tokens * top_k, intermediate),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    s1_block_n = 64
    s1_block_k = 64
    s1_warps = {64: 4, 128: 8}[block_m]
    if _SMALL_M_WARPS and block_m == _SMALL_M_WARPS_BLOCK_M:
        # Small-M arm: weight-fetch bound, so widen the CTA rather than the
        # tile. Same BLOCK_M/BLOCK_N/BLOCK_K and the same K-loop, so the
        # result is bit-identical to the four-warp schedule.
        s1_warps = _SMALL_M_WARPS_COUNT
    s1_grid = triton.cdiv(em, block_m) * triton.cdiv(intermediate, s1_block_n)
    _grouped_a16w4_situ_stage1_kernel[(s1_grid,)](
        hidden_states,
        w13_weight,
        w13_scale,
        inter,
        sorted_ids,
        sorted_experts,
        num_valid,
        hidden_dim,
        em,
        num_tokens,
        top_k,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w13_weight.stride(0),
        w13_weight.stride(1),
        w13_weight.stride(2),
        w13_scale.stride(0),
        w13_scale.stride(1),
        w13_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        I_R=intermediate,
        SITU_BETA=float(situ_beta),
        SITU_LINEAR_BETA=(1.0 if situ_linear_beta is None else float(situ_linear_beta)),
        HAS_LINEAR_BETA=situ_linear_beta is not None,
        BLOCK_M=block_m,
        BLOCK_N=s1_block_n,
        BLOCK_K=s1_block_k,
        NUM_WARPS=s1_warps,
        num_warps=s1_warps,
    )

    fuse_combine = num_tokens <= GROUPED_ATOMIC_COMBINE_MAX_TOKENS
    if fuse_combine:
        stage2_out = torch.zeros(
            (num_tokens, hidden_dim),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        stage2_strides = (stage2_out.stride(0), 0, stage2_out.stride(1))
    else:
        stage2_out = torch.empty(
            (num_tokens, top_k, hidden_dim),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        stage2_strides = stage2_out.stride()
    s2_block_n = 128
    s2_block_k = 64
    s2_warps = 4
    if _SMALL_M_WARPS and block_m == _SMALL_M_WARPS_BLOCK_M:
        # Same reasoning as stage 1. Guarded on BM64: at block_m=128 eight
        # warps measured slower here (4096 tokens, 434.6 -> 509.1 us), so the
        # large-M shape keeps the original four.
        s2_warps = _SMALL_M_WARPS_COUNT
    s2_grid =triton.cdiv(em, block_m) * triton.cdiv(hidden_dim, s2_block_n)
    _grouped_a16w4_stage2_kernel[(s2_grid,)](
        inter,
        w2_weight,
        w2_scale,
        stage2_out,
        sorted_ids,
        sorted_experts,
        sorted_weights,
        num_valid,
        hidden_dim,
        intermediate,
        em,
        num_tokens,
        top_k,
        inter.stride(0),
        inter.stride(1),
        w2_weight.stride(0),
        w2_weight.stride(1),
        w2_weight.stride(2),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        stage2_strides[0],
        stage2_strides[1],
        stage2_strides[2],
        BLOCK_M=block_m,
        BLOCK_N=s2_block_n,
        BLOCK_K=s2_block_k,
        NUM_WARPS=s2_warps,
        FUSE_COMBINE=fuse_combine,
        num_warps=s2_warps,
    )
    if fuse_combine:
        if out is None:
            return stage2_out.to(torch.bfloat16)
        out.copy_(stage2_out)
        return out

    if out is None:
        out = torch.empty(
            (num_tokens, hidden_dim),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
    reduce_block_m = 64
    reduce_block_n = 256
    if (
        _REDUCE_TUNE_SMALL_M
        and num_tokens <= _REDUCE_TUNE_MAX_TOKENS
        and hidden_dim % _REDUCE_TUNE_BLOCK_N == 0
    ):
        # Decode-shaped reduction. The (64, 256) tiling above is prefill
        # tuning: at num_tokens <= 64 it yields a 14-workgroup grid on a
        # 256-CU part, so the kernel is occupancy bound rather than
        # bandwidth bound. Same kernel, same fp32 accumulation order over
        # TOP_K -- only the tiling changes, so the result is bit-identical.
        reduce_block_m = _REDUCE_TUNE_BLOCK_M
        reduce_block_n = _REDUCE_TUNE_BLOCK_N
    reduce_grid = triton.cdiv(num_tokens, reduce_block_m) * triton.cdiv(
        hidden_dim, reduce_block_n
    )
    _masked_topk_reduce_kernel[(reduce_grid,)](
        stage2_out,
        local_topk_ids,
        topk_weights,
        out,
        num_tokens,
        hidden_dim,
        stage2_out.stride(0),
        stage2_out.stride(1),
        stage2_out.stride(2),
        local_topk_ids.stride(0),
        local_topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        out.stride(0),
        out.stride(1),
        TOP_K=top_k,
        NUM_EXPERTS=num_experts,
        EXPERT_START=expert_start,
        BLOCK_M=reduce_block_m,
        BLOCK_N=reduce_block_n,
        num_warps=4,
    )
    return out


__all__ = ["gluon_a16w4_situ_grouped_ep_gfx950"]
