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

"""GFX1250 Gluon KDA decode kernels using V-major recurrent state."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

gfx1250 = gl.amd.gfx1250

_GFX1250_NUM_CUS = 256


@gluon.jit
def _kda_recurrent_decode_kernel(
    q,
    k,
    v,
    raw_g,
    raw_beta,
    state_pool,
    read_indices,
    write_indices,
    output,
    cu_seqlens,
    a_log,
    dt_bias,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    Q_TOKEN_STRIDE: gl.constexpr,
    K_TOKEN_STRIDE: gl.constexpr,
    V_TOKEN_STRIDE: gl.constexpr,
    G_TOKEN_STRIDE: gl.constexpr,
    BETA_TOKEN_STRIDE: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    STATE_PAGE_STRIDE: gl.constexpr,
    HAS_LOWER_BOUND: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
):
    """One-token KDA decode with direct V-major indexed state-pool IO."""
    value_block = gl.program_id(0)
    sequence_head = gl.program_id(1)
    sequence_idx = sequence_head // H
    head_idx = sequence_head % H

    if BK == 128 and BV == 32:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [4, 8],
            [4, 8],
            [2, 2],
            [1, 0],
        )
    elif BK == 128 and BV == 8:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [1, 8],
            [4, 8],
            [2, 1],
            [1, 0],
        )
    else:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1],
            [32, 1],
            [gl.num_warps(), 1],
            [1, 0],
        )
    value_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    key_layout: gl.constexpr = gl.SliceLayout(0, state_layout)

    begin = gl.load(cu_seqlens + sequence_idx)
    end = gl.load(cu_seqlens + sequence_idx + 1)
    value_offsets = value_block * BV + gl.arange(0, BV, layout=value_layout)
    value_mask = value_offsets < V
    if begin == end:
        gl.store(
            output + (sequence_idx * H + head_idx) * V + value_offsets,
            0.0,
            mask=value_mask,
        )
        return

    read_idx = gl.load(read_indices + sequence_idx)
    write_idx = gl.load(write_indices + sequence_idx)
    valid_read = (read_idx >= 0) & (read_idx < NUM_SLOTS)
    if not valid_read:
        gl.store(
            output + (sequence_idx * H + head_idx) * V + value_offsets,
            0.0,
            mask=value_mask,
        )
        return

    key_offsets = gl.arange(0, BK, layout=key_layout)
    key_mask = key_offsets < K
    read_base = read_idx * STATE_PAGE_STRIDE + head_idx * K * V

    token_idx = begin
    q_value = gl.load(
        q + token_idx * Q_TOKEN_STRIDE + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    k_value = gl.load(
        k + token_idx * K_TOKEN_STRIDE + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    gate_value = gl.load(
        raw_g + token_idx * G_TOKEN_STRIDE + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    gate_value += gl.load(
        dt_bias + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    a_value = gl.exp(gl.load(a_log + head_idx).to(gl.float32))
    if HAS_LOWER_BOUND:
        log_decay = LOWER_BOUND / (1.0 + gl.exp(-(a_value * gate_value)))
    else:
        softplus = gl.maximum(gate_value, 0.0) + gl.log(
            1.0 + gl.exp(-gl.abs(gate_value))
        )
        log_decay = -a_value * softplus
    beta_value = gl.load(raw_beta + token_idx * BETA_TOKEN_STRIDE + head_idx).to(
        gl.float32
    )
    beta_value = 1.0 / (1.0 + gl.exp(-beta_value))

    scale: gl.constexpr = K**-0.5
    q_value *= gl.rsqrt(gl.sum(q_value * q_value, axis=0) + 1e-6) * scale
    k_value *= gl.rsqrt(gl.sum(k_value * k_value, axis=0) + 1e-6)
    valid_write = (write_idx >= 0) & (write_idx < NUM_SLOTS)
    safe_write_idx = gl.where(valid_write, write_idx, 0)
    write_base = safe_write_idx * STATE_PAGE_STRIDE + head_idx * K * V
    decay = gl.exp(log_decay)
    state_mask = value_mask[:, None] & key_mask[None, :]
    read_offsets = value_offsets[:, None] * K + key_offsets[None, :]
    running = gfx1250.buffer_load(
        state_pool + read_base,
        read_offsets.to(gl.int32),
        mask=state_mask,
        other=0.0,
    ).to(gl.float32)
    v_value = gl.load(
        v + token_idx * V_TOKEN_STRIDE + head_idx * V + value_offsets,
        mask=value_mask,
        other=0.0,
    ).to(gl.float32)
    running *= decay[None, :]
    prediction = gl.sum(running * k_value[None, :], axis=1)
    delta = beta_value * (v_value - prediction)
    running += delta[:, None] * k_value[None, :]
    out_value = gl.sum(running * q_value[None, :], axis=1)
    gl.store(
        output + (sequence_idx * H + head_idx) * V + value_offsets,
        out_value.to(output.dtype.element_ty),
        mask=value_mask,
    )
    write_offsets = value_offsets[:, None] * K + key_offsets[None, :]
    gfx1250.buffer_store(
        running,
        state_pool + write_base,
        write_offsets.to(gl.int32),
        mask=valid_write & state_mask,
    )


@gluon.jit
def _kda_fused_decode_kernel(
    mixed_qkv,
    conv_weights,
    conv_pool,
    conv_scratch,
    raw_g,
    beta_logits,
    output_gate,
    norm_weight,
    state_pool,
    state_scratch,
    read_indices,
    write_indices,
    output,
    cu_seqlens,
    a_log,
    dt_bias,
    H: gl.constexpr,
    D: gl.constexpr,
    TOKENS_PER_SEQUENCE: gl.constexpr,
    VERIFY_MODE: gl.constexpr,
    MIXED_ROW_STRIDE: gl.constexpr,
    CONV_WEIGHT_ROW_STRIDE: gl.constexpr,
    CONV_WEIGHT_COL_STRIDE: gl.constexpr,
    CONV_POOL_PAGE_STRIDE: gl.constexpr,
    CONV_POOL_CHANNEL_STRIDE: gl.constexpr,
    CONV_POOL_HISTORY_STRIDE: gl.constexpr,
    CONV_SCRATCH_PAGE_STRIDE: gl.constexpr,
    CONV_SCRATCH_CHANNEL_STRIDE: gl.constexpr,
    CONV_SCRATCH_HISTORY_STRIDE: gl.constexpr,
    GATE_ROW_STRIDE: gl.constexpr,
    BETA_ROW_STRIDE: gl.constexpr,
    OUTPUT_GATE_ROW_STRIDE: gl.constexpr,
    STATE_POOL_PAGE_STRIDE: gl.constexpr,
    STATE_SCRATCH_PAGE_STRIDE: gl.constexpr,
    NUM_POOL_SLOTS: gl.constexpr,
    NUM_SCRATCH_SLOTS: gl.constexpr,
    HAS_LOWER_BOUND: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
    NORM_EPS: gl.constexpr,
    PIPELINE_DEPTH: gl.constexpr,
):
    """Fuse the K3 decode/verify convolution, recurrence, and gated RMSNorm."""
    head_idx = gl.program_id(0)
    sequence_idx = gl.program_id(1)

    # Physical state tiles are [V, K]. Giving each wave32 warp two value rows
    # and the whole key axis keeps the per-panel reductions over keys inside a
    # wave, which is what makes the panel loop cheap here.
    state_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [2, 16],
        [8, 1],
        [1, 0],
    )
    key_layout: gl.constexpr = gl.SliceLayout(0, state_layout)
    value_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    key_offsets = gl.arange(0, D, layout=key_layout)
    compact_layout: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
    output_offsets = gl.arange(0, D, layout=compact_layout)
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    # Pack Q, K, V, and decay into each row; keep output in D-wide rows.
    shared_vectors = gl.allocate_shared_memory(
        gl.float32, [TOKENS_PER_SEQUENCE, 4 * D], shared_layout
    )
    output_alloc = gl.allocate_shared_memory(
        gl.float32, [TOKENS_PER_SEQUENCE, D], shared_layout
    )

    # Verify rows are dense [B, T]; decode uses cu_seqlens so graph-padding
    # rows can be empty.
    output_base = (sequence_idx * TOKENS_PER_SEQUENCE * H + head_idx) * D
    if VERIFY_MODE:
        token_base = sequence_idx * TOKENS_PER_SEQUENCE
    else:
        begin = gl.load(cu_seqlens + sequence_idx)
        end = gl.load(cu_seqlens + sequence_idx + 1)
        if begin == end:
            gl.store(output + output_base + output_offsets, 0.0)
            return
        token_base = begin

    read_idx = gl.load(read_indices + sequence_idx)
    valid_read = (read_idx >= 0) & (read_idx < NUM_POOL_SLOTS)
    if not valid_read:
        for token_offset in gl.static_range(TOKENS_PER_SEQUENCE):
            gl.store(
                output + output_base + token_offset * H * D + output_offsets,
                0.0,
            )
        return

    read_page_offset = read_idx.to(gl.int64)
    read_base = read_page_offset * STATE_POOL_PAGE_STRIDE + head_idx * D * D
    panel_value_offsets = gl.arange(0, 16, layout=value_layout)

    # Keep PIPELINE_DEPTH state panels in flight to balance latency and VGPR use.
    # static_range permits the unrolled tuple window to change length.
    off_pipe = ()
    raw_pipe = ()
    for _pd in gl.static_range(PIPELINE_DEPTH):
        _off_pd = (panel_value_offsets[:, None] + _pd * 16) * D + key_offsets[None, :]
        _raw_pd = gl.load(state_pool + read_base + _off_pd).to(gl.float32)
        off_pipe = off_pipe + (_off_pd,)
        raw_pipe = raw_pipe + (_raw_pd,)

    # Process Q/K/V/decay together so Q/K/V share convolution loads.
    qkvd_layout: gl.constexpr = gl.BlockedLayout([2], [32], [8], [0])
    qkvd_offsets = gl.arange(0, 4 * D, layout=qkvd_layout)
    slot_id = qkvd_offsets // D
    local_offset = qkvd_offsets - slot_id * D
    is_k = slot_id == 1
    is_v = slot_id == 2
    is_decay = slot_id == 3
    projection_width: gl.constexpr = H * D

    # Decay lanes use Q as an in-bounds dummy address and skip state writes.
    qkv_channel = (
        gl.where(is_k, projection_width, gl.where(is_v, 2 * projection_width, 0))
        + head_idx * D
        + local_offset
    )

    qkv_history0 = gl.load(
        conv_pool
        + read_page_offset * CONV_POOL_PAGE_STRIDE
        + qkv_channel * CONV_POOL_CHANNEL_STRIDE
    ).to(gl.float32)
    qkv_history1 = gl.load(
        conv_pool
        + read_page_offset * CONV_POOL_PAGE_STRIDE
        + qkv_channel * CONV_POOL_CHANNEL_STRIDE
        + CONV_POOL_HISTORY_STRIDE
    ).to(gl.float32)
    qkv_history2 = gl.load(
        conv_pool
        + read_page_offset * CONV_POOL_PAGE_STRIDE
        + qkv_channel * CONV_POOL_CHANNEL_STRIDE
        + 2 * CONV_POOL_HISTORY_STRIDE
    ).to(gl.float32)
    qkv_weight_base = conv_weights + qkv_channel * CONV_WEIGHT_ROW_STRIDE
    weight0 = gl.load(qkv_weight_base)
    weight1 = gl.load(qkv_weight_base + CONV_WEIGHT_COL_STRIDE)
    weight2 = gl.load(qkv_weight_base + 2 * CONV_WEIGHT_COL_STRIDE)
    weight3 = gl.load(qkv_weight_base + 3 * CONV_WEIGHT_COL_STRIDE)
    decay_channel = head_idx * D + local_offset
    a_value = gl.exp(gl.load(a_log + head_idx).to(gl.float32))

    # Both modes write through scratch pointers: decode aliases them to the
    # persistent pools, while verify supplies rollback buffers.
    valid_writes = ()
    safe_write_indices = ()
    for token_offset in gl.static_range(TOKENS_PER_SEQUENCE):
        write_idx = gl.load(
            write_indices + sequence_idx * TOKENS_PER_SEQUENCE + token_offset
        )
        valid_write = (write_idx >= 0) & (write_idx < NUM_SCRATCH_SLOTS)
        valid_writes = valid_writes + (valid_write,)
        safe_write_indices = safe_write_indices + (
            gl.where(valid_write, write_idx, 0).to(gl.int64),
        )
        token_idx = token_base + token_offset
        qkv_input = gl.load(mixed_qkv + token_idx * MIXED_ROW_STRIDE + qkv_channel).to(
            gl.float32
        )
        qkv_value = (
            qkv_history0 * weight0
            + qkv_history1 * weight1
            + qkv_history2 * weight2
            + qkv_input * weight3
        ).to(gl.float32)
        qkv_value *= 1.0 / (1.0 + gl.exp(-qkv_value))

        valid_write = valid_writes[token_offset]
        safe_write_idx = safe_write_indices[token_offset]
        qkv_write_base = (
            conv_scratch
            + safe_write_idx * CONV_SCRATCH_PAGE_STRIDE
            + qkv_channel * CONV_SCRATCH_CHANNEL_STRIDE
        )
        write_mask = valid_write & (slot_id != 3)
        gl.store(qkv_write_base, qkv_history1, mask=write_mask)
        gl.store(
            qkv_write_base + CONV_SCRATCH_HISTORY_STRIDE,
            qkv_history2,
            mask=write_mask,
        )
        gl.store(
            qkv_write_base + 2 * CONV_SCRATCH_HISTORY_STRIDE,
            qkv_input,
            mask=write_mask,
        )

        gate_value = gl.load(raw_g + token_idx * GATE_ROW_STRIDE + decay_channel).to(
            gl.float32
        ) + gl.load(dt_bias + decay_channel).to(gl.float32)
        if HAS_LOWER_BOUND:
            log_decay = LOWER_BOUND / (1.0 + gl.exp(-(a_value * gate_value)))
        else:
            softplus = gl.maximum(gate_value, 0.0) + gl.log(
                1.0 + gl.exp(-gl.abs(gate_value))
            )
            log_decay = -a_value * softplus
        combined = gl.where(is_decay, gl.exp(log_decay), qkv_value)
        shared_vectors.index(token_offset).store(combined)

        qkv_history0 = qkv_history1
        qkv_history1 = qkv_history2
        qkv_history2 = qkv_input

    # Publish all convolution results before cross-warp normalization; retain
    # per-token vectors for reuse across state panels.
    gl.barrier()
    q_values = ()
    k_values = ()
    decay_values = ()
    beta_values = ()
    key_queries = ()
    value_panels_by_token = ()
    for token_offset in gl.static_range(TOKENS_PER_SEQUENCE):
        token_idx = token_base + token_offset
        vectors = shared_vectors.index(token_offset)
        q_value = vectors.slice(0, D, dim=0).load(key_layout)
        k_value = vectors.slice(D, D, dim=0).load(key_layout)
        q_value *= gl.rsqrt(gl.sum(q_value * q_value, axis=0) + 1e-6) * (D**-0.5)
        k_value *= gl.rsqrt(gl.sum(k_value * k_value, axis=0) + 1e-6)
        decay = vectors.slice(3 * D, D, dim=0).load(key_layout)
        beta_value = gl.load(beta_logits + token_idx * BETA_ROW_STRIDE + head_idx).to(
            gl.float32
        )
        beta_value = 1.0 / (1.0 + gl.exp(-beta_value))
        value_panels = (
            vectors.slice(2 * D + 0, 16, dim=0).load(value_layout),
            vectors.slice(2 * D + 16, 16, dim=0).load(value_layout),
            vectors.slice(2 * D + 32, 16, dim=0).load(value_layout),
            vectors.slice(2 * D + 48, 16, dim=0).load(value_layout),
            vectors.slice(2 * D + 64, 16, dim=0).load(value_layout),
            vectors.slice(2 * D + 80, 16, dim=0).load(value_layout),
            vectors.slice(2 * D + 96, 16, dim=0).load(value_layout),
            vectors.slice(2 * D + 112, 16, dim=0).load(value_layout),
        )
        q_values = q_values + (q_value,)
        k_values = k_values + (k_value,)
        decay_values = decay_values + (decay,)
        beta_values = beta_values + (beta_value,)
        key_queries = key_queries + (gl.sum(k_value * q_value, axis=0),)
        value_panels_by_token = value_panels_by_token + (value_panels,)

    # Carry each state panel through tokens in causal order; verify stores every
    # intermediate state for partial acceptance.
    if not VERIFY_MODE:
        output_squares = gl.zeros([16], gl.float32, layout=value_layout)
    for panel_idx in gl.static_range(8):
        panel_offsets = off_pipe[0]
        running = raw_pipe[0]
        for token_offset in gl.static_range(TOKENS_PER_SEQUENCE):
            q_value = q_values[token_offset]
            k_value = k_values[token_offset]
            decay = decay_values[token_offset]
            value_panel = value_panels_by_token[token_offset][panel_idx]
            beta_value = beta_values[token_offset]

            running *= decay[None, :]
            prediction = gl.sum(running * k_value[None, :], axis=1)
            prior_output = gl.sum(running * q_value[None, :], axis=1)
            delta = beta_value * (value_panel - prediction)
            running += delta[:, None] * k_value[None, :]
            key_query = key_queries[token_offset]
            panel_out = prior_output + delta * key_query
            output_alloc.index(token_offset).slice(panel_idx * 16, 16, dim=0).store(
                panel_out
            )

            valid_write = valid_writes[token_offset]
            safe_write_idx = safe_write_indices[token_offset]
            write_base = (
                safe_write_idx * STATE_SCRATCH_PAGE_STRIDE + head_idx * D * D
            )
            gl.store(
                state_scratch + write_base + panel_offsets,
                running,
                mask=valid_write,
            )
            if not VERIFY_MODE:
                output_squares += panel_out * panel_out

        next_idx = panel_idx + PIPELINE_DEPTH
        if next_idx < 8:
            next_offsets = (
                panel_value_offsets[:, None] + next_idx * 16
            ) * D + key_offsets[None, :]
            next_raw = gl.load(state_pool + read_base + next_offsets).to(gl.float32)
            off_pipe = off_pipe[1:] + (next_offsets,)
            raw_pipe = raw_pipe[1:] + (next_raw,)
        else:
            off_pipe = off_pipe[1:] + (off_pipe[-1],)
            raw_pipe = raw_pipe[1:] + (raw_pipe[-1],)

    # Synchronize value-layout panel writers before compact-layout output loads;
    # only decode applies gated RMSNorm.
    gl.barrier()
    for token_offset in gl.static_range(TOKENS_PER_SEQUENCE):
        token_idx = token_base + token_offset
        out_value = output_alloc.index(token_offset).load(compact_layout)
        if VERIFY_MODE:
            gl.store(
                output + output_base + token_offset * H * D + output_offsets,
                out_value.to(output.dtype.element_ty),
            )
        else:
            output_sumsq = gl.sum(output_squares, axis=0)
            inverse_rms = gl.rsqrt(output_sumsq / D + NORM_EPS)
            gate = gl.load(
                output_gate
                + token_idx * OUTPUT_GATE_ROW_STRIDE
                + head_idx * D
                + output_offsets,
            ).to(gl.float32)
            weight = gl.load(norm_weight + output_offsets).to(gl.float32)
            out_value *= inverse_rms * weight * (1.0 / (1.0 + gl.exp(-gate)))
            gl.store(
                output + output_base + output_offsets,
                out_value.to(output.dtype.element_ty),
            )


def gluon_kda_recurrent_decode_gfx1250(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run one-token indexed KDA decode on a V-major recurrent-state pool."""
    tensors = (q, k, v, g_raw, beta_logits, A_log, dt_bias, state_pool)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx1250 Gluon KDA decode requires GPU tensors")
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q must have shape [1, batch, heads, key_dim]")
    if read_indices.ndim != 1 or write_indices.shape != read_indices.shape:
        raise ValueError("read_indices and write_indices must be matching vectors")
    if q.shape[1] != read_indices.numel():
        raise ValueError("gfx1250 Gluon KDA decode requires one token per sequence")
    if q.shape != k.shape or q.shape != g_raw.shape:
        raise ValueError("q, k, and raw_g must have identical shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("v must match q through the head dimension")

    _, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if beta_logits.shape != (1, tokens, heads):
        raise ValueError("beta_logits must have shape [1, batch, heads]")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() != tokens + 1:
        raise ValueError("cu_seqlens must contain one boundary per decode row")
    expected_tail = (heads, value_dim, key_dim)
    if state_pool.ndim != 4 or state_pool.shape[1:] != expected_tail:
        raise ValueError(
            f"state_pool must have physical shape [pages, H, V, K] "
            f"with tail {expected_tail}"
        )
    expected_strides = (value_dim * key_dim, key_dim, 1)
    if state_pool.stride()[1:] != expected_strides:
        raise ValueError("state_pool inner [H, V, K] dimensions must be contiguous")
    if state_pool.stride(0) < heads * key_dim * value_dim:
        raise ValueError("state_pool pages must not overlap")
    if A_log.shape != (heads,) or dt_bias.numel() != heads * key_dim:
        raise ValueError("invalid KDA gate parameter shapes")
    expected_inner_strides = (
        (q, key_dim),
        (k, key_dim),
        (g_raw, key_dim),
        (v, value_dim),
    )
    if any(
        tensor.stride(-1) != 1 or tensor.stride(-2) != width
        for tensor, width in expected_inner_strides
    ):
        raise ValueError("KDA inputs must have contiguous head vectors")
    if beta_logits.stride(-1) != 1:
        raise ValueError("KDA beta logits must have contiguous heads")

    A_log = A_log.contiguous()
    dt_bias = dt_bias.view(heads, key_dim).contiguous()
    read_indices = read_indices.to(device=q.device, dtype=torch.int32).contiguous()
    write_indices = write_indices.to(device=q.device, dtype=torch.int32).contiguous()
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()

    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(32, triton.next_power_of_2(value_dim))
    _kda_recurrent_decode_kernel[(triton.cdiv(value_dim, block_value), tokens * heads)](
        q,
        k,
        v,
        g_raw,
        beta_logits,
        state_pool,
        read_indices,
        write_indices,
        output,
        cu_seqlens,
        A_log,
        dt_bias,
        H=heads,
        K=key_dim,
        V=value_dim,
        Q_TOKEN_STRIDE=q.stride(1),
        K_TOKEN_STRIDE=k.stride(1),
        V_TOKEN_STRIDE=v.stride(1),
        G_TOKEN_STRIDE=g_raw.stride(1),
        BETA_TOKEN_STRIDE=beta_logits.stride(1),
        BK=block_key,
        BV=block_value,
        NUM_SLOTS=state_pool.shape[0],
        STATE_PAGE_STRIDE=state_pool.stride(0),
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        num_warps=4 if block_key == 128 and block_value == 32 else 2,
        num_stages=2,
    )
    return output


def gluon_kda_fused_decode_gfx1250(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    raw_g: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run fused single-token K3 decode against paged convolution/KDA state.

    Args:
        mixed_qkv: BF16 pre-convolution Q/K/V rows with shape
            ``[batch, 3 * num_heads * head_dim]``.
        conv_weights: Contiguous four-tap depthwise convolution weights with
            shape ``[3 * num_heads * head_dim, 4]``.
        conv_states: Mutable BF16 convolution state pool with shape
            ``[pages, 3 * num_heads * head_dim, 3]``.
        raw_g: Projected decay-gate values with shape
            ``[batch, num_heads * head_dim]``.
        beta_logits: Per-token, per-head update logits with shape
            ``[batch, num_heads]``.
        A_log: Per-head FP32 decay parameters with shape ``[num_heads]``.
        dt_bias: Per-head, per-channel FP32 bias with shape
            ``[num_heads * head_dim]``.
        output_gate: Gated-RMSNorm logits with shape
            ``[batch, num_heads * head_dim]``.
        norm_weight: Gated-RMSNorm weights with shape ``[head_dim]``.
        norm_eps: Gated-RMSNorm epsilon.
        state_pool: Mutable FP32 recurrent state, physical shape
            ``[pages, num_heads, head_dim(V), head_dim(K)]`` (V-major).
        read_indices: Source state page per batch row.
        write_indices: Destination state page per batch row.
        num_heads: Number of local heads; this specialization requires 12.
        head_dim: Per-head width; this specialization requires 128.
        cu_seqlens: Packed row boundaries with shape ``[batch + 1]``.
        lower_bound: Optional safe lower bound for the log-decay gate.

    Returns:
        Gated-RMSNorm output with shape ``[1, batch, 12, 128]``. Valid
        destination convolution and recurrent states are updated in place.
    """
    tensors = (
        mixed_qkv,
        conv_weights,
        conv_states,
        raw_g,
        beta_logits,
        A_log,
        dt_bias,
        output_gate,
        norm_weight,
        state_pool,
        read_indices,
        write_indices,
        cu_seqlens,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx1250 fused KDA decode requires GPU tensors")
    if num_heads != 12 or head_dim != 128:
        raise ValueError("gfx1250 fused KDA decode requires 12 heads of width 128")

    tokens = read_indices.numel()
    projection_width = num_heads * head_dim
    if mixed_qkv.ndim != 2 or mixed_qkv.shape != (tokens, 3 * projection_width):
        raise ValueError("mixed_qkv must have shape [batch, 3 * heads * head_dim]")
    if mixed_qkv.stride(1) != 1:
        raise ValueError("mixed_qkv channels must be contiguous")
    if (
        conv_weights.shape != (3 * projection_width, 4)
        or not conv_weights.is_contiguous()
    ):
        raise ValueError("conv_weights must be contiguous [3 * heads * head_dim, 4]")
    if conv_states.ndim != 3 or conv_states.shape[1:] != (3 * projection_width, 3):
        raise ValueError("conv_states must have shape [pages, 3 * heads * head_dim, 3]")
    if conv_states.stride()[1:] != (3, 1):
        raise ValueError(
            "conv_states inner channel/history dimensions must be contiguous"
        )
    if raw_g.shape != (tokens, projection_width) or raw_g.stride(1) != 1:
        raise ValueError("raw_g must have shape [batch, heads * head_dim]")
    if beta_logits.shape != (tokens, num_heads) or beta_logits.stride(1) != 1:
        raise ValueError("beta_logits must have shape [batch, heads]")
    if output_gate.shape != (tokens, projection_width) or output_gate.stride(1) != 1:
        raise ValueError("output_gate must have shape [batch, heads * head_dim]")
    if A_log.shape != (num_heads,) or not A_log.is_contiguous():
        raise ValueError("A_log must have shape [heads]")
    if dt_bias.shape != (projection_width,) or not dt_bias.is_contiguous():
        raise ValueError("dt_bias must have shape [heads * head_dim]")
    if norm_weight.shape != (head_dim,) or not norm_weight.is_contiguous():
        raise ValueError("norm_weight must have shape [head_dim]")
    if state_pool.ndim != 4 or state_pool.shape[1:] != (
        num_heads,
        head_dim,
        head_dim,
    ):
        raise ValueError("state_pool must have physical shape [pages, heads, V, K]")
    if state_pool.stride()[1:] != (head_dim * head_dim, head_dim, 1):
        raise ValueError("state_pool inner dimensions must be contiguous")
    if conv_states.shape[0] != state_pool.shape[0]:
        raise ValueError(
            "convolution and recurrent state pools must have equal capacity"
        )
    if read_indices.shape != (tokens,) or write_indices.shape != (tokens,):
        raise ValueError("read_indices and write_indices must match the decode batch")
    if read_indices.dtype != torch.int32 or write_indices.dtype != torch.int32:
        raise ValueError("read_indices and write_indices must be int32")
    if not read_indices.is_contiguous() or not write_indices.is_contiguous():
        raise ValueError("read_indices and write_indices must be contiguous")
    if cu_seqlens.shape != (tokens + 1,) or cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be an int32 boundary vector")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")

    output = torch.empty(
        (1, tokens, num_heads, head_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    # Beyond one CTA per CU, reduce VGPR pressure to favor higher occupancy.
    # Past two CTAs per CU the state reads saturate memory, so a shallower
    # window trades prefetch distance for the occupancy that matters there.
    pipeline_depth = 3 if num_heads * tokens > 2 * _GFX1250_NUM_CUS else 4
    _kda_fused_decode_kernel[(num_heads, tokens)](
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_states,
        raw_g,
        beta_logits,
        output_gate,
        norm_weight,
        state_pool,
        state_pool,
        read_indices,
        write_indices,
        output,
        cu_seqlens,
        A_log,
        dt_bias,
        H=num_heads,
        D=head_dim,
        TOKENS_PER_SEQUENCE=1,
        VERIFY_MODE=False,
        MIXED_ROW_STRIDE=mixed_qkv.stride(0),
        CONV_WEIGHT_ROW_STRIDE=conv_weights.stride(0),
        CONV_WEIGHT_COL_STRIDE=conv_weights.stride(1),
        CONV_POOL_PAGE_STRIDE=conv_states.stride(0),
        CONV_POOL_CHANNEL_STRIDE=conv_states.stride(1),
        CONV_POOL_HISTORY_STRIDE=conv_states.stride(2),
        CONV_SCRATCH_PAGE_STRIDE=conv_states.stride(0),
        CONV_SCRATCH_CHANNEL_STRIDE=conv_states.stride(1),
        CONV_SCRATCH_HISTORY_STRIDE=conv_states.stride(2),
        GATE_ROW_STRIDE=raw_g.stride(0),
        BETA_ROW_STRIDE=beta_logits.stride(0),
        OUTPUT_GATE_ROW_STRIDE=output_gate.stride(0),
        STATE_POOL_PAGE_STRIDE=state_pool.stride(0),
        STATE_SCRATCH_PAGE_STRIDE=state_pool.stride(0),
        NUM_POOL_SLOTS=state_pool.shape[0],
        NUM_SCRATCH_SLOTS=state_pool.shape[0],
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        NORM_EPS=norm_eps,
        PIPELINE_DEPTH=pipeline_depth,
        num_warps=8,
        num_stages=2,
    )
    return output


def gluon_kda_fused_verify_gfx1250(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_pool: torch.Tensor,
    conv_scratch: torch.Tensor,
    raw_g: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run fused KDA verify and materialize rollback states."""
    if num_heads != 12 or head_dim != 128:
        raise ValueError("gfx1250 fused KDA verify requires H=12, D=128")
    if draft_token_num <= 0:
        raise ValueError("draft_token_num must be positive")
    batch = read_indices.numel()
    tokens = batch * draft_token_num
    projection_width = num_heads * head_dim
    tensors = (
        mixed_qkv,
        conv_weights,
        conv_pool,
        conv_scratch,
        raw_g,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        state_scratch,
        read_indices,
        write_indices,
    )
    if not all(
        tensor.is_cuda and tensor.device == mixed_qkv.device for tensor in tensors
    ):
        raise ValueError("gfx1250 fused KDA verify requires colocated GPU tensors")
    if mixed_qkv.shape != (tokens, 3 * projection_width) or mixed_qkv.stride(1) != 1:
        raise ValueError("mixed_qkv must be dense [batch * T, 3 * H * D]")
    if (
        conv_weights.shape != (3 * projection_width, 4)
        or not conv_weights.is_contiguous()
    ):
        raise ValueError("conv_weights must be contiguous [3 * H * D, 4]")
    conv_tail = (3 * projection_width, 3)
    if conv_pool.ndim != 3 or conv_pool.shape[1:] != conv_tail:
        raise ValueError("conv_pool must have shape [pages, 3 * H * D, 3]")
    if conv_scratch.ndim != 3 or conv_scratch.shape[1:] != conv_tail:
        raise ValueError("conv_scratch must have shape [rows, 3 * H * D, 3]")
    if conv_pool.stride()[1:] != (3, 1) or conv_scratch.stride()[1:] != (3, 1):
        raise ValueError("convolution state inner dimensions must be contiguous")
    if raw_g.shape != (tokens, projection_width) or raw_g.stride(1) != 1:
        raise ValueError("raw_g must be dense [batch * T, H * D]")
    if beta_logits.shape != (tokens, num_heads) or beta_logits.stride(1) != 1:
        raise ValueError("beta_logits must be dense [batch * T, H]")
    state_tail = (num_heads, head_dim, head_dim)
    if state_pool.ndim != 4 or state_pool.shape[1:] != state_tail:
        raise ValueError("state_pool must have shape [pages, H, V, K]")
    if state_scratch.ndim != 4 or state_scratch.shape[1:] != state_tail:
        raise ValueError("state_scratch must have shape [rows, H, V, K]")
    expected_state_strides = (head_dim * head_dim, head_dim, 1)
    if (
        state_pool.stride()[1:] != expected_state_strides
        or state_scratch.stride()[1:] != expected_state_strides
    ):
        raise ValueError("recurrent state inner dimensions must be contiguous")
    if read_indices.shape != (batch,) or write_indices.shape != (
        batch,
        draft_token_num,
    ):
        raise ValueError("verify indices must have shapes [batch] and [batch, T]")
    if A_log.shape != (num_heads,) or not A_log.is_contiguous():
        raise ValueError("A_log must be contiguous [H]")
    if dt_bias.shape != (projection_width,) or not dt_bias.is_contiguous():
        raise ValueError("dt_bias must be contiguous [H * D]")

    read_indices = read_indices.to(
        device=mixed_qkv.device, dtype=torch.int32
    ).contiguous()
    write_indices = write_indices.to(
        device=mixed_qkv.device, dtype=torch.int32
    ).contiguous()
    output = torch.empty(
        (1, tokens, num_heads, head_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    pipeline_depth = 3 if num_heads * batch > 2 * _GFX1250_NUM_CUS else 4
    _kda_fused_decode_kernel[(num_heads, batch)](
        mixed_qkv,
        conv_weights,
        conv_pool,
        conv_scratch,
        raw_g,
        beta_logits,
        raw_g,
        dt_bias,
        state_pool,
        state_scratch,
        read_indices,
        write_indices,
        output,
        read_indices,
        A_log,
        dt_bias,
        H=num_heads,
        D=head_dim,
        TOKENS_PER_SEQUENCE=draft_token_num,
        VERIFY_MODE=True,
        MIXED_ROW_STRIDE=mixed_qkv.stride(0),
        CONV_WEIGHT_ROW_STRIDE=conv_weights.stride(0),
        CONV_WEIGHT_COL_STRIDE=conv_weights.stride(1),
        CONV_POOL_PAGE_STRIDE=conv_pool.stride(0),
        CONV_POOL_CHANNEL_STRIDE=conv_pool.stride(1),
        CONV_POOL_HISTORY_STRIDE=conv_pool.stride(2),
        CONV_SCRATCH_PAGE_STRIDE=conv_scratch.stride(0),
        CONV_SCRATCH_CHANNEL_STRIDE=conv_scratch.stride(1),
        CONV_SCRATCH_HISTORY_STRIDE=conv_scratch.stride(2),
        GATE_ROW_STRIDE=raw_g.stride(0),
        BETA_ROW_STRIDE=beta_logits.stride(0),
        OUTPUT_GATE_ROW_STRIDE=raw_g.stride(0),
        STATE_POOL_PAGE_STRIDE=state_pool.stride(0),
        STATE_SCRATCH_PAGE_STRIDE=state_scratch.stride(0),
        NUM_POOL_SLOTS=state_pool.shape[0],
        NUM_SCRATCH_SLOTS=state_scratch.shape[0],
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        NORM_EPS=0.0,
        PIPELINE_DEPTH=pipeline_depth,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = [
    "gluon_kda_fused_decode_gfx1250",
    "gluon_kda_fused_verify_gfx1250",
    "gluon_kda_recurrent_decode_gfx1250",
]
