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

"""KPool cache planning, writes, and sparse-history selection for DSA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from tokenspeed_kernel.ops.attention.kpool import (
    KPoolPreparedQuery,
    kpool_decode_append,
    kpool_decode_topk,
    kpool_prefill_compress,
    kpool_prefill_prepare_query,
    kpool_prefill_tail_write,
    kpool_prefill_topk,
)

from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.tensor import upload_packed

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext


@dataclass(frozen=True)
class KPoolWritePlan:
    """Physical pooled-index writes and request-local tail updates."""

    pool_request_slots: torch.Tensor
    pool_n_from_tail: torch.Tensor
    pool_chunk_src: torch.Tensor
    pool_tail_logical_base: torch.Tensor
    pool_write_slots: torch.Tensor
    tail_request_slots: torch.Tensor
    tail_chunk_src: torch.Tensor
    tail_dst_positions: torch.Tensor
    tail_write_counts: torch.Tensor


@dataclass(frozen=True)
class KPoolPrefillPlan:
    """Layer-invariant KPool cache-write and ragged-selection metadata."""

    write: KPoolWritePlan
    num_prefill_tokens: int
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    req_ids: torch.Tensor
    causal_lens: torch.Tensor
    pool_workspace_slots: torch.Tensor
    row_starts: torch.Tensor
    row_ends: torch.Tensor
    max_num_pools: int


@dataclass(frozen=True)
class KPoolPrefillTopK:
    """KPool selection output consumed by the generic DSA prefill path."""

    workspace_indices: torch.Tensor
    topk_lens: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    kv_seq_lens: torch.Tensor
    max_seq_len: int
    kv_workspace_slots: torch.Tensor


def dsa_prefill_host_lengths(metadata: Any, num_extends: int) -> tuple[int, int]:
    """Return prefill token count and maximum sequence length from CPU mirrors."""
    prefix = metadata.extend_prefix_lens_cpu[:num_extends]
    extend = metadata.extend_seq_lens_cpu[:num_extends]
    if prefix.device.type != "cpu" or extend.device.type != "cpu":
        raise RuntimeError("DSA prefill length mirrors must remain on CPU")
    prefix_lens = [int(value) for value in prefix]
    extend_lens = [int(value) for value in extend]
    return sum(extend_lens), max(
        (p + e for p, e in zip(prefix_lens, extend_lens)), default=0
    )


@dataclass(frozen=True)
class _KPoolHostWritePlan:
    pool_req_ids: torch.Tensor
    pool_n_from_tail: torch.Tensor
    pool_chunk_src: torch.Tensor
    pool_ids: torch.Tensor
    pool_tail_logical_base: torch.Tensor
    tail_req_ids: torch.Tensor
    tail_chunk_src: torch.Tensor
    tail_dst_positions: torch.Tensor
    tail_write_counts: torch.Tensor

    def parts(self) -> tuple[torch.Tensor, ...]:
        return (
            self.pool_req_ids,
            self.pool_n_from_tail,
            self.pool_chunk_src,
            self.pool_ids,
            self.pool_tail_logical_base,
            self.tail_req_ids,
            self.tail_chunk_src,
            self.tail_dst_positions,
            self.tail_write_counts,
        )


def _host_int64(values: torch.Tensor) -> torch.Tensor:
    if values.device.type != "cpu":
        raise RuntimeError("KPool plans require scheduler CPU length mirrors")
    return values.reshape(-1).to(torch.int64)


def _kpool_host_write_plan(
    starts: torch.Tensor, lengths: torch.Tensor, kpool: int
) -> _KPoolHostWritePlan:
    """Decompose each request's chunk into completed pools and a tail remainder.

    The first pool of a request splices ``start % kpool`` tokens already held
    in its tail ring with the chunk's leading tokens; later pools read whole
    ``kpool``-token runs of the chunk. Tokens past the last completed pool
    become the request's new tail.
    """
    request_ids = torch.arange(starts.numel(), dtype=torch.int64)
    chunk_begin = torch.cumsum(lengths, 0) - lengths
    live = lengths > 0
    first_slot = starts % kpool
    base_pool = starts // kpool
    num_pools = torch.where(live, (starts + lengths) // kpool - base_pool, 0)

    pool_req_ids = torch.repeat_interleave(request_ids, num_pools)
    pool_local = torch.arange(pool_req_ids.numel(), dtype=torch.int64)
    pool_local -= torch.repeat_interleave(
        torch.cumsum(num_pools, 0) - num_pools, num_pools
    )
    pool_ids = base_pool[pool_req_ids] + pool_local
    pool_first_slot = first_slot[pool_req_ids]
    splice = pool_local == 0
    pool_n_from_tail = torch.where(splice, pool_first_slot, 0)
    pool_chunk_src = chunk_begin[pool_req_ids] + torch.where(
        splice, 0, kpool - pool_first_slot + (pool_local - 1) * kpool
    )

    consumed = torch.where(num_pools > 0, num_pools * kpool - first_slot, 0)
    tail_count = torch.where(live, lengths - consumed, 0)
    has_tail = tail_count > 0
    return _KPoolHostWritePlan(
        pool_req_ids=pool_req_ids,
        pool_n_from_tail=pool_n_from_tail.to(torch.int32),
        pool_chunk_src=pool_chunk_src,
        pool_ids=pool_ids,
        pool_tail_logical_base=pool_ids * kpool,
        tail_req_ids=request_ids[has_tail],
        tail_chunk_src=(chunk_begin + lengths - tail_count)[has_tail],
        tail_dst_positions=(starts + consumed)[has_tail],
        tail_write_counts=tail_count[has_tail].to(torch.int32),
    )


def _index_cache_slots(
    index_block_table: torch.Tensor,
    req_ids: torch.Tensor,
    pool_ids: torch.Tensor,
    index_rows_per_page: int,
) -> torch.Tensor:
    if pool_ids.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=index_block_table.device)
    pages = index_block_table[
        req_ids, torch.div(pool_ids, index_rows_per_page, rounding_mode="floor")
    ].to(torch.int64)
    return pages * index_rows_per_page + torch.remainder(pool_ids, index_rows_per_page)


def _finalize_kpool_write_plan(
    uploaded: tuple[torch.Tensor, ...],
    *,
    index_block_table: torch.Tensor,
    request_slots: torch.Tensor,
    index_rows_per_page: int,
) -> KPoolWritePlan:
    (
        pool_req_ids,
        pool_n_from_tail,
        pool_chunk_src,
        pool_ids,
        pool_tail_logical_base,
        tail_req_ids,
        tail_chunk_src,
        tail_dst_positions,
        tail_write_counts,
    ) = uploaded
    request_slots = request_slots.to(device=index_block_table.device, dtype=torch.int64)
    return KPoolWritePlan(
        pool_request_slots=request_slots.index_select(0, pool_req_ids),
        pool_n_from_tail=pool_n_from_tail,
        pool_chunk_src=pool_chunk_src,
        pool_tail_logical_base=pool_tail_logical_base,
        pool_write_slots=_index_cache_slots(
            index_block_table, pool_req_ids, pool_ids, index_rows_per_page
        ),
        tail_request_slots=request_slots.index_select(0, tail_req_ids),
        tail_chunk_src=tail_chunk_src,
        tail_dst_positions=tail_dst_positions,
        tail_write_counts=tail_write_counts,
    )


def build_kpool_write_plan(
    *,
    req_start_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    index_block_table: torch.Tensor,
    request_slots: torch.Tensor,
    kpool: int,
    index_rows_per_page: int,
) -> KPoolWritePlan:
    """Map completed pools to index pages and remainders to request tail slots."""
    starts = _host_int64(req_start_positions)
    offsets = _host_int64(query_start_loc)
    if request_slots.numel() != starts.numel():
        raise ValueError(
            "KPool request-slot count differs from the request count: "
            f"{request_slots.numel()} != {starts.numel()}"
        )
    host = _kpool_host_write_plan(starts, offsets[1:] - offsets[:-1], kpool)
    return _finalize_kpool_write_plan(
        upload_packed(host.parts(), index_block_table.device),
        index_block_table=index_block_table,
        request_slots=request_slots,
        index_rows_per_page=index_rows_per_page,
    )


def build_kpool_prefill_plan(
    *,
    prefix_lens_cpu: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    index_block_table: torch.Tensor,
    request_slots: torch.Tensor,
    kpool: int,
    index_rows_per_page: int,
    token_capacity: int | None = None,
) -> KPoolPrefillPlan:
    """Build cache-write and ragged-selection metadata for a prefill batch.

    Every per-request and per-token vector is derived on the host with
    vectorized integer arithmetic and shipped in one pinned upload; only the
    page-table gathers run on the device.

    Args:
        prefix_lens_cpu: Per-request prefix lengths on CPU.
        extend_lens_cpu: Per-request extend lengths on CPU.
        index_block_table: Logical-pool-page to physical-index-page table.
        request_slots: Stable request-pool row for each batch request.
        kpool: Raw tokens represented by one compressed index row.
        index_rows_per_page: Compressed rows stored in one index page.
        token_capacity: Optional fixed token-row capacity. Rows after the real
            prefill tokens are emitted as inactive padding metadata.

    Returns:
        Cache-write metadata plus the ragged selection workspace mapping.
    """
    starts = _host_int64(prefix_lens_cpu)
    lengths = _host_int64(extend_lens_cpu)
    if starts.numel() != lengths.numel():
        raise ValueError(
            "KPool prefix and extend length counts differ: "
            f"{starts.numel()} != {lengths.numel()}"
        )
    if request_slots.numel() != starts.numel():
        raise ValueError(
            "KPool request-slot count differs from the request count: "
            f"{request_slots.numel()} != {starts.numel()}"
        )
    negative = torch.nonzero((starts < 0) | (lengths < 0)).reshape(-1)
    if negative.numel():
        req_id = int(negative[0])
        raise ValueError(
            "KPool prefill lengths must be non-negative, got "
            f"prefix={int(starts[req_id])}, extend={int(lengths[req_id])} "
            f"for request {req_id}"
        )

    request_ids = torch.arange(starts.numel(), dtype=torch.int64)
    query_offsets = torch.zeros(starts.numel() + 1, dtype=torch.int64)
    torch.cumsum(lengths, 0, out=query_offsets[1:])
    num_prefill_tokens = int(query_offsets[-1])
    final_num_pools = (starts + lengths) // kpool
    workspace_starts = torch.cumsum(final_num_pools, 0) - final_num_pools
    max_num_pools = int(final_num_pools.max()) if final_num_pools.numel() else 0

    req_ids = torch.repeat_interleave(request_ids, lengths)
    positions = torch.arange(num_prefill_tokens, dtype=torch.int64)
    positions -= torch.repeat_interleave(query_offsets[:-1], lengths)
    positions += starts[req_ids]
    causal_lens = positions + 1
    row_starts = workspace_starts[req_ids]
    row_ends = row_starts + causal_lens // kpool
    workspace_req_ids = torch.repeat_interleave(request_ids, final_num_pools)
    workspace_pool_ids = torch.arange(workspace_req_ids.numel(), dtype=torch.int64)
    workspace_pool_ids -= torch.repeat_interleave(workspace_starts, final_num_pools)

    if token_capacity is None:
        token_capacity = num_prefill_tokens
    token_capacity = int(token_capacity)
    if token_capacity < num_prefill_tokens:
        raise ValueError(
            "KPool token capacity is smaller than the prefill token count: "
            f"capacity={token_capacity}, tokens={num_prefill_tokens}"
        )
    num_padding_tokens = token_capacity - num_prefill_tokens
    if num_padding_tokens:
        # Empty ranges make selection a no-op. A zero causal length also keeps
        # selected-slot attention from reading cache entries for these rows.
        padding = torch.zeros(num_padding_tokens, dtype=torch.int64)
        positions = torch.cat((positions, padding))
        req_ids = torch.cat((req_ids, padding))
        causal_lens = torch.cat((causal_lens, padding))
        row_starts = torch.cat((row_starts, padding))
        row_ends = torch.cat((row_ends, padding))

    host_write = _kpool_host_write_plan(starts, lengths, kpool)
    write_parts = host_write.parts()
    uploaded = upload_packed(
        (
            *write_parts,
            positions.to(torch.int32),
            query_offsets.to(torch.int32),
            req_ids.to(torch.int32),
            causal_lens.to(torch.int32),
            row_starts.to(torch.int32),
            row_ends.to(torch.int32),
            workspace_req_ids,
            workspace_pool_ids,
        ),
        index_block_table.device,
    )
    write = _finalize_kpool_write_plan(
        uploaded[: len(write_parts)],
        index_block_table=index_block_table,
        request_slots=request_slots,
        index_rows_per_page=index_rows_per_page,
    )
    (
        positions,
        query_start_loc,
        req_ids,
        causal_lens,
        row_starts,
        row_ends,
        workspace_req_ids,
        workspace_pool_ids,
    ) = uploaded[len(write_parts) :]
    return KPoolPrefillPlan(
        write=write,
        num_prefill_tokens=num_prefill_tokens,
        positions=positions,
        query_start_loc=query_start_loc,
        req_ids=req_ids,
        causal_lens=causal_lens,
        pool_workspace_slots=_index_cache_slots(
            index_block_table,
            workspace_req_ids,
            workspace_pool_ids,
            index_rows_per_page,
        ),
        row_starts=row_starts,
        row_ends=row_ends,
        max_num_pools=max_num_pools,
    )


class KPoolRuntime:
    """Own KPool-derived state and operations for a DSA backend."""

    def __init__(self, pool_size: int, index_topk: int) -> None:
        self.pool_size = pool_size
        self.index_topk = index_topk
        self.prefill_plan: KPoolPrefillPlan | None = None
        self.req_pool_indices: torch.Tensor | None = None

    def reset_forward(self, req_pool_indices: torch.Tensor | None = None) -> None:
        """Reset KPool state for a new forward."""
        self.prefill_plan = None
        self.req_pool_indices = req_pool_indices

    def ensure_prefill_plan(
        self,
        ctx: ForwardContext,
        backend: Any,
        layer_id: int,
        token_capacity: int | None = None,
    ) -> None:
        """Build the layer-invariant prefill plan once for the current forward."""
        if ctx.num_extends <= 0 or self.prefill_plan is not None:
            return

        metadata = backend.chunked_prefill_metadata
        index_table = backend.kpool_prefill_page_table(ctx.num_extends)
        index_cache = ctx.token_to_kv_pool.get_kpool_buffers(layer_id)[0]
        # The slots come from the backend's per-forward publication, not the
        # prefill metadata: paged leaves' metadata carries no pool indices.
        if self.req_pool_indices is None:
            raise RuntimeError("DSA KPool prefill requires request-pool indices")
        self.prefill_plan = build_kpool_prefill_plan(
            prefix_lens_cpu=metadata.extend_prefix_lens_cpu[: ctx.num_extends],
            extend_lens_cpu=metadata.extend_seq_lens_cpu[: ctx.num_extends],
            index_block_table=index_table,
            request_slots=self.req_pool_indices[: ctx.num_extends],
            kpool=self.pool_size,
            index_rows_per_page=index_cache.shape[1],
            token_capacity=token_capacity,
        )

    def write_prefill(
        self,
        *,
        key: torch.Tensor,
        gate: torch.Tensor,
        compress_ape: torch.Tensor,
        ctx: ForwardContext,
        backend: Any,
        layer_id: int,
    ) -> None:
        """Write completed prefill pools and preserve their incomplete tails."""
        pool = ctx.token_to_kv_pool
        index_cache, tail_k, tail_gate = pool.get_kpool_buffers(layer_id)
        shared_plan = self.prefill_plan
        if shared_plan is None:
            raise RuntimeError("DSA KPool prefill plan was not initialized")
        plan = shared_plan.write

        if plan.pool_write_slots.numel() > 0:
            index_values, index_scales = pool.index_k_block_views(index_cache)
            kpool_prefill_compress(
                key,
                gate,
                tail_k,
                tail_gate,
                plan.pool_request_slots,
                plan.pool_n_from_tail,
                plan.pool_chunk_src,
                plan.pool_tail_logical_base,
                plan.pool_write_slots,
                index_values,
                index_scales,
                compress_ape,
            )

        if plan.tail_write_counts.numel() > 0:
            kpool_prefill_tail_write(
                key,
                gate,
                tail_k,
                tail_gate,
                plan.tail_chunk_src,
                plan.tail_request_slots,
                plan.tail_dst_positions,
                plan.tail_write_counts,
                pool_size=self.pool_size,
            )

    def write_decode(
        self,
        *,
        key: torch.Tensor,
        gate: torch.Tensor,
        compress_ape: torch.Tensor,
        ctx: ForwardContext,
        backend: Any,
        layer_id: int,
        num_reqs: int,
        q_len_per_req: int,
    ) -> None:
        """Append decode tokens to request tails and flush completed pools."""
        metadata = backend.forward_decode_metadata
        row_start = int(metadata.num_extends or 0)
        seq_lens = metadata.seq_lens_k[row_start : row_start + num_reqs].to(torch.int32)
        index_table = backend.kpool_decode_page_table(row_start, num_reqs)
        request_slots = self.req_pool_indices
        if request_slots is None:
            raise RuntimeError("DSA KPool decode requires request-pool indices")
        request_slots = request_slots[row_start : row_start + num_reqs].to(torch.int32)

        pool = ctx.token_to_kv_pool
        index_cache, tail_k, tail_gate = pool.get_kpool_buffers(layer_id)
        index_values, index_scales = pool.index_k_block_views(index_cache)
        keys = key.view(num_reqs, q_len_per_req, -1)
        gates = gate.view(num_reqs, q_len_per_req, -1)

        kpool_decode_append(
            keys,
            gates,
            tail_k,
            tail_gate,
            seq_lens,
            request_slots,
            index_table,
            index_values,
            index_scales,
            compress_ape,
        )

    def select_decode(
        self,
        *,
        query: torch.Tensor,
        weights: torch.Tensor,
        softmax_scale: float,
        ctx: ForwardContext,
        layer_id: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        q_len_per_req: int,
        decode_start: int,
        num_decode_tokens: int,
        out: torch.Tensor,
        lens_out: torch.Tensor,
    ) -> None:
        """Select decode history slots into caller-owned reusable workspaces."""
        history_table = page_table[: seq_lens.numel()]
        index_cache = ctx.token_to_kv_pool.get_kpool_buffers(layer_id)[0]
        kpool_decode_topk(
            query[decode_start : decode_start + num_decode_tokens].contiguous(),
            index_cache,
            weights[decode_start : decode_start + num_decode_tokens],
            seq_lens,
            history_table,
            history_table,
            pool_size=self.pool_size,
            page_size=index_cache.shape[1],
            kv_page_size=ctx.token_to_kv_pool.arena.kv_page_size,
            topk_pools=self.index_topk // self.pool_size,
            softmax_scale=softmax_scale,
            q_len_per_req=q_len_per_req,
            max_seq_len=ctx.attn_backend.max_context_len,
            out=out[decode_start : decode_start + num_decode_tokens],
            lens_out=lens_out[decode_start : decode_start + num_decode_tokens],
        )

    def prepare_prefill_query(
        self,
        *,
        query: torch.Tensor,
        weights: torch.Tensor,
        softmax_scale: float,
        ctx: ForwardContext,
        layer_id: int,
    ) -> KPoolPreparedQuery | None:
        """Build the query-side top-k inputs that need no pooled-cache reads.

        Returns whatever the planned prefill top-k solution consumes ahead of
        time, so the caller can issue it while this layer's pools are still
        being compressed and written.
        """
        index_cache = ctx.token_to_kv_pool.get_kpool_buffers(layer_id)[0]
        return kpool_prefill_prepare_query(
            query.contiguous(),
            index_cache,
            weights,
            pool_size=self.pool_size,
            page_size=index_cache.shape[1],
            topk_pools=self.index_topk // self.pool_size,
            softmax_scale=softmax_scale,
            apply_relu=True,
        )

    def select_prefill(
        self,
        *,
        query: torch.Tensor,
        weights: torch.Tensor,
        softmax_scale: float,
        prepared_query: KPoolPreparedQuery | None,
        ctx: ForwardContext,
        backend: Any,
        layer_id: int,
        num_prefill_tokens: int,
    ) -> KPoolPrefillTopK | None:
        """Select causal prefill history and return the generic DSA inputs."""
        metadata = backend.chunked_prefill_metadata
        prefix_lens = metadata.extend_prefix_lens[: ctx.num_extends].to(torch.int32)
        extend_lens = metadata.extend_seq_lens[: ctx.num_extends].to(torch.int32)
        seq_lens = prefix_lens + extend_lens
        if seq_lens.numel() == 0:
            return None
        host_token_count, max_seq_len = dsa_prefill_host_lengths(
            metadata, ctx.num_extends
        )
        if host_token_count != num_prefill_tokens:
            raise RuntimeError(
                "DSA KPool prefill token count mismatch: "
                f"metadata={host_token_count}, tokens={num_prefill_tokens}"
            )

        history_table = backend.kpool_prefill_page_table(ctx.num_extends)
        index_cache = ctx.token_to_kv_pool.get_kpool_buffers(layer_id)[0]
        shared_plan = self.prefill_plan
        if shared_plan is None:
            raise RuntimeError("DSA KPool prefill plan was not initialized")
        positions = shared_plan.positions
        if shared_plan.num_prefill_tokens != num_prefill_tokens:
            raise RuntimeError(
                "DSA KPool plan token count mismatch: "
                f"plan={shared_plan.num_prefill_tokens}, tokens={num_prefill_tokens}"
            )
        if positions.numel() != query.shape[0]:
            raise RuntimeError(
                "DSA KPool plan capacity mismatch: "
                f"plan={positions.numel()}, query_rows={query.shape[0]}"
            )
        selected_indices, selected_lens = kpool_prefill_topk(
            query.contiguous(),
            index_cache,
            weights,
            positions,
            shared_plan.query_start_loc,
            history_table,
            history_table,
            pool_size=self.pool_size,
            page_size=index_cache.shape[1],
            kv_page_size=ctx.token_to_kv_pool.arena.kv_page_size,
            topk_pools=self.index_topk // self.pool_size,
            softmax_scale=softmax_scale,
            prepared_query=prepared_query,
            req_ids=shared_plan.req_ids,
            causal_lens=shared_plan.causal_lens,
            pool_workspace_slots=shared_plan.pool_workspace_slots,
            row_starts=shared_plan.row_starts,
            row_ends=shared_plan.row_ends,
            max_num_pools=shared_plan.max_num_pools,
            max_logits_bytes=max(
                1,
                int(
                    global_server_args_dict["deepseek_v4_indexer_prefill_max_logits_mb"]
                ),
            )
            * 1024
            * 1024,
        )
        workspace_indices = torch.arange(
            selected_indices.numel(),
            dtype=torch.int32,
            device=selected_indices.device,
        ).view_as(selected_indices)
        workspace_indices.masked_fill_(selected_indices < 0, -1)

        return KPoolPrefillTopK(
            workspace_indices=workspace_indices,
            topk_lens=selected_lens,
            page_table=history_table,
            seq_lens=seq_lens,
            kv_seq_lens=shared_plan.causal_lens,
            max_seq_len=max_seq_len,
            kv_workspace_slots=selected_indices.reshape(-1).contiguous(),
        )
