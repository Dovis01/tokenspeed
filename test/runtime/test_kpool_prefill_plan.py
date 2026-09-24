"""KPool prefill planning stays host-vectorized and feeds fused cache writes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention import kpool as kpool_runtime
from tokenspeed.runtime.layers.attention.kpool import (
    KPoolRuntime,
    build_kpool_prefill_plan,
    build_kpool_write_plan,
)

_HEAD_DIM = 128


def _reference_prefill_plan(
    starts: list[int],
    lengths: list[int],
    index_block_table: torch.Tensor,
    request_slots: torch.Tensor,
    kpool: int,
    rows_per_page: int,
    token_capacity: int | None,
) -> dict[str, list[int] | int]:
    """Per-token Python loop the vectorized planner must reproduce exactly."""
    plan: dict[str, list[int] | int] = {
        key: []
        for key in (
            "positions",
            "req_ids",
            "causal_lens",
            "row_starts",
            "row_ends",
            "pool_workspace_slots",
            "pool_request_slots",
            "pool_n_from_tail",
            "pool_chunk_src",
            "pool_tail_logical_base",
            "pool_write_slots",
            "tail_request_slots",
            "tail_chunk_src",
            "tail_dst_positions",
            "tail_write_counts",
        )
    }
    workspace_start = 0
    chunk_begin = 0
    max_num_pools = 0
    for req_id, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        final_num_pools = (start + length) // kpool
        max_num_pools = max(max_num_pools, final_num_pools)
        for pool_id in range(final_num_pools):
            page = int(index_block_table[req_id, pool_id // rows_per_page])
            plan["pool_workspace_slots"].append(
                page * rows_per_page + pool_id % rows_per_page
            )
        for offset in range(length):
            causal = start + offset + 1
            plan["positions"].append(causal - 1)
            plan["req_ids"].append(req_id)
            plan["causal_lens"].append(causal)
            plan["row_starts"].append(workspace_start)
            plan["row_ends"].append(workspace_start + causal // kpool)
        workspace_start += final_num_pools

        if length > 0:
            first_slot = start % kpool
            base_pool = start // kpool
            num_pools = (start + length) // kpool - base_pool
            consumed = num_pools * kpool - first_slot if num_pools else 0
            for pool_index in range(num_pools):
                pool_id = base_pool + pool_index
                page = int(index_block_table[req_id, pool_id // rows_per_page])
                plan["pool_request_slots"].append(int(request_slots[req_id]))
                plan["pool_n_from_tail"].append(first_slot if pool_index == 0 else 0)
                plan["pool_chunk_src"].append(
                    chunk_begin
                    + (
                        0
                        if pool_index == 0
                        else (kpool - first_slot) + (pool_index - 1) * kpool
                    )
                )
                plan["pool_tail_logical_base"].append(pool_id * kpool)
                plan["pool_write_slots"].append(
                    page * rows_per_page + pool_id % rows_per_page
                )
            tail_count = length - consumed
            if tail_count:
                plan["tail_request_slots"].append(int(request_slots[req_id]))
                plan["tail_chunk_src"].append(chunk_begin + length - tail_count)
                plan["tail_dst_positions"].append(start + consumed)
                plan["tail_write_counts"].append(tail_count)
        chunk_begin += length

    num_prefill_tokens = chunk_begin
    padding = 0 if token_capacity is None else token_capacity - num_prefill_tokens
    for key in ("positions", "req_ids", "causal_lens", "row_starts", "row_ends"):
        plan[key].extend([0] * padding)
    plan["max_num_pools"] = max_num_pools
    plan["num_prefill_tokens"] = num_prefill_tokens
    return plan


_CASES = [
    pytest.param([70, 5], [70, 133], 4, 16, None, id="mid-pool-prefixes"),
    pytest.param([0, 64, 129], [1, 64, 300], 4, 16, None, id="aligned-and-odd"),
    pytest.param([12, 3, 40], [0, 9, 0], 4, 16, None, id="zero-length-extends"),
    pytest.param([4096], [4096 + 128], 4, 16, 8192, id="long-chunk-with-padding"),
    pytest.param([5, 33], [18, 47], 16, 4, 70, id="pool-16"),
    pytest.param([], [], 4, 16, 3, id="empty-batch-padding"),
]


@pytest.mark.parametrize(
    ("starts", "lengths", "kpool", "rows_per_page", "token_capacity"), _CASES
)
def test_prefill_plan_matches_per_token_reference(
    starts: list[int],
    lengths: list[int],
    kpool: int,
    rows_per_page: int,
    token_capacity: int | None,
) -> None:
    requests = len(starts)
    columns = max(
        (
            (start + length) // kpool // rows_per_page + 1
            for start, length in zip(starts, lengths, strict=True)
        ),
        default=1,
    )
    generator = torch.Generator().manual_seed(requests + kpool)
    index_block_table = torch.randint(
        1, 500, (requests, columns), dtype=torch.int32, generator=generator
    )
    request_slots = torch.randint(
        1, 64, (requests,), dtype=torch.int64, generator=generator
    )
    expected = _reference_prefill_plan(
        starts,
        lengths,
        index_block_table,
        request_slots,
        kpool,
        rows_per_page,
        token_capacity,
    )

    plan = build_kpool_prefill_plan(
        prefix_lens_cpu=torch.tensor(starts, dtype=torch.int64),
        extend_lens_cpu=torch.tensor(lengths, dtype=torch.int32),
        index_block_table=index_block_table,
        request_slots=request_slots,
        kpool=kpool,
        index_rows_per_page=rows_per_page,
        token_capacity=token_capacity,
    )

    assert plan.num_prefill_tokens == expected["num_prefill_tokens"]
    assert plan.max_num_pools == expected["max_num_pools"]
    assert plan.query_start_loc.tolist() == [
        0,
        *torch.cumsum(torch.tensor(lengths, dtype=torch.int64), 0).tolist(),
    ]
    for name in ("positions", "req_ids", "causal_lens", "row_starts", "row_ends"):
        assert getattr(plan, name).tolist() == expected[name], name
        assert getattr(plan, name).dtype == torch.int32, name
    assert plan.pool_workspace_slots.tolist() == expected["pool_workspace_slots"]
    write = plan.write
    for name in (
        "pool_request_slots",
        "pool_n_from_tail",
        "pool_chunk_src",
        "pool_tail_logical_base",
        "pool_write_slots",
        "tail_request_slots",
        "tail_chunk_src",
        "tail_dst_positions",
        "tail_write_counts",
    ):
        assert getattr(write, name).tolist() == expected[name], name
    assert write.pool_n_from_tail.dtype == torch.int32
    assert write.tail_write_counts.dtype == torch.int32
    assert write.pool_write_slots.dtype == torch.int64


def test_write_plan_matches_prefill_plan_write_half() -> None:
    starts = [70, 5, 8]
    lengths = [70, 133, 0]
    index_block_table = torch.arange(1, 3 * 4 + 1, dtype=torch.int32).view(3, 4)
    request_slots = torch.tensor([3, 0, 7], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 70, 203, 203], dtype=torch.int64)

    standalone = build_kpool_write_plan(
        req_start_positions=torch.tensor(starts, dtype=torch.int64),
        query_start_loc=query_start_loc,
        index_block_table=index_block_table,
        request_slots=request_slots,
        kpool=4,
        index_rows_per_page=16,
    )
    combined = build_kpool_prefill_plan(
        prefix_lens_cpu=torch.tensor(starts, dtype=torch.int64),
        extend_lens_cpu=torch.tensor(lengths, dtype=torch.int64),
        index_block_table=index_block_table,
        request_slots=request_slots,
        kpool=4,
        index_rows_per_page=16,
        token_capacity=None,
    ).write

    for name in standalone.__dataclass_fields__:
        assert torch.equal(getattr(standalone, name), getattr(combined, name)), name


def test_prefill_plan_uploads_host_metadata_once(monkeypatch) -> None:
    uploads: list[int] = []
    real_upload = kpool_runtime.upload_packed

    def counting_upload(parts, device):
        uploads.append(len(parts))
        return real_upload(parts, device)

    monkeypatch.setattr(kpool_runtime, "upload_packed", counting_upload)
    build_kpool_prefill_plan(
        prefix_lens_cpu=torch.tensor([70, 5], dtype=torch.int64),
        extend_lens_cpu=torch.tensor([70, 133], dtype=torch.int64),
        index_block_table=torch.arange(1, 9, dtype=torch.int32).view(2, 4),
        request_slots=torch.tensor([3, 0], dtype=torch.int64),
        kpool=4,
        index_rows_per_page=16,
        token_capacity=256,
    )

    assert uploads == [17]


def test_prefill_plan_rejects_device_length_mirrors() -> None:
    lengths = torch.tensor([4], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="CPU length mirrors"):
        build_kpool_prefill_plan(
            prefix_lens_cpu=SimpleNamespace(
                device=SimpleNamespace(type="cuda"), reshape=lambda *_: lengths
            ),
            extend_lens_cpu=lengths,
            index_block_table=torch.zeros((1, 1), dtype=torch.int32),
            request_slots=torch.zeros(1, dtype=torch.int64),
            kpool=4,
            index_rows_per_page=16,
            token_capacity=None,
        )


def test_prefill_plan_rejects_negative_lengths() -> None:
    with pytest.raises(ValueError, match="request 1"):
        build_kpool_prefill_plan(
            prefix_lens_cpu=torch.tensor([4, -1], dtype=torch.int64),
            extend_lens_cpu=torch.tensor([4, 4], dtype=torch.int64),
            index_block_table=torch.zeros((2, 1), dtype=torch.int32),
            request_slots=torch.zeros(2, dtype=torch.int64),
            kpool=4,
            index_rows_per_page=16,
            token_capacity=None,
        )


def _fake_forward(
    starts: list[int], lengths: list[int], token_capacity: int
) -> tuple[KPoolRuntime, SimpleNamespace, SimpleNamespace, tuple[torch.Tensor, ...]]:
    requests = len(starts)
    index_cache = torch.zeros((8, 16, _HEAD_DIM + 4), dtype=torch.uint8)
    tail_k = torch.zeros((requests + 2, 4, _HEAD_DIM), dtype=torch.bfloat16)
    tail_gate = torch.zeros_like(tail_k)
    views = (
        torch.zeros((8, 16, _HEAD_DIM), dtype=torch.float8_e4m3fn),
        torch.zeros((8, 16, 1), dtype=torch.float32),
    )
    pool = SimpleNamespace(
        get_kpool_buffers=lambda _layer_id: (index_cache, tail_k, tail_gate),
        index_k_block_views=lambda _buf: views,
        arena=SimpleNamespace(kv_page_size=64),
    )
    ctx = SimpleNamespace(
        num_extends=requests,
        token_to_kv_pool=pool,
        attn_backend=SimpleNamespace(max_context_len=4096),
    )
    backend = SimpleNamespace(
        chunked_prefill_metadata=SimpleNamespace(
            extend_prefix_lens_cpu=torch.tensor(starts, dtype=torch.int64),
            extend_seq_lens_cpu=torch.tensor(lengths, dtype=torch.int64),
        ),
        kpool_prefill_page_table=lambda n: torch.arange(
            1, 4 * n + 1, dtype=torch.int32
        ).view(n, 4),
    )
    runtime = KPoolRuntime(pool_size=4, index_topk=2048)
    runtime.reset_forward(torch.arange(1, requests + 1, dtype=torch.int64))
    runtime.ensure_prefill_plan(ctx, backend, 3, token_capacity=token_capacity)
    return runtime, ctx, backend, (tail_k, tail_gate, *views)


def test_write_prefill_issues_one_compress_and_one_tail_launch(monkeypatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(
        kpool_runtime,
        "kpool_prefill_compress",
        lambda *args, **kwargs: calls.append(("compress", args, kwargs)),
    )
    monkeypatch.setattr(
        kpool_runtime,
        "kpool_prefill_tail_write",
        lambda *args, **kwargs: calls.append(("tail", args, kwargs)),
    )
    runtime, ctx, backend, (tail_k, tail_gate, values, scales) = _fake_forward(
        [70, 5], [70, 133], 210
    )
    key = torch.zeros((210, _HEAD_DIM), dtype=torch.bfloat16)
    gate = torch.zeros_like(key)
    ape = torch.zeros((4, _HEAD_DIM), dtype=torch.float32)

    runtime.write_prefill(
        key=key, gate=gate, compress_ape=ape, ctx=ctx, backend=backend, layer_id=3
    )

    plan = runtime.prefill_plan.write
    assert [name for name, _, _ in calls] == ["compress", "tail"]
    compress_args = calls[0][1]
    assert compress_args[0] is key and compress_args[1] is gate
    assert compress_args[2] is tail_k and compress_args[3] is tail_gate
    assert compress_args[4] is plan.pool_request_slots
    assert compress_args[5] is plan.pool_n_from_tail
    assert compress_args[6] is plan.pool_chunk_src
    assert compress_args[7] is plan.pool_tail_logical_base
    assert compress_args[8] is plan.pool_write_slots
    assert compress_args[9] is values and compress_args[10] is scales
    assert compress_args[11] is ape
    tail_args, tail_kwargs = calls[1][1], calls[1][2]
    assert tail_args[4] is plan.tail_chunk_src
    assert tail_args[5] is plan.tail_request_slots
    assert tail_args[6] is plan.tail_dst_positions
    assert tail_args[7] is plan.tail_write_counts
    assert tail_kwargs == {"pool_size": 4}


def test_write_prefill_skips_launches_without_rows(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        kpool_runtime, "kpool_prefill_compress", lambda *a, **k: calls.append("c")
    )
    monkeypatch.setattr(
        kpool_runtime, "kpool_prefill_tail_write", lambda *a, **k: calls.append("t")
    )
    runtime, ctx, backend, _ = _fake_forward([4, 8], [4, 8], 12)
    key = torch.zeros((12, _HEAD_DIM), dtype=torch.bfloat16)

    runtime.write_prefill(
        key=key,
        gate=torch.zeros_like(key),
        compress_ape=torch.zeros((4, _HEAD_DIM), dtype=torch.float32),
        ctx=ctx,
        backend=backend,
        layer_id=3,
    )

    assert calls == ["c"]


def test_prepare_prefill_query_selects_with_planned_topk_traits(monkeypatch) -> None:
    captured = {}

    def fake_prepare(q, cache, weights, **kwargs):
        captured["q"] = q
        captured["cache"] = cache
        captured["weights"] = weights
        captured["kwargs"] = kwargs
        return "prepared"

    monkeypatch.setattr(kpool_runtime, "kpool_prefill_prepare_query", fake_prepare)
    runtime, ctx, _, _ = _fake_forward([70], [70], 70)
    query = torch.zeros((70, 2, _HEAD_DIM), dtype=torch.bfloat16)
    weights = torch.zeros((70, 2), dtype=torch.bfloat16)

    prepared = runtime.prepare_prefill_query(
        query=query, weights=weights, softmax_scale=0.25, ctx=ctx, layer_id=3
    )

    assert prepared == "prepared"
    assert captured["q"] is query and captured["weights"] is weights
    assert captured["cache"] is ctx.token_to_kv_pool.get_kpool_buffers(3)[0]
    assert captured["kwargs"] == {
        "pool_size": 4,
        "page_size": 16,
        "topk_pools": 512,
        "softmax_scale": 0.25,
        "apply_relu": True,
    }


def test_select_prefill_forwards_prepared_query(monkeypatch) -> None:
    captured = {}

    def fake_topk(*args, **kwargs):
        captured["kwargs"] = kwargs
        tokens = args[0].shape[0]
        return (
            torch.full((tokens, 3), -1, dtype=torch.int32),
            torch.zeros(tokens, dtype=torch.int32),
        )

    monkeypatch.setattr(kpool_runtime, "kpool_prefill_topk", fake_topk)
    monkeypatch.setitem(
        kpool_runtime.global_server_args_dict,
        "deepseek_v4_indexer_prefill_max_logits_mb",
        1,
    )
    runtime, ctx, backend, _ = _fake_forward([70], [70], 70)
    backend.chunked_prefill_metadata.extend_prefix_lens = torch.tensor([70])
    backend.chunked_prefill_metadata.extend_seq_lens = torch.tensor([70])
    prepared = object()

    selected = runtime.select_prefill(
        query=torch.zeros((70, 2, _HEAD_DIM), dtype=torch.bfloat16),
        weights=torch.zeros((70, 2), dtype=torch.bfloat16),
        softmax_scale=0.25,
        prepared_query=prepared,
        ctx=ctx,
        backend=backend,
        layer_id=3,
        num_prefill_tokens=70,
    )

    assert captured["kwargs"]["prepared_query"] is prepared
    assert captured["kwargs"]["max_num_pools"] == 35
    assert selected is not None
    assert selected.kv_workspace_slots.tolist() == [-1] * (70 * 3)
