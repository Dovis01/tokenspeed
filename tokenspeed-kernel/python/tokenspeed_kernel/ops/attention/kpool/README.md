# KPool ops

KPool is the compressed sparse-attention index used by GLM-5.3-Flash's DSA
layers: every `pool_size` consecutive raw index keys of a request are pooled
into one FP8 row (softmax over the learned per-channel gate plus intra-pool
bias, Hadamard rotation, per-row absmax scale) and the indexer selects top-k
pooled rows instead of raw tokens. Incomplete pools live in a request-local
tail ring until the pool closes.

The GLM-5.3-Flash specialization has fixed geometry:

- 32 query heads with head dimension 128
- 4 raw tokens per compressed pool
- 16 compressed rows per index-cache page
- 512 selected pools
- BF16 queries, signed BF16 or FP32 head weights, and scaled FP8 E4M3 keys
- weighted per-head ReLU scoring and global FlatKV-slot output

## Ops

| Op | Role |
|---|---|
| `kpool_prefill_compress` | One launch per prefill chunk: gathers each completed pool's slots from the tail ring (`n_from_tail` leading slots) or the chunk (`chunk_src` onward), runs a single-pass online softmax, rotates, quantizes and writes the row to `write_slots`. |
| `kpool_prefill_write` | Same compression for pre-assembled `[rows, pool_size, head_dim]` slots; the reference for the fused path and the two-step fallback. |
| `kpool_prefill_tail_write` | Copies each request's incomplete trailing pool into its tail ring. |
| `kpool_decode_append` | Appends a decode/verify window to the tail ring; the pool row is compressed and written only when the window closes a pool. |
| `kpool_prefill_prepare_query` | Query-side inputs of the planned prefill top-k that depend only on the indexer projections: FP8 queries plus head weights with the query scale and softmax scale folded in for logits-based solutions, `None` for solutions that score BF16 queries directly. |
| `kpool_prefill_topk` / `kpool_decode_topk` | Score the pooled cache and expand the selected pools to FlatKV slots. |

## Solutions

* `triton.py` registers portable implementations of every op (all vendors).
* `deep_gemm.py` registers the Hopper+ prefill selection (`fp8_mqa_logits`
  scoring, TRT-LLM top-k) and its matching `kpool_prefill_prepare_query`.
  Both share one capability, signature and trait set so selection resolves
  them together: whatever `kpool_prefill_prepare_query` returns is exactly
  what the selected `kpool_prefill_topk` consumes through `prepared_query`.
* `gluon.py` registers the gfx950 / gfx1250 prefill selection; it scores BF16
  queries and ignores `prepared_query`.

## Prefill contract

The runtime builds one layer-invariant plan per forward (`KPoolPrefillPlan`)
and every DSA layer issues: `kpool_prefill_compress` + `kpool_prefill_tail_write`
(cache side, only `key`/`gate` inputs), `kpool_prefill_prepare_query`
(query side, only `query`/`weights` inputs), then `kpool_prefill_topk`, which
reads the freshly written pools. The two halves have no data dependency on
each other, so a caller may run them on different streams and join before the
top-k.

All compression kernels share `_kpool_online_softmax_step` and
`_kpool_quantize`, so the fused prefill path, the two-step path and the decode
append produce bit-identical rows for identical inputs.

## Prefill selection

The hybrid Gluon implementation keeps one orchestration path across AMD
architectures. The architecture backend supplies three stages:

1. Score a bounded pool window through either the request page table or the
   precomputed physical-slot plan.
2. Select logical columns with the architecture's radix top-k for long or
   merged windows.
3. Reuse the portable Triton payload gather, deterministic short-window sort,
   and pool-to-FlatKV expansion.

GFX950 uses Wave64 MFMA scoring and its logical radix selector. GFX1250 uses
Wave32 WMMA-v3 scoring and its existing Wave32 radix selector. Short
single-window rows of at most 2048 pools fold signed head contributions in
logical head order before deterministic sorting, preserving stable pool IDs
for equal scores.

Production prefill supplies `pool_workspace_slots`, `row_starts`, and
`row_ends`. These tensors preserve request-local logical pool order while
addressing physical cache rows directly. The eager-only compatibility path
reconstructs request IDs and causal lengths and scores through the index page
table.

Scoring workspaces are row-tiled under `max_logits_bytes`. The cap includes
the persistent sort or radix intermediates as well as logits; one row remains
legal when its workspace exceeds the cap.
