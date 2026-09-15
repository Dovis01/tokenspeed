# KPool ops

KPool is the compressed sparse-attention index used by GLM-5.3-Flash's DSA
layers: every `pool_size` consecutive raw index keys of a request are pooled
into one FP8 row (softmax over the learned per-channel gate plus intra-pool
bias, Hadamard rotation, per-row absmax scale) and the indexer selects top-k
pooled rows instead of raw tokens. Incomplete pools live in a request-local
tail ring until the pool closes.

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
* `gluon.py` registers the gfx950 prefill selection; it scores BF16 queries
  and ignores `prepared_query`.

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
