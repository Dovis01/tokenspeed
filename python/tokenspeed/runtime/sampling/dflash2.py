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

"""Lossless DFlash2 rejection sampling from a sparse q distribution."""

from __future__ import annotations

import torch

from tokenspeed.runtime.sampling.draft_distribution import SparseDraftDistribution


def verify_sparse_draft_distribution(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    draft_distribution: SparseDraftDistribution,
    acceptance_coins: torch.Tensor,
    final_coins: torch.Tensor,
) -> None:
    """Losslessly verify a candidate chain against sparse DFlash2 q.

    ``candidates`` and ``target_probs`` use TokenSpeed's anchor-plus-drafts
    layout ``[B, N]`` and ``[B, N, V]``. The first candidate is the verified
    anchor. ``draft_distribution`` holds q for candidates 1..N-1. Accepted
    drafts use ``min(1, p(token) / q(token))``; the first rejection is sampled
    from normalized ``relu(p - q)``. If every draft is accepted, the final
    token is sampled from the target's bonus row.

    All control flow is static in ``N`` and all random values arrive through
    persistent buffers, so the operation is CUDA-graph capturable.
    """

    batch_size, num_tokens = candidates.shape
    num_steps = num_tokens - 1
    if draft_distribution.candidate_ids.shape[:2] != (batch_size, num_steps):
        raise ValueError("DFlash2 sparse candidate shape does not match verify width")
    if draft_distribution.probabilities.shape != draft_distribution.candidate_ids.shape:
        raise ValueError("DFlash2 sparse candidate IDs and probabilities must align")
    if target_probs.shape[:2] != (batch_size, num_tokens):
        raise ValueError("DFlash2 target probability shape does not match candidates")

    predict_rows = predicts.view(batch_size, num_tokens)
    predict_rows.zero_()
    accept_index.fill_(-1)
    accept_token_num.zero_()

    row_ids = torch.arange(batch_size, device=candidates.device, dtype=torch.int64)
    flat_base = row_ids * num_tokens
    accept_index[:, 0].copy_(flat_base.to(accept_index.dtype))
    alive = torch.ones(batch_size, dtype=torch.bool, device=candidates.device)

    sparse_ids = draft_distribution.candidate_ids.to(torch.int64)
    sparse_probs = draft_distribution.probabilities.float()
    eps = torch.finfo(torch.float32).tiny

    for step in range(num_steps):
        proposed = candidates[:, step + 1].to(torch.int64)
        p_token = target_probs[:, step].gather(1, proposed[:, None]).squeeze(1)
        q_token = (
            sparse_probs[:, step]
            * (sparse_ids[:, step] == proposed[:, None]).to(torch.float32)
        ).sum(dim=-1)
        ratio = (p_token / q_token.clamp_min(eps)).clamp(max=1.0)
        accepted = alive & (acceptance_coins[:, step] <= ratio)
        predict_rows[:, step].copy_(
            torch.where(accepted, proposed, torch.zeros_like(proposed)).to(
                predict_rows.dtype
            )
        )
        accept_index[:, step + 1].copy_(
            torch.where(
                accepted,
                flat_base + step + 1,
                torch.full_like(flat_base, -1),
            ).to(accept_index.dtype)
        )
        accept_token_num.add_(accepted.to(accept_token_num.dtype))
        alive = alive & accepted

    final_position = accept_token_num.to(torch.int64)
    target_final = target_probs[row_ids, final_position]
    rejected = ~alive
    q_position = final_position.clamp(max=max(num_steps - 1, 0))
    q_ids = sparse_ids[row_ids, q_position]
    q_values = sparse_probs[row_ids, q_position] * rejected[:, None]

    residual = target_final.clone()
    residual.scatter_add_(1, q_ids, -q_values)
    residual.clamp_min_(0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = torch.where(residual_mass > 0, residual, target_final)
    cumulative = residual.cumsum(dim=-1)
    cutoff = final_coins[:, None] * cumulative[:, -1:]
    sampled = (cumulative < cutoff).sum(dim=-1).clamp(max=target_probs.shape[-1] - 1)
    predict_rows.scatter_(
        1, final_position[:, None], sampled[:, None].to(predict_rows.dtype)
    )
