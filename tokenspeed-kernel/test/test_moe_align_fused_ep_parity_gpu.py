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

"""GPU parity for the expert-parallel fused MoE block-align on gfx950.

The alignment is only defined up to the order of routes *within* a block --
the grouped GEMM derives every destination row from the packed ``(slot, token)``
payload, never from the row's position -- so parity is checked per expert block
as a multiset of ``(packed_id, weight)`` pairs.
"""

from __future__ import annotations

import pytest
import torch


def _is_gfx950() -> bool:
    if not (torch.cuda.is_available() and torch.version.hip is not None):
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(not _is_gfx950(), reason="requires AMD gfx950")


def _routes(num_tokens, top_k, num_experts, expert_start, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # Top-k holds every expert at most once per token, which is the
    # precondition of the single-block collapse. Sample without replacement.
    scores = torch.rand((num_tokens, num_experts), device="cuda", generator=gen)
    ids = scores.topk(top_k, dim=-1).indices.to(torch.int32) + expert_start
    weights = torch.rand(
        (num_tokens, top_k), dtype=torch.float32, device="cuda", generator=gen
    )
    return ids.contiguous(), weights


def _blocks(sorted_ids, sorted_experts, sorted_weights, num_valid, block_m, sentinel):
    em = int(num_valid.item())
    out = {}
    for b in range(em // block_m):
        expert = int(sorted_experts[b].item())
        assert expert not in out, "an expert must own at most one block"
        rows = sorted_ids[b * block_m : (b + 1) * block_m].tolist()
        vals = sorted_weights[b * block_m : (b + 1) * block_m].tolist()
        out[expert] = sorted(
            (r, v) for r, v in zip(rows, vals) if (r & 0xFFFFFF) != sentinel
        )
    return out


@pytest.mark.parametrize(
    "num_tokens,top_k,num_experts,block_m,expert_start",
    [
        (32, 16, 112, 64, 0),  # the decode shape: 512 routes, EP=8 shard
        (32, 16, 112, 64, 224),  # global expert ids, remote routes dropped
        (1, 16, 112, 64, 0),
        (8, 16, 112, 64, 0),
        (16, 8, 112, 64, 0),
        (64, 16, 112, 128, 0),
        (32, 16, 896, 64, 0),  # num_experts > num_routes
        (2, 1, 112, 64, 0),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_ep_align_matches_device_align(
    num_tokens, top_k, num_experts, block_m, expert_start, seed
):
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
        moe_align_block_size_device,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_fused_ep import (
        moe_align_block_size_fused_ep,
    )

    ids, weights = _routes(num_tokens, top_k, num_experts, expert_start, seed)
    reference = moe_align_block_size_device(
        ids, weights, num_experts, block_m, expert_start=expert_start
    )
    actual = moe_align_block_size_fused_ep(
        ids, weights, num_experts, block_m, expert_start=expert_start
    )
    torch.cuda.synchronize()

    assert int(reference[3].item()) == int(actual[3].item())
    assert _blocks(*reference, block_m, num_tokens) == _blocks(
        *actual, block_m, num_tokens
    )


def test_fused_ep_align_rejects_large_m():
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_fused_ep import (
        moe_align_block_size_fused_ep,
    )

    ids, weights = _routes(128, 8, 112, 0, 0)
    with pytest.raises(ValueError):
        moe_align_block_size_fused_ep(ids, weights, 112, 64)
