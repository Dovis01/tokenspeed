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

"""DFlash2 draft model using the official grouped-conv and selector nodes."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.models.dflash import (
    DFlashDecoderLayer,
    DFlashDraftModel,
)


def _dflash2_config_value(config: Any, key: str, default: Any = None) -> Any:
    nested = getattr(config, "dflash_config", {}) or {}
    return nested.get(key, getattr(config, key, default))


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """Apply DFlash2's grouped dynamic depthwise convolution to flat blocks."""
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output = output + coefficients[:, tap] * shifted * (position >= tap).view(
            -1, 1, 1
        )
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    """Official DFlash2 grouped convolution, kept as ordinary PyTorch nodes."""

    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        if taps < 1 or taps > block_size:
            raise ValueError(
                f"conv_kernel_size={taps} must be in [1, block_size={block_size}]."
            )
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, self.taps, hidden_size, dtype=params_dtype)
        )
        self.kernel_projection = nn.Linear(
            hidden_size,
            2 * self.taps * self.num_groups,
            bias=False,
            dtype=params_dtype,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


class CandidateSelector(nn.Module):
    """Score top-k token lattices with DFlash2's low-rank transition model."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"selector_rank must be positive, got {rank}.")
        if not 2 <= top_k <= vocab_size:
            raise ValueError(
                f"selector_top_k must be in [2, {vocab_size}], got {top_k}."
            )
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype)
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype)
        )
        self.hidden_projection = nn.Linear(
            hidden_size, rank, bias=False, dtype=params_dtype
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2DecoderLayer(DFlashDecoderLayer):
    """DFlash layer with dynamic convolutions around attention and the MLP."""

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
        )
        conv_args = dict(
            hidden_size=int(config.hidden_size),
            taps=int(_dflash2_config_value(config, "conv_kernel_size")),
            group_size=int(_dflash2_config_value(config, "conv_group_size")),
            block_size=int(_dflash2_config_value(config, "block_size")),
        )
        self.attention_conv = DFlashGroupedConv(**conv_args)
        self.mlp_conv = DFlashGroupedConv(**conv_args)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx.forward_mode.is_idle():
            return super().forward(
                positions, hidden_states, ctx, out_cache_loc, residual
            )

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
        )
        if self.mapping.attn.tp_size > 1:
            hidden_states = all_reduce(hidden_states, self.mapping.attn.tp_group)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mapping.dense.tp_size > 1:
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


class DFlash2DraftModel(DFlashDraftModel):
    """DFlash2 checkpoint entry class."""

    decoder_layer_cls = DFlash2DecoderLayer

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )
        dtype = self.fc.weight.dtype
        self.input_embedding_scale = float(
            _dflash2_config_value(config, "input_embedding_scale", 1.0)
        )
        self.block_size = int(_dflash2_config_value(config, "block_size"))
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            rank=int(_dflash2_config_value(config, "selector_rank")),
            top_k=int(_dflash2_config_value(config, "selector_top_k")),
            params_dtype=dtype,
        )

    @torch.no_grad()
    def forward(self, *args, input_embeds: torch.Tensor | None = None, **kwargs):
        if input_embeds is not None and self.input_embedding_scale != 1.0:
            input_embeds = input_embeds * self.input_embedding_scale
        return super().forward(*args, input_embeds=input_embeds, **kwargs)


EntryClass = [DFlash2DraftModel]
