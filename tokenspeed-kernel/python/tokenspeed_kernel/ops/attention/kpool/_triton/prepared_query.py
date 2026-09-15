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

"""Query-side inputs a KPool prefill top-k solution can consume pre-built."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KPoolPreparedQuery:
    """FP8 queries and combined head weights for logits-based selection.

    Attributes:
        q_fp8: Quantized queries shaped ``[tokens, heads, head_dim]``.
        scaled_weights: FP32 ``[tokens, heads]`` head weights with the query
            dequant scale and the softmax scale folded in.
    """

    q_fp8: torch.Tensor
    scaled_weights: torch.Tensor
