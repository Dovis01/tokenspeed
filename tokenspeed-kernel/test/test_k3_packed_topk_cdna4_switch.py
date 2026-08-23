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

"""The CDNA4 packed-K3 top-k row cap is opt-in and defaults to the old value."""

from __future__ import annotations

import pytest

from tokenspeed_kernel.ops.moe.sigmoid_topk import (
    _K3_PACKED_TOPK_CDNA4_SWITCH,
    _K3_PACKED_TOPK_MAX_ROWS_CDNA4,
    _K3_PACKED_TOPK_MAX_ROWS_CDNA4_TUNED,
    _k3_packed_topk_max_rows_cdna4,
)


def test_default_is_the_untuned_cap(monkeypatch):
    monkeypatch.delenv(_K3_PACKED_TOPK_CDNA4_SWITCH, raising=False)
    assert _k3_packed_topk_max_rows_cdna4() == _K3_PACKED_TOPK_MAX_ROWS_CDNA4


@pytest.mark.parametrize("value", ["0", "", "off", "no", "false", "2"])
def test_unset_like_values_keep_the_default(monkeypatch, value):
    monkeypatch.setenv(_K3_PACKED_TOPK_CDNA4_SWITCH, value)
    assert _k3_packed_topk_max_rows_cdna4() == _K3_PACKED_TOPK_MAX_ROWS_CDNA4


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "On"])
def test_switch_raises_the_cap(monkeypatch, value):
    monkeypatch.setenv(_K3_PACKED_TOPK_CDNA4_SWITCH, value)
    assert _k3_packed_topk_max_rows_cdna4() == _K3_PACKED_TOPK_MAX_ROWS_CDNA4_TUNED
    assert _K3_PACKED_TOPK_MAX_ROWS_CDNA4_TUNED > _K3_PACKED_TOPK_MAX_ROWS_CDNA4
