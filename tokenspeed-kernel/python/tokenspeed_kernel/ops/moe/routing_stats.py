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

"""AMD routing-statistics kernel entry points used by the runtime profiler."""

from tokenspeed_kernel.platform import current_platform

if current_platform().is_cdna4:
    from tokenspeed_kernel_amd.ops.gfx950.moe.routing_stats import (
        ROUTING_STATS_FIELDS,
        accumulate_routing_stats,
        record_routing_counts,
    )
else:
    ROUTING_STATS_FIELDS = (
        "observations",
        "routes",
        "nonempty_blocks",
        "largest_block_sum",
        "largest_block_max",
        "busiest_rank_sum",
        "busiest_rank_max",
        "lightest_rank_sum",
        "lightest_rank_min",
        "rank_spread_sum",
        "rank_spread_max",
        "rank_squared_error_sum",
    )

    def _unsupported(*args, **kwargs):
        raise RuntimeError("MoE routing statistics require AMD CDNA4")

    accumulate_routing_stats = _unsupported
    record_routing_counts = _unsupported

__all__ = [
    "ROUTING_STATS_FIELDS",
    "accumulate_routing_stats",
    "record_routing_counts",
]
