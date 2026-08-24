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

"""Graph-safe online statistics for expert-parallel routing metadata."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from tokenspeed_kernel.ops.moe.routing_stats import (
    ROUTING_STATS_FIELDS,
    accumulate_routing_stats,
    record_routing_counts,
)

from tokenspeed.runtime.moe.distribution_recorder import ExpertDistributionRecorder
from tokenspeed.runtime.utils import Withable

logger = logging.getLogger(__name__)

_INT32_MAX = 2**31 - 1


def _divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _histogram_quantile(histogram: list[int], quantile: float) -> int | None:
    total = sum(histogram[1:-1])
    if total == 0:
        return None
    target = math.ceil(total * quantile)
    cumulative = 0
    for occupancy, frequency in enumerate(histogram[1:-1], start=1):
        cumulative += frequency
        if cumulative >= target:
            return occupancy
    return len(histogram) - 2


def _summarize_aggregate(
    values: dict[str, int],
    histogram: list[int],
    *,
    num_experts: int,
    num_ranks: int,
    num_routes: int,
    block_size: int,
) -> dict[str, Any]:
    observations = values["observations"]
    active_blocks = values["nonempty_blocks"]
    routes = values["routes"]
    ideal_rank_routes = _divide(num_routes, num_ranks)
    histogram_total = sum(histogram[1:-1])

    occupancy_fractions = {}
    for threshold in (1, 2, 4, 8, 16):
        if threshold >= len(histogram) - 1:
            continue
        occupancy_fractions[f"at_most_{threshold}"] = _divide(
            sum(histogram[1 : threshold + 1]), histogram_total
        )

    return {
        "observations": observations,
        "routes": routes,
        "route_conservation_ok": routes == observations * num_routes,
        "average_nonempty_block_size": _divide(routes, active_blocks),
        "average_nonempty_block_utilization": _divide(
            routes, active_blocks * block_size
        ),
        "average_nonempty_blocks": _divide(active_blocks, observations),
        "average_nonempty_expert_fraction": _divide(
            active_blocks, observations * num_experts
        ),
        "all_expert_slot_utilization": _divide(
            routes, observations * num_experts * block_size
        ),
        "largest_block_mean": _divide(values["largest_block_sum"], observations),
        "largest_block_max": values["largest_block_max"],
        "busiest_rank_routes_mean": _divide(values["busiest_rank_sum"], observations),
        "busiest_rank_excess_over_uniform_mean": _divide(
            values["busiest_rank_sum"], observations
        )
        - ideal_rank_routes,
        "busiest_rank_excess_over_uniform_max": values["busiest_rank_max"]
        - ideal_rank_routes,
        "lightest_rank_routes_mean": _divide(values["lightest_rank_sum"], observations),
        "lightest_rank_routes_min": values["lightest_rank_min"],
        "rank_spread_mean": _divide(values["rank_spread_sum"], observations),
        "rank_spread_max": values["rank_spread_max"],
        "rank_load_rms_error_from_uniform": math.sqrt(
            _divide(values["rank_squared_error_sum"], observations * num_ranks)
        ),
        "block_occupancy_quantiles": {
            "p50": _histogram_quantile(histogram, 0.50),
            "p90": _histogram_quantile(histogram, 0.90),
            "p95": _histogram_quantile(histogram, 0.95),
            "p99": _histogram_quantile(histogram, 0.99),
        },
        "block_occupancy_fractions": occupancy_fractions,
        "block_occupancy_overflow": histogram[-1],
        "block_occupancy_conservation_ok": histogram_total == active_blocks,
    }


def summarize_routing_stats(
    aggregate: list[list[int]],
    occupancy_histogram: list[list[int]],
    *,
    layer_ids: tuple[int, ...],
    num_experts: int,
    num_ranks: int,
    num_routes: int,
    block_size: int,
) -> dict[str, Any]:
    """Convert exact integer accumulators into per-layer and global metrics."""
    if len(aggregate) != len(layer_ids) or len(occupancy_histogram) != len(layer_ids):
        raise ValueError("routing-stat rows must match layer_ids")

    field_count = len(ROUTING_STATS_FIELDS)
    combined_values = {field: 0 for field in ROUTING_STATS_FIELDS}
    combined_histogram = [0] * (len(occupancy_histogram[0]) if layer_ids else 0)
    layer_summaries = []

    for layer_number, (layer_id, row, histogram) in enumerate(
        zip(layer_ids, aggregate, occupancy_histogram, strict=True)
    ):
        if len(row) != field_count:
            raise ValueError("routing-stat aggregate has the wrong field count")
        values = dict(zip(ROUTING_STATS_FIELDS, row, strict=True))
        for field in ROUTING_STATS_FIELDS:
            if field in {"largest_block_max", "busiest_rank_max", "rank_spread_max"}:
                combined_values[field] = max(combined_values[field], values[field])
            elif field == "lightest_rank_min":
                if layer_number == 0:
                    combined_values[field] = values[field]
                else:
                    combined_values[field] = min(combined_values[field], values[field])
            else:
                combined_values[field] += values[field]
        combined_histogram = [
            current + value
            for current, value in zip(combined_histogram, histogram, strict=True)
        ]
        layer_summaries.append(
            {
                "layer": layer_id,
                **_summarize_aggregate(
                    values,
                    histogram,
                    num_experts=num_experts,
                    num_ranks=num_ranks,
                    num_routes=num_routes,
                    block_size=block_size,
                ),
                "block_occupancy_histogram": histogram,
            }
        )

    observations = [row[0] for row in aggregate]
    return {
        "decode_iterations_min": min(observations, default=0),
        "decode_iterations_max": max(observations, default=0),
        "overall": {
            **_summarize_aggregate(
                combined_values,
                combined_histogram,
                num_experts=num_experts,
                num_ranks=num_ranks,
                num_routes=num_routes,
                block_size=block_size,
            ),
            "block_occupancy_histogram": combined_histogram,
        },
        "layers": layer_summaries,
    }


class MoERoutingStatsRecorder(ExpertDistributionRecorder):
    """Collect exact routing occupancy from one fixed decode batch shape."""

    def __init__(
        self,
        *,
        output_dir: str,
        rank: int,
        layer_ids: tuple[int, ...],
        num_experts: int,
        top_k: int,
        ep_size: int,
        batch_size: int,
        block_size: int,
        max_replays: int,
        device: torch.device | str = "cuda",
    ) -> None:
        if not layer_ids:
            raise ValueError("routing statistics require at least one MoE layer")
        if ep_size <= 0:
            raise ValueError("ep_size must be positive")
        if num_experts <= 0 or num_experts % ep_size:
            raise ValueError("num_experts must be positive and divisible by ep_size")
        if top_k <= 0 or batch_size <= 0 or block_size <= 0:
            raise ValueError("top_k, batch_size, and block_size must be positive")
        if batch_size > block_size:
            raise ValueError("routing-stat batch_size must not exceed block_size")
        if max_replays <= 0:
            raise ValueError("routing-stat max_replays must be positive")
        if batch_size * top_k % ep_size:
            raise ValueError("routes per layer must be divisible by ep_size")

        num_routes = batch_size * top_k
        ideal_rank_routes = num_routes // ep_size
        worst_rank_squared_error = (num_routes - ideal_rank_routes) ** 2 + (
            ep_size - 1
        ) * ideal_rank_routes**2
        largest_accumulator_increment = max(
            num_routes,
            num_experts,
            worst_rank_squared_error,
        )
        max_safe_replays = _INT32_MAX // largest_accumulator_increment
        if max_replays > max_safe_replays:
            raise ValueError(
                "routing-stat max_replays exceeds the int32 accumulator limit "
                f"({max_safe_replays} for this configuration)"
            )

        self.output_dir = Path(output_dir).expanduser()
        self.rank = rank
        self.layer_ids = layer_ids
        self.layer_to_row = {layer_id: row for row, layer_id in enumerate(layer_ids)}
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_replays = max_replays
        self.num_routes = num_routes
        self._current_layer = Withable()
        self._counts = torch.zeros(
            (len(layer_ids), num_experts), dtype=torch.int32, device=device
        )
        self._aggregate = torch.zeros(
            (len(layer_ids), len(ROUTING_STATS_FIELDS)),
            dtype=torch.int32,
            device=device,
        )
        self._expert_totals = torch.zeros_like(self._counts)
        self._occupancy_histogram = torch.zeros(
            (len(layer_ids), batch_size + 2), dtype=torch.int32, device=device
        )
        self._enabled = torch.zeros(1, dtype=torch.int32, device=device)
        self._recording = False
        self._dumped = False
        self._host_replays = 0

    def matches(
        self,
        *,
        rank: int,
        layer_ids: tuple[int, ...],
        num_experts: int,
        top_k: int,
        ep_size: int,
    ) -> bool:
        return (
            self.rank == rank
            and self.layer_ids == layer_ids
            and self.num_experts == num_experts
            and self.top_k == top_k
            and self.ep_size == ep_size
        )

    def with_current_layer(self, layer_idx):
        return self._current_layer.with_value(layer_idx)

    def on_select_experts(self, topk_ids: torch.Tensor, num_experts=None):
        del num_experts
        if tuple(topk_ids.shape) != (self.batch_size, self.top_k):
            if self._recording:
                self.stop_record()
                self.dump_record()
            return
        if not self._recording and not self._dumped:
            self.start_record()
        layer_idx = self._current_layer.value
        if layer_idx not in self.layer_to_row:
            raise RuntimeError(
                f"routing metadata for unconfigured MoE layer {layer_idx}"
            )
        record_routing_counts(topk_ids, self._counts[self.layer_to_row[layer_idx]])

    def on_model_forward_end(self, *, num_tokens: int, is_decode: bool) -> None:
        if not is_decode or num_tokens != self.batch_size:
            return
        accumulate_routing_stats(
            self._counts,
            self._expert_totals,
            self._aggregate,
            self._occupancy_histogram,
            self._enabled,
            num_routes=self.num_routes,
            num_ranks=self.ep_size,
            max_occupancy=self.batch_size,
        )
        self._host_replays += 1
        if self._host_replays >= self.max_replays:
            self.stop_record()
            self.dump_record()

    def start_record(self) -> None:
        self._counts.zero_()
        self._expert_totals.zero_()
        self._aggregate.zero_()
        self._aggregate[:, ROUTING_STATS_FIELDS.index("lightest_rank_min")].fill_(
            self.num_routes
        )
        self._occupancy_histogram.zero_()
        self._enabled.fill_(1)
        torch.cuda.synchronize(self._enabled.device)
        self._host_replays = 0
        self._recording = True
        logger.info(
            "Started B%d MoE routing statistics on rank %d",
            self.batch_size,
            self.rank,
        )

    def stop_record(self) -> None:
        if not self._recording:
            return
        torch.cuda.synchronize(self._enabled.device)
        self._enabled.zero_()
        torch.cuda.synchronize(self._enabled.device)
        self._recording = False

    def dump_record(self, output_mode: str = "file"):
        torch.cuda.synchronize(self._aggregate.device)
        aggregate = self._aggregate.cpu().tolist()
        expert_totals = self._expert_totals.cpu().tolist()
        histogram = self._occupancy_histogram.cpu().tolist()
        report = {
            "configuration": {
                "rank": self.rank,
                "batch_size": self.batch_size,
                "top_k": self.top_k,
                "routes_per_layer": self.num_routes,
                "num_experts": self.num_experts,
                "ep_size": self.ep_size,
                "experts_per_rank": self.num_experts // self.ep_size,
                "grouped_gemm_block_size": self.block_size,
                "layer_ids": self.layer_ids,
                "host_graph_replays": self._host_replays,
            },
            **summarize_routing_stats(
                aggregate,
                histogram,
                layer_ids=self.layer_ids,
                num_experts=self.num_experts,
                num_ranks=self.ep_size,
                num_routes=self.num_routes,
                block_size=self.block_size,
            ),
            "expert_route_totals": {
                str(layer_id): totals
                for layer_id, totals in zip(
                    self.layer_ids, expert_totals, strict=True
                )
            },
        }
        self._dumped = True
        if output_mode == "object":
            return report
        if output_mode != "file":
            raise ValueError(f"unsupported output mode: {output_mode}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"routing-stats-rank{self.rank}.json"
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote MoE routing statistics to %s", output_path)
        return output_path

    @property
    def recording(self) -> bool:
        return self._recording


__all__ = [
    "MoERoutingStatsRecorder",
    "summarize_routing_stats",
]
