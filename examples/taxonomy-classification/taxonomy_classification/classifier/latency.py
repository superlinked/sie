from __future__ import annotations

import math

from taxonomy_classification.classifier.models import LatencySummary


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0

    ordered_values = sorted(values)
    if len(ordered_values) == 1:
        return ordered_values[0]

    position = (len(ordered_values) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered_values[lower_index]

    lower_value = ordered_values[lower_index]
    upper_value = ordered_values[upper_index]
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def latency_summary(
    latencies_ms: list[float],
    *,
    batch_size: int | None = None,
    items_per_request: int | None = None,
) -> LatencySummary:
    return LatencySummary(
        p50=percentile(latencies_ms, 0.5),
        p95=percentile(latencies_ms, 0.95),
        request_count=len(latencies_ms),
        batch_size=batch_size,
        items_per_request=items_per_request,
    )
