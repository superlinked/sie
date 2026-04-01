"""Prometheus metrics collection for WebSocket status.

Extracts current counter and histogram values from Prometheus registry
for real-time display in the Terminal UI.

See DESIGN.md Section 5.6.
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus_client import REGISTRY

logger = logging.getLogger(__name__)


def collect_prometheus_metrics() -> dict[str, Any]:
    """Collect current Prometheus metrics.

    Extracts counters and histograms from the default registry,
    organized by metric name and model label.

    Returns:
        Dict with "counters" and "histograms" keys.
    """
    counters: dict[str, dict[str, float]] = {}
    histograms: dict[str, dict[str, dict[str, Any]]] = {}

    try:
        # Iterate over all metrics in the registry
        for metric in REGISTRY.collect():
            # Skip internal metrics
            if metric.name.startswith("python_") or metric.name.startswith("process_"):
                continue

            for sample in metric.samples:
                name = sample.name
                labels = sample.labels
                value = sample.value

                # Handle SIE counters (sie_*_total)
                # Sum across all label combinations (endpoint, status) for each model
                if name.endswith("_total") and name.startswith("sie_"):
                    model = labels.get("model")
                    if model:
                        base_name = name[:-6]  # Remove _total suffix
                        if base_name not in counters:
                            counters[base_name] = {}
                        # Sum values for same model across different label combinations
                        counters[base_name][model] = counters[base_name].get(model, 0) + value

                # Handle SIE histograms (sie_*_bucket, sie_*_sum, sie_*_count)
                # Only use phase="total" for request duration histograms
                elif name.startswith("sie_") and "_bucket" in name:
                    # Extract base histogram name
                    base_name = name.replace("_bucket", "")
                    model = labels.get("model")
                    le = labels.get("le")
                    phase = labels.get("phase")

                    # Skip non-total phases for duration histograms
                    if phase is not None and phase != "total":
                        continue

                    if model and le:
                        if base_name not in histograms:
                            histograms[base_name] = {}
                        if model not in histograms[base_name]:
                            histograms[base_name][model] = {"buckets": [], "counts": []}

                        # Skip +Inf bucket
                        if le != "+Inf":
                            try:
                                bucket_val = float(le)
                                histograms[base_name][model]["buckets"].append(bucket_val)
                                histograms[base_name][model]["counts"].append(int(value))
                            except ValueError:
                                pass

    except Exception:
        logger.exception("Error collecting Prometheus metrics")

    return {
        "counters": counters,
        "histograms": histograms,
    }
