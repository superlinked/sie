from types import SimpleNamespace

import pytest
from sie_server.api import helpers
from sie_server.api.helpers import InferenceErrorHandler


def test_inference_error_handler_skips_telemetry_timer_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "worker_telemetry_enabled", lambda: False)

    def unexpected_timer() -> float:
        raise AssertionError("disabled telemetry must not sample its completion timer")

    monkeypatch.setattr(helpers.time, "perf_counter", unexpected_timer)
    monkeypatch.setattr(
        helpers,
        "worker_telemetry",
        lambda: SimpleNamespace(
            item_completed=lambda **_kwargs: pytest.fail("disabled telemetry must not emit a completion")
        ),
    )

    handler = InferenceErrorHandler(model="model-a", endpoint="encode", span=SimpleNamespace())
    handler._record_completion("error")
