"""Shared pytest fixtures for the Dusk test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from dusk.config import reset_config
from dusk.trace.vector import reset_sie_circuit, reset_sie_client


@pytest.fixture(autouse=True)
def _isolate_config() -> Iterator[None]:
    """Reset the config singleton before and after every test.

    Ensures one test's ``set_config`` / environment overrides never leak into
    another, keeping detection thresholds deterministic.
    """
    reset_config()
    yield
    reset_config()


@pytest.fixture(autouse=True)
def _isolate_sie_circuit() -> Iterator[None]:
    """Reset the SIE circuit breaker before and after every test.

    Several tests deliberately make ``sie_encode``/``sie_score``/``sie_extract``
    raise to exercise the fallback path. Without this, three such tests
    running back to back in the same process would trip the real circuit
    and starve an unrelated, later test of a SIE call it expects to succeed.
    """
    reset_sie_circuit()
    yield
    reset_sie_circuit()


@pytest.fixture(autouse=True)
def _isolate_sie_client() -> Iterator[None]:
    """Reset the cached SIEClient before and after every test.

    ``_sie_client`` now caches its client across calls, keyed on
    (endpoint, api_key, timeout). Without this, a test that fakes a client
    for one (endpoint, api_key, timeout) triple -- e.g. by monkeypatching
    ``sys.modules['sie_sdk']`` -- could bleed its cached fake into a later
    test that happens to use the same triple with real sie-sdk installed.
    """
    reset_sie_client()
    yield
    reset_sie_client()
