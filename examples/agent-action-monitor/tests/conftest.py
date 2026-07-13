"""Shared pytest fixtures for the Dusk test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from dusk.config import reset_config


@pytest.fixture(autouse=True)
def _isolate_config() -> Iterator[None]:
    """Reset the config singleton before and after every test.

    Ensures one test's ``set_config`` / environment overrides never leak into
    another, keeping detection thresholds deterministic.
    """
    reset_config()
    yield
    reset_config()
