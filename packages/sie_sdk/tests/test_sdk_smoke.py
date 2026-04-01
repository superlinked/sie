"""Smoke tests for sie_sdk package."""

import sie_sdk


def test_import() -> None:
    """Verify package can be imported."""
    assert sie_sdk.__version__ == "0.1.0"
