"""Smoke tests for sie_server package."""

import sie_server


def test_import() -> None:
    """Verify package can be imported."""
    assert sie_server.__version__ == "0.1.0"
