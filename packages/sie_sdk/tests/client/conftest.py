from unittest.mock import MagicMock

import httpx
import msgpack_numpy as m
import pytest

# Patch msgpack for numpy support
m.patch()


@pytest.fixture
def mock_httpx_client() -> MagicMock:
    """Create a mock httpx.Client."""
    return MagicMock(spec=httpx.Client)
