# ruff: noqa: INP001

import os
import unittest
from unittest.mock import MagicMock, patch

import quickstart


class QuickStartTest(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "SIE_CLUSTER_URL": "https://sie.example.com",
            "SIE_API_KEY": "test-key",
            "SIE_MODEL": "example/model",
        },
        clear=True,
    )
    @patch("quickstart.SIEClient")
    def test_uses_configured_sie_deployment(self, client_type: MagicMock) -> None:
        client = client_type.return_value.__enter__.return_value
        client.encode.return_value = {"dense": [0.1, 0.2, 0.3]}

        with patch("builtins.print") as print_output:
            quickstart.main()

        client_type.assert_called_once_with(
            "https://sie.example.com",
            api_key="test-key",
        )
        client.encode.assert_called_once_with(
            "example/model",
            {"text": "Embeddings make meaning searchable."},
        )
        print_output.assert_called_once_with(
            "Created a 3-dimensional embedding.",
        )


if __name__ == "__main__":
    unittest.main()
