"""Reader wiring and one-call local-encoder checks using the actual SDK."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from transformers import AutoTokenizer

from embedding import selfhost
import live
import router
from test_selfhost_encode import FakeServer, Tokenizer
import workflow


class ReaderSelfHostTests(unittest.TestCase):
    def test_reader_encoder_omits_cloud_budget_and_returns_feature_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = router.parser().parse_args(["encode", "--data", str(root / "data"),
                "--output", str(root / "encoded"), "--self-hosted", "--sie-base-url", "http://127.0.0.1:8000"])
            with patch.object(workflow, "checked_data", return_value=root / "data"), \
                    patch.object(workflow, "invoke") as invoke:
                result = router.execute(args)
            argv = list(map(str, invoke.call_args.args))
            self.assertNotIn("--budget-usd", argv)
            self.assertIn("--self-hosted", argv)
            self.assertEqual(result["features"], str(root / "encoded/features"))

    def test_live_local_encoder_preserves_one_sdk_receipt_without_cloud_key(self):
        original_client = selfhost.make_client
        for fail in (False, True):
            server = FakeServer(fail=fail)
            with self.subTest(failure=fail), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "encoder"
                item = {"id": "synthetic-live", "request": "Synthetic question"}
                def factory(base_url, capture):
                    return original_client(base_url, capture, transport=httpx.MockTransport(server))
                with patch.object(AutoTokenizer, "from_pretrained", return_value=Tokenizer()), \
                        patch.object(selfhost, "make_client", side_effect=factory), \
                        patch.dict(os.environ, {"SIE_API_KEY": "synthetic-never-forwarded", "SIE_SELF_HOSTED_API_KEY": ""}):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "Automatic retries disabled"):
                            live.embed(item, output, base_url="http://127.0.0.1:8000", self_hosted=True)
                    else:
                        matrix, fee = live.embed(item, output, base_url="http://127.0.0.1:8000", self_hosted=True)
                        np.testing.assert_allclose(matrix[0, :2], [.6, .8], atol=1e-7)
                        self.assertIsNone(fee["encoder_usd"])
                        self.assertIsNone(fee["hosting_usd"])
                        self.assertTrue((output / "receipt.json").exists())
                self.assertEqual(sum(request.method == "POST" for request in server.requests), 1)
                self.assertTrue(all("authorization" not in request.headers for request in server.requests))
                for name in ("intent.json", "response.bin", "http.json"):
                    self.assertTrue((output / name).exists())


if __name__ == "__main__":
    unittest.main()
