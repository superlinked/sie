"""Source-download recovery tests with synthetic bytes and no network access."""

import hashlib
import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import fetch


class FetchTests(unittest.TestCase):
    def write_pins(self, root):
        payloads = {"tasks/first.json": b'{"input":"first"}', "tasks/second.json": b'{"input":"second"}'}
        pins = root / "pins"
        pins.mkdir()
        for name, (relative, data) in zip(
            ("source-download.json", "full-source-download.json"), payloads.items(), strict=True
        ):
            (pins / name).write_text(
                json.dumps(
                    {
                        "revision": fetch.REVISION,
                        "files": {
                            relative: {
                                "url": fetch.PREFIX + relative,
                                "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest(),
                            }
                        },
                    }
                )
            )
        return payloads

    def test_interrupted_download_cleans_partial_snapshot_and_allows_retry(self):
        for failure in (TimeoutError("download interrupted"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "snapshot"
                payloads = self.write_pins(root)

                def interrupted(url, timeout, *, payloads=payloads, failure=failure):
                    relative = url.removeprefix(fetch.PREFIX)
                    if relative == "tasks/second.json":
                        raise failure
                    return BytesIO(payloads[relative])

                def complete(url, timeout, *, payloads=payloads):
                    return BytesIO(payloads[url.removeprefix(fetch.PREFIX)])

                with patch.object(fetch, "__file__", str(root / "fetch.py")):
                    with (
                        patch.object(fetch, "urlopen", side_effect=interrupted),
                        self.assertRaises(type(failure)),
                    ):
                        fetch.fetch(output)
                    self.assertFalse(output.exists())
                    with patch.object(fetch, "urlopen", side_effect=complete):
                        result = fetch.fetch(output)
                self.assertEqual(result["files"], 2)
                self.assertEqual(result["bytes"], sum(map(len, payloads.values())))
                for relative, data in payloads.items():
                    self.assertEqual((output / relative).read_bytes(), data)
                for name in ("source-download.json", "full-source-download.json"):
                    self.assertEqual((output / name).read_bytes(), (root / "pins" / name).read_bytes())

    def test_hash_mismatch_cleans_new_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "snapshot"
            payloads = self.write_pins(root)

            def corrupted(url, timeout):
                data = payloads[url.removeprefix(fetch.PREFIX)]
                return BytesIO(b"x" * len(data))

            with (
                patch.object(fetch, "__file__", str(root / "fetch.py")),
                patch.object(fetch, "urlopen", side_effect=corrupted),
                self.assertRaisesRegex(ValueError, "size/hash validation"),
            ):
                fetch.fetch(output)
            self.assertFalse(output.exists())

    def test_preexisting_snapshot_is_preserved_without_network_access(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "snapshot"
            output.mkdir()
            existing = output / "existing.txt"
            existing.write_bytes(b"preserve this data")
            with (
                patch.object(fetch, "urlopen") as download,
                self.assertRaisesRegex(ValueError, "Preserve existing source snapshots"),
            ):
                fetch.fetch(output)
            self.assertEqual(existing.read_bytes(), b"preserve this data")
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
