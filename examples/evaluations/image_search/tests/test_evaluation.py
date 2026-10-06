import io
import math
import tarfile
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from fetch import download, unpack
from run import encode_vectors
from score import cosine, quote, rank, vector


class FakeClient:
    def __init__(self, width=1152):
        self.calls = []
        self.width = width

    def encode(self, model, items, **kwargs):
        self.calls.append((model, items, kwargs))
        return [{"dense": [1.0] + [0.0] * (self.width - 1)} for _ in items]


class NativeRequestsTest(unittest.TestCase):
    def test_exact_bytes_are_indexed_in_order_and_queries_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = []
            original = []
            for i in range(20):
                content = b"\xff\xd8" + bytes([i]) + b"\xff\xd9"
                file = f"images/photo-{i}.jpg"
                path = root / "inputs/original-context" / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                original.append(content)
                images.append({"image_id": str(i), "file": file})
            queries = [{"query_id": str(i), "query": f"Exact query {i}."} for i in range(8)]
            client = FakeClient()
            result = encode_vectors(client, root, images, queries, "sie-siglip384")
            self.assertEqual([len(items) for _, items, _ in client.calls], [10, 10, 8])
            self.assertEqual([options["is_query"] for _, _, options in client.calls], [False, False, True])
            sent = [item["images"][0]["data"] for _, batch, _ in client.calls[:2] for item in batch]
            self.assertEqual(sent, original)
            self.assertEqual(client.calls[2][1], [{"text": row["query"]} for row in queries])
            self.assertEqual(len(result["index"]), 20)
            self.assertEqual(len(result["query"]), 8)
            for _, _, options in client.calls:
                self.assertFalse(options["wait_for_capacity"])
                self.assertEqual(options["max_oom_retries"], 0)

    def test_a_wrong_native_width_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            encode_vectors(
                FakeClient(width=1), Path(temporary), [], [{"query_id": "q", "query": "A shoe."}], "sie-siglip384"
            )


class ScoringTest(unittest.TestCase):
    def test_cosine_uses_direction_and_rejects_invalid_vectors(self):
        self.assertAlmostEqual(cosine([1, 2], [2, 1]), 0.8)
        self.assertAlmostEqual(cosine([10, 20], [2, 1]), 0.8)
        for values in ([0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0]):
            with self.assertRaises(ValueError):
                vector(values, 2)
        with self.assertRaises(ValueError):
            cosine([1], [1, 0])

    def test_equal_cosines_keep_catalogue_order_and_missing_candidates_fail(self):
        ranking, _ = rank({"b": [1, 0], "a": [1, 0]}, [1, 0], ["a", "b"])
        self.assertEqual(ranking, ["a", "b"])
        with self.assertRaises(ValueError):
            rank({"a": [1, 0]}, [1, 0], ["a", "b"])

    def test_pixel_and_token_tariffs_use_their_distinct_units(self):
        call = {"kind": "voyage", "reported_model": "v", "usage": {"text_tokens": 50, "image_pixels": 500000}}
        rates = {"v": {"text_per_million_usd": "0.12", "image_pixels_per_billion_usd": "0.60"}}
        self.assertEqual(quote(call, rates), Decimal("0.000306"))

    def test_prompt_cache_units_are_subtracted_before_pricing_uncached_input(self):
        call = {
            "kind": "caption",
            "reported_model": "c",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 100,
                "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 100},
            },
        }
        rates = {
            "c": {
                "input_per_million_usd": "2.00",
                "cached_input_per_million_usd": "0.10",
                "cache_write_per_million_usd": "2.50",
                "output_per_million_usd": "10.00",
            }
        }
        self.assertEqual(quote(call, rates), Decimal("0.00286"))


class FetchTest(unittest.TestCase):
    def test_download_rejects_changed_bytes_before_unpacking(self):
        with patch("fetch.urlopen", return_value=io.BytesIO(b"changed")), self.assertRaises(ValueError):
            download("archive.tar.gz", "0" * 64)

    def test_archive_rejects_traversal_links_and_duplicate_members(self):
        for name, linked in (("../escape", False), ("image20-8-semantic/link", True)):
            content = io.BytesIO()
            with tarfile.open(fileobj=content, mode="w:gz") as archive:
                member = tarfile.TarInfo(name)
                member.size = 1
                if linked:
                    member.type = tarfile.SYMTYPE
                    member.linkname = "../../escape"
                    archive.addfile(member)
                else:
                    archive.addfile(member, io.BytesIO(b"x"))
            with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                unpack(content.getvalue(), Path(temporary))
        content = io.BytesIO()
        with tarfile.open(fileobj=content, mode="w:gz") as archive:
            for _ in range(2):
                member = tarfile.TarInfo("image20-8-semantic/same.txt")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            unpack(content.getvalue(), Path(temporary))


if __name__ == "__main__":
    unittest.main()
