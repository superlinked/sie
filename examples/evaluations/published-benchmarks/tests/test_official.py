import contextlib
import io
import json
import shlex
import tempfile
import unittest
import urllib.error
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from test_quality import FakeClient, clock, parse, reference_file, tiny_loaders
from test_verdict import DENSE_FLAGS, DenseClient, judged_qrels

import official
import run
import score
from official import COMMIT_URL, CONTENTS_URL, RAW_URL, LookupFailed, NotFound, http_fetch, lookup_official
from run import load_quality_data, official_baseline, plan_quality, run_quality
from score import recompute_verdict

MODEL = "example/dense"
FOLDER = "example__dense"
SHA = "a" * 40
PINNED = score.HF_DATASETS["mteb/SciFact"][1]
LOCAL = "http://localhost:8080"
SPARSE_FLAGS = ("--model", MODEL, "--task", "mteb/SciFact", "--output", "sparse", "--items-per-request", "2")
NO_LOOKUP = AssertionError("nothing may be looked up")


def result_file(value=0.31411, *, split="test", metric="ndcg_at_10", dataset_revision=PINNED, version="1.36.32"):
    return {
        "dataset_revision": dataset_revision,
        "task_name": "SciFact",
        "mteb_version": version,
        "scores": {split: [{"hf_subset": "default", metric: value, "main_score": value, "languages": ["eng-Latn"]}]},
    }


class FakeGitHub:
    def __init__(self, files=None, *, directories=None, commit=None, listing=None):
        self.files = {"rev-a": result_file()} if files is None else files
        self.directories = list(self.files) if directories is None else directories
        self.commit = SHA.encode() if commit is None else commit
        self.listing = listing
        self.urls = []
        self.accepts = []

    def _route(self, url):
        if url == COMMIT_URL:
            return self.commit
        if url == f"{CONTENTS_URL}/{FOLDER}?ref={SHA}":
            if self.listing is not None:
                return self.listing
            return json.dumps([{"name": name, "type": "dir"} for name in self.directories]).encode()
        prefix = f"{RAW_URL}/{SHA}/results/{FOLDER}/"
        directory, _, file_name = url.removeprefix(prefix).partition("/")
        if url.startswith(prefix) and directory in self.files and file_name.endswith(".json"):
            body = self.files[directory]
            return body if isinstance(body, bytes) else json.dumps(body).encode()
        raise NotFound(url)

    def __call__(self, url, accept=None):
        self.urls.append(url)
        self.accepts.append(accept)
        return self._route(url)


class Failing(FakeGitHub):
    def __init__(self, url_part, error, **kwargs):
        super().__init__(**kwargs)
        self.url_part, self.error = url_part, error

    def _route(self, url):
        if self.url_part in url:
            raise self.error
        return super()._route(url)


def lookup(fetch, *, model=MODEL, task="mteb/SciFact", split="test", metric="ndcg_at_10", served=None):
    return lookup_official(model, task, split, metric, served, fetch)


class LookupFoundTest(unittest.TestCase):
    def test_the_baseline_names_the_exact_file_at_the_resolved_commit(self):
        fetch = FakeGitHub()
        found = lookup(fetch)
        self.assertEqual(found.status, "found")
        source = f"https://github.com/embeddings-benchmark/results/blob/{SHA}/results/{FOLDER}/rev-a/SciFact.json"
        self.assertEqual(
            found.baseline,
            {
                "kind": "official",
                "metric": "ndcg_at_10",
                "value": 0.31411,
                "source": source,
                "model_revision": "rev-a",
                "dataset_revision": PINNED,
                "mteb_version": "1.36.32",
            },
        )
        self.assertEqual(found.detail, f"ndcg_at_10 0.31411 from {source}")
        self.assertEqual(
            fetch.urls,
            [
                "https://api.github.com/repos/embeddings-benchmark/results/commits/main",
                f"https://api.github.com/repos/embeddings-benchmark/results/contents/results/{FOLDER}?ref={SHA}",
                f"https://raw.githubusercontent.com/embeddings-benchmark/results/{SHA}/results/{FOLDER}/rev-a/SciFact.json",
            ],
        )
        self.assertEqual(fetch.accepts, ["application/vnd.github.sha", "application/vnd.github+json", None])

    def test_the_commit_can_also_come_as_json(self):
        self.assertEqual(lookup(FakeGitHub(commit=json.dumps({"sha": SHA}).encode())).status, "found")

    def test_the_split_and_metric_of_the_task_are_read(self):
        fetch = FakeGitHub({"rev-a": result_file(0.5, split="dev", metric="map_at_1000")})
        found = lookup(fetch, task="mteb/T2Reranking", split="dev", metric="map_at_1000")
        self.assertEqual((found.status, found.baseline["value"]), ("found", 0.5))
        self.assertTrue(fetch.urls[-1].endswith("/rev-a/T2Reranking.json"))

    def test_a_value_of_one_and_an_integer_one_are_accepted(self):
        for value in (1.0, 1):
            with self.subTest(value=value):
                self.assertEqual(lookup(FakeGitHub({"rev-a": result_file(value)})).baseline["value"], 1.0)

    def test_another_subset_beside_the_default_one_is_left_alone(self):
        body = result_file()
        body["scores"]["test"].append({"hf_subset": "other", "ndcg_at_10": 0.9})
        self.assertEqual(lookup(FakeGitHub({"rev-a": body})).baseline["value"], 0.31411)

    def test_missing_or_odd_revision_fields_are_recorded_as_null(self):
        body = result_file()
        del body["dataset_revision"]
        body["mteb_version"] = "line\nbreak"
        found = lookup(FakeGitHub({"rev-a": body}))
        self.assertIsNone(found.baseline["dataset_revision"])
        self.assertIsNone(found.baseline["mteb_version"])


class LookupNotFoundTest(unittest.TestCase):
    def assert_not_found(self, result, text):
        self.assertEqual(result.status, "not_found")
        self.assertIn(text, result.detail)
        self.assertIsNone(result.baseline)

    def test_a_404_at_each_step(self):
        self.assert_not_found(lookup(Failing("/commits/main", NotFound("x"))), "no main branch")
        self.assert_not_found(lookup(Failing("/contents/", NotFound("x"))), f"{MODEL} is not in")
        self.assert_not_found(lookup(FakeGitHub({}, directories=["rev-a", "rev-b"])), "no SciFact.json")

    def test_a_file_with_no_such_split_or_metric_has_no_number(self):
        self.assert_not_found(lookup(FakeGitHub({"rev-a": result_file(split="train")})), "has no test scores")
        older = {"scores": {"dev": [{"hf_subset": "default", "map": 0.66, "main_score": 0.66}]}}
        fetch = FakeGitHub({"rev-a": older})
        self.assert_not_found(
            lookup(fetch, task="mteb/T2Reranking", split="dev", metric="map_at_1000"),
            "rev-a/T2Reranking.json has no map_at_1000",
        )

    def test_a_newer_file_is_used_beside_one_with_no_number(self):
        older = {"scores": {"dev": [{"hf_subset": "default", "map": 0.66, "main_score": 0.66}]}}
        fetch = FakeGitHub({"rev-a": older, "rev-b": result_file(0.65, split="dev", metric="map_at_1000")})
        found = lookup(fetch, task="mteb/T2Reranking", split="dev", metric="map_at_1000", served="rev-a")
        self.assertEqual((found.status, found.baseline["model_revision"]), ("found", "rev-b"))

    def test_a_listing_with_no_directories(self):
        self.assert_not_found(lookup(FakeGitHub(listing=b'[{"name": "README.md", "type": "file"}]')), "no SciFact.json")

    def test_a_model_id_that_is_not_plain_builds_no_url(self):
        for model in ("noslash", "a/b/c", "a/b?x=1", "a b/c", "a/b\n", "", "org/na%2Fme", "../x/y"):
            with self.subTest(model=model):
                fetch = FakeGitHub()
                self.assert_not_found(lookup(fetch, model=model), "not an <org>/<name> model id")
                self.assertEqual(fetch.urls, [])

    def test_directory_names_that_are_not_plain_are_never_fetched(self):
        names = ["../../secrets", "a/b", "a b", ".", "..", "x?y", ""]
        listing = json.dumps([{"name": name, "type": "dir"} for name in names] + [{"name": 7, "type": "dir"}]).encode()
        fetch = FakeGitHub(listing=listing)
        self.assert_not_found(lookup(fetch), "no SciFact.json")
        self.assertEqual(len(fetch.urls), 2)


class LookupFailedTest(unittest.TestCase):
    def assert_failed(self, result, text=""):
        self.assertEqual(result.status, "failed")
        self.assertIn(text, result.detail)
        self.assertIsNone(result.baseline)

    def test_a_rate_limit_or_network_error_at_each_step(self):
        for part in ("/commits/main", "/contents/", "/SciFact.json"):
            for error in (LookupFailed("HTTP 403 from api.github.com"), TimeoutError("slow"), RuntimeError("x")):
                with self.subTest(part=part, error=error):
                    self.assert_failed(lookup(Failing(part, error)))

    def test_a_failure_on_one_directory_fails_the_lookup(self):
        fetch = Failing("/rev-b/", LookupFailed("HTTP 500"), files={"rev-a": result_file(), "rev-b": result_file()})
        self.assert_failed(lookup(fetch), "HTTP 500")

    def test_a_commit_that_is_no_commit_id(self):
        for commit in (b"<html>", b'{"sha": "main"}', b'{"sha": 7}', b"[]", b"A" * 40, b"../" * 14):
            with self.subTest(commit=commit):
                fetch = FakeGitHub(commit=commit)
                self.assert_failed(lookup(fetch))
                self.assertEqual(len(fetch.urls), 1)

    def test_a_listing_of_the_wrong_type(self):
        for listing in (b"not json", b'{"message": "x"}', b"[1, 2]", b"null"):
            with self.subTest(listing=listing):
                self.assert_failed(lookup(FakeGitHub(listing=listing)), "listing")

    def test_a_result_file_that_cannot_be_used(self):
        cases = {
            "not JSON": b"<html>",
            "deeply nested JSON": b"[" * 100000,
            "not an object": [],
            "no scores": {"task_name": "SciFact"},
            "scores of the wrong type": {"scores": []},
            "split of the wrong type": {"scores": {"test": {"hf_subset": "default"}}},
            "entries of the wrong type": {"scores": {"test": [1, "x", None]}},
            "no default subset": {"scores": {"test": [{"hf_subset": "a", "ndcg_at_10": 0.3}]}},
            "several subsets, none default": {
                "scores": {"test": [{"hf_subset": "a", "ndcg_at_10": 0.3}, {"hf_subset": "b", "ndcg_at_10": 0.3}]}
            },
            "two default subsets": {
                "scores": {
                    "test": [{"hf_subset": "default", "ndcg_at_10": 0.3}, {"hf_subset": "default", "ndcg_at_10": 0.4}]
                }
            },
            "metric as text": result_file("0.3"),
            "metric as a boolean": result_file(True),
            "metric null": result_file(None),
            "metric zero": result_file(0.0),
            "metric negative": result_file(-0.1),
            "metric in percent": result_file(31.4),
            "metric not finite": b'{"scores": {"test": [{"hf_subset": "default", "ndcg_at_10": NaN}]}}',
            "metric infinite": b'{"scores": {"test": [{"hf_subset": "default", "ndcg_at_10": Infinity}]}}',
            "metric too large for a float": result_file(10**400),
        }
        for name, body in cases.items():
            with self.subTest(name):
                self.assert_failed(lookup(FakeGitHub({"rev-a": body})), "rev-a/SciFact.json")

    def test_a_bad_file_does_not_hide_a_good_one(self):
        found = lookup(FakeGitHub({"rev-a": b"<html>", "rev-b": result_file(0.4)}))
        self.assertEqual((found.status, found.baseline["model_revision"]), ("found", "rev-b"))

    def test_a_bad_file_is_reported_ahead_of_a_file_with_no_number(self):
        self.assert_failed(lookup(FakeGitHub({"rev-a": result_file(split="train"), "rev-b": b"<html>"})), "not JSON")


class LookupPreferenceTest(unittest.TestCase):
    FILES: ClassVar[dict] = {
        "rev-a": result_file(0.1, version="1.9.0"),
        "rev-b": result_file(0.2, version="1.36.32"),
        "rev-c": result_file(0.3, version="1.36.32"),
        "rev-d": result_file(0.4, version=None),
    }

    def test_the_served_revision_wins(self):
        self.assertEqual(lookup(FakeGitHub(self.FILES), served="rev-a").baseline["model_revision"], "rev-a")

    def test_otherwise_the_highest_mteb_version_numerically_and_ties_by_sorted_name(self):
        for served in (None, "rev-z"):
            with self.subTest(served=served):
                self.assertEqual(lookup(FakeGitHub(self.FILES), served=served).baseline["model_revision"], "rev-b")

    def test_a_listing_order_changes_nothing(self):
        fetch = FakeGitHub(self.FILES, directories=["rev-d", "rev-c", "rev-b", "rev-a"])
        self.assertEqual(lookup(fetch).baseline["model_revision"], "rev-b")

    def test_a_file_with_no_version_is_the_last_choice(self):
        found = lookup(FakeGitHub({"rev-a": result_file(version=None), "rev-b": result_file(version="0.1")}))
        self.assertEqual(found.baseline["model_revision"], "rev-b")


class HttpFetchTest(unittest.TestCase):
    def test_a_plain_request_with_no_credential(self):
        environ = {"GITHUB_TOKEN": "t0ken", "GH_TOKEN": "t0ken", "HF_TOKEN": "t0ken", "SIE_API_KEY": "t0ken"}
        with (
            patch.dict("os.environ", environ),
            patch("official.urllib.request.urlopen", return_value=io.BytesIO(b"body")) as urlopen,
        ):
            self.assertEqual(http_fetch(COMMIT_URL, "application/vnd.github.sha"), b"body")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, COMMIT_URL)
        headers = dict(request.header_items())
        self.assertEqual(set(headers), {"User-agent", "Accept"})
        self.assertNotIn("@", headers["User-agent"])
        self.assertNotIn("t0ken", json.dumps(headers))
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": 30.0})

    def test_without_an_accept_only_the_user_agent_is_sent(self):
        with patch("official.urllib.request.urlopen", return_value=io.BytesIO(b"body")) as urlopen:
            http_fetch(f"{RAW_URL}/x")
        self.assertEqual(set(dict(urlopen.call_args.args[0].header_items())), {"User-agent"})

    def test_the_bytes_read_are_capped(self):
        with patch("official.urllib.request.urlopen", return_value=io.BytesIO(b"x" * official.MAX_BYTES)):
            self.assertEqual(len(http_fetch(COMMIT_URL)), official.MAX_BYTES)
        with (
            patch("official.urllib.request.urlopen", return_value=io.BytesIO(b"x" * (official.MAX_BYTES + 1))),
            self.assertRaises(LookupFailed),
        ):
            http_fetch(COMMIT_URL)

    def http_error(self, code):
        return urllib.error.HTTPError(COMMIT_URL, code, "x", None, None)

    def test_a_404_is_not_found(self):
        with patch("official.urllib.request.urlopen", side_effect=self.http_error(404)), self.assertRaises(NotFound):
            http_fetch(COMMIT_URL)

    def test_a_rate_limit_is_named_and_short(self):
        for code in (403, 429):
            with (
                self.subTest(code=code),
                patch("official.urllib.request.urlopen", side_effect=self.http_error(code)),
                self.assertRaisesRegex(LookupFailed, f"^HTTP {code} from api.github.com, probably .*rate limit$"),
            ):
                http_fetch(COMMIT_URL)

    def test_a_network_error_is_a_failure(self):
        for error in (urllib.error.URLError("no route"), TimeoutError("slow"), ConnectionResetError("reset")):
            with (
                self.subTest(error=error),
                patch("official.urllib.request.urlopen", side_effect=error),
                self.assertRaisesRegex(LookupFailed, "could not reach api.github.com"),
            ):
                http_fetch(COMMIT_URL)


class InfoClient(DenseClient):
    def __init__(self, *, outputs=("dense",), revision="rev-a", info_error=None, **kwargs):
        super().__init__(**kwargs)
        self.info = {"outputs": list(outputs) if outputs is not None else None, "revision": revision}
        self.info_error = info_error
        self.info_calls = []

    def get_model(self, model):
        self.info_calls.append(model)
        if self.info_error is not None:
            raise self.info_error
        return self.info


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def resolve(self, client, fetch, *flags):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            plan, record = official_baseline(plan_quality(parse(*flags)), client, fetch)
        return plan, record, stdout.getvalue().splitlines()

    def run_official(self, client, fetch, *flags):
        plan, lookup_record, _ = self.resolve(client, fetch, *flags)
        return run_quality(
            plan,
            load_quality_data(plan, tiny_loaders()),
            client,
            results_dir=self.directory / "results",
            now=clock,
            official_lookup=lookup_record,
        )


class OfficialBaselineTest(TemporaryCase):
    def test_a_found_number_becomes_the_baseline_and_no_inference_request_is_made(self):
        client, fetch = InfoClient(), FakeGitHub()
        plan, record, lines = self.resolve(client, fetch, *DENSE_FLAGS)
        self.assertEqual(plan.baseline["kind"], "official")
        self.assertEqual(plan.baseline["value"], 0.31411)
        self.assertEqual(record["status"], "found")
        self.assertEqual(lines, [f"official: ndcg_at_10 0.31411 from {plan.baseline['source']}"])
        self.assertEqual(record["detail"], lines[0].removeprefix("official: "))
        self.assertEqual(client.calls, [])
        self.assertEqual(client.info_calls, [MODEL])

    def test_a_miss_and_a_failure_keep_the_plan_and_say_so(self):
        for fetch, status, text in (
            (Failing("/contents/", NotFound("x")), "not_found", f"{MODEL} is not in"),
            (Failing("/commits/main", LookupFailed("HTTP 403 from api.github.com")), "failed", "HTTP 403"),
            (FakeGitHub({"rev-a": b"<html>"}), "failed", "not JSON"),
        ):
            with self.subTest(status=status):
                plan, record, lines = self.resolve(InfoClient(), fetch, *DENSE_FLAGS)
                self.assertIsNone(plan.baseline)
                self.assertEqual(record["status"], status)
                self.assertIn(text, record["detail"])
                self.assertEqual(
                    lines,
                    [f"official: none, {record['detail']}; the run reads MEASUREMENT unless you pass --baseline"],
                )

    def test_notes_when_the_revisions_differ(self):
        fetch = FakeGitHub({"rev-a": result_file(dataset_revision="other-dataset")})
        _, record, lines = self.resolve(InfoClient(revision="rev-served"), fetch, *DENSE_FLAGS)
        self.assertTrue(lines[0].startswith("official: ndcg_at_10 0.31411"))
        self.assertEqual(
            lines[1:],
            [
                "note: the official number is for model revision rev-a, the endpoint serves rev-served",
                f"note: the official number is for dataset revision other-dataset, this run uses {PINNED}",
            ],
        )
        self.assertEqual(record["status"], "found")
        self.assertIn("the endpoint serves rev-served", record["detail"])

    def test_no_note_when_they_agree_or_are_unknown(self):
        for client, body in (
            (InfoClient(), result_file()),
            (InfoClient(revision=None), result_file()),
            (InfoClient(info_error=RuntimeError("down")), result_file()),
            (InfoClient(), result_file(dataset_revision=None)),
        ):
            with self.subTest(info=client.info, error=client.info_error):
                _, _, lines = self.resolve(client, FakeGitHub({"rev-a": body}), *DENSE_FLAGS)
                self.assertEqual(len(lines), 1)

    def test_the_served_revision_picks_the_file(self):
        files = {"rev-a": result_file(0.1, version="1.0"), "rev-b": result_file(0.2, version="2.0")}
        plan, _, _ = self.resolve(InfoClient(revision="rev-a"), FakeGitHub(files), *DENSE_FLAGS)
        self.assertEqual(plan.baseline["value"], 0.1)

    def assert_skipped(self, client, flags, text):
        fetch = FakeGitHub()
        plan, record, lines = self.resolve(client, fetch, *flags)
        self.assertIsNone(plan.baseline)
        self.assertEqual(record["status"], "skipped")
        self.assertIn(text, record["detail"])
        self.assertEqual(fetch.urls, [])
        self.assertEqual(len(lines), 1)
        self.assertEqual(
            lines, [f"official: none, {record['detail']}; the run reads MEASUREMENT unless you pass --baseline"]
        )

    def test_a_limit_key_is_skipped_and_the_line_does_not_promise_a_baseline_helps(self):
        client = InfoClient()
        fetch = FakeGitHub()
        plan, record, lines = self.resolve(
            client,
            fetch,
            "--model",
            MODEL,
            "--task",
            "mteb/SciFact?limit=2",
            "--output",
            "dense",
            "--items-per-request",
            "2",
        )
        self.assertIsNone(plan.baseline)
        self.assertEqual(record["status"], "skipped")
        self.assertIn("limit", record["detail"])
        self.assertEqual(lines, [f"official: none, {record['detail']}; the run reads MEASUREMENT"])
        self.assertEqual((fetch.urls, client.info_calls), ([], []))

    def test_a_two_stage_key_is_skipped(self):
        client = InfoClient()
        key = "mteb/SciFact?candidates_model=org/encoder&candidates_k=2"
        self.assert_skipped(client, ("--model", "org/reranker", "--task", key), "retriever and reranker")
        self.assertEqual(client.info_calls, [])

    def test_a_profile_other_than_default_is_skipped(self):
        client = InfoClient()
        flags = ("--model", f"{MODEL}:sparse", *SPARSE_FLAGS[2:])
        self.assert_skipped(client, flags, "profile")
        self.assertEqual(client.info_calls, [])

    def test_a_non_dense_output_on_a_model_with_several_outputs_is_skipped(self):
        for flags, output in (
            (SPARSE_FLAGS, "sparse"),
            (SPARSE_FLAGS[:5] + ("multivector",) + SPARSE_FLAGS[6:], "multivector"),
        ):
            with self.subTest(output=output):
                client = InfoClient(outputs=("dense", "sparse", "multivector"))
                self.assert_skipped(
                    client, flags, f"several outputs and the official number is for dense, not {output}"
                )

    def test_a_non_dense_output_is_skipped_when_the_outputs_are_unknown(self):
        for client in (
            InfoClient(info_error=RuntimeError("down")),
            InfoClient(outputs=None),
            InfoClient(outputs=()),
        ):
            with self.subTest(info=client.info, error=client.info_error):
                self.assert_skipped(client, SPARSE_FLAGS, "did not say what the model outputs")

    def test_a_model_with_one_output_is_looked_up_on_that_output(self):
        for outputs in (("sparse",), ("sparse", "score")):
            with self.subTest(outputs=outputs):
                plan, record, _ = self.resolve(InfoClient(outputs=outputs), FakeGitHub(), *SPARSE_FLAGS)
                self.assertEqual((plan.baseline["kind"], record["status"]), ("official", "found"))

    def test_a_dense_output_is_looked_up_whatever_the_model_exposes(self):
        for client in (
            InfoClient(outputs=("dense", "sparse", "multivector")),
            InfoClient(info_error=RuntimeError("down")),
        ):
            with self.subTest(info=client.info, error=client.info_error):
                plan, _, _ = self.resolve(client, FakeGitHub(), *DENSE_FLAGS)
                self.assertEqual(plan.baseline["kind"], "official")

    def test_a_score_task_is_looked_up_with_its_split_and_metric(self):
        fetch = FakeGitHub({"rev-a": result_file(0.6, split="dev", metric="map_at_1000")})
        plan, _, _ = self.resolve(
            InfoClient(outputs=("score",), info_error=RuntimeError("down")),
            fetch,
            "--model",
            MODEL,
            "--task",
            "mteb/T2Reranking",
        )
        self.assertEqual((plan.baseline["metric"], plan.baseline["value"]), ("map_at_1000", 0.6))

    def test_an_alias_looks_up_the_canonical_file(self):
        fetch = FakeGitHub()
        self.resolve(InfoClient(), fetch, "--model", MODEL, "--task", "mteb/CMedQAv1Reranking")
        self.assertTrue(fetch.urls[-1].endswith("/rev-a/CMedQAv1-reranking.json"))

    def test_a_supplied_baseline_wins_and_nothing_is_looked_up(self):
        client, fetch = InfoClient(), FakeGitHub()
        plan, record, lines = self.resolve(
            client, fetch, *DENSE_FLAGS, "--baseline", "ndcg_at_10=0.3", "--baseline-source", "a paper"
        )
        self.assertEqual(plan.baseline["kind"], "supplied")
        self.assertIsNone(record)
        self.assertEqual((lines, fetch.urls, client.info_calls), ([], [], []))

    def test_a_reference_run_looks_nothing_up(self):
        client, fetch = InfoClient(), FakeGitHub()
        plan, record, lines = self.resolve(client, fetch, "--reference", str(reference_file(self.directory)))
        self.assertEqual(plan.baseline["kind"], "published")
        self.assertIsNone(record)
        self.assertEqual((lines, fetch.urls, client.info_calls), ([], [], []))


class OfficialRecordTest(TemporaryCase):
    def test_the_result_records_the_lookup_and_compares_with_the_number(self):
        record, _ = self.run_official(InfoClient(), FakeGitHub(), *DENSE_FLAGS)
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["official_lookup"]["status"], "found")
        self.assertEqual(record["baseline"]["kind"], "official")
        self.assertEqual(record["verdict"]["baseline"], 0.31411)
        self.assertEqual(record["verdict"]["metric"], "ndcg_at_10")
        self.assertIn(record["verdict"]["reason"], {"above_margin", "below_margin", "within_margin"})

    def test_a_miss_records_the_status_and_reads_measurement(self):
        record, _ = self.run_official(InfoClient(), Failing("/contents/", NotFound("x")), *DENSE_FLAGS)
        self.assertEqual(record["official_lookup"]["status"], "not_found")
        self.assertIsNone(record["baseline"])
        self.assertEqual((record["verdict"]["verdict"], record["verdict"]["reason"]), ("MEASUREMENT", "no_baseline"))

    def test_the_field_is_absent_beside_a_baseline_or_a_reference(self):
        supplied = (*DENSE_FLAGS, "--baseline", "ndcg_at_10=0.3", "--baseline-source", "a paper")
        record, _ = self.run_official(InfoClient(), FakeGitHub(), *supplied)
        self.assertNotIn("official_lookup", record)
        record, _ = self.run_official(InfoClient(), FakeGitHub(), "--reference", str(reference_file(self.directory)))
        self.assertNotIn("official_lookup", record)

    def test_the_command_repeats_the_run_and_does_not_gain_a_baseline(self):
        plan = plan_quality(parse(*DENSE_FLAGS))
        record, _ = self.run_official(InfoClient(), FakeGitHub(), *DENSE_FLAGS)
        self.assertNotIn("--baseline", record["run"]["command"])
        self.assertEqual(plan_quality(parse(*shlex.split(record["run"]["command"])[5:])), plan)

    def test_score_py_recomputes_the_verdict_from_the_stored_value_with_no_network(self):
        record, result_path = self.run_official(InfoClient(), FakeGitHub(), *DENSE_FLAGS)
        stdout = io.StringIO()
        with (
            patch("score.load_qrels", return_value=judged_qrels()),
            patch("official.urllib.request.urlopen", side_effect=NO_LOOKUP) as urlopen,
            contextlib.redirect_stdout(stdout),
        ):
            _, rows, _ = score.rescore(result_path)
            score.main([str(result_path), "--write"])
        urlopen.assert_not_called()
        self.assertEqual(rows, record["metrics"])
        self.assertEqual(recompute_verdict(record, rows), record["verdict"])
        self.assertIn(f"from {record['baseline']['source']}", stdout.getvalue())
        written = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(written["official_lookup"], record["official_lookup"])
        self.assertEqual(written["baseline"], record["baseline"])


class OfficialCommandTest(TemporaryCase):
    def main(self, client, fetch, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        seen = []

        def first_encode(model, items, **kwargs):
            seen.append(stdout.getvalue())
            return DenseClient.encode(client, model, items, **kwargs)

        client.encode = first_encode
        code = None
        with (
            contextlib.chdir(self.directory),
            patch.dict("os.environ", {}, clear=True),
            patch.object(run, "load_quality_data", side_effect=lambda plan: load_quality_data(plan, tiny_loaders())),
            patch.object(run, "SIEClient") as factory,
            patch.object(run, "http_fetch", fetch),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            factory.return_value.__enter__.return_value = client
            try:
                run.main(["quality", *argv, "--endpoint", LOCAL])
            except SystemExit as caught:
                code = caught.code
        return code, stdout.getvalue().splitlines(), stderr.getvalue(), seen

    def test_the_line_is_printed_before_the_first_inference_request(self):
        code, lines, stderr, seen = self.main(InfoClient(revision="rev-b"), FakeGitHub(), *DENSE_FLAGS)
        self.assertIsNone(code)
        self.assertEqual(stderr, "")
        self.assertTrue(lines[0].startswith("official: ndcg_at_10 0.31411 from https://github.com/"))
        self.assertTrue(lines[1].startswith("note: the official number is for model revision rev-a"))
        self.assertIn("official: ndcg_at_10", seen[0])
        self.assertTrue(lines[-1].startswith("verdict: "))
        self.assertIn(" against 0.31411 from https://github.com/", lines[-1])

    def test_a_lookup_failure_does_not_block_the_run(self):
        fetch = Failing("/commits/main", TimeoutError("slow"))
        code, lines, _, _ = self.main(InfoClient(), fetch, *DENSE_FLAGS)
        self.assertIsNone(code)
        self.assertEqual(
            lines[0],
            "official: none, the lookup failed (TimeoutError); the run reads MEASUREMENT unless you pass --baseline",
        )
        self.assertEqual(lines[-1], "verdict: MEASUREMENT no baseline")

    def test_a_supplied_baseline_never_calls_the_fetch(self):
        fetch = FakeGitHub()
        _, lines, _, _ = self.main(
            InfoClient(), fetch, *DENSE_FLAGS, "--baseline", "recall_at_1=1", "--baseline-source", "s"
        )
        self.assertEqual(fetch.urls, [])
        self.assertFalse([line for line in lines if line.startswith("official:")])

    def test_a_reference_run_never_calls_the_fetch_and_has_no_new_output_or_field(self):
        fetch = FakeGitHub()
        reference = reference_file(self.directory)
        client = FakeClient()
        client.get_model = lambda model: self.fail("a reference run must not ask for the model")
        _, lines, _, _ = self.main(client, fetch, "--reference", str(reference))
        self.assertEqual(fetch.urls, [])
        self.assertFalse([line for line in lines if line.startswith(("official:", "note:"))])
        (result_path,) = (self.directory / "results").glob("*-????????T??????Z.json")
        self.assertNotIn("official_lookup", json.loads(result_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
