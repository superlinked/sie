"""Real CPU smoke plus negative controls for the public acceptance harness."""

# Owned Python subprocesses run only this checkout's benchmark, without a shell.
# ruff: noqa: S603

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import httpx
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from tools.benchmarks.cross_encoder_lora import prepare as source_preparer
from tools.benchmarks.cross_encoder_lora.adapter import require_forward_coverage
from tools.benchmarks.cross_encoder_lora.client import describe_response, validate_response
from tools.benchmarks.cross_encoder_lora.evidence import Evidence, atomic_json, vector_delta
from tools.benchmarks.cross_encoder_lora.fixtures import prepare
from tools.benchmarks.cross_encoder_lora.protocol import BASE_REVISION, ORDERS, compare, paired_gate, percentile
from tools.benchmarks.cross_encoder_lora.runner import ROOT, extract, source_environment
from tools.benchmarks.cross_encoder_lora.worker import ORDER, matching_vector


def tiny_snapshot(directory: Path) -> Path:
    directory.mkdir()
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "password",
        "reset",
        "link",
        "email",
        "paris",
        "account",
        "support",
        "billing",
        "address",
        ".",
        "?",
        "how",
        "do",
        "i",
        "my",
        "use",
        "to",
        "a",
        "is",
        "of",
        "france",
    ]
    (directory / "vocab.txt").write_text("\n".join(vocabulary) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(directory / "vocab.txt"))
    tokenizer.save_pretrained(directory)
    config = BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=1,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    config.sbert_ce_default_activation_function = "torch.nn.modules.linear.Identity"
    torch.manual_seed(29)
    BertForSequenceClassification(config).save_pretrained(directory)
    return directory


def test_gate_positive_negative_and_uncertainty() -> None:
    assert paired_gate([1.0] * 6)["passed"]
    assert not paired_gate([1.02] * 6)["passed"]  # point gate
    uncertain = paired_gate([0.94, 0.94, 0.94, 1.06, 1.06, 1.06])
    assert uncertain["geometric_ratio"] < 1.01
    assert uncertain["one_sided_95_upper"] > 1.03
    assert not uncertain["passed"]
    assert not paired_gate([0.97] * 5 + [1.11])["passed"]
    with pytest.raises(AssertionError, match="six"):
        paired_gate([1.0] * 5)
    with pytest.raises(AssertionError, match="invalid"):
        paired_gate([float("nan")] * 6)


def test_nearest_rank() -> None:
    assert percentile(list(range(1, 21)), 0.95) == 19
    assert percentile([1, 2, 3], 0.95) == 3


def trials() -> list[dict]:
    performance = {str(c): {"mean_ms": 10.0, "p95_ms": 12.0, "rps": 100.0} for c in (1, 4, 16)}
    return [
        {"pair": i, "arm": arm, "passed": True, "shutdown_complete": True, "performance": performance}
        for i, order in enumerate(ORDERS)
        for arm in order
    ]


def test_full_schedule_and_cleanup_are_required() -> None:
    assert compare(trials())["passed"]
    wrong = trials()
    wrong[0]["arm"] = "R"
    with pytest.raises(AssertionError, match="identities"):
        compare(wrong)
    incomplete = trials()
    incomplete[0]["shutdown_complete"] = False
    with pytest.raises(AssertionError, match="failed"):
        compare(incomplete)


@pytest.mark.parametrize("value", [-1.0, 0.0, float("nan"), float("inf")])
def test_raw_metrics_must_be_positive_finite(value: float) -> None:
    invalid = trials()
    for trial in invalid:
        trial["performance"]["1"]["mean_ms"] = value
    with pytest.raises(AssertionError, match="raw performance"):
        compare(invalid)


def test_nonfinite_evidence_retained_but_never_accepted(tmp_path: Path) -> None:
    evidence = Evidence(tmp_path / "out")
    evidence.emit("bad_output", values=[float("nan"), float("inf")])
    assert json.loads((tmp_path / "out/events.jsonl").read_text())["values"] == ["nan", "inf"]
    with pytest.raises(AssertionError, match="nonfinite"):
        vector_delta([float("nan")], [1.0])
    atomic_json(tmp_path / "out/final.json", {"passed": False})
    assert json.loads((tmp_path / "out/final.json").read_text()) == {"passed": False}


@pytest.mark.parametrize("arm", ["B", "R"])
def test_real_loader_sdk_lifecycle(tmp_path: Path, arm: str) -> None:
    snapshot = tiny_snapshot(tmp_path / "model")
    fixtures = tmp_path / "fixtures"
    prepare(fixtures, snapshot, smoke=True)
    # Use real historical B and committed candidate imports, not the editable
    # checkout accidentally standing in for both arms. Full clones contain B;
    # a shallow CI checkout can still run the other deterministic controls.
    available = subprocess.run(["git", "cat-file", "-e", BASE_REVISION], cwd=ROOT, capture_output=True, check=False)  # noqa: S607
    if available.returncode:
        pytest.skip("immutable archive smoke requires the documented baseline commit in git history")
    inputs = tmp_path / "source-inputs"
    source_preparer.prepare(ROOT, inputs, "HEAD")
    sources = tmp_path / "sources"
    extract(inputs, sources)
    source = sources / ("B" if arm == "B" else "candidate")
    output = tmp_path / "evidence"
    command = [
        sys.executable,
        "-m",
        "tools.benchmarks.cross_encoder_lora.worker",
        "--source",
        str(source),
        "--fixtures",
        str(fixtures),
        "--output",
        str(output),
        "--arm",
        arm,
        "--smoke",
        "--deadline-s",
        "90",
    ]
    result = subprocess.run(
        command, env=source_environment(source), cwd=source, capture_output=True, text=True, timeout=130, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "trial.json").read_text())
    assert summary["passed"]
    assert summary["shutdown_complete"]
    assert summary["manager_present"] == (arm == "R")
    assert summary["environment"]["qualification"] is False
    assert Path(summary["environment"]["adapter_source"]).is_relative_to(source)
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert {"shape_oracles", "performance", "finished"} <= {event["event"] for event in events}
    batches = next(event["batches"] for event in events if event["event"] == "all_batches")
    assert batches
    for batch in batches:
        require_forward_coverage(batch)
    if arm == "R":
        assert {"effects", "concurrent_batches", "resident_set", "restored_state"} <= {
            event["event"] for event in events
        }


def test_worker_failure_persists_terminal_record(tmp_path: Path) -> None:
    output = tmp_path / "failed"
    command = [
        sys.executable,
        "-m",
        "tools.benchmarks.cross_encoder_lora.worker",
        "--source",
        str(ROOT),
        "--fixtures",
        str(tmp_path / "missing"),
        "--output",
        str(output),
        "--arm",
        "R",
        "--smoke",
    ]
    result = subprocess.run(
        command, env=source_environment(ROOT), cwd=ROOT, capture_output=True, text=True, timeout=45, check=False
    )
    assert result.returncode != 0
    summary = json.loads((output / "trial.json").read_text())
    assert not summary["passed"]
    assert summary["error"]["type"] == "FileNotFoundError"
    assert (output / "events.jsonl").read_text().endswith("\n")


def test_source_archive_resolves_moving_refs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args])  # noqa: S607

    git("init", "-q")
    source = root / "packages/sie_server/src"
    source.mkdir(parents=True)
    (source / "value.txt").write_text("baseline")
    (root / "uv.lock").write_text("fixed lock")
    git("add", ".")

    def commit() -> None:
        git("-c", "user.name=Harness Test", "-c", "user.email=harness@example.invalid", "commit", "-qm", "fixture")

    commit()
    baseline = git("rev-parse", "HEAD").decode().strip()
    git("checkout", "-qb", "candidate")
    (source / "value.txt").write_text("candidate before move")
    git("add", ".")
    commit()
    candidate = git("rev-parse", "HEAD").decode().strip()
    original = source_preparer.git

    def moving_git(path: Path, *arguments: str) -> bytes:
        value = original(path, *arguments)
        if arguments[-1] == "candidate^{commit}":
            (source / "value.txt").write_text("candidate after move")
            git("add", ".")
            commit()
        return value

    monkeypatch.setattr(source_preparer, "BASE_REVISION", baseline)
    monkeypatch.setattr(source_preparer, "git", moving_git)
    output = tmp_path / "archives"
    manifest = source_preparer.prepare(root, output, "candidate", baseline)
    assert manifest["sources"]["candidate"]["commit"] == candidate
    assert git("rev-parse", "candidate").decode().strip() != candidate
    with tarfile.open(output / "candidate.tar") as archive:
        stream = archive.extractfile("packages/sie_server/src/value.txt")
        assert stream is not None
        assert stream.read() == b"candidate before move"


def test_tampered_archive_refused_before_extract(tmp_path: Path) -> None:
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "B.tar").write_bytes(b"not the declared archive")
    atomic_json(
        inputs / "manifest.json",
        {
            "schema": 1,
            "dependency_lock_identical": True,
            "sources": {"B": {"commit": BASE_REVISION, "archive": "B.tar", "sha256": "0" * 64}},
        },
    )
    with pytest.raises(AssertionError, match="archive hash"):
        extract(inputs, tmp_path / "output")


def test_timed_diagnostics_require_one_complete_oracle_vector() -> None:
    first = [0.0, 1.0, 2.0, 3.0]
    second = [10.0, 11.0, 12.0, 13.0]
    oracles = {
        "first": {"copies": 1, "scores": [first[index] for index in ORDER]},
        "second": {"copies": 2, "scores": [second[index] for index in ORDER for _ in range(2)]},
    }
    assert matching_vector(first, oracles) == {"workload_sha256": "first", "copies": 1, "request_position": 0}
    assert matching_vector([0.0, 11.0, 2.0, 13.0], oracles) is None


def test_hidden_retry_rejected_and_malformed_error_retained() -> None:
    row = {
        "label": "test",
        "responses": [httpx.Response(200), httpx.Response(200)],
        "result": {"scores": [{"item_id": f"test:{i}", "score": float(4 - i), "rank": i} for i in range(4)]},
    }
    with pytest.raises(AssertionError, match="hidden SDK retry"):
        validate_response(row)
    bad = describe_response(httpx.Response(503, content=b"not JSON"))
    assert bad["body_decode_error"] == "JSONDecodeError"
    assert bad["status"] == 503


def test_live_server_failure_and_failed_evidence_emit_still_cleanup(tmp_path: Path) -> None:
    snapshot = tiny_snapshot(tmp_path / "model")
    fixtures = tmp_path / "fixtures"
    prepare(fixtures, snapshot, smoke=True)
    output = tmp_path / "failed-live"
    script = """
import argparse, asyncio, sys
from pathlib import Path
from tools.benchmarks.cross_encoder_lora import worker
original = worker.Evidence.emit
def failing_emit(self, event, **fields):
    if event == 'original_state':
        raise OSError('injected post-load evidence failure')
    if event == 'client_wire_responses':
        raise OSError('injected cleanup evidence failure')
    return original(self, event, **fields)
worker.Evidence.emit = failing_emit
asyncio.run(worker.run(argparse.Namespace(source=Path(sys.argv[1]), fixtures=Path(sys.argv[2]),
    output=Path(sys.argv[3]), snapshot=None, arm='R', pair=0, smoke=True, deadline_s=60)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT), str(fixtures), str(output)],
        env=source_environment(ROOT),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )
    assert result.returncode != 0
    summary = json.loads((output / "trial.json").read_text())
    assert not summary["passed"]
    assert summary["shutdown_complete"]
    assert summary["error"]["message"] == "injected cleanup evidence failure"
    assert {row["message"] for row in summary["error"]["context"]} == {"injected post-load evidence failure"}
