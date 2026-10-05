"""Evidence admits only complete, fresh suites bounded by measured local noise."""

import base64
import hashlib
import importlib.util
import json
import os
import sys
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from sie_sdk._msgpack import packb, unpackb
from sie_server.config.equivalence import (
    EquivalenceRecord,
    ErrorMeasurement,
    ProbeCase,
    canonical_digest,
    measure_values,
    model_contract_digest,
    read_equivalence_record,
    remote_profile_contract_digest,
)
from sie_server.config.fleet_equivalence import (
    MAX_FLEET_BYTES,
    FleetEquivalenceRecord,
    equivalence_record_digest,
    read_equivalence_evidence,
    read_fleet_equivalence_record,
)
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import load_upstreams

NOW = datetime(2026, 10, 3, tzinfo=UTC)
OTHER_IDENTITY = "v1:sha256:" + "e" * 64
_PROBE_PATH = Path(__file__).resolve().parents[4] / "tools" / "remote_equivalence.py"
_PROBE_SPEC = importlib.util.spec_from_file_location("remote_equivalence_probe", _PROBE_PATH)
assert _PROBE_SPEC is not None
assert _PROBE_SPEC.loader is not None
probe = importlib.util.module_from_spec(_PROBE_SPEC)
sys.modules[_PROBE_SPEC.name] = probe
_PROBE_SPEC.loader.exec_module(probe)
_FLEET_SPEC = importlib.util.spec_from_file_location(
    "remote_fleet_equivalence", _PROBE_PATH.with_name("remote_fleet_equivalence.py")
)
assert _FLEET_SPEC is not None
assert _FLEET_SPEC.loader is not None
fleet_probe = importlib.util.module_from_spec(_FLEET_SPEC)
_FLEET_SPEC.loader.exec_module(fleet_probe)


def record(*, outputs: frozenset[str] = frozenset({"dense"})) -> EquivalenceRecord:
    cases = []
    for operation in ("encode", "score"):
        selected = {value for value in outputs if (value == "score") == (operation == "score")}
        if not selected:
            continue
        categories = [
            "short",
            "long",
            "boundary_before",
            "boundary_after",
            "query_prefix",
            "query_default",
            "empty_prefix",
            "document_prefix",
        ]
        if operation == "score":
            categories.append("score_scale")
        for category in categories:
            cases.append(
                {
                    "operation": operation,
                    "category": category,
                    "input_sha256": "a" * 64,
                    "token_counts": [513 if category == "boundary_after" else 512],
                    "outcomes": ["ok"] * 3,
                    "measurements": {
                        output: {"noise_floor": 0.0, "remote_error": 0.0, "values": 384} for output in selected
                    },
                }
            )
    return EquivalenceRecord.model_validate_json(
        json.dumps(
            {
                "version": 3,
                "measured_at": NOW.isoformat(),
                "upstream_name": "upstream",
                "upstream_model": "vendor/model",
                "upstream_contract_sha256": "b" * 64,
                "model_contract_sha256": "c" * 64,
                "probe_sources_sha256": "d" * 64,
                "remote_contract_sha256": "d" * 64,
                "remote_execution_sha256": "9" * 64,
                "local_observation_sha256": "e" * 64,
                "runtime_options_sha256": "f" * 64,
                "output_dtype": "float32",
                "local_identity": "v1:sha256:" + "f" * 64,
                "model": "local/model",
                "remote_profile": "remote",
                "context_length": 512,
                "outputs": list(outputs),
                "cases": cases,
            }
        )
    )


def test_noise_floor_is_measured_and_remote_must_match_both_local_runs() -> None:
    result = measure_values(np.array([1.0, 2.0]), np.array([1.25, 2.0]), np.array([1.125, 2.0]))
    assert result.noise_floor == 0.25
    assert result.remote_error == 0.125
    assert result.passed
    assert not measure_values(np.array([1.0]), np.array([1.25]), np.array([1.5])).passed


def test_deterministic_local_runs_require_identical_remote_values() -> None:
    assert measure_values(np.array([1.0]), np.array([1.0]), np.array([1.0])).passed
    assert not measure_values(np.array([1.0]), np.array([1.0]), np.array([1.000001])).passed


@pytest.mark.parametrize("remote", [np.array([np.nan]), np.array([np.inf]), np.array([1 + 1j]), np.array(["1"])])
def test_invalid_output_cannot_be_measured(remote: np.ndarray) -> None:
    with pytest.raises(ValueError, match="probe output"):
        measure_values(np.array([1.0]), np.array([1.0]), remote)


def test_output_shape_changes_and_empty_values_are_refused() -> None:
    for remote in (np.array([[1.0]]), np.array([])):
        with pytest.raises(ValueError, match="probe output layout"):
            measure_values(np.array([1.0]), np.array([1.0]), remote)


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_record_cannot_assert_nonfinite_or_negative_tolerances(value: float) -> None:
    with pytest.raises(ValidationError):
        ErrorMeasurement(noise_floor=value, remote_error=0.0, values=1)


def test_all_declared_outputs_and_operations_are_measured() -> None:
    assert record(outputs=frozenset({"dense", "sparse", "multivector", "score"})).passed


def test_missing_or_duplicate_categories_cannot_make_a_complete_suite() -> None:
    data = record().model_dump(mode="json")
    data["cases"][-1] = data["cases"][0]
    with pytest.raises(ValidationError):
        EquivalenceRecord.model_validate_json(json.dumps(data))


def test_success_requires_every_declared_output() -> None:
    data = record(outputs=frozenset({"dense", "sparse"})).model_dump(mode="json")
    del data["cases"][0]["measurements"]["sparse"]
    with pytest.raises(ValidationError):
        EquivalenceRecord.model_validate_json(json.dumps(data))


def test_matching_refusals_are_not_a_numerical_proof() -> None:
    data = record().model_dump(mode="json")
    for case in data["cases"]:
        case["outcomes"] = ["invalid_input"] * 3
        case["measurements"] = {}
    assert not EquivalenceRecord.model_validate_json(json.dumps(data)).passed


def test_matching_boundary_refusal_can_preserve_truncation_contract() -> None:
    data = record().model_dump(mode="json")
    after = next(case for case in data["cases"] if case["category"] == "boundary_after")
    after["outcomes"] = ["input_too_long"] * 3
    after["measurements"] = {}
    assert EquivalenceRecord.model_validate_json(json.dumps(data)).passed
    after["outcomes"][2] = "invalid_input"
    assert not EquivalenceRecord.model_validate_json(json.dumps(data)).passed


def test_boundary_labels_require_observed_token_counts() -> None:
    data = record().model_dump(mode="json")
    data["cases"][3]["token_counts"] = [512]
    with pytest.raises(ValidationError):
        EquivalenceRecord.model_validate_json(json.dumps(data))


def test_freshness_rejects_expired_future_naive_and_disabled_records() -> None:
    evidence = record()
    assert evidence.is_fresh(max_age_s=60, now=NOW + timedelta(seconds=60))
    assert not evidence.is_fresh(max_age_s=60, now=NOW + timedelta(seconds=61))
    assert not evidence.is_fresh(max_age_s=60, now=NOW - timedelta(seconds=1))
    assert not evidence.is_fresh(max_age_s=60, now=NOW.replace(tzinfo=None))
    assert not evidence.is_fresh(max_age_s=0, now=NOW)


def test_rejected_or_malformed_outputs_cannot_claim_measurements() -> None:
    with pytest.raises(ValidationError):
        ProbeCase(
            operation="encode",
            category="short",
            input_sha256="a" * 64,
            token_counts=(5,),
            outcomes=("shape_mismatch",) * 3,
            measurements={"dense": ErrorMeasurement(noise_floor=0.0, remote_error=0.0, values=384)},
        )


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [1] * (len(text.split()) + (2 if add_special_tokens else 0))


@pytest.mark.parametrize(
    ("remote_offset", "contract_mismatch", "instance_mismatch", "remote_execution"),
    [
        (0.0, False, None, "8" * 64),
        (0.01, False, None, "8" * 64),
        (0.0, True, None, "8" * 64),
        (0.0, False, "local", "8" * 64),
        (0.0, False, "remote", "8" * 64),
        (0.0, False, None, None),
    ],
)
def test_full_cli_runs_real_sdk_transport_and_only_writes_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_offset: float,
    contract_mismatch: bool,
    instance_mismatch: str | None,
    remote_execution: str | None,
) -> None:
    requests: list[tuple[dict[str, Any], bool]] = []
    remote_contract: str | None = None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.reply(
                json.dumps(
                    {
                        "name": "local/model",
                        "revision": "a" * 40,
                        "max_sequence_length": 32,
                        "profiles": {
                            "default": {"identity": "v1:sha256:" + "f" * 64, "runtime_instance_id": "a" * 64},
                            "remote": {
                                "remote_contract_sha256": remote_contract,
                                "remote_execution_sha256": remote_execution,
                            },
                        },
                    }
                ).encode(),
                "application/json",
            )

        def do_POST(self) -> None:
            body = unpackb(self.rfile.read(int(self.headers["Content-Length"])), numeric_arrays=False)
            forbid = self.headers.get("X-SIE-Remote") == "forbid"
            requests.append((body, forbid))
            offset = 0.0 if forbid else remote_offset
            rows = [
                {"id": item["id"], "dense": {"dims": 2, "values": np.asarray([1.0 + offset, 2.0], dtype=np.float32)}}
                for item in body["items"]
            ]
            self.reply(packb({"model": "local/model", "items": rows}), "application/msgpack", remote=not forbid)

        def reply(self, body: bytes, content_type: str, *, remote: bool = False) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-SIE-Served-By", "remote" if remote else "local")
            mismatch = instance_mismatch == ("remote" if remote else "local")
            self.send_header("X-SIE-Runtime-Instance", ("b" if mismatch else "a") * 64)
            if remote:
                self.send_header("X-SIE-Upstream", "upstream")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    config_file = tmp_path / "model.yaml"
    config_file.write_text(
        """sie_id: local/model
hf_id: weights/model
hf_revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
inputs: {text: true}
tasks: {encode: {dense: {dim: 2}}}
max_sequence_length: 32
profiles:
  default:
    adapter_path: sie_server.adapters.bge_m3:BGEM3Adapter
    max_batch_tokens: 8192
    compute_precision: float32
    adapter_options:
      runtime: {normalize: true}
  remote:
    adapter_path: sie_server.adapters.remote.openai:OpenAIUpstreamAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime: {upstream: upstream, upstream_model: vendor/model}
      runtime: {normalize: true}
"""
    )
    upstream_file = tmp_path / "upstreams.yaml"
    upstream_file.write_text(
        """upstreams:
  upstream:
    kind: openai
    base_url: https://vendor.example/v1
    endpoints: [embeddings]
    rate_cap: {requests_per_minute: 600, max_concurrency: 32}
"""
    )
    monkeypatch.setattr(probe, "load_tokenizer", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(probe, "_execution_code", lambda: {"sources": "e" * 64})
    config = ModelConfig.model_validate(yaml.safe_load(config_file.read_bytes()))
    remote_contract = remote_profile_contract_digest(config, "remote", load_upstreams(upstream_file))
    if contract_mismatch:
        remote_contract = "0" * 64
    output = tmp_path / "evidence.json"
    try:
        result = probe.main(
            [
                "--model-file",
                str(config_file),
                "--upstreams-file",
                str(upstream_file),
                "--local-url",
                f"http://127.0.0.1:{server.server_port}",
                "--output",
                str(output),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    if contract_mismatch or remote_execution is None:
        assert result == 2
        assert not requests
        assert not output.exists()
        return
    if instance_mismatch:
        assert result == 2
        assert len(requests) == (3 if instance_mismatch == "remote" else 1)
        assert not output.exists()
        return
    assert result == (0 if remote_offset == 0 else 1)
    evidence = EquivalenceRecord.model_validate_json(output.read_bytes())
    assert evidence.passed == (remote_offset == 0)
    assert evidence.version == 3
    assert evidence.remote_execution_sha256 == remote_execution
    assert "local_instance_id" not in evidence.model_dump()
    assert evidence.runtime_options_sha256 == canonical_digest({"normalize": True})
    assert all(body["params"]["options"]["normalize"] is True for body, _ in requests)
    assert all(body["params"]["output_dtype"] == "float32" for body, _ in requests)
    assert len(requests) == 24
    assert [forbid for _, forbid in requests] == [True, True, False] * 8
    assert [body["params"]["options"]["profile"] for body, _ in requests] == ["default", "default", "remote"] * 8
    serialized = output.read_text()
    assert "retrieval " not in serialized
    assert "Represent this text" not in serialized
    assert 'values"' in serialized  # Measurement count, never vector contents.
    assert "1.01" not in serialized


@pytest.mark.parametrize(
    ("local", "remote"),
    [({}, {"normalize": False}), ({"normalize": True}, {"normalize": False}), ({"normalize": True}, {})],
)
def test_probe_refuses_a_remote_profile_that_runs_with_other_runtime_options(
    local: dict[str, Any], remote: dict[str, Any]
) -> None:
    config = ModelConfig.model_validate(
        {
            "sie_id": "local/model",
            "hf_id": "weights/model",
            "hf_revision": "a" * 40,
            "inputs": {"text": True},
            "tasks": {"encode": {"dense": {"dim": 2}}},
            "max_sequence_length": 32,
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.bge_m3:BGEM3Adapter",
                    "max_batch_tokens": 8192,
                    "compute_precision": "float32",
                    "adapter_options": {"runtime": local},
                },
                "remote": {
                    "adapter_path": "sie_server.adapters.remote.openai:OpenAIUpstreamAdapter",
                    "max_batch_tokens": 8192,
                    "adapter_options": {
                        "loadtime": {"upstream": "upstream", "upstream_model": "vendor/model"},
                        "runtime": remote,
                    },
                },
            },
        }
    )
    local_client, remote_client = MagicMock(), MagicMock()
    with pytest.raises(ValueError, match="remote profile runtime differs"):
        probe.run_probe(config, {}, local_client, remote_client, "remote")
    assert not local_client.mock_calls
    assert not remote_client.mock_calls


def test_boundary_counts_include_actual_default_and_empty_instructions() -> None:
    tokenizer = _Tokenizer()
    cases = probe._cases(tokenizer, 64, default_instruction="profile prefix")
    for case in cases:
        prefix = "profile prefix" if case.instruction is None else case.instruction
        assert case.token_counts == tuple(
            len(tokenizer.encode(f"{prefix} {text}", add_special_tokens=True)) for text in case.texts
        )
    assert cases[2].token_counts[0] <= 64 < cases[3].token_counts[0]


def test_sparse_layout_and_score_order_are_compared_by_identity() -> None:
    sparse = probe._values(
        [
            [{"sparse": {"indices": [1, 2], "values": [0.1, 0.2]}}],
            [{"sparse": {"indices": [2, 1], "values": [0.2, 0.1]}}],
            [{"sparse": {"indices": [1, 3], "values": [0.1, 0.2]}}],
        ],
        "sparse",
        dimension=4,
    )
    assert not measure_values(*sparse).passed
    scores = probe._values(
        [
            {"scores": [{"item_id": "b", "score": 0.2}, {"item_id": "a", "score": 0.1}]},
            {"scores": [{"item_id": "a", "score": 0.1}, {"item_id": "b", "score": 0.2}]},
            {"scores": [{"item_id": "b", "score": 0.2}, {"item_id": "a", "score": 0.1}]},
        ],
        "score",
    )
    assert measure_values(*scores).passed


@pytest.mark.parametrize("index", [-1, 1.5, True, "not-an-index", 4])
def test_sparse_indices_cannot_claim_invalid_vocabulary_positions(index: Any) -> None:
    values = [[{"sparse": {"indices": [index], "values": [0.1]}}]] * 3
    with pytest.raises(ValueError, match="sparse output"):
        probe._values(values, "sparse", dimension=4)


def test_consistently_wrong_dimensions_are_not_equivalence() -> None:
    values = [[{"dense": np.array([1.0])}]] * 3
    with pytest.raises(ValueError, match="declared dimensions"):
        probe._values(values, "dense", dimension=4)


def test_integer_conversion_cannot_hide_numerical_differences() -> None:
    with pytest.raises(ValueError, match="floating precision"):
        measure_values(np.array([2**53]), np.array([2**53]), np.array([2**53 + 1]))


@pytest.mark.parametrize("shape", [(1, 1), (0, 4), (33, 4), (4,)])
def test_multivector_dimensions_and_token_bounds_are_required(shape: tuple[int, ...]) -> None:
    values = [[{"multivector": np.ones(shape)}]] * 3
    with pytest.raises(ValueError, match="declared dimensions"):
        probe._values(values, "multivector", dimension=4, context_length=32)


@pytest.mark.parametrize("value", [2**53 + 1, True, "1.0", float("nan")])
def test_sparse_padding_cannot_coerce_invalid_values_into_equivalence(value: Any) -> None:
    values = [
        [{"sparse": {"indices": [0], "values": [float(2**53)]}}],
        [{"sparse": {"indices": [0, 1], "values": [float(2**53), 0.0]}}],
        [{"sparse": {"indices": [0, 2], "values": [value, 0.0]}}],
    ]
    with pytest.raises(ValueError, match="finite floating"):
        probe._values(values, "sparse", dimension=4)


def test_mixed_integer_scores_cannot_hide_conversion_error() -> None:
    values = [{"scores": [{"item_id": "a", "score": 2**53 + 1}, {"item_id": "b", "score": 0.0}]}] * 3
    with pytest.raises(ValueError, match="floating values"):
        probe._values(values, "score")


def _fleet_record(**changes: Any) -> EquivalenceRecord:
    data = record(outputs=frozenset({"dense", "sparse"})).model_dump(mode="json")
    data.update(changes)
    return EquivalenceRecord.model_validate_json(json.dumps(data))


def test_bundle_holds_one_record_per_execution_identity() -> None:
    first = _fleet_record()
    second = _fleet_record(local_identity=OTHER_IDENTITY)
    fleet = FleetEquivalenceRecord(records=(first, second))
    assert set(fleet.record_digests) == {first.local_identity, OTHER_IDENTITY}
    assert fleet.passed


@pytest.mark.parametrize("count", [0, 257])
def test_fleet_inventory_has_bounded_nonempty_membership(count: int) -> None:
    with pytest.raises(ValidationError):
        FleetEquivalenceRecord(records=(record(),) * count)


def test_fleet_inventory_rejects_a_repeated_execution_identity() -> None:
    with pytest.raises(ValidationError, match="repeat an execution identity"):
        FleetEquivalenceRecord(records=(record(), record()))
    with pytest.raises(ValidationError, match="repeat an execution identity"):
        remeasured = _fleet_record(measured_at=(NOW + timedelta(seconds=1)).isoformat())
        FleetEquivalenceRecord(records=(_fleet_record(), remeasured))


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "other/model"},
        {"remote_profile": "other"},
        {"upstream_name": "other"},
        {"upstream_model": "other/model"},
        {"upstream_contract_sha256": "0" * 64},
        {"model_contract_sha256": "0" * 64},
        {"remote_contract_sha256": "0" * 64},
        {"remote_execution_sha256": "0" * 64},
        {"runtime_options_sha256": "0" * 64},
        {"probe_sources_sha256": "0" * 64},
        {"context_length": 513},
    ],
)
def test_fleet_inventory_rejects_mixed_comparison_contracts(changes: dict[str, Any]) -> None:
    # The before/after boundary also moves when the declared context changes.
    if "context_length" in changes:
        cases = record(outputs=frozenset({"dense", "sparse"})).model_dump(mode="json")["cases"]
        for case in cases:
            if case["category"] == "boundary_after":
                case["token_counts"] = [514]
        changes = {**changes, "cases": cases}
    with pytest.raises(ValidationError, match="same comparison contract"):
        FleetEquivalenceRecord(records=(_fleet_record(), _fleet_record(local_identity=OTHER_IDENTITY, **changes)))


def test_fleet_inventory_cannot_mix_probe_inputs_or_token_counts() -> None:
    first = _fleet_record()
    for field, value in (("input_sha256", "0" * 64), ("token_counts", [1])):
        cases = first.model_dump(mode="json")["cases"]
        cases[0][field] = value
        with pytest.raises(ValidationError, match="same comparison contract"):
            FleetEquivalenceRecord(records=(first, _fleet_record(local_identity=OTHER_IDENTITY, cases=cases)))


def test_fleet_digests_bind_all_measurements_but_ignore_record_case_and_output_order() -> None:
    first = _fleet_record()
    data = first.model_dump(mode="json")
    data["outputs"].reverse()
    data["cases"].reverse()
    reordered = EquivalenceRecord.model_validate_json(json.dumps(data))
    assert equivalence_record_digest(first) == equivalence_record_digest(reordered)
    second = _fleet_record(local_identity=OTHER_IDENTITY)
    fleet = FleetEquivalenceRecord(records=(first, second))
    assert fleet.digest == FleetEquivalenceRecord(records=(second, reordered)).digest
    data["cases"][0]["measurements"]["dense"]["remote_error"] = 0.1
    failed = EquivalenceRecord.model_validate_json(json.dumps(data))
    assert equivalence_record_digest(first) != equivalence_record_digest(failed)
    assert fleet.digest != FleetEquivalenceRecord(records=(failed, second)).digest
    assert not FleetEquivalenceRecord(records=(failed, second)).passed


@pytest.mark.parametrize("offset", [-3601, 1])
def test_one_expired_or_future_process_invalidates_fleet_freshness(offset: int) -> None:
    fleet = FleetEquivalenceRecord(
        records=(
            _fleet_record(),
            _fleet_record(local_identity=OTHER_IDENTITY, measured_at=(NOW + timedelta(seconds=offset)).isoformat()),
        )
    )
    assert not fleet.is_fresh(max_age_s=3600, now=NOW)


@pytest.mark.parametrize("age", [True, 0, 604801, 1.5])
def test_fleet_age_is_strict_and_bounded(age: Any) -> None:
    assert not FleetEquivalenceRecord(records=(record(),)).is_fresh(max_age_s=age, now=NOW)


def test_fleet_age_accepts_a_seven_day_window() -> None:
    fleet = FleetEquivalenceRecord(records=(record(),))
    assert fleet.is_fresh(max_age_s=604800, now=NOW + timedelta(days=7))
    assert not fleet.is_fresh(max_age_s=604800, now=NOW + timedelta(days=7, seconds=1))


def test_fleet_cli_preserves_failed_evidence_and_never_overwrites(tmp_path: Path) -> None:
    cases = record().model_dump(mode="json")["cases"]
    cases[0]["measurements"]["dense"]["remote_error"] = 0.1
    failed = _fleet_record(cases=cases, outputs=["dense"], measured_at=datetime.now(UTC).isoformat())
    source, output = tmp_path / "worker.json", tmp_path / "fleet.json"
    source.write_text(failed.model_dump_json())
    args = ["--record", str(source), "--output", str(output)]
    assert fleet_probe.main(args) == 1
    inventory = read_fleet_equivalence_record(output)
    assert not inventory.passed
    assert inventory.record_digests == {failed.local_identity: equivalence_record_digest(failed)}
    original = output.read_bytes()
    assert fleet_probe.main(args) == 2
    assert output.read_bytes() == original


def test_fleet_cli_collects_fresh_records_and_records_expiry(tmp_path: Path) -> None:
    source = tmp_path / "worker.json"
    source.write_text(_fleet_record(measured_at=datetime.now(UTC).isoformat()).model_dump_json())
    assert fleet_probe.main(["--record", str(source), "--output", str(tmp_path / "fresh.json")]) == 0
    source.write_text(_fleet_record(measured_at=(datetime.now(UTC) - timedelta(days=2)).isoformat()).model_dump_json())
    assert fleet_probe.main(["--record", str(source), "--output", str(tmp_path / "expired.json")]) == 1
    assert read_fleet_equivalence_record(tmp_path / "expired.json").passed


def test_fleet_cli_failed_write_leaves_no_partial_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "worker.json", tmp_path / "fleet.json"
    source.write_text(_fleet_record(measured_at=datetime.now(UTC).isoformat()).model_dump_json())

    def failed_write(path: Path, data: bytes) -> None:
        with path.open("wb") as stream:
            stream.write(data[:10])
        raise OSError("simulated write failure")

    monkeypatch.setattr(Path, "write_bytes", failed_write)
    assert fleet_probe.main(["--record", str(source), "--output", str(output)]) == 2
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [source]


def test_fleet_cli_never_replaces_a_symlink(tmp_path: Path) -> None:
    source, output = tmp_path / "worker.json", tmp_path / "fleet.json"
    source.write_text(_fleet_record(measured_at=datetime.now(UTC).isoformat()).model_dump_json())
    output.symlink_to(source)
    original = source.read_bytes()
    assert fleet_probe.main(["--record", str(source), "--output", str(output)]) == 2
    assert output.is_symlink()
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    ("reader", "limit"),
    [
        (read_equivalence_record, 512 << 10),
        (read_fleet_equivalence_record, MAX_FLEET_BYTES),
        (read_equivalence_evidence, MAX_FLEET_BYTES),
    ],
)
def test_evidence_readers_refuse_oversized_files_and_nonblocking_fifos(tmp_path: Path, reader: Any, limit: int) -> None:
    source = tmp_path / "evidence.json"
    source.write_bytes(b" " * (limit + 1))
    with pytest.raises(ValueError, match="byte limit"):
        reader(source)
    source.unlink()
    os.mkfifo(source)
    with pytest.raises(ValueError, match="regular file"):
        reader(source)


def test_evidence_file_holds_one_record_or_a_bundle(tmp_path: Path) -> None:
    single, bundle = tmp_path / "record.json", tmp_path / "bundle.json"
    first, second = _fleet_record(), _fleet_record(local_identity=OTHER_IDENTITY)
    single.write_text(first.model_dump_json())
    bundle.write_text(FleetEquivalenceRecord(records=(first, second)).model_dump_json())
    assert read_equivalence_evidence(single) == (first,)
    assert read_equivalence_evidence(bundle) == (first, second)
    for invalid in ({"version": 2, "records": []}, {"version": 1, "records": [first.model_dump(mode="json")]}, {}):
        single.write_text(json.dumps(invalid))
        with pytest.raises(ValidationError):
            read_equivalence_evidence(single)


def test_process_bound_records_are_refused() -> None:
    data = record().model_dump(mode="json")
    data.update(version=2, local_instance_id="a" * 64)
    with pytest.raises(ValidationError):
        EquivalenceRecord.model_validate_json(json.dumps(data))


_CLUSTER_MODEL = """sie_id: local/model
hf_id: weights/model
hf_revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
inputs: {text: true}
tasks: {encode: {dense: {dim: 2}}}
max_sequence_length: 32
profiles:
  default:
    adapter_path: sie_server.adapters.bge_m3:BGEM3Adapter
    max_batch_tokens: 8192
    compute_precision: float32
  remote:
    adapter_path: sie_server.adapters.remote.openai:OpenAIUpstreamAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime: {upstream: upstream, upstream_model: vendor/model}
"""
_CLUSTER_UPSTREAMS = """upstreams:
  upstream:
    kind: openai
    base_url: https://vendor.example/v1
    endpoints: [embeddings]
    rate_cap: {requests_per_minute: 600, max_concurrency: 32}
"""
_IDENTITY_L4 = "v2:sha256:" + "1" * 64
_IDENTITY_A100 = "v2:sha256:" + "2" * 64
_REMOTE_EXECUTION = "7" * 64


def _websocket_frame(payload: bytes, opcode: int) -> bytes:
    size = len(payload)
    if size < 126:
        header = bytes([0x80 | opcode, size])
    elif size < 1 << 16:
        header = bytes([0x80 | opcode, 126]) + size.to_bytes(2, "big")
    else:
        header = bytes([0x80 | opcode, 127]) + size.to_bytes(8, "big")
    return header + payload


def _worker(name: str, gpu: str, bundle: str, profiles: list[dict[str, Any]], instance: str) -> dict[str, Any]:
    return {
        "name": name,
        "url": f"http://{name}:8080",
        "gpu": gpu,
        "bundle": bundle,
        "loaded_models": [],
        "numerical_process_inventory": {
            "observed_at_unix_ms": 1,
            "children": [
                {
                    "child_index": 0,
                    "status": "observed",
                    "snapshot": {"runtime_instance_id": instance, "complete": True, "profiles": profiles},
                }
            ],
        },
    }


def _cluster_fixture(tmp_path: Path) -> tuple[Path, Path, ModelConfig, str, str]:
    config_file, upstream_file = tmp_path / "model.yaml", tmp_path / "upstreams.yaml"
    config_file.write_text(_CLUSTER_MODEL)
    upstream_file.write_text(_CLUSTER_UPSTREAMS)
    config = ModelConfig.model_validate(yaml.safe_load(_CLUSTER_MODEL))
    remote_contract = remote_profile_contract_digest(config, "remote", load_upstreams(upstream_file))
    assert remote_contract is not None
    return config_file, upstream_file, config, model_contract_digest(config), remote_contract


def _fleet(contract: str, remote_contract: str, *, identities: tuple[str, str]) -> list[dict[str, Any]]:
    def local(identity: str) -> list[dict[str, Any]]:
        return [{"model_id": "local/model", "model_contract_sha256": contract, "local_identity": identity}]

    return [
        _worker("l4-0", "l4", "default", local(identities[0]), "a" * 64),
        _worker("a100-0", "a100-80gb", "default", local(identities[1]), "b" * 64),
        _worker(
            "remote-0",
            "cpu",
            "remote",
            [
                {
                    "model_id": "local/model",
                    "model_contract_sha256": contract,
                    "local_identity": None,
                    "remote_contract_sha256": remote_contract,
                    "remote_execution_sha256": _REMOTE_EXECUTION,
                }
            ],
            "c" * 64,
        ),
    ]


events: list[str] = []


def _run_cluster_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, statuses: list[list[dict[str, Any]]], extra: list[str]
) -> tuple[int, list[tuple[dict[str, Any], dict[str, str]]], Path]:
    config_file, upstream_file = tmp_path / "model.yaml", tmp_path / "upstreams.yaml"
    requests: list[tuple[dict[str, Any], dict[str, str]]] = []
    connections: list[int] = []
    events.clear()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if self.path != "/ws/cluster-status" or self.headers.get("Upgrade", "").lower() != "websocket":
                self.send_error(404)
                return
            accept = base64.b64encode(
                hashlib.sha1(  # noqa: S324 - the WebSocket handshake digest is defined as SHA-1
                    (self.headers["Sec-WebSocket-Key"] + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
                ).digest()
            ).decode()
            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            workers = statuses[min(len(connections), len(statuses) - 1)]
            connections.append(1)
            events.append("status")
            self.wfile.write(_websocket_frame(json.dumps({"timestamp": 1.0, "workers": workers}).encode(), 0x1))
            self.wfile.write(_websocket_frame(b"\x03\xe8", 0x8))
            self.wfile.flush()
            self.close_connection = True

        def do_POST(self) -> None:
            body = unpackb(self.rfile.read(int(self.headers["Content-Length"])), numeric_arrays=False)
            headers = {key.lower(): value for key, value in self.headers.items() if key.lower().startswith("x-sie")}
            requests.append((body, headers))
            events.append("wake" if body["items"][0]["id"] == "probe-wake" else "probe")
            remote = self.headers.get("X-SIE-Remote") != "forbid"
            rows = [
                {"id": item["id"], "dense": {"dims": 2, "values": np.asarray([1.0, 2.0], dtype=np.float32)}}
                for item in body["items"]
            ]
            payload = packb({"model": "local/model", "items": rows})
            self.send_response(200)
            self.send_header("Content-Type", "application/msgpack")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-SIE-Served-By", "remote" if remote else "local")
            if remote:
                self.send_header("X-SIE-Upstream", "upstream")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    monkeypatch.setattr(probe, "load_tokenizer", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(probe, "_execution_code", lambda: {"sources": "e" * 64})
    output = tmp_path / "evidence.json"
    try:
        result = probe.main(
            [
                "--model-file",
                str(config_file),
                "--upstreams-file",
                str(upstream_file),
                "--local-url",
                f"http://127.0.0.1:{server.server_port}",
                "--output",
                str(output),
                "--cluster",
                *extra,
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return result, requests, output


def test_cluster_probe_attributes_observations_to_one_identity_from_cluster_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, contract, remote_contract = _cluster_fixture(tmp_path)
    fleet = _fleet(contract, remote_contract, identities=(_IDENTITY_L4, _IDENTITY_L4))
    result, requests, output = _run_cluster_probe(tmp_path, monkeypatch, [fleet], [])
    assert result == 0
    evidence = EquivalenceRecord.model_validate_json(output.read_bytes())
    assert evidence.local_identity == _IDENTITY_L4
    assert evidence.remote_execution_sha256 == _REMOTE_EXECUTION
    assert evidence.remote_contract_sha256 == remote_contract
    assert evidence.local_observation_sha256 == canonical_digest({"identity": _IDENTITY_L4, "revision": "a" * 40})
    assert events == ["wake", "wake", "status", *["probe"] * 24, "status"]
    assert [headers.get("x-sie-remote") for _, headers in requests] == ["forbid", None] + ["forbid", "forbid", None] * 8
    assert all("x-sie-machine-profile" not in headers for _, headers in requests)
    assert all(body["params"]["options"]["profile"] in ("default", "remote") for body, _ in requests)


def test_cluster_probe_refuses_two_identities_unless_one_machine_profile_is_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, contract, remote_contract = _cluster_fixture(tmp_path)
    fleet = _fleet(contract, remote_contract, identities=(_IDENTITY_L4, _IDENTITY_A100))
    result, _, output = _run_cluster_probe(tmp_path, monkeypatch, [fleet], [])
    assert (result, events, output.exists()) == (2, ["wake", "wake", "status"], False)
    result, requests, output = _run_cluster_probe(tmp_path, monkeypatch, [fleet], ["--gpu", "a100-80gb"])
    assert result == 0
    assert EquivalenceRecord.model_validate_json(output.read_bytes()).local_identity == _IDENTITY_A100
    local = [headers for body, headers in requests if body["params"]["options"]["profile"] == "default"]
    remote = [headers for body, headers in requests if body["params"]["options"]["profile"] == "remote"]
    assert {headers.get("x-sie-machine-profile") for headers in local} == {"a100-80gb"}
    assert {headers.get("x-sie-machine-profile") for headers in remote} == {None}


def test_cluster_probe_refuses_a_fleet_that_changes_during_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, contract, remote_contract = _cluster_fixture(tmp_path)
    before = _fleet(contract, remote_contract, identities=(_IDENTITY_L4, _IDENTITY_L4))
    after = json.loads(json.dumps(before))
    after[0]["numerical_process_inventory"]["children"][0]["snapshot"]["runtime_instance_id"] = "d" * 64
    result, _, output = _run_cluster_probe(tmp_path, monkeypatch, [before, after], [])
    assert result == 2
    assert events == ["wake", "wake", "status", *["probe"] * 24, "status"]
    assert not output.exists()


@pytest.mark.parametrize("change", ["remote_contract", "remote_execution", "model_contract", "unattributed", "status"])
def test_cluster_probe_refuses_incomplete_or_mismatched_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    _, _, _, contract, remote_contract = _cluster_fixture(tmp_path)
    fleet = _fleet(contract, remote_contract, identities=(_IDENTITY_L4, _IDENTITY_L4))
    remote_profile = fleet[2]["numerical_process_inventory"]["children"][0]["snapshot"]["profiles"][0]
    if change == "remote_contract":
        remote_profile["remote_contract_sha256"] = "0" * 64
    elif change == "remote_execution":
        remote_profile["remote_execution_sha256"] = None
    elif change == "model_contract":
        fleet[0]["numerical_process_inventory"]["children"][0]["snapshot"]["profiles"][0]["model_contract_sha256"] = (
            "0" * 64
        )
    elif change == "unattributed":
        fleet.append({**_worker("legacy-0", "l4", "default", [], "e" * 64), "loaded_models": ["local/model"]})
        fleet[-1]["numerical_process_inventory"] = None
    else:
        fleet[1]["numerical_process_inventory"]["children"][0]["status"] = "incomplete"
    result, _, output = _run_cluster_probe(tmp_path, monkeypatch, [fleet], [])
    assert (result, events, output.exists()) == (2, ["wake", "wake", "status"], False)


@pytest.mark.parametrize("malformed", ["inventory", "children", "child", "snapshot", "profiles", "profile"])
def test_cluster_probe_refuses_malformed_status_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], malformed: str
) -> None:
    _, _, _, contract, remote_contract = _cluster_fixture(tmp_path)
    fleet = _fleet(contract, remote_contract, identities=(_IDENTITY_L4, _IDENTITY_L4))
    inventory = fleet[0]["numerical_process_inventory"]
    if malformed == "inventory":
        fleet[0]["numerical_process_inventory"] = ["not", "a", "map"]
    elif malformed == "children":
        inventory["children"] = {"0": inventory["children"][0]}
    elif malformed == "child":
        inventory["children"] = ["not a child"]
    elif malformed == "snapshot":
        inventory["children"][0]["snapshot"] = ["not", "a", "snapshot"]
    elif malformed == "profiles":
        inventory["children"][0]["snapshot"]["profiles"] = {"model_id": "local/model"}
    else:
        inventory["children"][0]["snapshot"]["profiles"] = ["not a profile"]
    result, _, output = _run_cluster_probe(tmp_path, monkeypatch, [fleet], [])
    assert (result, events, output.exists()) == (2, ["wake", "wake", "status"], False)
    assert "Traceback" not in capsys.readouterr().err


def test_machine_profile_selection_requires_cluster_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        probe.main(
            [
                "--model-file",
                str(tmp_path / "model.yaml"),
                "--upstreams-file",
                str(tmp_path / "upstreams.yaml"),
                "--local-url",
                "http://127.0.0.1:9",
                "--output",
                str(tmp_path / "evidence.json"),
                "--gpu",
                "l4",
            ]
        )
