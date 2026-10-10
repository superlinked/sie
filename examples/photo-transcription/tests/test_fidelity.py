"""Credential-free source integrity and real-SDK collector controls."""

from __future__ import annotations

import json

import collect as collector
import dataset
import httpx
import msgpack
import pytest
from fidelity import load_packet, native_report, recorded_report


@pytest.fixture
def packet(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    directory.mkdir()
    records = []
    providers = []
    for index in range(24):
        image = b"\xff\xd8\xff" + f"complete-original-photo-{index}".encode() + b"\xff\xd9"
        image_path = f"images/{index}.jpg"
        (directory / "images").mkdir(exist_ok=True)
        (directory / image_path).write_bytes(image)
        text = f"Source value {index}"
        row = {
            "id": f"photo-{index}",
            "source_cluster": f"document-{index}",
            "domain": "receipt" if index < 12 else "handwriting",
            "gold_text": text,
            "gold_sha256": dataset.digest(text.encode()),
            "critical_values": [{"role": "value", "surface": str(index)}],
            "reviewed_lines": [text],
            "source_image": {"path": image_path, "sha256": dataset.digest(image)},
            "input_image": {"path": image_path, "sha256": dataset.digest(image)},
        }
        records.append(row)
        for arm in ("luna", "sol"):
            providers.append(
                {
                    "arm": arm,
                    "case_id": row["id"],
                    "text": text,
                    "image_sha256": row["input_image"]["sha256"],
                    "transcription_sha256": dataset.digest(text.encode()),
                    "phase": "main",
                    "attempts": 1,
                    "source_grounded": {
                        "source_grounded_primary_outcome": "pass" if arm == "sol" or index < 16 else "fail"
                    },
                }
            )
    (directory / "manifest.json").write_text(json.dumps({"n": 24, "records": records}))
    (directory / "provider-projection.json").write_text(json.dumps({"provider_records": providers}))
    checksum_bytes = "".join(
        f"{dataset.digest(path.read_bytes())}  {path.relative_to(directory).as_posix()}\n"
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ).encode()
    (directory / "SHA256SUMS").write_bytes(checksum_bytes)
    monkeypatch.setattr(dataset, "CHECKSUM_SHA256", dataset.digest(checksum_bytes))
    return directory


def test_saved_counts_and_every_packet_file_are_verified(packet):
    _, projection = load_packet(packet)
    report = recorded_report(projection)
    assert report["luna"]["passed"] == 16
    assert report["sol"]["passed"] == 24
    assert report["native"] is None
    path = packet / "provider-projection.json"
    path.write_text(path.read_text().replace('"pass"', '"fail"', 1))
    with pytest.raises(ValueError, match="Packet file changed"):
        load_packet(packet)


def test_checksum_list_cannot_be_rewritten(packet):
    (packet / "SHA256SUMS").write_text("0" * 64 + "  manifest.json\n")
    with pytest.raises(ValueError, match="pinned source packet"):
        load_packet(packet)


@pytest.mark.parametrize("relative", ["../secret", "/secret"])
def test_packet_paths_cannot_escape(packet, relative):
    with pytest.raises(ValueError, match="relative"):
        dataset.packet_path(packet, relative)


def calls_and_decisions(manifest):
    calls = []
    decisions = []
    for source in manifest["records"]:
        calls.append(
            {
                "case_id": source["id"],
                "image_sha256": source["input_image"]["sha256"],
                "attempts": 1,
                "status": "completed",
                "text": source["gold_text"],
                "transcription_sha256": source["gold_sha256"],
            }
        )
        decisions.append(
            {
                "case_id": source["id"],
                "image_sha256": source["input_image"]["sha256"],
                "transcription_sha256": source["gold_sha256"],
                "reviewer": "Synthetic test reviewer",
                "critical_outcomes": ["preserved"] * len(source["critical_values"]),
                "line_outcomes": ["represented"] * len(source["reviewed_lines"]),
            }
        )
    return calls, decisions


def test_unknowns_and_submitted_failures_are_distinct(packet):
    manifest, _ = load_packet(packet)
    calls, decisions = calls_and_decisions(manifest)
    calls[0].update(status="unattempted", attempts=0)
    calls[1]["status"] = "error"
    calls[2]["status"] = "empty"
    decisions = decisions[4:]
    report = native_report(manifest, calls, decisions)
    assert (report["passed"], report["failed"], report["unattempted"], report["unreviewed"]) == (20, 2, 1, 1)
    assert report["complete_reviewed_cohort"] is False
    with pytest.raises(ValueError, match="every frozen case"):
        native_report(manifest, calls[:-1], decisions)


def test_exact_transcription_binding_and_source_roles(packet):
    manifest, _ = load_packet(packet)
    calls, decisions = calls_and_decisions(manifest)
    decisions[0]["critical_outcomes"] = ["misassociated"]
    assert native_report(manifest, calls, decisions)["failed"] == 1
    decisions[1]["transcription_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact transcription binding"):
        native_report(manifest, calls, decisions)


def test_only_declared_blurred_source_can_be_excluded(packet):
    manifest, _ = load_packet(packet)
    calls, decisions = calls_and_decisions(manifest)
    decisions[0]["critical_outcomes"] = ["not_assessable"]
    with pytest.raises(ValueError, match="undeclared critical"):
        native_report(manifest, calls, decisions)
    source = manifest["records"][0]
    source.update(id="cord/train-0012", critical_values=[{}] * 16, reviewed_lines=["blurred logo", "visible line"])
    calls[0]["case_id"] = source["id"]
    decisions[0].update(
        case_id=source["id"],
        critical_outcomes=["preserved"] * 15 + ["not_assessable"],
        line_outcomes=["not_assessable", "represented"],
    )
    assert native_report(manifest, calls, decisions)["passed"] == 24


def mock_transport(monkeypatch, handler):
    original = httpx.Client

    def configured(**kwargs):
        return original(**kwargs, transport=httpx.MockTransport(handler), trust_env=False)

    monkeypatch.setattr(httpx, "Client", configured)
    monkeypatch.delenv("SIE_API_KEY", raising=False)


@pytest.mark.parametrize("base_url", ["https://ocr.example.test", "https://ocr.example.test/prefix"])
@pytest.mark.parametrize(
    ("model", "instruction", "max_new_tokens"),
    [("test/native", "Text Recognition:", 8192), ("lightonai/LightOnOCR-3-4B", None, 12288)],
)
def test_actual_sdk_sends_full_jpeg_and_exact_controls_once(
    packet, tmp_path, monkeypatch, base_url, model, instruction, max_new_tokens
):
    seen = []

    def handle(request):
        seen.append(request)
        payload = msgpack.unpackb(request.content, raw=False)
        item = payload["items"][0]
        body = {
            "model": model,
            "items": [
                {"id": item["id"], "entities": [{"text": "Full source transcript", "label": "text", "score": 1.0}]}
            ],
        }
        return httpx.Response(
            200, content=msgpack.packb(body, use_bin_type=True), headers={"content-type": "application/msgpack"}
        )

    mock_transport(monkeypatch, handle)
    monkeypatch.setenv("SIE_BASE_URL", base_url)
    monkeypatch.setenv("SIE_API_KEY", "synthetic-fixture-credential")
    output = tmp_path / "run"
    report = collector.collect(packet, output, model, base_url, instruction, max_new_tokens)
    assert report["physical_inference_posts"] == 24
    assert len(seen) == 24
    manifest, _ = load_packet(packet)
    for request, source in zip(seen, manifest["records"], strict=True):
        body = msgpack.unpackb(request.content, raw=False)
        assert body["items"][0]["id"] == source["id"]
        assert body["items"][0]["images"][0]["data"] == (packet / source["input_image"]["path"]).read_bytes()
        assert body["items"][0]["images"][0]["format"] == "jpeg"
        expected_params = {"options": {"profile": "default", "max_new_tokens": max_new_tokens, "num_beams": 1}}
        if instruction is not None:
            expected_params["instruction"] = instruction
        assert body["params"] == expected_params
    assert (output / "wire/00-request.msgpack").read_bytes() == seen[0].content
    assert seen[0].headers["Authorization"] == "Bearer synthetic-fixture-credential"
    assert all(b"synthetic-fixture-credential" not in path.read_bytes() for path in output.rglob("*") if path.is_file())
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    assert all(row["returned_model"] == model for row in calls)
    assert len(json.loads((output / "review-template.json").read_text())) == 24
    with pytest.raises(FileExistsError):
        collector.collect(packet, output, "test/native", "https://ocr.example.test")


@pytest.mark.parametrize(
    ("model", "max_new_tokens", "limit"),
    [
        ("lightonai/LightOnOCR-3-4B", 12289, 12288),
        ("lightonai/LightOnOCR-3-4B", 0, 12288),
        ("zai-org/GLM-OCR", 8193, 8192),
        ("lightonai/LightOnOCR-3-4B-other", 12288, 8192),
        ("test/native", 0, 8192),
        ("", 8192, 8192),
    ],
)
def test_model_output_bounds_fail_before_transport_or_output(
    packet, tmp_path, monkeypatch, model, max_new_tokens, limit
):
    seen = []

    def handle(request):
        seen.append(request)
        raise AssertionError("Invalid controls must not reach the endpoint")

    mock_transport(monkeypatch, handle)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=f"between 1 and {limit}"):
        collector.collect(packet, output, model, "https://ocr.example.test", max_new_tokens=max_new_tokens)
    assert seen == []
    assert not output.exists()


def test_loading_error_is_one_post_and_preserves_all_remaining_cases(packet, tmp_path, monkeypatch):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(503, json={"error": {"code": "MODEL_LOADING", "message": "Still loading"}})

    mock_transport(monkeypatch, handle)
    output = tmp_path / "failed"
    report = collector.collect(packet, output, "test/native", "https://ocr.example.test")
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    assert len(seen) == report["physical_inference_posts"] == 1
    assert calls[0]["status"] == "error"
    assert calls[0]["error_type"] == "PhysicalAttemptError"
    assert len(calls) == 24
    assert all(row["status"] == "unattempted" and row["attempts"] == 0 for row in calls[1:])
    assert json.loads((output / "review-template.json").read_text()) == []


def test_in_flight_result_get_is_recorded_without_replaying_inference(packet, tmp_path, monkeypatch):
    seen = []
    item_id = ""

    def handle(request):
        nonlocal item_id
        seen.append(request.method)
        if request.method == "POST":
            item_id = msgpack.unpackb(request.content, raw=False)["items"][0]["id"]
            return httpx.Response(303, headers={"location": "/result?__modal_attempt_token=synthetic-token"})
        body = {
            "items": [{"id": item_id, "entities": [{"text": "Completed source text", "label": "text", "score": 1.0}]}]
        }
        return httpx.Response(
            200, content=msgpack.packb(body, use_bin_type=True), headers={"content-type": "application/msgpack"}
        )

    mock_transport(monkeypatch, handle)
    output = tmp_path / "continuations"
    report = collector.collect(packet, output, "test/native", "https://ocr.example.test")
    assert seen == ["POST", "GET"] * 24
    assert report["physical_inference_posts"] == 24
    assert report["completed"] == 24
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    for row in calls:
        assert [hop["method"] for hop in row["response_hops"]] == ["POST", "GET"]
        assert row["response_body_sha256"] == row["response_hops"][-1]["body_sha256"]
    assert (output / "wire/00-response-01.bin").exists()
