from __future__ import annotations

import base64
import copy
import io
import json
import shutil
from importlib.metadata import version
from unittest.mock import patch

import httpx
import pytest
from jsonschema import ValidationError
from sie_sdk import SIEClient

import evidence
import run
import score


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must use the fetched evidence or mocked transport")

    monkeypatch.setattr(evidence.urllib.request, "urlopen", blocked)


@pytest.fixture
def recorded():
    evidence.verify(evidence.DEFAULT_EVIDENCE)
    return evidence.DEFAULT_EVIDENCE


def test_offline_recorded_answer(recorded):
    with patch.object(SIEClient, "chat_completions", side_effect=AssertionError("No inference")):
        result = score.score(recorded)
    assert (result["checks"], result["passed"], result["fully_right"], result["schema_valid"]) == (26, 26, True, True)
    inputs = evidence.verify(recorded)
    assert inputs["request"]["messages"][0]["content"].endswith("\n")
    assert len(inputs["screen"]["gold"]["subtasks"]) == 3


def test_anonymous_fetch_uses_fixed_verified_files(recorded, tmp_path, monkeypatch):
    files = {pin["url"]: (recorded / name).read_bytes() for name, pin in evidence.PINS.items()}
    visited = []

    def download(request, timeout):
        assert timeout == 120
        assert not request.has_header("Authorization")
        visited.append(request.full_url)
        return io.BytesIO(files[request.full_url])

    monkeypatch.setattr(evidence.urllib.request, "urlopen", download)
    evidence.fetch(tmp_path)
    assert set(visited) == set(files)
    evidence.fetch(tmp_path)
    assert len(visited) == len(files)
    assert score.score(tmp_path)["passed"] == 26


@pytest.mark.parametrize("corrupt", [b"short", b"x" * 173994])
def test_bad_download_never_promoted(tmp_path, monkeypatch, corrupt):
    monkeypatch.setattr(evidence.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(corrupt))
    with pytest.raises(ValueError, match="digest mismatch"):
        evidence.fetch(tmp_path)
    assert not list(tmp_path.rglob("*"))


@pytest.mark.parametrize("name", list(evidence.PINS))
def test_each_cached_file_is_authenticated(recorded, tmp_path, name):
    shutil.copytree(recorded, tmp_path, dirs_exist_ok=True)
    path = tmp_path / name
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="digest mismatch"):
        evidence.verify(tmp_path)
    with pytest.raises(ValueError, match="digest mismatch"):
        score.score(tmp_path)


def test_missing_file_fails(recorded, tmp_path):
    shutil.copytree(recorded, tmp_path, dirs_exist_ok=True)
    (tmp_path / "source/cases/KB01/image.png").unlink()
    with pytest.raises(FileNotFoundError):
        evidence.verify(tmp_path)


def test_adjacent_manifest_cannot_reauthorize_response(recorded, tmp_path):
    shutil.copytree(recorded, tmp_path, dirs_exist_ok=True)
    response = tmp_path / "native/KB01-response.json"
    response.write_text("{}")
    manifest = tmp_path / "native/manifest.json"
    altered = json.loads(manifest.read_text())
    altered["files"]["KB01-response.json"]["sha256"] = evidence.hashlib.sha256(b"{}").hexdigest()
    manifest.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="digest mismatch"):
        evidence.verify(tmp_path)


def test_released_sdk_full_request_and_response(recorded):
    assert version("sie-sdk") == "0.10.1"
    calls = []
    saved = json.loads((recorded / "native/KB01-response.json").read_text())

    def transport(request):
        calls.append(request)
        assert request.method == "POST"
        assert str(request.url) == "http://example.test/v1/chat/completions"
        body = json.loads(request.content)
        assert "max_tokens" not in body
        body["max_tokens"] = body.pop("max_completion_tokens")
        assert body == json.loads((recorded / "native/KB01-request.json").read_text())
        image_url = body["messages"][1]["content"][0]["image_url"]["url"]
        assert (
            base64.b64decode(image_url.split(",", 1)[1], validate=True)
            == (recorded / "source/cases/KB01/image.png").read_bytes()
        )
        return httpx.Response(200, json=saved)

    with httpx.Client(base_url="http://example.test", transport=httpx.MockTransport(transport)) as http:
        with SIEClient("http://example.test", api_key="", http_client=http) as client:
            result = run.extract(client, recorded)
    assert len(calls) == 1
    assert result == saved


@pytest.mark.parametrize("status", [500, 503, 504])
def test_terminal_http_failure_does_not_resend(recorded, status):
    calls = []

    def transport(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "Recorded mock failure", "type": "server_error"}})

    with httpx.Client(base_url="http://example.test", transport=httpx.MockTransport(transport)) as http:
        with SIEClient("http://example.test", api_key="", http_client=http) as client:
            with pytest.raises(Exception):
                run.extract(client, recorded)
    assert len(calls) == 1


@pytest.mark.parametrize("change,passed", [("value", 25), ("missing_row", 23), ("extra_row", 26)])
def test_original_scoring_penalizes_wrong_values_and_rows(recorded, tmp_path, change, passed):
    response = json.loads((recorded / "native/KB01-response.json").read_text())
    answer = json.loads(response["choices"][0]["message"]["content"])
    if change == "value":
        answer["priority"] = 2
    elif change == "missing_row":
        answer["subtasks"].pop()
    else:
        answer["subtasks"].append(copy.deepcopy(answer["subtasks"][0]))
    response["choices"][0]["message"]["content"] = json.dumps(answer)
    path = tmp_path / "fresh-response.json"
    path.write_text(json.dumps(response))
    result = score.score(recorded, path)
    assert result["passed"] == passed
    assert result["checks"] == (27 if change == "extra_row" else 26)
    assert not result["fully_right"]


@pytest.mark.parametrize("change", ["type", "extra_field", "truncated", "invalid_json", "empty"])
def test_schema_and_incomplete_answers_fail(recorded, tmp_path, change):
    response = json.loads((recorded / "native/KB01-response.json").read_text())
    choice = response["choices"][0]
    answer = json.loads(choice["message"]["content"])
    if change == "type":
        answer["priority"] = "1"
    elif change == "extra_field":
        answer["invented"] = "hidden"
    elif change == "truncated":
        choice["finish_reason"] = "length"
    choice["message"]["content"] = {"invalid_json": "{", "empty": ""}.get(change, json.dumps(answer))
    path = tmp_path / "fresh-response.json"
    path.write_text(json.dumps(response))
    with pytest.raises((ValueError, ValidationError)):
        score.score(recorded, path)


def test_cli_retains_complete_envelope(recorded, tmp_path, monkeypatch):
    saved = json.loads((recorded / "native/KB01-response.json").read_text())
    calls = []

    def client_factory(endpoint, api_key):
        def transport(request):
            calls.append(request)
            return httpx.Response(200, json=saved)

        http = httpx.Client(base_url=endpoint, transport=httpx.MockTransport(transport))
        return SIEClient(endpoint, api_key=api_key, http_client=http)

    output = tmp_path / "response.json"
    monkeypatch.setattr(run, "SIEClient", client_factory)
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--endpoint", "http://example.test", "--evidence", str(recorded), "--output", str(output)],
    )
    run.main()
    assert json.loads(output.read_text()) == saved
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        run.main()
    assert len(calls) == 1
    assert json.loads(output.read_text()) == saved


def test_cli_failed_request_leaves_no_answer(recorded, tmp_path, monkeypatch):
    def client_factory(endpoint, api_key):
        http = httpx.Client(
            base_url=endpoint,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(500, json={"error": {"message": "mock failure"}})
            ),
        )
        return SIEClient(endpoint, api_key=api_key, http_client=http)

    output = tmp_path / "response.json"
    monkeypatch.setattr(run, "SIEClient", client_factory)
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--endpoint", "http://example.test", "--evidence", str(recorded), "--output", str(output)],
    )
    with pytest.raises(Exception):
        run.main()
    assert not output.exists()
