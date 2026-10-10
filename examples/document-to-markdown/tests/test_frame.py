from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import msgpack
import pytest
from sie_sdk import SIEClient

from document_to_markdown import frame

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aN3cAAAAASUVORK5CYII=")


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "inputs"
    (root / "pngs").mkdir(parents=True)
    pages = []
    for ordinal in range(3):
        path = f"pngs/page-{ordinal}.png"
        (root / path).write_bytes(PNG)
        pages.append(
            {
                "pdf": f"tables/page-{ordinal}.pdf",
                "png": path,
                "png_sha256": hashlib.sha256(PNG).hexdigest(),
                "png_bytes": len(PNG),
                "category": "tables",
                "family": f"family-{ordinal}",
                "role": "frame",
            }
        )
    pages.append({"pdf": "excluded.pdf", "png": "not-present.png", "role": "quote"})
    manifest = root / "frame-manifest.json"
    manifest.write_text(json.dumps({"pages": pages}), encoding="utf-8")
    return manifest, root


def response(text: str = "# Résumé\n\n| A | B |\n|---|---|\n") -> httpx.Response:
    return httpx.Response(
        200,
        content=msgpack.packb({"items": [{"entities": [], "data": {"markdown": text}}]}, use_bin_type=True),
        headers={"Content-Type": "application/msgpack"},
    )


def rows(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("url", ["http://localhost:8080", "https://chosen.example/gateway"])
def test_actual_sdk_preserves_base_path_png_options_intent_and_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    manifest, root = inputs(tmp_path)
    out = tmp_path / "run"
    calls = []

    def reply(request: httpx.Request) -> httpx.Response:
        ordinal = len(calls)
        assert json.loads((out / "intents" / f"{ordinal:06d}.json").read_text())["status"] == "attempt_status_unknown"
        assert rows(out)[ordinal]["status"] == "attempt_status_unknown"
        body = msgpack.unpackb(request.content, raw=False)
        assert body["items"][0]["id"] == f"tables/page-{ordinal}.pdf"
        assert body["items"][0]["images"][0]["data"] == PNG
        assert body["params"] == {"options": frame.OPTIONS}
        assert str(request.url) == f"{url}/v1/extract/example/model"
        assert request.headers["Authorization"] == "Bearer test-key"
        calls.append(request)
        return response()

    monkeypatch.setenv("SIE_API_KEY", "test-key")
    actual = frame.run_frame(manifest, root, out, sie_url=url, model="example/model", inner=httpx.MockTransport(reply))
    assert len(calls) == 3
    assert all(row["status"] == "received" and row["physical_sends"] == 1 for row in actual)
    assert rows(out) == actual
    assert (out / "markdown" / "tables" / "page-0.md").read_bytes() == "# Résumé\n\n| A | B |\n|---|---|\n".encode()
    assert len(list((out / "sdk").glob("*.json"))) == 3
    assert len(list((out / "responses").glob("*.bin"))) == 3
    assert "test-key" not in (out / "plan.json").read_text()


def test_fixed_selection_keeps_manifest_order(tmp_path: Path) -> None:
    manifest, root = inputs(tmp_path)
    selection = tmp_path / "selected.json"
    selection.write_text(json.dumps(["tables/page-2.pdf", "tables/page-0.pdf"]))
    selected, plan = frame.load_pages(manifest, root, selection)
    assert [page["pdf"] for page in selected] == ["tables/page-0.pdf", "tables/page-2.pdf"]
    assert plan["selection_sha256"] == frame.digest(selection.read_bytes())


@pytest.mark.parametrize("selection", [{"pdfs": []}, ["unknown.pdf"], ["tables/page-0.pdf"] * 2, [], [3]])
def test_malformed_selection_is_rejected_before_dispatch(tmp_path: Path, selection: Any) -> None:
    manifest, root = inputs(tmp_path)
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(selection))
    calls = []
    with pytest.raises(ValueError):
        frame.run_frame(
            manifest,
            root,
            tmp_path / "run",
            selection=selected,
            sie_url="https://chosen.example",
            inner=httpx.MockTransport(lambda request: calls.append(request) or response()),
        )
    assert calls == []
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("png", "../outside.png"),
        ("png", "/outside.png"),
        ("pdf", "../outside.pdf"),
        ("pdf", "/outside.pdf"),
        ("pdf", "C:/outside.pdf"),
        ("pdf", "tables/./page.pdf"),
        ("png_sha256", "0" * 64),
        ("png_bytes", 0),
    ],
)
def test_bad_later_input_is_rejected_before_any_dispatch(tmp_path: Path, field: str, value: Any) -> None:
    manifest, root = inputs(tmp_path)
    data = json.loads(manifest.read_text())
    data["pages"][2][field] = value
    manifest.write_text(json.dumps(data))
    calls = []
    with pytest.raises(ValueError):
        frame.run_frame(
            manifest,
            root,
            tmp_path / "run",
            sie_url="https://chosen.example",
            inner=httpx.MockTransport(lambda request: calls.append(request) or response()),
        )
    assert calls == []


def test_symlink_outside_input_root_is_rejected(tmp_path: Path) -> None:
    manifest, root = inputs(tmp_path)
    image = root / "pngs" / "page-1.png"
    image.unlink()
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    image.symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        frame.load_pages(manifest, root, None)


@pytest.mark.parametrize("failure", ["transport", "http", "loading", "redirect"])
def test_nonreceived_page_never_resends_and_keeps_full_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    manifest, root = inputs(tmp_path)
    calls = []

    def reply(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if failure == "transport":
            raise httpx.ReadError("interrupted", request=request)
        if failure == "http":
            return httpx.Response(400, json={"error": "invalid request"})
        if failure == "loading":
            return httpx.Response(503, json={"error": {"code": "MODEL_LOADING", "message": "loading"}})
        if failure == "redirect":
            return httpx.Response(307, headers={"Location": "https://elsewhere.example"})
        raise AssertionError("Unexpected fixture")

    monkeypatch.setattr("sie_sdk.client.sync.time.sleep", lambda delay: None)
    out = tmp_path / "run"
    actual = frame.run_frame(manifest, root, out, sie_url="https://chosen.example", inner=httpx.MockTransport(reply))
    assert len(calls) == 1
    assert actual[0]["status"] == ("attempt_status_unknown" if failure == "transport" else "failed")
    assert actual[0]["physical_sends"] == 1
    assert [row["status"] for row in actual[1:]] == ["known_unattempted", "known_unattempted"]
    assert all(row["reason"] == "stopped_after_nonreceived_page" for row in actual[1:])
    assert len(rows(out)) == 3
    assert not list((out / "markdown").rglob("*.md"))


def test_complete_empty_reply_is_one_failure_and_later_pages_continue(tmp_path: Path) -> None:
    manifest, root = inputs(tmp_path)
    calls = []

    def reply(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return response("" if len(calls) == 1 else "# Received\n")

    out = tmp_path / "run"
    actual = frame.run_frame(manifest, root, out, sie_url="https://chosen.example", inner=httpx.MockTransport(reply))
    assert len(calls) == 3
    assert [row["status"] for row in actual] == ["failed", "received", "received"]
    assert all(row["physical_sends"] == 1 for row in actual)
    assert actual[0]["reason"] == "MarkdownOutputError"
    assert len(rows(out)) == 3
    assert (out / "responses" / "000000.bin").is_file()
    assert len(list((out / "markdown").rglob("*.md"))) == 2


def test_metadata_get_cannot_qualify_a_later_unknown_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, root = inputs(tmp_path)
    calls = []

    def reply(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"metadata": "complete"})
        raise httpx.ReadError("interrupted extraction", request=request)

    def client(*args: Any, **kwargs: Any) -> SIEClient:
        actual = SIEClient(*args, **kwargs)
        extract = actual.extract

        def after_metadata(*extract_args: Any, **extract_kwargs: Any) -> Any:
            kwargs["http_client"].get("/metadata")
            return extract(*extract_args, **extract_kwargs)

        monkeypatch.setattr(actual, "extract", after_metadata)
        return actual

    monkeypatch.setattr(frame, "SIEClient", client)
    out = tmp_path / "run"
    actual = frame.run_frame(manifest, root, out, sie_url="https://chosen.example", inner=httpx.MockTransport(reply))
    assert calls == ["GET", "POST"]
    assert actual[0]["status"] == "attempt_status_unknown"
    assert actual[0]["physical_sends"] == 1
    assert "http_status" not in actual[0]
    assert not list((out / "responses").iterdir())
    assert all(row["status"] == "known_unattempted" for row in actual[1:])


def test_existing_directory_never_dispatches_or_replaces_files(tmp_path: Path) -> None:
    manifest, root = inputs(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    (out / "saved.md").write_text("retain")
    calls = []
    with pytest.raises(FileExistsError):
        frame.run_frame(
            manifest,
            root,
            out,
            sie_url="https://chosen.example",
            inner=httpx.MockTransport(lambda request: calls.append(request) or response()),
        )
    assert calls == []
    assert (out / "saved.md").read_text() == "retain"


def test_failed_intent_persistence_never_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, root = inputs(tmp_path)
    original = frame.save_new
    calls = []

    def save(path: Path, raw: bytes) -> None:
        if path.parent.name == "intents":
            raise OSError("cannot persist")
        original(path, raw)

    monkeypatch.setattr(frame, "save_new", save)
    with pytest.raises(OSError):
        frame.run_frame(
            manifest,
            root,
            tmp_path / "run",
            sie_url="https://chosen.example",
            inner=httpx.MockTransport(lambda request: calls.append(request) or response()),
        )
    assert calls == []


def test_markdown_entity_and_error_contract() -> None:
    assert frame.markdown_text({"entities": [{"label": "markdown", "text": "# Heading\n"}]}) == "# Heading\n"
    with pytest.raises(ValueError):
        frame.markdown_text({"error": {"message": "failed"}, "data": {"markdown": "ignored"}})
