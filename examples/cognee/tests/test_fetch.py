"""Offline download boundaries using tiny stand-ins, without a checked-in dataset."""

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

import fetch
import make_document


@pytest.fixture
def stand_ins(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    files = {name: f"unit-test:{name}".encode() for name in fetch.REQUIRED}
    manifest = {
        "schema_version": 1,
        "license": "CC0-1.0",
        "files": {
            name: {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()} for name, body in files.items()
        },
    }
    files["manifest.json"] = json.dumps(manifest).encode()
    monkeypatch.setattr(fetch, "MANIFEST_SHA256", hashlib.sha256(files["manifest.json"]).hexdigest())
    return files


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args, **kwargs):
        raise AssertionError("Offline checks must not access the network")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", blocked)


def transport(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> list[str]:
    visited = []

    def open_response(request, timeout):
        assert timeout == 60
        assert not request.has_header("Authorization")
        assert request.full_url.startswith(fetch.BASE_URL + "/")
        name = request.full_url.removeprefix(fetch.BASE_URL + "/")
        visited.append(name)
        return io.BytesIO(files[name])

    monkeypatch.setattr(fetch.urllib.request, "urlopen", open_response)
    return visited


def test_anonymous_pinned_fetch_and_offline_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stand_ins: dict[str, bytes]
) -> None:
    visited = transport(monkeypatch, stand_ins)
    fetch.fetch(tmp_path)
    assert set(visited) == set(stand_ins)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()}
    fetch.fetch(tmp_path)
    assert len(visited) == len(stand_ins)
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("problem", ["manifest", "hash", "short", "oversized"])
def test_corrupt_download_is_never_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stand_ins: dict[str, bytes], problem: str
) -> None:
    name = "manifest.json" if problem == "manifest" else "projector-check.txt"
    original = stand_ins[name]
    stand_ins[name] = {
        "manifest": original + b" ",
        "hash": b"x" * len(original),
        "short": original[:-1],
        "oversized": original + b"x",
    }[problem]
    transport(monkeypatch, stand_ins)
    with pytest.raises(ValueError):
        fetch.fetch(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["manifest.json", "access-note.txt"])
def test_cached_tampering_is_rejected_without_network_or_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stand_ins: dict[str, bytes], name: str
) -> None:
    for filename, body in stand_ins.items():
        (tmp_path / filename).write_bytes(body)
    (tmp_path / name).write_bytes(b"edited")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        fetch.fetch(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("problem", ["traversal", "missing", "boolean-size", "large-file", "total", "digest", "schema"])
def test_manifest_shape_paths_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stand_ins: dict[str, bytes], problem: str
) -> None:
    manifest = json.loads(stand_ins["manifest.json"])
    files = manifest["files"]
    if problem == "traversal":
        files["../outside.txt"] = files.pop("access-note.txt")
    elif problem == "missing":
        del files["access-note.txt"]
    elif problem == "boolean-size":
        files["access-note.txt"]["bytes"] = True
    elif problem == "large-file":
        files["access-note.txt"]["bytes"] = fetch.FILE_LIMIT + 1
    elif problem == "total":
        monkeypatch.setattr(fetch, "TOTAL_LIMIT", 1)
    elif problem == "digest":
        files["access-note.txt"]["sha256"] = "not-a-digest"
    else:
        manifest["schema_version"] = 2
    stand_ins["manifest.json"] = json.dumps(manifest).encode()
    monkeypatch.setattr(fetch, "MANIFEST_SHA256", hashlib.sha256(stand_ins["manifest.json"]).hexdigest())
    visited = transport(monkeypatch, stand_ins)
    with pytest.raises(ValueError):
        fetch.fetch(tmp_path)
    assert visited == ["manifest.json"]
    assert not list(tmp_path.iterdir())
    assert not (tmp_path.parent / "outside.txt").exists()


def test_manifest_response_read_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Oversized(io.BytesIO):
        def read(self, size=-1):
            assert size == fetch.MANIFEST_LIMIT + 1
            return b"x" * size

    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *args, **kwargs: Oversized())
    with pytest.raises(ValueError, match="byte limit"):
        fetch.fetch(tmp_path)
    assert not list(tmp_path.iterdir())


def test_interrupted_fetch_preserves_cache_without_partial_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stand_ins: dict[str, bytes]
) -> None:
    (tmp_path / "access-note.txt").write_bytes(stand_ins["access-note.txt"])
    calls = []

    def interrupted(request, timeout):
        name = request.full_url.removeprefix(fetch.BASE_URL + "/")
        calls.append(name)
        if len(calls) == 4:
            raise OSError("interrupted stand-in download")
        return io.BytesIO(stand_ins[name])

    monkeypatch.setattr(fetch.urllib.request, "urlopen", interrupted)
    with pytest.raises(OSError):
        fetch.fetch(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == {
        "access-note.txt": stand_ins["access-note.txt"]
    }


def test_renderer_preserves_pinned_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data/access-note.txt").write_text("Small authored renderer test.", encoding="utf-8")
    pinned = tmp_path / "data/access-note.png"
    pinned.write_bytes(b"pinned-image-stand-in")
    monkeypatch.setattr(make_document, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_document.py"])
    make_document.main()
    assert pinned.read_bytes() == b"pinned-image-stand-in"
    with Image.open(tmp_path / "runs/access-note-rendered.png") as rendered:
        assert rendered.format == "PNG"
        assert rendered.size == (1200, 480)
    monkeypatch.setattr(sys, "argv", ["make_document.py", "--output", str(pinned)])
    with pytest.raises(SystemExit) as error:
        make_document.main()
    assert error.value.code == 2
    assert pinned.read_bytes() == b"pinned-image-stand-in"


def test_renderer_missing_download_has_fetch_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(make_document, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_document.py"])
    with pytest.raises(SystemExit) as error:
        make_document.main()
    assert error.value.code == 2
    assert "uv run fetch.py" in capsys.readouterr().err
