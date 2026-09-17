from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pytest

from tools.ci import release_openapi

DOCUMENTS = sorted(release_openapi.OPENAPI_VERSION_SOURCES)


@pytest.mark.parametrize("document", DOCUMENTS)
def test_stamp_rewrites_only_the_info_version_line(document) -> None:
    committed = (release_openapi.ROOT / document).read_text(encoding="utf-8")
    version = release_openapi.manifest_version(document)
    assert json.loads(committed)["info"]["version"] == version
    assert release_openapi.stamp(committed, version) == committed

    stamped = release_openapi.stamp(committed, "99.100.101")
    changed = [
        (before, after)
        for before, after in zip(committed.split("\n"), stamped.split("\n"), strict=True)
        if before != after
    ]
    assert len(changed) == 1
    before, after = changed[0]
    assert before.startswith(f'    "version": "{version}"')
    assert after == before.replace(version, "99.100.101")


@pytest.mark.parametrize(
    "document",
    [
        '{"info": {"version": "0.7.3"}}\n',
        '{\n  "info": {\n    "title": "SIE"\n  },\n  "x-other": {\n    "version": "0.7.3"\n  }\n}\n',
    ],
)
def test_stamp_rejects_documents_without_one_pretty_printed_info_version(document) -> None:
    with pytest.raises(ValueError, match=r"info\.version"):
        release_openapi.stamp(document, "0.8.0")


def test_main_stamps_each_document_from_its_own_manifest(monkeypatch, tmp_path: Path) -> None:
    for document, (manifest, _) in release_openapi.OPENAPI_VERSION_SOURCES.items():
        for relative in (document, manifest):
            (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(release_openapi.ROOT / relative, tmp_path / relative)
    monkeypatch.setattr(release_openapi, "ROOT", tmp_path)
    releases = dict(zip(DOCUMENTS, ("98.0.1", "98.0.2"), strict=True))
    for document, release in releases.items():
        manifest, table = release_openapi.OPENAPI_VERSION_SOURCES[document]
        path = tmp_path / manifest
        text = path.read_text(encoding="utf-8")
        current = tomllib.loads(text)[table]["version"]
        path.write_text(text.replace(f'version = "{current}"', f'version = "{release}"', 1), encoding="utf-8")
        assert release_openapi.manifest_version(document) == release

    release_openapi.main()

    for document, release in releases.items():
        assert json.loads((tmp_path / document).read_text(encoding="utf-8"))["info"]["version"] == release
