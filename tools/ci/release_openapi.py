#!/usr/bin/env python3
"""Stamp each package version into the committed OpenAPI document generated from it."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPENAPI_VERSION_SOURCES = {
    "packages/sie_server/openapi.json": ("packages/sie_server/pyproject.toml", "project"),
    "packages/sie_gateway/openapi.json": ("packages/sie_gateway/Cargo.toml", "package"),
}
INFO_VERSION = re.compile(r'^(    "version": )"[^"\\]*"(,?)$', re.MULTILINE)


def stamp(document: str, version: str) -> str:
    stamped, members = INFO_VERSION.subn(lambda match: f"{match[1]}{json.dumps(version)}{match[2]}", document)
    expected = json.loads(document)
    expected["info"]["version"] = version
    if members != 1 or json.loads(stamped) != expected:
        raise ValueError("OpenAPI document must carry exactly one pretty-printed info.version member")
    return stamped


def manifest_version(document: str) -> str:
    manifest, table = OPENAPI_VERSION_SOURCES[document]
    return tomllib.loads((ROOT / manifest).read_text(encoding="utf-8"))[table]["version"]


def main() -> None:
    for document in OPENAPI_VERSION_SOURCES:
        path = ROOT / document
        path.write_text(stamp(path.read_text(encoding="utf-8"), manifest_version(document)), encoding="utf-8")


if __name__ == "__main__":
    main()
