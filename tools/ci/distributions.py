#!/usr/bin/env python3
"""Build, inspect and consume the public distributions without editable installs."""

from __future__ import annotations

import argparse
import base64
import email
import hashlib
import http
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_PATHS = (
    "packages/sie_sdk",
    "packages/sie_server",
    "packages/sie_config",
    "packages/sie_mcp",
    "integrations/sie_langchain",
    "integrations/sie_llamaindex",
    "integrations/sie_haystack",
    "integrations/sie_dspy",
    "integrations/sie_crewai",
    "integrations/sie_chroma",
    "integrations/sie_lancedb",
    "integrations/sie_qdrant",
    "integrations/sie_weaviate",
)
NPM_PATHS = (
    "packages/sie_ts_sdk",
    "integrations/sie_ts_chroma",
    "integrations/sie_ts_langchain",
    "integrations/sie_ts_llamaindex",
    "integrations/sie_ts_lancedb",
)


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)  # noqa: S603


def manifests(family: str, version: str = "") -> dict[str, tuple[str, Path]]:
    result = {}
    for relative in PYTHON_PATHS if family == "python" else NPM_PATHS:
        path = ROOT / relative
        data = (
            tomllib.loads((path / "pyproject.toml").read_text())["project"]
            if family == "python"
            else json.loads((path / "package.json").read_text())
        )
        if version and data["version"] != version:
            raise ValueError(f"{relative}: actual {data['version']} != release {version}")
        result[data["name"]] = (data["version"], path)
    return result


def archive_metadata(path: Path) -> tuple[str, str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            entries = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(entries) != 1:
                raise ValueError(f"{path.name}: expected one wheel metadata record")
            data = email.message_from_bytes(archive.read(entries[0]))
            return data["Name"], data["Version"]
    with tarfile.open(path, "r:gz") as archive:
        if path.suffix == ".tgz":
            member = archive.extractfile("package/package.json")
            if member is None:
                raise ValueError("npm archive has no package manifest")
            data = json.load(member)
            names = set(archive.getnames())
            for entry in (data.get("main"), data.get("module"), data.get("types")):
                if entry and "package/" + entry.removeprefix("./") not in names:
                    raise ValueError(f"{path.name}: missing packed entrypoint {entry}")
            if "workspace:" in json.dumps(data.get("dependencies", {})):
                raise ValueError("packed dependencies still reference the workspace")
            return data["name"], data["version"]
        entries = [
            entry for entry in archive.getmembers() if entry.name.count("/") == 1 and entry.name.endswith("/PKG-INFO")
        ]
        if len(entries) != 1:
            raise ValueError(f"{path.name}: expected one sdist metadata record")
        member = archive.extractfile(entries[0])
        if member is None:
            raise ValueError("sdist has no package metadata")
        data = email.message_from_bytes(member.read())
        return data["Name"], data["Version"]


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def verify(family: str, directory: Path, version: str = "") -> list[Path]:
    expected = {normalize(name): value[0] for name, value in manifests(family, version).items()}
    found: dict[tuple[str, str], Path] = {}
    for path in sorted(directory.iterdir()):
        if path.name == "provenance.json":
            continue
        suffix = ".whl" if path.suffix == ".whl" else ".tar.gz" if path.name.endswith(".tar.gz") else path.suffix
        if suffix not in ({".whl", ".tar.gz"} if family == "python" else {".tgz"}) or path.is_symlink():
            raise ValueError(f"unexpected package output: {path.name}")
        name, actual = archive_metadata(path)
        name = normalize(name)
        if expected.get(name) != actual or (name, suffix) in found:
            raise ValueError(f"{path.name}: wrong/duplicate distribution name or version")
        found[name, suffix] = path
    wanted = {
        (name, suffix) for name in expected for suffix in ({".whl", ".tar.gz"} if family == "python" else {".tgz"})
    }
    if set(found) != wanted:
        raise ValueError(f"incomplete {family} archives: missing {sorted(wanted - set(found))}")
    return list(found.values())


def clean_python(archives: list[Path]) -> None:
    sdk = next(path for path in archives if path.name.startswith("sie_sdk-") and path.suffix == ".whl")
    for archive in archives:
        name, version = archive_metadata(archive)
        module = name.replace("-", "_")
        module = {
            "sie_config": "sie_config.cli",
            "sie_mcp": "sie_mcp.cli",
            "sie_server": "sie_server.bundle_requirements",
        }.get(module, module)
        script = (
            "import importlib,importlib.metadata,pathlib,sys; "
            f"m=importlib.import_module({module!r}); "
            f"assert importlib.metadata.version({name!r})=={version!r}; "
            "assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())"
        )
        requirements = ["--with", str(archive)]
        if normalize(name) != "sie-sdk":
            requirements += ["--with", str(sdk)]
        with tempfile.TemporaryDirectory(prefix="sie-packed-python-") as temporary:
            run(
                "uv",
                "--no-config",
                "run",
                "--no-project",
                "--isolated",
                "--python",
                "3.12",
                *requirements,
                "python",
                "-I",
                "-c",
                script,
                cwd=Path(temporary),
            )


def clean_npm(archives: list[Path]) -> None:
    with tempfile.TemporaryDirectory(prefix="sie-packed-npm-") as temporary:
        path = Path(temporary)
        (path / "package.json").write_text(json.dumps({"name": "sie-packed-consumer", "private": True}))
        run("npm", "install", "--ignore-scripts", "--no-audit", "--no-fund", *map(str, archives), cwd=path)
        for archive in archives:
            name, version = archive_metadata(archive)
            run("node", "--input-type=module", "-e", f"await import({json.dumps(name)})", cwd=path)
            run("node", "-e", f"require({json.dumps(name)})", cwd=path)
            installed = json.loads((path / "node_modules" / name / "package.json").read_text())
            if installed["version"] != version:
                raise ValueError(f"{name}: consumer loaded a different version")


def build(family: str, directory: Path, version: str) -> None:
    packages = manifests(family, version)
    directory.mkdir(parents=True, exist_ok=False)
    if family == "python":
        run("uv", "lock", "--check", "--project", str(ROOT))
        for name in packages:
            with tempfile.TemporaryDirectory(prefix="sie-package-build-") as temporary:
                run("uv", "build", "--package", name, "--out-dir", temporary)
                for archive in Path(temporary).iterdir():
                    if archive.name.endswith((".whl", ".tar.gz")):
                        destination = directory / archive.name
                        if destination.exists():
                            raise ValueError(f"duplicate built archive: {archive.name}")
                        shutil.copyfile(archive, destination)
    else:
        run("pnpm", "install", "--frozen-lockfile")
        run("pnpm", "-r", "build")
        for _, path in packages.values():
            run("pnpm", "--dir", str(path), "pack", "--pack-destination", str(directory))
    archives = verify(family, directory, version)
    (clean_python if family == "python" else clean_npm)(archives)


def prepare_pypi(directory: Path, destination: Path, version: str) -> None:
    archives = verify("python", directory, version)
    destination.mkdir(parents=True, exist_ok=False)
    releases = {}
    for archive in archives:
        name, _ = archive_metadata(archive)
        if name not in releases:
            try:
                with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
                    releases[name] = json.load(response)["urls"]
            except urllib.error.HTTPError as error:
                if error.code != http.HTTPStatus.NOT_FOUND:
                    raise
                releases[name] = []
        existing = next((item for item in releases[name] if item["filename"] == archive.name), None)
        if existing:
            if existing["digests"]["sha256"] != hashlib.sha256(archive.read_bytes()).hexdigest():
                raise ValueError(f"PyPI already has different bytes for {archive.name}")
        else:
            shutil.copyfile(archive, destination / archive.name)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"pending={'true' if any(destination.iterdir()) else 'false'}\n")


def publish_npm(directory: Path, version: str) -> None:
    archives = verify("npm", directory, version)
    pending = []
    for archive in archives:
        name, _ = archive_metadata(archive)
        result = subprocess.run(  # noqa: S603
            ["npm", "view", f"{name}@{version}", "dist.integrity", "--json", "--registry=https://registry.npmjs.org"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            if "E404" not in result.stderr:
                raise ValueError(f"cannot check existing npm version: {result.stderr}")
            pending.append(archive)
        else:
            integrity = "sha512-" + base64.b64encode(hashlib.sha512(archive.read_bytes()).digest()).decode()
            if json.loads(result.stdout) != integrity:
                raise ValueError(f"npm already has different bytes for {name}@{version}")
    for archive in pending:
        run(
            "npm",
            "publish",
            str(archive),
            "--access",
            "public",
            "--provenance",
            "--ignore-scripts",
            "--registry=https://registry.npmjs.org",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "verify", "prepare-pypi", "publish-npm"])
    parser.add_argument("family", choices=["python", "npm"])
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--version", default="")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.mode == "build":
        build(args.family, directory, args.version)
    elif args.mode == "verify":
        verify(args.family, directory, args.version)
    elif args.mode == "prepare-pypi":
        prepare_pypi(directory, args.destination, args.version)
    else:
        publish_npm(directory, args.version)


if __name__ == "__main__":
    main()
