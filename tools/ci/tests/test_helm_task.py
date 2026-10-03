from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

from tools.mise_tasks import helm


def test_dependency_command_uses_checked_in_chart() -> None:
    assert helm.dependency_command() == ["dependency", "build", "deploy/helm/sie-cluster"]


def test_dependency_repositories_are_derived_and_idempotent() -> None:
    commands = helm.dependency_repository_commands()
    assert commands
    assert all(command[:2] == ["repo", "add"] for command in commands)
    assert all(command[-1] == "--force-update" for command in commands)
    urls = [command[-2] for command in commands]
    assert len(urls) == len(set(urls))
    assert "https://kedacore.github.io/charts" in urls
    assert all(not url.startswith("oci://") for url in urls)


def test_validation_defaults_disable_payload_store(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(helm, "run_helm", lambda args: calls.append(args) or 0)
    assert helm.cmd_lint([]) == 0
    assert calls == [["lint", "deploy/helm/sie-cluster", "--set", "payloadStore.enabled=false"]]


def test_config_staging_is_removed_after_render(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    bundles = root / "packages/sie_server/bundles"
    models = root / "packages/sie_server/models"
    bundles.mkdir(parents=True)
    models.mkdir(parents=True)
    (bundles / "default.yaml").write_text("name: default\n", encoding="utf-8")
    (models / "model.yaml").write_text("id: model\n", encoding="utf-8")
    monkeypatch.setattr(helm, "resolve_project_root", lambda: root)

    helm._sync_configs_to_helm()
    staged = root / "deploy/helm/sie-cluster/files"
    assert (staged / "bundles/default.yaml").is_file()
    assert (staged / "models/model.yaml").is_file()

    helm._cleanup_helm_configs()
    assert not (staged / "bundles").exists()
    assert not (staged / "models").exists()


@pytest.fixture
def package_source(tmp_path: Path, monkeypatch) -> Path:
    chart = tmp_path / "deploy/helm/sie-cluster"
    (chart / "templates").mkdir(parents=True)
    (chart / "Chart.yaml").write_text("apiVersion: v2\nname: sie-cluster\nversion: 0.8.1\n")
    (chart / "values.yaml").write_text("gateway:\n  embeddedConfigs:\n    enabled: false\n")
    for kind in ("bundles", "models"):
        source = tmp_path / "packages/sie_server" / kind
        source.mkdir(parents=True)
        (source / "example.yaml").write_text(f"name: {kind}\n")
        (chart / "templates" / f"{kind}.yaml").write_text(
            "{{- if .Values.gateway.embeddedConfigs.enabled }}\n"
            "apiVersion: v1\nkind: ConfigMap\nmetadata:\n"
            f"  name: {{{{ .Release.Name }}}}-{kind}\ndata:\n"
            f'{{{{- range $path, $_ := .Files.Glob "files/{kind}/*.yaml" }}}}\n'
            "  {{ base $path }}: |\n    {{- $.Files.Get $path | nindent 4 }}\n"
            "{{- end }}\n{{- end }}\n"
        )
    monkeypatch.setattr(helm, "resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr(helm, "CHART_DIR", chart)
    monkeypatch.setattr(helm, "get_usage_flag", lambda _: "package")
    monkeypatch.setattr(helm.sys, "argv", ["helm.py", "package", "--destination", str(tmp_path / "retained")])
    return tmp_path


def test_package_retains_catalogs_and_renders_archive_after_cleanup(package_source: Path, monkeypatch) -> None:
    execute = helm.subprocess.run
    rendered = []
    embedded_configs = []

    def check_archive(command, **kwargs):
        if "template" in command:
            archive = Path(command[command.index("template") + 2])
            assert archive.suffix == ".tgz"
            assert archive.is_file()
            assert not (package_source / "deploy/helm/sie-cluster/files/bundles").exists()
            assert not (package_source / "deploy/helm/sie-cluster/files/models").exists()
            rendered.append(command)
        result = execute(command, **kwargs)
        if "gateway.embeddedConfigs.enabled=true" in command:
            embedded_configs.extend(doc for doc in yaml.safe_load_all(result.stdout) if doc)
        return result

    monkeypatch.setattr(helm.subprocess, "run", check_archive)
    assert helm.main() == 0
    archive = package_source / "retained/sie-cluster-0.8.1.tgz"
    helm.validate_packaged_configs(archive)
    assert len(rendered) == 2
    assert "gateway.embeddedConfigs.enabled=true" in rendered[1]
    assert "config.enabled=false" in rendered[1]
    assert {doc["metadata"]["name"]: doc["data"] for doc in embedded_configs} == {
        "sie-bundles": {"example.yaml": "name: bundles\n"},
        "sie-models": {"example.yaml": "name: models\n"},
    }


@pytest.mark.parametrize("damage", ["missing", "changed", "extra", "duplicate", "symlink"])
def test_packaged_catalog_must_match_exact_source(package_source: Path, damage: str) -> None:
    archive = package_source / "bad.tgz"
    contents = {
        "sie-cluster/files/bundles/example.yaml": b"name: bundles\n",
        "sie-cluster/files/models/example.yaml": b"name: models\n",
    }
    target = "sie-cluster/files/models/example.yaml"
    if damage == "missing":
        contents.pop(target)
    elif damage == "changed":
        contents[target] = b"name: different\n"
    elif damage == "extra":
        contents["sie-cluster/files/models/unreviewed.yaml"] = b"name: extra\n"
    with tarfile.open(archive, "w:gz") as chart:
        for name, data in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            if damage == "symlink" and name == target:
                member.type = tarfile.SYMTYPE
                member.linkname = "other.yaml"
            chart.addfile(member, io.BytesIO(data))
            if damage == "duplicate" and name == target:
                chart.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="packaged chart"):
        helm.validate_packaged_configs(archive)


@pytest.mark.parametrize("failure", ["package", "archive-check", "lint", "embedded-render"])
def test_failed_package_checks_clean_staging_and_never_retain_archive(
    package_source: Path, monkeypatch, failure
) -> None:
    execute = helm.subprocess.run

    def fail(command, **kwargs):
        if (failure in {"package", "lint"} and failure in command) or (
            failure == "embedded-render" and "gateway.embeddedConfigs.enabled=true" in command
        ):
            return subprocess.CompletedProcess(command, 1, "", "deliberate chart validation failure")
        return execute(command, **kwargs)

    monkeypatch.setattr(helm.subprocess, "run", fail)
    if failure == "archive-check":

        def invalid(_archive):
            raise ValueError("invalid retained chart")

        monkeypatch.setattr(helm, "validate_packaged_configs", invalid)
        with pytest.raises(ValueError, match="invalid retained chart"):
            helm.main()
    else:
        assert helm.main() == 1
    assert not (package_source / "retained").exists()
    assert not (package_source / "deploy/helm/sie-cluster/files/bundles").exists()
    assert not (package_source / "deploy/helm/sie-cluster/files/models").exists()
