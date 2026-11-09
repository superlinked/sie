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


def test_locked_archives_follow_the_checked_in_lock() -> None:
    """The task's line parser and a YAML parser agree on the pinned archives."""
    lock = yaml.safe_load((helm.resolve_project_root() / helm.CHART_DIR / "Chart.lock").read_text())
    expected = {f"{entry['name']}-{entry['version']}.tgz" for entry in lock["dependencies"]}
    assert expected
    assert helm.locked_dependency_archives() == expected


@pytest.fixture
def locked_chart(tmp_path: Path, monkeypatch) -> Path:
    """A chart whose lock pins two dependencies, both present in charts/."""
    chart = tmp_path / "deploy/helm/sie-cluster"
    (chart / "charts").mkdir(parents=True)
    (chart / "Chart.yaml").write_text(
        "apiVersion: v2\nname: sie-cluster\nversion: 0.8.1\ndependencies:\n"
        "  - name: keda\n    repository: https://example.test/charts\n    version: 2.14.x\n"
        "  - name: cert-manager\n    repository: https://example.test/charts\n    version: 1.16.x\n"
    )
    (chart / "Chart.lock").write_text(
        "dependencies:\n"
        "- name: keda\n  repository: https://example.test/charts\n  version: 2.14.3\n"
        '- name: cert-manager\n  repository: https://example.test/charts\n  version: "v1.16.5"\n'
        "digest: sha256:0000\n"
        'generated: "2026-01-01T00:00:00Z"\n'
    )
    for archive in ("keda-2.14.3.tgz", "cert-manager-v1.16.5.tgz"):
        (chart / "charts" / archive).write_bytes(b"archive")
    monkeypatch.setattr(helm, "resolve_project_root", lambda: tmp_path)
    return chart


def test_dependencies_are_present_when_charts_match_the_lock(locked_chart: Path) -> None:
    """Every pinned archive on disk and nothing else means no build is needed."""
    assert helm.locked_dependency_archives() == {"keda-2.14.3.tgz", "cert-manager-v1.16.5.tgz"}
    assert helm.locked_dependencies_present()


@pytest.mark.parametrize("state", ["missing", "other-version", "leftover", "directory", "no-charts", "no-lock"])
def test_dependencies_are_absent_unless_charts_match_the_lock_exactly(locked_chart: Path, state: str) -> None:
    """Anything short of an exact match sends the task back to a full build."""
    charts = locked_chart / "charts"
    if state == "missing":
        (charts / "keda-2.14.3.tgz").unlink()
    elif state == "other-version":
        (charts / "keda-2.14.3.tgz").rename(charts / "keda-2.14.2.tgz")
    elif state == "leftover":
        (charts / "keda-2.14.2.tgz").write_bytes(b"archive")
    elif state == "directory":
        (charts / "keda-2.14.3.tgz").unlink()
        (charts / "keda-2.14.3.tgz").mkdir()
    elif state == "no-charts":
        for archive in charts.iterdir():
            archive.unlink()
        charts.rmdir()
    else:
        (locked_chart / "Chart.lock").unlink()
    assert not helm.locked_dependencies_present()


@pytest.fixture
def helm_calls(locked_chart: Path, monkeypatch) -> list[list[str]]:
    """Record the Helm commands one task run issues, without running Helm."""
    calls: list[list[str]] = []
    monkeypatch.setattr(helm, "check_helm", lambda: True)
    monkeypatch.setattr(helm, "run_helm", lambda args: calls.append(args) or 0)
    monkeypatch.setattr(helm, "_sync_configs_to_helm", lambda: None)
    monkeypatch.setattr(helm, "_cleanup_helm_configs", lambda: None)
    return calls


def run_task(monkeypatch, command: str) -> int:
    """Run the task's entry point as `mise run helm -- <command>` would."""
    monkeypatch.setattr(helm, "get_usage_flag", lambda _: command)
    monkeypatch.setattr(helm.sys, "argv", ["helm.py", command])
    return helm.main()


@pytest.mark.parametrize("command", ["lint", "template"])
def test_render_commands_reuse_dependencies_that_match_the_lock(helm_calls, monkeypatch, command: str) -> None:
    """A render command downloads nothing when charts/ already matches the lock."""
    assert run_task(monkeypatch, command) == 0
    assert [call[0] for call in helm_calls] == [command]


def test_render_commands_build_dependencies_that_do_not_match(locked_chart: Path, helm_calls, monkeypatch) -> None:
    """A missing archive brings back the full build before the command runs."""
    (locked_chart / "charts" / "keda-2.14.3.tgz").unlink()
    assert run_task(monkeypatch, "lint") == 0
    assert [call[:2] for call in helm_calls] == [
        ["repo", "add"],
        ["dependency", "build"],
        ["lint", "deploy/helm/sie-cluster"],
    ]


def test_explicit_dependencies_always_build(helm_calls, monkeypatch) -> None:
    """`dependencies` rebuilds even when charts/ already matches the lock."""
    assert helm.locked_dependencies_present()
    assert run_task(monkeypatch, "dependencies") == 0
    assert [call[:2] for call in helm_calls] == [["repo", "add"], ["dependency", "build"]]


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
