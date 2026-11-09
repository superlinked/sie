from __future__ import annotations

import io
import re
import shutil
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


def test_locked_dependencies_follow_the_checked_in_lock() -> None:
    """The task's line parser and a YAML parser agree on the pinned dependencies."""
    lock = yaml.safe_load((helm.resolve_project_root() / helm.CHART_DIR / "Chart.lock").read_text())
    expected = {(entry["name"], entry["version"]) for entry in lock["dependencies"]}
    assert expected
    assert helm.locked_dependencies() == expected


def test_lock_digest_is_computed_the_way_helm_writes_it() -> None:
    """The digest Helm wrote into the checked-in lock is reproduced exactly."""
    assert helm.lock_matches_chart()


def write_chart_archive(archive: Path, name: str, version: str) -> None:
    """Write a chart archive that declares the given name and version."""
    manifest = f"apiVersion: v2\nname: {name}\nversion: {version}\n".encode()
    member = tarfile.TarInfo(f"{name}/Chart.yaml")
    member.size = len(manifest)
    with tarfile.open(archive, "w:gz") as chart:
        chart.addfile(member, io.BytesIO(manifest))


@pytest.fixture
def locked_chart(tmp_path: Path, monkeypatch) -> Path:
    """A copy of the checked-in chart and lock, with every pinned archive in charts/."""
    source = helm.resolve_project_root() / helm.CHART_DIR
    chart = tmp_path / helm.CHART_DIR
    (chart / "charts").mkdir(parents=True)
    for name in ("Chart.yaml", "Chart.lock"):
        shutil.copy2(source / name, chart / name)
    for entry in yaml.safe_load((chart / "Chart.lock").read_text())["dependencies"]:
        archive = chart / "charts" / f"{entry['name']}-{entry['version']}.tgz"
        write_chart_archive(archive, entry["name"], entry["version"])
    monkeypatch.setattr(helm, "resolve_project_root", lambda: tmp_path)
    return chart


def a_locked_dependency(chart: Path) -> tuple[Path, str, str]:
    """Return one pinned dependency: its archive in charts/, its name and its version."""
    entry = yaml.safe_load((chart / "Chart.lock").read_text())["dependencies"][0]
    return chart / "charts" / f"{entry['name']}-{entry['version']}.tgz", entry["name"], entry["version"]


def test_dependencies_are_present_when_charts_match_the_lock(locked_chart: Path) -> None:
    """Every pinned archive on disk and nothing else means no build is needed."""
    assert helm.locked_dependencies_present()


def test_dependencies_are_present_whatever_the_archives_are_called(locked_chart: Path) -> None:
    """Helm keeps the file name a repository publishes, which need not be ``<name>-<version>.tgz``."""
    archive, _, _ = a_locked_dependency(locked_chart)
    archive.rename(archive.with_name(f"{archive.stem} copy.tgz"))
    assert helm.locked_dependencies_present()


@pytest.mark.parametrize(
    "state",
    [
        "missing",
        "leftover",
        "duplicate",
        "directory",
        "no-charts",
        "no-lock",
        "stale-lock",
        "tags",
        "declares-other-version",
        "declares-other-chart",
        "unreadable",
        "named-tar-gz",
        "named-hidden",
        "named-underscore",
    ],
)
def test_dependencies_are_absent_unless_charts_match_a_current_lock(locked_chart: Path, state: str) -> None:
    """Anything short of an exact match with a current lock sends the task back to a full build."""
    charts = locked_chart / "charts"
    archive, name, version = a_locked_dependency(locked_chart)
    manifest = locked_chart / "Chart.yaml"
    # The first dependency's version line: the chart's own version is not indented.
    constraint = re.compile(r"(?m)^(    version: ).*$")
    if state == "missing":
        archive.unlink()
    elif state == "leftover":
        # Another version of a locked chart, left beside the locked one.
        write_chart_archive(charts / f"{name}-0.0.0.tgz", name, "0.0.0")
    elif state == "duplicate":
        shutil.copy2(archive, charts / f"{archive.stem} copy.tgz")
    elif state == "directory":
        archive.unlink()
        archive.mkdir()
    elif state == "no-charts":
        shutil.rmtree(charts)
    elif state == "no-lock":
        (locked_chart / "Chart.lock").unlink()
    elif state == "stale-lock":
        # Chart.yaml asks for another version and the lock was not regenerated.
        manifest.write_text(constraint.sub(r"\g<1>0.0.x", manifest.read_text(), count=1))
    elif state == "declares-other-version":
        # The file is named after the locked version, the chart inside is another one.
        write_chart_archive(archive, name, "0.0.0")
    elif state == "declares-other-chart":
        write_chart_archive(archive, "another-chart", version)
    elif state == "unreadable":
        archive.write_bytes(b"not an archive")
    elif state == "named-tar-gz":
        # The right chart under a name Helm does not load as a dependency.
        archive.rename(archive.with_suffix(".tar.gz"))
    elif state == "named-hidden":
        archive.rename(archive.with_name(f".{archive.name}"))
    elif state == "named-underscore":
        archive.rename(archive.with_name(f"_{archive.name}"))
    else:
        # A field the task does not read: the digest cannot be confirmed.
        manifest.write_text(constraint.sub(r"\g<0>\n    tags:\n      - extra", manifest.read_text(), count=1))
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
    a_locked_dependency(locked_chart)[0].unlink()
    assert run_task(monkeypatch, "lint") == 0
    commands = [call[:2] for call in helm_calls]
    assert set(map(tuple, commands[:-2])) == {("repo", "add")}
    assert commands[-2:] == [["dependency", "build"], ["lint", "deploy/helm/sie-cluster"]]


def test_explicit_dependencies_always_build(helm_calls, monkeypatch) -> None:
    """`dependencies` rebuilds even when charts/ already matches the lock."""
    assert helm.locked_dependencies_present()
    assert run_task(monkeypatch, "dependencies") == 0
    commands = [call[:2] for call in helm_calls]
    assert set(map(tuple, commands[:-1])) == {("repo", "add")}
    assert commands[-1] == ["dependency", "build"]


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
