from __future__ import annotations

from pathlib import Path

from tools.mise_tasks import helm


def test_dependency_command_uses_checked_in_chart() -> None:
    assert helm.dependency_command() == ["dependency", "build", "deploy/helm/sie-cluster"]


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
