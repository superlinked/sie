from __future__ import annotations

import fnmatch
import json
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from tools.ci import real_models, required_ci

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/real-models.yml"


@pytest.fixture(autouse=True)
def _repository_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def pull_request_paths() -> list[str]:
    triggers = workflow().get("on", workflow().get(True))
    return triggers["pull_request"]["paths"]


def covered(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in pull_request_paths())


def test_every_pinned_model_selects_handwritten_tests():
    node_ids = real_models.pinned_node_ids()
    assert len(node_ids) == len(set(node_ids))
    for model in real_models.PINNED_MODELS:
        assert real_models.pinned_node_ids([model])


def test_unknown_pinned_model_is_rejected():
    with pytest.raises(ValueError, match="no handwritten model tests"):
        real_models.pinned_node_ids(["made-up/does-not-exist"])


def test_lane_models_pin_immutable_revisions():
    manifest = real_models.weights_manifest(ROOT / real_models.MODELS_DIR)
    assert set(manifest) == set(real_models.lane_models())
    for model, entry in manifest.items():
        assert re.fullmatch(r"[0-9a-f]{40}", entry["hf_revision"]), model
        for revision in entry["hf_tokenizer_dependencies"].values():
            assert re.fullmatch(r"[0-9a-f]{40}", revision), model


def test_parity_selections_name_checked_in_goldens():
    adapters = ROOT / "packages/sie_server/tests/adapters"
    stems = {path.stem for path in (adapters / "goldens").rglob("*.json")}
    stems |= {path.stem for path in (adapters / "fixtures").rglob("*.json")}
    for suite in real_models.PARITY_SUITES:
        assert (ROOT / suite.path).is_file()
        for keyword in suite.keyword.split(" or "):
            assert keyword in stems, keyword


def _copy_models(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    shutil.copytree(ROOT / real_models.MODELS_DIR, models)
    return models


def test_cache_key_follows_lane_revisions_only(tmp_path):
    models = _copy_models(tmp_path)
    key = real_models.cache_key(models)
    assert key.startswith(real_models.CACHE_KEY_PREFIX)
    assert real_models.cache_key(models) == key

    unrelated = next(
        path
        for path in sorted(models.glob("*.yaml"))
        if yaml.safe_load(path.read_text()).get("sie_id") not in real_models.lane_models()
    )
    unrelated.write_text(unrelated.read_text() + "\n# unrelated edit\n")
    assert real_models.cache_key(models) == key

    bge = models / "BAAI__bge-m3.yaml"
    revision = yaml.safe_load(bge.read_text())["hf_revision"]
    bge.write_text(bge.read_text().replace(revision, "0" * 40))
    assert real_models.cache_key(models) != key


def test_missing_lane_model_config_is_rejected(tmp_path):
    models = _copy_models(tmp_path)
    (models / "BAAI__bge-m3.yaml").unlink()
    with pytest.raises(ValueError, match="BAAI/bge-m3 has no model config"):
        real_models.cache_key(models)


def test_suite_commands_select_the_marked_tests():
    [(_, pins)] = real_models.suite_commands("pins")
    assert pins[:3] == [sys.executable, "-m", "pytest"]
    assert pins[pins.index("-m", 3) + 1] == "model"
    assert set(real_models.pinned_node_ids()) <= set(pins)

    parity = real_models.suite_commands("parity")
    assert [command[command.index("-k") + 1] for _, command in parity] == [
        suite.keyword for suite in real_models.PARITY_SUITES
    ]

    [(_, server)] = real_models.suite_commands("server")
    assert server[server.index("-m", 3) + 1] == "integration"
    assert server[-len(real_models.SERVER_TESTS) :] == list(real_models.SERVER_TESTS)


@pytest.mark.parametrize("suite", sorted(real_models.SHARED_SERVER_SUITES))
def test_shared_server_suites_need_a_server(suite):
    with pytest.raises(ValueError, match="needs a running server"):
        real_models.suite_commands(suite)


def test_every_python_integration_with_live_tests_runs():
    live = {path.parts[-3] for path in ROOT.glob("integrations/sie_*/tests/test_integration.py")}
    assert live == set(real_models.PYTHON_INTEGRATIONS)
    commands = real_models.suite_commands("integrations", "http://127.0.0.1:1")
    assert [command[-1] for _, command in commands] == [
        f"integrations/{name}/tests" for name in real_models.PYTHON_INTEGRATIONS
    ]


def test_every_typescript_package_with_live_tests_runs():
    live = {
        path.relative_to(ROOT).parts[:2]
        for path in ROOT.glob("*/*/tests/integration/*.integration.test.ts")
        if path.relative_to(ROOT).parts[0] in {"packages", "integrations"}
    }
    assert {"/".join(parts) for parts in live} == set(real_models.TYPESCRIPT_PACKAGES)
    commands = real_models.suite_commands("typescript", "http://127.0.0.1:1")
    assert [command for _, command in commands] == [
        ["mise", "exec", "--", "pnpm", "--dir", package, "run", "test:integration"]
        for package in real_models.TYPESCRIPT_PACKAGES
    ]


def test_suite_environment_drops_credentials_and_disables_telemetry():
    base = {"PATH": "/bin", "HF_TOKEN": "hf-test", "SIE_API_KEY": "sie-test", "SIE_TELEMETRY_DISABLED": "0"}
    env = real_models.suite_environment(base, "http://127.0.0.1:9")
    assert env["PATH"] == "/bin"
    assert env["SIE_TELEMETRY_DISABLED"] == "1"
    assert env["SIE_SERVER_URL"] == "http://127.0.0.1:9"
    assert not set(real_models.SCRUBBED_ENV) & set(env)
    assert "SIE_SERVER_URL" not in real_models.suite_environment(base)


def test_server_serves_the_models_on_loopback_cpu():
    command = real_models.server_command(real_models.SERVED_MODELS, 8123)
    assert command[:4] == [sys.executable, "-m", "sie_server.cli", "serve"]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("-d") + 1] == "cpu"
    assert command[-2:] == ["-m", ",".join(real_models.SERVED_MODELS)]


def test_warm_up_loads_each_model_through_its_task(monkeypatch):
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(real_models, "SIEClient", Mock(return_value=client))
    configs = real_models.model_configs(ROOT / real_models.MODELS_DIR)
    real_models.warm_up("http://127.0.0.1:1", real_models.SERVED_MODELS, configs)
    assert [call.args[0] for call in client.encode.call_args_list] == ["BAAI/bge-m3"]
    assert [call.args[0] for call in client.score.call_args_list] == ["jinaai/jina-reranker-v2-base-multilingual"]
    assert [call.args[0] for call in client.extract.call_args_list] == ["urchade/gliner_multi-v2.1"]


def _fake_run(failing: set[str]):
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        failed = any(marker in " ".join(command) for marker in failing)
        return subprocess.CompletedProcess(command, 1 if failed else 0)

    return run, calls


def test_run_keeps_going_and_reports_every_failed_suite(monkeypatch):
    run, calls = _fake_run({"test_laya_parity.py", "integrations/sie_qdrant/tests"})
    monkeypatch.setattr(real_models.subprocess, "run", run)
    served: list[tuple[str, ...]] = []

    @contextmanager
    def serve(models):
        served.append(tuple(models))
        yield "http://127.0.0.1:1"

    monkeypatch.setattr(real_models, "serve", serve)
    failed = real_models.run(real_models.SUITES)
    assert failed == [
        "parity packages/sie_server/tests/adapters/test_laya_parity.py",
        "integration sie_qdrant",
    ]
    assert served == [real_models.SERVED_MODELS]
    expected = sum(len(real_models.suite_commands(suite, "http://127.0.0.1:1")) for suite in real_models.SUITES)
    assert len(calls) == expected


def test_a_command_that_cannot_start_is_reported_and_the_rest_still_run(monkeypatch):
    calls: list[str] = []

    def run(command, **kwargs):
        calls.append(command[0])
        if command[0] == "missing":
            raise FileNotFoundError(2, "No such file or directory", "missing")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(real_models.subprocess, "run", run)
    commands = [("first", ["missing"]), ("second", ["present"])]
    assert real_models.run_commands(commands, {}) == ["first"]
    assert calls == ["missing", "present"]


def test_shared_server_failure_is_reported(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run, calls = _fake_run(set())
    monkeypatch.setattr(real_models.subprocess, "run", run)

    @contextmanager
    def serve(_models):
        raise RuntimeError("SIE server exited before becoming ready")
        yield

    monkeypatch.setattr(real_models, "serve", serve)
    assert real_models.run(["server", "integrations", "typescript"]) == ["shared server"]
    assert len(calls) == 1


def test_main_prints_the_cache_key(capsys):
    assert real_models.main(["cache-key"]) == 0
    assert capsys.readouterr().out.strip() == real_models.cache_key()


@pytest.mark.parametrize(("failed", "code"), [([], 0), (["pins"], 1)])
def test_main_runs_selected_suites_in_lane_order(monkeypatch, failed, code):
    run = Mock(return_value=failed)
    monkeypatch.setattr(real_models, "run", run)
    assert real_models.main(["run", "--suite", "typescript", "--suite", "pins"]) == code
    run.assert_called_once_with(["pins", "typescript"])


def test_workflow_is_advisory_read_only_and_pinned():
    lane = workflow()
    triggers = lane.get("on", lane.get(True))
    assert set(triggers) == {"schedule", "workflow_dispatch", "pull_request"}
    assert lane["permissions"] == {"contents": "read"}
    assert set(lane["jobs"]) == {"cpu"}
    job = lane["jobs"]["cpu"]
    assert job["name"] == "Real models / CPU"
    assert re.fullmatch(r"blacksmith-[248]vcpu-ubuntu-2404", job["runs-on"])
    assert job["timeout-minutes"] <= 60
    for key in ("if", "environment", "secrets", "permissions", "continue-on-error"):
        assert key not in job
    for step in job["steps"]:
        if "uses" in step:
            assert re.fullmatch(r"[\w/-]+@[a-f0-9]{40}", step["uses"])
    for service in job["services"].values():
        assert re.fullmatch(r"[\w./-]+:[\w.-]+@sha256:[0-9a-f]{64}", service["image"])
    serialized = json.dumps(lane)
    for forbidden in ("pull_request_target", "id-token", "secrets."):
        assert forbidden not in serialized
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert "real-models" not in set(ci["jobs"]) | set(required_ci.MANDATORY_JOBS)


def test_workflow_caches_the_hub_directory_under_the_pinned_revision_key():
    lane = workflow()
    steps = lane["jobs"]["cpu"]["steps"]
    commands = [step.get("run", "") for step in steps]
    key_step = next(step for step in steps if step.get("id") == "weights")
    assert "python -m tools.ci.real_models cache-key" in key_step["run"]
    restore = next(step for step in steps if step.get("uses", "").startswith("actions/cache/restore@"))
    save = next(step for step in steps if step.get("uses", "").startswith("actions/cache/save@"))
    for step in (restore, save):
        assert step["with"] == {"path": ".cache/huggingface/hub", "key": "${{ steps.weights.outputs.key }}"}
    assert lane["env"]["HF_HOME"] == "${{ github.workspace }}/.cache/huggingface"
    suites = next(step for step in steps if step.get("id") == "suites")
    assert "python -m tools.ci.real_models run" in suites["run"]
    assert steps.index(key_step) < steps.index(restore) < steps.index(suites) < steps.index(save)
    assert commands.index("mise run ts -- build") < steps.index(suites)
    assert "!cancelled()" in save["if"]
    assert "steps.restore.outputs.cache-hit != 'true'" in save["if"]
    assert "steps.suites.outcome" in save["if"]


def test_pull_request_paths_cover_every_lane_input():
    inputs = [
        ".github/workflows/real-models.yml",
        "tools/ci/real_models.py",
        "tools/ci/tests/test_real_models.py",
        str(real_models.MODEL_TEST),
        *real_models.SERVER_TESTS,
        *(suite.path for suite in real_models.PARITY_SUITES),
        *(f"integrations/{name}/tests/test_integration.py" for name in real_models.PYTHON_INTEGRATIONS),
        *(f"{package}/src/index.ts" for package in real_models.TYPESCRIPT_PACKAGES),
        "packages/sie_server/models/BAAI__bge-m3.yaml",
        "packages/sie_server/src/sie_server/adapters/gliner/__init__.py",
        "packages/sie_server/src/sie_server/api/extract.py",
        "packages/sie_sdk/src/sie_sdk/client/sync.py",
        "uv.lock",
    ]
    assert [path for path in inputs if not covered(path)] == []
