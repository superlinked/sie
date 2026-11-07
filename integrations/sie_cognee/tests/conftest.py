import pytest


@pytest.fixture(autouse=True)
def isolated_import_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("COGNEE_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TELEMETRY_DISABLED", "true")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
