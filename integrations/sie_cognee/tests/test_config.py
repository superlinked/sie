"""Inspect actual released Cognee settings in a fresh isolated process."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from sie_cognee.config import CogneeSettings, Endpoint, configure


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:8080", "http://localhost:8080/v1"),
        ("http://localhost:8080/v1/", "http://localhost:8080/v1"),
        ("https://models.example.test/prefix", "https://models.example.test/prefix/v1"),
    ],
)
def test_compatible_url(url, expected):
    assert Endpoint(url).compatible_url() == expected


@pytest.mark.parametrize(
    "url", ["file:///tmp/model", "https://user:password@example.test", "https://example.test/?key=x"]
)
def test_rejects_credential_bearing_or_invalid_urls(url):
    with pytest.raises(ValueError, match="Endpoint must"):
        Endpoint(url).compatible_url()


def test_local_placeholder_never_silently_authenticates_remote():
    assert Endpoint("http://127.0.0.1:8080").compatible_key() == "sie-local-no-key"
    with pytest.raises(ValueError, match="explicit API key"):
        Endpoint("https://models.example.test").compatible_key()
    assert "test-key" not in repr(Endpoint("https://models.example.test", "test-key"))


@pytest.fixture
def settings(tmp_path):
    return CogneeSettings(
        Endpoint("http://localhost:8080"),
        Endpoint("http://localhost:8081"),
        "generation",
        "embedding",
        3,
        tmp_path / "state",
    )


@pytest.mark.parametrize("dimensions", [0, -1, True, 1.5])
def test_rejects_invalid_dimensions_before_import(monkeypatch, settings, dimensions):
    monkeypatch.delitem(sys.modules, "cognee", raising=False)
    with pytest.raises(ValueError, match="positive integer"):
        configure(replace(settings, embedding_dimensions=dimensions))


def test_rejects_stage_fallback_and_unowned_storage(monkeypatch, settings):
    monkeypatch.delitem(sys.modules, "cognee", raising=False)
    monkeypatch.setenv("LLM_QUERY_ENDPOINT", "https://unrelated.example.test")
    with pytest.raises(ValueError, match="LLM_QUERY_ENDPOINT"):
        configure(settings)
    monkeypatch.delenv("LLM_QUERY_ENDPOINT")
    settings.storage_root.mkdir()
    (settings.storage_root / "existing.db").touch()
    with pytest.raises(ValueError, match="unowned"):
        configure(settings)


def test_actual_configuration_and_storage_isolation(tmp_path):
    script = Path(__file__).with_name("config_probe.py")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("LLM_", "EMBEDDING_", "FALLBACK_"))}
    env.update({"HF_HUB_OFFLINE": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "TELEMETRY_DISABLED": "true"})
    result = subprocess.run(  # noqa: S603 - fixed local contract probe
        [sys.executable, str(script), str(tmp_path / "state")],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-4B", "openai/test-model"])
def test_released_cognee_remember_recall_http_contract(tmp_path, model):
    script = Path(__file__).with_name("http_contract_probe.py")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("LLM_", "EMBEDDING_", "FALLBACK_"))}
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "TELEMETRY_DISABLED": "true",
            "KUZU_BUFFER_POOL_SIZE": "16777216",
            "KUZU_MAX_DB_SIZE": "536870912",
            "KUZU_NUM_THREADS": "1",
            "GRAPH_DATABASE_SUBPROCESS_ENABLED": "false",
            "VECTOR_DB_SUBPROCESS_ENABLED": "false",
        }
    )
    result = subprocess.run(  # noqa: S603 - fixed local contract probe
        [sys.executable, str(script), str(tmp_path / "state"), model],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
