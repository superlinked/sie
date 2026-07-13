"""Tests for the configuration system: defaults, YAML, env vars, validation."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from dusk.config import Config, ConfigError, load_config


def test_defaults() -> None:
    """A default Config exposes the documented threshold values."""
    config = Config()
    assert config.sweep_threshold == 15
    assert config.sweep_window_seconds == 10.0
    assert config.sweep_timing_std_threshold == 0.05
    assert config.boundary_port_threshold == 10
    assert config.boundary_window_seconds == 30.0
    assert config.alert_log_path == "dusk-alerts.json"
    assert config.log_level == "WARNING"
    assert config.enforce is False
    assert config.sie_endpoint == "http://sie:8080"
    assert config.sie_encode_model == "BAAI/bge-m3"
    assert config.sie_score_model == "BAAI/bge-reranker-v2-m3"
    assert config.sie_extract_model == "urchade/gliner_multi-v2.1"
    assert config.sie_timeout_ms == 10000
    assert config.n8n_alert_url == ""
    assert config.n8n_report_url == ""
    assert config.n8n_decision_url == ""
    assert config.n8n_max_workers == 8
    assert config.n8n_max_queued == 200
    assert config.offense_memory_path == ""
    assert config.repeat_offense_max_contribution == 0.3
    assert config.repeat_offense_half_life_days == 30.0


def test_load_defaults_when_no_file(tmp_path: Path) -> None:
    """load_config falls back to defaults when no YAML file is present."""
    config = load_config(str(tmp_path / "missing.yaml"))
    assert config.sweep_threshold == 15


def test_yaml_overrides(tmp_path: Path) -> None:
    """Values in dusk.yaml override the dataclass defaults."""
    yaml_path = tmp_path / "dusk.yaml"
    yaml_path.write_text("sweep_threshold: 5\nboundary_port_threshold: 3\n")
    config = load_config(str(yaml_path))
    assert config.sweep_threshold == 5
    assert config.boundary_port_threshold == 3
    # Untouched fields keep their defaults.
    assert config.sweep_window_seconds == 10.0


@pytest.fixture
def _clean_env() -> Iterator[None]:
    """Remove DUSK_* env vars around a test."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("DUSK_")}
    for key in saved:
        del os.environ[key]
    yield
    for key in list(os.environ):
        if key.startswith("DUSK_"):
            del os.environ[key]
    os.environ.update(saved)


def test_env_overrides_yaml(tmp_path: Path, _clean_env: None) -> None:
    """DUSK_* environment variables take precedence over YAML and defaults."""
    yaml_path = tmp_path / "dusk.yaml"
    yaml_path.write_text("sweep_threshold: 5\n")
    os.environ["DUSK_SWEEP_THRESHOLD"] = "42"
    config = load_config(str(yaml_path))
    assert config.sweep_threshold == 42


def test_invalid_threshold_raises() -> None:
    """A non-positive threshold is rejected at construction."""
    with pytest.raises(ConfigError):
        Config(sweep_threshold=0)


def test_invalid_log_level_raises() -> None:
    """An unrecognised log level name is rejected."""
    with pytest.raises(ConfigError):
        Config(log_level="NONSENSE")


def test_invalid_sie_timeout_raises() -> None:
    """A non-positive SIE timeout is rejected at construction."""
    with pytest.raises(ConfigError):
        Config(sie_timeout_ms=0)


@pytest.mark.parametrize(
    "field", ["sie_endpoint", "n8n_alert_url", "n8n_report_url", "n8n_decision_url"]
)
def test_invalid_url_scheme_raises(field: str) -> None:
    """A configured URL without an http(s) scheme is rejected."""
    with pytest.raises(ConfigError):
        Config(**{field: "ftp://example.com"})


def test_empty_n8n_urls_are_valid() -> None:
    """An unset (empty) n8n URL is valid -- it just means that webhook is off."""
    config = Config(n8n_alert_url="", n8n_report_url="", n8n_decision_url="")
    assert config.n8n_alert_url == ""


def test_enforce_env_override(_clean_env: None) -> None:
    """DUSK_ENFORCE overrides the default, via the same mechanism as every other field."""
    os.environ["DUSK_ENFORCE"] = "true"
    config = load_config("nonexistent-dusk.yaml")
    assert config.enforce is True


def test_sie_endpoint_env_override(_clean_env: None) -> None:
    """DUSK_SIE_ENDPOINT overrides the self-hosted default."""
    os.environ["DUSK_SIE_ENDPOINT"] = "https://hosted-sie.example.com"
    config = load_config("nonexistent-dusk.yaml")
    assert config.sie_endpoint == "https://hosted-sie.example.com"


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    """A YAML file that is not a mapping raises ConfigError."""
    yaml_path = tmp_path / "dusk.yaml"
    yaml_path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        load_config(str(yaml_path))


def test_unknown_key_ignored(tmp_path: Path) -> None:
    """Unknown YAML keys are ignored rather than fatal."""
    yaml_path = tmp_path / "dusk.yaml"
    yaml_path.write_text("sweep_threshold: 7\nnot_a_real_key: 1\n")
    config = load_config(str(yaml_path))
    assert config.sweep_threshold == 7
