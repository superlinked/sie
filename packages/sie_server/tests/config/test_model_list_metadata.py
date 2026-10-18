"""System One models carry the ``description`` and ``release_date`` /v1/models reports.

System One clients (the TypeSafe SDKs) list models and require both fields as
strings. /v1/models reports "" for a config that sets neither, so only the
System One models need real values.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import yaml

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

SYSTEM_ONE_CONFIGS = (
    "convaiinnovations__laya.yaml",
    "convaiinnovations__laya-multilingual.yaml",
    "convaiinnovations__laya-typed-decisions.yaml",
    "fastino__GLiNER2.5-Decide.yaml",
    "fastino__GLiNER2.5-Decide-1B.yaml",
    "fastino__GLiNER2.5-multi-Decide.yaml",
)


@pytest.mark.parametrize("name", SYSTEM_ONE_CONFIGS)
def test_system_one_config_has_list_metadata(name: str) -> None:
    config = yaml.safe_load((MODELS_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(config.get("description"), str), "no description"
    assert config["description"].strip(), "blank description"
    release_date = config.get("release_date")
    assert isinstance(release_date, str), "release_date must be a quoted YYYY-MM-DD string"
    assert dt.date.fromisoformat(release_date).isoformat() == release_date
