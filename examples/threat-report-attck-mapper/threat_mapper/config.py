from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
CACHE_DIR = ROOT / "data" / "cache"
RUNS_DIR = ROOT / "runs"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["cluster"]["url"] = os.getenv("SIE_CLUSTER_URL", config["cluster"]["url"])
    config["cluster"]["api_key"] = os.getenv("SIE_API_KEY", config["cluster"]["api_key"])
    return config
