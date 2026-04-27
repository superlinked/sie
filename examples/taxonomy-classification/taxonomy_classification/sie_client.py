from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from sie_sdk import SIEClient

DEFAULT_SIE_BASE_URL = "http://localhost:8080"


def project_env_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".env"


def read_sie_settings() -> tuple[str, str | None]:
    values = dotenv_values(project_env_path())
    base_url = (
        os.environ.get("SIE_BASE_URL")
        or values.get("SIE_BASE_URL")
        or DEFAULT_SIE_BASE_URL
    )
    api_key = os.environ.get("SIE_API_KEY") or values.get("SIE_API_KEY")
    return base_url, api_key


def create_sie_client(*, timeout_s: float = 30) -> SIEClient:
    base_url, api_key = read_sie_settings()
    return SIEClient(base_url, timeout_s=timeout_s, api_key=api_key)
