from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
RECORDED_DIR = ROOT / "recorded" / "modal-direct"
RUNS_DIR = ROOT / "runs"

SOURCE_IMAGE = ASSETS_DIR / "042-pharmacy-oos-sign.jpg"

DINO_MODEL = "IDEA-Research/grounding-dino-base"
DINO_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
OCR_MODEL = "lightonai/LightOnOCR-2-1B"
OCR_REVISION = "c97bd377f04481830395218fa8951df9deaba756"
DETECTION_LABELS = ["empty shelf space", "out of stock sign", "price tag"]
DETECTION_OPTIONS = {"box_threshold": 0.2, "text_threshold": 0.2}


@dataclass(frozen=True)
class RuntimeConfig:
    base_url: str
    api_key: str
    request_timeout_s: float
    provision_timeout_s: float


def load_config() -> RuntimeConfig:
    load_dotenv(ROOT / ".env")
    return RuntimeConfig(
        base_url=os.environ.get("SIE_BASE_URL", "http://localhost:8080").rstrip("/"),
        api_key=os.environ.get("SIE_API_KEY", ""),
        request_timeout_s=float(os.environ.get("SIE_REQUEST_TIMEOUT_S", "900")),
        provision_timeout_s=float(os.environ.get("SIE_PROVISION_TIMEOUT_S", "900")),
    )
