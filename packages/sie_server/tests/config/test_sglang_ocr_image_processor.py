"""SGLang OCR profiles keep SGLang's fast image processor."""

from pathlib import Path

import pytest
import yaml

_MODELS = Path(__file__).resolve().parents[2] / "models"


@pytest.mark.parametrize("config", ["zai-org__GLM-OCR.yaml", "PaddlePaddle__PaddleOCR-VL-1.5.yaml"])
def test_default_profile_uses_the_fast_image_processor(config: str) -> None:
    default = yaml.safe_load((_MODELS / config).read_text())["profiles"]["default"]
    assert default["adapter_path"].endswith(":SGLangVisionExtractAdapter")
    loadtime = default["adapter_options"]["loadtime"]
    assert "--disable-fast-image-processor" not in loadtime.get("extra_launch_args", [])
    assert "processor_use_fast" not in loadtime
