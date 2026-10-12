"""GLM-OCR's SGLang profile keeps SGLang's fast image processor."""

from pathlib import Path

import yaml

_CONFIG = Path(__file__).resolve().parents[2] / "models" / "zai-org__GLM-OCR.yaml"


def test_glm_ocr_default_profile_uses_the_fast_image_processor() -> None:
    default = yaml.safe_load(_CONFIG.read_text())["profiles"]["default"]
    assert default["adapter_path"].endswith(":SGLangVisionExtractAdapter")
    loadtime = default["adapter_options"]["loadtime"]
    assert "--disable-fast-image-processor" not in loadtime.get("extra_launch_args", [])
    assert "processor_use_fast" not in loadtime
