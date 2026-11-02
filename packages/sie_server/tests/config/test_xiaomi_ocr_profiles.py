"""Xiaomi-OCR-0 serves native weights with the model card's prompts."""

from pathlib import Path

import yaml

_CONFIG = Path(__file__).resolve().parents[2] / "models" / "xiaomi-research__Xiaomi-OCR-0.yaml"
_PROCESSOR_MAX_PAGE_TOKENS = 16_777_216 // (32 * 32)
_PROMPT_TOKENS = 256


def _profiles() -> dict:
    return yaml.safe_load(_CONFIG.read_text())["profiles"]


def test_sglang_profile_never_trusts_repository_code() -> None:
    default = _profiles()["default"]
    assert default["adapter_path"].endswith(":SGLangVisionExtractAdapter")
    assert default["adapter_options"]["loadtime"]["trust_remote_code"] is False


def test_both_profiles_default_to_the_same_card_prompt_and_decoding() -> None:
    profiles = _profiles()
    reference = profiles["transformers"]["adapter_options"]
    sglang = profiles["default"]["adapter_options"]
    task = sglang["loadtime"]["default_task"]
    assert sglang["loadtime"]["task_prompts"][task] == reference["loadtime"]["user_text"]
    assert sglang["runtime"] == reference["runtime"] == {"max_new_tokens": 4096, "num_beams": 1}


def test_a_full_resolution_page_and_its_output_fit_the_sequence_window() -> None:
    config = yaml.safe_load(_CONFIG.read_text())
    for profile in config["profiles"].values():
        loadtime = profile["adapter_options"]["loadtime"]
        assert "--mm-process-config" not in loadtime.get("extra_launch_args", [])
        budget = _PROCESSOR_MAX_PAGE_TOKENS + profile["adapter_options"]["runtime"]["max_new_tokens"] + _PROMPT_TOKENS
        assert budget <= config["max_sequence_length"]


def test_sglang_profile_keeps_its_l4_calibrated_reservation() -> None:
    loadtime = _profiles()["default"]["adapter_options"]["loadtime"]
    assert loadtime["mem_fraction_static"] == 0.70
    assert loadtime["max_concurrent_requests"] == 32
