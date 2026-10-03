from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from sie_server.config.model import ModelConfig
from sie_server.processors.streaming import _VISION_TOKENS_PER_VIDEO_ESTIMATE

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MAX_TOTAL_PIXELS = _VISION_TOKENS_PER_VIDEO_ESTIMATE * 1024
# SGLang's Qwen-VL video preprocessing never shrinks a frame below
# 1.05 x ``min_pixels`` (default 128 x 28 x 28), so ``total_pixels`` only bounds
# the visual tokens while ``max_frames`` keeps that per-frame floor inside it.
DEFAULT_VIDEO_MIN_PIXELS = 128 * 28 * 28


def _positive_int(value: object) -> int | None:
    """Return a positive ``int`` pixel/frame count, or ``None`` for anything else (``bool`` included)."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _video_budget_violations(config: ModelConfig) -> list[str]:
    if not config.inputs.video or config.tasks.generate is None:
        return []
    violations: list[str] = []
    for name in config.profiles:
        args = [str(arg) for arg in config.resolve_profile(name).loadtime.get("extra_launch_args") or []]
        video: dict[str, Any] | None = {}
        flag = args.index("--mm-process-config") if "--mm-process-config" in args else -1
        if flag >= 0:
            value = args[flag + 1] if flag + 1 < len(args) else ""
            if not value or value.startswith("--"):
                violations.append(f"{config.sie_id}:{name} --mm-process-config has no value")
                continue
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                violations.append(f"{config.sie_id}:{name} --mm-process-config is not JSON")
                continue
            video = decoded.get("video") or {} if isinstance(decoded, dict) else None
            if not isinstance(video, dict):
                violations.append(f"{config.sie_id}:{name} --mm-process-config is not an object")
                continue
        total_pixels = _positive_int(video.get("total_pixels"))
        max_frames = _positive_int(video.get("max_frames"))
        min_pixels = _positive_int(video.get("min_pixels", DEFAULT_VIDEO_MIN_PIXELS))
        if total_pixels is None or total_pixels > MAX_TOTAL_PIXELS:
            violations.append(f"{config.sie_id}:{name} video.total_pixels={video.get('total_pixels')!r}")
        # Integer arithmetic throughout: a float conversion of a pathological
        # count would raise OverflowError instead of recording a violation.
        # ``frames * min_pixels * 1.05 / 2 > budget`` <=> ``frames * min_pixels * 105 > budget * 200``.
        elif max_frames is None or min_pixels is None or max_frames * min_pixels * 105 > MAX_TOTAL_PIXELS * 200:
            violations.append(
                f"{config.sie_id}:{name} video.max_frames={video.get('max_frames')!r} "
                f"min_pixels={video.get('min_pixels', DEFAULT_VIDEO_MIN_PIXELS)!r}"
            )
    return violations


def test_video_generation_profiles_bound_visual_tokens() -> None:
    violations = []
    for path in sorted(MODELS_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if (data.get("inputs") or {}).get("video") and (data.get("tasks") or {}).get("generate") is not None:
            violations += _video_budget_violations(ModelConfig.model_validate(data))
    assert not violations, (
        "video generation profiles must set --mm-process-config video.total_pixels "
        f"<= {MAX_TOTAL_PIXELS} (the worker's per-video token estimate): {violations}"
    )


def _synthetic(video_block: dict[str, Any] | None, *, trailing: list[str] | None = None) -> ModelConfig:
    args = ["--mm-process-config", json.dumps({"video": video_block})] if video_block is not None else []
    if trailing is not None:
        args = ["--mm-process-config", *trailing]
    return ModelConfig.model_validate(
        {
            "sie_id": "org/video-model",
            "hf_id": "org/video-model",
            "inputs": {"text": True, "video": True},
            "tasks": {"generate": {"context_length": 32768, "max_output_tokens": 1024}},
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.sglang.generation:SGLangGenerationAdapter",
                    "max_batch_tokens": 1024,
                    "kv_budget_tokens": 32768,
                    "adapter_options": {"loadtime": {"extra_launch_args": args}},
                }
            },
        }
    )


@pytest.mark.parametrize(
    ("video_block", "violates"),
    [
        ({"fps": 2, "max_frames": 64, "total_pixels": MAX_TOTAL_PIXELS}, False),
        ({"fps": 2, "max_frames": 64, "total_pixels": MAX_TOTAL_PIXELS + 1}, True),
        ({"fps": 2, "total_pixels": MAX_TOTAL_PIXELS}, True),
        ({"fps": 2, "max_frames": 768, "total_pixels": MAX_TOTAL_PIXELS}, True),
        ({"fps": 2, "max_frames": 64}, True),
        ({"fps": 2, "max_frames": 64, "total_pixels": MAX_TOTAL_PIXELS, "min_pixels": None}, True),
        ({"fps": 2, "max_frames": 64, "total_pixels": MAX_TOTAL_PIXELS, "min_pixels": "100352"}, True),
        ({"fps": 2, "max_frames": True, "total_pixels": MAX_TOTAL_PIXELS}, True),
        ({"fps": 2, "max_frames": 64, "total_pixels": True}, True),
        ({"fps": 2, "max_frames": 0, "total_pixels": MAX_TOTAL_PIXELS}, True),
        ({"fps": 2, "max_frames": 10**400, "total_pixels": MAX_TOTAL_PIXELS}, True),
        (None, True),
    ],
)
def test_video_budget_check_detects_unbounded_profiles(video_block: dict[str, Any] | None, violates: bool) -> None:
    assert bool(_video_budget_violations(_synthetic(video_block))) is violates


@pytest.mark.parametrize(
    ("trailing", "expected"),
    [
        ([], "--mm-process-config has no value"),
        (["--kv-cache-dtype", "bfloat16"], "--mm-process-config has no value"),
        (["not json"], "--mm-process-config is not JSON"),
        (["null"], "--mm-process-config is not an object"),
        (['["video"]'], "--mm-process-config is not an object"),
        (['{"video": 5}'], "--mm-process-config is not an object"),
    ],
)
def test_video_budget_check_reports_a_flag_without_a_usable_value(trailing: list[str], expected: str) -> None:
    assert _video_budget_violations(_synthetic(None, trailing=trailing)) == [f"org/video-model:default {expected}"]
