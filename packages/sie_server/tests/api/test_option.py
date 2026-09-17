from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from sie_server.api.options import resolve_runtime_options_with_profile
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks


def _make_config() -> ModelConfig:
    return ModelConfig(
        sie_id="test",
        hf_id="org/test",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=8))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.base:ModelAdapter",
                max_batch_tokens=8,
            )
        },
    )


def test_invalid_overflow_policy_type_returns_400() -> None:
    span = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        resolve_runtime_options_with_profile(
            _make_config(),
            {"overflow_policy": ["truncate_text"]},
            span,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "INVALID_INPUT"
