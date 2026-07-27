from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from sie_server.adapters import _colbert_projection
from sie_server.adapters._colbert_projection import (
    ColBERTProjectionLoadError,
    load_standalone_colbert_projection,
)


def _write_root_weights(root: Path, state: dict[str, torch.Tensor]) -> str:
    root.mkdir()
    save_file(state, str(root / "model.safetensors"))
    return str(root)


def test_loads_exact_bias_free_projection_weights_and_shape(tmp_path: Path) -> None:
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    checkpoint = _write_root_weights(tmp_path / "checkpoint", {"linear.weight": weight})

    projection = load_standalone_colbert_projection(
        checkpoint,
        revision="exact-revision",
        device="cpu",
        dtype=torch.bfloat16,
        expected_in_features=4,
        expected_out_features=3,
        allow_bias=False,
        required=True,
    )

    assert projection is not None
    assert projection.in_features == 4
    assert projection.out_features == 3
    assert projection.bias is None
    assert projection.weight.dtype == torch.bfloat16
    torch.testing.assert_close(projection.weight.float(), weight)


def test_hub_download_receives_exact_model_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weight = torch.eye(2)
    checkpoint = _write_root_weights(tmp_path / "checkpoint", {"linear.weight": weight})
    calls: list[tuple[str, str, str | None]] = []

    def fake_download(model_id: str, filename: str, *, revision: str | None) -> str:
        calls.append((model_id, filename, revision))
        return str(Path(checkpoint) / "model.safetensors")

    monkeypatch.setattr(_colbert_projection, "hf_hub_download", fake_download)

    projection = load_standalone_colbert_projection(
        "jinaai/jina-colbert-v2",
        revision="4552c4dc",
        device="cpu",
        dtype=torch.float32,
        required=True,
    )

    assert projection is not None
    assert calls == [("jinaai/jina-colbert-v2", "model.safetensors", "4552c4dc")]


def test_optional_contract_preserves_missing_projection_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args, **_kwargs):
        raise OSError("missing")

    monkeypatch.setattr(_colbert_projection, "hf_hub_download", missing)

    projection = load_standalone_colbert_projection(
        "checkpoint-without-root-projection",
        revision="abc",
        device="cpu",
        dtype=torch.float32,
    )

    assert projection is None


def test_required_contract_fails_closed_when_projection_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args, **_kwargs):
        raise OSError("missing")

    monkeypatch.setattr(_colbert_projection, "hf_hub_download", missing)

    with pytest.raises(ColBERTProjectionLoadError, match=r"Unable to resolve root model\.safetensors"):
        load_standalone_colbert_projection(
            "jinaai/jina-colbert-v2",
            revision="4552c4dc",
            device="cpu",
            dtype=torch.float32,
            required=True,
        )


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"linear.weight": torch.ones(2, 5)}, "in_features=5 does not match model hidden_size=4"),
        (
            {"linear.weight": torch.ones(2, 4), "linear.bias": torch.ones(2)},
            "unsupported linear.bias",
        ),
    ],
)
def test_required_contract_rejects_malformed_projection(
    tmp_path: Path,
    state: dict[str, torch.Tensor],
    message: str,
) -> None:
    checkpoint = _write_root_weights(tmp_path / "checkpoint", state)

    with pytest.raises(ColBERTProjectionLoadError, match=message):
        load_standalone_colbert_projection(
            checkpoint,
            revision="4552c4dc",
            device="cpu",
            dtype=torch.float32,
            expected_in_features=4,
            expected_out_features=2,
            allow_bias=False,
            required=True,
        )
