from __future__ import annotations

from types import SimpleNamespace

import torch
from sie_server.adapters.colbert_modernbert_flash.adapter import ColBERTModernBERTFlashAdapter


def _bare_adapter() -> ColBERTModernBERTFlashAdapter:
    """Construct without load() (which requires CUDA); set only what methods need."""
    return object.__new__(ColBERTModernBERTFlashAdapter)


def _fake_model(hidden_size: int, **config_extra) -> SimpleNamespace:
    """Minimal stand-in for the loaded backbone; only config attrs are read."""
    return SimpleNamespace(config=SimpleNamespace(hidden_size=hidden_size, **config_extra))


def test_project_applies_chain_then_truncates() -> None:
    """WITH a Dense chain ending at token_dim: sequential matmuls, then a no-op truncate."""
    adapter = _bare_adapter()
    adapter._token_dim = 4
    w0 = torch.randn(6, 8)  # [out_features, in_features]
    w1 = torch.randn(4, 6)
    adapter._dense_chain = [w0, w1]

    hidden = torch.randn(5, 8)
    out = adapter._project(hidden)

    assert out.shape == (5, 4)
    expected = (hidden @ w0.T) @ w1.T
    assert torch.equal(out, expected)


def test_project_single_weight_equivalent_to_old_math() -> None:
    """A length-1 chain must be byte-identical to the pre-#1680 single-head math
    (``hidden @ W.T`` then truncate) — GTE/Reason representations must NOT change.
    """
    adapter = _bare_adapter()
    adapter._token_dim = 4
    weight = torch.randn(6, 8)
    adapter._dense_chain = [weight]

    hidden = torch.randn(5, 8)
    out = adapter._project(hidden)

    assert out.shape == (5, 4)
    assert torch.equal(out, (hidden @ weight.T)[:, :4])


def test_project_pure_truncation_without_chain() -> None:
    """WITHOUT a chain: falls back to backbone truncation (backward-compatible)."""
    adapter = _bare_adapter()
    adapter._token_dim = 4
    adapter._dense_chain = None

    hidden = torch.randn(5, 8)
    out = adapter._project(hidden)

    assert out.shape == (5, 4)
    assert torch.equal(out, hidden[:, :4])


def test_compute_rope_uses_layer_appropriate_theta() -> None:
    """Global layers use global_rope_theta (160k), local layers local_rope_theta (10k).

    With different bases the rotation angles at position >= 1 must differ; at
    position 0 both are identity (cos=1, sin=0).
    """
    adapter = _bare_adapter()
    adapter._device = "cpu"
    adapter._compute_precision = "float32"
    adapter._model = _fake_model(
        hidden_size=8,
        num_attention_heads=2,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
    )

    position_ids = torch.tensor([0, 1, 2, 3])
    global_cos, global_sin = adapter._compute_rope(position_ids, use_global=True)
    local_cos, local_sin = adapter._compute_rope(position_ids, use_global=False)

    assert global_cos.shape == local_cos.shape == (4, 4)  # [total_tokens, head_dim]
    # Position 0 is base-independent (angle 0).
    assert torch.equal(global_cos[0], local_cos[0])
    assert torch.equal(global_sin[0], local_sin[0])
    # Positions >= 1 rotate at different frequencies for different bases.
    assert not torch.allclose(global_cos[1:], local_cos[1:])
    assert not torch.allclose(global_sin[1:], local_sin[1:])
