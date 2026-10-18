"""Small, offline native Qwen3.5 checks in the optional Transformers 5 lane."""

import copy

import pytest
import torch
from safetensors.torch import load_file, save_file
from sie_server.adapters.pplx_decider.model import enable_noncausal_full_attention, load_model

qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5", reason="requires the transformers5 bundle")
config_module = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")


@pytest.fixture
def native():
    torch.manual_seed(790)
    config = config_module.Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [3, 3, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 2,
            "num_position_embeddings": 16,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    config._attn_implementation = "sdpa"
    return qwen.Qwen3_5Model(config).eval()


def test_noncausal_hook_changes_future_attention_preserves_padding_and_recurrence(native):
    causal = copy.deepcopy(native)
    enable_noncausal_full_attention(native.language_model)
    rows = {
        "input_ids": torch.tensor([[0, 2, 3, 4], [2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]),
    }
    with torch.inference_mode():
        actual = native(**rows, use_cache=False).last_hidden_state
        before = causal(**rows, use_cache=False).last_hidden_state
        single = native(
            input_ids=rows["input_ids"][:1, 1:], attention_mask=torch.ones(1, 3, dtype=torch.long), use_cache=False
        ).last_hidden_state
    assert torch.isfinite(actual).all()
    assert not torch.allclose(actual[0, 1], before[0, 1])
    torch.testing.assert_close(actual[0, 1:], single[0], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="no KV cache"):
        native(**rows, use_cache=True)


def test_saved_native_checkpoint_loads_complete_backbone_and_separate_readout(native, tmp_path, monkeypatch):
    native.save_pretrained(tmp_path, safe_serialization=True)
    head = torch.nn.Linear(32, 255, bias=False)
    save_file({"weight": head.weight.detach()}, tmp_path / "readout.safetensors")
    saved = {"temperature": 1.7, "pooling": "last", "attention_mode": "noncausal_full_attention"}
    loader = qwen.Qwen3_5Model.from_pretrained
    devices = []

    def observe_load(*args, **kwargs):
        devices.append(kwargs.get("device_map"))
        return loader(*args, **kwargs)

    monkeypatch.setattr(qwen.Qwen3_5Model, "from_pretrained", observe_load)
    loaded = load_model(tmp_path, saved, "cpu", torch.float32)
    assert devices == ["cpu"]
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    enable_noncausal_full_attention(native.language_model)
    inputs = {"input_ids": torch.tensor([[2, 3, 4]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    with torch.inference_mode():
        actual = loaded(inputs, [3])
        hidden = native(**inputs, use_cache=False).last_hidden_state[:, -1]
        expected = (head(hidden).float()[:, :3] / 1.7).softmax(-1)
    torch.testing.assert_close(actual[:, :3], expected, rtol=1e-6, atol=1e-7)
    assert actual[:, 3:].count_nonzero() == 0
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_missing_backbone_tensor_is_rejected(native, tmp_path):
    native.save_pretrained(tmp_path, safe_serialization=True)
    state = load_file(tmp_path / "model.safetensors")
    del state[next(iter(state))]
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="backbone weights"):
        load_model(tmp_path, {"temperature": 1.0, "pooling": "last"}, "cpu", torch.float32)
