"""Native mixed-recurrence Qwen parity against an independent dense attention mask."""

import copy

import pytest
import torch
from safetensors.torch import load_file, save_file
from sie_server.adapters._qwen3_5_packed import reference_kernels
from sie_server.adapters.decision.packed import PackedDecisionReadout
from sie_server.adapters.decision.qwen import QwenReadout, _strict_load, d3_contract

qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5", reason="requires transformers5")
configs = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")


def reference_varlen(query, key, value, cu_query, cu_key, max_query, max_key, *, softmax_scale, causal):
    assert not causal
    outputs = []
    for i in range(len(cu_query) - 1):
        q = query[cu_query[i] : cu_query[i + 1]].transpose(0, 1)[None]
        k = key[cu_key[i] : cu_key[i + 1]].transpose(0, 1)[None]
        v = value[cu_key[i] : cu_key[i + 1]].transpose(0, 1)[None]
        assert q.shape[2] <= max_query
        assert k.shape[2] <= max_key
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=softmax_scale, enable_gqa=True)
        outputs.append(out[0].transpose(0, 1))
    return torch.cat(outputs)


def tiny():
    config = configs.Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [3, 3, 2],
        },
    )
    config._attn_implementation = "sdpa"
    return qwen.Qwen3_5TextModel(config).eval()


def test_native_block_attention_preserves_recurrence_and_matches_dense_mask():
    from transformers.masking_utils import create_recurrent_attention_mask

    torch.manual_seed(41)
    reference = tiny()
    weight = torch.randn(255, 32)
    model = QwenReadout(copy.deepcopy(reference), weight)
    ids = torch.tensor([[0, 0, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]])
    present = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    bounds = ((2, 4), (0, 3))
    mask = torch.zeros(2, 1, 6, 6, dtype=torch.bool)
    for row, (first, end) in enumerate(bounds):
        mask[row, :, first:end, first:end] = True
        mask[row, :, end:, first:] = True
    with torch.inference_mode():
        embeddings = reference.embed_tokens(ids)
        linear = create_recurrent_attention_mask(
            config=reference.config, inputs_embeds=embeddings, attention_mask=present
        )
        expected = (
            reference(
                inputs_embeds=embeddings,
                position_ids=torch.tensor([[0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5]]),
                attention_mask={"full_attention": mask, "linear_attention": linear},
                use_cache=False,
            ).last_hidden_state[:, -1]
            @ weight.T
        )
        packed = PackedDecisionReadout(copy.deepcopy(reference), model.readout, reference_kernels(), reference_varlen)
        torch.testing.assert_close(packed(ids, present, bounds), expected, atol=1e-5, rtol=1e-5)
        actual = model(ids, present, bounds)
        single = model(ids[:1, 2:], present[:1, 2:], ((0, 2),))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[:1], single, atol=1e-5, rtol=1e-5)


def test_native_loader_refuses_missing_backbone_weights(tmp_path):
    model = tiny()
    model.save_pretrained(tmp_path)
    loaded = _strict_load(qwen.Qwen3_5TextModel, tmp_path, "cpu", torch.float32)
    torch.testing.assert_close(loaded.embed_tokens.weight, model.embed_tokens.weight)
    weights = load_file(tmp_path / "model.safetensors")
    weights.pop(next(key for key in weights if "q_proj" in key))
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="exactly match"):
        _strict_load(qwen.Qwen3_5TextModel, tmp_path, "cpu", torch.float32)


def test_d3_contract_accepts_header_prefix_and_plain_noncausal_checkpoints():
    shared = {
        "format_version": 1,
        "format_id": "d3-code-readout-v1",
        "prompt": "d3",
        "pooling": "last",
        "readout_dtype": "float32",
    }
    assert d3_contract({**shared, "attention_mode": "prefix_bidir", "prompt_layout": "header-v1"}) == (
        "prefix_bidir",
        "header-v1",
    )
    assert d3_contract({**shared, "attention_mode": "noncausal_full_attention", "temperature": 1.0}) == (
        "noncausal_full_attention",
        "plain",
    )
    with pytest.raises(ValueError, match="Unsupported d3 checkpoint contract"):
        d3_contract({**shared, "attention_mode": "noncausal_full_attention", "prompt_layout": "header-v1"})
    with pytest.raises(ValueError, match="Unsupported d3 checkpoint contract"):
        d3_contract({**shared, "attention_mode": "prefix_bidir"})
    with pytest.raises(ValueError, match="Unsupported d3 checkpoint contract"):
        d3_contract({**shared, "attention_mode": "causal", "prompt_layout": "header-v1"})


def test_noncausal_full_attention_matches_unmasked_prompt_tokens():
    from transformers.masking_utils import create_recurrent_attention_mask

    torch.manual_seed(41)
    reference = tiny()
    weight = torch.randn(255, 32)
    model = QwenReadout(copy.deepcopy(reference), weight, attention_mode="noncausal_full_attention")
    ids = torch.tensor([[0, 0, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]])
    present = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    bounds = ((2, 4), (0, 3))
    with torch.inference_mode():
        embeddings = reference.embed_tokens(ids)
        rows, width = present.shape
        ramp = torch.arange(width).view(1, 1, -1).expand(4, rows, -1)
        text_positions, mrope = ramp[0], ramp[1:]
        rope = reference.rotary_emb(embeddings, mrope)
        keys = present.bool()[:, None, None, :]
        scan = create_recurrent_attention_mask(
            config=reference.config,
            inputs_embeds=embeddings,
            attention_mask=present,
            past_key_values=None,
            position_ids=text_positions,
        )
        hidden = embeddings
        depth = reference.config.num_hidden_layers
        for layer, kind in zip(reference.layers[:depth], reference.config.layer_types[:depth], strict=True):
            output = layer(
                hidden,
                position_embeddings=rope,
                attention_mask=keys if kind == "full_attention" else scan,
                position_ids=text_positions,
                past_key_values=None,
                use_cache=False,
            )
            hidden = output[0] if isinstance(output, tuple) else output
        expected = reference.norm(hidden)[:, -1] @ weight.T
        actual = model(ids, present, bounds)
        single = model(ids[:1, 2:], present[:1, 2:], ((0, 2),))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[:1], single, atol=1e-5, rtol=1e-5)
