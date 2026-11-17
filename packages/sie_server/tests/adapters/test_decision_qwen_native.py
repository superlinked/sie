"""Native mixed-recurrence Qwen parity against an independent dense attention mask."""

import copy

import pytest
import torch
from safetensors.torch import load_file, save_file
from sie_server.adapters._qwen3_5_packed import reference_kernels
from sie_server.adapters.decision.packed import PackedDecisionReadout
from sie_server.adapters.decision.qwen import QwenReadout, _strict_load

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


def full_hidden_states(model, ids, present, bounds):
    """Every position's final hidden state on the default block-attention path."""
    from transformers.masking_utils import create_recurrent_attention_mask

    embeddings = model.backbone.embed_tokens(ids)
    scan = create_recurrent_attention_mask(
        config=model.backbone.config, inputs_embeds=embeddings, attention_mask=present
    )
    return model.backbone(
        inputs_embeds=embeddings,
        position_ids=(present.long().cumsum(-1) - 1).clamp_min(0),
        attention_mask={"full_attention": bounds, "linear_attention": scan},
        use_cache=False,
    ).last_hidden_state


def moved(before, after, position):
    return (before[0, position] - after[0, position]).abs().max().item() > 1e-4


def test_noncausal_full_attention_lets_earlier_positions_read_later_tokens():
    torch.manual_seed(41)
    reference = tiny()
    weight = torch.randn(255, 32)
    noncausal = QwenReadout(copy.deepcopy(reference), weight, attention_mode="noncausal_full_attention")
    block = QwenReadout(copy.deepcopy(reference), weight)
    ids = torch.tensor([[0, 0, 2, 3, 4, 5]])
    present = torch.tensor([[0, 0, 1, 1, 1, 1]])
    changed = ids.clone()
    changed[0, -1] = 9
    bounds = ((2, 4),)
    with torch.inference_mode():
        before = noncausal._noncausal_hidden(ids, present)
        after = noncausal._noncausal_hidden(changed, present)
        causal_before = reference(input_ids=ids[:, 2:], use_cache=False).last_hidden_state
        causal_after = reference(input_ids=changed[:, 2:], use_cache=False).last_hidden_state
        block_before = full_hidden_states(block, ids, present, bounds)
        block_after = full_hidden_states(block, changed, present, bounds)
    # Noncausal: every earlier real token sees the perturbed last token through full attention.
    assert all(moved(before, after, position) for position in (2, 3, 4, 5))
    # Same weights under the stock causal mask: earlier positions cannot see it (and the recurrent
    # layers do not leak it backwards), only the perturbed position moves.
    torch.testing.assert_close(causal_before[:, :-1], causal_after[:, :-1], atol=1e-6, rtol=1e-6)
    assert moved(causal_before, causal_after, 3)
    # Default prefix_bidir path: the prefix block (positions 2-3) is isolated from the suffix.
    torch.testing.assert_close(block_before[:, 2:4], block_after[:, 2:4], atol=1e-6, rtol=1e-6)
    assert moved(block_before, block_after, 4)


def test_noncausal_full_attention_ignores_left_padding():
    torch.manual_seed(41)
    weight = torch.randn(255, 32)
    model = QwenReadout(tiny(), weight, attention_mode="noncausal_full_attention")
    ids = torch.tensor([[0, 0, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]])
    present = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    bounds = ((2, 4), (0, 3))
    other_pad = ids.clone()
    other_pad[0, :2] = 30
    with torch.inference_mode():
        batch = model(ids, present, bounds)
        repadded = model(other_pad, present, bounds)
        single = model(ids[:1, 2:], present[:1, 2:], ((0, 2),))
        neighbour = model(ids[1:], present[1:], ((0, 3),))
    torch.testing.assert_close(batch[:1], single, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batch[1:], neighbour, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(repadded, batch, atol=1e-6, rtol=1e-6)


def test_noncausal_readout_keeps_the_native_sdpa_backbone():
    backbone = tiny()
    QwenReadout(backbone, torch.randn(255, 32), attention_mode="noncausal_full_attention")
    assert backbone.config._attn_implementation == "sdpa"
    eager = tiny()
    eager.config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="requires the backbone's sdpa"):
        QwenReadout(eager, torch.randn(255, 32), attention_mode="noncausal_full_attention")
