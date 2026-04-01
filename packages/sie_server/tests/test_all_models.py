from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from sie_server.core.loader import load_adapter, load_model_configs
from sie_server.types.inputs import ImageInput, Item

pytestmark = pytest.mark.model

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
ALL_CONFIGS: dict[str, Any] | None = None

# Cache loaded adapters to avoid reloading per test
_adapter_cache: dict[str, Any] = {}


def _get_adapter(model_name: str) -> Any:
    global ALL_CONFIGS
    if ALL_CONFIGS is None:
        ALL_CONFIGS = load_model_configs(MODELS_DIR)

    if model_name not in _adapter_cache:
        config = ALL_CONFIGS[model_name]
        adapter = load_adapter(config, MODELS_DIR, device="cpu")
        adapter.load("cpu")
        _adapter_cache[model_name] = adapter
    return _adapter_cache[model_name]


def _test_image() -> ImageInput:
    # 16x16 red JPEG generated via Pillow
    img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return ImageInput(data=buf.getvalue(), format="jpeg")


def _to_f16(arr: np.ndarray) -> list[float]:
    return arr.astype(np.float16).tolist()


def _assert_dense(
    model_name: str,
    expected_dim: int,
    expected_first3: list[float] | None,
) -> None:
    adapter = _get_adapter(model_name)
    output = adapter.encode([Item(text="test")], output_types=["dense"])
    assert output.dense is not None, f"{model_name}: no dense output"
    assert output.dense.shape == (1, expected_dim)
    assert not np.isnan(output.dense[0]).any()
    actual = _to_f16(output.dense[0, :3])
    if expected_first3 is None:
        msg = f"FILL: {model_name} dense = {actual}"
        raise AssertionError(msg)
    np.testing.assert_array_equal(
        np.array(actual, dtype=np.float16),
        np.array(expected_first3, dtype=np.float16),
    )


def _assert_dense_image(
    model_name: str,
    expected_dim: int,
    expected_first3: list[float] | None,
) -> None:
    adapter = _get_adapter(model_name)
    output = adapter.encode(
        [Item(images=[_test_image()])],
        output_types=["dense"],
    )
    assert output.dense is not None, f"{model_name}: no dense output for image"
    assert output.dense.shape == (1, expected_dim)
    assert not np.isnan(output.dense[0]).any()
    actual = _to_f16(output.dense[0, :3])
    if expected_first3 is None:
        msg = f"FILL: {model_name} image_dense = {actual}"
        raise AssertionError(msg)
    np.testing.assert_array_equal(
        np.array(actual, dtype=np.float16),
        np.array(expected_first3, dtype=np.float16),
    )


def _assert_sparse(
    model_name: str,
    expected_indices3: list[int] | None,
    expected_values3: list[float] | None,
) -> None:
    adapter = _get_adapter(model_name)
    output = adapter.encode([Item(text="test")], output_types=["sparse"])
    assert output.sparse is not None, f"{model_name}: no sparse output"
    assert len(output.sparse) == 1
    sv = output.sparse[0]
    actual_idx = sv.indices[:3].tolist()
    actual_val = _to_f16(sv.values[:3])
    if expected_indices3 is None or expected_values3 is None:
        msg = f"FILL: {model_name} sparse_indices = {actual_idx}, sparse_values = {actual_val}"
        raise AssertionError(msg)
    assert actual_idx == expected_indices3
    np.testing.assert_array_equal(
        np.array(actual_val, dtype=np.float16),
        np.array(expected_values3, dtype=np.float16),
    )


def _assert_multivector(
    model_name: str,
    expected_token_dim: int,
    expected_first3: list[float] | None,
) -> None:
    adapter = _get_adapter(model_name)
    output = adapter.encode([Item(text="test")], output_types=["multivector"])
    assert output.multivector is not None, f"{model_name}: no multivector output"
    assert len(output.multivector) == 1
    mv = output.multivector[0]
    assert mv.shape[1] == expected_token_dim
    actual = _to_f16(mv[0, :3])
    if expected_first3 is None:
        msg = f"FILL: {model_name} multivector = {actual}"
        raise AssertionError(msg)
    np.testing.assert_array_equal(
        np.array(actual, dtype=np.float16),
        np.array(expected_first3, dtype=np.float16),
    )


def _assert_multivector_image(
    model_name: str,
    expected_token_dim: int,
    expected_first3: list[float] | None,
) -> None:
    adapter = _get_adapter(model_name)
    output = adapter.encode(
        [Item(images=[_test_image()])],
        output_types=["multivector"],
    )
    assert output.multivector is not None
    assert len(output.multivector) == 1
    mv = output.multivector[0]
    assert mv.shape[1] == expected_token_dim
    actual = _to_f16(mv[0, :3])
    if expected_first3 is None:
        msg = f"FILL: {model_name} image_multivector = {actual}"
        raise AssertionError(msg)
    np.testing.assert_array_equal(
        np.array(actual, dtype=np.float16),
        np.array(expected_first3, dtype=np.float16),
    )


def _assert_score(
    model_name: str,
    expected_score: list[float] | None,
) -> None:
    adapter = _get_adapter(model_name)
    output = adapter.score_pairs(
        queries=[Item(text="what is artificial intelligence")],
        docs=[Item(text="Artificial intelligence is the simulation of human intelligence by machines")],
    )
    actual = _to_f16(output.scores[:1])
    if expected_score is None:
        msg = f"FILL: {model_name} score = {actual}"
        raise AssertionError(msg)
    np.testing.assert_array_equal(
        np.array(actual, dtype=np.float16),
        np.array(expected_score, dtype=np.float16),
    )


def _assert_extract(
    model_name: str,
    labels: list[str],
    expected_entity_labels: list[str] | None,
) -> None:
    adapter = _get_adapter(model_name)
    output = adapter.extract(
        [Item(text="John Smith works at Google in New York City")],
        labels=labels,
    )
    assert len(output.entities) == 1
    actual_labels = sorted({e["label"] for e in output.entities[0]})
    if expected_entity_labels is None:
        msg = f"FILL: {model_name} extract_labels = {actual_labels}"
        raise AssertionError(msg)
    assert actual_labels == sorted(expected_entity_labels)


# =============================================================================
# Dense-only text models
# =============================================================================


def test_alibaba_nlp_gte_multilingual_base_dense() -> None:
    _assert_dense("Alibaba-NLP/gte-multilingual-base", 768, [-0.055389404296875, 0.06341552734375, -0.029815673828125])


@pytest.mark.xfail(reason="1.5B model too slow for CPU unit tests", strict=False)
def test_alibaba_nlp_gte_qwen2_1_5b_instruct_dense() -> None:
    _assert_dense("Alibaba-NLP/gte-Qwen2-1.5B-instruct", 1536, None)


@pytest.mark.xfail(reason="7B model too large for CPU unit tests", strict=False)
def test_alibaba_nlp_gte_qwen2_7b_instruct_dense() -> None:
    _assert_dense("Alibaba-NLP/gte-Qwen2-7B-instruct", 3584, None)


def test_google_embeddinggemma_300m_dense() -> None:
    _assert_dense("google/embeddinggemma-300m", 768, [0.01641845703125, 0.052001953125, -0.0009312629699707031])


@pytest.mark.xfail(reason="7B model too large for CPU unit tests", strict=False)
def test_gritlm_gritlm_7b_dense() -> None:
    _assert_dense("GritLM/GritLM-7B", 4096, None)


def test_intfloat_e5_base_v2_dense() -> None:
    _assert_dense("intfloat/e5-base-v2", 768, [-0.00885772705078125, -0.03472900390625, -0.0255279541015625])


def test_intfloat_e5_large_v2_dense() -> None:
    _assert_dense("intfloat/e5-large-v2", 1024, [0.018524169921875, -0.0706787109375, 0.0183563232421875])


def test_intfloat_e5_small_v2_dense() -> None:
    _assert_dense("intfloat/e5-small-v2", 384, [-0.074462890625, 0.041748046875, 0.0362548828125])


@pytest.mark.xfail(reason="7B model too large for CPU unit tests", strict=False)
def test_intfloat_e5_mistral_7b_instruct_dense() -> None:
    _assert_dense("intfloat/e5-mistral-7b-instruct", 4096, None)


def test_intfloat_multilingual_e5_large_dense() -> None:
    _assert_dense("intfloat/multilingual-e5-large", 1024, [0.0318603515625, 0.022491455078125, -0.01427459716796875])


def test_intfloat_multilingual_e5_large_instruct_dense() -> None:
    _assert_dense(
        "intfloat/multilingual-e5-large-instruct",
        1024,
        [0.0282745361328125, 0.0178070068359375, -0.0012836456298828125],
    )


def test_intfloat_multilingual_e5_large_sentence_transformer_dense() -> None:
    _assert_dense(
        "intfloat/multilingual-e5-large:sentence_transformer",
        1024,
        [0.0318603515625, 0.022491455078125, -0.01427459716796875],
    )


@pytest.mark.xfail(reason="7B model too large for CPU unit tests", strict=False)
def test_linq_embed_mistral_dense() -> None:
    _assert_dense("Linq-AI-Research/Linq-Embed-Mistral", 4096, None)


def test_nomic_ai_nomic_embed_text_v2_moe_dense() -> None:
    _assert_dense(
        "nomic-ai/nomic-embed-text-v2-moe", 768, [0.01473236083984375, -0.01526641845703125, 0.0231475830078125]
    )


@pytest.mark.xfail(reason="1.5B model too slow for CPU unit tests", strict=False)
def test_novasearch_stella_en_1_5b_v5_dense() -> None:
    _assert_dense("NovaSearch/stella_en_1.5B_v5", 1024, None)


def test_novasearch_stella_en_400m_v5_dense() -> None:
    _assert_dense("NovaSearch/stella_en_400M_v5", 1024, [0.0458984375, 0.01031494140625, -0.09454345703125])


@pytest.mark.xfail(reason="8B model too large for CPU unit tests", strict=False)
def test_nvidia_llama_embed_nemotron_8b_dense() -> None:
    _assert_dense("nvidia/llama-embed-nemotron-8b", 4096, None)


@pytest.mark.xfail(reason="7B model too large for CPU unit tests", strict=False)
def test_nvidia_nv_embed_v2_dense() -> None:
    _assert_dense("nvidia/NV-Embed-v2", 4096, None)


def test_qwen_qwen3_embedding_0_6b_dense() -> None:
    _assert_dense("Qwen/Qwen3-Embedding-0.6B", 1024, [-0.0165252685546875, -0.04132080078125, -0.0137786865234375])


def test_qwen_qwen3_embedding_0_6b_custom_dense() -> None:
    _assert_dense(
        "Qwen/Qwen3-Embedding-0.6B:custom", 1024, [-0.0165252685546875, -0.04132080078125, -0.0137786865234375]
    )


@pytest.mark.xfail(reason="4B model too large for CPU unit tests", strict=False)
def test_qwen_qwen3_embedding_4b_dense() -> None:
    _assert_dense("Qwen/Qwen3-Embedding-4B", 2560, None)


@pytest.mark.xfail(reason="7B model too large for CPU unit tests", strict=False)
def test_salesforce_sfr_embedding_2_r_dense() -> None:
    _assert_dense("Salesforce/SFR-Embedding-2_R", 4096, None)


@pytest.mark.xfail(reason="7B model too large for CPU unit tests", strict=False)
def test_salesforce_sfr_embedding_mistral_dense() -> None:
    _assert_dense("Salesforce/SFR-Embedding-Mistral", 4096, None)


def test_sentence_transformers_all_minilm_l6_v2_dense() -> None:
    _assert_dense(
        "sentence-transformers/all-MiniLM-L6-v2", 384, [0.0167694091796875, 0.035125732421875, -0.0259857177734375]
    )


# =============================================================================
# Dense models with image support
# =============================================================================


def test_google_siglip_so400m_patch14_224_dense() -> None:
    _assert_dense("google/siglip-so400m-patch14-224", 1152, [-0.00983428955078125, -0.0234375, 0.013824462890625])


def test_google_siglip_so400m_patch14_224_image_dense() -> None:
    _assert_dense_image(
        "google/siglip-so400m-patch14-224", 1152, [0.0090789794921875, -0.006679534912109375, -0.0123138427734375]
    )


def test_google_siglip_so400m_patch14_384_dense() -> None:
    _assert_dense(
        "google/siglip-so400m-patch14-384", 1152, [-0.027801513671875, -0.01428985595703125, 0.004940032958984375]
    )


def test_google_siglip_so400m_patch14_384_image_dense() -> None:
    _assert_dense_image(
        "google/siglip-so400m-patch14-384", 1152, [0.00960540771484375, -0.0045623779296875, -0.009002685546875]
    )


def test_laion_clip_vit_b_32_laion2b_dense() -> None:
    _assert_dense(
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", 512, [-0.020416259765625, -0.0300140380859375, -0.0004265308380126953]
    )


def test_laion_clip_vit_b_32_laion2b_image_dense() -> None:
    _assert_dense_image("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", 512, [0.02642822265625, 0.06329345703125, -0.046875])


@pytest.mark.xfail(reason="Large ViT-H model too large for CPU unit tests", strict=False)
def test_laion_clip_vit_h_14_laion2b_dense() -> None:
    _assert_dense("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 1024, None)


@pytest.mark.xfail(reason="Large ViT-H model too large for CPU unit tests", strict=False)
def test_laion_clip_vit_h_14_laion2b_image_dense() -> None:
    _assert_dense_image("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 1024, None)


def test_openai_clip_vit_base_patch32_dense() -> None:
    _assert_dense("openai/clip-vit-base-patch32", 512, [-0.003787994384765625, 0.00225830078125, -0.007648468017578125])


def test_openai_clip_vit_base_patch32_image_dense() -> None:
    _assert_dense_image(
        "openai/clip-vit-base-patch32", 512, [-0.0030384063720703125, -0.01465606689453125, -0.0303497314453125]
    )


def test_openai_clip_vit_large_patch14_dense() -> None:
    _assert_dense("openai/clip-vit-large-patch14", 768, [0.009063720703125, 0.0189361572265625, 0.0005855560302734375])


def test_openai_clip_vit_large_patch14_image_dense() -> None:
    _assert_dense_image(
        "openai/clip-vit-large-patch14", 768, [0.009979248046875, 0.0163726806640625, 0.0226898193359375]
    )


# =============================================================================
# BAAI/bge-m3: dense + sparse + multivector
# =============================================================================


def test_baai_bge_m3_dense() -> None:
    _assert_dense("BAAI/bge-m3", 1024, [0.001247406005859375, 0.0228271484375, -0.02349853515625])


def test_baai_bge_m3_sparse() -> None:
    _assert_sparse("BAAI/bge-m3", [3034], [0.342041015625])


def test_baai_bge_m3_multivector() -> None:
    _assert_multivector("BAAI/bge-m3", 1024, [0.01641845703125, 0.0304718017578125, -0.036468505859375])


def test_baai_bge_m3_banking_dense() -> None:
    _assert_dense("BAAI/bge-m3:banking", 1024, [0.001247406005859375, 0.0228271484375, -0.02349853515625])


def test_baai_bge_m3_banking_sparse() -> None:
    _assert_sparse("BAAI/bge-m3:banking", [3034], [0.342041015625])


def test_baai_bge_m3_banking_multivector() -> None:
    _assert_multivector("BAAI/bge-m3:banking", 1024, [0.01641845703125, 0.0304718017578125, -0.036468505859375])


def test_baai_bge_m3_bge_m3_flag_dense() -> None:
    _assert_dense("BAAI/bge-m3:bge_m3_flag", 1024, [0.001247406005859375, 0.0228271484375, -0.02349853515625])


def test_baai_bge_m3_bge_m3_flag_sparse() -> None:
    _assert_sparse("BAAI/bge-m3:bge_m3_flag", [3034], [0.342041015625])


def test_baai_bge_m3_bge_m3_flag_multivector() -> None:
    _assert_multivector("BAAI/bge-m3:bge_m3_flag", 1024, [0.01641845703125, 0.0304718017578125, -0.036468505859375])


def test_baai_bge_m3_dense_profile_dense() -> None:
    _assert_dense("BAAI/bge-m3:dense", 1024, [0.001247406005859375, 0.0228271484375, -0.02349853515625])


def test_baai_bge_m3_dense_profile_sparse() -> None:
    _assert_sparse("BAAI/bge-m3:dense", [3034], [0.342041015625])


def test_baai_bge_m3_dense_profile_multivector() -> None:
    _assert_multivector("BAAI/bge-m3:dense", 1024, [0.01641845703125, 0.0304718017578125, -0.036468505859375])


def test_baai_bge_m3_medical_vn_dense() -> None:
    _assert_dense("BAAI/bge-m3:medical-vn", 1024, [0.001247406005859375, 0.0228271484375, -0.02349853515625])


def test_baai_bge_m3_medical_vn_sparse() -> None:
    _assert_sparse("BAAI/bge-m3:medical-vn", [3034], [0.342041015625])


def test_baai_bge_m3_medical_vn_multivector() -> None:
    _assert_multivector("BAAI/bge-m3:medical-vn", 1024, [0.01641845703125, 0.0304718017578125, -0.036468505859375])


def test_baai_bge_m3_multivector_profile_dense() -> None:
    _assert_dense("BAAI/bge-m3:multivector", 1024, [0.001247406005859375, 0.0228271484375, -0.02349853515625])


def test_baai_bge_m3_multivector_profile_sparse() -> None:
    _assert_sparse("BAAI/bge-m3:multivector", [3034], [0.342041015625])


def test_baai_bge_m3_multivector_profile_multivector() -> None:
    _assert_multivector("BAAI/bge-m3:multivector", 1024, [0.01641845703125, 0.0304718017578125, -0.036468505859375])


def test_baai_bge_m3_sparse_profile_dense() -> None:
    _assert_dense("BAAI/bge-m3:sparse", 1024, [0.001247406005859375, 0.0228271484375, -0.02349853515625])


def test_baai_bge_m3_sparse_profile_sparse() -> None:
    _assert_sparse("BAAI/bge-m3:sparse", [3034], [0.342041015625])


def test_baai_bge_m3_sparse_profile_multivector() -> None:
    _assert_multivector("BAAI/bge-m3:sparse", 1024, [0.01641845703125, 0.0304718017578125, -0.036468505859375])


# =============================================================================
# ibm-granite sparse
# =============================================================================


def test_ibm_granite_embedding_30m_sparse() -> None:
    _assert_sparse(
        "ibm-granite/granite-embedding-30m-sparse", [4, 114, 1296], [0.33837890625, 0.53759765625, 1.3681640625]
    )


# =============================================================================
# Sparse-only models
# =============================================================================


def test_naver_splade_cocondenser_selfdistil_sparse() -> None:
    _assert_sparse(
        "naver/splade-cocondenser-selfdistil", [2054, 2470, 2726], [0.061248779296875, 0.1495361328125, 0.147705078125]
    )


def test_naver_splade_v3_sparse() -> None:
    _assert_sparse("naver/splade-v3", [1000, 1037, 1056], [0.313232421875, 0.31005859375, 0.277587890625])


def test_opensearch_neural_sparse_encoding_doc_v2_distill_sparse() -> None:
    _assert_sparse(
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
        [1001, 1007, 1008],
        [0.02764892578125, 0.2174072265625, 0.08843994140625],
    )


def test_opensearch_neural_sparse_encoding_doc_v2_mini_sparse() -> None:
    _assert_sparse(
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
        [141, 196, 388],
        [0.0018644332885742188, 0.006130218505859375, 0.0025119781494140625],
    )


def test_opensearch_neural_sparse_encoding_doc_v3_distill_sparse() -> None:
    _assert_sparse(
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
        [1005, 1007, 1008],
        [0.00753021240234375, 0.00579071044921875, 0.034271240234375],
    )


def test_opensearch_neural_sparse_encoding_doc_v3_gte_sparse() -> None:
    _assert_sparse(
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
        [1011, 1024, 1029],
        [0.078125, 0.0782470703125, 0.138916015625],
    )


def test_opensearch_neural_sparse_encoding_v1_sparse() -> None:
    _assert_sparse(
        "opensearch-project/opensearch-neural-sparse-encoding-v1",
        [2470, 2671, 2726],
        [0.132080078125, 0.08245849609375, 0.07867431640625],
    )


def test_opensearch_neural_sparse_encoding_v2_distill_sparse() -> None:
    _assert_sparse(
        "opensearch-project/opensearch-neural-sparse-encoding-v2-distill",
        [1078, 1602, 2054],
        [0.06439208984375, 0.6044921875, 0.4658203125],
    )


def test_prithivida_splade_pp_en_v2_sparse() -> None:
    _assert_sparse(
        "prithivida/Splade_PP_en_v2", [2668, 2671, 2674], [0.1822509765625, 0.1221923828125, 0.061126708984375]
    )


def test_rasyosef_splade_mini_sparse() -> None:
    _assert_sparse("rasyosef/splade-mini", [1037, 1996, 2773], [1.0810546875, 0.380615234375, 0.59521484375])


# =============================================================================
# Multivector-only text models
# =============================================================================


def test_answerdotai_answerai_colbert_small_v1_multivector() -> None:
    _assert_multivector(
        "answerdotai/answerai-colbert-small-v1", 96, [-0.08148193359375, 0.05230712890625, -0.04718017578125]
    )


def test_colbert_ir_colbertv2_0_multivector() -> None:
    _assert_multivector("colbert-ir/colbertv2.0", 128, [0.03387451171875, 0.02313232421875, 0.02557373046875])


def test_jinaai_jina_colbert_v2_multivector() -> None:
    _assert_multivector("jinaai/jina-colbert-v2", 128, [0.0228271484375, 0.0301513671875, -0.2294921875])


def test_lightonai_gte_moderncolbert_v1_multivector() -> None:
    _assert_multivector(
        "lightonai/GTE-ModernColBERT-v1", 128, [0.0014638900756835938, -0.07513427734375, -0.005741119384765625]
    )


def test_lightonai_reason_moderncolbert_multivector() -> None:
    _assert_multivector(
        "lightonai/Reason-ModernColBERT", 128, [-0.0040740966796875, -0.08502197265625, 0.0003883838653564453]
    )


def test_mixedbread_ai_mxbai_colbert_large_v1_multivector() -> None:
    _assert_multivector(
        "mixedbread-ai/mxbai-colbert-large-v1", 128, [-0.084228515625, 0.00972747802734375, 0.056182861328125]
    )


def test_mixedbread_ai_mxbai_edge_colbert_v0_32m_multivector() -> None:
    _assert_multivector(
        "mixedbread-ai/mxbai-edge-colbert-v0-32m", 64, [0.199462890625, -0.08306884765625, 0.05841064453125]
    )


# =============================================================================
# Multivector models with image support
# =============================================================================


@pytest.mark.xfail(reason="3B model too large for CPU unit tests", strict=False)
def test_nvidia_llama_nemoretriever_colembed_3b_v1_multivector() -> None:
    _assert_multivector("nvidia/llama-nemoretriever-colembed-3b-v1", 128, None)


@pytest.mark.xfail(reason="3B model too large for CPU unit tests", strict=False)
def test_nvidia_llama_nemoretriever_colembed_3b_v1_image_multivector() -> None:
    _assert_multivector_image("nvidia/llama-nemoretriever-colembed-3b-v1", 128, None)


@pytest.mark.xfail(reason="3B model too large for CPU unit tests", strict=False)
def test_vidore_colpali_v1_3_hf_multivector() -> None:
    _assert_multivector("vidore/colpali-v1.3-hf", 128, None)


@pytest.mark.xfail(reason="3B model too large for CPU unit tests", strict=False)
def test_vidore_colpali_v1_3_hf_image_multivector() -> None:
    _assert_multivector_image("vidore/colpali-v1.3-hf", 128, None)


@pytest.mark.xfail(reason="2B+ model too large for CPU unit tests", strict=False)
def test_vidore_colqwen2_5_v0_2_multivector() -> None:
    _assert_multivector("vidore/colqwen2.5-v0.2", 128, None)


@pytest.mark.xfail(reason="2B+ model too large for CPU unit tests", strict=False)
def test_vidore_colqwen2_5_v0_2_image_multivector() -> None:
    _assert_multivector_image("vidore/colqwen2.5-v0.2", 128, None)


# =============================================================================
# Multivector models with muvera profiles (dense via postprocessor)
# =============================================================================


def test_answerdotai_answerai_colbert_small_v1_muvera_multivector() -> None:
    _assert_multivector(
        "answerdotai/answerai-colbert-small-v1:muvera", 96, [-0.08148193359375, 0.05230712890625, -0.04718017578125]
    )


def test_colbert_ir_colbertv2_0_muvera_multivector() -> None:
    _assert_multivector("colbert-ir/colbertv2.0:muvera", 128, [0.03387451171875, 0.02313232421875, 0.02557373046875])


def test_jinaai_jina_colbert_v2_muvera_multivector() -> None:
    _assert_multivector("jinaai/jina-colbert-v2:muvera", 128, [0.0228271484375, 0.0301513671875, -0.2294921875])


def test_lightonai_gte_moderncolbert_v1_muvera_multivector() -> None:
    _assert_multivector(
        "lightonai/GTE-ModernColBERT-v1:muvera", 128, [0.0014638900756835938, -0.07513427734375, -0.005741119384765625]
    )


def test_lightonai_reason_moderncolbert_muvera_multivector() -> None:
    _assert_multivector(
        "lightonai/Reason-ModernColBERT:muvera", 128, [-0.0040740966796875, -0.08502197265625, 0.0003883838653564453]
    )


def test_mixedbread_ai_mxbai_colbert_large_v1_muvera_multivector() -> None:
    _assert_multivector(
        "mixedbread-ai/mxbai-colbert-large-v1:muvera", 128, [-0.084228515625, 0.00972747802734375, 0.056182861328125]
    )


def test_mixedbread_ai_mxbai_edge_colbert_v0_32m_muvera_multivector() -> None:
    _assert_multivector(
        "mixedbread-ai/mxbai-edge-colbert-v0-32m:muvera", 64, [0.199462890625, -0.08306884765625, 0.05841064453125]
    )


@pytest.mark.xfail(reason="3B model too large for CPU unit tests", strict=False)
def test_nvidia_llama_nemoretriever_colembed_3b_v1_muvera_multivector() -> None:
    _assert_multivector("nvidia/llama-nemoretriever-colembed-3b-v1:muvera", 128, None)


@pytest.mark.xfail(reason="3B model too large for CPU unit tests", strict=False)
def test_vidore_colpali_v1_3_hf_muvera_multivector() -> None:
    _assert_multivector("vidore/colpali-v1.3-hf:muvera", 128, None)


@pytest.mark.xfail(reason="2B+ model too large for CPU unit tests", strict=False)
def test_vidore_colqwen2_5_v0_2_muvera_multivector() -> None:
    _assert_multivector("vidore/colqwen2.5-v0.2:muvera", 128, None)


# =============================================================================
# Score (reranker) models
# =============================================================================


def test_alibaba_nlp_gte_reranker_modernbert_base_score() -> None:
    _assert_score("Alibaba-NLP/gte-reranker-modernbert-base", [0.97314453125])


def test_baai_bge_reranker_base_score() -> None:
    _assert_score("BAAI/bge-reranker-base", [1.0])


def test_baai_bge_reranker_large_score() -> None:
    _assert_score("BAAI/bge-reranker-large", [0.99951171875])


def test_baai_bge_reranker_v2_m3_score() -> None:
    _assert_score("BAAI/bge-reranker-v2-m3", [1.0])


@pytest.mark.xfail(
    reason="PyTorch 2.9 ARM CPU GEMM bug: non-contiguous transpose in nn.Linear produces NaN/Inf",
    strict=False,
)
def test_cross_encoder_ms_marco_minilm_l_6_v2_score() -> None:
    _assert_score("cross-encoder/ms-marco-MiniLM-L-6-v2", None)


def test_cross_encoder_ms_marco_minilm_l_12_v2_score() -> None:
    _assert_score("cross-encoder/ms-marco-MiniLM-L-12-v2", [10.8359375])


def test_jinaai_jina_reranker_v2_base_multilingual_score() -> None:
    _assert_score("jinaai/jina-reranker-v2-base-multilingual", [0.88671875])


@pytest.mark.xfail(
    reason="Non-deterministic: score.weight not in checkpoint, randomly initialized each run", strict=False
)
def test_mixedbread_ai_mxbai_rerank_base_v2_score() -> None:
    _assert_score("mixedbread-ai/mxbai-rerank-base-v2", None)


@pytest.mark.xfail(
    reason="Non-deterministic: score.weight not in checkpoint, randomly initialized each run", strict=False
)
def test_mixedbread_ai_mxbai_rerank_large_v2_score() -> None:
    _assert_score("mixedbread-ai/mxbai-rerank-large-v2", None)


# =============================================================================
# Extract models (text input - GLiNER / NLI classification)
# =============================================================================

_NER_LABELS = ["person", "organization", "location"]


def test_emergentmethods_gliner_large_news_v2_1_extract() -> None:
    _assert_extract("EmergentMethods/gliner_large_news-v2.1", _NER_LABELS, ["location", "organization", "person"])


def test_ihor_gliner_biomed_large_v1_0_extract() -> None:
    _assert_extract("Ihor/gliner-biomed-large-v1.0", _NER_LABELS, ["location", "organization", "person"])


def test_jackboyla_glirel_large_v0_extract() -> None:
    _assert_extract("jackboyla/glirel-large-v0", _NER_LABELS, [])


def test_knowledgator_gliclass_base_v1_0_extract() -> None:
    _assert_extract("knowledgator/gliclass-base-v1.0", ["technology", "sports", "politics"], [])


def test_knowledgator_gliclass_small_v1_0_extract() -> None:
    _assert_extract("knowledgator/gliclass-small-v1.0", ["technology", "sports", "politics"], [])


def test_moritzlaurer_deberta_v3_base_zeroshot_extract() -> None:
    _assert_extract("MoritzLaurer/deberta-v3-base-zeroshot-v2.0", ["technology", "sports", "politics"], [])


def test_moritzlaurer_deberta_v3_large_zeroshot_extract() -> None:
    _assert_extract("MoritzLaurer/deberta-v3-large-zeroshot-v2.0", ["technology", "sports", "politics"], [])


def test_neuml_gliner_bert_tiny_extract() -> None:
    _assert_extract("NeuML/gliner-bert-tiny", _NER_LABELS, ["person"])


def test_numind_nuner_zero_extract() -> None:
    _assert_extract("numind/NuNER_Zero", _NER_LABELS, ["location", "organization", "person"])


def test_numind_nuner_zero_span_extract() -> None:
    _assert_extract("numind/NuNER_Zero-span", _NER_LABELS, ["location", "organization", "person"])


def test_urchade_gliner_large_v2_1_extract() -> None:
    _assert_extract("urchade/gliner_large-v2.1", _NER_LABELS, ["location", "organization", "person"])


def test_urchade_gliner_medium_v2_1_extract() -> None:
    _assert_extract("urchade/gliner_medium-v2.1", _NER_LABELS, ["location", "organization", "person"])


def test_urchade_gliner_multi_pii_v1_extract() -> None:
    _assert_extract("urchade/gliner_multi_pii-v1", _NER_LABELS, ["location", "organization", "person"])


def test_urchade_gliner_multi_v2_1_extract() -> None:
    _assert_extract("urchade/gliner_multi-v2.1", _NER_LABELS, ["location", "organization", "person"])


def test_urchade_gliner_small_v2_1_extract() -> None:
    _assert_extract("urchade/gliner_small-v2.1", _NER_LABELS, ["location", "organization", "person"])


# =============================================================================
# Extract models (image input - Florence-2, Donut)
# These require image input, skipping for now as they need special handling
# =============================================================================


@pytest.mark.skip(reason="requires image input and special extract handling")
def test_microsoft_florence_2_base_extract() -> None:
    pass


@pytest.mark.skip(reason="requires image input and special extract handling")
def test_microsoft_florence_2_base_ft_extract() -> None:
    pass


@pytest.mark.skip(reason="requires image input and special extract handling")
def test_microsoft_florence_2_large_extract() -> None:
    pass


@pytest.mark.skip(reason="requires image input and special extract handling")
def test_mynkchaudhry_florence_2_ft_docvqa_extract() -> None:
    pass


@pytest.mark.skip(reason="requires image input and special extract handling")
def test_naver_clova_ix_donut_base_finetuned_cord_v2_extract() -> None:
    pass


@pytest.mark.skip(reason="requires image input and special extract handling")
def test_naver_clova_ix_donut_base_finetuned_docvqa_extract() -> None:
    pass


@pytest.mark.skip(reason="requires image input and special extract handling")
def test_naver_clova_ix_donut_base_finetuned_rvlcdip_extract() -> None:
    pass


# =============================================================================
# Extract models (image input - OWLv2 detection)
# =============================================================================


@pytest.mark.skip(reason="requires image input and detection-specific handling")
def test_google_owlv2_base_patch16_ensemble_extract() -> None:
    pass
