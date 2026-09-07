from __future__ import annotations

from pathlib import Path

import pytest

from tools.mise_tasks import model_test_selection

ROOT = Path(__file__).resolve().parents[3]
MODEL_TEST = ROOT / "packages/sie_server/tests/test_all_models.py"


def select(model_id: str) -> list[str]:
    return model_test_selection.select_node_ids(model_id, MODEL_TEST.read_text(), str(MODEL_TEST))


def test_base_bge_m3_selects_only_exact_base_tests():
    selected = select("BAAI/bge-m3")
    assert {node_id.rsplit("::", 1)[1] for node_id in selected} == {
        "test_baai_bge_m3_dense",
        "test_baai_bge_m3_sparse",
        "test_baai_bge_m3_multivector",
    }


def test_bge_m3_profile_selects_only_that_exact_profile():
    selected = select("BAAI/bge-m3:dense")
    assert selected
    assert all("bge_m3_dense_profile" in node_id for node_id in selected)
    assert not set(selected) & set(select("BAAI/bge-m3"))


def test_qwen_reranker_selects_every_exact_handwritten_test():
    selected = select("Qwen/Qwen3-Reranker-0.6B")
    assert {node_id.rsplit("::", 1)[1] for node_id in selected} == {
        "test_qwen_qwen3_reranker_0_6b_score",
        "test_qwen_qwen3_reranker_0_6b_score_in_range",
        "test_qwen_qwen3_reranker_0_6b_score_with_instruction",
    }


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B", "made-up/does-not-exist", "baai/bge-m3"])
def test_uncovered_unknown_and_wrong_case_ids_select_nothing(model_id):
    assert select(model_id) == []


def test_ambiguous_test_mapping_is_rejected():
    source = """
def test_ambiguous():
    _assert_dense("one/model")
    _assert_sparse("other/model")
"""
    with pytest.raises(ValueError, match="maps to multiple model IDs"):
        model_test_selection.select_node_ids("one/model", source, "synthetic.py")
