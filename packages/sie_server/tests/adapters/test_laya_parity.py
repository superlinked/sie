"""Laya adapter parity with the reference implementation, on real weights.

Downloads each checkpoint at the revision pinned in its model config and
compares against the reference fixtures in ``fixtures/laya/`` (the ``laya``
0.3.11 package, CPU float32; regenerate them with
``packages/sie_server/scripts/generate_laya_fixtures.py`` when a revision changes):

* real-tokenizer rows: exact ``input_ids`` and marker positions;
* CPU float32: raw decision-head logits within 1e-3 (same math as the reference);
* CUDA (bfloat16, flash-attention backbone when available): argmax agreement
  >= 98%, mean |dp| <= 0.02, max |dp| <= 0.08 against the float32 reference.
  The reference's own bfloat16-autocast forward differs from its float32
  forward by up to ~0.06 on these inputs, so exact equality is not expected.

Run with ``mise run test -- -m model packages/sie_server/tests/adapters/test_laya_parity.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sie_server.adapters.laya.questions import calibrated_probs, parse_questions
from sie_server.core.loader import load_adapter, load_model_configs
from sie_server.types.inputs import Item

pytestmark = pytest.mark.model

FIXTURES = Path(__file__).parent / "fixtures" / "laya"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
VARIANTS = ("laya", "laya-multilingual", "laya-typed-decisions")

_adapters: dict[tuple[str, str], Any] = {}


def _golden(variant: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{variant}.json").read_text(encoding="utf-8"))


def _adapter(variant: str, device: str) -> Any:
    key = (variant, device)
    if key not in _adapters:
        configs = load_model_configs(MODELS_DIR)
        config = configs[f"convaiinnovations/{variant}"]
        adapter = load_adapter(config, MODELS_DIR, device=device)
        adapter.load(device)
        _adapters[key] = adapter
    return _adapters[key]


def _item(state: Any) -> Item:
    return Item(text=state) if isinstance(state, str) else Item(metadata={"state": state})


def _logits(adapter: Any, golden: dict[str, Any]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Adapter logits for every golden row, with the matching golden rows."""
    questions = parse_questions(golden["questions"])
    states = [adapter._item_state(_item(case["state"])) for case in golden["cases"]]
    prefixes, rows = adapter._build_rows(
        states, questions, max_len=golden["max_len"], head_max_len=golden["head_max_len"]
    )
    logits = adapter._run_rows(rows, [prefixes[r % len(prefixes)] for r in range(len(rows))])
    expected = [row for case in golden["cases"] for row in case["rows"]]
    for row, ref in zip(rows, expected, strict=True):
        assert hashlib.sha256(json.dumps(row).encode()).hexdigest() == ref["sha256"], ref["qid"]
    return logits, expected


def _check_config_pins_fixture(variant: str, golden: dict[str, Any]) -> None:
    config = load_model_configs(MODELS_DIR)[f"convaiinnovations/{variant}"]
    assert config.hf_revision == golden["revision"]
    assert config.max_sequence_length == golden["max_len"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_real_tokenizer_rows_match_reference(variant: str) -> None:
    """Tokenizer loading (no cache rewrite) plus row assembly reproduce the reference ids exactly."""
    from huggingface_hub import snapshot_download
    from sie_server.adapters.laya.adapter import LayaAdapter

    golden = _golden(variant)
    _check_config_pins_fixture(variant, golden)
    local = Path(snapshot_download(golden["repo"], revision=golden["revision"], allow_patterns=["tokenizer/*"]))
    adapter = LayaAdapter(golden["repo"], revision=golden["revision"])
    adapter._tokenizer = LayaAdapter._load_tokenizer(local / "tokenizer")
    for section in (golden, golden["override"]):
        questions = parse_questions(section["questions"])
        states = [adapter._item_state(_item(case["state"])) for case in section["cases"]]
        prefixes, rows = adapter._build_rows(
            states, questions, max_len=section["max_len"], head_max_len=section["head_max_len"]
        )
        expected = [row for case in section["cases"] for row in case["rows"]]
        for n, (row, ref) in enumerate(zip(rows, expected, strict=True)):
            assert hashlib.sha256(json.dumps(row).encode()).hexdigest() == ref["sha256"], ref["qid"]
            assert prefixes[n % len(prefixes)].markers == ref["markers"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_cpu_float32_logits_match_reference(variant: str) -> None:
    golden = _golden(variant)
    _check_config_pins_fixture(variant, golden)
    adapter = _adapter(variant, "cpu")
    logits, expected = _logits(adapter, golden)
    for mine, ref in zip(logits, expected, strict=True):
        np.testing.assert_allclose(mine, np.array(ref["logits"], dtype=np.float32), atol=1e-3, err_msg=ref["qid"])

    output = adapter.extract([_item(case["state"]) for case in golden["cases"]], output_schema=golden["questions"])
    assert output.input_token_counts == [case["input_tokens"] for case in golden["cases"]]


@pytest.mark.gpu_hw
@pytest.mark.parametrize("variant", VARIANTS)
def test_cuda_bfloat16_matches_reference(variant: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from sie_server.core.inference import is_flash_attention_available

    golden = _golden(variant)
    adapter = _adapter(variant, "cuda:0")
    assert adapter._model.use_flash == is_flash_attention_available("cuda:0")
    logits, expected = _logits(adapter, golden)

    calibration = adapter._calibration
    agree, deltas = [], []
    for mine, ref in zip(logits, expected, strict=True):
        qtype = next(t for t, i in (("choice", 0), ("score", 1), ("noul", 2)) if i == ref["qtype"])
        p_ref = calibrated_probs(np.array(ref["logits"], dtype=np.float32), qtype, calibration)
        p_mine = calibrated_probs(mine, qtype, calibration)
        agree.append(int(p_ref.argmax()) == int(p_mine.argmax()))
        deltas.extend(np.abs(p_ref - p_mine).tolist())
    assert np.mean(agree) >= 0.98
    assert np.mean(deltas) <= 0.02
    assert np.max(deltas) <= 0.08
