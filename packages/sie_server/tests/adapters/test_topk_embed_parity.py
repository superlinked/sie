r"""Parity between the TopK-Embed-V1 adapter and TopK's own pipeline.

The goldens under ``goldens/topk_embed`` come from the checkpoint's
sentence-transformers stack (``MultiVectorEncoder`` with its remote code), run by
``packages/sie_server/scripts/generate_topk_embed_goldens.py`` without any SIE code.
That stack packs a batch into one sequence and needs CUDA-only kernels
(flash-linear-attention, compiled flex attention); the CPU goldens swap only those
kernels for reference implementations with the same math (``generated_with``).

The adapter runs on stock ``transformers.models.qwen3_5`` classes, either one
right-padded row per input or one packed sequence per batch (both are tested). Token ids, token counts and scoring masks must match exactly. Vectors are
compared through their projections onto fixed random unit directions, and MaxSim
scores directly. The adapter's vision tower is TopK's, so page vectors differ only
through the text tower's kernels (sdpa and the unpacked delta rule vs flex attention
and the packed one). Measured on CPU (bf16) against the CPU goldens, over both
models, the largest differences were 0.008 (text) and 0.009 (pages) per projection
(0.0006 on average for pages) and 0.023 per score, with the same top document for
every query. The tolerances below leave about 2.5x that.

These tests need transformers >= 5.2 (the transformers5 bundle); run them with the
bundle's requirements:

    python -m sie_server.cli resolve-deps --bundle transformers5 > /tmp/t5.txt
    uv run --no-sync --with-requirements /tmp/t5.txt pytest -c pyproject.toml -m model \\
        packages/sie_server/tests/adapters/test_topk_embed_parity.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from sie_server.adapters.topk_embed.adapter import TopkEmbedAdapter
from sie_server.core.loader import load_adapter, load_model_configs
from sie_server.types.inputs import ImageInput, Item

pytestmark = pytest.mark.model

_GOLDENS = sorted((Path(__file__).parent / "goldens" / "topk_embed").glob("*.json"))
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_TEXT_PROJECTION_TOLERANCE = 0.02
_PAGE_PROJECTION_TOLERANCE = 0.025
_PAGE_PROJECTION_MEAN_TOLERANCE = 0.0015
_SCORE_TOLERANCE = 0.05


def _golden(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(
    scope="module",
    params=[(path, packed) for path in _GOLDENS for packed in (False, True)],
    ids=[f"{path.stem}-{'packed' if packed else 'padded'}" for path in _GOLDENS for packed in (False, True)],
)
def case(request: pytest.FixtureRequest) -> tuple[dict[str, Any], TopkEmbedAdapter]:
    pytest.importorskip("transformers.models.qwen3_5", reason="needs transformers >= 5.2 (the transformers5 bundle)")
    path, packed = request.param
    golden = _golden(path)
    config = load_model_configs(_MODELS_DIR)[golden["model"]]
    assert config.hf_revision == golden["revision"], "golden was generated for a different checkpoint revision"
    adapter = load_adapter(config, _MODELS_DIR, device="cpu")
    assert isinstance(adapter, TopkEmbedAdapter)
    # On CPU the packed path runs its PyTorch reference kernels; on CUDA the fast ones.
    adapter._packed = packed
    adapter.load("cpu")
    assert (adapter._kernels is not None) == packed
    return golden, adapter


def _page(spec: dict[str, Any]) -> ImageInput:
    # Same synthetic page as the generator: dark bars on a light background.
    rng = np.random.default_rng(spec["seed"])
    height, width = spec["height"], spec["width"]
    page = np.full((height, width, 3), 245, dtype=np.uint8)
    line_height = max(6, height // 40)
    for top in range(height // 12, height - height // 12, line_height * 2):
        length = int(width * rng.uniform(0.3, 0.8))
        shade = int(rng.integers(10, 60))
        page[top : top + line_height, width // 10 : width // 10 + length] = shade
    buf = io.BytesIO()
    Image.fromarray(page).save(buf, "PNG")
    return ImageInput(data=buf.getvalue(), format="png")


def _directions(golden: dict[str, Any], dim: int) -> np.ndarray:
    spec = golden["generated_with"]["directions"]
    matrix = np.random.default_rng(spec["seed"]).standard_normal((spec["count"], dim))
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _maxsim(queries: list[np.ndarray], documents: list[np.ndarray]) -> np.ndarray:
    return np.array([[float((q @ d.T).max(axis=1).sum()) for d in documents] for q in queries])


def _encode(adapter: TopkEmbedAdapter, golden: dict[str, Any]) -> dict[str, list[np.ndarray]]:
    queries = adapter.encode([Item(text=q["text"]) for q in golden["queries"]], ["multivector"], is_query=True)
    documents = adapter.encode([Item(text=d["text"]) for d in golden["documents"]], ["multivector"])
    # One page per call, as the goldens were generated.
    pages = [adapter.encode([Item(images=[_page(p)])], ["multivector"]).multivector for p in golden["pages"]]
    assert queries.multivector is not None
    assert documents.multivector is not None
    return {
        "queries": queries.multivector,
        "documents": documents.multivector,
        "pages": [page[0] for page in pages if page is not None],
    }


def test_text_tokenization_matches(case: tuple[dict[str, Any], TopkEmbedAdapter]) -> None:
    golden, adapter = case
    tokenizer = adapter._tokenizer
    queries = [adapter._query_template + q["text"].strip() for q in golden["queries"]]
    documents = [(adapter._document_prompt + d["text"]).strip() for d in golden["documents"]]
    assert tokenizer(queries)["input_ids"] == [q["input_ids"] for q in golden["queries"]]
    assert tokenizer(documents)["input_ids"] == [d["input_ids"] for d in golden["documents"]]


def test_vectors_and_scores_match(case: tuple[dict[str, Any], TopkEmbedAdapter]) -> None:
    golden, adapter = case
    encoded = _encode(adapter, golden)
    basis = _directions(golden, encoded["queries"][0].shape[1])

    for kind, tolerance in (("queries", _TEXT_PROJECTION_TOLERANCE), ("documents", _TEXT_PROJECTION_TOLERANCE)):
        for vectors, expected in zip(encoded[kind], golden[kind], strict=True):
            assert len(vectors) == expected["tokens"], f"{kind}: {expected['text']!r}"
            np.testing.assert_allclose(basis @ vectors.T, expected["projections"], rtol=0, atol=tolerance)

    for vectors, expected in zip(encoded["pages"], golden["pages"], strict=True):
        assert len(vectors) == expected["tokens"], expected["name"]
        diff = np.abs(basis @ vectors.T - np.array(expected["projections"]))
        assert diff.max() <= _PAGE_PROJECTION_TOLERANCE, expected["name"]
        assert diff.mean() <= _PAGE_PROJECTION_MEAN_TOLERANCE, expected["name"]

    for key, documents in (("text_scores", encoded["documents"]), ("page_scores", encoded["pages"])):
        scores = _maxsim(encoded["queries"], documents)
        expected = np.array(golden[key])
        np.testing.assert_allclose(scores, expected, rtol=0, atol=_SCORE_TOLERANCE)
        # Retrieval order is unchanged for every query with a clear winner.
        for row, expected_row in zip(scores, expected, strict=True):
            ranked = np.sort(expected_row)
            if ranked[-1] - ranked[-2] > 2 * _SCORE_TOLERANCE:
                assert row.argmax() == expected_row.argmax()
