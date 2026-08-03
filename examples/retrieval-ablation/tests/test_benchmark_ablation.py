from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _load_benchmark() -> Any:
    path = ROOT / "benchmark_ablation.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark()


def test_sample_fixture_is_valid_and_bounded() -> None:
    corpus, queries, qrels = BENCHMARK.load_sample_dataset()

    assert len(corpus) == BENCHMARK.SAMPLE_MAX_DOCUMENTS == 12
    assert len(queries) == BENCHMARK.SAMPLE_MAX_QUERIES == 6
    document_ids = {document["corpus_id"] for document in corpus}
    assert len(document_ids) == len(corpus)
    assert {query["query_id"] for query in queries} == set(qrels)
    assert all(set(relevant) <= document_ids for relevant in qrels.values())
    assert all(score > 0 for relevant in qrels.values() for score in relevant.values())


def test_sample_rankings_use_canonical_evaluation() -> None:
    corpus, queries, qrels = BENCHMARK.load_sample_dataset()
    document_ids = [document["corpus_id"] for document in corpus]
    perfect_rankings = []
    for query in queries:
        relevant_ids = list(qrels[query["query_id"]])
        perfect_rankings.append(
            relevant_ids + [document_id for document_id in document_ids if document_id not in relevant_ids]
        )

    metrics = BENCHMARK.evaluate(perfect_rankings, queries, qrels)

    assert metrics == {
        "ndcg@10": 1.0,
        "mrr@10": 1.0,
        "recall@10": 1.0,
        "n_queries": 6,
    }


def test_sample_dry_run_uses_isolated_runtime_state() -> None:
    env = {**os.environ, "SIE_BASE_URL": "http://localhost:8080"}
    completed = subprocess.run(
        [sys.executable, str(ROOT / "benchmark_ablation.py"), "--sample", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Dataset: sample" in completed.stderr
    assert "Namespace: ablation-sample-baai-bge-m3" in completed.stderr
    assert "cache/ablation/sample" in completed.stderr
    assert "sample_ablation_results.csv" in completed.stderr
    assert "Sample loaded: 12 documents, 6 queries, 8 qrels" in completed.stderr
