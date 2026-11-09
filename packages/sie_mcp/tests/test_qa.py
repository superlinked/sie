from typing import Any

import pytest
from sie_mcp import qa
from sie_mcp.qa import AnswerQuestionsError
from sie_sdk import RequestError

# A tiny keyword vocabulary lets the fake produce deterministic embeddings:
# encode() returns per-keyword counts, so cosine / dot-product retrieval is
# meaningful without a real cluster.
_VOCAB = ("alpha", "bravo", "charlie", "delta", "echo")

_PARAMS: dict[str, Any] = {
    "encode_model": "enc",
    "rerank_model": "rr",
    "generate_model": "gen",
    "top_k": 1,
    "rerank_candidates": 20,
    "chunk_chars": 1000,
    "chunk_overlap_chars": 100,
    "max_tokens": 64,
    "max_document_chars": 1_000_000,
    "max_questions": 50,
    "max_chunks": 1000,
}


def _embed(text: str) -> list[float]:
    lowered = text.lower()
    return [float(lowered.count(word)) for word in _VOCAB]


class _FakeQAClient:
    """Keyword-embedding fake: encode → counts, score → dot product, chat → canned."""

    def __init__(self, answer: str = "grounded answer") -> None:
        self.answer = answer
        self.encode_calls: list[dict[str, Any]] = []
        self.score_calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []

    async def encode(self, model: str, items: Any, **kwargs: Any) -> Any:
        self.encode_calls.append({"model": model, "items": items, **kwargs})
        single = isinstance(items, dict)
        item_list = [items] if single else items
        results = [{"id": item.get("id"), "dense": _embed(item["text"])} for item in item_list]
        return results[0] if single else results

    async def score(self, model: str, query: Any, items: list[Any], **kwargs: Any) -> Any:
        self.score_calls.append({"model": model, "query": query, "items": items, **kwargs})
        q = _embed(query["text"])
        scored = sorted(
            ((item["id"], sum(a * b for a, b in zip(_embed(item["text"]), q, strict=False))) for item in items),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return {
            "model": model,
            "scores": [{"item_id": cid, "score": s, "rank": r} for r, (cid, s) in enumerate(scored)],
        }

    async def chat_completions(self, model: str, messages: list[Any], **kwargs: Any) -> Any:
        self.chat_calls.append({"model": model, "messages": messages, **kwargs})
        return {"choices": [{"message": {"role": "assistant", "content": self.answer}}]}


# Three single-chunk documents (each shorter than the window), one keyword apiece.
_DOCS = [
    "Alpha report: the alpha subsystem is green.",
    "Bravo report: the bravo subsystem is amber.",
    "Charlie report: the charlie subsystem is red.",
]


async def test_returns_grounded_answer_and_passages() -> None:
    client = _FakeQAClient(answer="The bravo subsystem is amber.")

    result = await qa.answer_questions(client, documents=_DOCS, questions=["What is the status of bravo?"], **_PARAMS)

    answers = result["answers"]
    assert len(answers) == 1
    answer = answers[0]
    assert answer["question"] == "What is the status of bravo?"
    assert answer["answer"] == "The bravo subsystem is amber."
    # The selected passage is the bravo document — retrieval grounded the answer.
    assert [p["text"] for p in answer["passages"]] == [_DOCS[1]]
    passage = answer["passages"][0]
    assert passage["doc_index"] == 1
    # start points back into the source document (single-chunk doc → offset 0).
    assert passage["start"] == 0
    assert _DOCS[passage["doc_index"]][passage["start"] :].startswith(passage["text"])


async def test_passages_are_substrings_of_the_documents() -> None:
    client = _FakeQAClient()

    result = await qa.answer_questions(client, documents=_DOCS, questions=["alpha?"], **_PARAMS)

    for passage in result["answers"][0]["passages"]:
        assert passage["text"] in _DOCS[passage["doc_index"]]


async def test_passes_configured_models_and_query_flags() -> None:
    client = _FakeQAClient()

    await qa.answer_questions(client, documents=_DOCS, questions=["charlie?"], **_PARAMS)

    # Chunks are encoded once as documents; the question is encoded as a query.
    chunk_encode = client.encode_calls[0]
    assert chunk_encode["model"] == "enc"
    assert chunk_encode["is_query"] is False
    assert isinstance(chunk_encode["items"], list)
    question_encode = client.encode_calls[1]
    assert question_encode["model"] == "enc"
    assert question_encode["is_query"] is True
    # Rerank and generation route to their configured models.
    assert client.score_calls[0]["model"] == "rr"
    assert client.chat_calls[0]["model"] == "gen"
    assert client.chat_calls[0]["temperature"] == 0.0
    assert client.chat_calls[0]["max_completion_tokens"] == _PARAMS["max_tokens"]


async def test_generator_is_fed_only_the_selected_passages() -> None:
    client = _FakeQAClient()

    await qa.answer_questions(client, documents=_DOCS, questions=["bravo?"], **_PARAMS)

    messages = client.chat_calls[0]["messages"]
    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    # The grounding passage is in the prompt; the unrelated documents are not.
    assert _DOCS[1] in user_content
    assert _DOCS[0] not in user_content


async def test_shortlist_limits_candidates_sent_to_reranker() -> None:
    docs = [
        "Alpha note: alpha alpha.",
        "Bravo note: bravo bravo.",
        "Charlie note: charlie charlie.",
        "Delta note: delta delta.",
        "Echo note: echo echo.",
    ]
    client = _FakeQAClient()

    await qa.answer_questions(
        client,
        documents=docs,
        questions=["bravo and charlie"],
        **{**_PARAMS, "rerank_candidates": 2, "top_k": 2},
    )

    reranked = client.score_calls[0]["items"]
    # Only the two dense-closest chunks (bravo, charlie) reach the reranker.
    assert len(reranked) == 2
    assert {item["id"] for item in reranked} == {"d1-c0", "d2-c0"}


async def test_works_on_document_larger_than_one_window() -> None:
    big_doc = "Bravo status update. " * 400  # ~8000 chars, far larger than the window.
    client = _FakeQAClient()

    result = await qa.answer_questions(
        client,
        documents=[big_doc],
        questions=["bravo?"],
        **{**_PARAMS, "chunk_chars": 200, "chunk_overlap_chars": 20, "top_k": 3},
    )

    # The oversized document chunked into many pieces, encoded in one batch.
    assert len(client.encode_calls[0]["items"]) > 1
    passages = result["answers"][0]["passages"]
    assert passages
    assert all(p["doc_index"] == 0 and p["text"] in big_doc for p in passages)


async def test_no_state_persisted_between_calls() -> None:
    client = _FakeQAClient()

    await qa.answer_questions(client, documents=_DOCS, questions=["alpha?"], **_PARAMS)
    chunk_encodes_after_first = sum(1 for c in client.encode_calls if c["is_query"] is False)

    await qa.answer_questions(client, documents=_DOCS, questions=["bravo?"], **_PARAMS)
    chunk_encodes_after_second = sum(1 for c in client.encode_calls if c["is_query"] is False)

    # Each call re-encodes its chunks from scratch — no standing index or cache.
    assert chunk_encodes_after_first == 1
    assert chunk_encodes_after_second == 2


async def test_rejects_empty_documents() -> None:
    client = _FakeQAClient()

    with pytest.raises(AnswerQuestionsError):
        await qa.answer_questions(client, documents=["", "   "], questions=["alpha?"], **_PARAMS)


async def test_rejects_empty_questions() -> None:
    client = _FakeQAClient()

    with pytest.raises(AnswerQuestionsError):
        await qa.answer_questions(client, documents=_DOCS, questions=["  "], **_PARAMS)


async def test_rejects_documents_over_char_limit() -> None:
    # The corpus is bounded before any chunking/encoding, so the client is never called.
    client = _FakeQAClient()

    with pytest.raises(AnswerQuestionsError, match="character limit"):
        await qa.answer_questions(
            client, documents=_DOCS, questions=["alpha?"], **{**_PARAMS, "max_document_chars": 10}
        )

    assert client.encode_calls == []


async def test_rejects_too_many_questions() -> None:
    client = _FakeQAClient()

    with pytest.raises(AnswerQuestionsError, match="number of questions"):
        await qa.answer_questions(
            client, documents=_DOCS, questions=["alpha?", "bravo?"], **{**_PARAMS, "max_questions": 1}
        )

    assert client.encode_calls == []


async def test_rejects_too_many_chunks() -> None:
    # _DOCS yields three chunks (one per doc); a max of one rejects before encoding.
    client = _FakeQAClient()

    with pytest.raises(AnswerQuestionsError, match="exceeds the limit"):
        await qa.answer_questions(client, documents=_DOCS, questions=["alpha?"], **{**_PARAMS, "max_chunks": 1})

    assert client.encode_calls == []


@pytest.mark.parametrize(
    "bad",
    [
        {"top_k": 0},
        {"rerank_candidates": 0},
        {"chunk_chars": 0},
        {"max_tokens": 0},
        {"max_document_chars": 0},
        {"max_questions": 0},
        {"max_chunks": 0},
        {"chunk_overlap_chars": 1000},  # overlap == chunk_chars (not < it)
    ],
)
async def test_rejects_invalid_knobs(bad: dict[str, Any]) -> None:
    # Invalid retrieval knobs raise the job's own error before any chunking/encoding,
    # rather than a raw ValueError from chunk_documents deeper in the pipeline.
    client = _FakeQAClient()

    with pytest.raises(AnswerQuestionsError):
        await qa.answer_questions(client, documents=_DOCS, questions=["alpha?"], **{**_PARAMS, **bad})

    assert client.encode_calls == []


async def test_raises_when_reranker_returns_no_scores() -> None:
    class _NoScores(_FakeQAClient):
        async def score(self, model: str, query: Any, items: list[Any], **kwargs: Any) -> Any:
            return {"model": model, "scores": []}

    with pytest.raises(AnswerQuestionsError, match="no scores"):
        await qa.answer_questions(_NoScores(), documents=_DOCS, questions=["alpha?"], **_PARAMS)


async def test_names_the_setting_when_the_reranker_is_not_served() -> None:
    class _NoReranker(_FakeQAClient):
        async def score(self, model: str, query: Any, items: list[Any], **kwargs: Any) -> Any:
            raise RequestError(f"Model '{model}' not found", code="MODEL_NOT_FOUND", status_code=404)

    with pytest.raises(AnswerQuestionsError, match=r"'rr' is not served.*SIE_MCP_RERANK_MODEL"):
        await qa.answer_questions(_NoReranker(), documents=_DOCS, questions=["alpha?"], **_PARAMS)


async def test_other_request_errors_pass_through() -> None:
    class _BadRequest(_FakeQAClient):
        async def score(self, model: str, query: Any, items: list[Any], **kwargs: Any) -> Any:
            raise RequestError("bad request", code="INVALID_INPUT", status_code=400)

    with pytest.raises(RequestError, match="bad request"):
        await qa.answer_questions(_BadRequest(), documents=_DOCS, questions=["alpha?"], **_PARAMS)


async def test_doc_index_points_to_original_documents_when_some_are_empty() -> None:
    # A leading empty document must not shift the surviving document's doc_index:
    # the bravo doc stays at its caller-supplied index 2.
    docs = ["", "   ", "Bravo report: the bravo subsystem is amber."]
    client = _FakeQAClient()

    result = await qa.answer_questions(client, documents=docs, questions=["bravo?"], **_PARAMS)

    passage = result["answers"][0]["passages"][0]
    assert passage["doc_index"] == 2
    assert passage["text"] == docs[2]


async def test_falls_back_to_dense_when_rerank_ids_unmatched() -> None:
    # Reranker returns scores whose item_ids don't map to any chunk; the job must
    # still answer, grounded in the dense-shortlisted top passages.
    class _ForeignIds(_FakeQAClient):
        async def score(self, model: str, query: Any, items: list[Any], **kwargs: Any) -> Any:
            self.score_calls.append({"model": model, "query": query, "items": items})
            return {"model": model, "scores": [{"item_id": "nonexistent", "score": 9.0, "rank": 0}]}

    client = _ForeignIds()
    result = await qa.answer_questions(client, documents=_DOCS, questions=["bravo?"], **_PARAMS)

    passages = result["answers"][0]["passages"]
    assert passages  # did not fail the question
    assert passages[0]["text"] == _DOCS[1]  # dense retrieval still surfaced bravo


async def test_dense_floats_accepts_numpy_array_from_real_encode() -> None:
    # The real cluster returns dense as a numpy array, not a list; pin that.
    np = pytest.importorskip("numpy")

    floats = qa._dense_floats({"dense": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)})

    assert floats == [1.0, 2.0, 3.0]


async def test_raises_when_encode_has_no_dense_embedding() -> None:
    class _NoDense(_FakeQAClient):
        async def encode(self, model: str, items: Any, **kwargs: Any) -> Any:
            single = isinstance(items, dict)
            result = {"id": "x"}  # no "dense" key
            return result if single else [result]

    with pytest.raises(AnswerQuestionsError, match="no dense embedding"):
        await qa.answer_questions(_NoDense(), documents=_DOCS, questions=["alpha?"], **_PARAMS)


async def test_raises_when_generation_returns_no_content() -> None:
    class _NoContent(_FakeQAClient):
        async def chat_completions(self, model: str, messages: list[Any], **kwargs: Any) -> Any:
            return {"choices": []}

    with pytest.raises(AnswerQuestionsError, match="no choices"):
        await qa.answer_questions(_NoContent(), documents=_DOCS, questions=["alpha?"], **_PARAMS)
