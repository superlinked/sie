"""GLiNERUnionAdapter reproduces the PII redaction study's composition, and superlinked/pii-redact serves it.

The fixture holds five recorded documents of that study with both models'
recorded spans for every window it sent, and the spans the study's own
``compose`` built from them. Fake members replay the recorded spans, so these
tests check the composition without loading a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.adapters.gliner import GLiNERAdapter
from sie_server.adapters.gliner_union.adapter import GLiNERUnionAdapter, word_windows
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.core.loader import load_model_configs, reject_unknown_loadtime_options, resolve_adapter_class
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import Entity

SERVER_ROOT = Path(__file__).resolve().parents[2]
MODELS = SERVER_ROOT / "models"
BUNDLES = SERVER_ROOT / "bundles"
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "gliner_union" / "redact-documents.json").read_text())
DOCUMENTS = FIXTURE["documents"]
FIRST, SECOND = FIXTURE["models"]
NUNER_REVISION = "c90187673f464518dca09f41689184ed6976242c"
STUDY_OPTIONS: dict[str, Any] = {
    "score_floor": 0.6,
    "window_words": 300,
    "window_overlap": 50,
    "propagate_label": "person",
    "name_min_letters": 3,
}
LABELS = ["person", "email address"]


class ReplayMember:
    """A member that returns one model's recorded spans for each window it is sent."""

    def __init__(self, model: str, *, errors: dict[str, ExtractItemError] | None = None) -> None:
        self.spans = {
            document["text"][window["start"] : window["end"]]: window["entities"][model]
            for document in DOCUMENTS
            for window in document["windows"]
        }
        self.errors = errors or {}
        self.calls: list[tuple[list[str], list[str] | None]] = []

    def extract(self, items: list[Item], *, labels: list[str] | None = None, **_: Any) -> ExtractOutput:
        texts = [item.text or "" for item in items]
        self.calls.append((texts, labels))
        return ExtractOutput(
            entities=[
                [
                    Entity(
                        text=text[s["start"] : s["end"]],
                        label=s["label"],
                        score=s["score"],
                        start=s["start"],
                        end=s["end"],
                    )
                    for s in self.spans.get(text, [])
                ]
                for text in texts
            ],
            errors=[self.errors.get(text) for text in texts] if self.errors else None,
            input_token_counts=[len(text.split()) for text in texts],
        )


def union_adapter(members: list[Any] | None = None, **runtime: Any) -> GLiNERUnionAdapter:
    adapter = GLiNERUnionAdapter(
        FIRST,
        members=[{"model": SECOND, "revision": NUNER_REVISION, "options": {"merge_adjacent_entities": True}}],
        revision="1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d",
        **{**STUDY_OPTIONS, **runtime},
    )
    adapter._members = cast("list[Any]", members or [ReplayMember(FIRST), ReplayMember(SECOND)])
    adapter._device = "cpu"
    return adapter


def spans_of(entities: list[Entity]) -> list[tuple[int | None, int | None, str]]:
    return [(e["start"], e["end"], e["label"]) for e in entities]


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda d: str(d["index"]))
def test_windows_are_the_requests_the_study_sent(document: dict[str, Any]) -> None:
    assert word_windows(document["text"], 300, 50) == [(w["start"], w["end"]) for w in document["windows"]]


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda d: str(d["index"]))
def test_reproduces_the_recorded_composition(document: dict[str, Any]) -> None:
    output = union_adapter().extract([Item(text=document["text"])], labels=LABELS)

    entities = output.entities[0]
    expected = document["expected"]
    assert spans_of(entities) == [(e["start"], e["end"], e["label"]) for e in expected]
    for entity, recorded in zip(entities, expected, strict=True):
        assert entity["text"] == document["text"][entity["start"] : entity["end"]]
        if recorded["score"] is None:
            assert entity["score"] >= STUDY_OPTIONS["score_floor"]
        else:
            assert entity["score"] == recorded["score"]
    assert output.errors is None


def test_fixture_covers_every_step_of_the_composition() -> None:
    windows = {len(d["windows"]) for d in DOCUMENTS}
    derived = sum(e["score"] is None for d in DOCUMENTS for e in d["expected"])
    recorded = [s for d in DOCUMENTS for w in d["windows"] for spans in w["entities"].values() for s in spans]
    assert windows == {1, 2, 3, 4}
    assert derived > 0
    assert any(s["score"] < STUDY_OPTIONS["score_floor"] for s in recorded)
    assert any(not d["expected"] for d in DOCUMENTS)


def test_composes_a_batch_of_documents_item_by_item() -> None:
    first, second = ReplayMember(FIRST), ReplayMember(SECOND)
    output = union_adapter([first, second]).extract([Item(text=d["text"]) for d in DOCUMENTS], labels=LABELS)

    for entities, document in zip(output.entities, DOCUMENTS, strict=True):
        assert spans_of(entities) == [(e["start"], e["end"], e["label"]) for e in document["expected"]]
    # Each member reads every window of the batch in one call.
    windows = sum(len(d["windows"]) for d in DOCUMENTS)
    assert [len(texts) for texts, _ in first.calls] == [windows]
    assert [len(texts) for texts, _ in second.calls] == [windows]
    assert output.input_token_counts == [
        2 * sum(len(d["text"][w["start"] : w["end"]].split()) for w in d["windows"]) for d in DOCUMENTS
    ]


def test_a_request_without_labels_uses_the_default_labels() -> None:
    first, second = ReplayMember(FIRST), ReplayMember(SECOND)
    adapter = union_adapter([first, second], default_labels=["person", "city"])

    adapter.extract([Item(text=DOCUMENTS[0]["text"])])
    adapter.extract([Item(text=DOCUMENTS[0]["text"])], labels=["email address"])

    assert [labels for _, labels in first.calls] == [["person", "city"], ["email address"]]
    assert [labels for _, labels in second.calls] == [["person", "city"], ["email address"]]


def test_a_request_without_labels_or_defaults_is_invalid() -> None:
    adapter = GLiNERUnionAdapter(FIRST, members=[{"model": SECOND, "revision": NUNER_REVISION}])
    adapter._device = "cpu"
    with pytest.raises(InvalidInputError, match="labels"):
        adapter.extract([Item(text="Ann Lee")])


def test_request_options_override_the_profile() -> None:
    document = next(d for d in DOCUMENTS if any(e["score"] is None for e in d["expected"]))
    item = [Item(text=document["text"])]

    without_names = union_adapter().extract(item, labels=LABELS, options={"propagate_label": None})
    everything = union_adapter().extract(item, labels=LABELS, options={"score_floor": 0.0, "propagate_label": ""})

    found = [(e["start"], e["end"], e["label"]) for e in document["expected"] if e["score"] is not None]
    assert spans_of(without_names.entities[0]) == found
    assert set(found) < set(spans_of(everything.entities[0]))


@pytest.mark.parametrize(
    "options",
    [
        {"score_floor": 1.5},
        {"score_floor": True},
        {"window_words": 1},
        {"window_overlap": 300},
        {"name_min_letters": 0},
        {"propagate_label": 3},
        {"default_labels": []},
    ],
)
def test_invalid_request_options_are_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(InvalidInputError):
        union_adapter().extract([Item(text="Ann Lee")], labels=LABELS, options=options)


def test_a_member_error_fails_only_its_item() -> None:
    failing = DOCUMENTS[1]
    window = failing["text"][failing["windows"][-1]["start"] : failing["windows"][-1]["end"]]
    error = ExtractItemError(code="INPUT_TOO_LONG", message="too long")
    adapter = union_adapter([ReplayMember(FIRST), ReplayMember(SECOND, errors={window: error})])

    output = adapter.extract([Item(text=DOCUMENTS[0]["text"]), Item(text=failing["text"])], labels=LABELS)

    assert output.errors == [None, error]
    assert output.entities[1] == []
    assert output.input_token_counts is not None
    assert output.input_token_counts[1] == 0
    assert spans_of(output.entities[0]) == [(e["start"], e["end"], e["label"]) for e in DOCUMENTS[0]["expected"]]


def test_extract_needs_a_loaded_model() -> None:
    adapter = GLiNERUnionAdapter(FIRST, members=[{"model": SECOND, "revision": NUNER_REVISION}])
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.extract([Item(text="Ann Lee")], labels=LABELS)


def test_members_are_pinned_gliner_adapters() -> None:
    adapter = GLiNERUnionAdapter(
        FIRST,
        members=[{"model": SECOND, "revision": NUNER_REVISION, "options": {"merge_adjacent_entities": True}}],
        revision="1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d",
        compute_precision="bfloat16",
    )
    first, second = adapter._members
    assert isinstance(first, GLiNERAdapter)
    assert isinstance(second, GLiNERAdapter)
    assert (first._model_name_or_path, first._revision) == (FIRST, "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d")
    assert (second._model_name_or_path, second._revision) == (SECOND, NUNER_REVISION)
    assert second._merge_adjacent_entities is True
    assert first._compute_precision == second._compute_precision == "bfloat16"


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ([], "at least one other member"),
        ([{"model": SECOND}], "full commit SHA"),
        ([{"model": SECOND, "revision": "main"}], "full commit SHA"),
        ([{"model": SECOND, "revision": NUNER_REVISION, "options": {"tresh": 0.5}}], "unsupported option"),
        ([{"model": SECOND, "revision": NUNER_REVISION, "weights": "x"}], "mapping of model, revision and options"),
        ([{"revision": NUNER_REVISION}], "needs a model"),
    ],
)
def test_unpinned_or_malformed_members_are_refused(members: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GLiNERUnionAdapter(FIRST, members=members)


def test_load_unload_and_footprint_cover_every_member() -> None:
    class Member:
        def __init__(self) -> None:
            self.events: list[str] = []

        def load(self, device: str) -> None:
            self.events.append(f"load {device}")

        def unload(self) -> None:
            self.events.append("unload")

        def memory_footprint(self) -> int:
            return 7

    members = [Member(), Member()]
    adapter = GLiNERUnionAdapter(FIRST, members=[{"model": SECOND, "revision": NUNER_REVISION}])
    adapter._members = cast("list[Any]", members)

    adapter.load("cpu")
    assert adapter.memory_footprint() == 14
    adapter.unload()

    assert [m.events for m in members] == [["load cpu", "unload"], ["load cpu", "unload"]]
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.extract([Item(text="Ann Lee")], labels=LABELS)


def test_a_failed_member_load_unloads_it_and_the_members_loaded_before_it() -> None:
    events: list[str] = []

    class Member:
        def __init__(self, name: str, *, fails: bool = False) -> None:
            self.name, self.fails = name, fails

        def load(self, device: str) -> None:
            if self.fails:
                raise MemoryError(f"{self.name} does not fit on {device}")
            events.append(f"load {self.name}")

        def unload(self) -> None:
            events.append(f"unload {self.name}")

    adapter = GLiNERUnionAdapter(FIRST, members=[{"model": SECOND, "revision": NUNER_REVISION}])
    adapter._members = cast("list[Any]", [Member("a"), Member("b"), Member("c", fails=True)])

    with pytest.raises(MemoryError, match="c does not fit"):
        adapter.load("cuda:0")

    # The failed member is unloaded too: GLiNERAdapter may hold its model before the device transfer fails.
    assert events == ["load a", "load b", "unload c", "unload b", "unload a"]
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.extract([Item(text="Ann Lee")], labels=LABELS)


def test_pii_redact_is_the_studied_composition() -> None:
    config = load_model_configs(MODELS)["superlinked/pii-redact"]
    profile = config.resolve_profile("default")

    assert (config.hf_id, config.hf_revision) == (FIRST, "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d")
    assert config.tasks.extract is not None
    assert config.tasks.encode is None
    assert config.tasks.score is None
    assert resolve_adapter_class(config, MODELS) is GLiNERUnionAdapter
    reject_unknown_loadtime_options(GLiNERUnionAdapter, profile.loadtime, model_name=config.sie_id)
    assert profile.loadtime["members"] == [
        {"model": SECOND, "revision": NUNER_REVISION, "options": {"merge_adjacent_entities": True}}
    ]
    runtime = dict(profile.runtime)
    labels = runtime.pop("default_labels")
    assert runtime == STUDY_OPTIONS
    assert len(labels) == len(set(labels)) == 36
    assert labels[0] == "person"
    assert "superlinked/pii-redact" in match_bundle_models(BUNDLES / "default.yaml", MODELS)
    # The members' own catalog entries pin the same checkpoints.
    catalog = load_model_configs(MODELS)
    assert catalog[FIRST].hf_revision == config.hf_revision
    assert catalog[SECOND].hf_revision == NUNER_REVISION
    assert catalog[SECOND].resolve_profile("default").runtime == {"merge_adjacent_entities": True}
    assert config.hf_tokenizer_dependencies == {
        **catalog[FIRST].hf_tokenizer_dependencies,
        **catalog[SECOND].hf_tokenizer_dependencies,
    }
