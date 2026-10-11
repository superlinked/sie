"""A json_schema response format is shown to the model as well as enforced by the grammar."""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sie_server.processors import streaming as streaming_mod
from sie_server.processors.schema_prompt import (
    json_schema_response_format,
    profile_schema_in_prompt,
    renders_system_role,
    response_format_section,
    schema_in_messages,
    with_response_format,
)
from sie_server.processors.streaming import StreamingProcessor
from sie_server.types.grammar import GrammarSpec

from .test_streaming import (
    _decode_chunks,
    _make_msg,
    _make_registry_with_chat_config,
    _make_work_item,
    _PreflightGenAdapter,
)

WEAKNESSES = ["cross-site scripting", "SQL injection", "memory corruption", "path traversal"]
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "weakness": {"type": "string", "enum": WEAKNESSES},
        "attack_vector": {"type": "string", "enum": ["network", "adjacent", "local", "physical"]},
    },
    "required": ["weakness", "attack_vector"],
    "additionalProperties": False,
}
GRAMMAR = GrammarSpec(kind="json_schema", value=SCHEMA, label="cvss", strict=True)
_SECTION = response_format_section(GRAMMAR)
assert _SECTION is not None
SECTION: str = _SECTION


# -- rendering -------------------------------------------------------------------------------------------------


def test_section_is_the_response_formats_block_with_compact_schema() -> None:
    compact = json.dumps(SCHEMA, ensure_ascii=False, separators=(",", ":"))
    assert compact == SECTION.removeprefix("# Response Formats\n\n## cvss\n\n")
    assert SECTION.startswith("# Response Formats\n\n## cvss\n\n")
    for option in WEAKNESSES:
        assert json.dumps(option) in SECTION


def test_section_names_an_unlabelled_format_response() -> None:
    section = response_format_section(GrammarSpec(kind="json_schema", value=SCHEMA))
    assert section is not None
    assert section.startswith("# Response Formats\n\n## response\n\n")


@pytest.mark.parametrize(
    "grammar",
    [
        None,
        GrammarSpec(kind="regex", value="^(yes|no)$"),
        GrammarSpec(kind="ebnf", value='root ::= "a"'),
        GrammarSpec(kind="json_schema", value={"type": "object", "additionalProperties": True}, label="json_object"),
        GrammarSpec(kind="json_schema", value={"type": "object"}),
    ],
)
def test_nothing_to_show_without_an_informative_schema(grammar: GrammarSpec | None) -> None:
    assert response_format_section(grammar) is None


def test_openai_response_format_reads_the_json_schema() -> None:
    grammar = json_schema_response_format(
        {"type": "json_schema", "json_schema": {"name": "cvss", "schema": SCHEMA, "strict": True}}
    )
    assert grammar == GrammarSpec(kind="json_schema", value=SCHEMA, label="cvss")
    assert json_schema_response_format({"type": "json_object"}) is None
    assert json_schema_response_format({"type": "json_schema", "json_schema": {"name": "x"}}) is None
    assert json_schema_response_format(None) is None


# -- placement -------------------------------------------------------------------------------------------------


def test_section_is_appended_to_the_system_message() -> None:
    messages = [{"role": "system", "content": "Score the report."}, {"role": "user", "content": "XSS in IBM Db2"}]
    out = with_response_format(messages, SECTION, system_role=True)
    assert out[0] == {"role": "system", "content": "Score the report.\n\n" + SECTION}
    assert out[1] == messages[1]
    assert messages[0]["content"] == "Score the report."  # input untouched


def test_section_is_appended_to_a_developer_message_and_to_text_parts() -> None:
    messages = [{"role": "developer", "content": [{"type": "text", "text": "Be terse."}]}]
    out = with_response_format(messages, SECTION, system_role=True)
    assert out[0]["content"] == [{"type": "text", "text": "Be terse."}, {"type": "text", "text": "\n\n" + SECTION}]


def test_section_becomes_the_system_message_when_there_is_none() -> None:
    messages = [{"role": "user", "content": "XSS in IBM Db2"}]
    assert with_response_format(messages, SECTION, system_role=True) == [
        {"role": "system", "content": SECTION},
        {"role": "user", "content": "XSS in IBM Db2"},
    ]


def test_section_opens_the_first_user_message_without_a_system_role() -> None:
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is on screen?"}]},
        {"role": "assistant", "content": "A form."},
        {"role": "user", "content": "Extract it."},
    ]
    out = with_response_format(messages, SECTION, system_role=False)
    assert out[0]["content"] == [
        {"type": "text", "text": SECTION + "\n\n"},
        {"type": "image"},
        {"type": "text", "text": "What is on screen?"},
    ]
    assert out[1:] == messages[1:]
    plain = with_response_format([{"role": "user", "content": "Extract it."}], SECTION, system_role=False)
    assert plain == [{"role": "user", "content": SECTION + "\n\nExtract it."}]


# -- the caller already shows the schema ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rendering",
    [
        json.dumps(SCHEMA, indent=2),
        json.dumps(SCHEMA),
        json.dumps(SCHEMA, separators=(",", ":")),
        json.dumps(SCHEMA, indent=4, sort_keys=True),
    ],
)
def test_schema_already_in_the_prompt_is_found(rendering: str) -> None:
    messages = [{"role": "system", "content": f"Score the report.\n\nJSON schema:\n{rendering}"}]
    assert schema_in_messages(messages, SCHEMA)


def test_schema_in_a_user_text_part_is_found_including_non_ascii() -> None:
    schema = {"type": "object", "properties": {"größe": {"type": "string", "enum": ["klein", "groß"]}}}
    for ascii_only in (False, True):
        text = "Respond with JSON matching this schema:\n" + json.dumps(schema, indent=2, ensure_ascii=ascii_only)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        assert schema_in_messages(messages, schema)


def test_a_different_schema_is_not_mistaken_for_this_one() -> None:
    other = {**SCHEMA, "required": ["weakness"]}
    messages = [{"role": "system", "content": "JSON schema:\n" + json.dumps(other, indent=2)}]
    assert not schema_in_messages(messages, SCHEMA)
    assert not schema_in_messages([{"role": "user", "content": "weakness attack_vector network"}], SCHEMA)


# -- templates and profiles --------------------------------------------------------------------------------------


def _template(*, system: str) -> Any:
    def apply_chat_template(messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        parts = []
        for message in messages:
            if message["role"] == "system":
                if system == "raise":
                    raise ValueError("System role not supported")
                if system == "drop":
                    continue
            parts.append(f"<{message['role']}>{message['content']}")
        return "".join(parts)

    return apply_chat_template


def test_system_role_probe() -> None:
    assert renders_system_role(_template(system="keep"))
    assert not renders_system_role(_template(system="raise"))
    assert not renders_system_role(_template(system="drop"))


def test_profile_runtime_opt_out() -> None:
    def config(runtime: dict[str, Any]) -> Any:
        profile = MagicMock()
        profile.runtime = MappingProxyType(runtime)
        cfg = MagicMock()
        cfg.resolve_profile.return_value = profile
        return cfg

    assert profile_schema_in_prompt(config({}))
    assert profile_schema_in_prompt(config({"schema_in_prompt": True}))
    assert not profile_schema_in_prompt(config({"schema_in_prompt": False}))


# -- queued generation (gateway and sidecar path) ----------------------------------------------------------------


class _RecordingTokenizer:
    def __init__(self, *, system: str = "keep") -> None:
        self.system = system
        self.rendered: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], *, tokenize: bool, add_generation_prompt: bool, **_):
        assert tokenize is False
        assert add_generation_prompt is True
        rendered = _template(system=self.system)(messages)
        self.rendered.append([dict(m) for m in messages])
        return rendered

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [0] * len(text.split())


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    generate: dict[str, Any],
    *,
    tokenizer: _RecordingTokenizer | None = None,
    profile_runtime: dict[str, Any] | None = None,
) -> tuple[_RecordingTokenizer, _PreflightGenAdapter, AsyncMock]:
    tok = tokenizer or _RecordingTokenizer()
    adapter = _PreflightGenAdapter()
    registry = _make_registry_with_chat_config(adapter)
    profile = registry.get_config.return_value.resolve_profile.return_value
    profile.runtime = MappingProxyType(profile_runtime or {})
    monkeypatch.setattr(streaming_mod, "load_tokenizer", lambda *_a, **_kw: tok)
    nc = AsyncMock()
    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    await proc.process(_make_msg(_make_work_item(generate=generate)), "test/model")
    return tok, adapter, nc


def _request(system: str | None = "Score the report.", **extra: Any) -> dict[str, Any]:
    messages = [{"role": "user", "content": "IBM Db2 Mirror could allow cross-site scripting."}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    return {
        "messages": messages,
        "max_new_tokens": 64,
        "temperature": 0.0,
        "grammar": {"kind": "json_schema", "value": SCHEMA, "label": "cvss", "strict": True},
        **extra,
    }


def _system_texts(tok: _RecordingTokenizer) -> list[str]:
    """System texts of the request render (the probe render, if any, comes first)."""
    return [m["content"] for m in tok.rendered[-1] if m["role"] == "system"]


@pytest.mark.asyncio
async def test_queued_request_shows_the_schema_and_keeps_the_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    tok, adapter, _ = await _run(monkeypatch, _request())

    (system,) = _system_texts(tok)
    assert system == "Score the report.\n\n" + SECTION
    for option in WEAKNESSES:
        assert option in system
    assert adapter.dispatched_parameters is not None
    assert adapter.dispatched_parameters["grammar"] == GRAMMAR
    assert "# Response Formats" in adapter.dispatched_parameters["prompt"]


@pytest.mark.asyncio
async def test_queued_request_without_system_message_gets_one(monkeypatch: pytest.MonkeyPatch) -> None:
    tok, _, _ = await _run(monkeypatch, _request(system=None))
    assert tok.rendered[-1][0] == {"role": "system", "content": SECTION}


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["raise", "drop"])
async def test_template_without_system_role_gets_the_schema_in_the_user_turn(
    monkeypatch: pytest.MonkeyPatch, system: str
) -> None:
    tok, adapter, _ = await _run(monkeypatch, _request(system=None), tokenizer=_RecordingTokenizer(system=system))
    (user,) = tok.rendered[-1]
    assert user["role"] == "user"
    assert user["content"] == SECTION + "\n\nIBM Db2 Mirror could allow cross-site scripting."
    assert adapter.dispatched_parameters is not None


@pytest.mark.asyncio
async def test_request_opt_out_leaves_the_prompt_as_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    tok, adapter, _ = await _run(monkeypatch, _request(schema_in_prompt=False))
    assert _system_texts(tok) == ["Score the report."]
    assert adapter.dispatched_parameters is not None
    assert adapter.dispatched_parameters["grammar"] == GRAMMAR


@pytest.mark.asyncio
async def test_profile_opt_out_and_request_override(monkeypatch: pytest.MonkeyPatch) -> None:
    tok, _, _ = await _run(monkeypatch, _request(), profile_runtime={"schema_in_prompt": False})
    assert _system_texts(tok) == ["Score the report."]

    tok, _, _ = await _run(monkeypatch, _request(schema_in_prompt=True), profile_runtime={"schema_in_prompt": False})
    assert _system_texts(tok) == ["Score the report.\n\n" + SECTION]


@pytest.mark.asyncio
async def test_schema_the_caller_already_sent_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    system = "Score the report.\n\nJSON schema:\n" + json.dumps(SCHEMA, indent=2)
    tok, _, _ = await _run(monkeypatch, _request(system=system))
    assert _system_texts(tok) == [system]


@pytest.mark.asyncio
async def test_requests_without_a_json_schema_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    regex = {**_request(), "grammar": {"kind": "regex", "value": "^(yes|no)$"}}
    tok, _, _ = await _run(monkeypatch, regex)
    assert _system_texts(tok) == ["Score the report."]

    plain = {key: value for key, value in _request().items() if key != "grammar"}
    tok, _, _ = await _run(monkeypatch, plain)
    assert _system_texts(tok) == ["Score the report."]


@pytest.mark.asyncio
async def test_non_boolean_opt_out_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    tok, adapter, nc = await _run(monkeypatch, _request(schema_in_prompt="no"))
    terminal = _decode_chunks(nc)[-1]
    assert terminal["error"]["code"] == "invalid_request"
    assert "schema_in_prompt" in terminal["error"]["message"]
    assert adapter.dispatched_parameters is None
    assert tok.rendered == []


@pytest.mark.asyncio
async def test_system_role_probe_runs_once_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    tok = _RecordingTokenizer(system="raise")
    adapter = _PreflightGenAdapter()
    registry = _make_registry_with_chat_config(adapter)
    registry.get_config.return_value.resolve_profile.return_value.runtime = MappingProxyType({})
    monkeypatch.setattr(streaming_mod, "load_tokenizer", lambda *_a, **_kw: tok)
    proc = StreamingProcessor(nc=AsyncMock(), registry=registry, worker_id="w1")
    probes = MagicMock(wraps=streaming_mod.renders_system_role)
    monkeypatch.setattr(streaming_mod, "renders_system_role", probes)
    for _ in range(2):
        registry.get.return_value = _PreflightGenAdapter()
        await proc.process(_make_msg(_make_work_item(generate=_request(system=None))), "test/model")
    assert probes.call_count == 1
    assert tok.rendered[-1][0]["content"].startswith(SECTION)
