import json
from typing import Any

import pytest
from sie_mcp import structured
from sie_mcp.structured import StructuredOutputError, validate_output_schema

_MODEL = "Qwen/Qwen3.5-4B"


class _FakeChatClient:
    """Records chat_completions() calls and returns a canned ChatCompletion dict."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    async def chat_completions(self, model: str, messages: Any, **kwargs: Any) -> Any:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return {"choices": [{"message": {"role": "assistant", "content": self._content}}]}


# ── schema-subset validation ────────────────────────────────────────────


def test_accepts_realistic_extraction_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "qty": {"type": "integer"}},
                    "required": ["name", "qty"],
                },
            },
            "total": {"type": "number"},
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    validate_output_schema(schema)  # no raise


def test_rejects_non_object_schema() -> None:
    with pytest.raises(StructuredOutputError):
        validate_output_schema("nope")


def test_rejects_dollar_ref() -> None:
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/Foo"}}}
    with pytest.raises(StructuredOutputError, match=r"\$ref"):
        validate_output_schema(schema)


@pytest.mark.parametrize(
    "keyword",
    ["$dynamicRef", "if", "then", "else", "unevaluatedProperties", "dependentSchemas"],
)
def test_rejects_each_unsupported_keyword(keyword: str) -> None:
    schema = {"type": "object", keyword: {"x": True}}
    with pytest.raises(StructuredOutputError, match=keyword.replace("$", r"\$")):
        validate_output_schema(schema)


def test_rejects_too_deep_schema() -> None:
    leaf: Any = {"type": "string"}
    for _ in range(structured._MAX_SCHEMA_DEPTH + 5):
        leaf = {"type": "object", "properties": {"x": leaf}}
    with pytest.raises(StructuredOutputError, match="depth"):
        validate_output_schema(leaf)


def test_accepts_schema_at_depth_limit() -> None:
    leaf: Any = {"type": "string"}
    # Each "properties" wrap adds one to depth; stay at the limit.
    for _ in range(structured._MAX_SCHEMA_DEPTH):
        leaf = {"type": "object", "properties": {"x": leaf}}
    validate_output_schema(leaf)  # no raise


def test_accepts_property_names_colliding_with_keywords() -> None:
    # A property *name* that collides with a JSON-Schema keyword is data, not a
    # keyword: under ``properties`` the keys are caller-supplied names. The subset
    # walk must accept them, matching the gateway and server validators.
    schema = {
        "type": "object",
        "properties": {
            "if": {"type": "string"},
            "then": {"type": "string"},
            "else": {"type": "string"},
            "$ref": {"type": "string"},
            "dependentSchemas": {"type": "boolean"},
            "unevaluatedProperties": {"type": "integer"},
        },
    }
    validate_output_schema(schema)  # no raise


def test_accepts_keyword_named_property_nested_in_pattern_properties() -> None:
    # ``patternProperties`` is also a schema-map: its keys are patterns (data),
    # so a keyword-named pattern key must be accepted. Using ``then`` here is
    # deliberate — if ``patternProperties`` were walked as a *schema* context
    # instead of ``schema_map``, this key would be rejected as a keyword.
    schema = {
        "type": "object",
        "patternProperties": {"then": {"type": "object", "properties": {"if": {"type": "string"}}}},
    }
    validate_output_schema(schema)  # no raise


def test_rejects_keyword_in_real_schema_position_under_properties() -> None:
    # The gate must not over-accept: a genuine keyword sitting in a *schema*
    # position (the value of a property is a schema) is still rejected.
    schema = {"type": "object", "properties": {"a": {"type": "object", "if": {"x": True}}}}
    with pytest.raises(StructuredOutputError, match="if"):
        validate_output_schema(schema)


# ── gateway safety caps (mirror handlers/grammar.rs) ─────────────────────


def test_rejects_wide_object_schema_over_caps() -> None:
    # Mirrors the review repro: a 20k-property object exceeds the gateway caps
    # (serialized size and/or node count) and must be rejected at the edge.
    schema = {"type": "object", "properties": {f"f{i}": {"type": "string"} for i in range(20_000)}}
    with pytest.raises(StructuredOutputError, match="exceeds limit"):
        validate_output_schema(schema)


def test_rejects_oversized_serialized_schema() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "string", "description": "a" * (64 * 1024)}}}
    with pytest.raises(StructuredOutputError, match="serialized size"):
        validate_output_schema(schema)


def test_rejects_too_many_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Node count is shadowed by the byte cap under default JSON serialization, so
    # exercise it directly with a lowered cap to prove the walker counts nodes.
    monkeypatch.setattr(structured, "_MAX_SCHEMA_NODES", 5)
    schema = {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(10)}}
    with pytest.raises(StructuredOutputError, match="node count"):
        validate_output_schema(schema)


# ── extract_structured ──────────────────────────────────────────────────


async def test_extract_returns_parsed_json_and_call_shape() -> None:
    client = _FakeChatClient(json.dumps({"city": "Lisbon", "country": "Portugal"}))
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
        "title": "location",
    }

    data = await structured.extract_structured(
        client, content="I flew to Lisbon, Portugal.", output_schema=schema, model=_MODEL
    )

    assert data == {"city": "Lisbon", "country": "Portugal"}
    call = client.calls[0]
    assert call["model"] == _MODEL
    assert call["temperature"] == 0.0
    rf = call["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "location"
    assert rf["json_schema"]["schema"] is schema
    assert rf["json_schema"]["strict"] is True
    # Content is grounded in the supplied document.
    assert "Lisbon" in call["messages"][-1]["content"]


async def test_extract_uses_default_schema_name_without_title() -> None:
    client = _FakeChatClient("{}")
    await structured.extract_structured(client, content="x", output_schema={"type": "object"}, model=_MODEL)
    assert client.calls[0]["response_format"]["json_schema"]["name"] == "extraction"


async def test_extract_includes_instruction() -> None:
    client = _FakeChatClient("{}")
    await structured.extract_structured(
        client,
        content="x",
        output_schema={"type": "object"},
        instruction="Pull out the invoice number.",
        model=_MODEL,
    )
    assert "invoice number" in client.calls[0]["messages"][-1]["content"]


async def test_extract_validates_schema_before_calling_cluster() -> None:
    client = _FakeChatClient("{}")
    with pytest.raises(StructuredOutputError):
        await structured.extract_structured(
            client,
            content="x",
            output_schema={"type": "object", "properties": {"a": {"$ref": "#/x"}}},
            model=_MODEL,
        )
    assert client.calls == []  # never reached the cluster


async def test_extract_raises_on_non_json_content() -> None:
    client = _FakeChatClient("not json at all")
    with pytest.raises(StructuredOutputError, match="non-JSON"):
        await structured.extract_structured(client, content="x", output_schema={"type": "object"}, model=_MODEL)


async def test_extract_raises_on_missing_content() -> None:
    class _Empty:
        async def chat_completions(self, model: str, messages: Any, **kwargs: Any) -> Any:
            return {"choices": []}

    with pytest.raises(StructuredOutputError):
        await structured.extract_structured(_Empty(), content="x", output_schema={"type": "object"}, model=_MODEL)


@pytest.mark.parametrize(
    "resp",
    [
        {"choices": [None]},  # malformed: non-dict choice
        {"choices": [{"message": None}]},  # malformed: null message
        {"choices": [{"message": {"content": None}}]},  # malformed: null content
        {"choices": ["oops"]},  # malformed: string choice
    ],
)
async def test_extract_raises_typed_error_on_malformed_response(resp: Any) -> None:
    class _Malformed:
        async def chat_completions(self, model: str, messages: Any, **kwargs: Any) -> Any:
            return resp

    # Must raise the typed error, never a raw AttributeError/TypeError.
    with pytest.raises(StructuredOutputError):
        await structured.extract_structured(_Malformed(), content="x", output_schema={"type": "object"}, model=_MODEL)


# ── generate_structured ─────────────────────────────────────────────────


async def test_generate_passes_response_format_through() -> None:
    client = _FakeChatClient("yes")
    rf = {"type": "regex", "regex": "(yes|no)"}

    content = await structured.generate_structured(client, prompt="Is the sky blue?", response_format=rf, model=_MODEL)

    assert content == "yes"
    call = client.calls[0]
    assert call["response_format"] is rf
    assert call["messages"] == [{"role": "user", "content": "Is the sky blue?"}]


async def test_generate_validates_json_schema_subset() -> None:
    client = _FakeChatClient("{}")
    rf = {
        "type": "json_schema",
        "json_schema": {"name": "x", "schema": {"type": "object", "if": {}}},
    }
    with pytest.raises(StructuredOutputError, match="if"):
        await structured.generate_structured(client, prompt="p", response_format=rf, model=_MODEL)
    assert client.calls == []


async def test_generate_rejects_ebnf_on_outlines_model() -> None:
    client = _FakeChatClient("x")
    rf = {"type": "grammar", "grammar": 'root ::= "yes" | "no"', "syntax": "ebnf"}
    with pytest.raises(StructuredOutputError, match="EBNF"):
        await structured.generate_structured(client, prompt="p", response_format=rf, model=_MODEL)
    assert client.calls == []


async def test_generate_rejects_ebnf_on_instruct_4b_outlines_model() -> None:
    # The other 4B generation profile on the test cluster is Outlines-backed too.
    client = _FakeChatClient("x")
    rf = {"type": "grammar", "grammar": 'root ::= "yes" | "no"', "syntax": "ebnf"}
    with pytest.raises(StructuredOutputError, match="EBNF"):
        await structured.generate_structured(
            client, prompt="p", response_format=rf, model="Qwen/Qwen3-4B-Instruct-2507"
        )
    assert client.calls == []


async def test_generate_allows_grammar_on_non_outlines_model() -> None:
    client = _FakeChatClient("yes")
    rf = {"type": "grammar", "grammar": 'root ::= "yes"', "syntax": "ebnf"}
    # An xgrammar-backed model is not in the Outlines set → passes through.
    content = await structured.generate_structured(client, prompt="p", response_format=rf, model="some/xgrammar-model")
    assert content == "yes"
    assert client.calls[0]["response_format"] is rf


async def test_generate_rejects_json_schema_missing_schema() -> None:
    client = _FakeChatClient("{}")
    rf = {"type": "json_schema", "json_schema": {"name": "x"}}
    with pytest.raises(StructuredOutputError, match="schema is required"):
        await structured.generate_structured(client, prompt="p", response_format=rf, model=_MODEL)


async def test_generate_rejects_non_dict_json_schema() -> None:
    # Caller-supplied json_schema that is a string (not an object) must raise a
    # typed error, not a raw AttributeError.
    client = _FakeChatClient("{}")
    rf = {"type": "json_schema", "json_schema": "oops"}
    with pytest.raises(StructuredOutputError, match="schema is required"):
        await structured.generate_structured(client, prompt="p", response_format=rf, model=_MODEL)
    assert client.calls == []


async def test_generate_rejects_non_object_response_format() -> None:
    client = _FakeChatClient("{}")
    with pytest.raises(StructuredOutputError):
        await structured.generate_structured(
            client,
            prompt="p",
            response_format="json",
            model=_MODEL,  # type: ignore[arg-type]
        )


async def test_generate_rejects_oversized_regex() -> None:
    client = _FakeChatClient("x")
    rf = {"type": "regex", "regex": "a" * (structured._MAX_REGEX_LEN + 1)}
    with pytest.raises(StructuredOutputError, match="regex length"):
        await structured.generate_structured(client, prompt="p", response_format=rf, model=_MODEL)
    assert client.calls == []  # never reached the cluster


async def test_generate_rejects_oversized_ebnf_on_non_outlines_model() -> None:
    client = _FakeChatClient("x")
    rf = {"type": "grammar", "grammar": "a" * (structured._MAX_EBNF_LEN + 1), "syntax": "ebnf"}
    with pytest.raises(StructuredOutputError, match="EBNF"):
        await structured.generate_structured(client, prompt="p", response_format=rf, model="some/xgrammar-model")
    assert client.calls == []


# ── gpu + token-budget thread-through ────────────────────────────────────


async def test_extract_threads_gpu_and_max_tokens() -> None:
    client = _FakeChatClient("{}")
    await structured.extract_structured(
        client,
        content="x",
        output_schema={"type": "object"},
        model=_MODEL,
        gpu="a100-40gb",
        max_completion_tokens=2048,
    )
    call = client.calls[0]
    assert call["gpu"] == "a100-40gb"
    assert call["max_completion_tokens"] == 2048


async def test_generate_threads_gpu_and_max_tokens() -> None:
    client = _FakeChatClient("yes")
    await structured.generate_structured(
        client,
        prompt="p",
        response_format={"type": "regex", "regex": "(yes|no)"},
        model=_MODEL,
        gpu="a100-40gb",
        max_completion_tokens=128,
    )
    call = client.calls[0]
    assert call["gpu"] == "a100-40gb"
    assert call["max_completion_tokens"] == 128


# ── post-validation of cluster output (defense in depth) ─────────────────


async def test_extract_rejects_nonconforming_output() -> None:
    # additionalProperties:False + a leaked extra key (the speculative-decoding
    # failure mode) must be caught post-hoc, not returned as schema-valid.
    client = _FakeChatClient(json.dumps({"city": "Lisbon", "leak": "x"}))
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    with pytest.raises(StructuredOutputError, match="does not conform"):
        await structured.extract_structured(client, content="x", output_schema=schema, model=_MODEL)


async def test_extract_rejects_output_missing_required_field() -> None:
    client = _FakeChatClient(json.dumps({"city": "Lisbon"}))
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
    }
    with pytest.raises(StructuredOutputError, match="does not conform"):
        await structured.extract_structured(client, content="x", output_schema=schema, model=_MODEL)


async def test_extract_accepts_conforming_output() -> None:
    client = _FakeChatClient(json.dumps({"city": "Lisbon", "country": "Portugal"}))
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
        "additionalProperties": False,
    }
    data = await structured.extract_structured(client, content="x", output_schema=schema, model=_MODEL)
    assert data == {"city": "Lisbon", "country": "Portugal"}


async def test_generate_rejects_nonconforming_json_schema_output() -> None:
    client = _FakeChatClient(json.dumps({"answer": "Paris", "leak": 1}))
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    rf = {"type": "json_schema", "json_schema": {"name": "qa", "schema": schema, "strict": True}}
    with pytest.raises(StructuredOutputError, match="does not conform"):
        await structured.generate_structured(client, prompt="p", response_format=rf, model=_MODEL)


async def test_generate_rejects_truncated_json_schema_output() -> None:
    client = _FakeChatClient('{"answer": "Par')  # truncated mid-JSON
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    rf = {"type": "json_schema", "json_schema": {"name": "qa", "schema": schema, "strict": True}}
    with pytest.raises(StructuredOutputError, match="non-JSON"):
        await structured.generate_structured(client, prompt="p", response_format=rf, model=_MODEL)
