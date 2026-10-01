"""The Python SDK's wire types must match the shared golden fixtures.

Round-trips ``packages/wire-fixtures/model_state.json`` against the SDK's typed
``ModelState`` so drift (for example a state added on one side only) fails in CI
rather than shipping. See ``packages/wire-fixtures/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args, get_type_hints

from sie_sdk.client._shared import (
    FALLBACK_ERROR_HEADER,
    FALLBACK_ERROR_PATTERN,
    FALLBACK_REASON_HEADER,
    FALLBACK_REASONS,
    REMOTE_HEADER,
    REQUEST_USAGE_HEADERS,
    SERVED_BY_HEADER,
    SERVED_BY_VALUES,
    UPSTREAM_HEADER,
    UPSTREAM_NAME_PATTERN,
    parse_extract_results,
)
from sie_sdk.types import (
    DECLARED_USAGE_FIELDS,
    SETTLED_CHARGE_FIELDS,
    TERMINAL_UNIT_FIELDS,
    ModelInfo,
    ModelState,
    RequestMetadata,
    RequestUsage,
)

_FIXTURES = Path(__file__).parents[2] / "wire-fixtures"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


def test_model_state_matches_golden_fixture() -> None:
    fixture = _load("model_state.json")
    assert set(get_args(ModelState)) == set(fixture["model_states"])


def test_request_usage_partition_matches_golden_fixture() -> None:
    """The `usage` partition is one contract declared in three codebases (#3063).

    The Cloud gateway asserts the same fixture over the fields it injects, so a
    gateway that starts publishing a third settled field turns this red until
    the SDK declares it — which is what stops a consumer from silently nulling
    its meter readings on the managed path.
    """
    fixture = _load("request_usage.json")
    assert set(fixture["terminal_unit_fields"]) == TERMINAL_UNIT_FIELDS
    assert set(fixture["settled_charge_fields"]) == SETTLED_CHARGE_FIELDS
    assert fixture["usage_container"] == "usage"


def test_request_usage_partition_covers_the_whole_block() -> None:
    """No `usage` field may belong to neither half, or to both.

    A field in neither half is invisible to every consumer that partitions the
    block; a field in both would be metered AND treated as an annotation.
    """
    assert TERMINAL_UNIT_FIELDS.isdisjoint(SETTLED_CHARGE_FIELDS)
    assert DECLARED_USAGE_FIELDS == TERMINAL_UNIT_FIELDS | SETTLED_CHARGE_FIELDS
    assert set(RequestUsage.__annotations__) == DECLARED_USAGE_FIELDS


def test_every_terminal_unit_has_a_meter_header() -> None:
    """The metered half is exactly what the gateway reports in headers."""
    assert set(REQUEST_USAGE_HEADERS) == TERMINAL_UNIT_FIELDS


def test_model_info_declares_every_typed_wire_field() -> None:
    """``ModelInfo`` must declare each key the gateway emits.

    An emitted-but-undeclared field is invisible to type checkers, so callers
    can't reach it without a cast — the drift that left ``state``,
    ``last_error``, ``profiles`` and ``pending_generation`` unusable.
    """
    fixture = _load("model_info.json")
    assert set(ModelInfo.__annotations__) == set(fixture["typed"])


def test_model_info_omits_deliberately_excluded_wire_fields() -> None:
    """OpenAI-compat keys must stay out of the SIE ``ModelInfo`` type.

    ``GET /v1/models/{model}`` merges ``id``/``object``/``created``/``owned_by``
    in for vanilla OpenAI clients. Declaring them would advertise envelope
    scaffolding as SIE model metadata; the fixture records the reason per key.
    """
    fixture = _load("model_info.json")
    excluded = set(fixture["excluded"])
    assert excluded, "fixture must record why each excluded key is excluded"
    assert set(ModelInfo.__annotations__).isdisjoint(excluded)


def test_extract_parser_preserves_data_and_item_error() -> None:
    [result] = parse_extract_results(
        [
            {
                "id": "page-1",
                "entities": [],
                "data": {"processed_pages": 3},
                "error": {
                    "code": "INFERENCE_ERROR",
                    "message": "Document export failed",
                },
            }
        ]
    )

    assert result["data"] == {"processed_pages": 3}
    assert result["error"] == {
        "code": "INFERENCE_ERROR",
        "message": "Document export failed",
    }


def test_extract_parser_preserves_malformed_item_failures() -> None:
    results = parse_extract_results(
        [
            {"entities": [], "error": "not-an-error-object"},
            {"entities": [], "error": {"code": "INFERENCE_ERROR"}},
            {"entities": [], "error": {"code": " ", "message": "\t"}},
        ]
    )

    assert [result["error"] for result in results] == [
        {"code": "INTERNAL_ERROR", "message": "Malformed extraction item error"},
        {"code": "INTERNAL_ERROR", "message": "Malformed extraction item error"},
        {"code": "INTERNAL_ERROR", "message": "Malformed extraction item error"},
    ]


def test_serving_disclosure_matches_golden_fixture() -> None:
    """The remote-serving headers are one contract shared by the server and both SDKs."""
    fixture = _load("serving_disclosure.json")
    headers = fixture["response_headers"]
    metadata = get_type_hints(RequestMetadata)

    assert fixture["request_header"] == {"name": REMOTE_HEADER, "values": ["forbid"]}
    assert {key: header["name"] for key, header in headers.items()} == {
        "served_by": SERVED_BY_HEADER,
        "upstream": UPSTREAM_HEADER,
        "fallback_reason": FALLBACK_REASON_HEADER,
        "fallback_error": FALLBACK_ERROR_HEADER,
    }
    assert set(headers) <= set(metadata)
    assert set(headers["served_by"]["values"]) == SERVED_BY_VALUES == set(get_args(metadata["served_by"]))
    assert set(headers["fallback_reason"]["values"]) == FALLBACK_REASONS == set(get_args(metadata["fallback_reason"]))
    assert headers["upstream"]["pattern"] == UPSTREAM_NAME_PATTERN.pattern
    assert headers["fallback_error"]["pattern"] == FALLBACK_ERROR_PATTERN.pattern
