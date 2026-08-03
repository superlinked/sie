"""Pure jobs helpers: build_job_body slot mapping + result decoding (no transport)."""

from __future__ import annotations

import msgpack
import numpy as np
import pytest
from sie_sdk import build_job_body, connection_name
from sie_sdk.jobs import decode_chunk_bytes, decode_result_item, job_chunks


def test_inline_list_source_becomes_items() -> None:
    body = build_job_body(source=["a", "b"], operation="encode", model="BAAI/bge-m3")
    assert body == {"operation": "encode", "model": "BAAI/bge-m3", "items": [{"text": "a"}, {"text": "b"}]}


def test_bare_string_source_is_one_text_item() -> None:
    body = build_job_body(source="embed me", operation="encode", model="m")
    assert body["items"] == [{"text": "embed me"}]
    assert "src" not in body


def test_connector_uri_source_derives_connection() -> None:
    body = build_job_body(
        source="postgres://warehouse?query=SELECT+1",
        operation="encode",
        model="m",
        sink="postgres://warehouse?table=vecs",
    )
    assert body["src"] == "postgres://warehouse?query=SELECT+1"
    assert body["connection"] == "warehouse"
    assert body["sink"] == "postgres://warehouse?table=vecs"
    # Same-store sink reuses the source connection — no redundant sink_connection.
    assert "sink_connection" not in body


def test_distinct_sink_threads_sink_connection() -> None:
    body = build_job_body(
        source="postgres://wh?query=x",
        operation="encode",
        model="m",
        sink="s3://out-bucket/vecs",
    )
    assert body["sink_connection"] == "out-bucket"


def test_explicit_connection_overrides_derived() -> None:
    body = build_job_body(
        source="postgres://wh?query=x",
        operation="encode",
        model="m",
        sink="postgres://wh?table=t",
        connection="prod-wh",
        sink_connection="prod-wh",
    )
    assert body["connection"] == "prod-wh"
    assert body["sink_connection"] == "prod-wh"


def test_inplace_sink_sentinel() -> None:
    body = build_job_body(source="postgres://wh?query=x", operation="encode", model="m", sink="inplace")
    assert body["sink"] == "inplace"


def test_return_sink_is_omitted() -> None:
    assert "sink" not in build_job_body(source=["a"], operation="encode", model="m", sink="return")
    assert "sink" not in build_job_body(source=["a"], operation="encode", model="m", sink=None)


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        ("now", {}),
        (None, {}),
        ("schedule:*/5 * * * *", {"when": "schedule", "schedule": "*/5 * * * *"}),
        ("*/5 * * * *", {"when": "schedule", "schedule": "*/5 * * * *"}),
        ("watch:s3://in", {"when": "watch", "watch": "s3://in"}),
    ],
)
def test_when_trigger_mapping(when: str | None, expected: dict[str, str]) -> None:
    body = build_job_body(source=["a"], operation="encode", model="m", when=when)
    for key, value in expected.items():
        assert body[key] == value
    if not expected:
        assert "when" not in body


def test_output_types_ride_only_when_set() -> None:
    assert build_job_body(source=["a"], operation="encode", model="m", output_types=["dense"])["output_types"] == [
        "dense"
    ]
    assert "output_types" not in build_job_body(source=["a"], operation="encode", model="m")


def test_options_forward_op_inputs_as_is() -> None:
    """Op inputs ride `options` (JOBS.md matrix): score query, extract labels/schema, generate sampling."""
    # score (connector form): options.query rides the wire untouched.
    score = build_job_body(
        source="postgres://wh?query=x",
        operation="score",
        model="BAAI/bge-m3",
        sink="postgres://wh?table=scores",
        options={"query": "rank these documents"},
    )
    assert score["options"] == {"query": "rank these documents"}

    # extract (inline form): labels + output_schema forwarded as-is.
    extract = build_job_body(
        source=["some text"],
        operation="extract",
        model="urchade/gliner_small-v2.1",
        options={"labels": ["PERSON", "ORG"], "output_schema": {"type": "object"}},
    )
    assert extract["options"] == {"labels": ["PERSON", "ORG"], "output_schema": {"type": "object"}}

    # generate sampling forwarded as-is.
    generate = build_job_body(
        source="postgres://wh?query=x",
        operation="generate",
        model="Qwen/Qwen3-4B-Instruct-2507",
        sink="postgres://wh?table=docs&mode=inplace&column=summary",
        options={"max_new_tokens": 64, "temperature": 0.0},
    )
    assert generate["options"] == {"max_new_tokens": 64, "temperature": 0.0}


def test_options_stay_off_the_wire_when_unset_or_empty() -> None:
    # Absent or empty options never ride (the shipped body stays byte-identical).
    assert "options" not in build_job_body(source=["a"], operation="encode", model="m")
    assert "options" not in build_job_body(source=["a"], operation="encode", model="m", options={})


def test_non_mapping_options_raises() -> None:
    with pytest.raises(ValueError, match="options must be a mapping"):
        build_job_body(source=["a"], operation="score", model="m", options=["query"])  # ty: ignore[invalid-argument-type]


def test_empty_inline_source_raises() -> None:
    with pytest.raises(ValueError, match="no items"):
        build_job_body(source=[], operation="encode", model="m")


# ---- §4.5.4 field_map / output_field ----------------------------------------


def test_field_map_and_output_field_ride_connector_jobs() -> None:
    # The §4.5.4 before/after example, byte-shaped for the wire.
    body = build_job_body(
        source="postgres://wh?query=select id, body, source_url from docs",
        operation="encode",
        model="BAAI/bge-m3",
        field_map={"id_field": "id", "input_field": "body", "carry": ["source_url"], "input_type": "text"},
        sink="postgres://wh?table=doc_vectors",
        output_field="embedding",
    )
    assert body["field_map"] == {
        "id_field": "id",
        "input_field": "body",
        "carry": ["source_url"],
        "input_type": "text",
    }
    assert body["output_field"] == "embedding"


def test_field_map_omitted_keys_do_not_ride() -> None:
    body = build_job_body(
        source="postgres://wh?query=x",
        operation="encode",
        model="m",
        sink="postgres://wh?table=t",
        field_map={"id_field": "id"},
    )
    assert body["field_map"] == {"id_field": "id"}
    assert "output_field" not in body
    # An all-empty field_map rides nothing (additive-only wire).
    body = build_job_body(
        source="postgres://wh?query=x", operation="encode", model="m", sink="postgres://wh?table=t", field_map={}
    )
    assert "field_map" not in body


@pytest.mark.parametrize(
    ("field_map", "match"),
    [
        ({"id_column": "id"}, "unknown field_map key"),  # alias params are URI-side, not field_map keys
        ({"carry": "source_url"}, "carry"),  # a bare string is not a list of fields
        ({"input_type": "rows"}, "input_type"),
    ],
)
def test_bad_field_map_raises(field_map: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_job_body(
            source="postgres://wh?query=x",
            operation="encode",
            model="m",
            sink="postgres://wh?table=t",
            field_map=field_map,
        )


def test_field_map_on_inline_items_raises() -> None:
    with pytest.raises(ValueError, match="connector-src"):
        build_job_body(source=["a"], operation="encode", model="m", field_map={"id_field": "id"})
    with pytest.raises(ValueError, match="connector-src"):
        build_job_body(source=["a"], operation="encode", model="m", output_field="embedding")


# ---- §4.5.4 upload:// (internal scheme — no connection derived) ----------------


def test_upload_source_derives_no_connection() -> None:
    body = build_job_body(
        source="upload://file-abc?format=csv",
        operation="encode",
        model="m",
        sink="upload://file-out",
        field_map={"id_field": "doc_id", "input_field": "text", "input_type": "text"},
    )
    assert body["src"] == "upload://file-abc?format=csv"
    assert body["sink"] == "upload://file-out"
    assert "connection" not in body
    assert "sink_connection" not in body


def test_upload_source_to_external_sink_threads_sink_connection() -> None:
    body = build_job_body(
        source="upload://file-abc",
        operation="encode",
        model="m",
        sink="postgres://wh?table=doc_vectors",
        sink_connection="wh",
    )
    assert "connection" not in body
    assert body["sink_connection"] == "wh"


def test_bad_sink_raises() -> None:
    with pytest.raises(ValueError, match="sink must be"):
        build_job_body(source=["a"], operation="encode", model="m", sink="garbage")


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("postgres://warehouse?query=x", "warehouse"),
        ("s3://customer-bucket/in/", "customer-bucket"),
        ("gs://my-bucket", "my-bucket"),
    ],
)
def test_connection_name(uri: str, expected: str) -> None:
    assert connection_name(uri) == expected


def _chunk_bytes(n: int, dims: int) -> bytes:
    items = [
        {
            "success": True,
            "id": str(i),
            "units": {"input_tokens": 5},
            "result_msgpack": msgpack.packb(
                {"dense": {"dims": dims, "values": [0.1 * j for j in range(dims)]}}, use_bin_type=True
            ),
        }
        for i in range(n)
    ]
    return msgpack.packb(items, use_bin_type=True)


def test_decode_result_item_yields_numpy_dense() -> None:
    raw = _chunk_bytes(n=1, dims=4)
    (item,) = msgpack.unpackb(raw, raw=False)
    decoded = decode_result_item(item)
    assert decoded["id"] == "0"
    assert decoded["success"] is True
    assert decoded["dims"] == 4
    assert isinstance(decoded["dense"], np.ndarray)
    assert decoded["dense"].shape == (4,)


def test_decode_chunk_bytes_all_items() -> None:
    decoded = decode_chunk_bytes(_chunk_bytes(n=3, dims=8))
    assert len(decoded) == 3
    assert all(d["dims"] == 8 for d in decoded)


def test_job_chunks_extracts_ref_metadata() -> None:
    """#2434: a settled chunk carries its charge under the unified name."""
    job = {
        "output": {
            "kind": "refs",
            "chunks": [
                {
                    "seq": 0,
                    "items": 3,
                    "state": "succeeded",
                    "ref": "payload-store/c0",
                    "units": 15,
                    "credits_charged": 15,
                    "rate_book_version": "book-v1",
                }
            ],
        }
    }
    chunks = job_chunks(job)
    assert chunks == [
        {
            "seq": 0,
            "items": 3,
            "state": "succeeded",
            "ref": "payload-store/c0",
            "units": 15,
            "credits_charged": 15,
            "rate_book_version": "book-v1",
            "error": None,
        }
    ]


def test_job_chunks_leave_an_unsettled_chunk_uncharged() -> None:
    """A chunk the control plane has not acked yet reports no charge at all."""
    job = {
        "output": {
            "kind": "refs",
            "chunks": [{"seq": 1, "items": 3, "state": "spawned", "ref": None, "units": None}],
        }
    }
    (chunk,) = job_chunks(job)
    assert chunk["credits_charged"] is None
    assert chunk["rate_book_version"] is None
