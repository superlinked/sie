"""Additive Rust output wire-format contract tests.

These tests pin the ``RawOutput`` / ``DenseOutput`` / ``ScoreOutputRaw``
shapes we just added to ``sie_server.ipc_types`` so that:

* New fields stay **additive** — older msgpack payloads without
  ``raw_output`` round-trip cleanly into the new ``ItemOutcome``
  with ``raw_output = None`` and every other field intact.
* The field ordering / defaults match the Rust side
  (``sie_server_sidecar::ipc_types``). The msgpack wire uses *named* maps
  in both directions, so this is a field-name / default contract test,
  not a byte-identity one.
* When the Python adapter starts emitting a ``RawOutput`` for dense or
  score variants, ``msgspec.msgpack`` encodes it into a shape the Rust
  publisher can decode with its `#[serde(default)]` `Option<_>` fields
  — i.e. both ends agree on which variants are present.
"""

from __future__ import annotations

import msgspec
from sie_server.ipc_types import (
    BatchedF16MultivectorItem,
    BatchedF16MultivectorOutput,
    BatchOutcome,
    DenseOutput,
    ItemOutcome,
    RawOutput,
    ScoreOutputRaw,
)


def _encode(outcome: ItemOutcome) -> bytes:
    return msgspec.msgpack.encode(BatchOutcome(outcomes=[outcome]))


def _decode(buf: bytes) -> ItemOutcome:
    return msgspec.msgpack.decode(buf, type=BatchOutcome).outcomes[0]


def test_item_outcome_roundtrips_without_raw_output() -> None:
    """An outcome without ``raw_output`` must keep round-tripping
    byte-for-byte through the new schema — this is the rolling-deploy
    contract.
    """
    legacy = ItemOutcome(
        work_item_id="wi-1",
        request_id="req-1",
        item_index=0,
        disposition="publish_and_ack",
        result_msgpack=b"\x80",
        inference_ms=12.5,
    )
    out = _decode(_encode(legacy))
    assert out.raw_output is None
    assert out.result_msgpack == b"\x80"
    assert out.disposition == "publish_and_ack"
    assert out.inference_ms == 12.5


def test_raw_output_dense_variant_roundtrips() -> None:
    rt = _decode(
        _encode(
            ItemOutcome(
                work_item_id="wi-dense",
                request_id="req-dense",
                item_index=3,
                disposition="publish_and_ack",
                raw_output=RawOutput(
                    dense=DenseOutput(
                        values=[0.25, -0.5, 0.75, 1.0],
                        dim=4,
                        normalize=True,
                    ),
                ),
            ),
        ),
    )
    assert rt.result_msgpack is None
    assert rt.raw_output is not None
    assert rt.raw_output.score is None
    dense = rt.raw_output.dense
    assert dense is not None
    assert dense.dim == 4
    assert dense.normalize is True
    assert dense.values == [0.25, -0.5, 0.75, 1.0]


def test_raw_output_score_variant_roundtrips_and_preserves_order() -> None:
    rt = _decode(
        _encode(
            ItemOutcome(
                work_item_id="wi-score",
                request_id="req-score",
                item_index=0,
                disposition="publish_and_ack",
                raw_output=RawOutput(
                    score=ScoreOutputRaw(
                        scores=[0.1, 0.9, 0.5],
                        item_ids=["doc-a", "doc-b", "doc-c"],
                    ),
                ),
            ),
        ),
    )
    assert rt.raw_output is not None
    assert rt.raw_output.dense is None
    score = rt.raw_output.score
    assert score is not None
    # Rust does the sort; the Python side MUST send parallel lists
    # in input order.
    assert score.scores == [0.1, 0.9, 0.5]
    assert score.item_ids == ["doc-a", "doc-b", "doc-c"]


def test_raw_output_defaults_match_rust_side() -> None:
    """All inner variants default to ``None`` (mirrors
    ``#[serde(default)] Option<_>`` on the Rust side).
    """
    empty = RawOutput()
    assert empty.dense is None
    assert empty.score is None
    dense_only = RawOutput(dense=DenseOutput(values=[1.0], dim=1))
    assert dense_only.dense is not None
    assert dense_only.dense.normalize is False
    assert dense_only.score is None


def test_unknown_raw_output_variant_is_tolerated() -> None:
    """Forward-compat: a future Python build emits a variant the current
    Rust build doesn't know. ``msgspec`` ignores unknown fields by default
    so the outcome still decodes, with the known variants preserved.

    We simulate a truly-unknown variant (``"extract_json"`` — reserved
    for future typed outputs) alongside a known one. As new variants land in
    this file (dense / score / sparse / multivector today), pick any
    field name not yet defined on ``RawOutput`` to keep the test
    meaningful.
    """
    encoded = msgspec.msgpack.encode(
        {
            "outcomes": [
                {
                    "work_item_id": "wi",
                    "request_id": "req",
                    "item_index": 0,
                    "disposition": "publish_and_ack",
                    "raw_output": {
                        "dense": {"values": [0.5, 0.5], "dim": 2, "normalize": False},
                        # Hypothetical future typed-output variant.
                        "extract_json": {"doc_id": "doc-1", "scores": [0.1, 0.2]},
                    },
                },
            ],
        },
    )
    rt = msgspec.msgpack.decode(encoded, type=BatchOutcome).outcomes[0]
    assert rt.raw_output is not None
    assert rt.raw_output.dense is not None
    assert rt.raw_output.dense.dim == 2
    # Unknown variant dropped silently — this is the forward-compat
    # guarantee. Rolling deploys work in either order because neither
    # side has to know about the other's future fields.
    assert not hasattr(rt.raw_output, "extract_json")


def test_batched_f16_multivector_roundtrips_as_one_byte_buffer() -> None:
    batch = BatchOutcome(
        outcomes=[
            ItemOutcome(
                work_item_id="wi-0",
                request_id="req-0",
                item_index=0,
                disposition="publish_and_ack",
            ),
            ItemOutcome(
                work_item_id="wi-1",
                request_id="req-1",
                item_index=1,
                disposition="publish_and_ack",
            ),
        ],
        batched_f16_multivectors=[
            BatchedF16MultivectorOutput(
                values_f16=b"\x00<\x00@\x00B",
                items=[
                    BatchedF16MultivectorItem(
                        work_item_id="wi-0",
                        byte_offset=0,
                        byte_len=4,
                        num_tokens=1,
                        token_dims=2,
                    ),
                    BatchedF16MultivectorItem(
                        work_item_id="wi-1",
                        byte_offset=4,
                        byte_len=2,
                        num_tokens=1,
                        token_dims=1,
                    ),
                ],
            )
        ],
    )

    rt = msgspec.msgpack.decode(msgspec.msgpack.encode(batch), type=BatchOutcome)

    assert len(rt.batched_f16_multivectors) == 1
    shared = rt.batched_f16_multivectors[0]
    assert shared.values_f16 == b"\x00<\x00@\x00B"
    assert [item.byte_offset for item in shared.items] == [0, 4]
    assert [item.byte_len for item in shared.items] == [4, 2]
