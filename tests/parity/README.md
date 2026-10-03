# RunBatch IPC Fixtures

This directory holds JSON fixtures for the `RunBatch` IPC contract between the
`worker-sidecar` container (`sie-server-sidecar` binary) and the Python
`sie_server` adapter process.

The fixtures pin the small set of fields that must stay stable across
implementations:

- work item and request identifiers
- item indexes and outcome ordering
- publish/ACK/NAK disposition
- error codes
- LoRA routing expectations

Timing fields and raw inference bytes are intentionally excluded from the
canonical comparison. Each fixture lists those elided fields under
`notes.elided_fields`.

## Running

```bash
tests/parity/run_parity.sh
tests/parity/run_parity.sh -v
```

Today this script runs the Python-side fixture consumer. The same JSON shape is
kept language-neutral so Rust-side coverage can consume it without rewriting the
fixtures.

## Fixtures

| Fixture | Pins |
| --- | --- |
| `run_batch_empty.json` | Empty batch returns an empty outcome without handler calls |
| `run_batch_encode_no_lora.json` | Encode base path, empty `lora_key` |
| `run_batch_encode_lora.json` | Encode LoRA plumbing plus invalid-before-valid ordering |
| `run_batch_extract_lora.json` | Extract LoRA plumbing plus invalid-before-valid ordering |
| `run_batch_mixed_op.json` | Mixed op rejection |
| `run_batch_score_basic.json` | Score dispatch path |
| `run_batch_score_lora_warns.json` | Score drops non-empty `lora_key` and serves base |
| `run_batch_unknown_op.json` | Unknown op rejection |

## Adding A Fixture

1. Add a `run_batch_*.json` file here.
2. Add the filename to `PARITY_FIXTURES` in
   `packages/sie_server/tests/test_parity_run_batch.py`.
3. Run `tests/parity/run_parity.sh`.
