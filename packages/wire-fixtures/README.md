# Wire-contract golden fixtures

Language-neutral golden fixtures for the shapes that cross the SIE wire and are
otherwise hand-maintained in several codebases (gateway `rs`, server `py`, SDK
`py`, SDK `ts`). Each implementation round-trips these fixtures in its own CI so
**drift is caught in CI, not production** — the parity promise becomes
executable.

## Files

- `model_state.json` — the canonical `ModelState` values (`available`,
  `loading`, `loaded`, `unloading`, `failed`).

## Adding a consumer

Point a test at the JSON and assert the implementation's enum/type matches the
fixture set. Current consumers:

- Python SDK — `packages/sie_sdk/tests/test_wire_contract.py` (asserts
  `typing.get_args(ModelState)` matches the fixture).
- TypeScript SDK — `packages/sie_ts_sdk/tests/wireContract.test.ts` (asserts the
  runtime `MODEL_STATES` array — the single source the `ModelState` type is
  derived from — matches the fixture).

## Scope

This is the first slice (issue #1637): `ModelState` only. Extend the same
fixture file (and the two tests) with `ModelCapabilities`, error codes, and
status messages as they are pinned. Codegen can come later if fixtures prove
insufficient.
