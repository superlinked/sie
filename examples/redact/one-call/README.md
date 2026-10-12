# Mask one complete document with `superlinked/pii-redact`

This caller sends one complete text item through `sie_sdk.SIEClient.extract`,
then masks the union of the returned intervals. The served model composes two
detectors internally, including score filtering, overlapping server windows and
name propagation. The caller only validates offsets and applies masks.

The example includes a complete synthetic trade confirmation, index `240`, uid
`gretel-en-34966`. It contains `Alejandra Santamaría-Castilla`, so the input itself
exercises non-ASCII offsets. [input.txt](input.txt) is copied unchanged: **962
Unicode code points, 963 UTF-8 bytes**, SHA-256
`94ec9f7160c81390866958d9b33c2c5a9e98f854782eafc938f4e57418adfc62`.
No newline, normalization, stripping or caller windows are added.

## Run from the repository root

Use the root frozen Python workspace and its current SDK, version `0.10.0`:

```sh
mise exec -- uv run --frozen --project . --package sie-sdk python examples/redact/one-call/run.py --show
mise exec -- uv run --frozen --project . --package sie-sdk python examples/redact/one-call/run.py --saved-composition
mise exec -- uv run --frozen --project . --package sie-sdk python examples/redact/one-call/run.py --url "$SIE_URL"
```

`--show` prints the logical request and dispatches nothing. `--saved-composition`
renders the included reference spans offline. Both modes also work with plain
`python3` and the standard library.

For a live request, supply a Cloud or self-hosted URL through `--url` or `SIE_URL`.
If the endpoint needs authentication, set `SIE_API_KEY`; the caller passes it
explicitly to the SDK for that chosen endpoint. `--no-auth` suppresses it. Keys
and authentication headers are never printed. URLs must have no embedded
credentials, query or fragment. The example does not choose an endpoint for you.

The request contains the whole input and `options={"profile": "default"}`;
it omits `labels` to use the server's configured 36 defaults. The SDK sends
MessagePack to `/v1/extract/superlinked/pii-redact`. The JSON from `--show` is a
logical-body display, not the actual wire encoding.

Ordinary mode makes one SDK extract operation and keeps normal SDK handling of
a cold deployment and transient capacity failures; that operation can send more
than one HTTP request. For an explicit illustration requiring **at most one
extract POST**, add `--single-attempt`. It disables transport retries, blocks a
second POST before transport, and stops on the first HTTP error before SDK retry
logic runs. A loading response therefore ends that attempt. Supported SDK
same-origin GET completion polling can still retrieve the submitted result.
`--timeout-seconds` sets the SDK provision budget and HTTP phase timeouts
(default 900 seconds); HTTP phase timeouts are not a hard whole-call deadline.

## Read the output

Output includes the evidence type, returned entity spans, merged intervals and
masked text. Integer `start` and `end` must satisfy
`0 <= start < end <= len(text)`, and each entity's text must equal exactly
`text[start:end]`. A mismatch fails; the caller never searches for another
occurrence or changes the input to repair offsets.

Python string offsets count Unicode code points. UTF-8 bytes, UTF-16 code units
and displayed graphemes can have different lengths: an astral character occupies
one Python offset, while a base letter followed by a combining mark occupies
two. The tests exercise these cases alongside the input's accented name.

Duplicate, nested, partial-overlap and touching intervals merge before masking.
Each covered code point becomes one `█`, preserving the original Python string
length and every unmasked substring. The caller does not filter scores or
recreate the model composition. Returned errors, malformed wire responses,
missing entity lists, invalid spans and uncertain transport outcomes exit
nonzero without masked output. A genuine empty list says **no spans returned**;
it does not establish that the text is free of personal data.

## Source and evidence

[source.json](source.json) records the complete request, input identity, the full
36-label list and expected configuration from public SIE source
`9f9eaa3d74684c081fce45386ec91ac137f3ac1a`:

- [Model configuration](https://github.com/superlinked/sie/blob/9f9eaa3d74684c081fce45386ec91ac137f3ac1a/packages/sie_server/models/superlinked__pii-redact.yaml),
  SHA-256 `7637549f4036cc91dc342fcfd19abb8218cb444673b5091968f80f993ea06f57`.
- `urchade/gliner_multi_pii-v1` at `1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d` and
  `numind/NuNER_Zero` at `c90187673f464518dca09f41689184ed6976242c`, with
  `merge_adjacent_entities=true` for NuNER.
- Tokenizer dependencies `microsoft/mdeberta-v3-base` at
  `a0484667b22365f84929a935b5e50a51f71f159d` and
  `microsoft/deberta-v3-large` at `64a8c8eab3e352a784c658aef62be1662607476f`.
- The default profile runs both members with score floor `0.6`, server windows
  of 300 GLiNER words and overlap 50, and case-sensitive whole-word propagation
  of `person` tokens containing at least three letters. GLiNER's word counting
  includes punctuation; it differs from `text.split()`.

The adapter retains overlaps and differently labelled spans, deduplicating
identical `(start, end, label)` spans at the maximum score. It returns original
document offsets. Usage counts each detector and each overlapping server window.
These pins describe the expected source configuration; they do not prove what
an arbitrary URL serves. Deployment, installed runtime, device, checkpoint-load
receipts and returned revision metadata remain unknown until observed. A returned
configuration digest is not a checkpoint revision or proof of device parity.

The complete text and reference spans come from the immutable
[public fixture](https://github.com/superlinked/sie/blob/9f9eaa3d74684c081fce45386ec91ac137f3ac1a/packages/sie_server/tests/adapters/fixtures/gliner_union/redact-documents.json),
whose SHA-256 is
`27a46485552385b096933d05eb4e96722e0914becae5261645c2ad9daea42677`.
The retained input is in
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/31383ae9986d9dc9edd4e3a06ee9fb40b223d6fb/redact/inputs/gretel-main.jsonl)
at `31383ae9986d9dc9edd4e3a06ee9fb40b223d6fb`, derived from
[Gretel's synthetic financial dataset](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual),
attributed as Apache 2.0 in the [recorded study](../README.md).

**No real one-call HTTP exchange is retained here.** The fixture's `expected`
spans compose two saved model responses from that study; propagated spans can
have `score: null`. Offline output explicitly says
`saved composition replay; not a one-call HTTP recording`. It retains overlapping
phone/email spans and unnecessary masks, including the car description and
placeholder text. It illustrates masking and provenance, not a new inference,
accuracy estimate or comparison. A live run needs a reachable endpoint serving
the model; a local CPU server also needs both pinned checkpoints and tokenizers.

## Check without credentials or weights

```sh
mise exec -- uv run --frozen --project . --package sie-sdk pytest examples/redact/one-call/test_one_call.py
```

The tests use the real SDK with HTTPX MockTransport to inspect its MessagePack
request, optional authentication, cold-deployment handling, strict failure
behavior and validation before the SDK can default a missing entities field to
an empty list. No model is loaded and no inference request is sent.
