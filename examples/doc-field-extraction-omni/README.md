# Read documents into your own schema

## What this shows

SIE serves Qwen3.8 27B FP8, which reads a business document image into that
document's own JSON schema. The documents, schemas and gold JSON come from the
[OmniAI OCR Benchmark](https://huggingface.co/datasets/getomni-ai/ocr-benchmark):
bank checks, shipping invoices, inspection forms, statements and the like.
Omni's own JSON accuracy grades every reply.

The recipe puts the schema text in the system message. SIE applies
`response_format` only as a decoding grammar, so a schema sent only there
shapes the output but the model never reads its field descriptions. The
previous version of this example did exactly that, and on the same documents
it scored 75.64 against 91.03 for this recipe.

The example reruns the published study against any SIE endpoint, grades the
replies with a port of Omni's grader, and reproduces the published numbers
offline with no key, no GPU and no model call.
[superlinked.com/doc-field-extraction](https://superlinked.com/doc-field-extraction)
reports the result.

## Measured results

Mean Omni JSON accuracy × 100, over every document in the set. Unparsed,
capped and failed replies stay in the denominator and score 0.

| Model | Pilot, 100 documents | Confirmation, 546 fresh documents |
| --- | --- | --- |
| SIE Qwen3.8 27B, this recipe | 90.15 | **91.03** (95% CI 89.75 to 92.20) |
| GPT-6 Luna | 89.05 | 90.11 |
| GPT-6 Sol | 93.66 | 94.70 |
| Claude Sonnet 5.5 | 90.01 | 90.44 |
| Claude Haiku 4.5 | 64.82 | 66.00 |
| SIE Qwen3.8 27B, schema only in `response_format` | 76.14 | 75.64 |

- **Pre-registered test:** SIE against GPT-6 Luna, non-inferiority at a
  3-point margin, one-sided alpha 0.025, format-stratified paired bootstrap
  with 10,000 resamples. The difference is +0.92 points with a lower 95% bound
  of -0.51, so non-inferiority holds. Superiority is not shown. With Luna's one
  capped reply rerun at 8,192 output tokens, the difference is +0.74 (lower
  bound -0.69).
- **GPT-6 Sol** scored higher, by 3.67 points.
- **Replies:** SIE parsed 546 of 546, with none capped. Documents with every
  gold field right: SIE 219, Luna 229, Sol 255, Sonnet 5.5 266, Haiku 4.5 62.
- **Cost** at list prices on the same 546 documents: USD 13.57 per 1,000
  documents for SIE on the measured profile, 0.61 for GPT-6 Luna.

Both runs were pre-registered with the recipe frozen, and the confirmation's
546 documents do not overlap the pilot's 100. The rival numbers are their
recorded replies, paired on the same documents.

**Evidence**, on the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
each folder at an immutable revision:

- [`doc-field-extraction-native/2026-10-07`](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/f10e813f2206348e31da363da2bc90b06086d7d9/doc-field-extraction-native/2026-10-07)
  (manifest sha256 `69823e6d88f3a5dccd0602427be28b65e0fd9641ac02c17b6cf76256389fbad9`):
  the SIE replies, the readable request bodies, the per-document scores, the
  bootstrap, cost and serving logs. Its
  [README](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/f10e813f2206348e31da363da2bc90b06086d7d9/doc-field-extraction-native/2026-10-07/README.md)
  and [confirmation README](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/f10e813f2206348e31da363da2bc90b06086d7d9/doc-field-extraction-native/2026-10-07/confirm/README.md)
  give every number above.
- [`doc-field-extraction-omni/2026-10-01`](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/61e751a5f7f374cacd84ccace621f71e8a498404/doc-field-extraction-omni/2026-10-01):
  the earlier study that recorded the rival replies.
- [`doc-field-extraction-native-rerun/2026-10-07`](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/d61c5f2525f257ede1e280aa5bae5b55f5987b24/doc-field-extraction-native-rerun/2026-10-07):
  the portable rerun kit this example is adapted from.

## The recipe

`build_body` in `run.py` makes one request per document:

- **Model:** `Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec`, the measured id.
  `--model` sends another.
- **System message:** `SYSTEM + "\n\nJSON schema:\n" + json.dumps(schema, indent=2)`,
  where `SYSTEM` is "Extract data from the document image into the JSON
  schema. Copy values as they appear on the document. Use null for a field
  the document does not contain. Return only the JSON."
- **User message:** one `image_url` with a base64 data URL. The media type
  (PNG or JPEG) comes from the image bytes, not the file name.
- **`response_format`:** `json_schema`, `strict: true`, name `fields`. The
  schema is converted the same way for every model in the study: every
  property is required and every leaf may be null. Types, properties, items,
  descriptions and string enums are kept; other keywords such as `format` or
  `minimum` are dropped. `$ref` is rejected; inline the referenced schema.
- **Sampling:** `temperature` 0, `presence_penalty` 0,
  `max_completion_tokens` 8192.
- **Thinking:** off. The request carries no thinking flag, and the measured
  profile serves the model with thinking off.
- **Failures:** a request that gets no HTTP response is resent at most twice,
  as in the study. Anything else is not resent. An error, a reply with no
  content and a reply that hits the output cap all score 0.

## Measured on

These are the conditions of the published numbers, not requirements for
running the example.

- **Image:** `ghcr.io/superlinked/sie-server@sha256:a745dfd0859d5397d4d025c79fd0b656b6df369abda1d988897cbc3fb1272961`
  (`v0.9.0-cuda13-sglang-cu130`).
- **Overlay:** the `sie_server` package and model catalog from public SIE
  revision [`6cccc04d`](https://github.com/superlinked/sie/tree/6cccc04de9c6b78d3217854ad2377de36c0ffbd2),
  copied over the image because no release then shipped the profile.
- **Checkpoint:** `Qwen/Qwen3.8-27B-FP8` at `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- **Profile:** `h100-256k-batch-no-spec`: SGLang with FA3, xgrammar, BF16 KV
  cache and no speculation, up to 16 running requests, a 32,768-token output
  limit and up to 3,211,264 pixels for a single image.
- **Hardware and load:** one H100 80GB, 8 requests in flight over loopback.
- **Tokens and latency:** on the confirmation set, inputs averaged 3,710
  tokens and outputs 731. Latency was p50 10.4 s and p95 43.5 s, measured
  inside the GPU container with no gateway.

## Run it

The offline steps need only the Python standard library. Sending needs
`sie-sdk` 0.9.0 or newer, which `uv run` installs from this folder's
`pyproject.toml`; `pip install 'sie-sdk>=0.9.0'` also works. Downloads go to
`data/` (about 200 MB for the confirmation set and 50 MB for the pilot), and
runs to `runs/<set>/`.

### Reproduce the published numbers offline

```sh
python3 fetch.py --set confirm
python3 score.py --set confirm --published   # mean 91.03; 546 of 546 documents identical
python3 run.py --set confirm --dry-run       # bodies sha256 8e796736..., the published value
```

`fetch.py` downloads the pinned Omni rows and images and the published
replies, and checks every file against its sha256. `score.py --published`
regrades the published replies and compares each document's score, parse
flag, gold field count and diff counts with the published `omni_scores.json`.
`run.py --dry-run` rebuilds every request body and hashes it in the published
canonical form; `runs/confirm/bodies.show.jsonl` is then byte-identical to the
published readable bodies.

The pilot works the same way and scores 90.15. One pilot page was re-encoded
to JPEG by the study, and only Pillow 12.2.0 is known to reproduce its bytes,
so use `uv run`, which installs that version:

```sh
uv run fetch.py --set pilot
uv run score.py --set pilot --published
uv run run.py --set pilot --dry-run
```

### Rerun against your own SIE server

Serve the measured profile from a checkout of this repository with GPU
dependencies, then send the confirmation set:

```sh
mise run serve -- -p 8080 -d cuda:0 \
  -m 'Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec' \
  --preload 'Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec'
uv run run.py --set confirm --base-url http://127.0.0.1:8080
```

### Rerun against SIE Cloud or another remote endpoint

Put the key in `SIE_API_KEY`, or name another variable with `--api-key-env`,
and pass a model id the deployment serves:

```sh
export SIE_API_KEY=...
uv run run.py --set confirm --base-url https://api.superlinked.com \
  --model '<a model id your deployment serves>'
```

- **Profile size:** Qwen3.8 27B's bare route has an 8,192-token window and a
  4,096-token output limit, which is too small for this recipe. Use a profile
  with a larger window.
- **Another model or profile** makes a new measurement, not a rerun of the
  published one. `--concurrency` (default 8) changes latency, not the requests.
- **Readiness:** an unscored text-only request goes first;
  `--no-readiness` skips it. `--no-wait-for-capacity` fails at once instead
  of waiting while the endpoint provisions capacity.

A rerun on other hardware or software can differ from the published replies;
the published replies are the reference. Each run writes:

| File | Contents |
| --- | --- |
| `dry_run.json` | The hash check of the request bodies |
| `bodies.show.jsonl` | The readable request bodies |
| `outcomes.jsonl` | Every raw reply and error |
| `replies.jsonl` | One grader row per document |
| `scores.json` | Per-document scores |
| `summary.json` | Mean, parse and cap counts, tokens and latency |

To grade replies from anywhere else, write one `{"id", "text"}` row per
document (add `"error"` for a failed one) and run
`python3 score.py --set confirm --replies replies.jsonl`.

### Read your own document

Write a schema for a PNG or JPEG, for example:

```json
{
  "type": "object",
  "properties": {
    "document_number": {"type": "string", "description": "The number printed after 'Invoice No.'"},
    "total": {"type": "number", "description": "The amount due, including tax"}
  }
}
```

Inspect the request offline, then send it:

```sh
python3 run.py --image form.png --schema fields.json --dry-run
uv run run.py --image form.png --schema fields.json --base-url http://127.0.0.1:8080
```

A new document is a trial, not part of the measured study.

### Check the send path without a GPU

`replay_server.py` stands in for `/v1/chat/completions`. It checks every body
that `SIEClient` puts on the wire against the published readable body and
answers with the published reply:

```sh
python3 replay_server.py --set confirm --port 8099 &
uv run run.py --set confirm --base-url http://127.0.0.1:8099 --out runs/replay-confirm
curl -s http://127.0.0.1:8099/replay-stats   # body_matches_published: 546
```

The replayed run scores 91.03, and its `replies.jsonl` is byte-identical to
the published one.

## Scoring

`score.py` computes, per document, the score of the grader the study ran:

1. **Null fill.** A key that the document's Omni schema declares, but that the
   JSON leaves out, is added with the value null, in the gold and in every
   reply alike. Strict-schema APIs return null for an absent field while
   others drop the key. Nothing else is changed.
2. **Omni JSON accuracy.** 1 minus (additions + deletions + modifications)
   divided by the gold field count, floored at 0 and rounded to 4 decimals.
   The diff is json-diff with `sort: true`, and case is kept.
3. **Failures.** A reply that is not a JSON object or array, or a row marked
   as an error, scores 0.

The original grader is JavaScript. `score.py` reimplements it in the Python
standard library, reproducing JavaScript number parsing, key order, equality
and sort order wherever they change a score. On the published replies it
matches every per-document record of both sets.

## Source licenses

- **Documents, schemas and gold:** [OmniAI OCR Benchmark](https://huggingface.co/datasets/getomni-ai/ocr-benchmark),
  MIT, at revision `4ed0d95271ca00107726230f7a0944ed9e90d897`. Its names and
  businesses are the benchmark's synthetic content.
- **Grader:** `score.py` ports OmniAI's
  [benchmark](https://github.com/getomni-ai/benchmark) `src/evaluation/json.ts`
  (MIT) and [json-diff](https://github.com/andreyvit/json-diff) 1.0.6 (MIT,
  Copyright (c) 2015 Andrey Tarantsov) to Python. Their license notice is
  reproduced at the top of `score.py`.
- **Set lists:** `sets/*.image_manifest.json` are the published image
  manifests, unchanged; `pins.json` holds every pinned revision, size and hash.
- **Replies:** model outputs, published in the evidence packet.
