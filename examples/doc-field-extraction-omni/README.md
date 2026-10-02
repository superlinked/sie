# Read documents into your own schema

Replay the recorded extraction comparison on 700 Omni business documents and
100 CORD receipts without an API key, installed dependencies, or model calls:

```bash
python3 fetch.py
python3 score.py
```

The fetch pins an immutable public [evidence packet](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/61e751a5f7f374cacd84ccace621f71e8a498404/doc-field-extraction-omni/2026-10-01)
and verifies its manifest, every file hash, and every file size. The replay
requires all 800 inputs for each of the five models and checks the full and
fresh scopes against the recorded aggregates. The fresh scope excludes the
30 pilot documents: 676 business documents and 94 receipts.

On the October 1, 2026 recording, SIE Qwen3.8 27B FP8 improved mean business
document JSON accuracy over Haiku 4.5: 75.26% versus 65.71% across 700 documents;
75.05% versus 65.78% on the fresh 676. The registered paired superiority gate
passes on both scopes. Sonnet 5.5, GPT-6 Luna and GPT-6 Sol scored higher on the
business documents; the packet retains all of their results. Luna also had a
lower token-list cost.

`counts.jsonl` contains the 4,000 selected-form score, count, token and response
hash records. `summary.json` retains the aggregate scores for both Claude
schema forms, the recorded 10,000-sample bootstrap and eligibility decisions.
Unparsed and capped replies remain in the denominators. This replay verifies
saved counts; raw model replies are not included, so it does not independently
regrade them or rerun the bootstrap. Omni's mean JSON-diff accuracy and CORD's
mean normalized field accuracy are separate metrics, not whole-document success.

The historical preregistered SIE $0.72/$0.72 token pair and current default
Cloud rate projection are separately identified in the packet. The measured
self-hosted profile differs from that default SKU; neither projection is its
billed Cloud cost or a GPU ownership estimate.

## Try your own document

The exact measured profile is
`Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec`, on an H100, with public server
source [`6cccc04de9c6b78d3217854ad2377de36c0ffbd2`](https://github.com/superlinked/sie/tree/6cccc04de9c6b78d3217854ad2377de36c0ffbd2).
In a checkout of that source with the GPU dependencies configured, start it:

```bash
mise run serve -- -p 8080 -d cuda:0 \
  -m 'Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec' \
  --preload 'Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec'
```

Write a schema for your own PNG or JPEG, for example:

```json
{
  "type": "object",
  "properties": {
    "document_number": {"type": "string"},
    "total": {"type": "number"}
  }
}
```

Inspect the request offline, then send one call to that running server using
the public `sie_sdk.SIEClient`:

```bash
python3 run.py --image form.png --schema fields.json --show
uv run --no-project --with 'sie-sdk>=0.7.3,<0.8' python run.py \
  --image form.png --schema fields.json --endpoint http://127.0.0.1:8080
```

The trial uses the study's common strict-schema conversion: every property is
required, leaf values may be null, and unsupported source-schema constraints
are omitted. Referenced property schemas must be inlined; `$ref` is rejected.
String enums retain their non-null choices while the nullable branch allows null.
It sends temperature 0, presence penalty 0, a 4,096-token cap,
one image and the exact profile. A new document call is a trial, not a replay
of the frozen benchmark. The endpoint may be a remote self-hosted server;
`SIE_API_KEY` supplies its credential when required.

## Source licenses

[OmniAI OCR Benchmark](https://huggingface.co/datasets/getomni-ai/ocr-benchmark)
is MIT licensed at revision `4ed0d95271ca00107726230f7a0944ed9e90d897`.
[CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2), by NAVER Clova,
is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The original CORD repository revision was not recorded; the packet preserves
each frozen image hash. The count packet includes no source images or raw replies.
