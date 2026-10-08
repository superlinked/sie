# Photo transcription fidelity

Audit a frozen set of 24 photographed receipts and handwritten documents, then
collect once-only transcriptions from an existing SIE deployment. The offline
scorer needs only Python; native collection uses `sie_sdk.SIEClient`.

The [public source packet](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/fd29e7a2813e90deec6ac27e1fcdc22a5e237c44/ocr-photo-transcription/2026-10-08/source-provider-v2)
contains twelve CORD receipts, twelve GNHK handwritten documents, original
JPEGs and annotations, full gold text, source-context decisions and all 48
saved provider transcriptions. The downloader pins the dataset revision and
checksum-list hash, then checks every file. Source images are CC BY 4.0; the
packet includes publisher attribution, license evidence and modification notices.

## Reproduce the saved comparison

From this directory, with Python 3.12:

```bash
python fetch.py data
python score.py data
```

This downloads about 32 MB and makes no inference calls. The source-adjacent-v2
decisions reproduce **16/24 Luna** and **24/24 Sol** photos with complete
transcription and no critical errors. `native` is `null` until you collect and
review native results. The saved provider file is a **projection**, with usage,
request semantics and detached original wire-body hashes; it is not the original
transport recording. Reproducing its decisions does not rerun those providers.

The unit is one photographed document. A pass requires every assessable
critical amount, date, identifier and proper name to preserve its source role,
plus every readable gold line to be represented. Equivalent grouping, case,
spacing and predeclared date/quantity variants are allowed; a correct number on
the wrong line is a failure. This is a purposively balanced exploratory pilot,
not a representative population estimate or a claim of statistically established
model superiority. Publisher training splits and unknown model training exposure
are recorded in the manifest.

## Collect from a remote SIE endpoint

Install the SDK from this checkout using the repository's ordinary development
setup, or run `python -m pip install -e packages/sie_sdk` from the repository
root in a Python 3.12 environment. Set `SIE_URL` to an existing deployment.
For authentication, set `SIE_BASE_URL` to the same endpoint and `SIE_API_KEY`
to its key; the SDK confines that environment credential to the matching origin.
The scripts do not start a server or run a model on your machine.

```bash
python collect.py data native-glm \
  --model zai-org/GLM-OCR \
  --instruction 'Text Recognition:'
```

An alternative endpoint can be supplied with `--sie-url`. Choose a model already
served there and freeze its checkpoint/runtime configuration separately. For
models with a plain default OCR task, omit `--instruction`; do not carry a
Florence task token into another model. The collector sends each full JPEG,
original item ID, `profile=default`, `max_new_tokens=8192` and `num_beams=1`.
`--max-new-tokens` accepts a smaller predeclared limit, never a larger one.

The public SDK uses an injected HTTP transport to save the actual MessagePack
request and reply bodies. It records at most one inference POST attempt per
photo, rejects non-success replies before automatic model-loading retries, and
stops after a failed request. No authentication headers are saved. Later photos
remain explicitly **unattempted**, rather than being silently retried or scored
as model failures. A timeout may have reached the server, so its attempt still
counts. Bounded same-origin result retrievals use GET, are recorded separately,
and do not replay inference. Endpoint health GETs are not inference attempts. Output directories must
be new, preserving an earlier run instead of overwriting it.

## Review complete source fidelity

`native-glm/review-template.json` has one entry for each nonempty completed
transcription. Review against the **full photograph**, gold lines, local roles,
normalization policy and `source-amendments.json`, with model identity withheld.
Copy it to `decisions.json`, identify the actual human or AI reviewer, and fill
every critical-occurrence and readable-line decision. Record the review rationale
separately; do not describe AI review as human adjudication.

Critical outcomes are `preserved`, `changed`, `missing`, `misassociated`,
`invented` or `not_assessable`. Line outcomes are `represented`, `missing` or
`not_assessable`. The last option is restricted to the declared publisher-blurred
logo in `cord/train-0012` (critical index 15, line 0). The source-only amendment
for `gnhk/train-eng_NA_144` corrects the handwritten year from the original gold
2019 to **2018**; the original gold bytes remain unchanged. Apply both amendments
uniformly, rather than choosing an interpretation based on which model benefits.

```bash
python score.py data \
  --native-calls native-glm/calls.jsonl \
  --decisions native-glm/decisions.json
```

The scorer binds reviews to the exact image and transcription hashes, requires
all frozen case slots, and separates unattempted/unreviewed cases from failures.
It reports paired counts against each preserved provider arm. String matching
alone cannot reproduce semantic source-context review, and the script does not
claim to replace that review or supply a confidence interval from this pilot.

## Validate without credentials or inference

After the repository's ordinary development setup, from the repository root:

```bash
mise exec -- uv run --frozen --project . --no-sync pytest -q examples/photo-transcription/tests
mise exec -- uv run --frozen --project . --no-sync ruff check --select E,F,I,UP,B examples/photo-transcription
mise exec -- uv run --frozen --project . --no-sync ruff format --check examples/photo-transcription
```

The tests use synthetic source packets and the real SDK with a mock HTTP
transport. They check full-JPEG request controls, once-only model-loading failure,
hash tampering, full-cohort retention and incomplete-review handling.
