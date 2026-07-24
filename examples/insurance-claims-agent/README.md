# Review a flood claim packet through one SIE cluster

This example reviews a realistic claim packet built from an official FEMA proof
of loss, the Standard Flood Insurance Policy, a public-domain damage photograph,
and a fictional repair estimate.

The packet contains two deliberate problems. The proof of loss is unsigned. Its
net claim total is $81,060, while the attached estimate and inventory support
$80,660 after deductibles.

## What SIE does

| Call | Model | Result |
|---|---|---|
| Parse the form, estimate, and policy | `docling` | Markdown with form labels, tables, and policy text |
| Read the claim identity | `fastino/gliner2-large-v1` | Typed name, policy number, loss date, and property address |
| Retrieve controlling policy language | `BAAI/bge-reranker-v2-m3` | Ranked passages about proof of loss and supporting records |
| Describe the submitted photograph | `microsoft/Florence-2-base-ft` | Visible conditions from the image |
| Produce the evidence review | `Qwen/Qwen3.5-4B:no-spec` | JSON with route, totals, sourced findings, and next actions |

Every model call goes through SIE. The review uses the generation endpoint; the
photograph uses Florence-2 through the extract endpoint.

## Run it

```bash
cd examples/insurance-claims-agent
cp .env.example .env
uv sync

uv run fetch-claim-sources
uv run prepare-claim
uv run review-claim --run-id local
uv run eval-claim runs/local
```

`prepare-claim` fills FEMA Form 086-0-09 with the fictional values in
`fixtures/claim.json`. It also creates a contractor-style repair estimate and
copies the public-domain photograph into the packet.

For SIE Cloud, set one URL and key:

```bash
SIE_CLUSTER_URL=https://api.superlinked.com
SIE_API_KEY=...
```

A self-hosted development setup can run the default and generation bundles on
separate ports:

```bash
# Terminal 1: Docling, GLiNER2, and reranking
sie-server serve --port 8080

# Terminal 2: Florence-2 and Qwen generation
sie-server serve --models microsoft/Florence-2-base-ft,Qwen/Qwen3.5-4B:no-spec --port 8081

SIE_GENERATION_URL=http://localhost:8081 uv run review-claim --run-id local
```

On one GPU, release the default bundle before loading the generation models:

```bash
uv run review-claim --run-id local --stage default
# Stop the default server, then start the Florence-2 + Qwen server on the same port.
uv run review-claim --run-id local --stage generation
```

## Evidence bundle

```text
runs/<run-id>/manifest.json           endpoints, models, and per-call latency
runs/<run-id>/source-manifest.json    source URLs, rights, sizes, and checksums
runs/<run-id>/packet-manifest.json    packet files and expected reconciliation
runs/<run-id>/markdown/*.md           parsed form, estimate, and policy
runs/<run-id>/policy-evidence.json    reranked policy passages
runs/<run-id>/photo-analysis.md       vision-model observations
runs/<run-id>/review.json             structured claim review
runs/<run-id>/evaluation.json         deterministic result checks
runs/<run-id>/raw/*.json              complete model responses
```

## Safety boundary

The output routes evidence for a human adjuster. It does not approve or deny
coverage, calculate a payment, label fraud, or make a legal determination. The
claim is fictional, and the generated estimate is marked as a software fixture.
