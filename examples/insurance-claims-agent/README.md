# Review a public flood-insurance appeal through one SIE cluster

This example reviews FEMA Flood Insurance Appeal Decision B8 and the controlling
Standard Flood Insurance Policy. The appeal is a public, redacted record. It
concerns a July 2019 Lake Ontario flood and a disputed request to remove stones
from underneath an insured building.

The record is useful because the outcome turns on scope. FEMA directed the
insurer to cover removal of 12 to 15 cubic yards of flood-borne stones from
underneath the building to its perimeter. Barge transport, handling, disposal,
and debris removal from the yard remain outside that covered scope.

## What SIE does

| Stage | Model | Result |
|---|---|---|
| Parse the appeal and policy | `docling` | Markdown with the record, amounts, rules, analysis, and conclusion |
| Extract claim facts | `fastino/gliner2-large-v1` | Amounts, debris volume, loss date, and coverage terms |
| Retrieve controlling policy language | `Qwen/Qwen3-Reranker-4B` | Ranked passages about non-owned debris removal |
| Produce the cited review | `Qwen/Qwen3.5-4B` | JSON separating covered work, excluded costs, and evidence still needed |

Every model call goes through SIE.

## Expected result

The evaluator checks facts stated in FEMA's published decision:

- amended proof of loss: `$182,552.00`;
- debris-removal estimate: `$49,500.00`;
- barge estimate: `$181,832.94`;
- covered physical scope: `12` to `15` cubic yards beneath the building;
- excluded scope: barge transport, handling, disposal, and yard removal;
- follow-up evidence: comparison estimates and proof of work from prior claims.

The model summarizes a completed public appeal. It does not decide a live claim.

The [`verified-run`](verified-run/) directory records an August 9, 2026, prod-US
run in which all ten factual checks passed.

## Run it

```bash
cd examples/insurance-claims-agent
cp .env.example .env
uv sync

uv run fetch-claim-sources
uv run review-claim --run-id local
uv run eval-claim runs/local
```

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

# Terminal 2: Qwen generation
sie-server serve --models Qwen/Qwen3.5-4B:no-spec --port 8081

SIE_GENERATION_URL=http://localhost:8081 uv run review-claim --run-id local
```

On one GPU, release the default bundle before loading the generation model:

```bash
uv run review-claim --run-id local --stage default
# Stop the default server, then start Qwen on the same port.
uv run review-claim --run-id local --stage generation
```

## Evidence bundle

```text
runs/<run-id>/manifest.json           endpoints, models, and per-call latency
runs/<run-id>/source-manifest.json    source URLs, rights, sizes, and checksums
runs/<run-id>/default-stage.json      default-bundle endpoint, models, and timings
runs/<run-id>/markdown/*.md           parsed appeal decision and policy
runs/<run-id>/claim-facts.json        extracted amounts, dates, and scope phrases
runs/<run-id>/policy-evidence.json    reranked policy passages
runs/<run-id>/review.json             structured appeal review
runs/<run-id>/evaluation.json         deterministic factual checks
runs/<run-id>/raw/*.json              complete model responses
```

## Safety boundary

The output summarizes a published FEMA appeal for software evaluation. It does
not approve or deny coverage, calculate a payment, label fraud, or replace an
adjuster.
