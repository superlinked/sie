# Turn complete threat reports into cited ATT&CK suggestions

This agent starts with the report text. It finds adversary behavior, resolves
the closest MITRE ATT&CK technique, and keeps the exact source passage beside
every suggestion. No annotated behavior spans are supplied at runtime.

Every mapping remains `needs_analyst_review`. The agent can suggest a technique
or abstain; it cannot accept a mapping.

## Model ensemble

| Job | Model | SIE primitive |
|---|---|---|
| Extract typed adversary events | `Qwen/Qwen3.6-27B:no-spec` | `generate` with a strict JSON schema |
| Tag actors, tools, credentials, and targets | `fastino/gliner2-large-v1` | `extract` |
| Retrieve ATT&CK definitions with one vector per text | `Qwen/Qwen3-Embedding-8B` | `encode` |
| Retrieve labeled report spans from AnnoCTR train | `Qwen/Qwen3-Embedding-8B` | `encode` |
| Retrieve ATT&CK definitions with one vector per token | `jinaai/jina-colbert-v2` | `encode` with `multivector` output |
| Rerank the joint candidate set | `Qwen/Qwen3-Reranker-4B` | `score` |
| Verify the technique against the quoted evidence | `Qwen/Qwen3.6-27B:no-spec` | `generate` with a strict JSON schema |

Dense retrieval and late interaction each search the complete ATT&CK catalog.
A third search compares the extracted behavior with labeled spans from the
AnnoCTR training split, then keeps the nearest example for each technique. The
agent fuses two top-50 definition pools with 20 example-backed candidates and
sends 75 candidates to the reranker. The verifier reads the best ten, including
the nearest labeled source example when one reached the pool.

A verified mapping becomes a direct suggestion when the selected technique is
also the top labeled-example match. Other verified mappings stay in the
analyst review queue. Both routes keep the quote, event fields, and candidate
ledger.

Late interaction uses one query vector per token against every ATT&CK document
multivector. MaxSim scores each query token against its closest document token,
then sums those matches. No query pooling happens before candidate fusion.

`config.yaml` pins every model revision. A managed SIE run also records request
IDs, the settled rate-book version, and execution-identity hashes.

Behavior extraction bounds the JSON array in the grammar for each report
chunk. If a response still ends before valid JSON closes, the retry splits the
text and halves the row limit. The failed response remains in the API ledger.

## Data

`attck-map fetch` downloads two pinned sources:

- MITRE ATT&CK Enterprise 19.2, with 697 active techniques in the August 5,
  2026 STIX bundle.
- AnnoCTR at commit `d510b694`, including its published train, dev, and test
  splits plus the corpus's 578-technique ATT&CK snapshot under CC BY-SA 4.0.

The agent indexes labeled technique spans from AnnoCTR train. Development and
test reports never enter that index. The full-report evaluation still begins
with raw report text; it supplies no target spans or labels for those reports.

The command verifies both SHA-256 hashes. See `fixtures/SOURCES.md` for source
and license details.

## Run the agent

You need Python 3.12 and a SIE endpoint that serves the models in `config.yaml`.
The example sends every model call to `SIE_CLUSTER_URL`.

```bash
cd examples/threat-report-attck-mapper
cp .env.example .env
uv sync --frozen

uv run attck-map fetch
uv run attck-map report path/to/report.pdf --run-id report-review
uv run attck-map demo --run-id mfa-psa
```

The copied `.env` points to `http://localhost:8080`. Set `SIE_CLUSTER_URL` and
`SIE_API_KEY` to use SIE Cloud.

`demo` reads Proofpoint's “MFA PSA, Oh My!” report from AnnoCTR. The report
describes reverse-proxy phishing without supplying the current ATT&CK mapping.
The agent must separate interception, cookie theft, and reuse of a stolen
cookie, including current sub-technique `T1550.004`.

The checked-in run selected `T1550.004` for the exact words “use the stolen
session cookie to log in as the victim.” Its nearest labeled AnnoCTR example
pointed to `T1539`, which describes stealing the cookie. The routing policy kept
the reuse mapping and placed it in the closer-review lane. See `verified-run/`
for the report, raw SIE calls, candidate ledger, and checksummed manifest.

## Run the raw-report evaluation

The end-to-end benchmark starts from every complete report in a published
AnnoCTR split:

```bash
uv run attck-map full-benchmark --split dev --run-id dev-full
uv run attck-map full-benchmark --split test --run-id test-frozen
```

Use development data while changing prompts or thresholds. `EXPERIMENT.md`
freezes the held-out test protocol and excludes the Proofpoint worked example
from the aggregate.

Each run writes the extracted event spans, candidate ledgers, final mappings,
gold mentions, metrics, API calls, and a manifest with artifact hashes. The API
ledger keeps the exact payload and raw response for extraction, reranking, and
verification. Dense and token-level vectors stay in compressed NumPy files.
The evaluation reports:

- direct-suggestion precision where AnnoCTR annotates the cited span;
- behavior and candidate recall from complete reports;
- report-pair reference matches as a coverage diagnostic;
- candidate contribution from dense retrieval, MaxSim, labeled examples, and
  their overlaps.

The older `benchmark` command starts from human-annotated spans. Keep it for
retrieval diagnostics; it is not the end-to-end result.

## Frozen held-out result

The fixed pipeline ran once on the 33-report AnnoCTR test split on August 21,
2026. The Proofpoint worked report was excluded before the run.

| Release check | Test result | Gate |
|---|---:|---:|
| Exact ATT&CK ID when a direct suggestion overlaps an annotated span | 91/109 (83.5%) | 85% |
| Report-technique pairs with an extracted behavior | 199/317 (62.8%) | 70% |
| Correct technique family reached the finalist set after extraction | 180/199 (90.5%) | 90% |
| Exact source offsets | Passed | Required |
| Worked report absent from the aggregate | Passed | Required |

The family finalist gate passed. The precision and behavior-recall gates did
not. No prompt, threshold, or rank weight was changed after seeing the test
result. `EXPERIMENT.md` records the development result and the frozen test
contract.

## Decision boundary

This project prepares ATT&CK suggestions for human review. It does not detect
an intrusion, change a security control, or claim that a report author endorsed
the suggested mapping.

MITRE ATT&CK is a registered trademark of The MITRE Corporation. AnnoCTR report
text and annotations are licensed CC BY-SA 4.0 by the corpus authors and source
contributors.
