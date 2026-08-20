# Turn threat reports into cited ATT&CK mapping suggestions

This example reads adversary behavior from a threat report, retrieves the
closest active MITRE ATT&CK Enterprise techniques, reranks the hard negatives,
and verifies the final suggestion against an exact quote. Every selected
technique remains `needs_analyst_review`. The model can suggest, request review,
or abstain; it cannot accept a mapping.

The checked configuration uses these SIE models:

| Job | Model | SIE primitive |
|---|---|---|
| Extract behavior spans | `Qwen/Qwen3.5-4B` | `generate` with strict JSON schema |
| Extract tools, credentials, actors, and protocols | `fastino/gliner2-large-v1` | `extract` |
| Retrieve active ATT&CK techniques | `Qwen/Qwen3-Embedding-8B` | `encode` |
| Rerank close technique definitions | `Qwen/Qwen3-Reranker-4B` | `score` |
| Verify the evidence and candidate | `Qwen/Qwen3.5-4B` | `generate` with strict JSON schema |
| Resolve ambiguous cases | `Qwen/Qwen3.6-27B` | `generate` with strict JSON schema |

`config.yaml` records the expected catalog revision for every checkpoint. A
managed run also records request IDs, the settled rate-book version, and SIE's
execution-identity hashes.

`Qwen3-Embedding-8B` returns 4096-dimensional dense vectors and accepts up to
40,960 tokens. The query uses a security-specific retrieval instruction. ATT&CK
definitions are encoded as documents, so they do not receive the query prefix.

## Data

`attck-map fetch` downloads two pinned sources:

- MITRE ATT&CK Enterprise 19.2, 697 active techniques in the August 5, 2026
  STIX bundle.
- AnnoCTR at commit `d510b694`, including its published train, dev, and test
  splits plus the corpus's 578-technique ATT&CK snapshot under CC BY-SA 4.0.

The command verifies both SHA-256 hashes. See `fixtures/SOURCES.md` for source,
license, and benchmark details.

## Run the benchmark

You need Python 3.12 and a GPU-backed SIE endpoint that serves the configured
models.

```bash
cd examples/threat-report-attck-mapper
cp .env.example .env
uv sync --frozen

uv run attck-map fetch
uv run attck-map demo --run-id mfa-psa
uv run attck-map report path/to/report.pdf --run-id report-review
uv run attck-map benchmark --split dev --stage retrieve --limit 50 --run-id dev-smoke
uv run attck-map evaluate runs/dev-smoke
```

`demo` runs the complete report agent on Proofpoint's “MFA PSA, Oh My!” report
from AnnoCTR's test split. The report agent maps against active ATT&CK 19.2. It
has to distinguish interception and cookie theft from later reuse of the stolen
cookie, including current sub-technique `T1550.004`.

Remove `--limit` for the complete benchmark split. The stages are cumulative:

- `retrieve` embeds the active ATT&CK catalog once and measures dense recall.
- `rerank` sends the top candidates through `Qwen3-Reranker-4B`.
- `verify` runs the strict verifier and invokes the 27B model only when the 4B
  verifier returns `ambiguous`.

Each run writes `predictions.jsonl`, compressed embedding matrices, an API-call
ledger, recomputed metrics, and a manifest with source and artifact hashes.
The saved predictions retain the exact AnnoCTR evidence span and all active gold
technique IDs.

## What the metrics mean

The AnnoCTR linking task starts from a human-annotated behavior span. Some spans
have more than one valid technique, so the loader groups duplicate spans into a
multi-label case. The report includes:

- dense hit rate, gold recall at 10, MRR, and document-macro hit rate;
- reranker hit rate and recall at 5;
- verifier coverage, selective precision, and review or abstention rate.

The benchmark uses the ATT&CK entity snapshot shipped with AnnoCTR because that
is the label space its annotations describe. Full-report review uses ATT&CK
19.2. Mixing those catalogs would count taxonomy changes as model misses. Any
gold ID missing from the bundled benchmark catalog appears under
`excluded_gold_ids`.

## Decision boundary

This project prepares ATT&CK suggestions for human review. It does not detect an
intrusion, change a security control, or assert that a report author endorsed
the suggested mapping. Full-report behavior extraction and annotated-span
linking are separate evaluation surfaces; the saved manifest states which one a
run measured.

MITRE ATT&CK is a registered trademark of The MITRE Corporation. AnnoCTR report
text and annotations are licensed CC BY-SA 4.0 by the corpus authors and source
contributors.
