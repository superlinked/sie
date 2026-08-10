# Reproduce CMS's L1851 documentation finding

CMS publishes an example in which a supplier submits an L1851 claim with a
correctly coded order, an adequate medical-necessity record, and proof of
delivery. The face-to-face encounter happened seven months before proof of
delivery. CMS requires it within six months.

This project parses the published example, retrieves and reranks its evidence,
extracts source spans, maps exact ranked fragments, then validates the one-month
timing gap in ordinary code. Its output reproduces the CMS result:
`insufficient documentation error`, followed by `MAC recoups payment`.

It does not decide coverage or medical necessity. It makes no diagnosis,
treatment recommendation, or prospective payment decision.

## What runs on SIE

| Step | Model | Output |
|---|---|---|
| Parse the CMS excerpt | `docling` | Markdown |
| Retrieve candidate passages | `BAAI/bge-m3` | Dense vectors and cosine ranking |
| Put the controlling passages first | `Qwen/Qwen3-Reranker-4B` | Ranked source evidence |
| Extract named requirement and case spans | `urchade/gliner_multi-v2.1` | Source spans with offsets and scores |
| Recover requirements, submission facts, and outcome spans | `fastino/gliner2-large-v1` | Labeled entity spans from three bounded source groups |

Every stage consumes the prior stage. The three GLiNER2 calls cover
requirements, the submitted record, and the published outcome. Each call sees
only the deduplicated shortest CMS fragments needed for that group. The run
requires the model to recover the controlling phrases, including `L1851`,
`6 months`, `7 months`, `insufficient documentation`, and `recoups payment`.
Ordinary code then maps the exact ranked CMS fragments to the review fields.
The raw response is written before the next group runs. The final outcome
fields retain those mapped source fragments; deterministic evaluation checks
their required terms rather than replacing them with canned summaries.

The deterministic validator then requires all of these facts:

- HCPCS code `L1851`
- a six-month face-to-face window
- a face-to-face encounter documented seven months before proof of delivery
- CMS's `insufficient documentation error` conclusion
- CMS's statement that the MAC recoups payment

Missing, conflicting, or untraceable model output stops the run. The one-month
gap is calculated from the two extracted month counts.

## Source

The only case input is an exact excerpt from CMS's [Lower Limb
Orthoses][cms-source] page, last modified February 11, 2026. It contains no
invented patient, packet, or evaluation set.

See [fixtures/SOURCES.md](fixtures/SOURCES.md) for the source boundary.

## Run it

```bash
cd examples/prior-authorization-review-agent
cp .env.example .env
uv sync

uv run review-pa --run-id local
uv run eval-pa runs/local
```

If the process crashes, it can leave `runs/.<run-id>.lock` and a
`runs/.<run-id>-*` staging directory. Remove those exact abandoned paths before
retrying the same run ID.

Set `SIE_CLUSTER_URL` and `SIE_API_KEY` to use SIE Cloud. The default points to
a local server at `http://localhost:8080`.

## Evidence bundle

```text
runs/<run-id>/manifest.json                    endpoint, model IDs, source hash, latency
runs/<run-id>/raw/parse.json                   complete Docling response
runs/<run-id>/raw/retrieve.json                embeddings and cosine ranking
runs/<run-id>/raw/rerank-request.json          exact reranker query and candidate IDs
runs/<run-id>/raw/rerank.json                  complete reranker response
runs/<run-id>/raw/entities.json                combined GLiNER entity spans
runs/<run-id>/raw/entities-requirement-<index>.json per-requirement GLiNER entity spans
runs/<run-id>/raw/entities-case-<index>.json   per-case GLiNER entity spans
runs/<run-id>/raw/gliner2-requirements.json    GLiNER2 requirement spans
runs/<run-id>/raw/gliner2-submission.json      GLiNER2 submission spans
runs/<run-id>/raw/gliner2-outcome.json         GLiNER2 outcome spans
runs/<run-id>/raw/mapped.json                  validated exact-fragment mapping
runs/<run-id>/parsed.md                        parsed CMS excerpt used downstream
runs/<run-id>/review.json                      validated reproduction of the CMS result
runs/<run-id>/evaluation.json                  source, arithmetic, and boundary checks
```

Recorded latency includes model provisioning and is not a performance claim.
The verified manifest preserves the upstream-style Docling name configured
during acquisition and the canonical `docling` ID that SIE actually served.
New runs use the canonical ID directly.

[cms-source]: https://www.cms.gov/training-education/medicare-learning-networkr-mln/compliance/medicare-provider-compliance-tips/lower-limb-orthoses
