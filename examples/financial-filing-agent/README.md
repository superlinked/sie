# Trace a restated figure back to the controlling filings

This example follows one Pathward Financial fact through exact table rows from
an original Form 10-Q and a restated Form 10-K/A, plus exact sentences from an
Item 4.02 Form 8-K. It returns both versions, calculates the change, and keeps
the source status and company caveat attached.

The result is narrow on purpose. It does not classify the restatement as fraud
or misconduct, and it does not make an investment recommendation.

## What runs on SIE

| Step | Model | Output |
|---|---|---|
| Parse the source packet | `docling` | Markdown with source headings |
| Retrieve candidate passages | `BAAI/bge-m3` | Dense vectors and cosine ranking |
| Rerank against the exact question | `Qwen/Qwen3-Reranker-4B` | Ordered evidence with scores |
| Verify the cited spans | `urchade/gliner_multi-v2.1` | Exact source spans |
| Recover exact source spans | `fastino/gliner2-large-v1` | Table values, period, company and reliance-status entities |

Each stage consumes the prior stage. The result fails closed if either entity
model omits a required source span. The original values come from the original
Form 10-Q table. The restated values come from the Form 10-K/A table, and the
pipeline checks that its “As Previously Reported” column matches the Form 10-Q.
The company caveat is retained verbatim only when the reranked evidence contains
its exact source sentence. Ordinary Python code calculates the difference with
decimal arithmetic.

## Verified result

The recorded component calls ran through the public SIE server on an NVIDIA L4.
The restated passage ranked first for the question about the restated Q3 FY2023
figure.

```text
Net income attributable to parent   $45.096M -> $36.080M
Change                               -$9.016M (-20.0%)
Diluted EPS                          $1.68 -> $1.34
Source status                        earlier affected filings should no longer be relied upon
```

Pathward's exact statement stays in the result: the accounting change does not
impact net income over the life of the portfolio, but changes when elements of
the programs are recognized.

## Run it

```bash
cd examples/financial-filing-agent
cp .env.example .env
uv sync

uv run review-filing --run-id local
uv run eval-filing runs/local
```

If the process crashes, it can leave `runs/.<run-id>.lock` and a
`runs/.<run-id>-*` staging directory. Remove those exact abandoned paths before
retrying the same run ID.

Set `SIE_CLUSTER_URL` and `SIE_API_KEY` to use SIE Cloud. The default points to
a local server at `http://localhost:8080`.

## Evidence bundle

Every run writes:

```text
runs/<run-id>/manifest.json       endpoint, model IDs, fixture hashes, latency
runs/<run-id>/raw/parse.json      complete Docling response
runs/<run-id>/raw/retrieve.json   embeddings and cosine ranking
runs/<run-id>/raw/rerank.json     complete reranker response
runs/<run-id>/raw/entities.json   combined entity spans
runs/<run-id>/raw/entities-original-10q.json   original Form 10-Q entity spans
runs/<run-id>/raw/entities-restated-10ka.json  restated Form 10-K/A entity spans
runs/<run-id>/raw/entities-restated-10ka-diluted.json  isolated diluted-EPS row spans
runs/<run-id>/raw/entities-status.json         reliance-status entity spans
runs/<run-id>/raw/gliner2-*.json raw GLiNER2 source-span responses
runs/<run-id>/raw/mapped.json     validated record mapped from Docling table coordinates
runs/<run-id>/parsed.md           parsed packet used downstream
runs/<run-id>/review.json         source-versioned result and calculated delta
runs/<run-id>/evaluation.json     deterministic checks
```

`verified-run/` contains the August 9, 2026, prod-US evidence. It is not a
latency benchmark. Its manifest pins the endpoint, run date, model IDs, source
hashes, diagnostic timings, and every artifact checksum.

See [fixtures/SOURCES.md](fixtures/SOURCES.md) for accession numbers, URLs, and
source checksums.
