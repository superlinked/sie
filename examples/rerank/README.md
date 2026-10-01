# Rank the right document type with a rule in the request

Your search can find the subject and still put a proposal first when the reader
needs an adopted rule. This example gives an open reranker a relevance rule per
request, so the same question and candidate set can serve either purpose.

Model: [`Qwen/Qwen3-Reranker-4B`](https://huggingface.co/Qwen/Qwen3-Reranker-4B).
The server and SDK are in this repository. Run the model beside your index or use
a compatible SIE endpoint.

## Recorded result

The held-out study contains **499 questions, each with 20 candidates**. Candidates
are Federal Register titles and abstracts: an adopted rule and a proposal on the
queried subject, plus 18 distractors from the same agency. Ground truth comes from
the source's `Rule` / `Proposed Rule` document type; it does **not** establish whether
a regulation is currently legally in force.

| Model | Adopted rule first | Proposal first | Both rules correct in every recorded repetition | Repetitions |
| --- | ---: | ---: | ---: | ---: |
| SIE · Qwen3 Reranker 4B | 483/499 (96.8%) | 469/499 (94.0%) | 455/499 (91.2%) | 1 |
| Cohere Rerank 4 Pro | 435/499 (87.2%) | 477/499 (95.6%) | 420/499 (84.2%) | 1 |
| Cohere Rerank 4 Fast | 389/499 (78.0%) | 447/499 (89.6%) | 357/499 (71.5%) | 1 |
| Claude Haiku 4.5 | 487/499 (97.6%) | 490/499 (98.2%) | 478/499 (95.8%) | 2 |
| GPT-6 Luna | 492/499 (98.6%) | 491/499 (98.4%) | 485/499 (97.2%) | 2 |
| Voyage rerank-2.5 | 484/499 (97.0%) | 478/499 (95.8%) | 464/499 (93.0%) | 1 |
| Voyage rerank-3 | 475/499 (95.2%) | 485/499 (97.2%) | 462/499 (92.6%) | 1 |

Per-arm language-model success requires the right document in **both** repetitions.
Rerankers were recorded once; their figures do not measure repeated-run reliability.
Each model's instruction wording was selected on a separate 100-question development
set. The baseline without an instruction is also recorded.

The SIE run initially had 18 transport timeouts. A bounded repair retried only
those requests, preserving all original results. Adopted/proposal hits rose from
477/462 to 483/469. The publication threshold passes after repair; the repair was
registered after observing the first results and is not a fresh confirmatory run.
The planned large lead over Cohere and non-inferiority to Haiku were not established.
These results do not establish a quality win over Voyage or hosted latency.

The recorded listwise parser accepts valid nonempty partial rankings. Haiku returned
668 partial rankings of 2,994 calls; Luna returned 6. All dedicated reranker responses
ranked all 20 candidates. The figures above measure the first result, and the scorer
reports completeness separately.

## Reproduce without inference

From this directory:

```sh
python3 fetch.py
python3 score.py
python3 run.py --check
python3 -m unittest discover -s tests -v
```

These commands use the standard library and need no credentials or inference spend.
The fetcher anonymously downloads a pinned revision of
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/e5b19d7589fc3a982fcfeeabb071cf833c22455f/rerank-relevance-rules),
checks the manifest and every file digest, and refuses to replace an unowned directory.
Missing or duplicate calls fail rather than reducing the denominator. The protocol,
first-attempt history, repair history, development cases and all seven model recordings
are included.

`python3 run.py --show 'CASE ID'` prints both request envelopes for an actual question.
`score.py --emit figures.json` writes the reproduced figures as JSON.

## Try the model

Use the SDK from this repository's current main checkout; initialize the repository
as described in [CONTRIBUTING.md](../../CONTRIBUTING.md). From the repository root:

```sh
mise exec -- uv run --package sie-sdk python examples/rerank/run.py \
  --data examples/rerank/data --record --limit 1 \
  --base-url http://localhost:8000 --out rerank-trial.json
```

This explicitly runs inference: one question under two rules by default. For a hosted
endpoint, use its base URL and provide `SIE_API_KEY` through your environment. The
runner checks the model revision before scoring, sends all 20 candidates through
`SIEClient.score`, and refuses to overwrite an existing trial file. Trials are separate
from the published study. Self-hosting setup is in the repository's [README](../../README.md).
GPU serving needs a suitable GPU host; the offline scorer does not.

## Cost scope

The recorded caller-content usage averages **3,502.27 tokens per 20-candidate query**.
At the published SIE rate of $0.0425 per million content tokens, that is an estimated
**$0.149 per 1,000 queries**. This prices recorded usage at the current rate; it is not
an observed hosted bill. The study used a self-hosted server and made no SIE API charge.
GPU infrastructure costs are separate, and hosted latency was not measured.

## Sources and license

The corpus comes from the [Federal Register public API](https://www.federalregister.gov/developers/documentation/api/v1).
The [Office of the Federal Register permits reproducing its content](https://www.archives.gov/federal-register/faqs).
The downloaded evidence preserves source URLs, the selection protocol, hashes and
public server revision. Derived annotations and recordings are Apache-2.0; the
Federal Register corpus is public-domain US government material. No seal or logo
is redistributed.
