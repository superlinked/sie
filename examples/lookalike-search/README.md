# Find the passage that answers a question when a look-alike sits beside it

## What this shows

Regulations and technical docs are full of near-duplicates: the construction
rule next to the general-industry one, the breach rule for 500 or more patients
next to the one for fewer, the refund deadline for cash next to the one for
credit cards. A search that returns the look-alike hands an assistant the wrong
answer, stated with confidence.

This example holds the recorded run behind
[superlinked.com/search](https://superlinked.com/search). Its sources are in
that page's [SOURCES.md](https://superlinked.com/reference/search/SOURCES.md).
It has two tests.

- **Look-alike search.** 5,658 passages from 40 public sources (eCFR parts
  from OSHA, FMCSA, DOT, CFPB, HHS, the Department of Labor and the FTC; IRS
  Publications 15 and 15-A; NIST SP 800-63B-4; Kubernetes and PostgreSQL
  documentation). 440 questions, each written against one answer passage with a
  named look-alike beside it, checked blind by a second model and read by eye,
  then frozen and hashed before any model embedded one.
- **Eight MTEB retrieval tasks, scored per query.** CQADupstackPhysics, CosQA,
  FiQA2018, LegalBenchConsumerContractsQA, NFCorpus, SCIDOCS, SciFact and
  StackOverflowQA: 6,200 queries.

Each model ranks every passage for every question by cosine similarity. The
figure is the share of questions whose first result is the answer.

| Model | Look-alike test, answer first (of 440) | MTEB, first result relevant |
| --- | --- | --- |
| Voyage 4 Large | 79.3% | 54.7% |
| Voyage 4 Lite | 76.4% | 50.5% |
| Cohere Embed v4 | 70.2% | 52.3% |
| SIE Qwen3 Embedding 4B | 69.5% | 54.6% |
| OpenAI text-embedding-3-large | 67.0% | 50.8% |
| OpenAI text-embedding-3-small | 60.5% | 47.5% |

On the look-alike test SIE Qwen3 Embedding 4B is level with OpenAI
text-embedding-3-large (47 questions only SIE got right, 36 only OpenAI did;
p = 0.27) and ahead of 3-small (p < 0.001). Both Voyage models are ahead of it
(p < 0.01), while Cohere Embed v4 is level. On the eight MTEB tasks SIE is level
with Voyage 4 Large and ahead of the rest.

`score.py` prints these figures, and also two further SIE runs. One is the same
4B model through the OpenAI-compatible route with no query instruction. The other
is Qwen3 Embedding 8B on the open-source server.

## Models and settings

| Model | Called as | Query encoding |
| --- | --- | --- |
| SIE Qwen3 Embedding 4B | `Qwen/Qwen3-Embedding-4B` on `https://api.superlinked.com/v1/encode` | `is_query=True`, the model's own query instruction |
| OpenAI text-embedding-3-large | OpenAI embeddings API, 3,072 dimensions | none (the API has none) |
| OpenAI text-embedding-3-small | OpenAI embeddings API, 1,536 dimensions | none |
| Voyage 4 Large, Voyage 4 Lite | Voyage embeddings API, 1,024 dimensions | `input_type="query"`, passages `"document"` |
| Cohere Embed v4 | Cohere v2 embed API, float, 1,536 dimensions | `input_type="search_query"`, passages `"search_document"` |

All models ran on 30 September 2026.

Through SIE's OpenAI-compatible `/v1/embeddings` route there is no query flag.
There, prefix each question with the model's instruction,
`"Instruct: Given a query, retrieve relevant passages that answer the query\nQuery: "`,
which gives the same vector as `is_query=True`. Without the prefix a question is
embedded as a document, and on the MTEB tasks it ranks the right result first
about 5 points less often.

## Run it

Download the recorded run, then score it. Neither step needs a key:

```sh
uv sync
uv run python fetch.py
uv run python score.py
```

`fetch.py` pulls the evidence from the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
pinned to one revision, and checks every file against the hash the dataset
lists. The download is about 70 MB. It holds:

- the corpus and the frozen questions;
- SIE's recorded vectors;
- every model's recorded top ten for every question;
- the per-query MTEB scores.

`score.py` ranks all 5,658 passages for every question from SIE's vectors and
fails unless that ranking matches the recording. It then counts each model's
first-place answers, compares every model with SIE by exact McNemar test, and
averages the per-query MTEB scores. The other vendors' vectors are not
redistributed; their rankings are read as recorded.

Encode it yourself on SIE Cloud, which needs a key and spends credits:

```sh
SIE_API_KEY=sk-sie-... uv run python run.py --smoke   # 20 questions, a fraction of a cent
SIE_API_KEY=sk-sie-... uv run python run.py           # the whole test, about 1.2M tokens
uv run python score.py --vectors run-output/qwen3-embedding-4b
```

A blank passage is not sent. It gets a zero vector, which scores 0 for every
question.
