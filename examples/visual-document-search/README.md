# Find the PDF page that answers a question, without parsing it to text first

## What this shows

Most document search parses each PDF page to text, then embeds the text. Whatever the parser drops or scrambles,
such as a table's header row, the numbers in a chart or a slide's layout, the search never sees. This example
ranks the page images themselves, with no OCR step, and compares that with the pipelines teams run today.

This is the recorded run behind
[superlinked.com/visual-document-search](https://superlinked.com/visual-document-search). Its sources are in that
page's [SOURCES.md](https://superlinked.com/reference/visual-document-search/SOURCES.md).

**The benchmark.** [ViDoRe v3](https://arxiv.org/abs/2601.08620) (CC BY 4.0) covers annual reports and 10-K filings,
FDA slide decks, European Commission reports, a textbook, and French energy and physics documents. It supplies human
relevance grades for every question. The run uses six public datasets and their English questions: 1,816 questions over
11,624 pages. Four datasets hold English documents: computer_science, finance_en, hr and pharmaceuticals. Two hold
French documents: energy and physics. Every page of a dataset is a candidate for each of its questions.

**The renders.** Every page was rendered once to a JPEG with a 1,650-pixel long side, a US-letter page at 150 dpi.
Every image model received the same bytes.

| Arm | nDCG@10 | Right page first |
| --- | --- | --- |
| SIE `TomoroAI/tomoro-colqwen3-embed-4b:compact` | 64.1 | 62.9% |
| SIE `TomoroAI/tomoro-colqwen3-embed-4b`, default profile | 65.5 | 65.3% |
| Cohere Embed v4 | 61.1 | 60.8% |
| Voyage multimodal-3.5 | 61.0 | 60.5% |
| Parse, then OpenAI text-embedding-3-large | 56.0 | 54.4% |
| Parse, then BM25 | 42.0 | 41.3% |

Both scores are averaged per dataset, then over the six datasets. nDCG@10 is ViDoRe's own metric. "Right page first"
means the first page returned has a positive relevance grade.

On nDCG@10, SIE's compact profile ranks the answer page higher than every other arm on all six datasets, except
SIE's own default profile.
- **Against parse-then-embed:** +8.1 points. The 95% interval, from a paired bootstrap within each dataset, is
  +6.9 to +9.3.
- **Against Voyage multimodal-3.5:** +3.1, interval +2.2 to +4.0.
- **Against Cohere Embed v4:** +3.0, interval +2.0 to +3.9.
- **Against the default profile:** the compact profile is 1.4 points below it, but it encodes about twice as many pages
  per second.

On the four English-document datasets alone (1,206 questions), SIE's lead over each arm holds, in
`evidence/stats-english-documents.json`. On energy and physics every arm searches French pages with English questions,
which is why BM25, with an English stopword list, collapses there.

The two parse arms read each page's markdown as ViDoRe ships it, which comes from a strong parser. A production OCR
step would do no better.

## Models and settings

| Arm | Called as | Input |
| --- | --- | --- |
| SIE, compact | open-source SIE server, `TomoroAI/tomoro-colqwen3-embed-4b:compact` (768 visual tokens per page), multivector, MaxSim | page image; question with `is_query=True` |
| SIE, default | the same model with its default 1,280 visual tokens | the same |
| Voyage multimodal-3.5 | `POST /v1/multimodalembeddings`, `input_type` document and query | page image; question |
| Cohere Embed v4 | `POST /v2/embed`, `embed-v4.0`, float, `search_document` and `search_query` | page image; question |
| OpenAI text-embedding-3-large | `POST /v1/embeddings`, 3,072 dimensions, cosine | the page's markdown, truncated to 8,191 tokens; question |
| BM25 | k1 1.2, b 0.75 | the page's markdown |

All arms ran on 30 September 2026, with the SIE arms on one H100.

A multivector model returns one vector per patch of the page, not one per page. A question's score for a page is
MaxSim: each question vector takes its best match anywhere on the page, and those matches are summed. That is how a
number in a chart or a row of a table is found without any text extraction.

## Run it

Download the recorded run, then score it. Neither step needs a key:

```sh
uv sync
uv run python fetch.py
uv run python score.py
```

`fetch.py` pulls the evidence from the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence), pinned to one
revision, and checks every file against the hash the dataset lists. The download is about 23 MB. It holds:
- the English questions with their relevance grades;
- every arm's top 100 pages for every question;
- the recorded scores;
- the throughput measurements.

`score.py` recomputes every score and interval from the rankings and the grades. It fails unless the scores match
the recording. It draws its own bootstrap samples, so an interval can differ from the recorded one in the last
digit. The intervals above are the recorded ones, from `evidence/stats.json`.

To rank the pages yourself, use a current SIE `main` checkout and follow
[the contributor setup](../../CONTRIBUTING.md#set-up-a-development-checkout).
In a separate shell, from that checkout’s repository root, start the GPU server:

```sh
mise exec -- uv run --package sie-server --extra local sie-server serve --device cuda -m TomoroAI/tomoro-colqwen3-embed-4b:compact
```

The compact profile is available from source; hosted access is coming soon.

Then run it and score your run beside the recording:

```sh
uv run python run.py --smoke                        # 20 questions, a minute: checks the server and the model
uv run python run.py                                # all six datasets, 11,624 pages
uv run python score.py --rankings run-output
```

`score.py` scores a run only when it covers all six datasets, the full benchmark, and names any missing file.
`run.py` downloads each dataset from HuggingFace at the recorded revision and renders the pages as the recording did.
It then ranks every page for every question. Set `SIE_BASE_URL` and `SIE_API_KEY` to call SIE Cloud instead of a local
server.

Each page returns up to 768 vectors of 320 floats. Pass `output_dtype="float16"` to `encode` to halve the response;
the recorded run used float32.
