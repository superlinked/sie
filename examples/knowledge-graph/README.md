# Build a citable graph from filing text

## What this shows

Ten paragraphs from public company filings and regulator notices go to SIE
Cloud as two `/v1/extract` calls each. The first call names the entity types
you want, such as company, subsidiary, product and date. The second sends the
same paragraph back with those entities as metadata and the relation types you
want, such as acquired, subsidiary of and develops. What comes back is a graph
whose nodes carry character offsets into the paragraph, so every node points at
the text it came from.

The run is already recorded. Forty calls cover ten paragraphs through GLiNER2
Large and Base. This example scores the twenty Large calls, each projected from
the unchanged full recording with its request, result, headers and timing.
Both forms live in the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
pinned to one revision by `fetch.py`. Download it and you can re-derive the
recorded counts with **no API key and no inference spend**. The current Large
results support the examples on
[superlinked.com/knowledge-graph](https://superlinked.com/knowledge-graph), whose
sources are in its [SOURCES.md](https://superlinked.com/reference/knowledge-graph/SOURCES.md).

You cannot verify this by cloning alone. The clone gives you the code; the
dataset gives you the evidence. Fetching it needs no account and no token.

- Model: `fastino/gliner2-large-v1`
- Endpoint: `https://api.superlinked.com/v1/extract/fastino%2Fgliner2-large-v1`
- Served deployment revision: `10333b84de80b402376b626eb25366fb081d3faeb893eb4b01cf32e8c27e4aff`, the `X-SIE-Model-Revision` every recorded call carries. Models served together share this value, so it names the deployment rather than the weights.
- HuggingFace revision the weights came from: `b122b11eeaee4dabd32bed80412f3234c0d0e943`
- SIE server version 0.7.3, recorded 2026-09-15
- No threshold or other option is sent, so the server default applies

A person read every edge the page displays against its source paragraph. Those
readings are human judgements, not anything the model returned: GLiNER2 gives
every relation a confidence score and no correctness signal, so the only way a
wrong edge is caught is that somebody read the paragraph.

`score.py` enforces that rather than asserting it. Every displayed edge has to
appear in the set a person read, and the run fails by name if one does not, so
a displayed edge nobody has read cannot score clean.

## Run it

Download the recorded run, then count it. Both steps are standard library
only, so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

Look at both requests for one paragraph without sending them:

```sh
python3 run.py --show flex-credit-facility
```

Send the calls yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output
```

## What result to expect

`score.py` prints a row per paragraph and ends with:

```
10 paragraphs recorded, 5 in the original display selection
8 edges across the 4 proof paragraphs and 5 in the original hero, 13 selected in total
```

These counts describe the frozen five-case page-role manifest. The current
page highlights Flex's credit agreement, Ford's vehicle recall and Tarsus's
acquisition filing. The original selection remains reproducible; it is not an
aggregate accuracy claim.

`canonical-calls.json` retains all forty original calls, including the recorded
usage headers and original digest scopes. `calls.json` is a Large-only per-item
projection for this example. `projection.json` pins the full recording's byte
digest and maps each projected call to its original entry. The scorer refuses
a changed source file or a projection whose requests, results, headers,
timings or model revisions differ from that source.

The two Flex calls each meter 92 input tokens. A hosted estimate must price
both calls at the current published Large rate and round each terminal event
as the credit book requires. The historical credits in a recording do not
establish a current list price.

The five paragraphs outside that original selection are printed too, each with
the edge count and the recorded selection reason. Four returned two edges or
fewer, and the fifth comes from a filing another task page already used.

`score.py` fails rather than skipping. It refuses a paragraph whose text no
longer matches its digest, a candidate with a missing call, a response that
does not match its `response_sha256`, a request the pinned text does not
rebuild, a relations call whose metadata is not the entities call's own output,
a reviewed edge that is missing while the relation it rests on is still being
asked for, and a displayed edge with no recorded reading.

## What this does NOT establish

- **Not that the graph is correct.** Nothing mechanical checks these edges.
  Every relation the model returns carries a confidence score and no
  correctness signal, and a high score is not evidence the paragraph says it.
  Check each returned relation against the retained source before using it as
  a verified fact.
- **Not an accuracy rate.** Eight edges over the four paragraphs outside the
  hero is far too small to support a percentage, and the ten paragraphs were
  chosen to be readable rather than sampled from anything.
- **Not that offsets pin a relation.** GLiNER2 names each end of a relation by
  its text, not by an offset, and returns at most one span per text and label.
  Where a paragraph mentions the same text twice, the relation does not say
  which mention it means, and no threshold changes that. The task page marks
  every occurrence in that case rather than choosing one.
- **Not a benchmark.** The recorded `duration_ms` values are provenance. One
  call took 15.8 seconds against about 0.5 seconds for every other, which is
  the shape of a cold model load rather than a measurement of anything.
- **Not a check of the current website.** `projection.json` identifies the
  immutable public source recording from `superlinked/sie-web`. Its byte
  digest ties the downloaded data to that recording; scoring it does not fetch
  or inspect a later website deployment.

## Inputs

`evidence/inputs/candidates.json` pins ten paragraphs, each with the SHA-256 of its own
text, the URL and SHA-256 of the document it was taken from, and a note saying
how the paragraph was derived from that document. Sources are SEC EDGAR
filings, NHTSA recall reports and FDA recall notices, all public records.

Each candidate also records `page_role`, the original selection and its reason.
`score.py` reads the counts off those frozen roles, so the five-case selection
and eight proof edges come from pinned data rather than the current display.

`evidence/inputs/review.json` holds the readings. `score.py` checks each
flagged triple against the recorded relations, so the review cannot flag an
edge the model never returned, and cannot reach an edge outside the eleven it
claims to be counting.
