# Build a citable graph from filing text

## What this shows

Ten paragraphs from public company filings and regulator notices go to SIE
Cloud as two `/v1/extract` calls each. The first call names the entity types
you want, such as company, subsidiary, product and date. The second sends the
same paragraph back with those entities as metadata and the relation types you
want, such as acquired, subsidiary of and develops. What comes back is a graph
whose nodes carry character offsets into the paragraph, so every node points at
the text it came from.

The run is already recorded. `calls.json` holds all twenty requests and the
entities and relations they returned, so you can re-derive the published counts
with no API key and no network. Those are the same values behind the figures on
[superlinked.com/knowledge-graph](https://superlinked.com/knowledge-graph).

- Model: `fastino/gliner2-large-v1`
- Endpoint: `https://api.superlinked.com/v1/extract/fastino%2Fgliner2-large-v1`
- Model revision: `b122b11eeaee4dabd32bed80412f3234c0d0e943`
- SIE server version 0.7.3, recorded 2026-09-15
- No threshold or other option is sent, so the server default applies

Three of the eleven edges in the proof paragraphs are wrong in a way no score
exposes. `data/review.json` records them, with the reason each was marked. They
are human judgements from reading the source paragraph, not anything the model
returned, and the example treats them as data you can disagree with.

## Run it

Count the recorded run. Nothing to install, no key, no network:

```sh
python3 score.py
```

Run the tests, which also cover the fail-closed paths:

```sh
python3 -m unittest discover -s tests -v
```

Look at both requests for one paragraph without sending them:

```sh
python3 run.py --show flex-credit-facility
```

Send the calls yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output/calls.json
```

## What result to expect

`score.py` prints a row per paragraph and ends with:

```
10 paragraphs recorded, 5 shown on the page
11 edges across the 4 proof paragraphs and 5 in the hero graph, 16 drawn in total
a hand review flagged 3 of the 11
```

Those are the figures the task page publishes. The five paragraphs the page
leaves out are printed too, each with the edge count and the reason it was not
displayed. Four of the five returned two edges or fewer, and the fifth comes
from a filing another task page already uses.

`score.py` fails rather than skipping. It refuses a paragraph whose text no
longer matches its digest, a candidate with a missing call, a response that
does not match its `response_sha256`, a request the pinned text does not
rebuild, a relations call whose metadata is not the entities call's own output,
and a flagged edge the model never returned.

## What this does NOT establish

- **Not that the graph is correct.** Three of eleven proof edges are flagged
  here, and nothing mechanical found them. Every relation the model returns
  carries a confidence score, and all three flagged edges score high. A graph
  built from this output without a person reading the source will contain
  claims the source does not make.
- **Not an accuracy rate.** Eleven edges over four paragraphs is far too small
  to support a percentage, and the ten paragraphs were chosen to be readable
  rather than sampled from anything. Read the three flagged edges as three
  failure modes worth knowing about, not as a rate.
- **Not that offsets pin a relation.** GLiNER2 names each end of a relation by
  its text, not by an offset. Where a paragraph mentions the same text twice,
  the relation does not say which mention it means. That is exactly what the
  `other-mention` verdict on `tarsus-alkeus` records.
- **Not a model comparison.** The same ten paragraphs were also run against
  `fastino/gliner2-base-v1`. No published figure rests on it, so those calls
  are not shipped here. They are in the `superlinked/sie-web` fixture.
- **Not a benchmark.** The recorded `duration_ms` values are provenance. One
  call took 15.8 seconds against about 0.5 seconds for every other, which is
  the shape of a cold model load rather than a measurement of anything.
- **Not bound to the website.** The same run backs the fixtures in
  `superlinked/sie-web` under
  `apps/site/tests/fixtures/reference/knowledge-graph/`, which is what that
  repository's CI checks. Nothing automatically ties the two copies together,
  so they could drift.

## Inputs

`data/candidates.json` pins ten paragraphs, each with the SHA-256 of its own
text, the URL and SHA-256 of the document it was taken from, and a note saying
how the paragraph was derived from that document. Sources are SEC EDGAR
filings, NHTSA recall reports and FDA recall notices, all public records.

Each candidate also records `page_role`, which says whether the task page shows
it and why. `score.py` reads the counts off those roles, so the "5 shown" and
"11 proof edges" figures come from the pinned data rather than from anything
the display trims.

`data/review.json` holds the three flagged edges. `score.py` checks each
flagged triple against the recorded relations, so the review cannot flag an
edge the model never returned, and cannot reach an edge outside the eleven it
claims to be counting.
