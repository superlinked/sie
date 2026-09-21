# Find the photograph a shopper described

## What this shows

Six photographs and one line of text go to SIE Cloud as vectors in the same
1152-dimensional space. The query is `a red leather handbag`. No captions are
generated, no keywords are indexed and no reranker runs. A cosine over what came
back puts the handbag first.

The six were chosen so that a colour match and a category match both sit in the
set. Red shoes share the colour. A black handbag shares the category. A green
backpack, a black camera and a blue running shoe fill out the rest. The full
match beats every partial one.

The run is already recorded. Both encode calls and the vectors they returned
live in the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
pinned to one revision by `fetch.py`. Download it and you can re-derive the
published number with **no API key and no inference spend**. Those are the same
vectors behind the figures on
[superlinked.com/image-search](https://superlinked.com/image-search).

You cannot verify this by cloning alone. The clone gives you the code; the
dataset gives you the evidence. Fetching it needs no account and no token.

- Model: `google/siglip-so400m-patch14-384`, 1152-dimensional dense vectors
- Endpoint: `https://api.superlinked.com/v1/encode/google/siglip-so400m-patch14-384`
- Model revision: `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`
- Recorded 2026-09-21
- 2 calls: the six photographs in one batch, then the query

## Run it

Download the recorded run, then rank it. Both steps are standard library only,
so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

Look at a request without sending it:

```sh
python3 run.py --show images
python3 run.py --show query
```

Encode it yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output
```

## What result to expect

`score.py` prints the ranking and ends with:

```
rank     score  image
   1     0.168  Red leather handbag <- the query's subject
   2     0.068  Black handbag
   3     0.067  Red shoes
   4     0.057  Green backpack
   5     0.028  Black camera
   6    -0.008  Blue running shoe

6 photographs, 1152-dimensional vectors, one text query
the red leather handbag ranks first at 0.168
the closest other photograph is the black handbag at 0.068
a gap of 0.100, or 2.5x
```

Those are the figures the task page publishes. The handbag clears the field by a
wide margin, and the two partial matches land close together well below it.

`score.py` fails rather than skipping. An image whose bytes no longer match
their digest, a response that does not match its `response_sha256`, a request
the pinned inputs do not rebuild, a call nothing scores, a call whose served
deployment revision is not the one the manifest names, an image with no recorded
vector, a vector returned twice, a vector whose declared width disagrees with
its own values, a non-finite value, or a ranking that no longer leads with the
handbag all exit non-zero.

## What this does NOT establish

- **Not a benchmark score.** Six photographs and one query is a demonstration,
  not a measurement. There is no held-out set and no confidence interval here,
  so nothing in this output supports a percentage you could quote.
- **Not a claim about attribution.** SigLIP returns one pooled vector per
  image. The scores say which photograph the sentence matches. They do not say
  which pixels earned the match, and this example draws no boxes.
- **Not a test of scale.** Six images are ranked by a Python loop. Nothing here
  speaks to recall or index behaviour at a million images.
- **Not evidence the set is unbiased.** The six photographs and the query were
  chosen together, with the colour match and the category match put in on
  purpose. It is a designed set, not a sample of a catalogue.
- **Not bound to the website.** The same run backs the fixtures in
  `superlinked/sie-web` under `apps/site/tests/fixtures/reference/image-search/`,
  which is what that repository's CI checks. Nothing automatically ties the two
  copies together, so they could drift.

## Inputs

`evidence/inputs/images.json` holds the six photographs with the subject,
author, source URL, licence and SHA-256 of each. `score.py` checks every digest
before ranking anything. The files come from Wikimedia Commons, Unsplash and
Flickr under public-domain, Creative Commons or Unsplash terms.

`evidence/inputs/query.json` holds the one query.

`ranking.py` builds the request bodies and does the cosine. `run.py` and
`score.py` both read it from there, so the request the scorer rebuilds is the
request the runner sends, and the scorer refuses a recording it cannot rebuild.

The photographs travel as raw file bytes. SIE's SDK keeps already encoded PNG
and JPEG unchanged on the wire, so the digest recorded in `calls.json` is the
digest of what SIE read, and `inputs/` holds those exact bytes rather than a
second base64 copy of them.
