# Tag people, organizations and locations in ordinary text

Extract named entities from 24 verbatim paragraphs published in 2026: science,
software, standards and transport. The task uses the same three labels across
all four domains, including seven paragraphs with no named entities.

The answer key and seven finite acceptable representations were reviewed before
model outputs. Repeated mentions count once per typed name; each independent
source family contributes equally. These paragraphs test a declared engineering
fit, rather than representing the distribution of all documents.

## Reproduce scores without inference

From this directory, run the offline checks:

```bash
python3 -m unittest discover -s tests -v
```

Score a recorded JSONL file, or compare several files against the same answer key:

```bash
python3 score.py --rows large.jsonl --rows medium.jsonl --rows comparison.jsonl
```

Each row contains `id`, `model`, `status` (`ok`, `failed` or `unattempted`) and
the input's `text_sha256`. Successful rows contain `entities` with `text` and
`type` (or the native `label` field). Native recordings also retain `start`,
`end` and `score`. Invented comparison names count as false positives rather
than being discarded. Gold and native spans must be exact source substrings; native offsets are
Python Unicode character positions, with an exclusive end.

The report shows strict and prospectively accepted tag-set F1, pooled precision
and recall, entity-bearing F1, exact documents and empty-source false tags.
Failed and missing planned calls score zero, including empty-source cases.
Comparisons use a paired bootstrap over source families, never individual
mentions. An acceptable representation is a complete fixed answer set;
arbitrary unions or new post-output interpretations are not accepted.

The prospective useful-fit gate requires all 24 calls to succeed, at least 12
entity-bearing sources, pooled precision of at least 90% and entity-bearing
mean F1 of at least 80%. Passing that point gate does not establish population
noninferiority against another model.

## Collect new native recordings

Install the Python SDK from the repository root:

```bash
python3 -m pip install ./packages/sie_sdk
```

Use an explicitly configured SIE endpoint serving the selected model. Set
`SIE_API_KEY` only when that endpoint requires authentication. From this
directory:

```bash
python3 record.py --url https://api.superlinked.com \
  --model gliner-community/gliner_large-v2.5 --output large.jsonl
python3 record.py --url https://api.superlinked.com \
  --model gliner-community/gliner_medium-v2.5 --output medium.jsonl
```

The recipe uses `SIEClient.extract`, plain `person`, `organization`, `location`
labels and the model profile's fixed threshold: 0.75 for Large, 0.55 for Medium.
The expected checkpoint revisions are recorded separately from the returned
model identity; this client does not independently attest the server's loaded
checkpoint.

Each invocation allows at most 24 physical extraction POSTs and starts no new
call after its 1,200-second wall budget. Calls run serially, with connect/read
timeouts capped at 60 seconds and the remaining budget. A request hook prevents
SDK retries from sending a second POST. The first
failed call stops further inference and writes every remaining source as
unattempted. Existing recording files are never overwritten. New inference can
incur endpoint charges; offline scoring does not.

Recordings allowlist semantic entities and token counters. They exclude API
keys, authentication headers, request IDs, endpoint addresses and exception
messages. Costs require a separately pinned rate card and complete metering;
the scorer does not substitute hardware spend or invent missing prices.

## Source rights and annotation

The bundled passages retain their source URLs, publication dates, content hashes
and rights in `inputs/cases.jsonl`. See [ATTRIBUTION.md](inputs/ATTRIBUTION.md),
the [annotation policy](inputs/annotation-policy.md), and the retained
[license notices](inputs/licenses). USGS and NTSB staff text is U.S. government
work; NIST permits redistribution of its unmarked text. Go content uses CC BY
4.0; Rust and Node.js content retain their respective MIT and Apache notices.
The passages are prose-only extractions, without images or logos.
