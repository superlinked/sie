# Read the numbers off a dashboard that has no API

Twelve screenshots of Superset, Argo CD, Airflow, the Kubernetes Dashboard,
GitLab and Jaeger, one call each to `Qwen/Qwen3.8-27B-FP8` with a strict
JSON-schema grammar, and a thirteenth call for the playground snippet.
The extractions behind
[superlinked.com/screenshot-mining](https://superlinked.com/screenshot-mining),
whose sources are in its
[SOURCES.md](https://superlinked.com/reference/screenshot-mining/SOURCES.md).

Across the twelve screens, 328 of 335 expected values came back exactly as the
screen shows them. Seven did not. Nine of the twelve screens were perfect,
including the 55-value GitLab DORA dashboard and the 32-value Argo CD
ApplicationSets table. The seven misses fall on three screens: a Superset tile
the reply left out entirely, costing three checks, plus a timezone count read as
410 where the screen shows 416; a GitLab branch read as `ruby_3_2` where the
screen shows `ruby3_2` and a merge request read as `!147325` where the screen
shows `147325`; and a Grafana version read as `v18.2.1` where the screen shows
`v19.2.1`.

## Where the evidence lives

The code is here. The screenshots and the recorded calls are in the
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence)
dataset on Hugging Face, pinned to a revision in `fetch.py`. So you
cannot verify this by cloning alone: clone, fetch, then score. What you do not
need is an API key, or a cent of inference spend, to re-derive the number.

```
screenshot-mining/
  inputs/inputs.json     expected values and scoring rules, read off each
                         screenshot at full resolution before any model call
  inputs/images/         the twelve screenshots, 3,735,115 bytes
  calls.json             13 entries: request, response, status, timing
  manifest.json          endpoint, model id, served revision, run date, sources
```

The screenshots were never committed to either repository. Each request record
carries a `$payload` descriptor instead of the base64 bytes, giving the media
type, byte length, SHA-256 and a commit-pinned raw URL. The twelve files were
re-fetched from those URLs on 2026-09-20, every digest still matched, and the
bytes went into the dataset. The descriptors inside the recorded entries still
say `stored: false`, because they are part of an entry whose digest has to keep
verifying; `calls.json` explains that in its `images_note` and maps each call to
the file that now holds its bytes.

## Run it

Download the recorded run, then score it. Both steps are standard library
only, so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

Expect a per-screen line, then:

```
Across all 12 recorded screens, 328 of 335 values matched the screen and 7 did not
The playground call adds 4 of 4, for 332 of 339 over every recorded call
```

`score.py` also checks three per-screen figures the page publishes, pinned in
`PAGE_PER_SCREEN`, and exits non-zero if any of them, or the headline, is not
what it computes. That catches a rescore moving a published number quietly.

It does not check which screens the page draws, and it cannot: nothing here
reads the page. Tampering shows the gap. Change a figure and the run fails;
swap the slug for another screen with its correct figures and the run stays
green. That is why this file no longer says what the page's grid contains. The
page's `SOURCES.md` is where the selection is recorded.

Look at a request without sending it:

```sh
python3 run.py --show gitlab-runner-fleet
```

Send the calls yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output
```

`run.py` writes into `run-output/`, never over the downloaded evidence, and in
the same entry shape the published `calls.json` uses. Your extractions will
differ from the recorded ones: nothing pins a seed, and the served model
revision moves. `python3 run.py --verify-images` hashes every fetched
screenshot against the digest `inputs.json` pins, and needs no key.

## How a value is scored

One check per expected leaf value, row keys included. A missing row fails every
leaf in it, and each run of extra rows is one more failed check. Strings compare
after NFKC normalization, with the minus sign and en dash read as a hyphen, the
ellipsis read as three dots, all whitespace removed and case folded. Numbers
compare numerically and a string never matches a number. An expected null
matches only null. Lists compare by position unless a case registers a by-key or
order-free rule. `inputs.json` records those rules, and they were fixed before
the run.

`score.py` checks the bytes before it scores them. The checks fall into two
layers, and they catch different attacks.

**The metadata files.** `calls.json`, `manifest.json` and `inputs/inputs.json`
are hashed whole, before anything is parsed, against object ids pinned in this
repository beside the dataset revision. That is the layer that defeats a
coordinated edit: every other digest here travels inside those three files, so
someone who changes a recorded response and recomputes the `recorded_sha256`
sitting beside it satisfies all of them, and the run still prints the published
figure. Only a value pinned outside the evidence sees that. These are the ids
HuggingFace publishes for the revision, so you can check them by hand:

```sh
curl -s "https://huggingface.co/api/datasets/superlinked/sie-task-evidence/tree/d28e188a47be3a8c5ce3f198c0ee967d3e6c1c65/screenshot-mining?recursive=true"
```

**The image files.** No image id is pinned here, so the object ids above would
not notice a modified image. Two digests cover that instead: each file is
hashed against the `$payload.sha256` of the request that sent it, and that
digest is held against the `file_name` and `image_sha256` the case registered
in `inputs.json`. The first catches an image edited while the metadata is left
untouched. The second catches an image that hashes correctly for the call
carrying it but is not the one the case pinned, so one screen cannot be scored
against another's expected values. Editing an image *and* the digests that
describe it means editing the metadata, which is the first layer again.

The remaining checks are consistency within the evidence: `inputs.json` against
the digest the run recorded, every response record re-digested the way the run
digested it, and each entry's `slug` against `<case>__<call>`, so an edit to
one field cannot have the evidence validated as one screen and the reply scored
against another's expected values. An unrecognised `kind` or `role`, and a
recorded call that no case reaches, are failures too rather than silent passes.
Anything missing, or that any check disagrees about, is a failure and nothing
is scored; it is never skipped past.

It does not recompute `entry_sha256`, the RFC 8785 canonical digest the
recording carries over each whole entry. That needs a JSON canonicalizer, which
this repository already ships at
`examples/document-to-markdown/document_to_markdown/canonical.py` and the
website ships in TypeScript. Nothing here duplicates it, and a fresh `run.py`
writes no `entry_sha256` either. It matters less than it looks: the object id
pinned above covers every byte of `calls.json`, including the entry fields this
scorer never reads on their own, such as `timing`, `retry_count` and
`model_revision`.

## The inputs are not ours

Every screenshot is third-party, and each one is the upstream project's own
published image of its own product, taken from that project's documentation at a
pinned commit. None were made by Superlinked, and none are AI-generated. Eight
are Apache-2.0: two from Apache Superset, two from Argo CD, two from Apache
Airflow, one from the Kubernetes Dashboard and one from Jaeger. Four are
CC-BY-SA-4.0 documentation images from GitLab. `manifest.json` records for each
one its application, screen, publisher, licence, the documentation page it came
from, the raw URL and its SHA-256. Licences vary by source and have not been
cleared for reuse beyond quotation here; treat the provenance record as the
starting point for that, not as a clearance.

## What this does not establish

- Not a benchmark. Twelve screens chosen to span dashboards, pipeline lists,
  admin tables and metric tiles is a demonstration, not a measurement.
- Not a claim about your screens. 328 of 335 is the score on these twelve. Nine
  screens were perfect and three were not: 28 of 32, 17 of 19 and 51 of 52.
- The seven misses are small: one omitted tile, one wrong count, two characters
  in a branch name, a stray `!`, one digit in a version string. Nothing in this
  run tests a screenshot at a resolution where the text is genuinely unreadable.
- Not reproducible against the live API. These are recordings. A rerun goes
  through a different served revision and unfixed sampling, so it will differ.
- Not a description of what the page shows. All twelve screens are scored here,
  and so is the playground call, which is scored separately and is not part of
  the 335. Which of them the page draws, and where, is the page's decision and
  is recorded in its own `SOURCES.md`.

  This bullet used to name the grid, the hero and the playground screens, and it
  was wrong twice inside two days: once naming four screens, then naming three
  from a page change that had not shipped. Every run stayed green through both,
  because nothing here can read the page. The claim is removed rather than
  corrected a third time. No figure moved and no call was rescored in any of it.
