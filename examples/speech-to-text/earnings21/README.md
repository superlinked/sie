# Earnings21: all 219 pieces through a SIE URL

This standalone example makes a fresh submission of every frozen Earnings21
piece through `sie_sdk.SIEClient`. Its scientific denominator is all 44 calls;
the transport submits 219 PCM16 mono 16 kHz WAVs. It produces transcripts and
request evidence. Scoring is a separate, explicit command.

The default arm is `Qwen/Qwen3-ASR-1.7B-hf`, with exactly
`{"language": "en", "max_new_tokens": 8192}`, no instruction, and concurrency 8.
Provide an HTTP(S) `SIE_URL` already serving the chosen model. A hostname, port,
and base path are supported. This example does not load a model or install a
serving deployment. The historical Qwen source is
[b1104d2d20e6800fa68a6283b65f796a2526195e](https://github.com/superlinked/sie/tree/b1104d2d20e6800fa68a6283b65f796a2526195e);
that model's catalog entry is absent from the current main checkout. The requested
model and source checkpoint are recorded separately from any revision actually
reported by the server. `--model` records a distinct override arm; it does not
establish which checkpoint the URL serves.

From a SIE clone, set up only this project's Python 3.12 environment:

```bash
cd examples/speech-to-text/earnings21
mise exec -- uv sync --frozen --project .
mise exec -- uv run --frozen --project . python fetch.py
```

The fetch command downloads and verifies only small public evidence: the manifest,
original call frame and references, unchanged builder, Amendments 3 and 4, source
README and scorer files. It neither downloads audio nor runs decoding or scoring.
`--from-dir` can copy an existing exact evidence directory instead. The immutable
[source.json](source.json) pins the
[public evidence revision](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/457c33de3d7188db55e0fb74b17f9f43c34c49c5/speech-to-text-e21/2026-10-08-amendment-4/manifest)
and [additive decoder notes](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/fba647ac74876f960f8e8840831bd9745562e748/speech-to-text-e21/2026-10-10/reproducibility).

To explicitly fetch the 44 frozen MP3s (about 560 MB), then prepare the roughly
4.5 GB of WAVs, install `ffmpeg` and run:

```bash
mise exec -- uv run --frozen --project . python fetch.py --audio-dir data/mp3
mise exec -- uv run --frozen --project . python prepare.py
mise exec -- uv run --frozen --project . python run.py --verify-only
```

`prepare.py` verifies every original MP3 size and hash before invoking the pinned
`make_pieces.py` byte-for-byte. The fixed-point MP3 decoder, decoded PCM hashes,
recorded sample offsets and Python wave headers define the expected bytes. It
never searches for new offsets or normalizes references. Decoder builds can differ;
a PCM mismatch stops preparation. No verified derived-WAV archive is identified
as a fallback. `prepare.json` records decoder/version provenance; only successful
qualification of all 219 size/hash pairs establishes compatible WAV bytes.

After verification, choose a fresh output directory and a caller wall window:

```bash
mkdir -p runs
export SIE_URL='http://localhost:8080'
# Set SIE_API_KEY explicitly if the chosen URL requires authentication.
mise exec -- uv run --frozen --project . python run.py \
  --output runs/fresh-001 --case-seconds 1200 --whole-seconds 21600
```

`--url` overrides `SIE_URL`; credentials, query strings and fragments in that base
URL are rejected. An absent `SIE_API_KEY` becomes an explicit empty SDK key.
Redirects, ambient proxies and HTTP transport retries are disabled. Defaults are
1,200 seconds per piece and 21,600 seconds for the whole run; both must be finite
and positive. They include startup, serialization, reading and worker closure;
the whole window reserves bounded shutdown time. These limits are caller choices,
not a promise of completing the frame on a particular server. `--concurrency`
accepts 1 through 8 and is recorded; the frozen arm's concurrency is 8.

Before dispatch, the caller qualifies the complete sorted 44-call / 219-piece
frame, reference hashes and every WAV path, size and hash. Each active worker
rechecks its bytes at physical handoff. The stock 0.10.0 SDK sends one item as
MessagePack with native audio bytes and `sample_rate: null`; the older evidence's
JSON/base64 description is not a replacement request implementation.

Every worker durably records an exclusive POST intent before physical egress.
A transport fence rejects every second POST, including SDK model-loading and
admission retries. Complete responses are persisted with actual status, safe
headers, body representation, digests, and an explicit completion flag before
the SDK interprets them. Gzip/deflate expansion is bounded. The unchanged SDK
return is saved separately. Known API-key redactions are marked; authentication
headers and continuation tokens are omitted from evidence.

A complete 303 is pending work. The stock SDK may follow its documented
same-origin continuation GET within the original piece budget, without another
POST. An intent without a complete terminal outcome, including a lost or partial
continuation, remains consumed `UNKNOWN`. UNKNOWN, persistence failures and fatal
closure failures stop replacement dispatch. Existing workers settle within their
remaining budgets and every owned process is terminated, killed if needed, and
joined. An existing output directory is never executable; there is no screen,
subset, resume or retry command.

`checkpoint.json` atomically binds the complete piece and call tables.
`pieces.jsonl` and `calls.jsonl` expose those tables; `hypotheses.jsonl` always
contains all 44 scorer rows `{id,hyp,error}`. Per-piece directories retain the
intent, complete/partial HTTP evidence, SDK return and terminal row. Empty,
short and capped successful text is retained with `output_tokens`,
`finish_reason`, `cap_state`, usage and timing. Complete failed responses contribute
empty piece text. Failed, UNKNOWN and unattempted positions are empty strings
when texts are joined in original piece order with literal single spaces.
Successful text survives in a partly failed call, with separate error and
completeness metadata. Exit 0 means all pieces succeeded, 1 means retained
incomplete/failed results, and 2 means a caller/setup failure.

The frozen primary comparator is P2, `gpt-transcribe` on the same piece bytes;
whole-call P1 is secondary. If a separately obtained P2 call JSONL is available,
run the unchanged scorer explicitly:

```bash
mise exec -- uv run --frozen --project . python data/evidence/scoring/e21_score.py \
  --refs data/evidence/references.jsonl --ids data/evidence/all_ids.txt \
  --calls data/evidence/calls.json --arm sie=runs/fresh-001/hypotheses.jsonl \
  --arm gpt-transcribe=/path/to/p2-calls.jsonl --vs gpt-transcribe \
  --seed 20261006 --out runs/fresh-001/score.json
```

The scorer's hash is
`9f4fa3110b20b55bed119a4c92b001af18770cef9acca8e842bc217e0ec9b6fd`.
It checks the three unchanged supporting scorer pins in the evidence manifest.
Content-word error rate governs the claim; report WER beside it. The unit is
the call, all 44 calls, with 4,000 call-bootstrap resamples and seed 20261006.
The 95% upper bound of S3 minus P2 must be below 0 for “fewer content-word
errors,” or at most +0.5 percentage points for “as accurate.” Report both the
primary analysis and the sensitivity that drops calls where either arm omits
at least 60 seconds; a claim must hold in both. Qwen training overlap with
Earnings21 is undisclosed. This caller supplies no new accuracy or cost result.

Offline controls use exact pinned small metadata and tiny synthetic PCM16 WAVs,
with external sockets forbidden. They do not qualify original audio, decode it,
run a model/provider, or execute the scorer:

```bash
mise exec -- uv run --frozen --project . python -m unittest discover -s tests -v
mise exec -- uv run --frozen --project . ruff check --select E,F,I,UP,B .
mise exec -- uv run --frozen --project . ruff format --check .
```

References and audio are Earnings21, Rev.com, under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), from
`hf-audio/asr-leaderboard-longform` revision
`d6797370d3189c618e722721ab5b6c9be78c022c`, `earnings21` test.
The frozen derivatives are MP3 re-encodings and piece cuts. The fetched source
README retains attribution and the normalizer's MIT/Apache-2.0 provenance.
Downloaded evidence, audio, environments and run outputs are ignored by Git.
