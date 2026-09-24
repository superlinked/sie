# Find the dose someone said in a noisy recording

Twelve clips of Supreme Court arguments, simulated telehealth consultations and
AMI meeting-room recordings, each sent twice to
`openai/whisper-large-v3-turbo` on SIE Cloud. The transcripts behind
[superlinked.com/speech-to-text](https://superlinked.com/speech-to-text), whose
sources are in its [SOURCES.md](https://superlinked.com/reference/speech-to-text/SOURCES.md).

Across the twelve clips a search of the transcripts finds 56 of the 61 doses,
figures, drug names and statute numbers that were registered before the run.
Five came back wrong: `800,000 GeForce units`, `DSH payment`, `Piriton`,
`Trimethoprim` and `twelve fifty Euros`. The pooled word error rate is 8.1%
over 594 human-transcribed words, and it is not evenly spread: one clip scores
2.4% and another 20.0%.

The last of the five misses is the interesting one. A speaker says "twelve
fifty Euros, so fifty percent of the selling price" of a twenty-five Euro
product, and Whisper wrote `€1250`. Anyone searching for that figure would find
a number a hundred times too large.

## Where the evidence lives

The code is here. The audio and the recorded calls are in the
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence)
dataset on Hugging Face, pinned to a revision in `fetch.py`. So you cannot
verify this by cloning alone: clone, fetch, then score. What you do not need is
an API key, or a cent of inference spend, to re-derive the numbers.

```
speech-to-text/
  inputs/inputs.json             the human transcripts, the key terms and the
                                 clip bounds, all fixed before any model call
  inputs/audio/                  the twelve clips as sent, 910,764 bytes
  inputs/normalizer.json         the Whisper spelling map both counts depend on
  inputs/scoring_amendment.json  two post-run changes to the key-term rule
  calls.json                     24 entries: request, response, status, timing
  manifest.json                  endpoint, model id, served revision, run date,
                                 sources
```

## Run it

Download the recorded run, then score it. Both steps are standard library only,
so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

Expect a per-clip line, then:

```
Across all 12 recorded clips a search finds 56 of 61 key terms, and 5 came back wrong
Pooled word error rate 8.1% over 594 human-transcribed words
```

`score.py` also checks the published per-clip figures for four of the clips. It
exits non-zero if any of those, or either headline, is not what it computes. All
twelve are scored either way.

It checks figures, never composition. Which clips the page plays, how many, and
which of the wrong terms it shows are decided in sie-web; nothing here can reach
the page to read them, so a constant asserting them would go stale on the next
reselection while this script still exited 0.

Look at a request without sending it, and check that this runner is the one
that sent them:

```sh
python3 run.py --show primock-uti-antibiotics
python3 run.py --check-requests   # 24 of 24 rebuilt identically
python3 run.py --verify-audio     # 12 of 12 clips verified
```

Send the calls yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output
```

`run.py` writes into `run-output/`, never over the downloaded evidence. Your
transcripts will differ from the recorded ones: the served model revision moves.

## The figures depend on the normalizer, and that is not a detail

Both published counts are computed after OpenAI's Whisper English text
normalizer. It lowercases, removes filler words and transcriber tags, expands
contractions, applies a British-to-American spelling map and writes spoken
numbers as digits. Run the same transcripts through a different normalizer and
you will get different figures, so `whisper_normalizer.py` here is a standard
library port of the Transformers 4.57.6 implementation, whose normalization is
unmodified from it. Its docstring names the three things that do differ, none of
them in the algorithm. It is loaded with the spelling map the run recorded, which
`fetch.py` downloads alongside the calls. `score.py` hashes that map before it
scores, and a missing one is a failure naming both figures it takes away, rather
than a silent fall back to an empty map, which would still produce numbers. The
same holds for every file the scorer opens: each is checked for existence before
it is read, so a partial fetch is a named failure and never a traceback.

Word error rate is `(substitutions + deletions + insertions) / reference words`
by word-level Levenshtein, pooled by summing edits and reference words over all
twelve clips rather than averaging per-clip rates.

A key term is found when some contiguous run of normalized transcript words
spells its content. `matching.py` holds that rule in one place, and normalizes
nothing itself: it compares tokens the normalizer has already rewritten. Two
amendments,
recorded in `scoring_amendment.json` and applied after the run to every term,
clip and call: one can only remove a hit, requiring a money term to write the
amount actually spoken, and one can only add a hit, dropping a currency symbol,
a leading article and the decimal point inside a number from both sides before
comparing. Neither touched a transcript and neither repeated a call.

## The recordings are not ours

Every clip is third-party and none is synthetic. Five are Supreme Court
argument recordings in the public domain, with the Court's own official
transcripts as the reference and Oyez sentence alignment used for timing only.
Four are PriMock57 simulated consultations, CC BY 4.0, from Babylon Health.
Three are AMI Meeting Corpus recordings, CC BY 4.0, from the University of
Edinburgh. `manifest.json` records for each one its title, setting, speaker,
licence, source URL, transcript URL, clip bounds and SHA-256. Licences vary by
source and have not been cleared for reuse beyond quotation here; treat the
provenance record as the starting point for that, not as a clearance.

## What this does not establish

- Not a benchmark. Twelve clips chosen to span courtroom, clinical and
  meeting-room audio is a demonstration, not a measurement, and 8.1% is not
  comparable to a published WER on a standard test set.
- Not a claim about your audio. Every clip is English, under a minute, and cut
  at a published human timing. Nothing here tests a long recording, a
  cross-talk-heavy one, or a language other than English.
- The key terms are ours. 61 terms across 12 clips were chosen before the run
  as the things an agent would search for, and a different choice would give a
  different ratio.
- The references are the publishers' transcripts, which are themselves human
  work with their own conventions and errors. Substituting a different
  reference would move the WER.
- The five wrong terms are not equally serious. `Piriton` and `Trimethoprim`
  are spelling failures on drug names; `€1250` for `€12.50` is a wrong number.
- Not reproducible against the live API. These are recordings, and a rerun goes
  through a different served revision.
- All twelve clips and all 61 terms are scored here. How many of either reach a
  reader is a display decision made in sie-web and is not checked here.
- The control call, sent with no instruction, is recorded and scored. It finds
  the same 56 of 61 at a pooled 8.75%, and it is in no published figure.
