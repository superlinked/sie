# Measure how translation and paraphrase weaken a text watermark

This standalone SIE example generates text with a watermark key we control,
translates or paraphrases it, then measures the remaining signal. It tests
our own marked output, not a commercial assistant's watermark or arbitrary
AI writing. The educational article is in [EXPERIMENT.md](EXPERIMENT.md).

## What this shows

The workflow is: **local marked text → SIE rewrite → local detector**.
An English → Arabic → English round trip uses two generation calls.
A separate one-pass paraphrase uses one. The detector compares distinct
token/context pairs under the original numeric watermark key; the SIE API
key only authorizes service access.

## Run it

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

For a hosted demo, sign in to the [Superlinked console](https://console.superlinked.com)
and obtain a key for your account. Confirm model access and credits first.
If hosted access is unavailable, [Superlinked](https://superlinked.com/)
offers an inference-grant application.

```bash
uv run --frozen python run_paraphrase.py --ask-key --limit 1 \
  --arms rt_ar paraphrase_1x --output runs/my-sie-demo.json
```

The hidden key prompt does not save your key. Alternatively supply
`SIE_API_KEY` using your environment or secret manager.
`SIE_BASE_URL` or `--sie-url` overrides `https://api.superlinked.com`.
Never put a real key in source code, notebooks or recorded results.

This command uses the first **saved** watermarked paragraph and makes up to
three hosted generations. Scoring runs locally on CPU; tokenizer and
embedding-model assets download if missing. No source-generation weights
are needed for this path.

To use your own self-hosted SIE generation endpoint instead, enable a model
using the [SIE text-generation setup](https://superlinked.com/docs/generate),
then select the same model in the demo:

```bash
SIE_GENERATOR_MODEL=Qwen/Qwen3-0.6B \
uv run --frozen python run_paraphrase.py --sie-url http://localhost:8080 \
  --limit 1 --arms rt_ar paraphrase_1x --output runs/local-sie-demo.json
```

An unauthenticated local server needs no key. Use the port where your
generation server actually runs (the documented Apple Silicon setup uses
8081). A different model is a new comparison, not a reproduction of the
archived cloud table.

### What you will see

The runner prints the source's mean z-score, then a score and detection count
for each selected transformation. Values are measured at runtime; there is
no required outcome for the one-passage demo.

Results go into a new JSON file with transformed text, source text and score,
watermark parameters and the requested model. Existing outputs are refused.
Failed or incomplete responses stop the run and leave completed records
intact. The demo does not automatically retry potentially billable generation.
Inspect account usage before manually retrying a timed-out request.
The hosted runner saves final English, not the intermediate Arabic.

For all eight saved sources and all four transformation arms, use
`--limit 8 --arms rt_ar paraphrase_1x paraphrase_2x paraphrase+rt_ar`
with a new output filename. This requests up to 64 generations.

## Models and SIE features used

| Stage | Model | Where it runs |
|---|---|---|
| Marked/control source | Qwen/Qwen2.5-1.5B-Instruct | Local Transformers, CPU |
| Translation and paraphrase | Qwen/Qwen3.8-27B-FP8 by default | `SIEClient.chat_completions` |
| Optional entity protection | urchade/gliner_multi-v2.1 | `SIEClient.extract` |
| Embedding comparison | sentence-transformers/all-MiniLM-L6-v2 | Local CPU |
| Separate strength pilot's translator | MADLAD-400-3B CTranslate2 | Direct local execution, **not SIE** |

`config.yaml` is a descriptive reference, not a runtime configuration file
or a complete set of revision pins. Scripts read their constants,
`SIE_GENERATOR_MODEL`, CLI arguments and saved source metadata. The archived
cloud run requested Qwen/Qwen3.6-27B; the updated default does not reproduce
that model. Check the configured endpoint's catalog before running.

## Inspect the recorded run

No key, model download or Python is needed to read the data:

- `samples.json`: sixteen archived watermarked/control source passages from
  eight prompts. This older file lacks exact generation-time token IDs and
  seeds; the revised generator saves these for new runs.
- `paraphrase-results.json`: archived SIE transformations. Original scores
  are historical; `review-results.json` contains the corrected scores.
- `results.json`, `results-cloud-qwen.json`, `distance-sweep-results.json`:
  other archived translation runs included for offline auditing.
- `fresh-translation-2026-09-13/records.csv`: all 168 fresh text/score rows.
  Match `id` across `arm` values to compare a source before and after.
- That fresh directory also contains `originals.json`, `translations.json`,
  `manifest.json` and `results.json`: prompts, source token IDs, Arabic
  intermediates, returned English, settings, revisions and quality flags.

These are synthetic ordinary-writing tasks, not a human-authored corpus.
The fresh run uses 21 prompts at four settings and two text versions:
84 sources and 168 scored records, **not 168 independent prompts**.
One of three keys is assigned to each seven-prompt group, not all three
keys crossed with every prompt.

### Recorded results

First, the archived eight-prompt SIE comparison, re-scored with the corrected
detector:

| Text | Mean z | Detected at z > 3 |
|---|---:|---:|
| Original watermarked | 10.86 | 8/8 |
| English → Arabic → English | 5.91 | 8/8 |
| One paraphrase pass | 0.69 | 0/8 |
| Two paraphrase passes | 1.10 | 0/8 |
| Unwatermarked model control | −0.09 | 0/8 |

Second, the separate local MADLAD pilot:

| Watermark bias | Mean z before | Mean z after | Detected before | Detected after |
|---|---:|---:|---:|---:|
| 0: control | −0.13 | −0.11 | 0/21 | 0/21 |
| 2 | 4.48 | 2.48 | 19/21 | 8/21 |
| 3 | 7.11 | 3.56 | 21/21 | 15/21 |
| 4 | 9.23 | 4.07 | 21/21 | 15/21 |

Three pairs have stop/length problems, including one repetitive translation
failure. The primary table keeps them; `summary.md` in the fresh directory
gives the sensitivity check. These measurements do not guarantee identical
results from a rerun or prove that any rewrite preserves every claim.

## Test offline

```bash
uv run --frozen python -m unittest -v test_review test_demo
```

The tests make no server calls. To re-score the older saved texts, fetch only
the generator's tokenizer and configuration first if uncached:

```bash
uv run --frozen python -c "from transformers import AutoConfig, AutoTokenizer; m='Qwen/Qwen2.5-1.5B-Instruct'; AutoConfig.from_pretrained(m); AutoTokenizer.from_pretrained(m)"
HF_HUB_OFFLINE=1 uv run --frozen python audit_saved_results.py
```

This writes a derived `review-results.json`, preserving all raw data.

## Generate new data

To generate new source/control passages on CPU, without replacing the archive:

```bash
uv run --frozen python generate_watermarked.py --output runs/new-sources/samples.json
uv run --frozen python run_paraphrase.py --ask-key \
  --samples runs/new-sources/samples.json --output runs/new-sources/sie-demo.json
```

Generating source texts downloads Qwen2.5-1.5B weights if absent. The separate
local strength pilot also needs MADLAD, so expect several gigabytes of
downloads and memory. It uses **no SIE key or API**:

```bash
uv run --frozen python fresh_translation.py generate --allow-downloads --out-dir runs/local-pilot
uv run --frozen --with ctranslate2==4.8.1 python fresh_translation.py translate --allow-downloads --out-dir runs/local-pilot
uv run --frozen python fresh_translation.py report --allow-downloads --out-dir runs/local-pilot
```

Use the same new `--out-dir` in all phases. Completed batches resume.
Omit `--allow-downloads` once assets are cached. The published dataset is
protected against generation/translation writes.

## Limitations

The detector fixes repeated-pair counting and uses stable normal-tail
p-values. These remain nominal, not empirically calibrated false-alarm rates.
A low score means insufficient evidence under this key, not human authorship.
An embedding cosine is not a percentage of meaning preserved.

`run_eval.py` includes the optional entity-protection arm; `distance_sweep.py`
explores seven languages. These advanced runners retain historical output
defaults: use a disposable copy if running them. They are not required for
the commands above. Archived cloud outputs lack full response metadata and
cannot be labeled a newly verified SDK run.
