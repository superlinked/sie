# Fresh local experiment

This run replaces the missing-text strength comparison with newly generated,
auditable data. It is separate from the older eight-passage cloud experiment.

Design: 21 distinct prompts, each generated without a watermark and with
green-list biases 2, 3 and 4. Each original is scored before and after an
English → Arabic → English round trip: 84 source passages and 168 scored
records. Three keys are used, one per group of seven prompts. Prompt/key
groups are shared across bias conditions. This is a paired exploratory
comparison, not 168 independent prompts or a calibration study.

The generator is cached Qwen2.5-1.5B-Instruct on CPU. The translator is cached
MADLAD-400-3B, using its float32 CTranslate2 artifact and greedy decoding.
The three watermarked settings are experimental choices; none is described
as representative of a commercial deployment. This run does not test paraphrasing.

Files:

- `manifest.json`: prompts, keys, model revision, generation settings and versions.
- `originals.json`: English source texts, exact token IDs, attention masks,
  sampling batch/seeds, finish reasons and initial scores.
- `translations.json`: intermediate Arabic and returned English, sentence-level
  token outputs, stop indicators, decoder settings and translation batch IDs.
- `results.json`: both detector scores for each pair, output-length ratios,
  embedding cosine similarities, completeness flags and summaries.
- `records.csv`: the 168 text-and-score records in spreadsheet-friendly form.
- `summary.md`: the generated result table and limitations.

The published data in this directory is preserved. To generate a separate
experiment, run the three phases from the parent folder, using the same
new output directory. Completed batches resume; omit `--allow-downloads`
when all assets are already cached:

```bash
uv run --frozen python fresh_translation.py generate --allow-downloads --out-dir runs/local-pilot
uv run --frozen --with ctranslate2==4.8.1 python fresh_translation.py translate --allow-downloads --out-dir runs/local-pilot
uv run --frozen python fresh_translation.py report --allow-downloads --out-dir runs/local-pilot
```

Generation/reporting require the existing evaluation dependencies. Translation
requires `ctranslate2==4.8.1` and the tokenizer/artifact. In the recorded
September 13 run, generation/reporting used the evaluation environment (Torch 2.14.0,
Transformers 4.57.6); translation used the cached SIE environment (Torch 2.9.1,
Transformers 4.57.6, CTranslate2 4.8.1). Reporting explicitly checks that
original scores agree with the scores saved at generation time.

All completed pairs are reported. No source is dropped based on its watermark
score. Length ratios outside 0.7–1.4 and generation/translation stop flags
identify passages for inspection. These checks and embedding similarities
do not substitute for human judgments of semantic equivalence.

The local artifact uses a T5 Unigram/Metaspace tokenizer. The runner explicitly
disables an inapplicable Mistral-tokenizer fix: Transformers 4.57.6's warning
heuristic misidentifies the converted model's config, which has no model_type.
The tokenizer's actual pre-tokenizer type is checked before translation.
