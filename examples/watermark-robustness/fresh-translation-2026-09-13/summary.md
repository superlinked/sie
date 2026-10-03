# Fresh translation and watermark-strength pilot

21 distinct prompts × four conditions × two versions = 168 scored records. The four conditions share prompts; these are 84 source generations, not 168 independent prompts.

Generation: Qwen2.5-1.5B-Instruct on CPU; three experimental keys. Translation: local MADLAD-400-3B float32, English → Arabic → English. Detector counts unique token pairs; threshold is strictly z > 3.

| Bias | Pairs | Mean z before | Mean z after | Detected before | Detected after | Mean length ratio | Mean cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (unwatermarked) | 21 | -0.13 | -0.11 | 0/21 | 0/21 | 0.99 | 0.946 |
| 2 | 21 | 4.48 | 2.48 | 19/21 | 8/21 | 1.00 | 0.937 |
| 3 | 21 | 7.11 | 3.56 | 21/21 | 15/21 | 1.01 | 0.932 |
| 4 | 21 | 9.23 | 4.07 | 21/21 | 15/21 | 1.05 | 0.928 |

All pairs are included; no results are removed for producing an inconvenient score.

Source token-cap flags: 2/84. Translation stop flags: 2/84. Length-review flags (ratio below 0.7 or above 1.4): 1/84.

Embedding input truncation flags: 1/84. These pairs exceed the embedding model's input limit on at least one side.

Cosine similarity is an embedding proxy, not factual equivalence or a percentage of meaning preserved. Length flags are prompts for review, not a semantic-quality test. Twenty-one controls per condition cannot calibrate a rare false-positive rate. The keys are assigned to groups of seven prompts, so key and prompt effects are not independently isolated.

Full prompts/settings and library versions are in manifest.json. Exact source token IDs and finish reasons are in originals.json. Both translation directions, sentence outputs and batch membership are in translations.json. Per-pair scores and quality proxies are in results.json.

## Sensitivity check

The primary table above includes every pair. The table below excludes source/translation stop flags and the predeclared length-review flags. Passing these checks does not establish semantic equivalence.

| Bias | Pairs passing checks | Mean z before | Mean z after | Detected before | Detected after |
|---|---:|---:|---:|---:|---:|
| 0 | 21 | -0.13 | -0.11 | 0/21 | 0/21 |
| 2 | 21 | 4.48 | 2.48 | 19/21 | 8/21 |
| 3 | 20 | 7.13 | 3.55 | 20/20 | 14/20 |
| 4 | 19 | 8.94 | 4.11 | 19/19 | 14/19 |

See manual-review.md for qualitative observations about this run. A low score on a failed translation is not evidence of meaning-preserving watermark removal.
