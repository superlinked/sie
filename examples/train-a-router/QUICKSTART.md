# Train a router: one-page guide

Train routers to choose between **GPT-6.1 Sol and GPT-6 Luna**, then compare money saved against answer accuracy. Try Laya, adapt it with LoRA, and train small classifiers on **Qwen3-Embedding-4B** vectors from SIE Cloud or your own SIE server.

**Results on 3,043 test questions:**

| Policy | Accuracy | Accuracy loss | Cost / 10,000 | Savings |
|---|---:|---:|---:|---:|
| Always Sol | 92.64% | 0.00 pp | $14.02 | 0.0% |
| Embedding router, more quality retained | 91.65% | 0.99 pp | $9.09 | 35.2% |
| Embedding router, more money saved | 90.67% | 1.97 pp | $6.94 | 50.5% |
| Always Luna | 88.63% | 4.01 pp | $1.52 | 89.2% |

Always Sol is the baseline. **pp means percentage points:** 92% to 91% accuracy is a 1 pp loss. Costs include Cloud encoder fees and exclude classifier application hosting. Step 4 compares the full curves, with Laya hosting estimated separately in the [README](README.md).

**Data:** 9,111 training, 3,056 calibration and 3,043 test questions from MMLU-Pro, BIG-Bench Hard and KOR-Bench. Related questions stay in the same split. Choose on calibration data; check on test data.

**Prerequisites:** Python 3.12, uv and about 4.4 GB for downloads. LoRA needs an NVIDIA GPU (58 minutes on H100); replay and embedding-classifier training work on CPU. The supplied data needs no API keys. Live requests need `OPENAI_API_KEY` and, for Cloud embeddings, `SIE_API_KEY`.

## 1. Install and compare the baselines [Sol: 92.64%; Luna: 88.63% accuracy]

```bash
# Enter the example, install dependencies, download data and compare the routers.
cd examples/train-a-router
uv sync --frozen --group outcomes --group training --group live
uv run --frozen python router.py fetch --output data --group core --group weights
uv run --frozen python router.py replay --data data --output replay.json
uv run --frozen python router.py show --report replay.json
```

## 2. Adapt Laya with LoRA [40.4% answer savings; 1.91 pp accuracy loss]

```bash
# Prepare a run, train LoRA on training labels, and score held-out questions with unchanged and adapted Laya.
uv run --frozen --group outcomes --group training python router.py prepare --data data --run run
uv run --frozen --group outcomes --group training python router.py fit-native --data data --run run
uv run --frozen --group outcomes --group training python router.py score --data data --run run --method native --device cuda
```

## 3. Train on embeddings [50.5% savings; 1.97 pp accuracy loss]

```bash
# Train two correctness predictors on fixed embeddings, then score held-out vectors.
uv run --frozen --group outcomes python router.py fit-embedding --data data --run run
uv run --frozen --group outcomes python router.py score --data data --run run --method embedding
```

## 4. Compare cost–quality frontiers

```bash
# Fit the simple control, build calibration curves, and open the resulting HTML to choose a method and candidate ID.
uv run --frozen --group outcomes python router.py fit-topic --data data --run run
uv run --frozen --group outcomes python router.py calibrate --data data --run run
uv run --frozen python router.py frontier --data data --run run --output calibration-frontier.html
```

Choose a method and candidate ID in `calibration-frontier.html`, then check its test result. The test view uses published answers; use fresh test questions to evaluate any further tuning.

```bash
# Inspect the same saved candidates on published test answers without retraining or making API calls.
uv run --frozen python router.py frontier --data data --run run --include-test --output test-frontier.html
```

## 5. Route a new request

```bash
# With API keys set, use the embedding candidate ID you chose in calibration-frontier.html.
CANDIDATE_ID=YOUR_SELECTED_CANDIDATE_ID
uv run --frozen --group live python router.py answer --run run --method embedding \
  --candidate-id "$CANDIDATE_ID" \
  --question 'Which planet is closest to the Sun? A. Mercury B. Venus' \
  --answer-type choice --choices A B --output answers/planet
```

To use the supplied models, omit `--run` from `frontier` and use `--run data/reference` for `answer`. Candidate IDs belong to a specific method and run. Use fresh output paths. The [README](README.md) covers CPU-only training, local embeddings, labels, serving costs and uncertainty.
