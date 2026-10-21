# Train a router: one-page guide

**Goal:** spend less than always using GPT-6.1 Sol while keeping answer quality close. Compare unchanged Laya, LoRA-adapted Laya and a classifier trained on Qwen3-Embedding-4B vectors. **Recorded result:** **35.2% lower cost including Cloud embedding fees**, with **91.65% answer accuracy versus 92.64%** for always Sol: a **0.99 percentage-point loss** on 3,043 final questions.

**Data and metrics:** 9,111 training, 3,056 calibration and 3,043 final questions from MMLU-Pro, BIG-Bench Hard and KOR-Bench, with related question families kept together. Measure final answer accuracy, its percentage-point loss versus always Sol, and dollars per 10,000 requests including the encoder fee. The primary policy targets a 1-point calibration loss; its final-test 95% interval is **0.56–1.46 points**, so it does not guarantee a loss below one point. LoRA/topic hosting is unpriced; the embedding fee excludes the small classifier's application hosting.

**Prerequisites:** Python 3.12, uv and roughly 4.4 GB for artifacts. Use an NVIDIA GPU locally or in a VM for LoRA training; the recorded run used an H100 and took about 58 minutes. CPU works for report replay and embedding-classifier training. Recorded-data work needs no API keys. SIE Cloud embedding calls need `SIE_API_KEY`; fresh Sol/Luna answers need `OPENAI_API_KEY`.

## 1. Install and inspect the baseline [Sol: 92.64% correct; $14.02 / 10,000]

```bash
# Enter the example, install pinned dependencies, download artifacts and print the recorded results.
cd examples/train-a-router
uv sync --frozen --group outcomes --group training --group live
uv run --frozen python router.py fetch --output data --group core --group weights
uv run --frozen python router.py replay --data data --output replay.json
uv run --frozen python router.py show --report replay.json
```

## 2. Adapt the decision model [25.5% answer savings; 1.12-point loss]

```bash
# Prepare a new run, train LoRA on training labels, and score held-out questions with both Laya variants.
uv run --frozen --group outcomes --group training python router.py prepare --data data --run run
uv run --frozen --group outcomes --group training python router.py fit-native --data data --run run
uv run --frozen --group outcomes --group training python router.py score --data data --run run --method native --device cuda
```

## 3. Train the embedding classifiers [35.2% savings including encoder; 0.99-point loss]

```bash
# Train two small correctness predictors on fixed embeddings, then score held-out vectors.
uv run --frozen --group outcomes python router.py fit-embedding --data data --run run
uv run --frozen --group outcomes python router.py score --data data --run run --method embedding
```

## 4. Select and evaluate the policy [35.2% savings at 0.99-point measured loss]

```bash
# Fit the simple comparison, choose policies on calibration answers, and evaluate the refit on published test answers.
uv run --frozen --group outcomes python router.py fit-topic --data data --run run
uv run --frozen --group outcomes python router.py calibrate --data data --run run
uv run --frozen --group outcomes python router.py evaluate-refit --data data --run run --output refit-report.json
uv run --frozen python router.py show --report refit-report.json
```

## 5. Make a live routed request [one selected answer-model call]

```bash
# With SIE_API_KEY and OPENAI_API_KEY set, route a new question using the published embedding policy.
uv run --frozen --group live python router.py answer --run data/reference --method embedding \
  --question 'Which planet is closest to the Sun? A. Mercury B. Venus' \
  --answer-type choice --choices A B --output answers/planet
```

Use a fresh output directory for each run. For regenerated Cloud embeddings, self-hosted SIE, CPU-only refits, label collection, uncertainty and serving tradeoffs, follow the [README](README.md). Your refit uses published test answers; it is explicitly a reproduction, not a new blind experiment.
