# Train a router: one-page guide

**Goal:** compare the cost–quality Pareto frontiers and choose the tradeoff appropriate for your application. The answer models are **GPT-6.1 Sol and GPT-6 Luna** through OpenAI. **Qwen3-Embedding-4B is the encoder**, served by SIE Cloud or your own SIE server. Compare unchanged Laya, LoRA-adapted Laya, embedding classifiers and simple controls.

**Recorded tradeoffs:** always Sol gives 92.64% accuracy at $14.02 per 10,000 requests. Two embedding-router points give 91.65% at $9.09 (**35.2% savings, 0.99-point loss**) and 90.67% at $6.94 (**50.5% savings, 1.97-point loss**), including Cloud encoder fees. Always Luna gives 88.63% at $1.52. These are examples along the curve; **there is no required one-point quality ceiling**. Native/topic hosting is unpriced, and encoder fees exclude the classifier's application hosting.

**Data:** 9,111 training, 3,056 calibration and 3,043 final questions from MMLU-Pro, BIG-Bench Hard and KOR-Bench, with related question families kept together. Measure answer accuracy, its percentage-point loss versus always Sol, and dollars per 10,000. Choose a policy on calibration data; use test data to check it.

**Prerequisites:** Python 3.12, uv and about 4.4 GB for artifacts. LoRA training needs an NVIDIA GPU; the recorded H100 run took about 58 minutes. CPU works for replay, frontier analysis and embedding-classifier training. Recorded-data work needs no keys. Live Cloud embeddings need `SIE_API_KEY`; fresh Sol/Luna answers need `OPENAI_API_KEY`.

## 1. Install and inspect the endpoints [Sol: 92.64%; Luna: 88.63%]

```bash
# Enter the example, install pinned dependencies, download artifacts and inspect all recorded operating points.
cd examples/train-a-router
uv sync --frozen --group outcomes --group training --group live
uv run --frozen python router.py fetch --output data --group core --group weights
uv run --frozen python router.py replay --data data --output replay.json
uv run --frozen python router.py show --report replay.json
```

## 2. Adapt the decision model [recorded points: 25.5–40.4% answer savings]

```bash
# Prepare a run, train LoRA on training labels, and score held-out questions with unchanged and adapted Laya.
uv run --frozen --group outcomes --group training python router.py prepare --data data --run run
uv run --frozen --group outcomes --group training python router.py fit-native --data data --run run
uv run --frozen --group outcomes --group training python router.py score --data data --run run --method native --device cuda
```

## 3. Train the embedding classifiers [recorded points: 35.2–50.5% savings including encoder]

```bash
# Train two correctness predictors on fixed embeddings, then score held-out vectors.
uv run --frozen --group outcomes python router.py fit-embedding --data data --run run
uv run --frozen --group outcomes python router.py score --data data --run run --method embedding
```

## 4. Compare frontiers and choose your operating point [your budget and quality needs]

```bash
# Fit the simple control, build calibration curves, and open the resulting HTML to choose a method and candidate ID.
uv run --frozen --group outcomes python router.py fit-topic --data data --run run
uv run --frozen --group outcomes python router.py calibrate --data data --run run
uv run --frozen python router.py frontier --data data --run run --output calibration-frontier.html
```

Record your choice from calibration, then inspect its test behavior. The expanded test curve is retrospective analysis of saved calibration candidates, not a new blind experiment. The original sampled-policy intervals remain available through `evaluate-refit` and `show`; they are not simultaneous confidence bands around the full curve.

```bash
# Inspect the same saved candidates on published test answers without retraining or making API calls.
uv run --frozen python router.py frontier --data data --run run --include-test --output test-frontier.html
```

## 5. Make a live request at your chosen point [one selected answer-model call]

```bash
# With API keys set, use the embedding candidate ID you chose in calibration-frontier.html.
CANDIDATE_ID=YOUR_SELECTED_CANDIDATE_ID
uv run --frozen --group live python router.py answer --run run --method embedding \
  --candidate-id "$CANDIDATE_ID" \
  --question 'Which planet is closest to the Sun? A. Mercury B. Venus' \
  --answer-type choice --choices A B --output answers/planet
```

To explore the published curves without refitting, omit `--run` from `frontier` and use `--run data/reference` for `answer`. Candidate IDs belong to a specific method and run. Use fresh output paths. For Cloud/local embeddings, CPU-only refits, label collection, serving costs and uncertainty, follow the [README](README.md).
