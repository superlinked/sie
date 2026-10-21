# Train a router to spend less on AI answers

An inexpensive model can answer many requests well. A router decides when to use it and when to pay for a stronger model. Train two kinds of router, compare **money saved against answer quality**, and choose the tradeoff that suits your application.

Requests go to **GPT-6.1 Sol or GPT-6 Luna**. The routers use Laya or Qwen embeddings:

| Role | Model |
|---|---|
| Stronger answer model | GPT-6.1 Sol, through OpenAI |
| Less expensive answer model | GPT-6 Luna, through OpenAI |
| Decision router, before and after LoRA | Laya, based on ModernBERT-large, about 421 million parameters |
| Fixed embedding encoder | Qwen3-Embedding-4B, through SIE Cloud or your SIE server |
| Classifier on those embeddings | Two small logistic regressions trained here |

Follow these steps:

1. **Understand the data and baselines.** Measure always-Sol and always-Luna quality and cost.
2. **Try Laya.** Estimate which answer model will be correct.
3. **Adapt Laya with LoRA.** Learn from both answer models' successes and failures.
4. **Train on embeddings.** Keep Qwen fixed and fit two small correctness predictors.
5. **Compare the cost–quality frontiers.** Choose a routing policy on calibration data, then check it on separate test questions.
6. **Route a new request.** Use live APIs, or serve the encoder yourself.

**Results on 3,043 test questions:**

| Policy | Answer accuracy | Accuracy loss | Cost / 10,000 | Savings |
|---|---:|---:|---:|---:|
| Always Sol | 92.64% | 0.00 pp | $14.02 | 0.0% |
| Embedding router, more quality retained | 91.65% | 0.99 pp | $9.09 | 35.2% |
| Embedding router, more money saved | 90.67% | 1.97 pp | $6.94 | 50.5% |
| Always Luna | 88.63% | 4.01 pp | $1.52 | 89.2% |

Always Sol is the baseline for accuracy loss and savings. **pp means percentage points:** 92% accuracy falling to 91% is a 1 pp loss. Savings use %, so spending $65 instead of $100 saves 35%.

Embedding costs include the Cloud encoder fee; classifier application hosting is separate. These policies illustrate the tradeoff. Step 5 compares the full curves, including Laya and simpler alternatives.

Use [the one-page guide](QUICKSTART.md) for a compact command list. Downloadable [data, checkpoints and results](https://huggingface.co/datasets/superlinked/train-a-router-reasoning/tree/63336f25fc54597a8e161a7245613f91bd60177e) support every step below.

## 1. Understand the data and baselines [Sol: 92.64% accuracy; $14.02 / 10,000]

The workload combines [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard), and [KOR-Bench](https://github.com/KOR-Bench/KOR-Bench): knowledge, multi-step reasoning and rule-based puzzles with checkable answers. The download includes the source versions, licenses and question transformations.

| Split | Questions | Purpose |
|---|---:|---|
| Training | 9,111 | Fit the correctness predictors and estimate answer costs |
| Calibration | 3,056 | Choose how much to favor the less expensive model |
| Development | 900 | Check the collection and serving setup; excluded from fitting |
| Final test | 3,043 | Evaluate all previously selected policies once |

Duplicates, detected close variants and questions sharing templates or KOR rules form a **family**. Families stay within one data split and one training fold to reduce leakage. Public benchmarks may still have appeared in an answer model's pretraining data.

We run **both answer models on every question** and grade their responses against the reference answer. Each training row has two correctness labels and two token costs. Invalid or truncated responses count as incorrect and still cost money. The router sees only the question.

The main metrics are:

- **Answer accuracy (%):** how often the answer returned to the user is correct.
- **Accuracy loss (pp):** always-Sol accuracy minus routed accuracy.
- **Cost savings (%):** `100 × (always-Sol cost − routed cost) / always-Sol cost`.
- **Uncertainty:** 95% intervals, calculated by repeatedly resampling whole question families within each benchmark.

Costs project serving 10,000 requests using the experiment's token prices and response lengths. Router fees are separate from answer-model costs. See step 5 for hosting costs and latency.

### Set up the example

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). **LoRA training requires an NVIDIA GPU**, locally or on a VM. We used an H100 with 80 GB of memory; peak reserved GPU memory was about 29 GB. Smaller GPUs and Apple Silicon training have not been validated for this recipe. A CPU can replay results, train the embedding classifiers and serve Laya more slowly.

Dependencies are pinned in this folder. The data and checkpoints take approximately 4.4 GB. Training and replay use the supplied answers; API keys are needed only for new model calls.

```bash
# From a clone of superlinked/sie, install the example's pinned training and API dependencies.
cd examples/train-a-router
uv sync --frozen --group outcomes --group training --group live
```

```bash
# Download and verify the published questions, outcomes, embeddings and Laya checkpoints.
uv run --frozen python router.py fetch --output data --group core --group weights
```

```bash
# Recompute the results from saved answers and compare the routers.
uv run --frozen python router.py replay --data data --output replay.json
uv run --frozen python router.py show --report replay.json
```

`fetch` verifies the downloads. `replay` checks the saved routing decisions and recomputes metrics and uncertainty intervals without model calls.

## 2. Try Laya [56.7% answer savings; 2.27 pp accuracy loss]

[Laya](https://huggingface.co/convaiinnovations/laya) is a roughly 421-million-parameter ModernBERT decision model. Here it receives two questions: “Will Luna answer this request correctly?” and “Will Sol answer it correctly?” Its outputs become estimated probabilities of correctness.

Start with Laya's existing weights. It estimates correctness; a cost penalty determines how much the router favors cheaper answers. Varying that penalty traces a cost–quality curve. The heading shows one policy from that curve.

```bash
# Inspect Laya's results alongside the other routers.
uv run --frozen python router.py show --report replay.json
```

Every router uses the same answer models and questions. Always-Sol and always-Luna remain available choices.

## 3. Adapt Laya with LoRA [40.4% answer savings; 1.91 pp accuracy loss]

**LoRA** trains small additional matrices inside the existing model instead of updating all its weights. We adapt Laya's attention layers and decision head to predict the two observed correctness labels. This step needs a GPU.

Training compares zero through eight passes using a held-out portion of the training data, then starts fresh and fits all training questions for the best number of passes. It selected **one pass**. Calibration and test answers play no part in that selection. The exported model includes the LoRA weights, ready for inference.

```bash
# Create a new run and fit the LoRA decision model using training questions and labels only.
uv run --frozen --group outcomes --group training python router.py prepare --data data --run run
uv run --frozen --group outcomes --group training python router.py fit-native --data data --run run
```

```bash
# Score held-out question text with unchanged and adapted Laya; this reads no held-out answers.
uv run --frozen --group outcomes --group training python router.py score --data data --run run --method native --device cuda
```

Training took about **58 minutes on an H100**, including selection and the final fit.

## 4. Train on embeddings [50.5% savings; 1.97 pp accuracy loss]

An **embedding** represents the question as a list of numbers that captures aspects of its meaning. We use the fixed **Qwen3-Embedding-4B** model and a 2,560-number vector. We then fit **two logistic regressions**: a weighted sum followed by a probability conversion, one for Luna correctness and one for Sol correctness. Together they have 5,122 weights and biases, with no hidden layers.

Only these small classifiers learn from the labels, so training runs on a **CPU**. Five training folds, keeping families together, select a penalty that limits overly large weights. The selected L2 penalty was 1.

```bash
# Fit the small correctness predictors on the supplied embeddings, then score held-out question vectors.
uv run --frozen --group outcomes python router.py fit-embedding --data data --run run
uv run --frozen --group outcomes python router.py score --data data --run run --method embedding
```

To regenerate the vectors, get an API key from the [SIE Console](https://console.superlinked.com) and set `SIE_API_KEY`. Embedding the full dataset in batches cost about **$0.21** at the experiment's prices.

```bash
# Regenerate the fixed question embeddings through SIE Cloud, keeping the downloaded cache unchanged.
uv run --frozen --group live python router.py encode --data data --output new-features
```

Use `--features new-features` on both `fit-embedding` and `score --method embedding` in a fresh run to use those vectors.

Qwen has 4B parameters; Laya has 421M. Model size may contribute to their results, but this comparison also changes the architecture and training. Compare serving cost and latency alongside accuracy.

## 5. Compare cost–quality frontiers

The **Pareto frontier** contains the policies for which no measured alternative is cheaper with equal or better accuracy, or more accurate at equal or lower cost. Choose a policy based on your budget and the consequences of incorrect answers.

Each router combines predicted correctness with answer costs estimated from training data. Sweep the cost penalty to explore its tradeoffs. Include a simple text/topic classifier and random routing to check whether the more complex approaches earn their cost.

```bash
# Fit the text/topic control on training data, then build calibration sweeps.
uv run --frozen --group outcomes python router.py fit-topic --data data --run run
uv run --frozen --group outcomes python router.py calibrate --data data --run run
```

```bash
# Build an interactive calibration frontier; open calibration-frontier.html and inspect candidate policies.
uv run --frozen python router.py frontier --data data --run run --output calibration-frontier.html
```

Open the HTML to compare routers. Choose **answer-model cost** to compare them before hosting fees, or **priced total cost** to include the Cloud encoder fee. Policies with unpriced hosting are omitted from total-cost view. The dashed line shows the expected cost and accuracy of randomly choosing Sol or Luna.

**Choose on calibration data; check on test data.** Save the method and candidate ID shown in the chart, then inspect its test result:

```bash
# Inspect the same saved calibration candidates on published test answers, without fitting or making model calls.
uv run --frozen python router.py frontier --data data --run run --include-test --output test-frontier.html
```

The test view replays the same candidate policies on published answers. Selecting a policy after seeing those answers would bias the result. If you train or tune further, evaluate the changes on fresh test questions. The displayed candidates cover all calibration routing patterns, which may differ from the patterns possible on test questions.

The report also summarizes policies selected with 0, 1, 2 and 5 pp calibration limits and includes their uncertainty intervals. You can choose any candidate in the frontier view.

```bash
# Evaluate your trained routers at the report's four calibration limits, including uncertainty intervals.
uv run --frozen --group outcomes python router.py evaluate-refit --data data --run run --output refit-report.json
uv run --frozen python router.py show --report refit-report.json
```

Here is how two calibration limits translate into test results. Both columns show **answer-cost savings / accuracy loss**:

| Router | 1 pp calibration limit | 2 pp calibration limit |
|---|---:|---:|
| Unchanged Laya | 0.0% / 0.00 pp | 56.7% / 2.27 pp |
| Laya + LoRA | 25.5% / 1.12 pp | 40.4% / 1.91 pp |
| Embedding + classifier | 36.5% / 0.99 pp | 51.8% / 1.97 pp |
| Simple text/topic control | 29.6% / 0.66 pp | 57.9% / 1.61 pp |
| Random mixture | 26.2% / 1.18 pp | 52.4% / 2.36 pp |

Including the Cloud encoder changes embedding savings to **35.2% and 50.5%**. Laya and text/topic hosting must be added separately. Random-routing results are expectations for mixtures chosen on calibration data.

LoRA has no clear advantage over random routing at these settings. The text/topic classifier is competitive: at the 2 pp calibration limit, it has lower answer cost and higher test accuracy than the embedding classifier. The best choice depends on where you operate on the curve.

<details>
<summary>Uncertainty and evaluation details</summary>

- The two embedding accuracy-loss estimates have 95% intervals of **0.56–1.46 pp** and **1.38–2.59 pp**. Intervals describe individual policies, not the entire frontier at once.
- Resampling keeps question families together. KOR contributes only nine independent families to the final test.
- The report's 26 policies were fixed before collecting final answers. The expanded test curve analyzes those published answers retrospectively.
- LoRA used one training recipe and one seed. These results do not isolate the effect of architecture or establish a universal winner.

</details>

For a CPU-only embedding exercise, prepare a fresh run and use `reuse-controls` instead of `fit-native` and `fit-topic`. This loads the supplied Laya and text/topic predictions.

```bash
# Reuse the published Laya and topic comparisons in a CPU-only embedding refit.
uv run --frozen python router.py reuse-controls --data data --run run
```

### Include serving cost and latency

Timings use the same 90 development questions, 30 per benchmark, sent one at a time. Laya runs on a warm worker; Cloud timings also include the network call and classifiers.

| Router deployment | Mean latency | 95th percentile | What is included |
|---|---:|---:|---|
| Unchanged Laya, H100 | 30 ms | 68 ms | Local model inference |
| Adapted Laya, H100 | 29 ms | 67 ms | Local model inference |
| Adapted Laya, eight CPU cores | 1.59 s | 4.16 s | Local model inference |
| Qwen embedding through SIE Cloud + classifier | 264 ms | 321 ms | Network, embedding and classifier |

The Cloud encoder adds **$0.179 per 10,000 requests** at the test questions' lengths. Adapting the serving measurements to the same dataset mix gives approximately **$0.27 per 10,000** for LoRA-adapted Laya on a busy H100 or **$1.28** on eight busy CPU cores. These Laya estimates exclude idle time, cold starts and networking. An always-on GPU can cost much more at low traffic. The embedding classifier's application hosting is separate too.

For the LoRA policy with 1.12 pp accuracy loss, adding those hosting estimates gives **$10.71 per 10,000 on H100** (23.6% savings) or **$11.73 on CPU** (16.4% savings). These are cost projections. The embedding policy with 0.99 pp loss costs **$9.09 including its encoder**, versus **$14.02** for always Sol.

## 6. Route a new request

Set `OPENAI_API_KEY` for Sol/Luna and `SIE_API_KEY` for Cloud embeddings. Pass the method, candidate ID and run you chose in step 5. Each request calls one answer model.

```bash
# Route a question using the candidate ID you chose in the calibration chart.
CANDIDATE_ID=YOUR_SELECTED_CANDIDATE_ID
uv run --frozen --group live python router.py answer --run run --method embedding \
  --candidate-id "$CANDIDATE_ID" \
  --question 'Which planet is closest to the Sun? A. Mercury B. Venus' \
  --answer-type choice --choices A B --output answers/planet
```

For Laya, use `--method lora` or `--method base`, the `training` dependency group and a candidate ID from that method. Add `--device cuda` for GPU inference; CPU is also supported. Laya loads its local checkpoint and needs no SIE key.

To use the supplied models, omit `--run` when generating the frontier, then use `--run data/reference` for live requests. Each request needs a fresh output directory and saves its route, answer, usage and timing. CLI timing includes model loading; the serving table above measures a warm worker.

### Use your own SIE encoder

Follow [the self-hosted encoder setup](self-host/README.md) to build SIE from public source and start Qwen on your GPU. The encoder has passed a GPU check; the Compose wrapper has only static validation. Pass its endpoint explicitly:

```bash
# Generate question embeddings with your local SIE server.
uv run --frozen --group live python router.py encode --data data --output local-features \
  --sie-base-url http://localhost:8000 --self-hosted
```

The same endpoint options work with `answer`, without a SIE key. **Sol and Luna still use OpenAI.** To use a different answer-model pair, collect its outcomes and retrain the predictors.

### Where do labels come from without reference answers?

This example grades against reference answers. Without references, a **teacher model** can judge both answers against a written rubric and supply the two training labels. Keep the actual answer costs, and use a separate human-reviewed sample to check the teacher's reliability.

Qwen 27B, GLM or a frontier API model could serve as the teacher. Compare their judgments with that human-reviewed sample before choosing. Teacher-generated labels have not been validated in this example.

To collect new Sol/Luna answers for this dataset, follow [the label collection instructions](LABELS.md).

## Files and reproducibility

All runnable code and its dependency lock live in this folder. Large public artifacts live on Hugging Face and are pinned by revision and file hashes.

| File or directory | Purpose |
|---|---|
| `router.py` | Download, prepare, train, score, inspect frontiers, evaluate and serve |
| `frontier_analysis.py`, `frontier.py` | Compare saved operating points and produce an interactive cost–quality view |
| `paired_*.py`, `utility_router.py` | Training, routing and evaluation |
| `encode.py`, `live.py` | SIE embeddings and live routed answers |
| `collect_labels.py`, `_collection/` | Optional fresh paired-answer collection and grading |
| `release.json`, `uv.lock` | Exact artifact and dependency versions |
| Downloaded `data/` | Questions, labels, embeddings, checkpoints, reports and serving evidence |
| Your `run/` and output files | Newly fitted models, selected policies and results |
