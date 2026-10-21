# Train a router to spend less on AI answers

An inexpensive model can answer many requests well. A router decides when to use it and when to pay for a stronger model. This example maps the **tradeoff between money saved and answer quality**. You choose the operating point that suits your application.

The answer-model pair in this study is **GPT-6.1 Sol → GPT-6 Luna**. Qwen is the embedding encoder, not either answer model:

| Role | Model |
|---|---|
| Stronger answer model | GPT-6.1 Sol, through OpenAI |
| Less expensive answer model | GPT-6 Luna, through OpenAI |
| Decision router, before and after LoRA | Laya, based on ModernBERT-large, about 421 million parameters |
| Fixed embedding encoder | Qwen3-Embedding-4B, through SIE Cloud or your SIE server |
| Classifier on those embeddings | Two small logistic regressions trained here |

Follow these steps:

1. **Understand the data and establish the endpoints.** Measure always-Sol and always-Luna quality and cost.
2. **Try Laya without additional training.** Estimate which answer model will be correct.
3. **Adapt Laya with LoRA.** Learn from both answer models' successes and failures.
4. **Train on embeddings.** Keep Qwen fixed and fit two small correctness predictors.
5. **Compare the cost–quality frontiers.** Find policies for which no other measured choice is both cheaper and at least as accurate, with a strict improvement in one dimension. Choose on calibration data, then check on separate test questions.
6. **Route a new request at your chosen operating point.** Use live APIs, or serve the encoder yourself.

**Two recorded embedding-router operating points** illustrate the tradeoff on 3,043 final questions:

| Policy | Answer accuracy | Loss vs always Sol | Cost / 10,000 | Savings |
|---|---:|---:|---:|---:|
| Always Sol | 92.64% | 0.00 pp | $14.02 | 0.0% |
| Embedding router, more quality retained | 91.65% | 0.99 pp | $9.09 | 35.2% |
| Embedding router, more money saved | 90.67% | 1.97 pp | $6.94 | 50.5% |
| Always Luna | 88.63% | 4.01 pp | $1.52 | 89.2% |

Embedding costs include the Cloud encoder fee; application hosting for the tiny classifier is separate. These are examples along the tradeoff, not a recommended quality ceiling or the complete frontier. The quality-loss intervals for the two embedding points are **0.56–1.46** and **1.38–2.59 percentage points**. Which point is useful depends on the value of correct answers in your application. Compare Laya, LoRA and the simple controls across the curve too; the embedding method does not win everywhere.

For a compact command list, use [the one-page guide](QUICKSTART.md). The complete recorded results, checkpoints and data are in the pinned [artifact release](https://huggingface.co/datasets/superlinked/train-a-router-reasoning/tree/63336f25fc54597a8e161a7245613f91bd60177e).

## 1. Understand the data and baseline [Sol: 92.64% correct; $14.02 / 10,000]

The workload combines public questions from [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard), and [KOR-Bench](https://github.com/KOR-Bench/KOR-Bench). They cover knowledge, multi-step reasoning and rule-based puzzles. We use tasks with exact checkable answers: choice labels, integers or specified text. The included source plan records the selected tasks, upstream revisions, licenses and transformations.

| Split | Questions | Purpose |
|---|---:|---|
| Training | 9,111 | Fit the correctness predictors and estimate answer costs |
| Calibration | 3,056 | Choose how much to favor the less expensive model |
| Development | 900 | Check the collection and serving setup; excluded from fitting |
| Final test | 3,043 | Evaluate all previously selected policies once |

Related questions stay together: exact duplicates, detected close variants, numeric templates and shared KOR rules belong to the same **family**. A family cannot cross training, calibration or final test, or the training folds used to select settings. This reduces leakage from near-duplicate questions. It does not establish whether an answer model saw a public benchmark during pretraining.

We run **both answer models on every question**, then compare each response with the benchmark's reference answer. Each training row therefore contains two correctness labels and two observed token costs. Invalid or truncated responses count as incorrect and still cost money. The router sees the question, never either model's generated answer or the reference answer.

The main metrics are:

- **Answer quality:** percentage of final answers that are correct after routing. This measures the user's answer, not agreement with a preferred route label.
- **Quality loss:** always-Sol accuracy minus routed accuracy, in **percentage points**. Going from 92% to 91% is a loss of 1 point, or one additional incorrect answer per 100 requests on average.
- **Savings:** `100 × (always-Sol cost − routed cost) / always-Sol cost`. We report answer-model cost separately from router cost.
- **Uncertainty:** 95% intervals from repeatedly resampling whole question families within each source. These are pointwise intervals, not simultaneous guarantees for every policy. KOR contributes only nine independent families to the final test.

Costs are projections per 10,000 requests using the recorded model-token prices and actual response lengths. They are not the bill for collecting the experiment. Embedding totals add the Cloud fee for one embedding request per question. Native-router hosting has a separate measured estimate below; an unknown hosting bill is not entered as zero.

### Set up the example

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The **full training sequence requires an NVIDIA GPU** locally or on a virtual machine. We validated LoRA training on an H100 with 80 GB of memory; peak reserved GPU memory was about 29 GB. Smaller GPUs and Apple Silicon training have not been validated for this recipe. A CPU can replay results, train the embedding classifiers and serve Laya more slowly.

The locked dependencies are self-contained in this folder. The complete data and checkpoint download is approximately 4.4 GB. API keys are only needed for fresh model calls, not for training on the supplied recordings or replaying the report.

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
# Recompute the recorded final evaluation, check it against the published report, and print the comparison.
uv run --frozen python router.py replay --data data --output replay.json
uv run --frozen python router.py show --report replay.json
```

`fetch` checks the pinned manifest and every downloaded file. `replay` makes no model calls. It verifies the original route decisions and final-answer collection, then recomputes the results and uncertainty intervals.

## 2. Try the unchanged decision model [one recorded point: 56.7% answer savings; 2.27-point loss]

[Laya](https://huggingface.co/convaiinnovations/laya) is a roughly 421-million-parameter ModernBERT decision model. Here it receives two questions: “Will Luna answer this request correctly?” and “Will Sol answer it correctly?” Its outputs become estimated probabilities of correctness.

Laya produces correctness estimates, not one fixed spending decision. Varying how much to favor lower cost produces different policies. One recorded setting keeps every request on Sol; another saves 56.7% in answer costs with a 2.27-point test-quality loss. The useful comparison is its whole curve against the other approaches. It is not assumed to understand this particular answer-model pair before adaptation.

```bash
# Inspect the unchanged Laya row alongside the other fixed policies; this does not train or call an API.
uv run --frozen python router.py show --report replay.json
```

The comparison uses the same answer models, questions and cost basis for every router. Always-Sol and always-Luna remain available choices, so the policy can bypass routing when doing so is preferable.

## 3. Adapt Laya with LoRA [recorded points: 25.5–40.4% answer savings; 1.12–1.91-point loss]

**LoRA** trains small additional matrices inside the existing model instead of updating all its weights. We adapt Laya's attention layers and decision head to predict the two observed correctness labels. This step needs a GPU.

The training process compares zero through eight training passes on one held-out training fold, selects the earliest best result, then starts fresh and trains on all training questions for that many passes. It selected **one pass** in the recorded run. Calibration and final answers do not influence this choice. The exported checkpoint merges the adapter for ordinary inference.

```bash
# Create a new run and fit the LoRA decision model using training questions and labels only.
uv run --frozen --group outcomes --group training python router.py prepare --data data --run run
uv run --frozen --group outcomes --group training python router.py fit-native --data data --run run
```

```bash
# Score held-out question text with unchanged and adapted Laya; this reads no held-out answers.
uv run --frozen --group outcomes --group training python router.py score --data data --run run --method native --device cuda
```

The recorded GPU fitting run took about 58 minutes, including selection and the fresh final fit. Changing the model, training settings or split defines a new experiment; do not tune those choices against the published final test and describe it as unseen.

## 4. Train a classifier on embeddings [recorded points: 35.2–50.5% savings; 0.99–1.97-point loss]

An **embedding** represents the question as a list of numbers that captures aspects of its meaning. We use the fixed **Qwen3-Embedding-4B** model and a 2,560-number vector. We then fit **two logistic regressions**: a weighted sum followed by a probability conversion, one for Luna correctness and one for Sol correctness. Together they have 5,122 weights and biases, with no hidden layers.

The encoder stays unchanged. Only these small classifiers learn from the labels, so this training step runs on a CPU. Five whole-family training folds choose the regularization strength, which limits overly large learned weights. The recorded run selected an L2 penalty of 1.

```bash
# Fit the small correctness predictors on the supplied embeddings, then score held-out question vectors.
uv run --frozen --group outcomes python router.py fit-embedding --data data --run run
uv run --frozen --group outcomes python router.py score --data data --run run --method embedding
```

To regenerate the vectors, create an account at the [SIE Console](https://console.superlinked.com), obtain an API key, and set `SIE_API_KEY` in your environment. The published full collection used about **$0.21** in batched embedding charges. Fresh charges depend on the current service pricing.

```bash
# Regenerate the fixed question embeddings through SIE Cloud, keeping the downloaded cache unchanged.
uv run --frozen --group live python router.py encode --data data --output new-features
```

Use `--features new-features` on both `fit-embedding` and `score --method embedding` in a fresh run to use those vectors. Completed batches are reused on resume. An interrupted call with unknown billing requires inspecting its receipt before explicitly retrying it with `--retry-unknown`.

The Qwen encoder is larger than Laya, so these results compare complete approaches, not equal-size architectures. A larger encoder may help; this experiment does not isolate model size as the cause. The practical tradeoff includes how the model is hosted and shared.

## 5. Compare frontiers and choose an operating point [you choose the cost–quality tradeoff]

Correctness predictions alone do not decide where to send a request. We also estimate each answer model's expected cost from training data. A **cost penalty** controls how much the router favors saving money over predicted answer quality. Sweeping that penalty produces many policies. The **Pareto frontier** keeps the choices that cannot be improved in cost without giving up quality, or in quality without spending more, among the measured choices.

There is **no required one-point quality-loss ceiling**. Pick a policy based on your budget, the consequences of incorrect answers, and uncertainty. A point that looks best on these benchmarks may not be best on your own traffic.

```bash
# Fit the simple control and build every method's calibration sweep using calibration answers only.
uv run --frozen --group outcomes python router.py fit-topic --data data --run run
uv run --frozen --group outcomes python router.py calibrate --data data --run run
```

```bash
# Build an interactive calibration frontier; open calibration-frontier.html and inspect candidate policies.
uv run --frozen python router.py frontier --data data --run run --output calibration-frontier.html
```

The interactive view lets you compare methods and inspect each operating point. It shows **answer-model cost** for a common comparison, or **priced total cost**, which includes the embedding fee and excludes native/topic policies whose hosting cost is unknown. The dashed random line is the expected result of randomly mixing always-Sol and always-Luna; it is an important inexpensive control. Lines do not imply that every interpolated point is a deterministic routing threshold.

**Choose an operating point from the calibration curve**, recording its method and candidate ID (the number identifying that saved policy). Then inspect test outcomes separately:

```bash
# Inspect the same saved calibration candidates on published test answers, without fitting or making model calls.
uv run --frozen python router.py frontier --data data --run run --include-test --output test-frontier.html
```

The additional test sweep is **retrospective analysis of saved calibration candidates**. It is useful for comparing curves, but the nondominated test points are a description of this test set, not a place to select a policy or claim a new blind result. The candidates exhaust the calibration routing patterns; they need not cover every distinct routing pattern on the test inputs. Choose on calibration data and validate once on a separate holdout for your own application.

The original study recorded 0, 1, 2 and 5-point calibration settings, labeling 1 and 2 as its primary and secondary comparisons. Those were experimental sampling choices, not product requirements. Their 26 frozen policies and original uncertainty intervals remain unchanged. The full calibration sweep was already saved; the example now exposes it instead of privileging one sampled setting.

```bash
# Reproduce the original sampled settings for your refit and inspect their existing uncertainty calculations.
uv run --frozen --group outcomes python router.py evaluate-refit --data data --run run --output refit-report.json
uv run --frozen python router.py show --report refit-report.json
```

Some recorded test points show why the whole comparison matters:

| Router | Answer savings / quality loss at recorded setting 1 | Answer savings / quality loss at recorded setting 2 |
|---|---:|---:|
| Unchanged Laya | 0.0% / 0.00 pp | 56.7% / 2.27 pp |
| Laya + LoRA | 25.5% / 1.12 pp | 40.4% / 1.91 pp |
| Embedding + classifier | 36.5% / 0.99 pp | 51.8% / 1.97 pp |
| Simple text/topic control | 29.6% / 0.66 pp | 57.9% / 1.61 pp |
| Random mixture | 26.2% / 1.18 pp | 52.4% / 2.36 pp |

This table uses **answer costs only** for comparability. Including the Cloud encoder changes the embedding savings to **35.2% and 50.5%**. Native/topic hosting remains unpriced; see the serving scenarios below. The random rows are expected mixtures fixed during calibration, not sampled sequences of random decisions.

LoRA has no clear advantage over random routing in these recorded comparisons. The simple text/topic router is competitive and has a better cost-and-quality point estimate than embeddings at the second sampled setting. We trained one LoRA recipe with one seed; these results do not establish a universally best architecture. The saved intervals are pointwise intervals for individual policies, not simultaneous confidence bands around a frontier selected after seeing test outcomes.

For a CPU-only embedding exercise, use `reuse-controls` in a fresh run instead of `fit-native` and `fit-topic`. This reuses the published comparison predictions and clearly marks them as reused; it does not claim to retrain Laya.

```bash
# Reuse the published Laya and topic comparisons in a CPU-only embedding refit.
uv run --frozen python router.py reuse-controls --data data --run run
```

### Include serving cost and latency

These measurements use the same fixed 90 development questions, with 30 from each source. Native timings include model inference on a warm worker. Cloud timings include the network call, vector handling and the two classifier predictions. They are sequential request timings, not a throughput benchmark.

| Router deployment | Mean latency | 95th percentile | What is included |
|---|---:|---:|---|
| Unchanged Laya, H100 | 30 ms | 68 ms | Local model inference |
| Adapted Laya, H100 | 29 ms | 67 ms | Local model inference |
| Adapted Laya, eight CPU cores | 1.59 s | 4.16 s | Local model inference |
| Qwen embedding through SIE Cloud + classifier | 264 ms | 321 ms | Network, embedding and classifier |

The main final report includes **$0.179 per 10,000 requests** for the Cloud encoder, projected using the actual final question lengths and checked pricing formula. A separate projection reweights the serving sample to the final dataset mix: approximately **$0.27 per 10,000** for busy H100 time or **$1.28** for busy eight-core CPU time for adapted Laya. Those native estimates exclude idle capacity, cold starts and network overhead. A dedicated always-on GPU can cost much more at low traffic; the small classifier's application hosting is also outside the Cloud encoder fee.

Adding those estimated busy-resource costs to the first sampled LoRA answer bill gives roughly **$10.71 per 10,000 on H100** (23.6% savings) or **$11.73 on eight CPU cores** (16.4% savings). These are deployment scenarios, not observed end-to-end bills. The embedding route costs **$9.09 including its Cloud encoder fee**, compared with **$14.02** for always Sol. The chosen policy sends about half the requests to Luna; its savings differ from that fraction because answer lengths and costs vary by question.

## 6. Route a new request [your chosen operating point; one answer-model call]

Set `OPENAI_API_KEY` for the fixed Sol/Luna pair. The embedding router needs `SIE_API_KEY` when it calls SIE Cloud; a self-hosted encoder does not need it. Native Laya loads its local checkpoint. The live command requires an explicit choice: either `--candidate-id` from the selected method's calibration frontier or `--target-gap-pp` for one of the four recorded settings. It has no default quality ceiling. Use the same `--run` that produced the frontier.

```bash
# Route at the calibration operating point you selected; replace the candidate ID with the one shown in your frontier.
CANDIDATE_ID=YOUR_SELECTED_CANDIDATE_ID
uv run --frozen --group live python router.py answer --run run --method embedding \
  --candidate-id "$CANDIDATE_ID" \
  --question 'Which planet is closest to the Sun? A. Mercury B. Venus' \
  --answer-type choice --choices A B --output answers/planet
```

Use `--method lora` or `--method base` with the `training` dependency group and a candidate ID from that method to run either Laya variant. Use `--run data/reference` with a candidate from the published reference frontier to skip refitting. `--device cuda` selects GPU inference; CPU inference is supported. Each call writes the route, answer, usage and timing to its own output directory. This command's elapsed time includes checkpoint loading or the Cloud call; use the resident measurements above for serving comparisons.

### Use your own SIE encoder

Follow [the self-hosted encoder setup](self-host/README.md) to build and start the pinned public SIE embedding server on your GPU. It includes the model configuration, a health check and a one-request smoke test. There is no published image for this bundle; the setup builds from public source. The encoder component has been exercised on a GPU, while this new Compose wrapper has only been checked statically. Pass the endpoint explicitly; there is no need to unset an environment variable or change the example code.

```bash
# Encode the fixed questions against your own SIE server; local compute is not assigned a Cloud bill.
uv run --frozen --group live python router.py encode --data data --output local-features \
  --sie-base-url http://localhost:8000 --self-hosted
```

The same endpoint options work with `answer`. This self-hosts the encoder; **Sol and Luna still use OpenAI**. A Qwen-4B/Qwen-27B development comparison showed a larger quality gap, but an adapted router for that pair has not been validated here. Changing the answer-model pair requires collecting its outcomes and fitting new predictors; the supplied Sol/Luna router is not claimed to transfer.

### Where do labels come from without reference answers?

Here, model outputs are generated by APIs and correctness comes from the benchmark's reference answers. That is different from **teacher-generated labels**. If your application has no reference answers, a teacher model can judge both candidate answers against a written rubric, producing two quality labels while preserving the observed costs. Keep a separate human-reviewed sample to measure whether those judgments are reliable, and prevent the teacher from using final-test answers to guide training.

A stronger teacher is useful only if its judgments are more reliable for your task. Qwen 27B, GLM or a frontier API model are candidates, not interchangeable ground truth. This harder-benchmark study has not validated a teacher-label variant; its measured results use the reference-answer grades described above.

To collect new paired API answers on this fixed workload, follow [the label collection instructions](LABELS.md). They preserve the same roles, request settings, first completed response and physical-attempt cost accounting. Fresh responses can differ from the recordings.

## Files and reproducibility

All runnable code and its dependency lock live in this folder. Large public artifacts live on Hugging Face and are pinned by revision and file hashes.

| File or directory | Purpose |
|---|---|
| `router.py` | Download, prepare, train, score, inspect frontiers, evaluate and serve |
| `frontier_analysis.py`, `frontier.py` | Compare saved operating points and produce an interactive cost–quality view |
| `paired_*.py`, `utility_router.py` | The unchanged training and evaluation recipe used for the recorded study |
| `encode.py`, `live.py` | SIE embeddings and live routed answers |
| `collect_labels.py`, `_collection/` | Optional fresh paired-answer collection and grading |
| `release.json`, `uv.lock` | Exact artifact and dependency versions |
| Downloaded `data/` | Questions, labels, embeddings, checkpoints, reports and serving evidence |
| Your `run/` and output files | Newly fitted models, selected policies and results |

The final report retains all methods and quality targets, including the simple controls and outcomes that are less favorable to a learned router. There is no GEPA dependency and no event-specific setup.
