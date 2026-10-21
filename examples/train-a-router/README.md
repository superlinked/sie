# Train a router to spend less on AI answers

An inexpensive model can answer many requests well. A router decides when to use it and when to pay for a stronger model. The goal here is **lower total cost with a small loss in answer quality**, measured against sending everything to the stronger model.

You will compare two ways to make that decision: adapt a small **Laya decision model** with LoRA, then train a tiny classifier on **Qwen embeddings**. Both learn from the same questions, model answers and costs. This example uses **GPT-6 Luna** as the inexpensive answer model and **GPT-6.1 Sol** as the stronger one. Qwen provides the embeddings; it is not the answer model in these results.

Follow these steps:

1. **Understand the data and establish the baseline.** Measure always-Sol and always-Luna quality and cost.
2. **Try Laya without additional training.** Ask the decision model which answer model is likely to be correct.
3. **Adapt Laya with LoRA.** Teach it from the observed successes and failures of both answer models.
4. **Train on embeddings.** Keep the Qwen encoder fixed and fit two small correctness predictors on top.
5. **Choose a spending policy and evaluate it.** Select the policy on calibration questions; inspect the separate final test and simple controls.
6. **Route a new request.** Use the saved policy with live APIs, or serve the encoder yourself.

**Final result:** **35.2% lower cost including Cloud embedding fees**, with **91.65% answer accuracy versus 92.64%** for always Sol: a **0.99 percentage-point loss** on 3,043 final questions.

| Approach | Final answer quality | Loss vs always Sol | Answer cost / 10,000 | Total cost / 10,000 | Savings |
|---|---:|---:|---:|---:|---:|
| Always Sol | 92.64% | 0.00 pp | $14.02 | $14.02 | 0.0% |
| Always Luna | 88.63% | 4.01 pp | $1.52 | $1.52 | 89.2% |
| Unchanged Laya | 92.64% | 0.00 pp | $14.02 | $14.02 | 0.0% |
| Laya + LoRA | 91.52% | 1.12 pp | $10.44 | hosting unpriced | 25.5% before hosting |
| Embedding + classifier | 91.65% | 0.99 pp | $8.91 | $9.09 | 35.2% |
| Simple text/topic control | 91.98% | 0.66 pp | $9.87 | hosting unpriced | 29.6% before hosting |
| Random control | 91.46% | 1.18 pp | $10.35 | $10.35 | 26.2% |

These are the policies chosen for a **1 percentage-point calibration target**, our primary comparison. The random row is the calculated average for a random mixture fixed during calibration, not one sampled sequence of random decisions. The target is a selection rule, not a promise about new requests. For the embedding router, the 95% interval is **33.0–37.4% savings** and **0.56–1.46 points of quality loss**. The measured loss is just under one point; the interval does not establish a strict one-point ceiling. At this workload and response-length mix, the measured costs project to about **$493 saved per million requests**. Application hosting for the tiny classifier is not included in the encoder fee.

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

## 2. Try the unchanged decision model [0% savings; 0.00-point loss at the primary target]

[Laya](https://huggingface.co/convaiinnovations/laya) is a roughly 421-million-parameter ModernBERT decision model. Here it receives two questions: “Will Luna answer this request correctly?” and “Will Sol answer it correctly?” Its outputs become estimated probabilities of correctness.

At the primary target, calibration selected always Sol for unchanged Laya. Its predictions did not provide a cheaper policy meeting that calibration target. This establishes what the off-the-shelf decision model achieves before adaptation; it is not assumed to understand this particular pair's strengths.

```bash
# Inspect the unchanged Laya row alongside the other fixed policies; this does not train or call an API.
uv run --frozen python router.py show --report replay.json --target-gap-pp 1
```

The comparison uses the same answer models and quality target for every router. Always-Sol and always-Luna remain available choices, so the policy can bypass routing when doing so is preferable.

## 3. Adapt Laya with LoRA [25.5% answer savings; 1.12-point loss]

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

## 4. Train a classifier on embeddings [35.2% savings including encoder; 0.99-point loss]

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

## 5. Select the spending policy and check the final result [35.2% savings at 0.99-point measured loss]

Correctness predictions alone do not decide where to send a request. We also estimate each answer model's expected cost from training data. The policy weighs the predicted quality benefit of Sol against its additional expected cost. Calibration selects the cheapest policy whose measured loss is within the chosen target.

We fix four targets in advance: 0, 1, 2 and 5 percentage points. The **1-point target is the primary comparison**; all targets remain in the report. We also fit a simple word-based topic predictor and compare with random routing, so any gain must be assessed against inexpensive alternatives.

```bash
# Fit the simple text/topic control, then select every policy using calibration answers only.
uv run --frozen --group outcomes python router.py fit-topic --data data --run run
uv run --frozen --group outcomes python router.py calibrate --data data --run run
```

```bash
# Evaluate your refit on the published final answers and print its cost-quality comparison.
uv run --frozen --group outcomes python router.py evaluate-refit --data data --run run --output refit-report.json
uv run --frozen python router.py show --report refit-report.json
```

The original study froze every policy **before collecting final model answers**. Your run uses already published answers and is explicitly labeled a reproduction, not a new blind test. Numerical differences can arise across hardware; the reference replay checks the exact original decisions independently.

The primary embedding policy saves **9.00 percentage points more than the random policy** selected before the test (95% interval: 6.78–11.23), while its accuracy difference is uncertain. A separate diagnostic compares random routing at the embedding router's observed test spending: the embedding router answers **0.60 percentage points more questions correctly** (interval: 0.19–1.00). That spending-matched random fraction is descriptive, not a policy fitted before testing.

**LoRA does not show a clear advantage over random routing**, and the simple text/topic router deserves attention: at the primary target it saves 29.6% in answer costs with a 0.66-point loss. At the secondary 2-point target, its point estimate saves 57.9% before hosting with a 1.61-point loss, versus 50.5% including encoder fees and a 1.97-point loss for the embedding router. These results support comparing the alternatives; they do not establish that each more complex method wins at every quality budget. We trained one LoRA recipe with one seed, not an exhaustive adaptation or full fine-tuning comparison.

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

Adding those estimated busy-resource costs to the primary LoRA answer bill gives roughly **$10.71 per 10,000 on H100** (23.6% savings) or **$11.73 on eight CPU cores** (16.4% savings). These are deployment scenarios, not observed end-to-end bills. The embedding route costs **$9.09 including its Cloud encoder fee**, compared with **$14.02** for always Sol. The chosen policy sends about half the requests to Luna; its savings differ from that fraction because answer lengths and costs vary by question.

## 6. Route a new request [live APIs; one selected answer model]

Set `OPENAI_API_KEY` for the fixed Sol/Luna pair. The embedding router needs `SIE_API_KEY` when it calls SIE Cloud; a self-hosted encoder does not need it. Native Laya loads its local checkpoint. The following command uses the published policy. Substitute `--run run` to use your trained policy.

```bash
# Route a new multiple-choice question with the embedding router and call only the selected answer model.
uv run --frozen --group live python router.py answer --run data/reference --method embedding \
  --question 'Which planet is closest to the Sun? A. Mercury B. Venus' \
  --answer-type choice --choices A B --output answers/planet
```

Use `--method lora` or `--method base` with the `training` dependency group to run either Laya variant. `--device cuda` selects GPU inference; CPU inference is supported. Each call writes the route, answer, usage and timing to its own output directory. This command's elapsed time includes checkpoint loading or the Cloud call; use the resident measurements above for serving comparisons.

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
| `router.py` | Download, prepare, train, score, calibrate, evaluate and serve |
| `paired_*.py`, `utility_router.py` | The unchanged training and evaluation recipe used for the recorded study |
| `encode.py`, `live.py` | SIE embeddings and live routed answers |
| `collect_labels.py`, `_collection/` | Optional fresh paired-answer collection and grading |
| `release.json`, `uv.lock` | Exact artifact and dependency versions |
| Downloaded `data/` | Questions, labels, embeddings, checkpoints, reports and serving evidence |
| Your `run/` and output files | Newly fitted models, selected policies and results |

The final report retains all methods and quality targets, including the simple controls and outcomes that are less favorable to a learned router. There is no GEPA dependency and no event-specific setup.
