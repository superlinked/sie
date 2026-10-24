# Train a router to spend less on reasoning

A router decides when a cheaper model is enough and when a stronger model is worth its extra cost. This example trains that decision on difficult reasoning questions, then measures **money saved and answer accuracy** on questions kept out of training.

**On 975 held-out questions, an embedding classifier with learned cost estimates saved 16.7% with a 0.9-point accuracy loss; another preselected policy saved 39.7% with a 6.1-point loss. Both savings figures include the Cloud encoder fee.** The complete cost–quality curves are included: there is no single required accuracy-loss threshold.

You will:

1. **Establish the baseline:** compare always using Sol with always using Luna.
2. **Try a decision model:** use Laya to estimate whether each answer model will succeed.
3. **Adapt Laya with LoRA:** teach it from the answer models’ successes and failures.
4. **Use embedding neighbors:** estimate success from similar questions, without adapting an encoder.
5. **Train a small embedding classifier:** learn two correctness predictions on top of fixed embeddings.
6. **Compare cost estimates:** predict each request’s answer cost, then choose a policy on the full curve.
7. **Route a new question:** call only the selected answer model.

[One-page instructions](QUICKSTART.md) · [Labels and new answer collection](LABELS.md) · [Serve the encoder yourself](self-host/README.md)

## 1. Understand the data and baseline [Sol 80.8% accuracy; Luna 55.6%]

We use **Big-Bench Extra Hard (BBEH)**: reasoning problems involving logic, counting, state tracking, language rules and other tasks. We retain **3,918 questions across 23 tasks** after source-only duplicate and ambiguity checks. Each question is answered independently by **`gpt-6-luna`** and **`gpt-6.1-sol`**, using medium reasoning and the same answer format. The supplied answer recordings let you train and evaluate without paying to collect them again.

| Split | Questions | Purpose |
|---|---:|---|
| Training | 1,969 | Learn from model successes, failures and token costs |
| Development | 589 | Choose training duration and model settings |
| Calibration | 385 | Choose routing policies at different spending levels |
| Final test | 975 | Measure the frozen policies once |

Repeated questions and identified source/context families stay together. Training uses training labels; model settings use development labels. After those choices, each learned router is fitted afresh on training + development. Calibration chooses the routing tradeoffs. In the original study, final-test answers were collected only after all models and policies were frozen.

The main measurements are:

- **Answer accuracy:** the percentage of chosen model answers judged correct against the benchmark reference. This measures the answers people receive, rather than whether a router matches an arbitrary “best model” label.
- **Cost per 10,000 requests:** the selected answers' input/output token cost plus the stated router cost, scaled to a common workload size. API charges and hosting estimates are identified separately.
- **Savings:** the percentage decrease from always calling Sol.
- **Accuracy loss:** always-Sol accuracy minus routed accuracy. For example, 80% becoming 78% is a **2-point loss**; a negative loss means the router did better on these questions.

Some questions are answered correctly only by Luna. A good router can therefore occasionally beat always-Sol. The gray random-routing line shows what happens if we choose models without reading the question; a useful router should improve on that tradeoff.

All dollar amounts below are **per 10,000 incoming requests**. Each request gets one answer, from either Sol or Luna. **Subtotals include the embedding API fee or the estimated Laya serving cost.** `≈` marks a hosting estimate; `—` means that model is not used. Application hosting, including the small CPU classifiers and cost predictors, idle capacity, and one-time training/data collection are extra.

| Method | Answer accuracy | Loss (points) | Sol answers | Luna answers | Laya, H100 estimate | Embeddings, Cloud API | Subtotal | Savings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always Sol | 80.8% | 0.0 | $70.77 | — | — | — | $70.77 | 0.0% |
| Always Luna | 55.6% | 25.2 | — | $12.23 | — | — | $12.23 | 82.7% |
| Unchanged Laya | 68.6% | 12.2 | $36.06 | $5.76 | ≈$1.08 | — | ≈$42.90 | ≈39.4% |
| Laya + LoRA | 78.3% | 2.6 | $54.16 | $2.64 | ≈$1.05 | — | ≈$57.85 | ≈18.2% |
| Embedding neighbors | 76.3% | 4.5 | $50.59 | $3.21 | — | $0.81 | $54.61 | 22.8% |
| Embedding classifier | 76.8% | 4.0 | $51.51 | $2.90 | — | $0.81 | $55.22 | 22.0% |
| LoRA + learned cost | 79.0% | 1.8 | $55.34 | $2.29 | ≈$1.05 | — | ≈$58.68 | ≈17.1% |
| Embedding classifier + learned cost | 79.9% | 0.9 | $56.06 | $2.10 | — | $0.81 | $58.97 | 16.7% |
| Text/length + learned cost | 79.0% | 1.8 | $52.84 | $2.44 | — | — | $55.28 | 21.9% |
| Text-predicted task + learned cost | 80.1% | 0.7 | $56.15 | $2.15 | — | — | $58.31 | 17.6% |

Laya's estimate prices the time actively serving requests on a resident H100; the embedding column uses SIE Cloud's per-request price. These are different deployment choices, not a comparison of both models on identical hardware. [Serving speed and costs](#serving-speed-and-costs) shows the calculation and its limits. Savings use the unrounded subtotal; rounded components can differ from it by a cent.

**Neighbors spend less here because they buy different answers.** They send **49.6%** of requests to Sol, compared with **59.7%** for LoRA. Their answer bill is **$53.80 versus $56.80**, before either router's inference cost, and their accuracy is lower: **76.3% versus 78.3%**. Laya is about 0.4B parameters and Qwen's encoder is 4B, but parameter count alone does not determine the full routing bill.

All router rows use a rule fixed before final collection: select the highest calibration accuracy under an **80% calibration answer-cost cap** relative to always-Sol. It is an illustrative spending level, not a required quality threshold or a guarantee of 20% test savings. The router costs above are added after that selection; they do not change the frozen policies. Every method, both cost variants and all preselected spending levels remain in the full report.

The table and default replay use the supplied reference checkpoints. Step 3 also compares a faster training recipe against them. The steps compare different ways to learn the decision; they do not promise that every added model improves every tradeoff. LoRA improves on unchanged Laya. Neighbors provide a simple embedding baseline. Learning request costs improves the embedding classifier at the illustrated spending levels. The inexpensive text controls remain competitive and belong in the comparison.

### Set up

For the **complete training sequence**, use a Linux machine with an NVIDIA GPU. The fast LoRA recipe was validated on one **H100 80 GB**; PyTorch reserved up to **75.8 GiB of GPU memory** during the measured validation stages. Other GPU sizes and Apple Silicon LoRA training have not been validated. A Linux or macOS laptop can replay the results, train the embedding classifiers and run Laya inference; full-population Laya scoring is much faster on a GPU.

Install Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). From a clone of this repository:

```bash
# Enter the self-contained example and install the locked CPU dependencies.
cd examples/train-a-router
uv sync --frozen --group outcomes
```

```bash
# Download the verified dataset, saved models and Laya weights from Hugging Face.
uv run --frozen python router.py fetch --output data --group core --group weights
```

```bash
# Recompute the published test results on CPU, without model or API calls.
uv run --frozen python router.py replay --data data --output published-results
```

Open `published-results/explorer/index.html` in your browser. It shows all saved policies, uncertainty intervals, simple controls and costs with or without the measured Cloud encoder fee. For a smaller CPU-only download, omit `--group weights`; weights are needed for Laya training or inference, not result replay.

```bash
# Create fresh working directories for your own training run; keep downloaded evidence intact.
uv run --frozen python router.py prepare --data data --run my-router
```

For an **embedding-only exercise on CPU**, install with `uv sync --frozen --group outcomes` and fetch only `--group core`. After `prepare`, run the following command and continue at step 4:

```bash
# Reuse the supplied Laya predictions for comparison while you train the embedding routers on CPU.
uv run --frozen python router.py reuse-native --data data --run my-router
```

This path uses the recorded Laya and LoRA comparisons; it does not train or copy a Laya checkpoint into `my-router`. The complete GPU path below performs the adaptation itself.

## 2. Try unchanged Laya [68.6% accuracy; 40.9% answer-cost savings]

**Laya is a decision model:** a roughly 0.4-billion-parameter ModernBERT encoder with a small decision head. For each request, we ask two questions: will Luna answer correctly, and will Sol answer correctly? Those scores, together with estimated answer costs, determine whether paying for Sol looks worthwhile. The request is the only input; the router never sees the reference answer.

```bash
# Install the locked CUDA environment and check a real GPU forward/backward step.
uv sync --project gpu --frozen
uv run --project gpu --frozen python gpu/smoke.py --output gpu-smoke.json
```

The [GPU setup guide](gpu/README.md) lists the supported system and public binary packages. `--project gpu` selects the GPU environment; keep working in `examples/train-a-router`.

```bash
# Score the held-out question inputs with unchanged Laya on the GPU.
uv run --project gpu --frozen python router.py base --data data --run my-router --device cuda
```

This establishes how much a general decision model knows before seeing this workload. The supplied calibration explorer already includes this baseline, so you can inspect its tradeoffs while your own scoring runs. For CPU inference on Linux or macOS, use the main environment with `--group training` in place of `--project gpu`, and `--device cpu`.

## 3. Adapt Laya with LoRA [78.3% accuracy; 19.7% answer-cost savings]

**LoRA learns small updates inside an existing model.** We keep Laya’s original encoder weights fixed, learn rank-16 attention updates, and train its existing decision head. The labels are two bits per question: whether Luna’s answer was correct and whether Sol’s was correct. They come from real model answers checked against the benchmark references.

The fast recipe reduces padding and uses FlashAttention during training. It keeps the same model, labels, effective batch size and development-selection rule. Validation and exported-model inference use the original full-precision computation.

The complete fast fit took **28.6 minutes on H100**, compared with **77.4 minutes** for the reference recipe: **2.7× faster**. This includes all eight training passes and development checks (**18.2 minutes**), then the selected three-pass refit and checkpoint exports (**10.4 minutes**). Dependency setup and held-out input scoring add time. The reference recipe remains available with `--recipe reference`.

```bash
# Train on training data, select duration on development, then refit from the original model on both.
uv run --project gpu --frozen python router.py fit-native --data data --run my-router --recipe fast
```

```bash
# Apply both unchanged and adapted Laya to calibration/test inputs, without reading their answer labels.
uv run --project gpu --frozen python router.py score --data data --run my-router --method native --device cuda
```

The recipe considers the unchanged model and eight training passes. **Three passes gave the best development loss** in the supplied run; later passes worsened it. We then restarted from the original model and trained for those three passes on training + development. This check is why the example does not simply train for as long as possible. Small differences across hardware can change the selected pass; your run records its own selection.

At the same 80% calibration spending cap, the faster run reached **77.8% accuracy at $55.94 / 10k answers**, compared with **78.3% at $56.80** for the supplied reference. Adding the learned costs in step 6 gave the faster model **78.9% accuracy at $55.84**: **21.1% answer-cost savings with a 1.9-point loss** versus always-Sol, before native hosting. [The complete training comparison](gpu/README.md#measured-speed-and-routing-quality) includes all spending levels, uncertainty and a download that needs no GPU.

## 4. Estimate success from similar questions [76.3% accuracy; 22.8% savings incl. encoder]

An **embedding** represents a question as a list of numbers that captures similarities in its content. We use **Qwen3-Embedding-4B**, fixed throughout the experiment. The neighbor router finds similar training questions and averages their two correctness labels. It selected **32 neighbors** on development data.

This gives a different way to improve with more examples: add their embeddings and observed outcomes to the reference collection. You do not retrain the large encoder.

```bash
# Fit neighbors and the small classifiers introduced next, using the supplied embeddings on CPU.
uv run --frozen --group outcomes python router.py fit-embedding --data data --run my-router
```

The command fits both embedding approaches and inexpensive text controls in one pass. Compare the neighbor method (`knn`) with the classifier (`embedding`) in the calibration explorer in step 6.

## 5. Train a small classifier on embeddings [76.8% accuracy; 22.0% savings incl. encoder]

Instead of looking up neighbors, we can learn a compact mapping from each embedding to two probabilities. The classifier is **two regularized logistic regressions**—one for Luna’s correctness and one for Sol’s—not another large language model. The frozen encoder produces 2,560 numbers; each predictor learns a weight for those numbers plus an offset.

The previous command trained these classifiers alongside the neighbors. Now generate predictions from both approaches:

```bash
# Generate their calibration/test predictions from question embeddings, without reading answer labels.
uv run --frozen --group outcomes python router.py score --data data --run my-router --method embedding
```

The text and task controls fitted in step 4 matter: if question length or vocabulary explains most of the benefit, a larger model may be unnecessary. The control given the benchmark's true task name is labeled **metadata-assisted** and cannot be selected for arbitrary live requests.

The encoder is about ten times larger than Laya. This comparison measures useful systems, not proof that one architecture is intrinsically better. Encoder quality, size, training data and task fit all differ. Its practical advantage here is that the expensive encoder stays fixed while the task-specific training is fast and small.

### Use SIE Cloud or your own encoder

The supplied vectors make the training exercise reproducible on CPU. To generate fresh ones, create an account in the [Superlinked Console](https://console.superlinked.com), obtain an API key, and use the available Cloud credits. **SIE credits pay for embeddings; Sol/Luna answers use OpenAI credits.** You need no OpenAI key to train using the supplied answer recordings.

```bash
# Enter your SIE Cloud key without writing it into this repository.
export SIE_API_KEY='your-sie-api-key'
```

```bash
# Regenerate all question embeddings and validate singleton encoder fees through SIE Cloud.
uv run --frozen --group encode python router.py encode --data data --output fresh-encoding
```

For a fresh run, add `--features fresh-encoding/features` to both embedding fitting and scoring commands. The [self-host instructions](self-host/README.md) run the same pinned Qwen encoder behind your own SIE endpoint. Self-hosting the encoder does not make Sol or Luna locally available; changing the answer models requires collecting their outcomes and training for that pair.

## 6. Compare cost estimates and choose the tradeoff [79.9% accuracy; 16.7% savings incl. encoder]

A request’s cost varies with how much a model generates. A router that predicts correctness well can still spend poorly if it assumes the wrong answer cost. We therefore also fit two small cost predictors using question words and length. They learn from the token costs in training + development; the saved final answers are not available to them.

```bash
# Fit request-cost predictors on CPU while preserving each router’s original cost-estimate variant.
uv run --frozen --group outcomes python router.py fit-cost --data data --run my-router
```

```bash
# Use calibration answers to save the complete cost–quality curve and freeze every test policy.
uv run --frozen python router.py calibrate --data data --run my-router
```

```bash
# Explore your calibration curve and copy a method plus candidate ID before opening test results.
uv run --frozen python router.py frontier --run my-router --output calibration.html
```

Open `calibration.html`. A **Pareto frontier** keeps choices where you cannot lower cost without losing accuracy, or improve accuracy without paying more. Choose a point that fits your application and record its **method and candidate ID together before opening the test report**. Candidate IDs belong to a particular method and run; the same number can mean a different policy elsewhere. There is no built-in one-point accuracy-loss requirement. Increasing the price penalty generally sends more requests to Luna; the saved policy also handles cases where a supposedly stronger answer is predicted to cost less.

The cost predictor helps some spending levels and hurts others, so the explorer keeps both variants. `embedding` uses the original cost estimate; `embedding__textcost` uses the learned request-cost estimate. The same choice is available for Laya, LoRA, neighbors and simple controls.

For the embedding classifier with learned costs, the preselected policies give:

| Calibration answer-cost cap | Final accuracy | Accuracy loss | Answers + encoder / 10k | Savings incl. encoder |
|---|---:|---:|---:|---:|
| 20% of always-Sol | 56.1% | 24.7 points | $13.52 | 80.9% |
| 40% of always-Sol | 66.5% | 14.4 points | $26.51 | 62.5% |
| 60% of always-Sol | 74.8% | 6.1 points | $42.70 | 39.7% |
| 80% of always-Sol | 79.9% | 0.9 points | $58.97 | 16.7% |

These are four examples from the curve, all chosen on calibration. They expose the price of additional savings; none is designated as the right quality loss.

```bash
# Evaluate your frozen policies on the published test answers and open the generated report.
uv run --frozen python router.py evaluate-refit --data data --run my-router --output my-results
```

Open `my-results/explorer/index.html`. Because these test answers are already public, this is a reproduction exercise, not a new blind evaluation. For a new dataset, preserve separate development, calibration and final-test populations before iterating.

## 7. Route a new question

Use the method and candidate ID copied from **calibration.html** for the same run. The command below uses `embedding__textcost`, the classifier with learned costs behind the headline result. Choose that method in the explorer and copy one of its frontier candidate IDs, or replace both values with another selected pair. The command predicts the route first, saves that decision, then calls exactly one answer model.

```bash
# Enter the OpenAI key used to pay for the selected Sol or Luna answer.
export OPENAI_API_KEY='your-openai-api-key'
```

```bash
# Route one question using a method and candidate ID selected together from calibration.
METHOD=embedding__textcost
CANDIDATE_ID=YOUR_CALIBRATION_CANDIDATE_ID
uv run --frozen --group collection --group encode --group outcomes python router.py answer \
  --run my-router --method "$METHOD" --candidate-id "$CANDIDATE_ID" \
  --question 'A robot starts at (0,0), facing north. It turns right, moves 3, turns left, moves 2, then turns left and moves 1. What is its final coordinate? Answer as (x,y).' \
  --output answers/robot
```

For Laya or LoRA, select `base` or `lora` (`base__textcost` or `lora__textcost` for learned costs), use `--group collection --group training` (add `--group outcomes` for learned costs), and choose `--device cpu` or `cuda`. These routes need no embedding API call.

To try the supplied models directly, fetch `--group weights`, use `--run data/reference` and select a policy from the supplied reference calibration explorer. This also provides live Laya inference after the CPU-only exercise, which reused its predictions without copying its checkpoint. Each request needs a new output directory so a failed call is not silently repeated.

## Serving speed and costs

| Router deployment | Mean request time | 95th-percentile time | Router cost / 10k requests |
|---|---:|---:|---:|
| Unchanged Laya, resident H100 | 80 ms | 243 ms | $1.08 busy-resource estimate |
| Adapted Laya, resident H100 | 78 ms | 235 ms | $1.05 busy-resource estimate |
| Adapted Laya, 8-core CPU / 32 GiB | 3.98 s | 13.70 s | $6.99 busy-resource estimate |
| SIE Cloud Qwen encoder | 204 ms | 226 ms | $0.81 validated singleton-fee projection |
| Resident embedding classifier | 0.049 ms | 0.065 ms | Application hosting extra |
| Classifier + text cost predictors | 2.40 ms | 4.00 ms | Application hosting extra |

Each model measurement uses the same 90 development questions after warmup. The Cloud encoder plus classifier averages approximately **204 ms**, or **207 ms** with the text cost predictors; these sums combine separately measured means, not an end-to-end latency benchmark. The encoder's longest observed request affects the mean more than its 95th percentile.

The Cloud encoder measurement includes network time and saving the raw response to disk; native model timings are resident-model measurements on the stated Modal hardware. The $0.81 fee projects the final-test question mix; the native busy-resource estimates use the 90-question development timing sample. Model loading and CLI startup are separate. These are sequential single-request measurements, not a throughput benchmark. The learned head adds very little work; deploying the 4B encoder is the larger serving requirement.

The opening table combines these router estimates with the selected answer costs. The explorer retains the original answer-only curves and can add per-request Cloud encoder fees; it does not insert a native hosting estimate into the frozen evaluation. Its unpriced native and CPU costs remain unknown, not zero.

The cost calculations are:

- **Sol and Luna:** sum each model's token cost only for questions sent to that model, then multiply by `10,000 / 975`. Input, cached-input and cache-write prices follow the saved usage; output tokens already include reasoning tokens, so reasoning is charged once. Multiplying a routing share by the model's overall mean cost would be wrong: routers select different questions, with different answer lengths.
- **Qwen embeddings:** the recorded Cloud rate is **$0.06 per million input tokens**, rounded up per request to a whole **$0.00001 credit**: `ceil(3 × input_tokens / 500)` credits. The 975 test questions total 1,235,875 encoder tokens and **7,910 credits after rounding**: `$0.07910 × 10,000 / 975 = $0.811282`. Both neighbors and classifiers need **one embedding for every incoming request**, whichever answer model is selected. The rate and rounding rule reproduce all 91 saved singleton charges and 245 successful batch charges. The table prices separate live requests, even when the exercise reuses cached vectors; it does not spread the original batch-collection bill across future requests.
- **Laya on H100:** the saved resource rate is **$0.00134388 per second** for an H100, eight CPU cores and 64 GiB RAM. For LoRA, `0.078329 seconds × $0.00134388 × 10,000 = $1.05265`; unchanged Laya uses its measured 0.080407 seconds and costs about $1.08. Each timing already includes **both correctness predictions**, tokenization and the original cost arithmetic. LoRA is merged into the model for inference; this is one adapted router, not an extra model call or a training charge.

The Laya estimate assumes the measured sequential serving time is billable active time. Startup, loading and idle capacity add cost; batching, different hardware or lower precision may change it. The CPU-only LoRA scenario above costs about **$6.99 / 10k**, rather than $1.05. Cloud API pricing and a dedicated server's resource cost should not be interpreted as equal-efficiency measurements. Small CPU heads, learned cost predictors and application hosting are still extra, as are one-time training and answer collection.

The downloaded evidence contains [per-question encoder fees](https://huggingface.co/datasets/superlinked/train-a-router-bbeh/blob/86dd7494a4c2396bc9ffa890f994b9722610b357/fees.json), [answer token usage and prices](https://huggingface.co/datasets/superlinked/train-a-router-bbeh/blob/86dd7494a4c2396bc9ffa890f994b9722610b357/grades/grades-final.json), and [Laya timings and resource rates](https://huggingface.co/datasets/superlinked/train-a-router-bbeh/blob/86dd7494a4c2396bc9ffa890f994b9722610b357/metadata/timing/native-hosting-comparison.json). These are pinned study prices, not a promise about future bills.

## What the results establish

The final curves replay policies chosen on calibration. Their 95% uncertainty intervals resample identified question families together within each task; they are pointwise intervals, not a guarantee over every possible threshold. The illustrated embedding + cost policy's 0.9-point accuracy loss has a 95% interval from −0.3 to 2.1 points: it does not establish a guaranteed loss below one point. The test covers new questions from this 23-task mix, not unseen kinds of reasoning. Public benchmark questions may also have appeared in model pretraining.

Answer scoring preserves meaningful punctuation and normalizes only source-validated equivalent forms. The primary test includes all 975 reserved questions. A supplementary 973-question check excludes two source questions whose answer formats were flagged as ambiguous before final answers were collected; it keeps every routing policy unchanged.

The example is supervised by observed outcomes, not by an LLM guessing which model is better. When reference answers are unavailable, a teacher can judge candidate answers, but those are noisy labels and need a separately checked evaluation set. [Labels and new answer collection](LABELS.md) explains that distinction and how to collect another paired run.

All code lives in this example; datasets, saved weights, answer recordings and measurements live in the pinned public Hugging Face release. The `source/`, `native/`, `embedding/` and `evaluation/` folders are parts of this one workflow. `frozen-source-manifest.json` identifies the exact calculation code used for the published results.
