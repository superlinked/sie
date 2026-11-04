# Lower Luna’s reasoning effort

**Luna low cost 54.9% less than Luna medium, with an 11.3-point accuracy loss.** Its mean answer time was 9.0 seconds versus 22.0 seconds. We compared all **975 BBEH test questions**, reusing the existing medium recordings; the runs happened on different days, so provider load also affects the latency comparison.

Sol stays at medium. The existing Laya, LoRA and embedding routers keep their original decisions, so this measures what happens when you change Luna’s reasoning setting without retraining the router.

## Results

Costs below are estimates per 10,000 requests. Accuracy loss and savings are relative to always using Sol. Mean and 95th-percentile times cover the selected answer request, before adding any router inference.

| Answer model | Accuracy | Accuracy loss (points) | Answer cost | Savings | Mean time | 95th-percentile time |
|---|---:|---:|---:|---:|---:|---:|
| Sol, medium | 80.8% | 0.0 | $70.77 | 0.0% | 9.87 s | 27.79 s |
| Luna, medium | 55.6% | 25.2 | $12.23 | 82.7% | 22.01 s | 72.03 s |
| Luna, low | 44.3% | 36.5 | $5.52 | 92.2% | 9.04 s | 19.40 s |

Luna low answered **432 questions correctly**, versus **542** at medium. The paired 95% interval for the accuracy loss is **8.2 to 14.4 points**. Average reasoning tokens fell from **2,126 to 783**. All 975 low responses completed with valid answer formats; there were no retries or truncated low answers.

Changing Luna also changes the answers delivered by the routers:

| Frozen policy | Luna effort | Accuracy | Accuracy loss (points) | Total cost | Savings | Mean answer time | 95th-percentile answer time |
|---|---|---:|---:|---:|---:|---:|---:|
| Laya + LoRA, 80% cap | medium | 78.3% | 2.6 | $57.85 | 18.2% | 12.45 s | 32.62 s |
| Laya + LoRA, 80% cap | low | 75.3% | 5.5 | $56.80 | 19.7% | 10.28 s | 27.52 s |
| LoRA + learned cost, 80% cap | medium | 79.0% | 1.8 | $58.68 | 17.1% | 12.08 s | 32.36 s |
| LoRA + learned cost, 80% cap | low | 77.1% | 3.7 | $57.79 | 18.3% | 10.25 s | 27.61 s |
| Embedding classifier + learned cost, 80% cap | medium | 79.9% | 0.9 | $58.97 | 16.7% | 11.81 s | 33.99 s |
| Embedding classifier + learned cost, 80% cap | low | 77.2% | 3.6 | $58.05 | 18.0% | 9.94 s | 27.13 s |
| Embedding classifier + learned cost, 60% cap | medium | 74.8% | 6.1 | $42.70 | 39.7% | 14.32 s | 44.65 s |
| Embedding classifier + learned cost, 60% cap | low | 68.8% | 12.0 | $40.19 | 43.2% | 9.49 s | 23.73 s |

The caps refer to the original calibration answer-cost budgets. These are the same policies illustrated in the main example. Their decisions and Sol shares stay fixed; we do not choose new thresholds from these test answers. All nine comparisons, including unchanged Laya, neighbors and text controls, are in the [full report](https://huggingface.co/datasets/superlinked/train-a-router-bbeh/blob/main/effort/2026-10-26/luna-low/report.md).

Total cost adds the original **$1.05 / 10k busy-H100 estimate for LoRA**, or **$0.81 / 10k SIE Cloud encoder fee** for embedding routers. Application hosting and idle capacity are extra. The router cost is unchanged between medium and low. [The main example](../README.md) explains the cost breakdown and serving measurements.

For the embedding policy at the 80% cap, low saves another **$0.92 / 10k requests** and reduces observed mean answer time by **1.88 seconds**, while losing another **2.7 accuracy points**. Sol still supplies 79.2% of answers, so cutting Luna’s bill has a small effect on the total. Adding the separately measured encoder/classifier mean gives an estimated **12.02 → 10.14 seconds** for router plus answer; this is not a measured end-to-end service time. Percentiles cannot be added this way.

[Download the responses and analysis](https://huggingface.co/datasets/superlinked/train-a-router-bbeh/tree/main/effort/2026-10-26/luna-low), or replay them below.

## Replay the results on CPU

No API key or GPU is needed. From a clone of this repository:

```bash
# Enter the example and install the locked analysis dependencies.
cd examples/train-a-router
uv sync --frozen --group collection --group outcomes
```

```bash
# Download the original questions, reference answers and frozen routing policies.
uv run --frozen python router.py fetch --output data --group core
```

```bash
# Download and verify the Luna-low results, then unpack their saved API responses.
uv run --frozen python router.py fetch --release effort/release.json --output luna-low-evidence
tar -xzf luna-low-evidence/evidence.tar.gz -C luna-low-evidence
```

```bash
# Grade all 975 answers and compare the original policies with Luna low substituted.
uv run --frozen --group collection --group outcomes python -m effort.analyze \
  --data data --collection luna-low-evidence/collection --output luna-low-report
```

Open `luna-low-report/report.md`. `report.json` also contains results by task, paired uncertainty intervals, token counts, missing-value coverage and all nine policy comparisons. `graded-rows.json` contains the question-level measurements. Use a new report directory for each replay; the analyzer preserves previous reports.

## Collect fresh Luna-low answers on CPU

Use the same downloaded `data` directory. This makes paid OpenAI calls with **up to 16 requests in flight**. It calls only `gpt-6-luna`, with `reasoning.effort=low`; it reuses the existing Luna-medium and Sol-medium recordings for comparison.

```bash
# Configure the OpenAI account that will pay for fresh Luna-low answers.
export OPENAI_API_KEY='your-openai-api-key'
```

```bash
# Collect Luna-low answers to all 975 questions into a fresh directory.
uv run --frozen --group collection python -m effort.collect_low \
  --inputs data/source/question-only/final-inputs.json \
  --manifest data/manifest.json --source-plan data/source/source-plan.json \
  --output my-luna-low --budget-usd 20.00 --deadline-seconds 5400 --execute
```

The budget is a ceiling, not a target. Our complete run used **$0.54 in estimated API usage**, plus worker hosting, and collection took **13.5 minutes on a Modal CPU worker**. Omitting `--execute` only saves the request plan. Repeating the exact command resumes that plan; completed answers and requests with uncertain execution are not sent again. Invalid or truncated answers are kept. Only explicit transient failures can be retried.

```bash
# Grade your new answers using the same references and routing decisions.
uv run --frozen --group collection --group outcomes python -m effort.analyze \
  --data data --collection my-luna-low --output my-luna-low-report
```

## How to read the comparison

- **Accuracy:** correct answers divided by all 975 questions, including invalid or truncated responses. Grading uses the original reference answers and normalization rules.
- **Cost:** input and output tokens at the study’s standard API prices, including cached-input accounting. Reasoning tokens are already part of output tokens and are charged once. Router costs are identified separately.
- **Answer time:** elapsed time for the selected API attempt, including client/network work. It excludes router inference, collection queue time and retry backoff. The 95th percentile is the time within which 95% of these answers finished; time to first token was not measured.
- **Uncertainty:** paired 95% intervals resample related question families together within each task. They describe this question mix, not a guarantee for new kinds of work.

Low and medium answers were collected on different days. Their latency difference includes changes in provider load and serving conditions as well as reasoning effort. This is a comparison of saved request times, not a simultaneous load test.

The routers learned from medium-effort answers and costs. Their route choices stay fixed here; no policy was selected using the new low-effort test answers. Training a router specifically for Luna low would require new training and calibration answers, followed by a fresh test. The main example’s medium-based results remain unchanged.

[Return to the full example](../README.md)
