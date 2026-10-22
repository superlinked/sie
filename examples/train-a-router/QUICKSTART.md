# Train a cost-aware router

**Goal:** send requests to Luna when it is sufficient and pay for Sol when its extra accuracy is worthwhile. The supplied BBEH experiment compares unchanged Laya → LoRA adaptation → embedding neighbors → a small embedding classifier, with an optional learned cost estimate. On 975 held-out questions, an embedding classifier with learned cost estimates saved 16.7% with a 0.9-point accuracy loss; another preselected policy saved 39.7% with a 6.1-point loss. Both savings figures include the Cloud encoder fee.

**You need:** Python 3.12 and `uv`; an NVIDIA GPU for the full LoRA exercise (validated on H100 80 GB). Replay and classifier training run on CPU. SIE Cloud can supply Qwen3-Embedding-4B vectors; OpenAI supplies live Sol/Luna answers. Cached vectors and answers are included, so replay needs neither API key.

1. **Set up and inspect the evidence.** Training/development/calibration/test contain 1,969/589/385/975 questions. Related question families stay in one split. Accuracy measures the chosen answer; savings compare its token cost with always-Sol. An accuracy change from 80% to 78% is a 2-point loss.

```bash
# From your repository clone, install dependencies and download the pinned data and decision-model weights.
cd examples/train-a-router
uv sync --frozen --group training --group outcomes
uv run --frozen python router.py fetch --output data --group core --group weights
uv run --frozen python router.py replay --data data --output published-results
uv run --frozen python router.py prepare --data data --run my-router
```

Open `published-results/explorer/index.html` for the full cost–quality curves. A CPU-only replay needs only `--group core`.

2. **Try and adapt Laya.** Start with unchanged weights, then let development data choose the LoRA training duration. The supplied run selected three passes and refitted from the original model.

```bash
# Score unchanged Laya, train LoRA, and save input-only predictions for both versions.
uv run --frozen --group training python router.py base --data data --run my-router --device cuda
uv run --frozen --group training python router.py fit-native --data data --run my-router
uv run --frozen --group training python router.py score --data data --run my-router --method native --device cuda
```

3. **Fit embedding routers on CPU.** The large Qwen encoder stays fixed. The neighbor method averages outcomes of similar questions; two small logistic classifiers learn whether Luna and Sol will answer correctly.

```bash
# Fit neighbors, classifiers and simple controls, then score held-out question embeddings.
uv run --frozen --group outcomes python router.py fit-embedding --data data --run my-router
uv run --frozen --group outcomes python router.py score --data data --run my-router --method embedding
```

4. **Estimate cost and choose your tradeoff.** Cost varies by question, so retain both the original and learned cost estimates. Select a policy using calibration, then inspect its test result.

```bash
# Fit cost predictors, freeze the full calibration frontier, and render the two reports.
uv run --frozen --group outcomes python router.py fit-cost --data data --run my-router
uv run --frozen python router.py calibrate --data data --run my-router
uv run --frozen python router.py frontier --run my-router --output calibration.html
uv run --frozen python router.py evaluate-refit --data data --run my-router --output my-results
```

Open `calibration.html` to select a method and candidate ID; open `my-results/explorer/index.html` to assess it. There is no required quality-loss threshold. Evaluation of your refit uses published test answers, so it is a reproduction, not another blind test.

5. **Try a live request.** Add a SIE key for an embedding route and an OpenAI key for the selected answer. Run `router.py answer` with the method and candidate copied from calibration. [The full walkthrough](README.md#7-route-a-new-question) gives the command. Laya routes need no embedding call.

To regenerate vectors, use [SIE Cloud](README.md#use-sie-cloud-or-your-own-encoder) or [your own SIE server](self-host/README.md). See [labels and collection](LABELS.md) for a new paired answer run. The [README](README.md) includes measured latency, hardware requirements, uncertainty and the effect of router fees.
