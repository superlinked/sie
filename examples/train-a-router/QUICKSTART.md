# Train a cost-aware router

**Goal:** send requests to Luna when it is sufficient and pay for Sol when its extra accuracy is worthwhile. The supplied BBEH experiment compares unchanged Laya → LoRA adaptation → embedding neighbors → a small embedding classifier, with an optional learned cost estimate. On 975 held-out questions, an embedding classifier with learned cost estimates saved 16.7% with a 0.9-point accuracy loss; another preselected policy saved 39.7% with a 6.1-point loss. Both savings figures include the Cloud encoder fee.

**You need:** Python 3.12 and `uv`; an NVIDIA GPU for the full LoRA exercise (validated on H100 80 GB). Replay and classifier training run on CPU. SIE Cloud can supply Qwen3-Embedding-4B vectors; OpenAI supplies live Sol/Luna answers. Cached vectors and answers are included, so replay needs neither API key.

**1. Set up and inspect the evidence.** Training/development/calibration/test contain 1,969/589/385/975 questions. Related question families stay in one split. Accuracy measures the chosen answer; savings compare its cost with always-Sol. An accuracy change from 80% to 78% is a 2-point loss. Step numbers match the [full walkthrough](README.md).

```bash
# From your repository clone, install dependencies and download the pinned data and decision-model weights.
cd examples/train-a-router
uv sync --frozen --group outcomes
uv run --frozen python router.py fetch --output data --group core --group weights
uv run --frozen python router.py replay --data data --output published-results
uv run --frozen python router.py prepare --data data --run my-router
```

Open `published-results/explorer/index.html` for the full cost–quality curves.

The [opening comparison table](README.md#1-understand-the-data-and-baseline-sol-808-accuracy-luna-556) separates Sol answers, Luna answers, Laya serving estimates and embedding API fees. Include the router's cost as well as the selected answer; the explorer leaves native hosting unpriced, while the README shows an explicit H100 estimate.

For the **CPU embedding exercise**, install only `--group outcomes`, fetch only `--group core`, and run the following after `prepare`. Then continue at steps 4–5. This uses recorded Laya comparisons; it does not perform LoRA training.

```bash
# Supply the saved Laya predictions needed for comparison during an embedding-only refit.
uv run --frozen python router.py reuse-native --data data --run my-router
```

**2–3. Try and adapt Laya on GPU.** Start with unchanged weights, then let development data choose the LoRA training duration. The fast recipe uses less padding and FlashAttention for training, with the same full-precision validation and export. The complete fit took **28.6 minutes on H100**, versus **77.4 minutes** for the supplied reference recipe; setup and input scoring add time. `--recipe reference` retains the original recipe. [GPU setup and comparison](gpu/README.md) explains the locked environment and measured results.

```bash
# Score unchanged Laya, train LoRA, and save input-only predictions for both versions.
uv sync --project gpu --frozen
uv run --project gpu --frozen python gpu/smoke.py --output gpu-smoke.json
uv run --project gpu --frozen python router.py base --data data --run my-router --device cuda
uv run --project gpu --frozen python router.py fit-native --data data --run my-router --recipe fast
uv run --project gpu --frozen python router.py score --data data --run my-router --method native --device cuda
```

**4–5. Fit embedding routers on CPU.** The large Qwen encoder stays fixed. Neighbors average outcomes of similar questions; two small classifiers learn whether Luna and Sol will answer correctly.

```bash
# Fit neighbors, classifiers and simple controls, then score held-out question embeddings.
uv run --frozen --group outcomes python router.py fit-embedding --data data --run my-router
uv run --frozen --group outcomes python router.py score --data data --run my-router --method embedding
```

**6. Estimate cost and choose your tradeoff.** Cost varies by question, so retain both the original and learned cost estimates.

```bash
# Fit cost predictors, freeze the calibration choices, and open the calibration explorer.
uv run --frozen --group outcomes python router.py fit-cost --data data --run my-router
uv run --frozen python router.py calibrate --data data --run my-router
uv run --frozen python router.py frontier --run my-router --output calibration.html
```

Open `calibration.html` and record a **method and its candidate ID together** before viewing test results. For the headline approach, choose `embedding__textcost`; candidate IDs are specific to a method and run. There is no required quality-loss threshold.

```bash
# After selecting your calibration policy, evaluate the frozen choices against the published test answers.
uv run --frozen python router.py evaluate-refit --data data --run my-router --output my-results
```

Open `my-results/explorer/index.html` to assess your choice. This is a reproduction using published test answers, not a new blind evaluation.

**7. Try a live request.** Add a SIE key for an embedding route and an OpenAI key for the selected answer. Run `router.py answer` with both the method and candidate copied from calibration for the same run. [The live command](README.md#7-route-a-new-question) uses `embedding__textcost`. Laya routes need no embedding call; after the CPU shortcut, use the supplied `data/reference` run and weights for live Laya inference.

To regenerate vectors, use [SIE Cloud](README.md#use-sie-cloud-or-your-own-encoder) or [your own SIE server](self-host/README.md). See [labels and collection](LABELS.md) for a new paired answer run. The [README](README.md) includes measured latency, hardware requirements, uncertainty and the effect of router fees.
