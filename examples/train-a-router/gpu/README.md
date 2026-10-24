# CUDA environment for Laya training

This separate project installs the faster native recipe without changing the
example's CPU environment. Use Linux x86_64, Python 3.12 and an NVIDIA H100 with
80 GB memory. Other GPU memory sizes have not been validated for the full
8,192-token recipe. The wheels require glibc 2.28 or newer and a CUDA 12.8-capable
driver; NVIDIA lists Linux driver 570.26 for
[CUDA 12.8 GA](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).

From `examples/train-a-router`, using uv 0.5.31 or newer:

```bash
# Install public binary wheels into gpu/.venv; no compiler or API key is needed.
uv sync --project gpu --frozen --python 3.12

# Check imports and a real synthetic CUDA forward/backward/optimizer step.
uv run --project gpu --frozen python gpu/smoke.py --output runs/gpu-smoke.json
```

Keep the working directory at `examples/train-a-router` when following the main
guide. [`--project gpu`](https://docs.astral.sh/uv/reference/cli/#uv-run--project)
selects this environment without changing the working directory, so
`uv run --project gpu --frozen python router.py ...` uses the same run and data
paths as the CPU commands. The smoke uses random synthetic tensors: it downloads
no model, reads no answer data and makes no API calls. It verifies installed
versions, package isolation, finite gradients and one completed optimizer step.
It does not measure routing quality or full-model memory use.

On 2026-10-24, a fresh Linux H100 environment installed this lock from the public
indexes and passed the smoke with uv 0.5.31 and Python 3.12.12. A separate check
using the pinned Laya checkpoint also completed a synthetic update on the
longest example input (6,702 combined tokens), followed by FP32 SDPA validation;
its peak allocated memory was 44.99 GiB. These are installation and memory
checks, separate from the full training comparison.

The GPU project is for native training and scoring. Keep collection, Cloud
embedding and live API commands in the main environment with their documented
dependency groups. On a preconfigured VM, clear inherited `UV_NO_SYNC`,
`UV_PROJECT_ENVIRONMENT`, `UV_NO_BUILD` and `VIRTUAL_ENV` settings before installing, so uv
creates this project's environment.

## Locked binaries

| Dependency | Version | Public source |
|---|---|---|
| PyTorch | `2.9.1+cu128` | [PyTorch CUDA 12.8 index](https://download.pytorch.org/whl/cu128/) |
| FlashAttention 2 | `2.8.3.post1+cu.12.8.torch.2.9` | [Astral CUDA 12.8 index](https://wheels.astral.sh/simple/cu128/) |
| Transformers | `4.57.6` | PyPI |
| Laya | `0.3.29` | PyPI |
| PEFT | `0.18.1` | PyPI |
| einops | `0.8.2` | PyPI |

Astral documents its binary extension index in the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-gpu-enabled-pytorch-extensions).
The lock contains all transitive versions and wheel hashes. The selected
[CPython 3.12 FlashAttention wheel](https://wheels.astral.sh/artifacts/f688c8526ab0633dde21c1b3153959a728f7d6b275353c48d3f346862691b06a/flash_attn-2.8.3.post1+cu.12.8.torch.2.9-cp312-cp312-manylinux_2_24_x86_64.whl)
has SHA-256 `f688c8526ab0633dde21c1b3153959a728f7d6b275353c48d3f346862691b06a`.
The locked Torch wheel has SHA-256
`7cb4018f4ce68b61fd3ef87dc1c4ca520731c7b5b200e360ad47b612d7844063`.
The project forbids source builds for Torch and FlashAttention with uv's
`no-build-package` setting. Avoid a blanket `--no-build` on uv 0.5.31: it also
rejects this project's intentionally unbuilt virtual package.

## Evaluated dependency scope

As of 2026-10-24, upstream [FlashAttention 2.8.3.post2](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post2)
contains newer fixes but has no matching published Torch 2.9 wheel in the checked
upstream release or Astral index. This recipe uses the pinned public binary above
within the following scope:

- The [checkpoint-loading advisory](https://github.com/advisories/GHSA-7g5w-pq96-8c5w)
  affects FlashAttention's model and distributed checkpoint helpers.
  [Transformers' ModernBERT integration](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/modernbert/modeling_modernbert.py)
  imports its attention and rotary kernels instead. The smoke checks that those
  checkpoint helpers were not imported; Laya loads the pinned safetensors
  checkpoint through its existing loader.
- The [archive-extraction advisory](https://github.com/advisories/GHSA-w9r7-4gwr-j959)
  concerns `hopper/setup.py`. Installing the locked FA2 wheel, with its source
  build disabled, does not execute that helper.
- The [FA2 backward index fix](https://github.com/Dao-AILab/flash-attention/pull/2260)
  concerns large allocation offsets. The
  [pinned Laya encoder](https://huggingface.co/convaiinnovations/laya/blob/aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8/encoder/config.json)
  has 16 heads of width 64. At microbatch 4 and sequence limit 8,192, the
  `batch × rounded_sequence × heads × head_width` bound is 33,554,432, below
  the signed 32-bit limit. This assessment does not cover arbitrary larger
  models or batches.

The fast recipe uses FA2 for mixed-precision training. Development selection,
export and scoring explicitly use the original FP32 SDPA path; disabling
autocast alone would not make FA2 evaluation equivalent to FP32 SDPA.

## Compare your trained model with the supplied results

After the walkthrough's `fit-native --recipe fast` and native `score` commands,
run this comparison on CPU. It reuses the supplied cost predictors and keeps the
original routers alongside your new LoRA model. Both cost-estimation variants
retain their complete curves.

```bash
# Choose comparison policies on calibration data and freeze them before checking test answers.
uv run --frozen python training_comparison.py calibrate \
  --data data --native-run my-router/native --output training-comparison
```

```bash
# Evaluate those frozen choices against the same published test answers.
uv run --frozen python training_comparison.py evaluate \
  --data data --output training-comparison
```

Open `training-comparison/explorer/index.html` for the overlaid curves.
`comparison-report.json` also pairs the two training recipes at calibration
answer-cost caps of 20%, 40%, 60% and 80% of always-Sol. Each pair reports the
accuracy and dollar differences with uncertainty intervals. The actual test
spending can differ between recipes at the same calibration cap. These are
reproductions on already public answers, not new blind tests; there is no
required quality-loss threshold.
