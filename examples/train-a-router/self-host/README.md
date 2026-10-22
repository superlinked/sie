# Serve the Qwen encoder on your own GPU

This folder starts **one SIE embedding server** for the router. The two small
correctness predictors run in the example's Python process. **Sol and Luna
remain OpenAI API answer models**; this setup does not serve them locally.

Use a Linux x86-64 GPU machine, Docker Engine with BuildKit, Docker Compose v2
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The encoder component has been exercised on an H100 80 GB. Smaller GPUs have
not been validated for this configuration. It reserves 85% of its GPU, so
stop it before running LoRA training on the same device.

Although its filename says `cuda12`, the SGLang runtime stage contains CUDA
**12.9.1** and compiles kernels at startup. Use a driver supporting that
toolkit; NVIDIA lists Linux driver **575.57.08 or newer** for CUDA 12.9 Update 1.
The generic CUDA-12 compatibility floor is insufficient to establish support
for these JIT kernels. See the [CUDA release notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cuda-driver).

There is no published SIE image for the `sglang-embedding` bundle at this
source revision. Compose builds the [public Dockerfile](https://github.com/superlinked/sie/blob/975d5309ba3af9eaeb13725b20dcf10d906c3e2a/packages/sie_server/Dockerfile.cuda12)
with `BUNDLE=sglang-embedding` from commit
`975d5309ba3af9eaeb13725b20dcf10d906c3e2a`. The image name in Compose is a local
build tag. No private repository, private image or package index is required.
The first build needs network access and substantial disk space for the CUDA
image and build layers. The first server start also downloads approximately
8.04 GB of encoder weights; a named volume retains weights and kernel caches.

Run these commands from
the example folder on the GPU machine:

```bash
# Check GPU visibility and the Compose configuration without starting a model.
nvidia-smi
docker compose -f self-host/compose.yaml config --quiet
```

```bash
# Build the public encoder image once; no GPU is required during image construction.
docker compose -f self-host/compose.yaml build encoder
```

```bash
# Start the encoder and wait for the exact model revision to be loaded.
docker compose -f self-host/compose.yaml up --detach --wait --wait-timeout 1200 encoder
```

Set `ENCODER_GPU` to a device ID before starting if GPU 0 is not the intended
device. The health check reads model metadata: it requires the pinned revision,
`loaded=true` and 2,560 output dimensions. It does not make inference requests.
Cold download/loading can take several minutes. Inspect a failed startup with
`docker compose -f self-host/compose.yaml logs --tail 100 encoder`.

```bash
# Make one synthetic embedding request through the SIE SDK; no answer model is called.
docker compose -f self-host/compose.yaml exec -T encoder \
  python /opt/check_encoder.py --smoke > encoder-smoke.json
```

The smoke validates the returned model, ID, vector size, finiteness and unit
norm. It uses the same instruction, `is_query=true` and float32 output as the
example. No API key is needed. GPU hosting remains a real cost; the smoke
does not assign it a Cloud debit or call it free.

```bash
# Regenerate question features locally while keeping the published vectors unchanged.
uv run --frozen --group encode python router.py encode \
  --data data --output local-features \
  --sie-base-url http://127.0.0.1:8000 --self-hosted
```

Use `--features local-features/features` for both classifier fitting and embedding
scoring in a new run. The model revision and encoding settings match the
recorded recipe, but a different inference runtime need not produce
bit-identical vectors. The existing recordings remain the exact replay path.

```bash
# Use your selected reference-frontier candidate with the local encoder; OPENAI_API_KEY is still needed for answers.
CANDIDATE_ID=YOUR_SELECTED_REFERENCE_CANDIDATE_ID
uv run --frozen --group encode --group collection --group outcomes python router.py answer \
  --run data/reference --method embedding --candidate-id "$CANDIDATE_ID" \
  --question 'Which planet is closest to the Sun? A. Mercury B. Venus. Answer A or B.' \
  --output answers/local-encoder-planet \
  --sie-base-url http://127.0.0.1:8000 --self-hosted
```

With `--self-hosted`, the example explicitly suppresses SIE Cloud credentials
for the encoder. Your shell environment does not need to change. Use a new
output directory for each live request.

```bash
# Stop the encoder while preserving the downloaded model and kernel caches.
docker compose -f self-host/compose.yaml down
```

For a GPU VM, run Compose on that machine. Run the example there too, or
forward its loopback endpoint over SSH:

```bash
# From your client machine, keep a tunnel to the GPU VM open while using the local endpoint.
ssh -N -L 8000:127.0.0.1:8000 your-user@your-gpu-vm
```

Only `127.0.0.1:8000` is published. Authentication is disabled for this trusted
local endpoint; do not expose the port publicly. The SGLang child stays inside
the container, and the server runs as UID 1000 with a writable named cache.

The source-built encoder component and pinned weights have passed a prior
GPU functionality check. This new single-service Compose wrapper has only
been checked statically; a complete Compose GPU launch has not yet been run.
Source/model pins do not make the Docker build fully reproducible: upstream
base-image tags and some transitive dependency ranges can move. Preserve the
resulting image and its digest for a deployment you intend to reproduce.
