# Serve the Qwen encoder on your own GPU

This folder starts **one SIE embedding server** for the router. The two small
correctness predictors run in the example's Python process. **Sol and Luna
remain OpenAI API answer models**; this setup does not serve them locally.

Use a Linux x86-64 GPU machine, Docker Engine, Docker Compose v2
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The target machine for this example is an H100 80 GB, which also supports the
LoRA training exercise. Stop the encoder before training on the same GPU.

Compose pulls the published **SIE v0.10.1 CUDA default image**, pinned by its
content digest. The model configuration in this folder selects the Flash
adapter for the same Qwen3-Embedding-4B checkpoint. No SIE source build is
needed. The first pull downloads the image; the first server start downloads
approximately 8.04 GB of encoder weights. A named volume retains the weights
and kernel caches.

This choice uses an available release image. SGLang was faster in most of
our [matched 4B measurements](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/193a7dc36da168693a64e6add67d1f57dd69ba45/serving-perf-encoders/2026-10-22/qwen3-embedding-4b-adapters/README.md);
it remains the main model catalog's default. Flash and
SGLang produce close, but not identical, vectors. Keep the supplied vectors
for exact result replay; fit and calibrate with your fresh vectors when
building your own router.

Run these commands from
the example folder on the GPU machine:

```bash
# Check GPU visibility and the Compose configuration without starting a model.
nvidia-smi
docker compose -f self-host/compose.yaml config --quiet
```

```bash
# Download the pinned public SIE image; no local image build is needed.
docker compose -f self-host/compose.yaml pull encoder
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

The smoke validates the returned model, ID, vector size, finiteness and
near-unit norm at the model's BF16 precision. Feature collection then
normalizes in float64 before storing float32 vectors, just as the Cloud
feature path does. The smoke uses the same instruction and `is_query=true`
as the example. No SIE API key is needed; GPU hosting is charged by your
provider rather than by SIE Cloud.

```bash
# Regenerate question features locally while keeping the published vectors unchanged.
uv run --frozen --group encode python router.py encode \
  --data data --output local-features \
  --sie-base-url http://127.0.0.1:8000 --self-hosted
```

Use `--features local-features/features` for both classifier fitting and embedding
scoring in a new run, then repeat the calibration step in the main walkthrough.

```bash
# Use your freshly calibrated policy with the local encoder; OpenAI still supplies the selected answer.
CANDIDATE_ID=YOUR_CALIBRATION_CANDIDATE_ID
uv run --frozen --group encode --group collection --group outcomes python router.py answer \
  --run my-router --method embedding --candidate-id "$CANDIDATE_ID" \
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
local endpoint; do not expose the port publicly. The server runs as UID 1000
with a writable named cache.

The exact image, profile, server arguments, health check and smoke test passed
on an H100 80 GB as UID/GID 1000 with an empty model cache. The GPU check ran
on Modal; Docker Compose orchestration and host port mapping were checked
statically rather than run there. The [validation receipt](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/193a7dc36da168693a64e6add67d1f57dd69ba45/serving-perf-encoders/2026-10-22/qwen3-embedding-4b-adapters/released-image-validation.json)
records the image digest and hashes of these files.
