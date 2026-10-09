# Server Plugin: Extending SIE with a Custom Adapter + LoRA

Everything the example's pipeline needs on the server side, packaged
as a thin extension you bake into a custom `sie-server` image. Build
once, run anywhere the parent `sie-server` runs.

## What's in here

| File | Role |
|------|------|
| `Dockerfile` | Builds a `sie-server` image with the custom adapter and model YAMLs below. |
| `adapters/stablebridge_pruner/__init__.py` | Custom `ModelAdapter` for unified reranking + token-level context pruning. 659 lines of production-ready Python. Wraps a frozen `BAAI/bge-reranker-v2-m3` with a trained `PruningHead` MLP (1024 → 512 → 1) and exposes both `score()` and `extract()` from one forward pass. |
| `models/answerdotai__ModernBERT-base.yaml` | Model config with a `us-regulatory` profile that activates `sugiv/modernbert-us-stablecoin-encoder` (a LoRA fine-tune) at request time. |
| `models/sugiv__stablebridge-pruner-highlighter.yaml` | Model config that wires the Stablebridge adapter up to `sie_id: sugiv/stablebridge-pruner-highlighter`, with `default` / `aggressive` / `conservative` pruning profiles. |

## Build + run

```bash
# From examples/regulatory-rag/
# GPU (recommended):
docker build -t sie-regulatory --build-arg SIE_TAG=latest-cuda12-default ./server-plugin
docker run --gpus all -p 8080:8080 sie-regulatory

# CPU (slow but works for the pipeline's small corpus):
docker build -t sie-regulatory --build-arg SIE_TAG=latest-cpu-default ./server-plugin
docker run -p 8080:8080 sie-regulatory
```

Then from the parent directory:

```bash
python rag_pipeline.py
```

## What this teaches

SIE is an inference **cluster**, not a closed box. Everything in this
folder is code you could write for your own domain:

- A **custom adapter** that wraps a frozen base model with a trained head to add a new primitive (here: token-level pruning under `extract()`).
- A **LoRA profile** registered in a model YAML so domain-adapted weights hot-load at request time without a separate deployment. The profile sets `options["lora_id"]`; sie-server already copies that onto worker `options["lora"]`, so no encode-route patch is required.
- Two new model IDs that the rest of the stack (SDK, router, autoscaler, monitoring) picks up automatically because they're registered the same way as any first-party SIE model.

## Version pinning

The `Dockerfile` accepts `SIE_TAG` for reproducibility. Pin that build
arg to the CPU or CUDA tag you want to publish against. LoRA id routing
ships in current sie-server, so moving to a newer upstream image does
not require re-applying a server patch.
