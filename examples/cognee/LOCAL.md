# Choose local inference endpoints

The examples use the same API with SIE Local, SIE Cluster and SIE Serverless.
Configure each model role explicitly when its serving runtime uses a different
port. These commands follow the repository's [server setup](../../README.md)
and [model catalog](../../packages/sie_server/models). Check support and memory
requirements on the hardware you will present with before a workshop.

## Linux CPU

The default CPU bundle provides supported embedding/reranking models:

```sh
docker run -p 127.0.0.1:8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cpu-default
```

Generation, OCR and decision models need their own supported runtimes; the
default CPU bundle does not supply the whole demonstration.

## Apple Silicon

Run embedding/reranking and MLX generation in separate environments and ports:

```sh
uvx --from 'sie-server[local]' sie-server serve -p 8080
# In a separate terminal:
uvx --with 'mlx-lm>=0.30.7' --from sie-server sie-server serve -b sglang -p 8081
```

The model catalog supplies MLX checkpoint mappings. Generation needs Apple
Silicon, sufficient unified memory and structured JSON support for the selected
model. These services cover the memory experiment's LLM and embeddings;
configure supported OCR and decision endpoints separately for those steps.

## Linux with NVIDIA GPU

Use separate bundle images for incompatible model runtimes:

| Role | Image suffix | Host port |
| --- | --- | --- |
| Embeddings and reranking | `latest-cuda12-default` | 8080 |
| Structured generation | `latest-cuda12-sglang` | 8081 |
| Default OCR profile | `latest-cuda12-sglang-vision-extract` | 8082 |
| GLiNER2.5-Decide | `latest-cuda12-transformers5` | 8083 |

For example, the generation service:

```sh
docker run --gpus all -p 127.0.0.1:8081:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda12-sglang
```

Use the corresponding image suffix and port for each other service. Several
models may not fit on one GPU simultaneously. Size workers for the selected
models, or use SIE Cluster to route requests to suitable workers.

## Connect the examples

Once each role is served, set these values in the example's `.env`:

```dotenv
SIE_BASE_URL=http://localhost:8080
SIE_API_KEY=
SIE_EMBEDDING_BASE_URL=http://localhost:8080
SIE_RERANK_BASE_URL=http://localhost:8080
SIE_LLM_BASE_URL=http://localhost:8081
SIE_OCR_BASE_URL=http://localhost:8082
SIE_DECISION_BASE_URL=http://localhost:8083
WORKSHOP_NAMESPACE=local-workshop-v1
```

Remove conflicting exported shell values; they take precedence over `.env`.
If a model has a different served ID, set its `SIE_<ROLE>_MODEL` override.

Unauthenticated loopback compatible clients receive a harmless placeholder
key. Authenticated services need actual role-specific keys. For SIE Cluster,
set the base URL to its gateway and supply keys as appropriate.

Combining local and Serverless roles is supported. Describe that as mixed
inference: local memory does not keep remote model inputs on your machine.
Fully local execution requires every model role to be served locally.
