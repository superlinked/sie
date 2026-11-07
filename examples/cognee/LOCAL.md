# Run SIE Local or SIE Cluster

The examples use the same API with SIE Local, SIE Cluster and SIE Serverless.
Configure each model role explicitly when its serving runtime uses a different
port. These commands follow the repository's [server setup](../../README.md)
and [model catalog](../../packages/sie_server/models). Check that your machine
supports the selected models and has enough memory to run them.

The commands below start SIE Local services on the machine where they run.
For model workers in your cloud, follow the [SIE Cluster section](#sie-cluster).

## Linux CPU

The default CPU bundle provides supported embedding/reranking models:

```sh
docker run -p 127.0.0.1:8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cpu-default
```

Generation, OCR and decision models need their own supported runtimes; the
default CPU bundle does not supply every example's model roles.

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

### CUDA 13 alternative

`Qwen/Qwen3.8-27B-FP8` uses the CUDA 13 SGLang bundle. To use this model
override, replace the CUDA 12 generation service above with this service.
Stop the previous generation container before reusing host port 8081:

```sh
docker run --gpus all -p 127.0.0.1:8081:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda13-sglang-cu130
```

Set the generation endpoint and model in the example's `.env`:

```dotenv
SIE_LLM_BASE_URL=http://localhost:8081
SIE_LLM_MODEL=Qwen/Qwen3.8-27B-FP8
```

The other model roles keep their own services and endpoints.

## Connect to SIE Local

Once each role is served, set these values in the example's `.env`:

```dotenv
SIE_BASE_URL=http://localhost:8080
SIE_API_KEY=
SIE_EMBEDDING_BASE_URL=http://localhost:8080
SIE_RERANK_BASE_URL=http://localhost:8080
SIE_LLM_BASE_URL=http://localhost:8081
SIE_OCR_BASE_URL=http://localhost:8082
SIE_DECISION_BASE_URL=http://localhost:8083
SIE_COGNEE_NAMESPACE=local-example-v1
```

Remove conflicting exported shell values; they take precedence over `.env`.
If a model has a different served ID, set its `SIE_<ROLE>_MODEL` override.

Unauthenticated loopback generation/embedding clients receive a harmless
placeholder key. Configure the authentication required by each other service.

## SIE Cluster

Use the [SIE Cluster Helm guide](../../deploy/helm/sie-cluster/README.md) to
deploy model workers in your cloud. Enable workers that support the selected
model roles, then point the examples at the Cluster gateway.

To use one gateway for every role, remove the `SIE_<ROLE>_BASE_URL` and
`SIE_<ROLE>_API_KEY` entries from `.env` for `LLM`, `EMBEDDING`, `OCR`, `RERANK`
and `DECISION`. Unset any exported copies as well:

```sh
unset SIE_LLM_BASE_URL SIE_LLM_API_KEY \
  SIE_EMBEDDING_BASE_URL SIE_EMBEDDING_API_KEY \
  SIE_OCR_BASE_URL SIE_OCR_API_KEY \
  SIE_RERANK_BASE_URL SIE_RERANK_API_KEY \
  SIE_DECISION_BASE_URL SIE_DECISION_API_KEY
```

Set `SIE_BASE_URL` to the gateway URL and `SIE_API_KEY` to its key. Exported
shell values take precedence over `.env`, including these general settings.
To route roles separately, set each role's URL and key explicitly. Changing
only `SIE_BASE_URL` leaves existing role-specific overrides in effect.

Combining local and Serverless roles is supported. Describe that as mixed
inference: local memory does not keep remote model inputs on your machine.
Fully local execution requires every model role to be served locally.
