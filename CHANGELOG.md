# Changelog

## v0.7.3 (2026-09-03)

### Highlights

This release adds MADLAD translation and improves how generation requests finish, fail, and cancel.

### Features

- Translate text with `google/madlad400-3b-mt`, served through CTranslate2 with native request batching. The initial profile uses FP32 and supports up to 512 input and 512 output tokens.
- An opt-in TensorRT-LLM bundle provides an encoder-decoder adapter on CUDA 13. It is not the default backend for the model catalog.
- The Rust/Candle worker implementation is now included, covering native embeddings and ColBERT scoring.
- Converted serving artifacts can be pinned and verified before loading, then reused from the local cache for offline serving.

### Bug fixes

- Cancelled generation streams now close their upstream requests, and model unloading drains active generation before releasing the runtime.
- Model-load failures and generation errors remain errors through streaming and buffered responses. Python and TypeScript clients retain useful error codes, invalid-parameter details, and validated retry hints without exposing raw backend diagnostics.
- Streaming support is reported in model capabilities. Unsupported streaming requests are rejected before execution.
- Encoder-decoder models enforce separate input and output limits instead of incorrectly treating them as one shared context window.

## v0.7.2 (2026-08-27)

### Highlights

Qwen3.8 joins the model catalog, Alibaba Cloud deployments gain native object-storage support, and a broad set of fixes improves SDK errors, batching, and self-hosted reliability.

### Features

- Added `Qwen/Qwen3.8-27B-FP8` for text and image generation, with tool calling and structured output. Alongside the conservative default profile, explicit H100, H200, and RTX PRO 6000 profiles support a 256K context window. These profiles return answers without a separate thinking mode.
- Added native Alibaba Object Storage Service support for queued payloads and SDK storage access, plus ACK Helm settings and RRSA workload identity.
- Self-hosted clusters can configure file-backed and replicated NATS work queues instead of relying only on the default in-memory, single-replica queue.

### Bug fixes

- Python and TypeScript SDKs detect incomplete batch responses instead of silently pairing results with the wrong inputs. Mixed text-and-image batches retain their input ordering, and multimodal embedding models batch text-only inputs correctly.
- Generation failures, including empty output and model-loading errors, are surfaced consistently. Retry handling better distinguishes a model still loading from a permanent failure, and stream errors retain their request identifiers.
- The OpenAI-compatible embeddings endpoint rejects unsupported `dimensions` values instead of silently returning a different vector width. Cold loads return a retryable response rather than holding the request open throughout model loading.
- Invalid image data receives a useful input error, unknown model identifiers include nearby matches, and non-finite reranker scores are rejected.
- Model eviction no longer leaves queued requests waiting indefinitely or blocks unrelated requests while a worker shuts down. Failed engine starts release their ports and report the underlying crash instead of a misleading timeout.
- Helm deployments gain gateway disruption protection and shutdown budgets, ingress timeouts aligned with model loading, and fixes for KEDA scale-to-zero checks and ACK storage permissions.
- ColPali and ColQwen checkpoint revisions are pinned correctly for repeatable loading.

### Performance improvements

- SGLang compiler caches can persist across worker restarts, with separate cache entries for incompatible GPU and runtime combinations.
- Qwen3.8 hardware profiles tune speculative decoding and scheduling.

### Upgrade notes

- The CUDA 13 `gemma` bundle is renamed to `sglang-cu130`. Update explicit bundle selections and image references from `cuda13-gemma` to `cuda13-sglang-cu130`.
- The TypeScript SDK adds the explicit `timeoutMs` option. The older `timeout` option remains a millisecond-based alias.
- File-backed queues are opt-in. Changing the setting does not convert existing NATS streams; existing streams need a planned drain and recreation. Replication also requires a NATS cluster.

## v0.7.1 (2026-08-09)

Docling OCR now explicitly uses the English recognizer, fixing English words being joined together. GLiNER rejects blank documents and label prompts that leave no room for document text before running extraction.

## v0.7.0 (2026-08-08)

### Bug fixes

- Python and TypeScript clients handle malformed generation responses and unexpected redirects as clear errors, without copying response bodies into diagnostics.
- Queued extraction results retain the input item identifier, so callers can associate results with their documents.
- Grounding DINO normalizes free-form detection prompts to the format expected by the model.
- Models selected through `SIE_PINNED_MODELS` remain resident when starting the server. Inconsistent pinned-model and model-filter settings are rejected at startup.

## v0.6.30 (2026-08-07)

Qwen3.6 thinking profiles enable CUDA graphs and overlapping scheduling while retaining non-speculative decoding for reasoning and structured-output compatibility.

## v0.6.29 (2026-08-07)

Gemma 4 thinking profiles gain speculative decoding with the matching assistant model. Separate non-speculative profiles remain available for structured output, preserving the selected context window and thinking mode.

## v0.6.28 (2026-08-07)

### Features

- Qwen3.6 and Gemma 4 non-thinking profiles gain tuned speculative-decoding configurations for long-context generation.
- Structured-output routing can select a profile-specific, non-speculative counterpart that preserves the requested context window and thinking mode.

### Bug fixes

- Speculative draft models use pinned revisions and are prepared alongside locally cached main models. An unrelated cached revision no longer satisfies an explicitly requested model revision.

## v0.6.27 (2026-08-06)

### Highlights

Generation gained a direct-server Responses endpoint and more reliable handling of model-specific reasoning, while the retrieval catalog expanded with multilingual dense and late-interaction models.

### Features

- Added `lightonai/mLateOn` for multivector embeddings and scoring, and `ibm-granite/granite-embedding-97m-multilingual-r2` for dense embeddings.
- Added stateless, non-streaming text requests at the direct server's `/v1/responses` endpoint, with explicit errors for unsupported options.
- Added hardware-specific long-context and thinking profiles for selected Qwen and Gemma models. Context and output limits remained profile-specific; the default model settings were not expanded globally.
- Enabled generation streams over the local worker-ingest connection, including request cancellation.

### Bug fixes

- Prevented Qwen and Gemma reasoning blocks from appearing in visible answers when thinking is disabled, including delimiters split across streamed chunks.
- Enabled Python clients to retry capacity failures received before generation starts, including failures delivered inside an SSE stream.
- Kept Docling's normal document parsing available when optional OCR assets fail to initialize.
- Corrected native extraction image preprocessing and preserved Qwen3-VL reranker batching with newer Transformers versions.
- Redirected the server's outdated root playground to its interactive API documentation.

## v0.6.26 (2026-08-02)

Added H100 FP8 generation profiles for `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B`, and `google/gemma-4-31B-it`. The newly added 35B Qwen and 31B Gemma configurations initially exposed an 8K context window, rather than their checkpoints' full native context.

## v0.6.25 (2026-07-30)

### Features

- Added `Qwen/Qwen3-Embedding-8B` for text embeddings.

### Bug fixes

- Added early validation of OpenAI-compatible embedding requests containing more than 256 inputs to prevent oversized result payloads.
- Applied supported sequence-length limits to GLiNER and NuNER model configurations.
- Bounded recovery work when an invalid input affects a shared encoding batch, preventing repeated decoding failures from triggering unbounded retries of smaller batches.

## v0.6.24 (2026-07-26)

### Features

- Added exact adapter-revision pinning to LoRA entries in model configuration for reproducible loading.

### Bug fixes

- Included missing tokenizer dependencies in the affected model configurations.

### Performance improvements

- Cached MUVERA projection state across requests, avoiding reconstruction of the same random projection structures for every encoding operation.

## v0.6.23 (2026-07-24)

### Highlights

The direct server's `/v1/generate` endpoint gained image-input and structured-output support. Python and TypeScript SDKs also gained helpers for image inputs, grammars, and Responses; the Responses helpers covered stateless, non-streaming text requests.

### Features

- Added `naver/v-splade-quality` for sparse text and image embeddings.

### Bug fixes

- Corrected context-sensitive structured-output schema validation and made malformed grammar errors consistent.
- Restored trained ColBERT projections and query-expansion behavior, including the ColBERTv2 and Jina ColBERT retrieval recipes.
- Serialized concurrent tokenizer and hidden-state operations that could otherwise interfere with one another.
- Preserved pinned revisions when loading trusted model code, including fallback paths.
- Removed unsupported visual MUVERA profiles from the advertised catalog.

## v0.6.22 (2026-07-22)

Docling gained support for verified, immutable model artifacts. Staged file-inventory and hash checks made missing or mismatched assets explicit instead of silently using a different artifact set.

## v0.6.21 (2026-07-21)

### Highlights

This release added a text-reranking compatibility API and expanded the embedding and transcription catalog, alongside fixes to retrieval and vision-model execution.

### Features

- Added Cohere-compatible text-only reranking at `/v1/rerank` and `/v2/rerank`, with strict validation of supported request fields and complete-result handling.
- Added `Snowflake/snowflake-arctic-embed-s` for text embeddings and `openai/whisper-large-v3-turbo` for audio transcription.

### Bug fixes

- Corrected `Alibaba-NLP/gte-Qwen2-7B-instruct` serving to use its checkpoint's bidirectional embedding implementation rather than a causal generation implementation.
- Restored the 512-token capacity of `prithivida/Splade_PP_en_v2` while keeping its retrieval-specific query and document limits separate.
- Prevented concurrent ColPali forwards from interfering with Transformers' output recording and stopped temporary tensors from accumulating between requests.
- Corrected model discovery for an explicitly empty selection, which previously advertised every model in the bundle.
- Added validation for blank reranking inputs and filtered relation results whose endpoints were not selected by the extraction request.

## v0.6.20 (2026-07-18)

### Highlights

The model catalog expanded with compact multilingual retrieval and text-classification options. Generative OCR models gained a shared SGLang serving path with continuous batching.

### Features

- Added `intfloat/multilingual-e5-small`, `tencent/R3-embedding-0.6b`, and `tencent/R3-rerank-0.6b`.
- Added `fastino/gliguard-LLMGuardrails-300M` for text classification.
- Enabled LightOnOCR, PaddleOCR-VL, and GLM-OCR serving through the generative OCR adapter.

### Bug fixes

- Corrected classification-threshold and multi-label handling in GLiNER2-based classification.
- Corrected GLiREL entity offsets and relation text when mapping tokenized inputs back to the original text.

## v0.6.19 (2026-07-14)

### Bug fixes

- Corrected unknown-model responses to return `404` instead of `500`.
- Marked permanently failed model loads as terminal failures, preventing queued requests from waiting for a model that cannot become ready.
- Preserved one multivector result per input, in input order, when encoding with the ColBERT adapter.
- Restored Florence-2 processor compatibility and pinned offline chat-template rendering to the configured tokenizer revision.
- Protected cross-encoder tokenization from concurrent access during inference and token counting.

## v0.6.18 (2026-07-12)

Gateway configuration diagnostics were updated to redact API bearer tokens, administrator tokens, and configuration-service credentials instead of including their values in debug output.

## v0.6.17 (2026-07-09)

### Features

- Added support for multiple GPU worker children behind one sidecar, distributing work according to each child's readiness and queue pressure.
- Added `vidore/colSmol-256M` for text and image multivector embeddings, with an optional MUVERA representation.

### Bug fixes

- Kept worker IPC health checks responsive when GPU health inspection fails.
- Released cancelled scheduler reservations and bounded queue admission through completion, preventing cancelled work from leaving a worker appearing permanently busy.

## v0.6.16 (2026-07-07)

### Bug fixes

- Restored NV-Embed-v2's native encoding recipe, including its instruction-aware latent-attention pooling, instead of generic embedding pooling.
- Applied trained PyLate Dense projection chains in ColBERT adapters and aligned the ModernBERT implementation with the checkpoint's forward pass.
- Restored query and document prefixes on the SentenceTransformer profile for `intfloat/multilingual-e5-large`.
- Fixed scale-from-zero handling for GPU-agnostic requests in multi-profile pools and normalized explicit GPU demand labels consistently.

## v0.6.15 (2026-07-03)

### Highlights

Apple Silicon gained an MLX generation backend for configured models, including Qwen3.5-4B. Self-hosted Helm deployments gained opt-in distributed tracing with a bundled OpenTelemetry collector and optional Tempo backend.

### Bug fixes

- Restored the checkpoint-specific embedding recipes for EmbeddingGemma, GTE-Qwen2, and Stella models, including their tokenizer and query-instruction handling.
- Applied profile runtime options consistently on queued encoding requests and preserved the distinction between unsigned-byte and binary embeddings.
- Corrected out-of-memory errors on the OpenAI-compatible embeddings endpoint to return `503 RESOURCE_EXHAUSTED`.
- Bounded continuous-batch draining so a busy LoRA adapter cannot indefinitely delay other adapters.
- Improved cleanup during model unload and serialized hot-reload changes to the model registry.
- Preserved trace context through embedding rewrites and queue dispatch.
- Prevented late replies from abandoned generation attempts from replacing the active result.

## v0.6.14 (2026-06-26)

Fixed ColBERT query–document pair scoring, including the ModernBERT and rotary variants, so retrieval results can be reranked without server errors. MUVERA encoding now returns the requested dense vectors instead of dropping them from the response.

## v0.6.13 (2026-06-25)

Live model-configuration updates now preserve profile-qualified variants, validate changes before applying them, and unload removed variants safely. Cleanup of large request payloads now tracks the exact stored objects and retries failed deletions.

MCP tools now support document summarization, entity extraction, and masking detected PII. Large-text extraction uses bounded overlapping chunks, and operators can choose models and GPU routing separately for different tools.

## v0.6.12 (2026-06-24)

GLiClass now returns an actionable input-too-long error when the combined label and text input exceeds its context window. The gateway also avoids unnecessary storage deletions for requests that were never offloaded to object storage.

## v0.6.11 (2026-06-23)

### Highlights

This release expands text generation with Gemma 4 and makes self-hosted capacity easier to keep ready for requests.

### Features

- Added Gemma 4 E2B, E4B, and 26B-A4B model configurations in a dedicated `gemma` bundle using CUDA 13.
- Pinned models are now loaded on assigned workers and protected from idle and memory-pressure eviction.
- Logical resource pools can use existing worker capacity through a configurable backing queue pool, with Python and TypeScript SDK support.

### Bug fixes

- Generation requests made while a model is loading receive a retryable response. Queue retries preserve their delivery limits, and worker fallback routing respects the requested pool.
- Helm reports incompatible bundle/platform selections before deploying workers with an unavailable image.

## v0.6.10 (2026-06-22)

### Highlights

This release corrects embedding and object-detection outputs and makes pending generation work visible in the model and cluster-status APIs.

### Features

- Added a self-hostable MCP server for document conversion, image descriptions, document question answering, and structured output. It uses SIE's public inference APIs and requires a configured SIE endpoint, the appropriate models, and authentication settings. Question answering works on the documents supplied to each call, without a persistent index.

### Bug fixes

- Qwen3-VL embeddings now use the model's final normalized hidden state and consistent instruction formatting, including a default for blank instructions.
- Florence-2 detection results now return pixel-space `[x, y, width, height]` bounding boxes.
- Python installations can resolve the `transformers5` bundle without conflicting with the server's dependency constraints.

### Breaking changes

- The bundled Florence-2 base-ft and large configurations now default to object detection instead of OCR. Set the extraction task explicitly if you need OCR.
- Removed the bundled `naver-clova-ix/donut-base-finetuned-rvlcdip` model configuration.
- GLiNER v2.5 default entity thresholds changed to `0.60` for small, `0.55` for medium, and `0.75` for large. Set an explicit threshold to preserve previous extraction behavior.

## v0.6.9 (2026-06-19)

Added per-pool pinned-model settings to the pool API and Python SDK, including profile-qualified model IDs. Helm model preloading now respects each worker's pool, hardware profile, and bundle. LoRA adapters also receive compatible adapter names when their model IDs contain characters that PEFT cannot use directly.

## v0.6.8 (2026-06-16)

Self-hosted pools can now enforce a minimum number of warm workers through KEDA. Active rerankers are no longer mistaken for idle models, model unloading is coordinated with in-flight work, and multimodal scoring accounts for media when sizing batches.

## v0.6.7 (2026-06-16)

### Features

- Added a 32K-token serving profile for Qwen3.6-27B on RTX PRO 6000. The base model configuration retains its 4K context window.

### Bug fixes

- Grammar-constrained Qwen3.5-4B requests use a non-speculative profile so structured-output constraints are enforced.
- Long model startups now use consistent readiness timeouts across the gateway, workers, and Helm configuration. Reapplying unchanged model configuration no longer needlessly unloads models.

### Performance improvements

- LightOnOCR processes multiple pages in bounded batches, preserving page order and handling different image sizes.

## v0.6.6 (2026-06-14)

Fixed false configuration mismatches between the configuration service and workers when models inherit profiles or belong to specific pools and bundles. Configuration checks also recover when previously missing bundle metadata becomes available.

## v0.6.5 (2026-06-13)

### Bug fixes

- Image-based reranking now transports Python SDK image inputs correctly and includes document images in the Qwen3-VL reranker's prompt.
- Structured generation correctly resolves JSON Schema references without discarding constraints beside a reference.
- SGLang model loading no longer blocks the server event loop, allowing health checks and other requests to remain responsive during startup.
- Configuration updates correctly account for pool ownership, replace stale snapshots, and stop advertising removed model configurations as ready.

### Breaking changes

- Provisioning responses now use HTTP `503` with retry information instead of HTTP `202`. The Python and TypeScript SDKs understand the updated response; direct API clients should handle the new status when waiting for capacity.

## v0.6.4 (2026-06-11)

Added an AKS Helm overlay with Azure Workload Identity support. Self-hosted clusters also recover worker health subscriptions after a stale NATS connection, and binary request data is preserved when passed from the sidecar to the inference worker.

## v0.6.3 (2026-06-10)

### Highlights

Chat requests can now combine text and images, while the model catalog gains more classification, extraction, and object-detection options.

### Features

- The gateway's OpenAI-compatible chat endpoint now accepts inline image data for vision-capable models while preserving text/image ordering.
- Added ModernBERT-base-zeroshot-v2.0 and BART-large-MNLI classification configurations, GLiNER2-large-v1 extraction, and OWLv2-large-patch14-ensemble object detection.
- Added Azure Blob support for model caching and large request payloads.

### Bug fixes

- Corrected image preprocessing and removed padding from document embeddings for `nvidia/llama-nemoretriever-colembed-3b-v1`.
- Workers are removed promptly from gateway discovery when they shut down, reducing routing to stale workers.

## v0.6.2 (2026-06-08)

Added three dense text-embedding models: `mixedbread-ai/mxbai-embed-large-v1`, `Snowflake/snowflake-arctic-embed-l-v2.0`, and `nomic-ai/modernbert-embed-base`.

Self-hosted startup is more reliable: the configuration service can serve health checks while NATS connects, single-profile bundles can scale from zero for requests without an explicit GPU selection, and CUDA images include the build tool needed for SGLang's first-use kernels. Qwen3-VL embedding models also accept and validate their configured output dimension.

## v0.6.1 (2026-06-07)

Added static, non-expiring queue pools in Helm, with startup validation for invalid pool settings. The default worker queue is again shared as `default`, so SDK requests that specify only a hardware profile route correctly; dedicated pools remain explicitly configurable.

## v0.6.0 (2026-06-07)

Queue routing and autoscaling now distinguish each pool, hardware profile, and model bundle, keeping work assigned to the intended worker group.

### Breaking changes

- Queue subjects now use `sie.work.{pool}.{machine_profile}.{bundle}.{model}`. Upgrade the gateway, sidecar, and Helm chart together; the previous subject format is not supported.
- Helm workers now default to their worker-group name as the queue pool. Set `workers.common.queuePool: "default"` explicitly to retain a shared queue.

## v0.5.0 (2026-06-04)

### Highlights

Self-hosted worker pools can now serve multiple bundles with separate replica limits, making it possible to scale embedding and generation workloads independently on the same machine profile.

### Features

- Added Granite Guardian 3.0 2B for content-safety verdicts, with a configurable verdict threshold. Added SQLCoder-7B-2 for completion-based SQL generation using its native prompt format.
- Added configurable `code`, `sql`, and `guard` model aliases and exposed matching capabilities in model metadata. The default `sql` alias uses Qwen3-4B-Instruct-2507, not SQLCoder.
- Added a separate gateway metrics listener for Prometheus scraping without opening inference endpoints to unauthenticated access.

### Bug fixes

- Florence-2 extraction now honors the supplied instruction.
- Guard-model verdict handling now keeps returned log probabilities consistent and rejects unsupported multi-candidate sampling.
- Helm rejects missing or invalid per-bundle replica limits during rendering.

### Breaking changes

- Move each pool's `bundle`, `minReplicas`, `maxReplicas`, `extraEnv`, and `imageBundle` settings into `workers.pools.<pool>.bundles.<bundle>`. The bundle name becomes the map key; `workers.common.bundle` is removed.
- Worker resource names change from `worker-<pool>` to `worker-<pool>-<bundle>`. Upgrades must explicitly remove obsolete StatefulSets, ScaledObjects, PodDisruptionBudgets, and image-prepull DaemonSets so old resources do not interfere with scaling or node drains.

## v0.4.2 (2026-06-03)

### Highlights

This release introduces the Rust worker sidecar for queued inference and expands document extraction and image-text model support.

### Features

- Added MinerU2.5-Pro-2604-1.2B for document OCR and Marqo fashionSigLIP for image-text embeddings. Docling now also accepts image inputs.
- Chat completions accept `min_tokens` and `chat_template_kwargs`; model profiles can supply default sampling settings.
- Added an FP8 serving profile for Qwen3.6-27B on RTX PRO 6000 and increased Qwen3-0.6B's configured context window to 4,096 tokens.
- Workers reconcile configuration changes after missed updates or reconnects.

### Bug fixes

- Kept generation dispatch separate from embedding, scoring, and extraction queues.
- Fixed `dense_dim` handling in CLIP and PyTorch embedding adapters and CUDA-cache cleanup in visual-document adapters.

### Breaking changes

- Cluster inference is now queue-only. Queue workers require the Rust worker sidecar and NATS JetStream; the Helm chart enables the sidecar by default. Custom deployments must include the sidecar alongside the Python inference worker.

## v0.4.1 (2026-05-28)

Added Qwen3.6-27B model support and updated the Linux GPU dependency stack to CUDA 12.9. Generation requests are now dispatched separately from shared inference queues.

## v0.4.0 (2026-05-27)

### Highlights

SIE now serves text generation alongside embeddings, reranking, and extraction, including streaming responses and structured output.

### Features

- Added Qwen3-0.6B, Qwen3-4B-Instruct-2507, and Qwen3.5-4B generation models through SGLang.
- Added a native generation API, OpenAI-compatible chat and legacy completions endpoints, and an initial gateway Responses API implementation. The Python and TypeScript SDKs expose generation options and streaming.
- Added multi-turn tool calls, multiple response candidates, log probabilities, seeded sampling, and per-request LoRA selection. JSON-schema, regex, and grammar-constrained output are available where supported by the selected model and backend.
- Added a browsable API reference at `/docs` and optional bundled certificate management with self-signed TLS for self-hosted clusters.

### Bug fixes

- Streaming now surfaces backpressure failures instead of silently dropping output chunks, and cancellation prevents duplicate generation attempts.
- Fixed decoding of base64 image inputs and included the system libraries needed by Docling in worker images.
- GPU-aware health checks detect unusable CUDA contexts so unhealthy workers can be taken out of service.

## v0.3.4 (2026-05-14)

### Features

- Python and TypeScript clients now expose `InputTooLongError` for extraction inputs that exceed model limits.
- Helm can use the model-cache bucket's `payloads` prefix for large inference payloads.

### Bug fixes

- Fixed gateway startup during concurrent configuration changes and made shared queue routing the default for workers.
- Relaxed the Python SDK's installation requirement to Python 3.12 or later, and removed unnecessary X11 dependencies from image-processing installations.

## v0.3.3 (2026-05-13)

### Highlights

Added ColQwen3 and Nemotron ColEmbed v2 for visual-document retrieval, with clearer failures for oversized extraction inputs and stalled model loading.

### Bug fixes

- GLiClass now enforces the configured overflow policy and returns `INPUT_TOO_LONG` with HTTP 400 instead of crashing on oversized inputs.
- Model loading now distinguishes stalled downloads from time spent loading downloaded weights and applies separate timeout bounds.
- Worker images now include the spatial-index library required by document extraction dependencies.

## v0.3.2 (2026-05-08)

### Features

- Enabled pairwise scoring for the supported ColBERT model variants.
- Added a Docling OCR profile, gateway OpenAPI discovery, configurable Kubernetes probe timing, cert-manager TLS support, and an optional S3-backed cluster model cache.

### Bug fixes

- Standardized gateway errors and health responses, preserved embedding timing headers, and fixed the scale-from-zero request path.
- Docling now reuses its converter and honors the selected device. PaddleOCR-VL generation now enables its key/value cache.
- Server wheels now include model and bundle configuration files, so installed packages can find their bundled defaults.

## v0.3.1 (2026-04-29)

Added BGE-M3 scoring with dense, sparse, ColBERT, and hybrid modes, plus Marqo e-commerce image-text embeddings. Failed model loads now enter an explicit failed state rather than remaining indefinitely in a loading state.

## v0.3.0 (2026-04-29)

### Highlights

Document extraction and multimodal retrieval expand substantially, while self-hosted clusters move to a Rust gateway with a dedicated configuration service.

### Features

- Added document inputs and structured extraction results across the server and SDKs, including Docling processing for PDF, DOCX, and HTML.
- Added GLM-OCR and PaddleOCR-VL-1.5; Qwen3-VL-Embedding-2B and Qwen3-VL-Reranker-2B; Qwen3-Reranker-0.6B and 4B; and SigLIP 2 image-text embeddings.
- Added GLiNER2, GLiNER-bi and Modern GLiNER-bi, Stablebridge token pruning and highlighting, and a ModernBERT-base embedding configuration.
- Added automatic GPU out-of-memory recovery and idle-model eviction, together with gateway and configuration-service metrics.

### Bug fixes

- Clients retry transient disconnects and capacity-related service errors without treating permanent connection failures as retryable.
- Improved propagation of model configuration changes and reporting of unknown or unroutable models.

### Breaking changes

- Self-hosted Helm configuration moves from `router` to `gateway` settings and adds a separate `config` service. Configuration writes belong to that service, not the inference gateway; review custom values and configuration clients when upgrading. The chart enables NATS JetStream queue routing by default.

## v0.2.0 (2026-04-17)

### Highlights

Added ModernBERT-based embedding models and LightOnOCR, along with startup model preloading and explicit concurrency control in the asynchronous Python client.

### Features

- Added GTE-ModernBERT-base, Snowflake Arctic Embed M v2.0, and IBM Granite English R2 embedding models, including the small variant.
- Added LightOnOCR-2-1B for OCR in the `transformers5` bundle.
- Added `max_concurrency` to `SIEAsyncClient` and Haystack-convention import aliases under `haystack_integrations`.
- Added anonymous usage telemetry, with opt-out through `SIE_TELEMETRY_DISABLED=true` or `DO_NOT_TRACK=1`.

### Bug fixes

- Model-affinity routing can spill requests to other workers instead of becoming stuck, and rejected requests now contribute to autoscaling demand.

### Breaking changes

- Worker startup no longer accepts `--model` to select models. Use `--preload` or `SIE_PRELOAD_MODELS` to load models at startup; otherwise models load on demand.

## v0.1.10 (2026-04-09)

### Highlights

Added LanceDB integrations for Python and TypeScript and a configuration-management API that distributes model changes to workers.

### Features

- Weaviate document enrichment now supports asynchronous processing, chunking, and streaming. LanceDB table enrichment processes batches incrementally without materializing the entire table.
- Added `get_model()` to the Python SDK and exposed queue-routing controls through Helm.

### Bug fixes

- Fixed queued score-response formatting, dead-letter routing, and reconnect handling.
- LlamaIndex embedding now handles `BytesIO` images, and Weaviate classification enrichment validates its configuration.

## v0.1.9 (2026-04-02)

Fixed Helm worker image tags to include the target platform and restored worker pool names to match machine profiles.

## v0.1.8 (2026-04-01)

Fixed duplicated platform suffixes in Helm worker image tags.

## v0.1.7 (2026-04-01)

### Highlights

Added Qdrant and Weaviate integrations and made self-hosted installation more complete through the Helm chart.

### Features

- Qdrant integration supports native sparse vectors; Weaviate integration supports the v4 client.
- ColBERT supports configurable document-length limits and custom prefix tokens.
- Pool creation accepts minimum worker counts and bundle selection. SDK/server version negotiation reports incompatible versions, and the Python SDK waits for capacity by default with a 900-second timeout.
- Helm now manages service accounts and model-access secrets, offers bundled autoscaling and monitoring components, and can pre-pull worker images onto GPU nodes.

### Bug fixes

- Corrected Qwen3 embedding attention behavior and LoRA-layer handling that could affect embedding results.
- Fixed asynchronous-client initialization outside a running event loop and extended pool leases to tolerate rolling upgrades.

### Breaking changes

- AWS and GCP Terraform deployments now separate cluster infrastructure from SIE application installation. Use Helm for the SIE application and review resource ownership before upgrading existing Terraform-managed installations.

## v0.1.6 (2026-03-12)

### Features

- Added Matryoshka embedding truncation for ColBERT.

### Bug fixes

- Fixed loading and configuration for NV-Embed-v2, Stella, BGE-M3, and instruction-based embedding models.
- Fixed ColBERT encoding on non-CUDA devices by selecting the native execution path.
- Aligned synchronous and asynchronous `encode()` and `score()` behavior. Malformed inference inputs now receive validation errors instead of server errors.
- Fixed spot-GPU resolution for autoscaling and increased default CPU-worker memory limits for the expanded bundle.

### Breaking changes

- The standalone `florence2` and `gliner` bundles are removed. Use the `default` bundle, which now includes Florence-2, GLiNER, GLiREL, and GLiClass dependencies.

## v0.1.5 (2026-02-27)

Added GLiNER v2.5 small, medium, and large models, GLiClass large models, and DeBERTa-based NLI classification. The gateway now streams request and response bodies, with corrected response headers for streaming.

## v0.1.4 (2026-02-27)

No user-facing changes.

## v0.1.3 (2026-02-26)

No user-facing changes.

## v0.1.2 (2026-02-26)

No user-facing changes.

## v0.1.1 (2026-02-26)

No user-facing changes.

## v0.1.0 (2026-02-26)

### Features

- Added structured gateway request logs and an `X-SIE-Worker` response header to identify the serving worker.

### Bug fixes

- Removed ColBERT's blanket CUDA-only model-loading restriction.
- GLiClass and NLI classification now populate `classifications` results correctly, and entity extraction handles typed dictionary results consistently.
- Corrected model-name resolution and bundle registration for classification models.

### Breaking changes

- Model configurations no longer accept per-model `dependencies`; adapter dependencies are defined by bundles. The `DEPENDENCY_CONFLICT` error and its HTTP 409 responses are removed. This does not remove HTTP 409 responses for incompatible bundle selections.
