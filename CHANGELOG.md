# Changelog

## [0.8.2](https://github.com/superlinked/sie/compare/v0.8.1...v0.8.2) (2026-09-23)


### Bug Fixes

* **release:** rehearse all artifacts before publication ([#342](https://github.com/superlinked/sie/issues/342)) ([cf2b732](https://github.com/superlinked/sie/commit/cf2b732028703eb6f20cf7db144664f2216eb7d4))

## [0.8.1](https://github.com/superlinked/sie/compare/v0.8.0...v0.8.1) (2026-09-23)


### Features

* **examples:** rebuild the chat example around the recorded multi-turn run ([#338](https://github.com/superlinked/sie/issues/338)) ([25c857e](https://github.com/superlinked/sie/commit/25c857e0ea1a639884ff1f9abbf50d20113327f3))


### Bug Fixes

* **release:** repair audio and CUDA 13 build checks ([#340](https://github.com/superlinked/sie/issues/340)) ([605b3fe](https://github.com/superlinked/sie/commit/605b3fea9e6e8630cf431dae0943fce529540c44))

## [0.8.0](https://github.com/superlinked/sie/compare/v0.7.3...v0.8.0) (2026-09-23)


### ⚠ BREAKING CHANGES

* **server:** `extra_launch_args` may not carry `--nccl-port`, `--host` or `--port`, in any unambiguous abbreviation or `--flag=value` spelling. A profile carrying one fails to load, and the config service refuses the write that would create it, so profiles written straight through the config API need the same edit as the ones shipped in a chart. Replace `--nccl-port` with the `adapter_options.loadtime.nccl_port` option, which applies above `tensor_parallel_size: 1` and is reserved before the flag is passed. Remove `--host` and `--port`: the server passes both for the HTTP listener it talks to.
* **server:** a load-time option that no adapter constructor accepts is refused when the model loads instead of being ignored. `extra_launch_args` may not carry placement flags (including unambiguous abbreviations) and `extra_env` may not set `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES`. The SGLang embedding adapter no longer accepts `pooling_method`. SGLang launch arguments spell the width as `--tensor-parallel-size` for every profile.

### Features

* carry inline video through the queued chat completions path ([#329](https://github.com/superlinked/sie/issues/329)) ([d0633d1](https://github.com/superlinked/sie/commit/d0633d141aa0ce088bed3f447d228c21658e94f5))
* **ci:** add CI and release automation ([4e110b9](https://github.com/superlinked/sie/commit/4e110b9ca86aaf10476ebda3cb789fc9209994b4))
* **config:** add GLM-5.3-Flash across eight accelerators ([#324](https://github.com/superlinked/sie/issues/324)) ([a913d0c](https://github.com/superlinked/sie/commit/a913d0cfd4d5110f685d277d9941c5d081a9ee43))
* **examples:** add detect, image-classify and speech-to-text evaluations ([#313](https://github.com/superlinked/sie/issues/313)) ([8b4f3d4](https://github.com/superlinked/sie/commit/8b4f3d460dec5ceabe1a09872f82996ae8b743fe))
* **examples:** add five recorded task evaluations, with evidence on HuggingFace ([#310](https://github.com/superlinked/sie/issues/310)) ([7d6e11d](https://github.com/superlinked/sie/commit/7d6e11d88452cb9cc72c378914e9625edfdc6032))
* **examples:** add image-search and visual-document-search evaluations ([#314](https://github.com/superlinked/sie/issues/314)) ([c303dfc](https://github.com/superlinked/sie/commit/c303dfc7f7a0c0000bc6338ca20fcf40e6ee2678))
* **examples:** add ocr-two-stage, a recorded two-stage OCR evaluation ([#303](https://github.com/superlinked/sie/issues/303)) ([4202f7d](https://github.com/superlinked/sie/commit/4202f7d96ab84261d8fb88c1a32d0179ea132e34))
* **examples:** add three recorded task evaluations, with evidence on HuggingFace ([#311](https://github.com/superlinked/sie/issues/311)) ([8be1e04](https://github.com/superlinked/sie/commit/8be1e04a78bf2c347d10076b97b1a840b7d23d02))
* **examples:** measure whether a second call earns its place on doc field extraction ([#336](https://github.com/superlinked/sie/issues/336)) ([f4521a9](https://github.com/superlinked/sie/commit/f4521a9077dcd334c138e8372b9088469d39f82b))
* **examples:** re-derive the yes-or-no figure for /structured-output ([#335](https://github.com/superlinked/sie/issues/335)) ([8333afa](https://github.com/superlinked/sie/commit/8333afa28f5bf6bc151b0cf5bbadab06fc1f9713))
* **examples:** score four guardrail models on the same twelve inputs ([#319](https://github.com/superlinked/sie/issues/319)) ([6794e34](https://github.com/superlinked/sie/commit/6794e348ea50d995ad8da7a611b58ec70da92e4e))
* **examples:** score grounded answers in chat, move the incident run to its own example ([#317](https://github.com/superlinked/sie/issues/317)) ([5cc5580](https://github.com/superlinked/sie/commit/5cc5580f110092eeffd67ce5b6bfe8db12311c60))
* **helm:** add a tested high-availability values composition ([#274](https://github.com/superlinked/sie/issues/274)) ([0560f01](https://github.com/superlinked/sie/commit/0560f017155fa4097ec9da0bd456100e4f33aca5))
* **helm:** make the worker /dev/shm size configurable per pool ([#302](https://github.com/superlinked/sie/issues/302)) ([e203682](https://github.com/superlinked/sie/commit/e2036826329065ee35cd84fccba696656a379cee))
* **models:** add google/translategemma-4b-it ([40ae372](https://github.com/superlinked/sie/commit/40ae37238b6a09b0abfeeaa6fbf645d292edcced))
* **models:** add google/translategemma-4b-it ([979961b](https://github.com/superlinked/sie/commit/979961b840604ef04ccc202bbd081e3534fb4552))
* **models:** add Qwen3-VL-8B-Instruct with text, image, and video input ([#323](https://github.com/superlinked/sie/issues/323)) ([92bd4bb](https://github.com/superlinked/sie/commit/92bd4bba8c93a026bd243780daf40e74e273eb13))
* **server:** accept inline video in local chat completions ([#321](https://github.com/superlinked/sie/issues/321)) ([0fbe99b](https://github.com/superlinked/sie/commit/0fbe99b446fe393e79088f9b3c927a40e52cf8b8))
* **server:** move the CUDA 13 SGLang bundle to 0.5.20 for glm5_next ([#320](https://github.com/superlinked/sie/issues/320)) ([ac2358c](https://github.com/superlinked/sie/commit/ac2358c7ed99e7f21943e089da90e3c35fbe7c1f))
* **server:** parse and force GLM tool calls on the queued route ([#328](https://github.com/superlinked/sie/issues/328)) ([84b1a7f](https://github.com/superlinked/sie/commit/84b1a7f8c6eb932af02516deae55d3b87aea63b4))
* **server:** serve one model across several GPUs with tensor parallelism ([#282](https://github.com/superlinked/sie/issues/282)) ([a03f5db](https://github.com/superlinked/sie/commit/a03f5dbf878db47b6a8418c6c4b9f530b486ff88))


### Bug Fixes

* **api:** constrain native stream execution evidence pairs ([7cfa50e](https://github.com/superlinked/sie/commit/7cfa50e73068354808f5b6430c248c494ec2216e))
* **api:** restrict stream evidence to successful terminal chunks ([f256d4e](https://github.com/superlinked/sie/commit/f256d4e496f066f5cb510e18e3f9ad61aa337387))
* bound diagnostic decoding and preserve grammar refusal metadata ([b70cca0](https://github.com/superlinked/sie/commit/b70cca055565590f88b07c7a65914bb468006bb4))
* **ci:** serialize mise bootstrap to protect Node GPG state ([538daff](https://github.com/superlinked/sie/commit/538daff41ca20ee27a69f26bda73dd190bcff35a))
* **ci:** serialize mise bootstrap to protect Node GPG state ([996b07f](https://github.com/superlinked/sie/commit/996b07feedc2a55cebd6962895e5d7a0d7afc0b8))
* **ci:** serialize mise bootstrap to protect Node GPG state ([#287](https://github.com/superlinked/sie/issues/287)) ([538daff](https://github.com/superlinked/sie/commit/538daff41ca20ee27a69f26bda73dd190bcff35a))
* **config:** add Qwen3.8 grammar fallback ([#258](https://github.com/superlinked/sie/issues/258)) ([1c7bbe8](https://github.com/superlinked/sie/commit/1c7bbe806c1f1fba9929f96e4b4a96b8c462d2fb))
* **config:** keep GLM-5.3-Flash reasoning private on every route ([#326](https://github.com/superlinked/sie/issues/326)) ([1ffb19c](https://github.com/superlinked/sie/commit/1ffb19c4a8b1b42c30dd344a6c85abe036186597))
* document observed images in server generation usage ([81aecec](https://github.com/superlinked/sie/commit/81aececaca31a1518102f209eeaefa878b81c9c2))
* **examples:** record Granite Guardian through chat completions on guardrails ([#316](https://github.com/superlinked/sie/issues/316)) ([86f4e1e](https://github.com/superlinked/sie/commit/86f4e1ed56b317614e040fd88548341a852a28c9))
* **gateway:** keep the served surface when an authoritative export shrinks it ([#270](https://github.com/superlinked/sie/issues/270)) ([c65aee5](https://github.com/superlinked/sie/commit/c65aee57224d93c036490f601092bf2c8e165ee9))
* **gateway:** pass the invalid_guard_verdict worker code through ([#297](https://github.com/superlinked/sie/issues/297)) ([7a8b09b](https://github.com/superlinked/sie/commit/7a8b09b6db1c206e21bb2b85b7af415e5f644802))
* **gateway:** refuse requests when the auth configuration is an error ([#269](https://github.com/superlinked/sie/issues/269)) ([8c84419](https://github.com/superlinked/sie/commit/8c84419ac4c4882b081aeda3413cd66b86dab191))
* **generate:** preserve image usage and complete execution evidence ([d266205](https://github.com/superlinked/sie/commit/d2662052bd53a017b300a64a246f4989f4f8fe30))
* **generate:** preserve image usage and complete execution evidence ([c9e9b7e](https://github.com/superlinked/sie/commit/c9e9b7e117ab8de4128a6ab10e7ce2be20734352))
* **generate:** preserve image usage and complete execution evidence ([#286](https://github.com/superlinked/sie/issues/286)) ([d266205](https://github.com/superlinked/sie/commit/d2662052bd53a017b300a64a246f4989f4f8fe30))
* **generation:** require explicit streamed candidate indexes ([c0aca44](https://github.com/superlinked/sie/commit/c0aca4438f1fe1359a8a0ef704ef592e49dd193f))
* **helm:** make KEDA hook Job resources configurable ([#268](https://github.com/superlinked/sie/issues/268)) ([5e726f6](https://github.com/superlinked/sie/commit/5e726f64be7150f47fb08397d4e444ed0450030c))
* identify unsupported Outlines JSON Schema type values ([729cadc](https://github.com/superlinked/sie/commit/729cadcfb5bd7f75c7af28899efe93ca319d794c))
* identify unsupported Outlines JSON Schema type values ([0c0408b](https://github.com/superlinked/sie/commit/0c0408b1de443f7309c71660bdd3cccd29ca37d0))
* identify unsupported Outlines JSON Schema type values ([#289](https://github.com/superlinked/sie/issues/289)) ([729cadc](https://github.com/superlinked/sie/commit/729cadcfb5bd7f75c7af28899efe93ca319d794c))
* isolate grammar followers and type malformed backend errors ([4c5e05f](https://github.com/superlinked/sie/commit/4c5e05fe1ce5cd4fafe99f4e79df6c83b6b7a0b8))
* **mcp:** parse the Host header before trusting it for the OAuth origin ([27ae3a1](https://github.com/superlinked/sie/commit/27ae3a18033a09a1c9c3f18de6bcef1d51f37db4))
* **mcp:** parse the Host header before trusting it for the OAuth origin ([604449a](https://github.com/superlinked/sie/commit/604449a4a435c0bc4fe2bb2957a1758f7d1e27f8))
* **mcp:** stop deriving the OAuth origin from X-Forwarded headers ([#292](https://github.com/superlinked/sie/issues/292)) ([f5c4451](https://github.com/superlinked/sie/commit/f5c4451f1cfc9a2d98dffc5cb94b08ad2d38efa0))
* **mcp:** validate bracketed IPv6 before advertising origins ([a66f6a0](https://github.com/superlinked/sie/commit/a66f6a096153e7dd8c41197e74bc8d9238c1823c))
* **mcp:** validate Host before parsing OAuth request origins ([d9507f5](https://github.com/superlinked/sie/commit/d9507f5a896ccdc2d0c8ffaa89e7b39d5c95b523))
* **models:** bound translategemma prompts to the documented 2K input context ([ac810ad](https://github.com/superlinked/sie/commit/ac810ad01fb2e9744b47a29c7537d28f4e4e520a))
* **models:** pin MADLAD to greedy sampling by default ([b0f0eeb](https://github.com/superlinked/sie/commit/b0f0eebf4370d6f151eba98cc1704750617996a5))
* **models:** pin MADLAD to greedy sampling by default ([a968874](https://github.com/superlinked/sie/commit/a968874ce724c4b9dc5c8a61313188655705f4e9))
* **models:** pin the triton attention backend for translategemma ([2dfe7b2](https://github.com/superlinked/sie/commit/2dfe7b25778f67ac37083a2e1244f627fce3e6c4))
* omit logprobs for rewritten guard verdicts ([ba8c3b1](https://github.com/superlinked/sie/commit/ba8c3b11e98711b974bef5f07d8ba15a6e5bcbbd))
* preserve generation progress and reject invalid guard verdicts ([424f7ae](https://github.com/superlinked/sie/commit/424f7ae15dc1fe4e7e66a171160322f38264dedd))
* preserve generation progress and reject invalid guard verdicts ([398c673](https://github.com/superlinked/sie/commit/398c673885bd94e0c5f047240ae3289a04f72754))
* preserve generation progress and reject invalid guard verdicts ([#285](https://github.com/superlinked/sie/issues/285)) ([424f7ae](https://github.com/superlinked/sie/commit/424f7ae15dc1fe4e7e66a171160322f38264dedd))
* preserve grammar-safe generation defaults ([#261](https://github.com/superlinked/sie/issues/261)) ([3b16ffb](https://github.com/superlinked/sie/commit/3b16ffbca4ee5fa816f9636b9b998a3e740830ff))
* preserve TensorRT-LLM completion text and stop alignment ([#267](https://github.com/superlinked/sie/issues/267)) ([d345bc0](https://github.com/superlinked/sie/commit/d345bc0a0663e587687ec535c75d3aa7004f844b))
* **release:** format generated SDK package metadata ([#339](https://github.com/superlinked/sie/issues/339)) ([36bd49e](https://github.com/superlinked/sie/commit/36bd49e15f6817bbb26d91f988fcba8cab3a49c4))
* **sdk:** declare terminal streaming execution evidence ([0261183](https://github.com/superlinked/sie/commit/026118304fdd167a224678ac93bf0e7f70a124e7))
* **sdk:** declare terminal streaming execution evidence ([#283](https://github.com/superlinked/sie/issues/283)) ([0261183](https://github.com/superlinked/sie/commit/026118304fdd167a224678ac93bf0e7f70a124e7))
* **sdk:** document terminal streaming execution evidence ([f3a05ba](https://github.com/superlinked/sie/commit/f3a05ba287f4d5a96afe4d5f72be3abceacdd352))
* **server:** bound native encode, score and extract request bodies ([#271](https://github.com/superlinked/sie/issues/271)) ([8479f1c](https://github.com/superlinked/sie/commit/8479f1c8beff090da711456dcddbe03abb899965))
* **server:** evict the least recently used group when two blocks cost the same ([#294](https://github.com/superlinked/sie/issues/294)) ([120059f](https://github.com/superlinked/sie/commit/120059fd632d119c6d8cb91bcf96068ef452515e))
* **server:** fetch model weights outside the registry load lock ([#272](https://github.com/superlinked/sie/issues/272)) ([394df63](https://github.com/superlinked/sie/commit/394df634824eb7cde4fafa38eee70cc46258eed7))
* **server:** force complete GLM argument pairs and reject oversized GLM calls ([#330](https://github.com/superlinked/sie/issues/330)) ([079647d](https://github.com/superlinked/sie/commit/079647d52261063aa7fa9066b1393a14d0b2976c))
* **server:** keep inline media payloads out of SGLang child logs ([#325](https://github.com/superlinked/sie/issues/325)) ([d8573f2](https://github.com/superlinked/sie/commit/d8573f24fc357b3e57df94c06ac7a12f2c6784b0))
* **server:** map GroundingDINO detections back onto the caller's labels ([#277](https://github.com/superlinked/sie/issues/277)) ([958414c](https://github.com/superlinked/sie/commit/958414c5b376bb370da608b6bbb957b20556457e))
* **server:** narrow validated terminal result type ([56efe5f](https://github.com/superlinked/sie/commit/56efe5fbe2abf18f30f3d19d9feb14cbe04b6ee4))
* **server:** pin cuda-tile for TensorRT-LLM ([#254](https://github.com/superlinked/sie/issues/254)) ([6484fab](https://github.com/superlinked/sie/commit/6484fabdcf5365ff9e4d0381e95a301a524e7ab4))
* **server:** read Qwen2.5-style tool calls as Hermes JSON on the queued route ([#332](https://github.com/superlinked/sie/issues/332)) ([d8494f3](https://github.com/superlinked/sie/commit/d8494f3ff9642ebdd0ccf06c1d8c03fc23badb97))
* **server:** refuse listener flags in extra_launch_args ([#304](https://github.com/superlinked/sie/issues/304)) ([7a68262](https://github.com/superlinked/sie/commit/7a68262a0f05e6cc42790b77fc779e58ec2696e0))
* **server:** reject failed SGLang generation terminals ([829f80c](https://github.com/superlinked/sie/commit/829f80ca4a5dceea1defc9c9891b6e1111bdf9e7))
* **server:** reject failed SGLang generation terminals ([3bc140d](https://github.com/superlinked/sie/commit/3bc140d29214c8410267771f24b988da59864ebf))
* **server:** reject failed SGLang generation terminals ([#284](https://github.com/superlinked/sie/issues/284)) ([829f80c](https://github.com/superlinked/sie/commit/829f80ca4a5dceea1defc9c9891b6e1111bdf9e7))
* **server:** reject malformed backend finish metadata ([2be0155](https://github.com/superlinked/sie/commit/2be0155f8ef1ff0eda201739e87fca1a0ed5d6e4))
* **server:** report a valueless --mm-process-config instead of raising ([#333](https://github.com/superlinked/sie/issues/333)) ([2f7ad9d](https://github.com/superlinked/sie/commit/2f7ad9db0dff6781210bb28bf3567eff2436a9f1))
* **server:** require a usable startup budget for SGLang profiles ([#301](https://github.com/superlinked/sie/issues/301)) ([038a8d9](https://github.com/superlinked/sie/commit/038a8d9116a0bf6f776809c9d8355b3741f5d4dc))
* **server:** require complete generation candidates ([afaab57](https://github.com/superlinked/sie/commit/afaab574e6d035f399fce98029281e9c2f2ef507))
* **server:** ship FFmpeg shared libraries in the SGLang runtime image ([#322](https://github.com/superlinked/sie/issues/322)) ([9d98037](https://github.com/superlinked/sie/commit/9d9803790a7c582da0d7232be326177a78508d92))
* **server:** tell SGLang when a native generate request starts inside reasoning ([#327](https://github.com/superlinked/sie/issues/327)) ([26702c7](https://github.com/superlinked/sie/commit/26702c7a9e5b78326692b2c811c36f91ad8c9db1))
* **server:** validate video pixel and frame counts before the budget arithmetic ([#337](https://github.com/superlinked/sie/issues/337)) ([703841a](https://github.com/superlinked/sie/commit/703841a3871cbd0838b6c3be93886e8cc603241a))
* **tasks:** stop expanding possibly-empty arrays bare under set -u ([#278](https://github.com/superlinked/sie/issues/278)) ([e0084c7](https://github.com/superlinked/sie/commit/e0084c78c3cd67b044c4968fc2d2212967a74bb3))
* **toolchain:** upgrade Rust to 1.98.1 and patch rustls ([#276](https://github.com/superlinked/sie/issues/276)) ([a1f6fab](https://github.com/superlinked/sie/commit/a1f6fab356cbff88609a7d999c8e3b3fd02e2365))
* **tooling:** use canonical Rust LLVM component ([#256](https://github.com/superlinked/sie/issues/256)) ([2183bc1](https://github.com/superlinked/sie/commit/2183bc1c3e7b034a8219e8446f99954f601eb87e))
* **ts-sdk:** decode base64 data URLs with media-type parameters ([#247](https://github.com/superlinked/sie/issues/247)) ([21289a1](https://github.com/superlinked/sie/commit/21289a1798a54fcce1b556bcd01ec0b07ff025c2))
* validate complete first guard verdict evidence ([514b07f](https://github.com/superlinked/sie/commit/514b07fffc4b5f84985d5861cb12fc1de0564fd5))
* validate unsupported template placeholders ([#249](https://github.com/superlinked/sie/issues/249)) ([0508e17](https://github.com/superlinked/sie/commit/0508e17aa323c7773be0d9d58c85038d7d9b043e))

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
