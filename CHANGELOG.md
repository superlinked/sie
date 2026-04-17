# Changelog

## v0.2.0

- chore(main): release 0.2.0
- docs: add telemetry documentation and clarify deployment guidelines
- refactor(observability): update tmp_path type hints from str to Path
- test(observability): fix telemetry test type hints
- fix(haystack): rename namespace alias to sie
- feat(observability): add anonymous usage telemetry
- feat(haystack): add haystack_integrations namespace aliases
- feat(deps): migrate from pynvml to nvidia-ml-py package
- feat(benchmarks): add MTEB NFCorpus evaluation results for ModernBERT-based embedders
- fix: install uv via curl instead of COPY --from ghcr.io
- feat(sdk): add max_concurrency param to SIEAsyncClient to prevent connection pool exhaustion
- fix(build): downgrade dockerfile syntax version to 1 for broader compatibility
- refactor(deployment): add model preloading configuration to Terraform and CLI validation
- feat(adapters): add ModernBERT flash dense embedding support with fallback mechanism
- fix(router): harden affinity spill with bounds check, clamp, and debug log
- feat(workers): implement model preloading at startup to reduce first-request latency
- fix(router): add overflow spill to break model affinity deadlock
- fix(router): make rejected requests visible to KEDA scaling metrics
- fix(adapters): remove redundant tokenizer validation and unused template parameter
- refactor(adapters): simplify template application logic and add chat token caching
- fix: address PR review — panel title, namespace variable
- fix(dashboard): queue routing dashboard accuracy and usability
- fix(release): add LanceDB integrations to release-please config
- refactor(adapters): consolidate adapter infrastructure with declarative specs
- perf(server): use sdpa attention for LightOnOCR (6x faster, no OOM)
- fix(server): harden LightOnOCR adapter and rename next bundle
- fix(server): add collate() and fix prepare() signature on LightOnOCRPreprocessor
- refactor(adapters): consolidate adapter infrastructure with declarative specs
- refactor(adapters): improve flash attention device handling and resource cleanup
- refactor(adapters): extract common flash adapter logic into reusable base and utilities
- fix(server): trim next bundle to only verified transformers 5.x adapters
- fix: restore accidentally deleted test file
- fix(server): remove accidental data symlink, default to bfloat16
- Delete packages/sie_server/tests/adapters/test_lighton_ocr.py
- feat(server): add next bundle and fix LightOnOCR for transformers 5.x
- feat(server): add lightonai/LightOnOCR-2-1B OCR adapter

## v0.1.10

- chore(main): release 0.1.10
- feat: add async, chunking, and streaming to Weaviate document enricher
- perf(lancedb): decouple scanner and SIE batch sizes in enrich_table
- fix(helm): use generic updater for both Chart.yaml version fields
- fix: handle BytesIO images in LlamaIndex and validate Weaviate classify config
- perf(lancedb): stream enrich_table batch-by-batch instead of full materialization
- perf(lancedb): use Lance scanner for column projection in enrich_table
- fix(observability): queue routing dashboard PromQL for NATS wait
- feat(observability): queue routing dashboard, NATS prom exporter, router image tag
- fix: update adapter tests and address code review feedback
- fix(integrations): address CodeRabbit review findings for LanceDB PR
- fix(helm): use generic release-please updater for appVersion
- test(router): add queue-mode score response key regression tests
- fix(router): use "scores" key in queue-mode score responses
- fix(helm): restore Chart.yaml deps from main, keep appVersion v-prefix
- feat(terraform): add evaluation cluster setup for AWS with multi-GPU support and updated configurations
- make sure extraction uses not just the entity response type but the other response types also, including integrations
- remove the code that made it seem as if we were passing topk param to server from typescript - this optimization doesn't make that much sense as the request is anyway much larger than response
- PY<>TS parity in terms of integration depth  for langchain and llamaindex
- multivector support update in integrations
- deduplicate integration logic into ts/py sdk package that the integrations already depend on
- feat(sdk): add get_model() and configure LanceDB release workflows
- feat(integrations): add LanceDB integration (Python + TypeScript)
- haystack and llamaindex get multi-modal integration
- weaviate integration improvements
- test(queue): add NATS server fixture and improve integration test reliability
- feat(dlq,pull-loop): improve DLQ routing and score response handling
- fix(config,queue,nats): correct cluster routing condition, stream max_age units, and reconnect state ordering
- fix(queue-routing): score response format and DLQ fallback routing key
- fix(queue-routing): configurable NATS fetch budget, Helm-wired queue params
- perf(router): bypass FastAPI for hot proxy paths via raw ASGI middleware
- perf(sdk+router): lazy msgpack_numpy.patch and pure ASGI middleware
- perf(router): remove msgpack_numpy global patch and BaseHTTPMiddleware
- perf(router): replace stdlib json with orjson for 3-10x faster serialization
- perf(router): reduce thread pool pressure by inlining small deserialization
- refactor(router): remove lock-free synchronization and batch work item serialization
- fix(helm): add recreate strategy for router deployment when nats config restore is enabled
- test(cluster): fix kind cluster tests for config API and port-forward restart support
- fix(queue-routing): resolve 5 bugs blocking NATS pull-based worker routing
- feat(worker): improve batch cost bounds and fix template logic
- fix: address review findings — error handling, validation, public APIs, tests
- fix(server): update tests to use max_seq_length instead of max_length
- test(adaptive-batching): replace time.sleep with mocked monotonic for deterministic integral accumulation test
- feat(adaptive-batching): add auto-calibration and PI controller with per-model profile overrides
- refactor(batcher): implement proportional coalesce scaling and adaptive batch wait precision
- feat: add pull-based worker queue routing and adaptive batching
- feat(config-api): implement idempotency and auth separation with profile conflict detection
- docs(config): clarify write auth token precedence and filter updates
- fix(server): align cross-encoder max_seq_length parameter with loader
- chore: review items
- chore: review items
- chore: review items
- chore: fix review items
- chore: remove git sync
- feat: implement Config Management API with NATS-based distribution and review fixes Add runtime model config additions via REST API (POST /v1/configs/models), with NATS pub/sub distribution to workers and S3/GCS/local persistence using epoch-based CAS. Workers track config convergence via SHA-256 bundle_config_hash reported through WebSocket status. Key components: - config_api.py: REST endpoints (list/get/add models, resolve, bundles) - config_store.py: Epoch CAS persistence (local/S3/GCS backends) - nats_manager.py: Router-side NATS publish + cross-router sync - nats_subscriber.py: Worker-side NATS subscribe + config apply - Helm/Terraform: NATS sub-chart, config store env vars, restore flow Also addresses 25 code review findings: M2/M12 medium fixes, L1-L16 low-severity fixes, T3-T10 test coverage gaps, D1-D7 doc/feature discrepancies, and H13 S3 conditional writes documentation.

## v0.1.9

- chore(main): release 0.1.9
- fix(helm): revert pool names to machine profile names
- fix: increase docker smoke test timeouts and add retry
- fix(helm): include $platform in worker image tag format

## v0.1.8

- chore(main): release 0.1.8
- fix(helm): correct image.tag comment to reflect actual format
- fix(helm): remove duplicate platform suffix from worker image tag
- fix: add sie-qdrant and sie-weaviate to release-please config
- fix: update README example to use new pool naming convention
- fix: correct worker image tag format to {version}-{platform}-{bundle}

## v0.1.7

- chore(main): release 0.1.7
- fix(readme): correct helm chart path
- fix(chart): update home URL and Helm install command in README
- fix: pool error types, add pool/progress test coverage
- fix(router): use effective_pool instead of pool_name for default pool GPU extraction
- fix: address PR #478 review feedback
- perf(florence2): switch OCR configs from beam search to greedy decoding
- fix(sdk): defer aiohttp session creation to fix "no running event loop" in SIEAsyncClient
- sdk: switch SIEAsyncClient from httpx to aiohttp
- feat(helm): default router to image-embedded model configs
- feat(helm): enable image pre-pull DaemonSet by default

