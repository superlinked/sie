# Changelog

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

