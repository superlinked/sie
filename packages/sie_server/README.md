# SIE Server

GPU inference server for embeddings, reranking, and entity extraction.

## Features

- Multi-model serving with LRU eviction
- Token-based dynamic batching
- Hot reload model configs without restart
- Unified API: `encode()`, `score()`, `extract()`
- Prometheus metrics and OpenTelemetry tracing

## Installation

```bash
pip install sie-server
```

## Quick Start

```bash
sie-server serve --port 8080 --device cuda:0
```

## API

See the [API documentation](https://sie.dev/docs) for details.

## License

Apache 2.0
