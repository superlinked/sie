# text2vec-sie: Weaviate Server-Side Module (Spec)

> **Status:** Spec/placeholder. This module needs to be contributed to the
> [Weaviate repo](https://github.com/weaviate/weaviate) as part of the
> partnership. The code below is a reference implementation showing what
> the Go module would look like.

## What this would enable

Once merged into Weaviate, users could configure SIE as a native vectorizer:

```python
import weaviate.classes as wvc

collection = client.collections.create(
    "Articles",
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_sie(
        api_endpoint="http://sie-server:8080",
        model="BAAI/bge-m3",
    ),
)

# Weaviate handles embedding automatically — no manual vectors needed
collection.data.insert(properties={"text": "hello world"})
results = collection.query.near_text("search query", limit=5)
```

This would list SIE under:
- **Locally hosted models**: https://docs.weaviate.io/weaviate/model-providers#locally-hosted
- **API-based** (later, with managed offering): https://docs.weaviate.io/weaviate/model-providers#api-based

## Module structure

Based on `text2vec-ollama` as reference. The SIE module would be nearly
identical since SIE's HTTP API is similar to Ollama's:

```
weaviate/modules/text2vec-sie/
├── module.go              # Module lifecycle, implements Vectorizer interface
├── config.go              # ClassConfigDefaults, ValidateClass
├── nearText.go            # GraphQL nearText search integration
├── clients/
│   ├── sie.go             # HTTP client calling SIE's /v1/encode/{model}
│   ├── sie_test.go        # Client unit tests
│   └── meta.go            # Module metadata
└── ent/
    ├── class_settings.go  # Configuration: apiEndpoint, model
    └── class_settings_test.go
```

## Key differences from text2vec-ollama

| Aspect | Ollama | SIE |
|--------|--------|-----|
| Endpoint | `POST /api/embed` | `POST /v1/encode/{model}` |
| Request body | `{"model": "...", "input": [...]}` | `{"items": [{"text": ...}], "params": {"output_types": [...]}}` |
| Response | `{"embeddings": [[...]]}` | `{"model": "...", "items": [{"dense": {"dims": N, "dtype": "...", "values": [...]}}]}` |
| Auth | None | Optional API key header |
| Batch | Single request, all texts | Single request, all items |
| Model in URL | No (in body) | Yes (path parameter) |
| Content type | JSON | msgpack (default) or JSON (via Accept header) |

## SIE HTTP client (reference)

The Go client would call SIE's encode endpoint:

```go
// Request (model slash must be URL-encoded in the path segment)
POST /v1/encode/BAAI%2Fbge-m3
Content-Type: application/json

{
    "items": [
        {"text": "first document"},
        {"text": "second document"}
    ],
    "params": {
        "output_types": ["dense"]
    }
}

// Response
{
    "model": "BAAI/bge-m3",
    "items": [
        {
            "dense": {
                "dims": 1024,
                "dtype": "float32",
                "values": [0.123, -0.456, ...]
            }
        },
        {
            "dense": {
                "dims": 1024,
                "dtype": "float32",
                "values": [0.789, -0.012, ...]
            }
        }
    ],
    "timing": {
        "queue_ms": 0.5,
        "inference_ms": 12.3
    }
}
```

Note: `output_types` is nested under `params`, not at the top level.
The response wraps items in an object with `model` and `timing` metadata.
Each dense vector is a structured object with `dims`, `dtype`, and `values`
(not a bare float array). The Go client needs to extract `values` from each
item's `dense` object.

## Configuration defaults

```go
const (
    DefaultApiEndpoint = "http://localhost:8080"
    DefaultModel       = "BAAI/bge-m3"
    Name               = "text2vec-sie"
)
```

## Next steps

1. **Partnership discussion**: Propose `text2vec-sie` to the Weaviate team
2. **Go implementation**: Write the module following `text2vec-ollama` as template
3. **PR to weaviate/weaviate**: Submit for review
4. **Python client support**: Weaviate team adds `Configure.Vectorizer.text2vec_sie()` to the Python client
5. **Docs**: Add SIE to https://docs.weaviate.io/weaviate/model-providers
