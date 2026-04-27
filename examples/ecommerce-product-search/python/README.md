# E-Commerce Product Search (Python)

Python implementation of the gallery example using `sie-sdk`, FastAPI, and NumPy.

Pipeline:

1. `extract()` pulls structured attributes from product descriptions
2. `encode()` embeds the catalog into dense vectors
3. `search.py` retrieves by cosine similarity and reranks with `score()`
4. `server.py` exposes the shared UI and REST endpoints

## Files

- `ingest.py` builds `../data/embeddings.npy` and `../data/metadata.json`
- `search.py` runs CLI demo searches against the saved index
- `server.py` serves `/`, `/api/search`, `/api/categories`, and `/api/stats`
- `requirements.txt` contains the Python dependencies for this path and the shared dataset fetch step

## Requirements

- Python 3.12 (required by `sie-sdk`)
- A reachable SIE server. Local Docker is the default:
  ```bash
  docker run -p 8080:8080 ghcr.io/superlinked/sie-server:latest-cpu-default
  ```
  Or a managed SIE cluster via `SIE_CLUSTER_URL` and `SIE_API_KEY` env vars.

## Configuration

This implementation reads `../config.yaml`. The defaults target a local SIE server at `http://localhost:8080`. Tune only when you need to.

See the [top-level README](../README.md#customize) for the full config.

To point at a managed cluster instead:

```bash
export SIE_CLUSTER_URL="https://your-cluster-url"
export SIE_API_KEY="your-api-key"
```

## Run It

From `examples/ecommerce-product-search/`:

```bash
pip install -r python/requirements.txt
python data/fetch_dataset.py
cd python
python ingest.py
python search.py
uvicorn server:app --host 0.0.0.0 --port 8888
```

Open `http://localhost:8888`.

This also serves the shared browser UI, so you can type a query, choose a category, and explore results interactively without using the API directly.

Generated files:

- `../data/products.json`
- `../data/embeddings.npy`
- `../data/metadata.json`

## Exact Verification

After the server starts:

```bash
curl http://localhost:8888/api/stats
curl "http://localhost:8888/api/search?q=wireless+bluetooth+headphones&category=All+Electronics"
```

Expected response shapes:

```json
{
  "total_products": 3000,
  "embedding_dims": 1024,
  "categories": 7,
  "models": {
    "embedding": "NovaSearch/stella_en_400M_v5",
    "reranker": "BAAI/bge-reranker-v2-m3",
    "extractor": "urchade/gliner_multi-v2.1"
  }
}
```

```json
{
  "query": "wireless bluetooth headphones",
  "filters": {
    "category": "All Electronics"
  },
  "results": [
    {
      "id": "<product-id>",
      "title": "<product-title>",
      "description": "<truncated-product-description>",
      "category": "All Electronics",
      "price": "<number-or-null>",
      "rating": "<number-or-null>",
      "image": "<image-url-or-null>",
      "features": ["<feature-text>", "<feature-text>"],
      "attributes": {
        "brand": "<extracted-brand>"
      },
      "scores": {
        "vector": "<similarity-score>",
        "rerank": "<rerank-score>"
      }
    }
  ]
}
```

## API

### `GET /api/search`

Parameters:

- `q` required query string
- `category` optional category filter
- `brand` optional extracted brand filter

Response fields:

- `query`
- `filters`
- `results[].id`
- `results[].title`
- `results[].description`
- `results[].category`
- `results[].price`
- `results[].rating`
- `results[].image`
- `results[].features`
- `results[].attributes`
- `results[].scores.vector`
- `results[].scores.rerank`

### `GET /api/categories`

Returns category counts as `[name, count]` tuples.

Example:

```json
[
  ["AMAZON FASHION", 613],
  ["Amazon Home", 549]
]
```

### `GET /api/stats`

Returns:

```json
{
  "total_products": 3000,
  "embedding_dims": 1024,
  "categories": 7,
  "models": {
    "embedding": "NovaSearch/stella_en_400M_v5",
    "reranker": "BAAI/bge-reranker-v2-m3",
    "extractor": "urchade/gliner_multi-v2.1"
  }
}
```

## Troubleshooting

### Missing dataset files

From `examples/ecommerce-product-search/`, run:

```bash
python data/fetch_dataset.py
```

### Search returns no results

- Verify `../data/metadata.json` and `../data/embeddings.npy` exist
- Check that the query is broad enough to retrieve candidates
- Confirm the category filter matches the dataset values

### Cannot connect to the cluster

- Check `cluster.url` and `cluster.api_key` in `../config.yaml`
- Or set `SIE_CLUSTER_URL` and `SIE_API_KEY`
- Verify the cluster can serve the configured models
