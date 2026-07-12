# AI Patronus browser memory

AI Patronus is a Chrome extension that lets a small browser companion remember
pages on request and recall them semantically through SIE embeddings.

This example focuses on a privacy-preserving memory loop:

1. The user asks Patronus to `remember this page`.
2. The extension sends the current page title, URL, and readable text to a local
   memory bridge.
3. The bridge calls SIE's OpenAI-compatible `/v1/embeddings` endpoint and stores
   the vector locally.
4. The user asks `what did I save about browser memory?` and Patronus returns the
   closest saved pages.

No browser history is uploaded automatically. The save action is manual, and the
default setup runs against a local SIE server with local JSON storage.

## SIE primitives

| Flow | Endpoint | Model |
|---|---|---|
| Page memory | `/v1/embeddings` | `sentence-transformers/all-MiniLM-L6-v2` |
| Semantic recall | `/v1/embeddings` + local cosine similarity | `sentence-transformers/all-MiniLM-L6-v2` |

The Chrome extension also contains optional hooks for chat, voice, and web
research providers from the original hackathon prototype, but the SIE memory
flow works without those keys.

## Run it locally

You need Docker, Python 3.12, and Chrome.

Start SIE:

```bash
docker run -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface ghcr.io/superlinked/sie-server:latest-cpu-default
```

Start the memory bridge:

```bash
cd examples/ai-patronus-browser-memory/memory_server
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload --port 8800
```

Load the extension:

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Choose **Load unpacked**.
4. Select `examples/ai-patronus-browser-memory/extension`.
5. Open the Patronus popup and set `Superlinked URL` to `http://localhost:8800`.

Try it on any normal website:

```text
remember this page
what did I save about browser memory?
```

You can also seed sample memories after the memory bridge is running:

```bash
cd examples/ai-patronus-browser-memory/memory_server
python seed_sample.py
```

Then ask:

```text
what did I save about privacy?
```

## Configuration

The memory bridge reads `memory_server/.env`:

| Variable | Default | Purpose |
|---|---|---|
| `SIE_URL` | `http://localhost:8080` | SIE endpoint |
| `SIE_API_KEY` | empty | Optional bearer token for auth-enabled SIE clusters |
| `SIE_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `PATRONUS_MEMORY_PATH` | `data/memory.json` | Local memory store |
| `PATRONUS_MAX_TEXT_CHARS` | `12000` | Per-page text cap |

The extension popup can store optional provider keys in Chrome local storage.
For a hosted/auth-enabled SIE memory endpoint, add that endpoint's origin to
`extension/manifest.json` `host_permissions` before loading the extension.

## API

The local bridge exposes:

### `POST /ingest`

```json
{
  "title": "Example page",
  "url": "https://example.com",
  "text": "Page text to remember"
}
```

### `POST /query`

```json
{
  "query": "browser memory",
  "limit": 5
}
```

Response:

```json
{
  "results": [
    {
      "title": "Example page",
      "url": "https://example.com",
      "text": "Short excerpt...",
      "score": 0.82
    }
  ]
}
```

## Project layout

```text
ai-patronus-browser-memory/
├── extension/          # Manifest V3 Chrome extension
└── memory_server/     # FastAPI bridge from extension to SIE embeddings
```

## Privacy notes

- The extension sends page text only when the user explicitly asks it to
  remember the page.
- The bridge stores memories locally by default in `memory_server/data/`.
- Do not enable automatic history ingestion without explicit consent, filtering,
  and deletion controls.
