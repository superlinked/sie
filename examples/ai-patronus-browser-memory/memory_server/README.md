# Patronus SIE Memory Bridge

This is the open-source-safe Superlinked/SIE part of Patronus AI: a tiny local
bridge that lets the Chrome extension manually save pages and later recall them
with semantic search.

It does not use hackathon keys or hosted partner endpoints. By default it talks
to a local SIE server at `http://localhost:8080` and stores page memories in
`data/memory.json`.

## Run

Start SIE:

```bash
docker run -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface ghcr.io/superlinked/sie-server:latest-cpu-default
```

Start the memory bridge:

```bash
cd memory_server
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload --port 8800
```

Optional: seed sample memories after the server is running:

```bash
python seed_sample.py
```

Then load `../extension/` unpacked in Chrome and set:

```text
Superlinked URL: http://localhost:8800
Superlinked token: blank
```

Ask Patronus to:

```text
remember this page
what did I save about browser memory?
```

## API

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

## Privacy

The extension only sends page text when the user explicitly asks it to remember
the page. Do not enable automatic ingestion without adding clear consent,
filtering, and deletion controls.
