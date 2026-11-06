"""Exercise released Cognee remember/recall against deterministic loopback HTTP."""

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tiktoken
from sie_cognee.config import CogneeSettings, Endpoint, configure
from sie_cognee.documents import SourceDocument
from sie_cognee.retrieval import retrieve_chunks

# A tiny local tokenizer makes this fixture independent of downloaded tokenizer
# caches. Model transport, structured schemas, storage and retrieval stay real.
encoding = tiktoken.Encoding(
    name="byte-fixture", pat_str=r"[\s\S]", mergeable_ranks={bytes([i]): i for i in range(256)}, special_tokens={}
)
tiktoken.get_encoding = lambda *_args, **_kwargs: encoding
tiktoken.encoding_for_model = lambda *_args, **_kwargs: encoding
text = "The projector is confirmed for the workshop on 2026-11-14."
requests = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append((self.path, body))
        if self.path == "/v1/embeddings":
            assert body["model"] == "text-embedding-3-small"
            payload = {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": index, "embedding": [1.0, 0.5, 0.25]}
                    for index, _ in enumerate(body["input"])
                ],
                "model": body["model"],
                "usage": {"prompt_tokens": len(body["input"]), "total_tokens": len(body["input"])},
            }
        elif self.path == "/v1/chat/completions":
            assert body["model"] == "gpt-4o"
            schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
            if not schema:
                output = "test"
            elif "nodes" in schema["properties"]:
                output = {
                    "nodes": [{"id": "projector", "name": "Projector", "type": "Equipment", "description": text}],
                    "edges": [],
                }
            else:
                assert "summary" in schema["properties"], schema["properties"]
                output = {"summary": text, "description": "Synthetic equipment note"}
            payload = {
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(output)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        else:
            raise AssertionError(f"Unexpected path: {self.path}")
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


async def main():
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = Endpoint(f"http://127.0.0.1:{server.server_port}")
            configure(CogneeSettings(endpoint, endpoint, "gpt-4o", "text-embedding-3-small", 3, Path(sys.argv[1])))
            import cognee

            document = SourceDocument(text, "workshop-projector", "note.txt")
            await cognee.remember(document.to_data_item(), dataset_name="fixture", self_improvement=False)
            chunks = await retrieve_chunks("projector", dataset="fixture", top_k=2)
            assert chunks
            assert text in chunks[0].text
            assert chunks[0].chunk_id
            assert chunks[0].metadata.get("data_id") or chunks[0].metadata.get("doc_id")
            assert any(path == "/v1/chat/completions" for path, _ in requests)
            assert any(path == "/v1/embeddings" for path, _ in requests)
            assert os.environ["GRAPH_EXTRACTOR"] == "llm"
        finally:
            server.shutdown()
            thread.join(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
