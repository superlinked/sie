"""A fake OpenAI-compatible embeddings endpoint for the kind remote-worker smoke test.

tools/ci/kind_remote_smoke.py runs this, unmodified, as the main process of a
``kubectl debug`` ephemeral container attached to the remote worker's own pod.
Containers in one pod share a network namespace, so this process listens on
the pod's loopback address and the remote worker's ``openai`` upstream adapter
reaches it at ``http://127.0.0.1:<port>/v1`` with no TLS: a loopback upstream
is the one case ``sie_server.config.upstreams.validate_upstream_url`` accepts
plain HTTP for (see packages/sie_server/REMOTE_BACKENDS.md, "Controls and
failure behavior"). It issues no outbound calls and holds no credentials; it
only records what it received, in-memory, for the test to read back over the
same pod's loopback.

Standard library only: the ephemeral container runs a stock Python image with
nothing installed.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("FAKE_UPSTREAM_PORT", "8300"))
DIM = int(os.environ.get("FAKE_UPSTREAM_DIM", "8"))

_lock = threading.Lock()
_calls: list[dict[str, object]] = []


def _embedding(index: int) -> list[float]:
    return [round((index + 1) / 10 + slot * 0.01, 4) for slot in range(DIM)]


class Handler(BaseHTTPRequestHandler):
    server_version = "FakeOpenAIUpstream/1"

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        if self.path != "/v1/embeddings":
            with _lock:
                _calls.append({"path": self.path, "method": "POST", "client_port": self.client_address[1]})
            self._send_json(404, {"error": {"message": f"no fake route for {self.path}"}})
            return
        try:
            request = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            request = {}
        inputs = request.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        with _lock:
            _calls.append(
                {
                    "path": self.path,
                    "method": "POST",
                    "model": request.get("model"),
                    "input_count": len(inputs),
                    "authorization": self.headers.get("Authorization", ""),
                    "client_port": self.client_address[1],
                }
            )
        data = [{"object": "embedding", "index": i, "embedding": _embedding(i)} for i in range(len(inputs))]
        token_count = sum(len(str(item).split()) for item in inputs) or 1
        self._send_json(
            200,
            {
                "object": "list",
                "data": data,
                "model": request.get("model", "fake-embedding-model"),
                "usage": {"prompt_tokens": token_count, "total_tokens": token_count},
            },
        )

    def do_GET(self) -> None:
        if self.path == "/calls":
            with _lock:
                self._send_json(200, {"count": len(_calls), "calls": list(_calls)})
            return
        self._send_json(404, {"error": {"message": f"no fake route for {self.path}"}})

    def log_message(self, format: str, *args: object) -> None:  # matches BaseHTTPRequestHandler's signature
        sys.stderr.write(f"[fake-upstream] {self.address_string()} {format % args}\n")


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)  # noqa: S104 - must be reachable from a sibling container
    print(f"[fake-upstream] listening on 0.0.0.0:{PORT}, dim={DIM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
