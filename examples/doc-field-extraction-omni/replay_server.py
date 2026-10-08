#!/usr/bin/env python3
"""Check the send path offline: a local stand-in for /v1/chat/completions that answers with the published replies.

    python3 replay_server.py --set confirm --port 8099
    uv run run.py --set confirm --base-url http://127.0.0.1:8099 --out runs/replay-confirm
    curl -s http://127.0.0.1:8099/replay-stats

No model runs. For every request, the server takes the body that sie_sdk.SIEClient put on the wire, replaces the
image with its sha256 and compares the result with the published readable body of that document. It then answers
with the document's published reply; a published capped row is answered with finish_reason "length". Scoring the
replayed run must give the published mean, and /replay-stats must show every body matching.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from fetch import DATA, SETS, fetch_published, sha256
from score import read_rows


def load(set_name: str, cache: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """The published readable bodies keyed by image sha256, and the published grader rows."""
    published = fetch_published(set_name, cache)
    show = {}
    for line in published["show"].read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["slot"] == "study":
            if row["image_sha256"] in show:
                raise SystemExit("Two study rows share an image")
            show[row["image_sha256"]] = row
    return show, read_rows(published["replies"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=SETS, required=True)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--cache", type=Path, default=DATA, help="download folder (default data/)")
    args = parser.parse_args()
    show, replies = load(args.set, args.cache)
    stats: dict[str, Any] = {
        "requests": 0,
        "readiness": 0,
        "study": 0,
        "body_matches_published": 0,
        "body_differs": [],
        "unknown_image": 0,
        "other_paths": [],
    }
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def reply(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/replay-stats":
                with lock:
                    self.reply(200, stats)
                return
            with lock:
                stats["other_paths"].append(f"GET {self.path}")
            self.reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path != "/v1/chat/completions":
                with lock:
                    stats["other_paths"].append(f"POST {self.path}")
                self.reply(404, {"error": "not found"})
                return
            with lock:
                stats["requests"] += 1
            user = body["messages"][-1]["content"]
            if isinstance(user, str):  # the text-only readiness request
                with lock:
                    stats["readiness"] += 1
                content, finish = json.dumps({"ok": True}), "stop"
            else:
                url = user[0]["image_url"]["url"]
                prefix, encoded = url.split(",", 1)
                digest = sha256(base64.b64decode(encoded))
                row = show.get(digest)
                if row is None:
                    with lock:
                        stats["unknown_image"] += 1
                    self.reply(400, {"error": "image not in the published set"})
                    return
                user[0]["image_url"]["url"] = f"{prefix},<image sha256:{digest}>"
                same = json.dumps(body, sort_keys=True) == json.dumps(row["body"], sort_keys=True)
                with lock:
                    stats["study"] += 1
                    if same:
                        stats["body_matches_published"] += 1
                    else:
                        stats["body_differs"].append(row["id"])
                published = replies[row["id"]]
                content = published.get("text") or ""
                finish = "length" if published.get("error") == "capped" else "stop"
            self.reply(
                200,
                {
                    "id": "replay",
                    "object": "chat.completion",
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}
                    ],
                },
            )

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Replaying {len(show)} published {args.set} replies on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
