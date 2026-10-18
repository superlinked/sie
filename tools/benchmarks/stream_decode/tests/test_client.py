import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tools.benchmarks.stream_decode.client import stop_child
from tools.benchmarks.stream_decode.observer import SOURCE_FILES, VERSION, digest


def test_real_sdk_child_controller_records_serial_streams(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://invalid-proxy.invalid:9")
    monkeypatch.setenv("ALL_PROXY", "http://invalid-proxy.invalid:9")
    monkeypatch.setenv("NO_PROXY", "")
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            chunks = [
                {"choices": [{"index": 0, "delta": {"content": "ab"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {"content": "cd"}, "finish_reason": "length"}]},
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
            ]
            body = ("".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("A fixed test prompt")
    manifest = tmp_path / "server.json"
    manifest.write_text(json.dumps({"cohort": "local-test"}))
    (tmp_path / "installed-1.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "source_files": SOURCE_FILES,
                "server_manifest_sha256": digest({"cohort": "local-test"}),
            }
        )
    )
    output = tmp_path / "result"
    try:
        run = subprocess.run(  # noqa: S603 - fixed test module, no shell
            [
                sys.executable,
                "-m",
                "tools.benchmarks.stream_decode.client",
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--model",
                "test/model",
                "--prompt",
                str(prompt),
                "--server-manifest",
                str(manifest),
                "--output",
                str(output),
                "--completion-tokens",
                "4",
                "--warmups",
                "1",
                "--repeats",
                "1",
                "--request-seconds",
                "5",
                "--total-seconds",
                "40",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert run.returncode == 0, run.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(bodies) == 2
    assert all(b["min_tokens"] == b["max_completion_tokens"] == 4 for b in bodies)
    assert all(b["stream"] and b["stream_options"]["include_usage"] for b in bodies)
    records = [json.loads(path.read_text()) for path in sorted(output.glob("trial-*.json"))]
    assert [r["warmup"] for r in records] == [True, False]
    assert all(r["completed"] and r["retry_count"] == 0 and r["http_requests"] == 1 for r in records)
    assert len({r["started_ns"] for r in records}) == 2
    assert records[0]["finished_ns"] < records[1]["started_ns"]
    assert json.loads((output / "controller.json").read_text())["completed"]


def test_owned_child_is_terminated_and_reaped():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert stop_child(process) == ["SIGTERM"]
        assert process.poll() is not None
    finally:
        stop_child(process)


def test_child_wall_deadline_works_without_controller(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import time\nfrom tools.benchmarks.stream_decode.client import child_deadline\n"
            "with child_deadline(0.1):\n    time.sleep(60)\n",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert "fixed child wall deadline expired" in result.stderr
