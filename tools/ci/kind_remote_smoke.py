"""Serve a remote_only model end to end through the gateway's queue path on a
local kind cluster.

This builds the sie-config, sie-gateway, sie-server-sidecar and sie-server
(cpu/default bundle) images from the checkout, loads them into an ephemeral
kind cluster, installs the sie-cluster Helm chart with its dedicated
`workers.pools.remote` lane enabled (see deploy/helm/sie-cluster/values.yaml),
registers one `remote_only` model against a fake OpenAI-compatible upstream,
and sends one request through the gateway with `sie_sdk.SIEClient`.

The fake upstream (tools/ci/fixtures/fake_openai_upstream.py) runs as a
`kubectl debug` ephemeral container attached to the remote worker's own pod,
reachable only at that pod's loopback address. Containers in one pod share a
network namespace, so this needs no Service, no DNS name and no TLS
certificate: a loopback upstream is the one case the server's upstream
validation accepts plain HTTP for. It also means no other pod, including the
gateway's, has any network path to it: a gateway pod's own loopback is its
own, separate network namespace. A successful call therefore could only have
come from a process inside the worker's pod.

Chart note: the remote lane's `imageBundle` is pinned to `default`, and the
chart refuses `imageBundle: remote` at render time ("no remote worker image
is published; set imageBundle: default to run the remote adapters on the
%s-default image", templates/worker-statefulset.yaml). The default bundle
(packages/sie_server/bundles/default.yaml) lists the same remote adapters as
the remote bundle, at a higher priority, so a single built image serves both;
this script therefore builds `--bundle default`, not `--bundle remote`.

Usage:
    mise exec -- uv run --frozen --project . --no-sync python -m tools.ci.kind_remote_smoke

Requires a local Docker daemon, and `kind` and `helm` (mise-managed; `kind` is
not, see docs/installation.md) on PATH. Creates and deletes its own
uniquely-named kind cluster and local images; touches no other cluster or
registry.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from sie_sdk import SIEClient

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_UPSTREAM_SERVER = ROOT / "tools/ci/fixtures/fake_openai_upstream.py"
NAMESPACE = "sie"
UPSTREAM_NAME = "fake-upstream"
FAKE_UPSTREAM_PORT = 8300
EMBEDDING_DIM = 8
MODEL_ID = "test/remote-fake-embedding"
MIN_FREE_GIB = 25.0
DEBUG_IMAGE = "python:3.12-slim-bookworm"

MODEL_CONFIG = f"""\
sie_id: {MODEL_ID}
remote_backed: true
routing:
  policy: remote_only
max_sequence_length: 256
tasks:
  encode:
    dense:
      dim: {EMBEDDING_DIM}
profiles:
  default:
    adapter_path: sie_server.adapters.remote.openai:OpenAIUpstreamAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        upstream: {UPSTREAM_NAME}
        upstream_model: fake-embedding-model
"""


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    kwargs.setdefault("cwd", ROOT)
    kwargs.setdefault("check", True)
    print(f"+ {' '.join(str(a) for a in args)}", flush=True)
    return subprocess.run(args, **kwargs)  # noqa: S603 - fixed argv, no shell


def capture(args: list[str], **kwargs) -> str:
    kwargs.setdefault("cwd", ROOT)
    result = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)  # noqa: S603
    return result.stdout.strip()


def require_disk(min_gib: float = MIN_FREE_GIB) -> None:
    free_gib = shutil.disk_usage(ROOT).free / (1024**3)
    print(f"[disk] {free_gib:.1f} GiB free on the volume holding {ROOT}")
    if free_gib < min_gib:
        raise RuntimeError(f"only {free_gib:.1f} GiB free, below the {min_gib} GiB gate; not starting")


def require_local_docker() -> None:
    info = json.loads(capture(["docker", "info", "--format", "{{json .}}"]))
    if info["OSType"] != "linux":
        raise RuntimeError("this smoke test requires a Linux Docker daemon (kind runs Linux nodes)")
    print(f"[docker] Linux {info['Architecture']}, {info['NCPU']} CPUs, {info['MemTotal'] // (1024**3)} GiB")


def build_images(registry: str, version: str, revision: str) -> dict[str, str]:
    common = ["--registry", registry, "--version", version, "--source-revision", revision]
    run(["mise", "run", "docker", "--", "build-server", "--platform", "cpu", "--bundle", "default", *common])
    images = {"sie-server": f"{registry}/sie-server:v{version}-cpu-default"}
    for service in ("sie-config", "sie-gateway", "sie-server-sidecar"):
        run(["mise", "run", "docker", "--", "build-service", "--service", service, *common])
        images[service] = f"{registry}/{service}:v{version}"
    return images


def write_values(images: dict[str, str], version: str) -> Path:
    def repo(image: str) -> str:
        return image.rsplit(":", 1)[0]

    content = f"""\
fullnameOverride: sie
telemetry:
  enabled: false
  deploymentEnv: ci
payloadStore:
  enabled: false
upstreams:
  {UPSTREAM_NAME}:
    kind: openai
    base_url: http://127.0.0.1:{FAKE_UPSTREAM_PORT}/v1
    endpoints: [embeddings]
    rate_cap:
      requests_per_minute: 60
      max_concurrency: 4
    api_key_secret:
      name: fake-upstream-key
      key: api-key
workers:
  common:
    image:
      repository: {repo(images["sie-server"])}
      tag: v{version}
    workerSidecar:
      image:
        repository: {repo(images["sie-server-sidecar"])}
        tag: v{version}
  pools:
    remote:
      enabled: true
gateway:
  image:
    repository: {repo(images["sie-gateway"])}
    tag: v{version}
config:
  image:
    repository: {repo(images["sie-config"])}
    tag: v{version}
"""
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="sie-remote-smoke-values-", delete=False
    )
    handle.write(content)
    handle.close()
    return Path(handle.name)


def wait_http(url: str, timeout: float = 180, *, headers: dict[str, str] | None = None) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=3) as response:  # noqa: S310 - fixed localhost URL
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(2)
    raise RuntimeError(f"{url} never became ready: {last_error}")


def port_forward(resource: str, remote_port: int) -> tuple[subprocess.Popen, int]:
    process = subprocess.Popen(  # noqa: S603
        ["kubectl", "-n", NAMESPACE, "port-forward", resource, f"0:{remote_port}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    line = process.stdout.readline()
    match = re.search(r"127\.0\.0\.1:(\d+)", line)
    if not match:
        process.terminate()
        raise RuntimeError(f"kubectl port-forward {resource} failed to start: {line!r}")
    return process, int(match.group(1))


def register_model(config_url: str, admin_token: str) -> None:
    request = urllib.request.Request(
        f"{config_url}/v1/configs/models",
        data=MODEL_CONFIG.encode("utf-8"),
        method="POST",
    )
    request.add_header("Authorization", f"Bearer {admin_token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - fixed localhost URL
            print(f"[config] registered {MODEL_ID}: HTTP {response.status}")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"registering {MODEL_ID} failed: HTTP {error.code} {error.read().decode()}") from None


def fake_upstream_calls(worker_port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{worker_port}/calls", timeout=10) as response:  # noqa: S310
        return json.loads(response.read())


def attach_fake_upstream(pod: str, timeout: float = 120) -> None:
    script = FIXTURE_UPSTREAM_SERVER.read_text()
    result = run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "debug",
            pod,
            "-c",
            "fake-upstream",
            f"--image={DEBUG_IMAGE}",
            "--attach=false",
            "--profile=general",
            "--",
            "python3",
            "-c",
            script,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    print(f"[fake-upstream] kubectl debug: {result.stdout!r}")

    deadline = time.monotonic() + timeout
    state = None
    while time.monotonic() < deadline:
        state = capture(
            [
                "kubectl",
                "-n",
                NAMESPACE,
                "get",
                "pod",
                pod,
                "-o",
                'jsonpath={.status.ephemeralContainerStatuses[?(@.name=="fake-upstream")]}',
            ]
        )
        print(f"[fake-upstream] ephemeral container status: {state}")
        if '"running"' in state:
            return
        if '"terminated"' in state:
            break
        time.sleep(3)
    logs = subprocess.run(  # noqa: S603, S607 - fixed argv, diagnostic only
        ["kubectl", "-n", NAMESPACE, "logs", pod, "-c", "fake-upstream"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    raise RuntimeError(f"fake-upstream ephemeral container never reached Running; last status {state}; logs:\n{logs}")


def encode_through_gateway(gateway_url: str, timeout: float = 120) -> dict:
    """Retry the encode call: sie-config -> gateway catalog propagation is async."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    with SIEClient(gateway_url, timeout_s=30) as client:
        while time.monotonic() < deadline:
            try:
                return client.encode(MODEL_ID, [{"text": "a fake upstream smoke probe"}])[0]
            except Exception as error:  # noqa: BLE001 - retry until the model is routable
                last_error = error
                time.sleep(3)
    raise RuntimeError(f"gateway never served {MODEL_ID}: {last_error}")


def main() -> None:
    require_disk()
    require_local_docker()

    suffix = uuid.uuid4().hex[:8]
    cluster = f"sie-remote-smoke-{suffix}"
    registry = f"localhost/sie-remote-smoke-{suffix}"
    version = "0.0.0"
    revision = capture(["git", "rev-parse", "HEAD"])
    kubeconfig = Path(tempfile.mkstemp(prefix="sie-remote-smoke-kubeconfig-")[1])
    os.environ["KUBECONFIG"] = str(kubeconfig)

    images = build_images(registry, version, revision)
    values_path = write_values(images, version)
    gateway_forward: subprocess.Popen | None = None
    config_forward: subprocess.Popen | None = None
    worker_forward: subprocess.Popen | None = None

    try:
        run(["kind", "create", "cluster", "--name", cluster, "--wait", "180s"])
        run(["kubectl", "taint", "nodes", "--all", "node-role.kubernetes.io/control-plane-"], check=False)

        for image in images.values():
            run(["kind", "load", "docker-image", image, "--name", cluster])

        run(["kubectl", "create", "namespace", NAMESPACE])
        run(
            [
                "kubectl",
                "-n",
                NAMESPACE,
                "create",
                "secret",
                "generic",
                "fake-upstream-key",
                "--from-literal=api-key=dummy-test-key",
            ]
        )

        run(["mise", "run", "helm", "--", "install", "--apply", "--values", str(values_path)])

        admin_token_b64 = capture(
            [
                "kubectl",
                "-n",
                NAMESPACE,
                "get",
                "secret",
                "sie-config-admin-token",
                "-o",
                "jsonpath={.data.SIE_ADMIN_TOKEN}",
            ]
        )
        admin_token = base64.b64decode(admin_token_b64).decode()

        config_forward, config_port = port_forward("svc/sie-config", 8080)
        config_url = f"http://127.0.0.1:{config_port}"
        wait_http(f"{config_url}/readyz")
        register_model(config_url, admin_token)

        worker_pod = capture(
            [
                "kubectl",
                "-n",
                NAMESPACE,
                "get",
                "pods",
                "-l",
                "app.kubernetes.io/component=worker",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ]
        )
        print(f"[worker] pod: {worker_pod}")
        attach_fake_upstream(worker_pod)

        worker_forward, worker_port = port_forward(f"pod/{worker_pod}", FAKE_UPSTREAM_PORT)
        wait_http(f"http://127.0.0.1:{worker_port}/calls")
        print("[fake-upstream] ready, reachable via the worker pod's own port-forward")

        gateway_forward, gateway_port = port_forward("svc/sie-gateway", 8080)
        gateway_url = f"http://127.0.0.1:{gateway_port}"
        wait_http(f"{gateway_url}/readyz")

        result = encode_through_gateway(gateway_url)
        served_by = result["request"].get("served_by")
        upstream = result["request"].get("upstream")
        print(f"[gateway] encode result request metadata: {result['request']}")
        if served_by != "remote" or upstream != UPSTREAM_NAME:
            raise RuntimeError(f"expected served_by=remote upstream={UPSTREAM_NAME}, got {result['request']}")

        calls = fake_upstream_calls(worker_port)
        print(f"[fake-upstream] /calls: {calls}")
        if calls["count"] != 1:
            raise RuntimeError(f"expected exactly one upstream call, saw {calls['count']}: {calls['calls']}")

        print("\n=== PASS ===")
        print(f"model={MODEL_ID} served_by={served_by} upstream={upstream}")
        print(f"fake upstream call: {calls['calls'][0]}")
        print(
            "the gateway pod has no network path to this address: it is the worker pod's own "
            "loopback, in a separate network namespace from the gateway's."
        )
    finally:
        for process in (gateway_forward, config_forward, worker_forward):
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        run(["kind", "delete", "cluster", "--name", cluster], check=False)
        run(["docker", "rmi", "-f", *images.values()], check=False)
        kubeconfig.unlink(missing_ok=True)
        values_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
