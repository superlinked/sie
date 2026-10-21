from __future__ import annotations

import base64
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import yaml

from tools.mise_tasks import helm

ROOT = Path(__file__).resolve().parents[3]
WORKER_TEMPLATE = "templates/worker-statefulset.yaml"
KEDA_APPLY_TEMPLATE = "templates/keda-scaledobject.yaml"
KEDA_CLEANUP_TEMPLATE = "templates/keda-lifecycle.yaml"
AUTOSCALING_VALUES = {
    "autoscaling": {"enabled": True},
    "observability": {
        "otel": {"collector": {"prometheus": {"networkPolicy": {"scrapeNamespaceNames": ["monitoring"]}}}}
    },
}
GENERATED_ADMIN_TOKEN_SECRET = "sie-sie-cluster-config-admin-token"
GENERATED_READ_TOKEN_SECRET = "sie-sie-cluster-config-read-token"
CONFIG_SERVICE = "sie-sie-cluster-config/config"
GATEWAY = "sie-sie-cluster-gateway/gateway"
CONFIG_CONSUMERS = (GATEWAY, "sie-sie-cluster-worker-l4-default/worker-sidecar")
L4_POOL = {"workers": {"pools": {"l4": {"enabled": True}}}}


def render_workers(tmp_path: Path, values: dict) -> subprocess.CompletedProcess[str]:
    return render_template(tmp_path, values, WORKER_TEMPLATE)


def render_template(tmp_path: Path, values: dict, template: str) -> subprocess.CompletedProcess[str]:
    values_file = tmp_path / "values.yaml"
    values_file.write_text(yaml.safe_dump(values), encoding="utf-8")
    return subprocess.run(
        [
            "mise",
            "exec",
            "--",
            "helm",
            "template",
            "sie",
            str(helm.CHART_DIR),
            "--namespace",
            "sie",
            *helm.validation_args(["-f", str(values_file), "--show-only", template]),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def render_collector_config(tmp_path: Path, *, remote_enabled: bool = True) -> tuple[dict, str]:
    chart = tmp_path / "collector-chart"
    (chart / "templates").mkdir(parents=True)
    (chart / "Chart.yaml").write_text("apiVersion: v2\nname: collector-test\nversion: 0.0.0\n")
    shutil.copyfile(ROOT / helm.CHART_DIR / "templates/_otel-collector-config.tpl", chart / "templates/_collector.tpl")
    (chart / "templates/config.yaml").write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: test\ndata:\n  collector.yaml: |\n"
        '    {{- include "sie-cluster.otel.collectorConfig" .Values | nindent 4 }}\n'
    )
    defaults = yaml.safe_load((ROOT / helm.CHART_DIR / "values.yaml").read_text())
    collector = defaults["observability"]["otel"]["collector"]
    collector["betterStack"].update({"enabled": remote_enabled, "endpoint": "https://example.invalid"})
    values = {
        "metricsEnabled": True,
        "logsEnabled": False,
        "tracesEnabled": False,
        "prometheusEnabled": True,
        "collector": collector,
        "traceEndpoint": "",
        "logEndpoint": "",
        "betterStack": collector["betterStack"],
        "deploymentEnvironment": "dev",
        "cloudRegion": "test-region",
        "localTraceExporters": [],
        "logExporters": [],
    }
    (chart / "values.yaml").write_text(yaml.safe_dump(values))
    rendered = subprocess.run(  # noqa: S603 - fixed local Helm test chart
        ["mise", "exec", "--", "helm", "template", "test", str(chart)],  # noqa: S607 - mise-managed Helm
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    image = collector["image"]
    return yaml.safe_load(yaml.safe_load(rendered)["data"]["collector.yaml"]), f"{image['repository']}:{image['tag']}"


def worker_statefulsets(tmp_path: Path, values: dict) -> list[dict]:
    result = render_workers(tmp_path, values)
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc and doc["kind"] == "StatefulSet"]


def shm_volume(statefulset: dict) -> dict:
    (volume,) = [volume for volume in statefulset["spec"]["template"]["spec"]["volumes"] if volume["name"] == "shm"]
    return volume["emptyDir"]


def shm_size_limits(tmp_path: Path, values: dict) -> dict[str, str]:
    limits = {}
    for statefulset in worker_statefulsets(tmp_path, values):
        volume = shm_volume(statefulset)
        assert volume["medium"] == "Memory"
        limits[statefulset["metadata"]["labels"]["sie.superlinked.com/pool"]] = volume["sizeLimit"]
    return limits


def readme_device_group_values() -> dict:
    readme = (ROOT / helm.CHART_DIR / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Tensor-Parallel Device Groups\n", 1)[1].split("\n### ", 1)[0]
    (pool_example,) = [
        block for block in re.findall(r"```yaml\n(.*?)```", section, flags=re.DOTALL) if block.startswith("workers:")
    ]
    return yaml.safe_load(pool_example)


def test_worker_shm_defaults_to_8gi(tmp_path: Path) -> None:
    assert shm_size_limits(tmp_path, {"workers": {"pools": {"l4": {"enabled": True}}}}) == {"l4": "8Gi"}


@pytest.mark.parametrize(
    ("common", "expected"),
    [
        ({}, {"l4": "32Gi", "cpu": "8Gi"}),
        ({"shmSize": "16Gi"}, {"l4": "32Gi", "cpu": "16Gi"}),
    ],
)
def test_pool_shm_size_overrides_the_common_size(tmp_path: Path, common: dict, expected: dict) -> None:
    pools = {"l4": {"enabled": True, "shmSize": "32Gi"}, "cpu": {"enabled": True}}
    assert shm_size_limits(tmp_path, {"workers": {"common": common, "pools": pools}}) == expected


@pytest.mark.parametrize(
    "workers",
    [
        {"common": {"shmSize": ""}, "pools": {"l4": {"enabled": True}}},
        {"common": {"shmSize": None}, "pools": {"l4": {"enabled": True}}},
        {"common": {"shmSize": " "}, "pools": {"l4": {"enabled": True, "shmSize": ""}}},
        {"pools": {"l4": {"enabled": True, "shmSize": " "}}},
    ],
)
def test_empty_shm_size_fails_the_render(tmp_path: Path, workers: dict) -> None:
    result = render_workers(tmp_path, {"workers": workers})
    assert result.returncode != 0
    assert "workers.pools.l4: the /dev/shm size is empty." in result.stderr


def test_readme_device_group_example_renders(tmp_path: Path) -> None:
    values = readme_device_group_values()
    ((pool_name, pool),) = values["workers"]["pools"].items()
    (statefulset,) = worker_statefulsets(tmp_path, values)
    assert statefulset["metadata"]["labels"]["sie.superlinked.com/pool"] == pool_name
    assert shm_volume(statefulset)["sizeLimit"] == pool["shmSize"]
    (worker,) = [
        container
        for container in statefulset["spec"]["template"]["spec"]["containers"]
        if container["name"] == "worker"
    ]
    assert worker["resources"]["limits"]["nvidia.com/gpu"] == str(pool["gpu"]["count"])
    assert worker["resources"]["limits"]["memory"] == pool["resources"]["limits"]["memory"]


def hook_events(tmp_path: Path, values: dict, template: str) -> dict[tuple[str, str], set[str]]:
    result = render_template(tmp_path, values, template)
    assert result.returncode == 0, result.stderr
    return {
        (doc["kind"], doc["metadata"]["name"]): set(doc["metadata"]["annotations"]["helm.sh/hook"].split(","))
        for doc in yaml.safe_load_all(result.stdout)
        if doc and "helm.sh/hook" in doc["metadata"].get("annotations", {})
    }


def test_worker_scaledobject_appends_the_sixth_trigger(tmp_path: Path) -> None:
    values = {**L4_POOL, **AUTOSCALING_VALUES}
    rendered = render_template(tmp_path, values, KEDA_APPLY_TEMPLATE)
    assert rendered.returncode == 0, rendered.stderr
    scaled_objects = [
        manifest
        for doc in yaml.safe_load_all(rendered.stdout)
        if doc and doc.get("kind") == "ConfigMap"
        for manifest in yaml.safe_load_all(doc.get("data", {}).get("scaledobjects.yaml", ""))
        if manifest and manifest.get("kind") == "ScaledObject"
    ]
    (worker,) = [
        scaled_object
        for scaled_object in scaled_objects
        if scaled_object["spec"]["scaleTargetRef"]["kind"] == "StatefulSet"
    ]
    triggers = worker["spec"]["triggers"]
    assert [trigger["type"] for trigger in triggers] == ["prometheus"] * 6
    # Indexes 0-4 match the pre-#293 HPA names. The loading trigger is last.
    assert [trigger["metadata"]["metricName"] for trigger in triggers] == [
        "sie_gateway_pending_demand_l4_default",
        "sie_gateway_queue_depth_l4_default",
        "sie_gateway_active_lease_gpus_l4_default",
        "sie_gateway_pool_warm_floor_l4_default",
        "sie_gateway_rejected_rate_l4_default",
        "sie_gateway_model_loads_in_progress_l4_default",
    ]
    loading = triggers[-1]["metadata"]
    assert "max(sie_gateway_model_loads_in_progress{" in loading["query"]
    assert "model_loads_in_progress_ratio" not in loading["query"]
    assert loading["ignoreNullValues"] == "false"

    hook = render_template(tmp_path, values, "templates/hooks/keda-ready-test.yaml")
    assert hook.returncode == 0, hook.stderr
    (job,) = [doc for doc in yaml.safe_load_all(hook.stdout) if doc and doc.get("kind") == "Job"]
    script = job["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "EXPECTED_METRIC_COUNT=7" in script
    # 1-based probe count: 6 is trigger index 5 and is allowed; 7 is rejected.
    assert re.search(r'\[ "\$object_metric_index" -lt 1 \] \|\| \[ "\$object_metric_index" -gt 6 \]', script)
    assert re.search(r"\.key >= 0 and \.key <= 5\b", script)
    scaled_objects_count = 2
    probe_batches = (scaled_objects_count + 16 - 1) // 16
    timeout = (30 * 3) + (probe_batches * 6 * 5 * 3) + 180
    assert f"TIMEOUT={timeout}" in script
    assert job["spec"]["activeDeadlineSeconds"] == timeout + 60


@pytest.mark.parametrize("remote_enabled", [False, True])
def test_model_loading_unit_override_is_prometheus_only(tmp_path: Path, remote_enabled: bool) -> None:
    name = "sie.gateway.model_loads_in_progress"
    contract = yaml.safe_load((ROOT / "telemetry/contract.yaml").read_text())
    (metric,) = [metric for metric in contract["metrics"] if metric["name"] == name]
    assert metric["unit"] == "1"
    assert metric["prometheus_name"] == "sie_gateway_model_loads_in_progress"
    assert contract["metric_processing"]["prometheus_compatibility"][name] == {
        "unit_override": "",
        "prometheus_name": metric["prometheus_name"],
    }
    config, _ = render_collector_config(tmp_path, remote_enabled=remote_enabled)
    processors = config["processors"]
    compatibility = processors["transform/prometheus_compatibility"]["metric_statements"]
    assert f'set(unit, "") where name == "{name}"' in compatibility[0]["statements"]
    canonical = processors["transform/contract_metrics"]["metric_statements"]
    (descriptor,) = [group for group in canonical if group["context"] == "metric"]
    assert any(
        statement.startswith('set(unit, "1")') and "sie[.]gateway[.]model_loads_in_progress" in statement
        for statement in descriptor["statements"]
    )
    pipelines = config["service"]["pipelines"]
    for receiver in ("gateway", "application"):
        local = pipelines[f"metrics/prometheus/{receiver}"]["processors"]
        assert local.index("transform/contract_metrics") < local.index("transform/prometheus_compatibility")
        if remote_enabled:
            assert (
                "transform/prometheus_compatibility" not in pipelines[f"metrics/betterstack/{receiver}"]["processors"]
            )


def test_scaledobject_apply_hook_also_runs_on_rollback(tmp_path: Path) -> None:
    hooks = hook_events(tmp_path, AUTOSCALING_VALUES, KEDA_APPLY_TEMPLATE)

    assert sorted(kind for kind, _ in hooks) == ["Job", "Role", "RoleBinding", "ServiceAccount"]
    assert all(events == {"post-install", "post-upgrade", "post-rollback"} for events in hooks.values()), hooks


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, {"pre-upgrade", "pre-rollback", "pre-delete"}),
        (AUTOSCALING_VALUES, {"pre-delete"}),
    ],
)
def test_scaledobject_cleanup_hook_events(tmp_path: Path, values: dict, expected: set[str]) -> None:
    hooks = hook_events(tmp_path, values, KEDA_CLEANUP_TEMPLATE)

    assert sorted(kind for kind, _ in hooks) == ["Job", "Role", "RoleBinding", "ServiceAccount"]
    assert all(events == expected for events in hooks.values()), hooks


def test_dashboards_cover_public_prometheus_metrics(tmp_path: Path) -> None:
    result = render_template(tmp_path, {"dashboards": {"enabled": True}}, "templates/dashboard-configmaps.yaml")
    assert result.returncode == 0, result.stderr
    dashboards = [
        json.loads(payload)
        for doc in yaml.safe_load_all(result.stdout)
        if doc and doc["kind"] == "ConfigMap"
        for filename, payload in doc.get("data", {}).items()
        if filename.endswith(".json")
    ]
    assert dashboards
    expressions = [
        target["expr"]
        for dashboard in dashboards
        for panel in dashboard.get("panels", [])
        for target in panel.get("targets", [])
        if "expr" in target
    ]
    contract = yaml.safe_load((ROOT / "telemetry/contract.yaml").read_text())
    queried_names = set(re.findall(r"\bsie_[A-Za-z0-9_:]+", "\n".join(expressions)))
    missing = sorted(
        metric["prometheus_name"]
        for metric in contract["metrics"]
        if "prometheus" in metric["export"]
        and not queried_names
        & {
            metric["prometheus_name"] + suffix
            for suffix in (("_bucket", "_sum", "_count") if metric["type"] == "histogram" else ("",))
        }
    )
    assert not missing, f"Grafana queries missing public Prometheus metrics: {missing}"
    deadline = next(
        metric for metric in contract["metrics"] if metric["name"] == "sie.worker.work_item.deadline_exceeded"
    )
    (expression,) = [expression for expression in expressions if deadline["prometheus_name"] in expression]
    grouping = re.search(r"sum by \(([^)]+)\)", expression)
    assert grouping is not None
    assert set(grouping.group(1).replace(" ", "").split(",")) == set(contract["attribute_sets"][deadline["attributes"]])
    assert "rate(" in expression
    assert 'producer_service="sie-worker-sidecar"' in expression
    assert all(
        selector in expression
        for selector in ('namespace="$namespace"', 'service="$collector"', 'endpoint="prometheus"')
    )


def render_chart(tmp_path: Path, values: dict, overlays: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    values_file = tmp_path / "chart-values.yaml"
    values_file.write_text(yaml.safe_dump(values), encoding="utf-8")
    overlay_args = [arg for overlay in overlays for arg in ("-f", str(helm.CHART_DIR / overlay))]
    return subprocess.run(
        [
            "mise",
            "exec",
            "--",
            "helm",
            "template",
            "sie",
            str(helm.CHART_DIR),
            "--namespace",
            "sie",
            *(helm.validation_args(overlay_args) if overlays else []),
            "-f",
            str(values_file),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def rendered_documents(tmp_path: Path, values: dict, overlays: tuple[str, ...] = ()) -> list[dict]:
    result = render_chart(tmp_path, values, overlays)
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def container_env(docs: list[dict], workload: str, container: str) -> dict[str, dict]:
    (doc,) = [
        doc for doc in docs if doc["kind"] in {"Deployment", "StatefulSet"} and doc["metadata"]["name"] == workload
    ]
    (spec,) = [spec for spec in doc["spec"]["template"]["spec"]["containers"] if spec["name"] == container]
    return {env["name"]: env for env in spec.get("env", [])}


def env_entries(docs: list[dict], name: str) -> dict[str, dict]:
    entries = {}
    for doc in docs:
        if doc["kind"] not in {"Deployment", "StatefulSet"}:
            continue
        for container in doc["spec"]["template"]["spec"]["containers"]:
            for env in container.get("env", []):
                if env["name"] == name:
                    entries[f"{doc['metadata']['name']}/{container['name']}"] = env.get("valueFrom", {}).get(
                        "secretKeyRef", env.get("value")
                    )
    return entries


def token_secrets(docs: list[dict]) -> dict[str, dict]:
    return {
        doc["metadata"]["name"]: doc
        for doc in docs
        if doc["kind"] == "Secret" and doc["metadata"]["name"].endswith(("-admin-token", "-read-token"))
    }


def secret_token(secret: dict, key: str) -> str:
    assert secret["metadata"]["annotations"] == {"helm.sh/resource-policy": "keep"}
    token = base64.b64decode(secret["data"][key]).decode()
    assert re.fullmatch(r"[A-Za-z0-9]{64}", token)
    return token


def assert_token_scopes(docs: list[dict], admin: dict, read: dict | str) -> None:
    assert env_entries(docs, "SIE_ADMIN_TOKEN") == {CONFIG_SERVICE: admin}
    assert env_entries(docs, "SIE_CONFIG_READ_TOKEN") == {CONFIG_SERVICE: read}
    assert env_entries(docs, "SIE_CONFIG_SERVICE_TOKEN") == dict.fromkeys(CONFIG_CONSUMERS, read)


GENERATED_ADMIN = {"name": GENERATED_ADMIN_TOKEN_SECRET, "key": "SIE_ADMIN_TOKEN"}
GENERATED_READ = {"name": GENERATED_READ_TOKEN_SECRET, "key": "SIE_CONFIG_READ_TOKEN"}
# Values as `helm upgrade --reuse-values` sees them from a release that predates
# these keys: null removes the chart default.
PREDATING_TOKEN_KEYS = {
    "config": {
        "auth": {
            "generateAdminToken": None,
            "readTokenSecretName": None,
            "readTokenSecretKey": None,
            "generateReadToken": None,
        }
    },
    "gateway": {"auth": {"adminTokenSecretName": None, "adminTokenSecretKey": None}},
}


def test_chart_defaults_render_without_overrides(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, {})
    gateway = container_env(docs, "sie-sie-cluster-gateway", "gateway")
    assert "SIE_PAYLOAD_STORE_URL" not in gateway
    assert "SIE_ADMIN_TOKEN" not in gateway
    assert gateway["SIE_CONFIG_SERVICE_TOKEN"]["valueFrom"]["secretKeyRef"] == GENERATED_READ
    assert container_env(docs, "sie-sie-cluster-config", "config")["SIE_DEPLOYMENT_ENV"]["value"] == "production"


EMPTY_GENERATION_FLAGS = {"config": {"auth": {"generateAdminToken": "", "generateReadToken": " "}}}


@pytest.mark.parametrize("values", [{}, PREDATING_TOKEN_KEYS, EMPTY_GENERATION_FLAGS])
def test_generated_tokens_are_wired_by_scope(tmp_path: Path, values: dict) -> None:
    docs = rendered_documents(tmp_path, {**L4_POOL, **values})
    secrets = token_secrets(docs)
    assert set(secrets) == {GENERATED_ADMIN_TOKEN_SECRET, GENERATED_READ_TOKEN_SECRET}
    admin = secret_token(secrets[GENERATED_ADMIN_TOKEN_SECRET], "SIE_ADMIN_TOKEN")
    read = secret_token(secrets[GENERATED_READ_TOKEN_SECRET], "SIE_CONFIG_READ_TOKEN")
    assert admin != read
    assert_token_scopes(docs, GENERATED_ADMIN, GENERATED_READ)


def test_operator_admin_token_secret_is_used_unchanged(tmp_path: Path) -> None:
    auth = {"adminTokenSecretName": "operator-admin", "adminTokenSecretKey": "token"}
    docs = rendered_documents(tmp_path, {**L4_POOL, "config": {"auth": auth}})
    assert set(token_secrets(docs)) == {GENERATED_READ_TOKEN_SECRET}
    assert_token_scopes(docs, {"name": "operator-admin", "key": "token"}, GENERATED_READ)


def test_operator_read_token_secret_is_used_unchanged(tmp_path: Path) -> None:
    auth = {
        "adminTokenSecretName": "operator-admin",
        "adminTokenSecretKey": "token",
        "readTokenSecretName": "operator-read",
        "readTokenSecretKey": "read",
        "generateReadToken": False,
    }
    docs = rendered_documents(tmp_path, {**L4_POOL, "config": {"auth": auth}})
    assert token_secrets(docs) == {}
    assert_token_scopes(docs, {"name": "operator-admin", "key": "token"}, {"name": "operator-read", "key": "read"})


def test_gateway_admin_token_is_wired_only_to_the_gateway(tmp_path: Path) -> None:
    auth = {"mode": "token", "tokenSecretName": "gateway-tokens", "adminTokenSecretName": "gateway-admin"}
    values = {**L4_POOL, "gateway": {"auth": auth}}
    docs = rendered_documents(tmp_path, values)
    assert env_entries(docs, "SIE_ADMIN_TOKEN") == {
        CONFIG_SERVICE: GENERATED_ADMIN,
        GATEWAY: {"name": "gateway-admin", "key": "SIE_ADMIN_TOKEN"},
    }
    assert env_entries(docs, "SIE_CONFIG_SERVICE_TOKEN") == dict.fromkeys(CONFIG_CONSUMERS, GENERATED_READ)


def test_gateway_without_sie_config_sends_no_config_credential(tmp_path: Path) -> None:
    values = {
        "config": {"enabled": False},
        "gateway": {"embeddedConfigs": {"enabled": True}, "auth": {"adminTokenSecretName": "gateway-admin"}},
    }
    gateway = container_env(rendered_documents(tmp_path, values), "sie-sie-cluster-gateway", "gateway")
    assert "SIE_CONFIG_SERVICE_URL" not in gateway
    assert gateway["SIE_CONFIG_SERVICE_TOKEN"] == {"name": "SIE_CONFIG_SERVICE_TOKEN", "value": ""}
    assert gateway["SIE_ADMIN_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "gateway-admin",
        "key": "SIE_ADMIN_TOKEN",
    }


@pytest.mark.parametrize(
    ("values", "error"),
    [
        (
            {
                "config": {
                    "auth": {
                        "readTokenSecretName": GENERATED_ADMIN_TOKEN_SECRET,
                        "readTokenSecretKey": "SIE_ADMIN_TOKEN",
                    }
                }
            },
            "config.auth.readTokenSecretName and readTokenSecretKey point at the sie-config admin token",
        ),
        (
            {
                "config": {
                    "auth": {
                        "adminTokenSecretName": "shared",
                        "adminTokenSecretKey": "token",
                        "readTokenSecretName": "shared",
                        "readTokenSecretKey": "token",
                    }
                }
            },
            "config.auth.readTokenSecretName and readTokenSecretKey point at the sie-config admin token",
        ),
        (
            {"gateway": {"auth": {"adminTokenSecretName": GENERATED_ADMIN_TOKEN_SECRET}}},
            "gateway.auth.adminTokenSecretName and adminTokenSecretKey point at the sie-config admin token",
        ),
        (
            {
                "gateway": {
                    "auth": {
                        "adminTokenSecretName": GENERATED_READ_TOKEN_SECRET,
                        "adminTokenSecretKey": "SIE_CONFIG_READ_TOKEN",
                    }
                }
            },
            "gateway.auth.adminTokenSecretName and adminTokenSecretKey point at the sie-config read token",
        ),
    ],
)
def test_shared_token_secrets_fail_the_render(tmp_path: Path, values: dict, error: str) -> None:
    result = render_chart(tmp_path, {**L4_POOL, **values})
    assert result.returncode != 0
    assert error in result.stderr


def test_tokens_in_one_secret_under_different_keys_render(tmp_path: Path) -> None:
    auth = {
        "adminTokenSecretName": "sie-tokens",
        "adminTokenSecretKey": "admin",
        "readTokenSecretName": "sie-tokens",
        "readTokenSecretKey": "read",
    }
    values = {**L4_POOL, "config": {"auth": auth}, "gateway": {"auth": {"adminTokenSecretName": "sie-tokens"}}}
    docs = rendered_documents(tmp_path, values)
    assert env_entries(docs, "SIE_ADMIN_TOKEN") == {
        CONFIG_SERVICE: {"name": "sie-tokens", "key": "admin"},
        GATEWAY: {"name": "sie-tokens", "key": "SIE_ADMIN_TOKEN"},
    }
    read = {"name": "sie-tokens", "key": "read"}
    assert env_entries(docs, "SIE_CONFIG_READ_TOKEN") == {CONFIG_SERVICE: read}
    assert env_entries(docs, "SIE_CONFIG_SERVICE_TOKEN") == dict.fromkeys(CONFIG_CONSUMERS, read)


@pytest.mark.parametrize("name", ["SIE_ADMIN_TOKEN", "SIE_CONFIG_SERVICE_TOKEN"])
@pytest.mark.parametrize(
    ("path", "values"),
    [
        ("gateway.extraEnv", lambda entry: {"gateway": {"extraEnv": [entry]}}),
        (
            "workers.common.workerSidecar.extraEnv",
            lambda entry: {"workers": {"common": {"workerSidecar": {"extraEnv": [entry]}}}},
        ),
    ],
)
def test_config_credentials_cannot_be_overridden_through_extra_env(
    tmp_path: Path, name: str, path: str, values
) -> None:
    rendered = values({"name": name, "value": "override"})
    result = render_chart(
        tmp_path, {**L4_POOL, **rendered, "workers": {**L4_POOL["workers"], **rendered.get("workers", {})}}
    )
    assert result.returncode != 0
    assert f"{path} must not override chart-owned variable {name}" in result.stderr


def test_production_without_an_admin_token_names_the_read_only_effect(tmp_path: Path) -> None:
    auth = {"generateAdminToken": False, "readTokenSecretName": "operator-read"}
    result = render_chart(tmp_path, {"config": {"auth": auth}})
    assert result.returncode != 0
    assert "so it would refuse every config write and serve only reads that present the read token" in result.stderr
    assert "refuse every /v1/configs request" not in result.stderr


@pytest.mark.parametrize("key", [None, "", " "])
def test_empty_admin_token_key_fails_the_render(tmp_path: Path, key: str | None) -> None:
    result = render_chart(tmp_path, {**L4_POOL, "config": {"auth": {"adminTokenSecretKey": key}}})
    assert result.returncode != 0
    assert "config.auth.adminTokenSecretKey is empty" in result.stderr


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"config": {"auth": {"readTokenSecretKey": ""}}}, "config.auth.readTokenSecretKey is empty"),
        ({"config": {"auth": {"readTokenSecretKey": " "}}}, "config.auth.readTokenSecretKey is empty"),
        (
            {"gateway": {"auth": {"adminTokenSecretName": "gateway-admin", "adminTokenSecretKey": ""}}},
            "gateway.auth.adminTokenSecretKey is empty",
        ),
        (
            {"gateway": {"auth": {"adminTokenSecretName": "gateway-admin", "adminTokenSecretKey": " "}}},
            "gateway.auth.adminTokenSecretKey is empty",
        ),
    ],
)
def test_empty_token_key_fails_the_render(tmp_path: Path, values: dict, error: str) -> None:
    result = render_chart(tmp_path, {**L4_POOL, **values})
    assert result.returncode != 0
    assert error in result.stderr


@pytest.mark.parametrize("disabled", [False, "false", " False "])
@pytest.mark.parametrize("admin", [{}, {"adminTokenSecretName": "operator-admin"}])
def test_admin_token_without_a_read_token_fails_the_render(tmp_path: Path, admin: dict, disabled: object) -> None:
    result = render_chart(tmp_path, {**L4_POOL, "config": {"auth": {**admin, "generateReadToken": disabled}}})
    assert result.returncode != 0
    assert "sie-config has an admin token but no read token" in result.stderr


@pytest.mark.parametrize("telemetry", [{}, {"deploymentEnv": "production"}, {"deploymentEnv": " Prod "}])
def test_production_config_service_without_a_token_fails_the_render(tmp_path: Path, telemetry: dict) -> None:
    result = render_chart(tmp_path, {"config": {"auth": {"generateAdminToken": False}}, "telemetry": telemetry})
    assert result.returncode != 0
    assert "no admin token (config.auth.generateAdminToken=false" in result.stderr
    assert "or set config.auth.generateAdminToken=true so the chart generates one" in result.stderr


@pytest.mark.parametrize("deployment_env", ["prodcution", "test", " "])
def test_unrecognized_environment_without_a_token_fails_the_render(tmp_path: Path, deployment_env: str) -> None:
    values = {"config": {"auth": {"generateAdminToken": False}}, "telemetry": {"deploymentEnv": deployment_env}}
    result = render_chart(tmp_path, values)
    assert result.returncode != 0
    assert "would serve /v1/configs without authentication" in result.stderr
    assert "supported only for telemetry.deploymentEnv staging, development, or ci" in result.stderr


@pytest.mark.parametrize("deployment_env", ["staging", "development", "ci", " CI "])
def test_non_production_config_service_may_opt_out_of_the_token(tmp_path: Path, deployment_env: str) -> None:
    values = {
        **L4_POOL,
        "config": {"auth": {"generateAdminToken": False}},
        "telemetry": {"deploymentEnv": deployment_env},
    }
    docs = rendered_documents(tmp_path, values)
    assert token_secrets(docs) == {}
    assert env_entries(docs, "SIE_ADMIN_TOKEN") == {}
    assert env_entries(docs, "SIE_CONFIG_READ_TOKEN") == {}
    assert env_entries(docs, "SIE_CONFIG_SERVICE_TOKEN") == dict.fromkeys(CONFIG_CONSUMERS, "")


@pytest.mark.parametrize(
    ("telemetry", "expected"),
    [
        ({}, {"SIE_TELEMETRY_ENABLED": "1"}),
        ({"enabled": True}, {"SIE_TELEMETRY_ENABLED": "1"}),
        ({"enabled": False}, {"SIE_TELEMETRY_DISABLED": "1"}),
        ({"enabled": None}, {}),
    ],
)
def test_worker_telemetry_setting_is_explicit(tmp_path: Path, telemetry: dict, expected: dict[str, str]) -> None:
    (statefulset,) = worker_statefulsets(tmp_path, {**L4_POOL, "telemetry": telemetry})
    (worker,) = [c for c in statefulset["spec"]["template"]["spec"]["containers"] if c["name"] == "worker"]
    rendered = {
        env["name"]: env["value"]
        for env in worker["env"]
        if env["name"] in {"SIE_TELEMETRY_ENABLED", "SIE_TELEMETRY_DISABLED"}
    }
    assert rendered == expected


@pytest.mark.parametrize("setting", ["admin", "read"])
@pytest.mark.parametrize(
    ("token", "error"),
    [
        ("", "Secret generated exists but has no TOKEN_KEY key. If config.auth.{setting}TokenSecretKey was renamed"),
        ("a" * 31, "Secret generated holds a TOKEN_KEY value shorter than 32 characters"),
        ("a" * 32, None),
    ],
)
def test_reused_token_must_exist_and_have_at_least_32_characters(
    tmp_path: Path, setting: str, token: str, error: str | None
) -> None:
    chart = tmp_path / "helper-check"
    (chart / "templates").mkdir(parents=True)
    (chart / "Chart.yaml").write_text("apiVersion: v2\nname: helper-check\nversion: 0.1.0\n", encoding="utf-8")
    shutil.copy(ROOT / helm.CHART_DIR / "templates" / "_helpers.tpl", chart / "templates" / "_helpers.tpl")
    data = base64.b64encode(token.encode()).decode()
    (chart / "templates" / "check.yaml").write_text(
        '{{- include "sie-cluster.config.validateReusedToken" '
        f'(dict "name" "generated" "key" "TOKEN_KEY" "data" "{data}" '
        f'"keySetting" "config.auth.{setting}TokenSecretKey" '
        f'"nameSetting" "config.auth.{setting}TokenSecretName") }}}}\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["mise", "exec", "--", "helm", "template", "check", str(chart)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if error is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert error.format(setting=setting) in result.stderr


INGRESS_GUARD = "Refusing to render the gateway Ingress"
CLOUD_PRESETS = ["values-aws.yaml", "values-gke.yaml", "values-aks.yaml", "values-ack.yaml"]


def gateway_ingresses(documents: list[dict]) -> list[dict]:
    return [
        doc
        for doc in documents
        if doc["kind"] == "Ingress"
        and any(
            path["backend"]["service"]["name"].endswith("-gateway")
            for rule in doc["spec"]["rules"]
            for path in rule["http"]["paths"]
        )
    ]


def worker_network_policies(documents: list[dict]) -> list[dict]:
    return [
        doc
        for doc in documents
        if doc["kind"] == "NetworkPolicy" and doc["metadata"]["labels"].get("app.kubernetes.io/component") == "worker"
    ]


@pytest.mark.parametrize("preset", CLOUD_PRESETS)
def test_cloud_presets_do_not_publish_an_ingress(tmp_path: Path, preset: str) -> None:
    assert gateway_ingresses(rendered_documents(tmp_path, {}, (preset,))) == []


AUTH_GUARD = f"{INGRESS_GUARD}: nothing authenticates its requests"
TLS_GUARD = f"{INGRESS_GUARD} without TLS"
GATEWAY_AUTH = {"auth": {"mode": "static", "tokenSecretName": "sie-gateway-auth"}}
SCOPED_TLS = {"hosts": ["sie.example.com"], "tlsConfig": {"enabled": True, "mode": "byo", "secretName": "sie-tls"}}
UPSTREAM_TLS = {"hosts": ["sie.example.com"], "tlsConfig": {"enabled": False, "mode": "disabled"}}
OAUTH2_EDGE = {"enabled": True, "oauth2Proxy": {"oidcIssuerUrl": "https://issuer.example.com"}}


def ingress_values(ingress: dict | None = None, **values: dict) -> dict:
    return {**values, "ingress": {"enabled": True, **(ingress or {})}}


def render_error(tmp_path: Path, values: dict, overlays: tuple[str, ...] = ()) -> str:
    result = render_chart(tmp_path, values, overlays)
    assert result.returncode != 0, "render unexpectedly succeeded"
    return result.stderr


@pytest.mark.parametrize("preset", CLOUD_PRESETS)
def test_enabling_an_ingress_on_a_preset_requires_authentication(tmp_path: Path, preset: str) -> None:
    assert AUTH_GUARD in render_error(tmp_path, ingress_values(), (preset,))


@pytest.mark.parametrize("ingress", [{}, SCOPED_TLS, UPSTREAM_TLS, {"allowPlaintext": True}])
def test_unauthenticated_ingress_fails_whatever_its_host_and_tls(tmp_path: Path, ingress: dict) -> None:
    stderr = render_error(tmp_path, ingress_values(ingress))
    assert AUTH_GUARD in stderr
    assert "A hostname or TLS is not access control" in stderr
    assert "ingress.allowUnauthenticated=true" in stderr


@pytest.mark.parametrize(
    "values",
    [
        {"gateway": GATEWAY_AUTH},
        {"gateway": {"auth": {"mode": "token", "tokenSecretName": "sie-gateway-auth"}}},
        {"auth": OAUTH2_EDGE},
        {"ingress": {"allowUnauthenticated": True}},
    ],
)
def test_authenticated_ingress_without_tls_fails(tmp_path: Path, values: dict) -> None:
    values = {**values, "ingress": {"enabled": True, **values.get("ingress", {})}}
    stderr = render_error(tmp_path, values)
    assert TLS_GUARD in stderr
    assert "ingress.allowPlaintext=true" in stderr


@pytest.mark.parametrize(
    "ingress",
    [
        SCOPED_TLS,
        {"host": "sie.example.com", "tls": {"enabled": True}},
        UPSTREAM_TLS,
        {"allowPlaintext": True},
    ],
)
def test_authenticated_ingress_renders_with_tls_or_the_plaintext_opt_in(tmp_path: Path, ingress: dict) -> None:
    assert len(gateway_ingresses(rendered_documents(tmp_path, ingress_values(ingress, gateway=GATEWAY_AUTH)))) == 1


def test_tls_without_a_host_does_not_count_as_tls(tmp_path: Path) -> None:
    values = ingress_values({"tlsConfig": {"enabled": True, "mode": "byo"}}, gateway=GATEWAY_AUTH)
    assert TLS_GUARD in render_error(tmp_path, values)


def test_both_opt_ins_keep_the_previous_catch_all_ingress(tmp_path: Path) -> None:
    values = ingress_values({"allowUnauthenticated": True, "allowPlaintext": True})
    (ingress,) = gateway_ingresses(rendered_documents(tmp_path, values))
    assert [rule.get("host") for rule in ingress["spec"]["rules"]] == [None]
    assert "tls" not in ingress["spec"]


def test_scoped_ingress_keeps_its_host_and_tls(tmp_path: Path) -> None:
    (ingress,) = gateway_ingresses(rendered_documents(tmp_path, ingress_values(SCOPED_TLS, gateway=GATEWAY_AUTH)))
    assert [rule["host"] for rule in ingress["spec"]["rules"]] == ["sie.example.com"]
    assert ingress["spec"]["tls"] == [{"hosts": ["sie.example.com"], "secretName": "sie-tls"}]


@pytest.mark.parametrize("field", ["allowUnauthenticated", "allowPlaintext"])
@pytest.mark.parametrize("value", ["false", "true", 1])
def test_ingress_opt_ins_accept_only_booleans(tmp_path: Path, field: str, value: object) -> None:
    stderr = render_error(tmp_path, ingress_values({field: value, **SCOPED_TLS}, gateway=GATEWAY_AUTH))
    assert f"ingress.{field} must be a boolean" in stderr


@pytest.mark.parametrize("class_name", ["alb", "traefik", "nginx-internal", ""])
def test_offline_render_credits_the_oauth2_edge_only_for_class_nginx(tmp_path: Path, class_name: str) -> None:
    values = ingress_values({"className": class_name, **SCOPED_TLS}, auth=OAUTH2_EDGE)
    stderr = render_error(tmp_path, values)
    assert "cannot inspect IngressClasses" in stderr
    assert "accepts only ingress.className=nginx" in stderr


def test_oauth2_edge_on_ingress_nginx_counts_as_authentication(tmp_path: Path) -> None:
    values = ingress_values({"className": "nginx", **SCOPED_TLS}, auth=OAUTH2_EDGE)
    (ingress,) = gateway_ingresses(rendered_documents(tmp_path, values))
    assert "nginx.ingress.kubernetes.io/auth-url" in ingress["metadata"]["annotations"]


@pytest.mark.parametrize(
    "extra_env",
    [
        [{"name": "SIE_AUTH_MODE", "value": "none"}, {"name": "SIE_AUTH_TOKEN", "value": ""}],
        [{"name": "SIE_AUTH_MODE", "valueFrom": {"configMapKeyRef": {"name": "gateway-auth", "key": "mode"}}}],
    ],
)
def test_extra_env_auth_mode_override_is_not_credited(tmp_path: Path, extra_env: list[dict]) -> None:
    values = ingress_values(SCOPED_TLS, gateway={**GATEWAY_AUTH, "extraEnv": extra_env})
    assert AUTH_GUARD in render_error(tmp_path, values)


def test_extra_env_auth_mode_enabling_auth_is_credited(tmp_path: Path) -> None:
    extra_env = [
        {"name": "SIE_AUTH_MODE", "value": "token"},
        {"name": "SIE_AUTH_TOKENS", "valueFrom": {"secretKeyRef": {"name": "gateway-auth", "key": "tokens"}}},
    ]
    values = ingress_values(SCOPED_TLS, gateway={"extraEnv": extra_env})
    assert len(gateway_ingresses(rendered_documents(tmp_path, values))) == 1


@pytest.mark.parametrize("mode", ["Static", "static ", "bearer"])
def test_unsupported_gateway_auth_mode_fails(tmp_path: Path, mode: str) -> None:
    stderr = render_error(tmp_path, {"gateway": {"auth": {"mode": mode, "tokenSecretName": "sie-gateway-auth"}}})
    assert "is not supported" in stderr


@pytest.mark.parametrize("mode", ["static", "token"])
def test_gateway_token_auth_needs_a_token_source(tmp_path: Path, mode: str) -> None:
    assert "needs tokens" in render_error(tmp_path, {"gateway": {"auth": {"mode": mode}}})


@pytest.mark.parametrize("service_type", ["LoadBalancer", "NodePort"])
def test_external_gateway_service_requires_auth_or_opt_in(tmp_path: Path, service_type: str) -> None:
    stderr = render_error(tmp_path, {"gateway": {"service": {"type": service_type, "allowPlaintext": True}}})
    assert f"Refusing to render gateway.service.type={service_type} without gateway auth" in stderr
    service = {"type": service_type, "allowPlaintext": True}
    rendered_documents(tmp_path, {"gateway": {**GATEWAY_AUTH, "service": service}})
    rendered_documents(tmp_path, {"gateway": {"service": {**service, "allowUnauthenticated": True}}})


@pytest.mark.parametrize("service_type", ["LoadBalancer", "NodePort"])
def test_external_gateway_service_requires_a_plaintext_acknowledgement(tmp_path: Path, service_type: str) -> None:
    stderr = render_error(tmp_path, {"gateway": {**GATEWAY_AUTH, "service": {"type": service_type}}})
    assert f"Refusing to render gateway.service.type={service_type} without an explicit TLS decision" in stderr
    assert "gateway.service.allowPlaintext=true" in stderr


@pytest.mark.parametrize("field", ["allowUnauthenticated", "allowPlaintext"])
def test_gateway_service_opt_ins_accept_only_booleans(tmp_path: Path, field: str) -> None:
    service = {"type": "LoadBalancer", "allowUnauthenticated": True, "allowPlaintext": True, field: "false"}
    assert f"gateway.service.{field} must be a boolean" in render_error(tmp_path, {"gateway": {"service": service}})


def test_token_secret_without_token_auth_fails(tmp_path: Path) -> None:
    stderr = render_error(tmp_path, {"gateway": {"auth": {"mode": "none", "tokenSecretName": "sie-gateway-auth"}}})
    assert "gateway.auth.tokenSecretName is set but gateway auth mode is none" in stderr


def test_ip_only_self_signed_certificate_is_not_ingress_tls(tmp_path: Path) -> None:
    ingress = {
        "tlsConfig": {"enabled": True, "mode": "self-signed", "selfSigned": {"leaf": {"ipAddresses": ["10.0.0.10"]}}}
    }
    stderr = render_error(tmp_path, ingress_values(ingress, gateway=GATEWAY_AUTH))
    assert TLS_GUARD in stderr
    assert "IP-only self-signed certificate is not supported" in stderr


MCP_EDGE = {"enabled": True, "ingress": {"enabled": True, "host": "mcp.example.com"}}


def mcp_edge_ingresses(documents: list[dict]) -> list[dict]:
    return [
        doc
        for doc in documents
        if doc["kind"] == "Ingress" and doc["metadata"]["labels"].get("app.kubernetes.io/component") == "mcp-edge"
    ]


def test_mcp_edge_ingress_without_tls_fails(tmp_path: Path) -> None:
    stderr = render_error(tmp_path, {"mcpEdge": MCP_EDGE})
    assert "Refusing to render the MCP edge Ingress without TLS" in stderr
    assert "mcpEdge.ingress.allowPlaintext=true" in stderr


@pytest.mark.parametrize(
    "values",
    [
        {"ingress": {"tlsConfig": {"enabled": True, "mode": "byo"}}},
        {"ingress": {"tlsConfig": {"enabled": False, "mode": "disabled"}}},
        {"mcpEdge": {"ingress": {"allowPlaintext": True}}},
    ],
)
def test_mcp_edge_ingress_renders_with_tls_or_the_plaintext_opt_in(tmp_path: Path, values: dict) -> None:
    mcp_edge = {**MCP_EDGE, "ingress": {**MCP_EDGE["ingress"], **values.get("mcpEdge", {}).get("ingress", {})}}
    documents = rendered_documents(tmp_path, {**values, "mcpEdge": mcp_edge})
    assert len(mcp_edge_ingresses(documents)) == 1


def test_mcp_edge_ingress_refuses_self_signed_mode_without_its_certificate(tmp_path: Path) -> None:
    tls = {"enabled": True, "mode": "self-signed", "selfSigned": {"leaf": {"dnsNames": ["mcp.example.com"]}}}
    stderr = render_error(tmp_path, {"mcpEdge": MCP_EDGE, "ingress": {"tlsConfig": tls}})
    assert "nothing issues the MCP edge certificate" in stderr


def test_mcp_edge_plaintext_opt_in_accepts_only_booleans(tmp_path: Path) -> None:
    mcp_edge = {**MCP_EDGE, "ingress": {**MCP_EDGE["ingress"], "allowPlaintext": "false"}}
    assert "mcpEdge.ingress.allowPlaintext must be a boolean" in render_error(tmp_path, {"mcpEdge": mcp_edge})


@pytest.mark.parametrize("service_type", ["LoadBalancer", "NodePort"])
def test_config_service_must_stay_cluster_ip(tmp_path: Path, service_type: str) -> None:
    stderr = render_error(tmp_path, {"config": {"service": {"type": service_type}}})
    assert f"config.service.type={service_type} is not supported" in stderr


def test_worker_network_policy_is_off_by_default(tmp_path: Path) -> None:
    assert (
        worker_network_policies(rendered_documents(tmp_path, {"workers": {"pools": {"l4": {"enabled": True}}}})) == []
    )


def test_ha_overlay_admits_only_gateway_pods_to_every_worker_port(tmp_path: Path) -> None:
    values = {"workers": {"pools": {"l4": {"enabled": True, "gpu": {"count": 2}}}}}
    documents = rendered_documents(tmp_path, values, ("values-aws.yaml", "values-ha.yaml"))
    (policy,) = worker_network_policies(documents)
    (gateway,) = [
        doc
        for doc in documents
        if doc["kind"] == "Deployment" and doc["metadata"]["labels"].get("app.kubernetes.io/component") == "gateway"
    ]
    workers = [
        doc for doc in documents if doc["kind"] == "StatefulSet" and doc["metadata"]["name"].startswith("sie-worker-")
    ]
    assert workers
    selector = policy["spec"]["podSelector"]["matchLabels"]
    for worker in workers:
        pod_labels = worker["spec"]["template"]["metadata"]["labels"]
        assert selector.items() <= pod_labels.items()
    worker_ports = {
        port["containerPort"]
        for worker in workers
        for container in worker["spec"]["template"]["spec"]["containers"]
        if container["name"] != "worker-sidecar"
        for port in container.get("ports", [])
    }
    assert policy["spec"]["policyTypes"] == ["Ingress"]
    (rule,) = policy["spec"]["ingress"]
    (peer,) = rule["from"]
    assert peer["podSelector"]["matchLabels"].items() <= gateway["spec"]["template"]["metadata"]["labels"].items()
    assert {port["port"] for port in rule["ports"]} == worker_ports == {8080, 8081}


def test_worker_network_policy_appends_extra_ingress_rules(tmp_path: Path) -> None:
    extra = {
        "from": [{"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "bench"}}}],
        "ports": [{"port": 8080, "protocol": "TCP"}],
    }
    values = {
        "workers": {
            "networkPolicy": {"enabled": True, "extraIngress": [extra]},
            "pools": {"l4": {"enabled": True}},
        }
    }
    (policy,) = worker_network_policies(rendered_documents(tmp_path, values))
    assert policy["spec"]["ingress"][1] == extra


@pytest.mark.parametrize(
    "rule",
    [
        {"from": [{"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "bench"}}}]},
        {"ports": [{"port": 8080, "protocol": "TCP"}]},
    ],
)
def test_worker_network_policy_rejects_rules_without_from_or_ports(tmp_path: Path, rule: dict) -> None:
    values = {
        "workers": {"networkPolicy": {"enabled": True, "extraIngress": [rule]}, "pools": {"l4": {"enabled": True}}}
    }
    assert "workers.networkPolicy.extraIngress[0] needs a non-empty from and ports" in render_error(tmp_path, values)


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"from": [{}], "ports": [{"port": 8080}]}, "extraIngress[0].from[0] admits every source"),
        (
            {"from": [{"namespaceSelector": {}}], "ports": [{"port": 8080}]},
            "extraIngress[0].from[0] admits every source",
        ),
        (
            {"from": [{"namespaceSelector": {"matchLabels": {"team": "bench"}}}], "ports": [{"protocol": "TCP"}]},
            "extraIngress[0].ports[0] admits every port",
        ),
    ],
)
def test_worker_network_policy_rejects_wildcard_peers_and_ports(tmp_path: Path, rule: dict, message: str) -> None:
    values = {
        "workers": {"networkPolicy": {"enabled": True, "extraIngress": [rule]}, "pools": {"l4": {"enabled": True}}}
    }
    assert message in render_error(tmp_path, values)


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"from": [{"ipBlock": {"cidr": "0.0.0.0/0"}}], "ports": [{"port": 8080}]}, "admits every address"),
        ({"from": [{"ipBlock": {"cidr": "::/0"}}], "ports": [{"port": 8080}]}, "admits every address"),
        (
            {"from": [{"ipBlock": {"cidr": "10.0.0.0/8"}}], "ports": [{"port": 1, "endPort": 65535}]},
            "spans nearly every port",
        ),
    ],
)
def test_worker_network_policy_rejects_full_range_sources_and_ports(tmp_path: Path, rule: dict, message: str) -> None:
    values = {
        "workers": {"networkPolicy": {"enabled": True, "extraIngress": [rule]}, "pools": {"l4": {"enabled": True}}}
    }
    assert message in render_error(tmp_path, values)


def test_worker_network_policy_accepts_a_scoped_ip_block_and_port_range(tmp_path: Path) -> None:
    rule = {
        "from": [{"ipBlock": {"cidr": "10.0.0.0/8"}}],
        "ports": [{"port": 8080, "endPort": 8081, "protocol": "TCP"}],
    }
    values = {
        "workers": {"networkPolicy": {"enabled": True, "extraIngress": [rule]}, "pools": {"l4": {"enabled": True}}}
    }
    (policy,) = worker_network_policies(rendered_documents(tmp_path, values))
    assert policy["spec"]["ingress"][1] == rule


INGRESS_CLASS_LOOKUP = 'lookup "networking.k8s.io/v1" "IngressClass" "" ""'


def ingress_class(name: str, controller: str, default: bool = False) -> dict:
    annotations = {"ingressclass.kubernetes.io/is-default-class": "true"} if default else {}
    return {"metadata": {"name": name, "annotations": annotations}, "spec": {"controller": controller}}


def render_with_ingress_classes(tmp_path: Path, values: dict, classes: list[dict]) -> subprocess.CompletedProcess[str]:
    chart = tmp_path / "chart"
    shutil.copytree(ROOT / helm.CHART_DIR, chart, symlinks=True)
    helpers = chart / "templates" / "_helpers.tpl"
    source = helpers.read_text(encoding="utf-8")
    assert source.count(INGRESS_CLASS_LOOKUP) == 1
    helpers.write_text(
        source.replace(INGRESS_CLASS_LOOKUP, "(default (dict) .Values.ingressClassFixture)"), encoding="utf-8"
    )
    values_file = tmp_path / "overrides.yaml"
    values_file.write_text(yaml.safe_dump({**values, "ingressClassFixture": {"items": classes}}), encoding="utf-8")
    return subprocess.run(
        [
            "mise",
            "exec",
            "--",
            "helm",
            "template",
            "sie",
            str(chart),
            "--namespace",
            "sie",
            *helm.validation_args(["-f", str(values_file)]),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


INGRESS_NGINX = "k8s.io/ingress-nginx"
NGINX_INC = "nginx.org/ingress-controller"


@pytest.mark.parametrize(
    ("class_name", "classes", "message"),
    [
        (
            "",
            [ingress_class("a-f5", NGINX_INC, default=True), ingress_class("z-nginx", INGRESS_NGINX, default=True)],
            'IngressClass "a-f5" (a default IngressClass',
        ),
        ("nginx", [ingress_class("nginx", NGINX_INC)], 'IngressClass "nginx" uses controller "nginx.org'),
        ("missing", [ingress_class("nginx", INGRESS_NGINX)], 'no IngressClass named "missing"'),
        ("", [ingress_class("nginx", INGRESS_NGINX)], "no IngressClass is marked as the cluster default"),
        (
            "internal",
            [ingress_class("internal", "k8s.io/internal-ingress-nginx")],
            "not in auth.ingress.acceptedControllers",
        ),
    ],
)
def test_oauth2_edge_rejects_ingress_classes_without_an_accepted_controller(
    tmp_path: Path, class_name: str, classes: list[dict], message: str
) -> None:
    values = ingress_values({"className": class_name, **SCOPED_TLS}, auth=OAUTH2_EDGE)
    result = render_with_ingress_classes(tmp_path, values, classes)
    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize(
    ("class_name", "classes", "accepted"),
    [
        ("", [ingress_class("a", INGRESS_NGINX, default=True), ingress_class("b", INGRESS_NGINX, default=True)], None),
        ("nginx-internal", [ingress_class("nginx-internal", INGRESS_NGINX)], None),
        (
            "internal",
            [ingress_class("internal", "k8s.io/internal-ingress-nginx")],
            [INGRESS_NGINX, "k8s.io/internal-ingress-nginx"],
        ),
    ],
)
def test_oauth2_edge_accepts_ingress_classes_with_an_accepted_controller(
    tmp_path: Path, class_name: str, classes: list[dict], accepted: list[str] | None
) -> None:
    auth = {**OAUTH2_EDGE, **({"ingress": {"acceptedControllers": accepted}} if accepted else {})}
    values = ingress_values({"className": class_name, **SCOPED_TLS}, auth=auth)
    result = render_with_ingress_classes(tmp_path, values, classes)
    assert result.returncode == 0, result.stderr
    assert len(gateway_ingresses([doc for doc in yaml.safe_load_all(result.stdout) if doc])) == 1


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"from": [{"ipBlock": {"cidr": "0::/0"}}], "ports": [{"port": 8080}]}, "admits every address"),
        ({"from": [{"ipBlock": {"cidr": "1.2.3.4/0"}}], "ports": [{"port": 8080}]}, "admits every address"),
        (
            {"from": [{"ipBlock": {"cidr": "10.0.0.0/8"}}], "ports": [{"port": 2, "endPort": 65535}]},
            "spans nearly every port",
        ),
    ],
)
def test_worker_network_policy_rejects_prefix_zero_and_near_full_port_ranges(
    tmp_path: Path, rule: dict, message: str
) -> None:
    values = {
        "workers": {"networkPolicy": {"enabled": True, "extraIngress": [rule]}, "pools": {"l4": {"enabled": True}}}
    }
    assert message in render_error(tmp_path, values)


NATS_SERVER_CONFIG_FIXTURE = ROOT / "tools/ci/fixtures/sie-cluster-nats.conf"
NATS_L4_POOL = {"workers": {"pools": {"l4": {"enabled": True}}}}
NATS_CLIENTS = {
    "config": ("sie-sie-cluster-config", "config"),
    "gateway": ("sie-sie-cluster-gateway", "gateway"),
    "worker": ("sie-sie-cluster-worker-l4-default", "worker-sidecar"),
}


def render_nats_chart(tmp_path: Path, values: dict, *extra: str) -> subprocess.CompletedProcess[str]:
    values_file = tmp_path / "nats-values.yaml"
    values_file.write_text(yaml.safe_dump(values), encoding="utf-8")
    return subprocess.run(
        [
            "mise",
            "exec",
            "--",
            "helm",
            "template",
            "sie",
            str(helm.CHART_DIR),
            "--namespace",
            "sie",
            *helm.validation_args([*extra, "-f", str(values_file)]),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def nats_documents(tmp_path: Path, values: dict, *extra: str) -> list[dict]:
    result = render_nats_chart(tmp_path, values, *extra)
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def workload_env(docs: list[dict], workload: str, container: str) -> dict[str, dict]:
    (doc,) = [
        doc for doc in docs if doc["kind"] in {"Deployment", "StatefulSet"} and doc["metadata"]["name"] == workload
    ]
    (spec,) = [spec for spec in doc["spec"]["template"]["spec"]["containers"] if spec["name"] == container]
    return {env["name"]: env for env in spec.get("env", [])}


def nats_server_config(docs: list[dict]) -> str:
    (config_map,) = [doc for doc in docs if doc["kind"] == "ConfigMap" and doc["metadata"]["name"] == "sie-nats-config"]
    return config_map["data"]["nats.conf"]


def nats_auth_secrets(docs: list[dict]) -> dict[str, dict]:
    return {
        doc["metadata"]["name"]: doc
        for doc in docs
        if doc["kind"] == "Secret" and doc["metadata"]["name"].startswith("sie-nats-auth-")
    }


def nats_client_credentials(docs: list[dict]) -> dict[str, tuple[str, dict]]:
    credentials = {}
    for component, (workload, container) in NATS_CLIENTS.items():
        env = workload_env(docs, workload, container)
        if "SIE_NATS_PASSWORD" in env:
            credentials[component] = (
                env["SIE_NATS_USER"]["value"],
                env["SIE_NATS_PASSWORD"]["valueFrom"]["secretKeyRef"],
            )
    return credentials


def test_nats_server_config_fixture_matches_the_chart(tmp_path: Path) -> None:
    docs = nats_documents(tmp_path, {})
    assert nats_server_config(docs) == NATS_SERVER_CONFIG_FIXTURE.read_text(encoding="utf-8"), (
        "tools/ci/fixtures/sie-cluster-nats.conf must equal the nats.conf the chart renders by default; "
        "the gateway, sidecar, CPU-stack, and permission tests run nats-server with it"
    )


@pytest.mark.parametrize("auth", [{}, {"nats": {"auth": None}}, {"nats": {"auth": {"enabled": None}}}])
def test_nats_authentication_is_on_by_default(tmp_path: Path, auth: dict) -> None:
    docs = nats_documents(tmp_path, {**NATS_L4_POOL, **auth})
    secrets = nats_auth_secrets(docs)
    assert sorted(secrets) == ["sie-nats-auth-config", "sie-nats-auth-gateway", "sie-nats-auth-worker"]
    for secret in secrets.values():
        assert secret["metadata"]["annotations"] == {"helm.sh/resource-policy": "keep"}
        assert re.fullmatch(r"[A-Za-z][A-Za-z0-9]{47}", base64.b64decode(secret["data"]["password"]).decode())
    assert nats_client_credentials(docs) == {
        component: (f"sie-{component}", {"name": f"sie-nats-auth-{component}", "key": "password"})
        for component in NATS_CLIENTS
    }
    worker = workload_env(docs, "sie-sie-cluster-worker-l4-default", "worker")
    assert not {"SIE_NATS_URL", "SIE_NATS_USER", "SIE_NATS_PASSWORD"} & set(worker)
    server_env = workload_env(docs, "sie-nats", "nats")
    for component in NATS_CLIENTS:
        assert server_env[f"SIE_NATS_AUTH_{component.upper()}_PASSWORD"]["valueFrom"]["secretKeyRef"] == {
            "name": f"sie-nats-auth-{component}",
            "key": "password",
        }
    config = nats_server_config(docs)
    for component in NATS_CLIENTS:
        assert f'"user": "sie-{component}"' in config
        assert f'"password": $SIE_NATS_AUTH_{component.upper()}_PASSWORD' in config
    assert "no_auth_user" not in config
    assert not [doc for doc in docs if "nats-box" in doc["metadata"]["name"]]


KUBERNETES_WRITE_VERBS = {"create", "update", "patch", "delete", "deletecollection", "*"}


def role_writes(role: dict, resource: str) -> bool:
    return any(
        {resource, "*"} & set(rule.get("resources", [])) and KUBERNETES_WRITE_VERBS & set(rule.get("verbs", []))
        for rule in role.get("rules", [])
    )


@pytest.mark.parametrize("overlays", [(), ("values-ha.yaml",)])
def test_only_gateway_pods_hold_a_token_that_can_write_configmaps(tmp_path: Path, overlays: tuple[str, ...]) -> None:
    overlay_args = [arg for overlay in overlays for arg in ("-f", str(helm.CHART_DIR / overlay))]
    docs = nats_documents(tmp_path, {**NATS_L4_POOL, "mcpEdge": {"enabled": True}}, *overlay_args)
    assert bool(overlays) == any(
        doc["kind"] == "NetworkPolicy" and doc["metadata"]["name"].endswith("-worker") for doc in docs
    )
    roles = {doc["metadata"]["name"]: doc for doc in docs if doc["kind"] == "Role"}
    assert not [name for name, role in roles.items() if role_writes(role, "secrets")]
    writers = {
        subject["name"]
        for doc in docs
        if doc["kind"] == "RoleBinding" and role_writes(roles[doc["roleRef"]["name"]], "configmaps")
        for subject in doc["subjects"]
        if subject["kind"] == "ServiceAccount"
    }
    assert writers == {"sie-server"}
    pods = {
        doc["metadata"]["name"]: doc["spec"]["template"]["spec"]
        for doc in docs
        if doc["kind"] in {"Deployment", "StatefulSet", "DaemonSet", "Job"}
    }
    assert {"sie-sie-cluster-config", "sie-sie-cluster-mcp", "sie-sie-cluster-worker-l4-default"} <= set(pods)
    mounting = sorted(
        name
        for name, pod in pods.items()
        if pod.get("serviceAccountName") in writers and pod.get("automountServiceAccountToken", True)
    )
    assert mounting == ["sie-sie-cluster-gateway"]


def test_nats_authentication_opt_out(tmp_path: Path) -> None:
    docs = nats_documents(tmp_path, {**NATS_L4_POOL, "nats": {"auth": {"enabled": False}}})
    assert nats_auth_secrets(docs) == {}
    assert nats_client_credentials(docs) == {}
    assert not [name for name in workload_env(docs, "sie-nats", "nats") if name.startswith("SIE_NATS")]
    assert "authorization" not in nats_server_config(docs)


def test_nats_operator_secrets_are_used_unchanged(tmp_path: Path) -> None:
    existing = {"config": "ops-config", "gateway": "ops-gateway", "worker": "ops-worker"}
    docs = nats_documents(tmp_path, {**NATS_L4_POOL, "nats": {"auth": {"existingSecrets": existing}}})
    assert nats_auth_secrets(docs) == {}
    assert nats_client_credentials(docs) == {
        component: (f"sie-{component}", {"name": name, "key": "password"}) for component, name in existing.items()
    }
    server_env = workload_env(docs, "sie-nats", "nats")
    for component, name in existing.items():
        assert server_env[f"SIE_NATS_AUTH_{component.upper()}_PASSWORD"]["valueFrom"]["secretKeyRef"]["name"] == name


def test_nats_cluster_routes_authenticate(tmp_path: Path) -> None:
    docs = nats_documents(tmp_path, {}, "-f", str(ROOT / helm.CHART_DIR / "values-ha.yaml"))
    assert "sie-nats-auth-route" in nats_auth_secrets(docs)
    config = nats_server_config(docs)
    assert '"password": $SIE_NATS_AUTH_ROUTE_PASSWORD' in config
    assert '"routes": $SIE_NATS_ROUTES' in config
    env_names = list(workload_env(docs, "sie-nats", "nats"))
    assert env_names.index("SIE_NATS_AUTH_ROUTE_PASSWORD") < env_names.index("SIE_NATS_ROUTES")
    routes = workload_env(docs, "sie-nats", "nats")["SIE_NATS_ROUTES"]["value"]
    assert routes == (
        "["
        + ",".join(
            f"nats://sie-route:$(SIE_NATS_AUTH_ROUTE_PASSWORD)@sie-nats-{i}.sie-nats-headless:6222" for i in range(3)
        )
        + "]"
    )


@pytest.mark.parametrize(
    ("values", "extra"),
    [
        ({"nats": {"config": {"merge": {"sieNatsAuth": None}}}}, ()),
        ({"nats": {"container": {"env": {"sieNatsAuth": None}}}}, ()),
        (
            {"nats": {"config": {"cluster": {"merge": {"sieNatsAuth": None}}}}},
            ("-f", str(ROOT / helm.CHART_DIR / "values-ha.yaml")),
        ),
    ],
)
def test_nats_values_without_the_server_wiring_fail_the_render(
    tmp_path: Path, values: dict, extra: tuple[str, ...]
) -> None:
    result = render_nats_chart(tmp_path, values, *extra)
    assert result.returncode != 0
    assert "lack the chart's server wiring" in result.stderr
    assert "--reset-then-reuse-values" in result.stderr


def test_external_nats_needs_operator_secrets(tmp_path: Path) -> None:
    external = {"install": False, "url": "nats://external-nats:4222"}
    result = render_nats_chart(tmp_path, {**NATS_L4_POOL, "nats": external})
    assert result.returncode != 0
    assert "needs nats.auth.existingSecrets.gateway" in result.stderr

    existing = {"config": "ops-config", "gateway": "ops-gateway", "worker": "ops-worker"}
    docs = nats_documents(tmp_path, {**NATS_L4_POOL, "nats": {**external, "auth": {"existingSecrets": existing}}})
    assert nats_auth_secrets(docs) == {}
    assert nats_client_credentials(docs) == {
        component: (f"sie-{component}", {"name": name, "key": "password"}) for component, name in existing.items()
    }

    docs = nats_documents(tmp_path, {**NATS_L4_POOL, "nats": {**external, "auth": {"enabled": False}}})
    assert nats_client_credentials(docs) == {}


def test_nats_anonymous_upgrade_aid(tmp_path: Path) -> None:
    docs = nats_documents(tmp_path, {"nats": {"auth": {"allowAnonymous": True}}})
    config = nats_server_config(docs)
    assert '"no_auth_user": "sie-anonymous"' in config
    assert '"user": "sie-anonymous"' in config

    external = {"install": False, "url": "nats://external-nats:4222", "auth": {"allowAnonymous": True}}
    result = render_nats_chart(tmp_path, {"nats": external})
    assert result.returncode != 0
    assert "allowAnonymous applies only to the bundled NATS server" in result.stderr


@pytest.mark.parametrize(
    ("password", "error"),
    [
        ("", "Secret nats exists but has no password key"),
        ("a" * 31, "shorter than 32 characters"),
        ("a" * 31 + "$", "not letters and digits starting with a letter"),
        ("1" + "a" * 31, "not letters and digits starting with a letter"),
        ("a" * 32, None),
    ],
)
def test_reused_nats_password_must_be_32_letters_and_digits(tmp_path: Path, password: str, error: str | None) -> None:
    chart = tmp_path / "helper-check"
    (chart / "templates").mkdir(parents=True)
    (chart / "Chart.yaml").write_text("apiVersion: v2\nname: helper-check\nversion: 0.1.0\n", encoding="utf-8")
    shutil.copy(ROOT / helm.CHART_DIR / "templates" / "_nats-auth.tpl", chart / "templates" / "_nats-auth.tpl")
    data = base64.b64encode(password.encode()).decode()
    (chart / "templates" / "check.yaml").write_text(
        f'{{{{- include "sie-cluster.nats.validateReusedPassword" (dict "name" "nats" "data" "{data}") }}}}\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["mise", "exec", "--", "helm", "template", "check", str(chart)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if error is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert error in result.stderr


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _route_count(monitor_port: int) -> int:
    with urllib.request.urlopen(f"http://127.0.0.1:{monitor_port}/routez", timeout=2) as response:
        return len({route["remote_id"] for route in json.load(response).get("routes") or []})


def test_rendered_nats_cluster_forms_only_with_route_credentials(tmp_path: Path) -> None:
    binary = shutil.which("nats-server")
    if binary is None:
        pytest.skip("nats-server is not on PATH")
    docs = nats_documents(tmp_path, {}, "-f", str(ROOT / helm.CHART_DIR / "values-ha.yaml"))
    rendered = nats_server_config(docs)
    assert rendered.count('"store_dir": "/data"') == 1
    assert rendered.count('"port": 6222') == 1
    route_password = "RoutePassword0123456789abcdefghijk"
    cluster_ports = [_free_port() for _ in range(3)]
    monitor_ports = [_free_port() for _ in range(3)]
    routes = ",".join(f"nats://sie-route:{route_password}@127.0.0.1:{port}" for port in cluster_ports)
    servers = []

    def start(index: int, *args: str, env: dict[str, str]) -> subprocess.Popen[bytes]:
        work = tmp_path / f"node-{index}"
        work.mkdir()
        if env:
            config = work / "nats.conf"
            node_config = rendered.replace('"store_dir": "/data"', f'"store_dir": {json.dumps(str(work / "js"))}')
            node_config = node_config.replace('"port": 6222', f'"port": {cluster_ports[index]}')
            config.write_text(node_config, encoding="utf-8")
            args = ("-c", str(config), *args)
        else:
            args = ("-sd", str(work / "js"), *args)
        server = subprocess.Popen(  # noqa: S603
            [binary, *args, "-a", "127.0.0.1", "-p", str(_free_port()), "-P", str(work / "pid")],
            env={**os.environ, **env},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        servers.append(server)
        return server

    try:
        for index in range(3):
            start(
                index,
                "-m",
                str(monitor_ports[index]),
                env={
                    "SERVER_NAME": f"node-{index}",
                    "SIE_NATS_AUTH_CONFIG_PASSWORD": "c" * 32,
                    "SIE_NATS_AUTH_GATEWAY_PASSWORD": "g" * 32,
                    "SIE_NATS_AUTH_WORKER_PASSWORD": "w" * 32,
                    "SIE_NATS_AUTH_ROUTE_PASSWORD": route_password,
                    "SIE_NATS_ROUTES": f"[{routes}]",
                },
            )
        deadline = time.monotonic() + 20
        while True:
            try:
                if all(_route_count(port) == 2 for port in monitor_ports):
                    break
            except OSError:
                pass
            assert time.monotonic() < deadline, "the three rendered nodes did not route to each other"
            time.sleep(0.2)

        start(
            3,
            "--cluster_name",
            "sie-nats",
            "-cluster",
            f"nats://127.0.0.1:{_free_port()}",
            "-routes",
            f"nats://127.0.0.1:{cluster_ports[0]}",
            env={},
        )
        time.sleep(3)
        assert _route_count(monitor_ports[0]) == 2, "a route without credentials joined the cluster"
    finally:
        for server in servers:
            server.terminate()
        for server in servers:
            server.wait(timeout=10)


UPSTREAMS_FIXTURE = ROOT / "tools/ci/fixtures/helm-upstreams.yaml"
UPSTREAMS_SECRET = "sie-sie-cluster-upstreams"
REMOTE_WORKER = ("StatefulSet", "sie-sie-cluster-worker-remote-remote", "worker")
REMOTE_SERVICE_ACCOUNT = "sie-sie-cluster-worker-remote"
CREDENTIAL_CANARY = "sk-canary-2f7c9e04b1d3a685"  # gitleaks:allow


def upstreams_fixture() -> dict:
    return yaml.safe_load(UPSTREAMS_FIXTURE.read_text(encoding="utf-8"))


def remote_pool_values(**pools: dict) -> dict:
    return {
        "upstreams": upstreams_fixture()["values"],
        "workers": {"networkPolicy": {"enabled": True}, "pools": {"remote": {"enabled": True}, **pools}},
    }


def with_worker_network_policy(values: dict) -> dict:
    return {**values, "workers": {**values.get("workers", {}), "networkPolicy": {"enabled": True}}}


def pod_specs(docs: list[dict]) -> list[tuple[dict, dict]]:
    specs = []
    for doc in docs:
        spec = doc.get("spec") or {}
        if doc["kind"] == "Pod":
            specs.append((doc, spec))
        elif doc["kind"] == "CronJob":
            specs.append((doc, spec["jobTemplate"]["spec"]["template"]["spec"]))
        elif isinstance(spec.get("template"), dict) and "spec" in spec["template"]:
            specs.append((doc, spec["template"]["spec"]))
    return specs


def secret_references(container: dict) -> set[str]:
    names = {
        env["valueFrom"]["secretKeyRef"]["name"]
        for env in container.get("env", [])
        if "secretKeyRef" in env.get("valueFrom", {})
    }
    names.update(source["secretRef"]["name"] for source in container.get("envFrom", []) if "secretRef" in source)
    return names


def containers_with_upstream_access(docs: list[dict], secret_names: set[str]) -> dict[str, set[tuple[str, str, str]]]:
    access: dict[str, set[tuple[str, str, str]]] = {"secret": set(), "file": set(), "file_env": set()}
    for doc, spec in pod_specs(docs):
        upstream_volumes = {
            volume["name"]
            for volume in spec.get("volumes", [])
            if volume.get("secret", {}).get("secretName") in {UPSTREAMS_SECRET, *secret_names}
            or volume.get("configMap", {}).get("name") == UPSTREAMS_SECRET
        }
        for container in [*spec.get("initContainers", []), *spec["containers"]]:
            owner = (doc["kind"], doc["metadata"]["name"], container["name"])
            if secret_references(container) & secret_names:
                access["secret"].add(owner)
            if any(mount["name"] in upstream_volumes for mount in container.get("volumeMounts", [])):
                access["file"].add(owner)
            if any(env["name"] == "SIE_UPSTREAMS_FILE" for env in container.get("env", [])):
                access["file_env"].add(owner)
    return access


def upstream_secret_names() -> set[str]:
    return {
        upstream["api_key_secret"]["name"]
        for upstream in upstreams_fixture()["values"].values()
        if "api_key_secret" in upstream
    }


def lane_pod_specs(docs: list[dict]) -> dict[str, dict]:
    return {doc["metadata"]["name"]: doc["spec"]["template"]["spec"] for doc in docs if doc["kind"] == "StatefulSet"}


def test_only_the_remote_lane_worker_receives_upstreams_and_credentials(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, remote_pool_values(l4={"enabled": True}, cpu={"enabled": True}))
    secret_names = upstream_secret_names()

    access = containers_with_upstream_access(docs, secret_names)

    assert access == {"secret": {REMOTE_WORKER}, "file": {REMOTE_WORKER}, "file_env": {REMOTE_WORKER}}
    assert not [doc for doc in docs if doc["kind"] == "Secret" and doc["metadata"]["name"] in secret_names]
    remote_serving = env_entries(docs, "SIE_REMOTE_SERVING")
    assert remote_serving == {
        "sie-sie-cluster-worker-cpu-default/worker": "0",
        "sie-sie-cluster-worker-l4-default/worker": "0",
        "sie-sie-cluster-worker-remote-remote/worker": "1",
    }
    remote = container_env(docs, REMOTE_WORKER[1], "worker")
    assert remote["SIE_UPSTREAMS_FILE"]["value"] == "/etc/sie/upstreams/upstreams.yaml"
    assert remote["SIE_UPSTREAM_KEY_TEAM_SIE"]["valueFrom"]["secretKeyRef"] == {
        "name": "team-sie-upstream",
        "key": "api-key",
    }
    (remote_lane,) = [
        doc for doc in docs if doc["kind"] == "StatefulSet" and doc["metadata"]["name"] == REMOTE_WORKER[1]
    ]
    (worker,) = [c for c in remote_lane["spec"]["template"]["spec"]["containers"] if c["name"] == "worker"]
    assert worker["image"].endswith("-cpu-transformers5")
    assert "--bundle=remote" in worker["args"]
    assert "nvidia.com/gpu" not in worker["resources"]["limits"]


def test_the_remote_lane_image_provides_every_remote_bundle_dependency() -> None:
    values = yaml.safe_load((ROOT / helm.CHART_DIR / "values.yaml").read_text(encoding="utf-8"))
    image_bundle = values["workers"]["pools"]["remote"]["bundles"]["remote"]["imageBundle"]
    bundles = ROOT / "packages/sie_server/bundles"
    remote = yaml.safe_load((bundles / "remote.yaml").read_text(encoding="utf-8"))
    image = yaml.safe_load((bundles / f"{image_bundle}.yaml").read_text(encoding="utf-8"))

    assert image_bundle == "transformers5"
    assert remote["deps"]
    assert remote["deps"].items() <= (image.get("deps") or {}).items()


def test_every_lane_names_a_bundle_the_server_ships(tmp_path: Path) -> None:
    pools = yaml.safe_load((ROOT / helm.CHART_DIR / "values.yaml").read_text(encoding="utf-8"))["workers"]["pools"]
    docs = rendered_documents(tmp_path, remote_pool_values(**{name: {"enabled": True} for name in pools}))

    rendered = {
        arg.removeprefix("--bundle=")
        for _, spec in pod_specs(docs)
        for container in spec["containers"]
        for arg in container.get("args", [])
        if arg.startswith("--bundle=")
    }

    assert rendered == {bundle for pool in pools.values() for bundle in pool["bundles"]}
    assert rendered <= {path.stem for path in (ROOT / "packages/sie_server/bundles").glob("*.yaml")}


def test_the_upstreams_file_is_a_secret_that_matches_the_fixture(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, remote_pool_values())

    (secret,) = [doc for doc in docs if doc["kind"] == "Secret" and doc["metadata"]["name"] == UPSTREAMS_SECRET]

    assert yaml.safe_load(secret["stringData"]["upstreams.yaml"]) == upstreams_fixture()["rendered"]
    assert not [doc for doc in docs if doc["kind"] == "ConfigMap" and doc["metadata"]["name"] == UPSTREAMS_SECRET]


def bound_rules(docs: list[dict], service_account: str) -> list[dict]:
    roles = {(doc["kind"], doc["metadata"]["name"]): doc for doc in docs if doc["kind"] in {"Role", "ClusterRole"}}
    rules = []
    for doc in docs:
        if doc["kind"] not in {"RoleBinding", "ClusterRoleBinding"}:
            continue
        subjects = doc.get("subjects") or []
        if not any(s.get("kind") == "ServiceAccount" and s.get("name") == service_account for s in subjects):
            continue
        rules.extend(roles[(doc["roleRef"]["kind"], doc["roleRef"]["name"])].get("rules", []))
    return rules


def test_no_role_of_the_gateway_reaches_secrets_and_the_remote_lane_has_none(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, remote_pool_values())
    (gateway,) = [
        doc
        for doc in docs
        if doc["kind"] == "Deployment" and doc["metadata"]["name"] == GATEWAY.split("/", maxsplit=1)[0]
    ]
    gateway_account = gateway["spec"]["template"]["spec"]["serviceAccountName"]

    gateway_rules = bound_rules(docs, gateway_account)

    assert any("configmaps" in rule["resources"] for rule in gateway_rules)
    assert not [rule for rule in gateway_rules if {"secrets", "*"} & set(rule["resources"])]
    assert bound_rules(docs, REMOTE_SERVICE_ACCOUNT) == []


def test_the_remote_lane_runs_as_its_own_account_without_identity_or_hugging_face_token(tmp_path: Path) -> None:
    values = remote_pool_values(l4={"enabled": True})
    values["serviceAccount"] = {"annotations": {"eks.amazonaws.com/role-arn": "arn:aws:iam::123456789012:role/sie"}}
    values["workers"]["common"] = {"hfCache": {"tokenSecret": "hf-token"}}

    docs = rendered_documents(tmp_path, values)

    lanes = lane_pod_specs(docs)
    (gateway,) = [
        doc
        for doc in docs
        if doc["kind"] == "Deployment" and doc["metadata"]["name"] == GATEWAY.split("/", maxsplit=1)[0]
    ]
    assert lanes[REMOTE_WORKER[1]]["serviceAccountName"] == REMOTE_SERVICE_ACCOUNT
    assert lanes[REMOTE_WORKER[1]]["automountServiceAccountToken"] is False
    assert (
        lanes["sie-sie-cluster-worker-l4-default"]["serviceAccountName"]
        == gateway["spec"]["template"]["spec"]["serviceAccountName"]
    )
    (account,) = [
        doc for doc in docs if doc["kind"] == "ServiceAccount" and doc["metadata"]["name"] == REMOTE_SERVICE_ACCOUNT
    ]
    assert account["automountServiceAccountToken"] is False
    assert "annotations" not in account["metadata"]
    assert set(env_entries(docs, "HF_TOKEN")) == {"sie-sie-cluster-worker-l4-default/worker"}


def test_annotations_can_bind_the_remote_account_to_a_payload_store_identity(tmp_path: Path) -> None:
    annotations = {"iam.gke.io/gcp-service-account": "payloads@example-project.iam.gserviceaccount.com"}
    values = remote_pool_values()
    values["workers"]["remote"] = {"serviceAccount": {"annotations": annotations}}

    docs = rendered_documents(tmp_path, values)

    (account,) = [
        doc for doc in docs if doc["kind"] == "ServiceAccount" and doc["metadata"]["name"] == REMOTE_SERVICE_ACCOUNT
    ]
    assert account["metadata"]["annotations"] == annotations


def test_remote_lanes_do_not_inherit_the_common_runtime_class(tmp_path: Path) -> None:
    values = remote_pool_values(l4={"enabled": True})
    values["workers"]["common"] = {"runtimeClassName": "nvidia"}

    lanes = lane_pod_specs(rendered_documents(tmp_path, values))

    assert "runtimeClassName" not in lanes[REMOTE_WORKER[1]]
    assert lanes["sie-sie-cluster-worker-l4-default"]["runtimeClassName"] == "nvidia"


def test_the_worker_network_policy_admits_only_the_gateway_to_the_remote_lane(tmp_path: Path) -> None:
    values = remote_pool_values()
    values["workers"]["remote"] = {"networkPolicy": {"enabled": False}}
    docs = rendered_documents(tmp_path, values)
    remote_labels = next(
        doc["spec"]["template"]["metadata"]["labels"]
        for doc in docs
        if doc["kind"] == "StatefulSet" and doc["metadata"]["name"] == REMOTE_WORKER[1]
    )

    (policy,) = [
        doc for doc in docs if doc["kind"] == "NetworkPolicy" and doc["metadata"]["name"] == "sie-sie-cluster-worker"
    ]

    assert policy["spec"]["podSelector"]["matchLabels"].items() <= remote_labels.items()
    (ingress,) = policy["spec"]["ingress"]
    (peer,) = ingress["from"]
    assert peer["podSelector"]["matchLabels"]["app.kubernetes.io/component"] == "gateway"


def test_a_remote_lane_needs_one_of_the_two_network_policies(tmp_path: Path) -> None:
    remote_policy_only = remote_pool_values()
    remote_policy_only["workers"]["networkPolicy"] = {"enabled": False}
    neither = remote_pool_values()
    neither["workers"]["networkPolicy"] = {"enabled": False}
    neither["workers"]["remote"] = {"networkPolicy": {"enabled": False}}

    result = render_workers(tmp_path, neither)

    assert remote_network_policy(rendered_documents(tmp_path, remote_policy_only)) is not None
    assert result.returncode != 0
    assert "a remote lane needs a NetworkPolicy" in result.stderr


def test_a_remote_lane_refuses_extra_ingress_sources(tmp_path: Path) -> None:
    extra = {
        "from": [{"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "benchmarks"}}}],
        "ports": [{"port": 8080, "protocol": "TCP"}],
    }
    values = remote_pool_values()
    values["workers"]["networkPolicy"] = {"enabled": True, "extraIngress": [extra]}
    values["workers"]["remote"] = {"networkPolicy": {"enabled": False}}

    result = render_workers(tmp_path, values)

    assert result.returncode != 0
    assert "workers.networkPolicy.extraIngress cannot be set while a remote lane is enabled" in result.stderr


def test_no_upstreams_file_is_rendered_without_a_remote_lane(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, {"upstreams": upstreams_fixture()["values"], **L4_POOL})

    assert not [doc for doc in docs if doc["metadata"]["name"] in {UPSTREAMS_SECRET, REMOTE_SERVICE_ACCOUNT}]
    assert env_entries(docs, "SIE_REMOTE_SERVING") == {"sie-sie-cluster-worker-l4-default/worker": "0"}


def test_the_global_switch_withholds_the_upstreams_and_credentials(tmp_path: Path) -> None:
    values = remote_pool_values()
    values["workers"]["remote"] = {"serving": False}

    docs = rendered_documents(tmp_path, values)

    assert env_entries(docs, "SIE_REMOTE_SERVING") == {"sie-sie-cluster-worker-remote-remote/worker": "0"}
    assert not [name for name in container_env(docs, REMOTE_WORKER[1], "worker") if name.startswith("SIE_UPSTREAM")]
    assert not [doc for doc in docs if doc["metadata"]["name"] == UPSTREAMS_SECRET]
    assert containers_with_upstream_access(docs, upstream_secret_names()) == {
        "secret": set(),
        "file": set(),
        "file_env": set(),
    }


def test_changing_an_upstream_restarts_the_remote_lane(tmp_path: Path) -> None:
    def checksum(values: dict) -> str:
        (statefulset,) = worker_statefulsets(tmp_path, values)
        return statefulset["spec"]["template"]["metadata"]["annotations"]["checksum/upstreams"]

    before = remote_pool_values()
    after = remote_pool_values()
    after["upstreams"]["open-host"]["base_url"] = "https://other.example.com/v1"

    assert checksum(before) != checksum(after)


EVIDENCE_CONFIG_MAP = "remote-equivalence-evidence"


def evidence_values(record_file: str = "/etc/sie/equivalence/local-model.json", **remote: object) -> dict:
    values = remote_pool_values(l4={"enabled": True})
    values["upstreams"]["open-host"]["equivalence"]["record_files"] = {"local/model": record_file}
    values["workers"]["remote"] = {"equivalence": {"configMap": EVIDENCE_CONFIG_MAP}, **remote}
    return values


def evidence_mounts(docs: list[dict]) -> dict[tuple[str, str, str], dict]:
    mounts = {}
    for doc, spec in pod_specs(docs):
        volumes = {
            volume["name"]
            for volume in spec.get("volumes", [])
            if volume.get("configMap", {}).get("name") == EVIDENCE_CONFIG_MAP
        }
        for container in [*spec.get("initContainers", []), *spec["containers"]]:
            for mount in container.get("volumeMounts", []):
                if mount["name"] in volumes:
                    mounts[(doc["kind"], doc["metadata"]["name"], container["name"])] = mount
    return mounts


def test_only_the_remote_lane_worker_mounts_equivalence_evidence(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, evidence_values())

    assert evidence_mounts(docs) == {
        REMOTE_WORKER: {"name": "equivalence", "mountPath": "/etc/sie/equivalence", "readOnly": True}
    }
    (secret,) = [doc for doc in docs if doc["kind"] == "Secret" and doc["metadata"]["name"] == UPSTREAMS_SECRET]
    rendered = yaml.safe_load(secret["stringData"]["upstreams.yaml"])
    assert rendered["upstreams"]["open-host"]["equivalence"]["record_files"] == {
        "local/model": "/etc/sie/equivalence/local-model.json"
    }


def test_a_remote_lane_without_remote_serving_mounts_no_evidence(tmp_path: Path) -> None:
    assert evidence_mounts(rendered_documents(tmp_path, evidence_values(serving=False))) == {}


def test_the_evidence_config_map_reaches_the_pod_template_only_as_its_volume(tmp_path: Path) -> None:
    def template(values: dict) -> dict:
        (statefulset,) = [
            doc
            for doc in rendered_documents(tmp_path, values)
            if doc["kind"] == "StatefulSet" and doc["metadata"]["name"] == REMOTE_WORKER[1]
        ]
        return statefulset["spec"]["template"]

    renamed = evidence_values()
    renamed["workers"]["remote"]["equivalence"]["configMap"] = "other-evidence"
    template_after = template(renamed)
    for volume in template_after["spec"]["volumes"]:
        if volume.get("configMap", {}).get("name") == "other-evidence":
            volume["configMap"]["name"] = EVIDENCE_CONFIG_MAP

    assert template(evidence_values()) == template_after


@pytest.mark.parametrize(
    "record_file",
    [
        "/proofs/local-model.json",
        "/etc/sie/equivalence/",
        "/etc/sie/equivalence/..",
        "/etc/sie/equivalence/nested/local-model.json",
        "/etc/sie/equivalence-other/local-model.json",
    ],
)
def test_mounted_evidence_requires_each_record_file_to_be_one_of_its_keys(tmp_path: Path, record_file: str) -> None:
    error = render_error(tmp_path, evidence_values(record_file))

    assert "record files must be keys of workers.remote.equivalence.configMap" in error
    assert "/proofs/" not in error


@pytest.mark.parametrize(
    "evidence",
    [
        {"configMap": "Upper"},
        {"configMap": "-leading"},
        {"configMap": "has_underscore"},
        {"configMap": "proofs..remote"},
        {"configMap": "proofs.-remote"},
        {"configMap": "proofs-.remote"},
        {"configMap": "proofs."},
        {"configMap": "a" * 254},
        {"configMap": 5},
        {"configMap": "evidence", "secret": "evidence"},
        "evidence",
    ],
)
def test_the_evidence_setting_names_one_valid_config_map(tmp_path: Path, evidence: object) -> None:
    values = evidence_values()
    values["workers"]["remote"]["equivalence"] = evidence

    assert "workers.remote.equivalence" in render_error(tmp_path, values)


@pytest.mark.parametrize("name", ["evidence", "remote.equivalence-evidence", "a" * 63 + "." + "b" * 63])
def test_dotted_config_map_names_are_accepted(tmp_path: Path, name: str) -> None:
    values = evidence_values()
    values["workers"]["remote"]["equivalence"]["configMap"] = name
    docs = rendered_documents(tmp_path, values)

    (statefulset,) = [
        doc for doc in docs if doc["kind"] == "StatefulSet" and doc["metadata"]["name"] == REMOTE_WORKER[1]
    ]
    volumes = {volume["name"]: volume for volume in statefulset["spec"]["template"]["spec"]["volumes"]}
    assert volumes["equivalence"]["configMap"]["name"] == name


def test_without_mounted_evidence_record_paths_are_only_checked_for_shape(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, remote_pool_values())

    assert evidence_mounts(docs) == {}


def upstream(**overrides: object) -> dict:
    definition: dict = {
        "kind": "sie",
        "base_url": "https://sie.example.internal",
        "rate_cap": {"requests_per_minute": 60, "max_concurrency": 4},
    }
    definition.update(overrides)
    return definition


def reads_secret(name: str) -> dict:
    return {"name": "SOME_KEY", "valueFrom": {"secretKeyRef": {"name": name, "key": "api-key"}}}


def remote_lane(**pool: object) -> dict:
    return {"workers": {"pools": {"remote": {"enabled": True, **pool}}}}


SECRET_UPSTREAM = {"team": upstream(api_key_secret={"name": "team-upstream", "key": "api-key"})}


@pytest.mark.parametrize(
    ("path", "key_path", "key"),
    [
        ("auth.oauth2Proxy.secret.name", "auth.oauth2Proxy.secret.clientIDKey", "api-key"),
        ("auth.oauth2Proxy.secret.name", "auth.oauth2Proxy.secret.clientSecretKey", "api-key"),
        ("auth.oauth2Proxy.secret.name", "auth.oauth2Proxy.secret.cookieSecretKey", "api-key"),
        (
            "observability.otel.collector.betterStack.existingSecret",
            "observability.otel.collector.betterStack.tokenKey",
            "api-key",
        ),
        ("mcpEdge.existingSecretName", None, "connector-secrets"),
        ("mcpEdge.clusterApiKey.existingSecretName", "mcpEdge.clusterApiKey.secretKey", "api-key"),
        ("ingress.tlsConfig.secretName", None, "tls.key"),
        ("ingress.tlsConfig.secretName", None, "tls.crt"),
        ("ingress.tlsConfig.selfSigned.rootCA.secretName", None, "tls.key"),
    ],
)
def test_other_chart_credentials_cannot_share_an_upstream_key(
    tmp_path: Path,
    path: str,
    key_path: str | None,
    key: str,
) -> None:
    values: dict = {"upstreams": {"team": upstream(api_key_secret={"name": "team-upstream", "key": key})}}
    entries = [(path, "team-upstream")]
    if key_path is not None:
        entries.append((key_path, key))
    for dotted_path, value in entries:
        parts = dotted_path.split(".")
        current = values
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    result = render_workers(tmp_path, values)

    assert result.returncode != 0
    assert "name the same Secret key" in result.stderr


def test_gateway_extra_env_cannot_read_the_generated_upstreams_file(tmp_path: Path) -> None:
    values = remote_pool_values()
    values["gateway"] = {"extraEnv": [reads_secret(UPSTREAMS_SECRET)]}

    result = render_workers(tmp_path, values)

    assert result.returncode != 0
    assert "gateway.extraEnv reads the upstream Secret" in result.stderr


def test_the_generated_upstreams_file_cannot_be_used_as_an_api_key(tmp_path: Path) -> None:
    values = {"upstreams": {"team": upstream(api_key_secret={"name": UPSTREAMS_SECRET, "key": "upstreams.yaml"})}}

    result = render_workers(tmp_path, values)

    assert result.returncode != 0
    assert "the upstreams file name the same Secret key" in result.stderr


def test_the_mcp_ingress_certificate_cannot_be_sent_upstream(tmp_path: Path) -> None:
    values = {"upstreams": {"team": upstream(api_key_secret={"name": "sie-sie-cluster-mcp-tls", "key": "tls.key"})}}

    result = render_workers(tmp_path, values)

    assert result.returncode != 0
    assert "the MCP ingress TLS certificate name the same Secret key" in result.stderr


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            remote_lane(gpu={"count": 1}),
            "workers.pools.remote.bundles.remote: a remote lane holds upstream credentials",
        ),
        (
            remote_lane(bundles={"remote": {"imageBundle": "remote", "minReplicas": 1, "maxReplicas": 1}}),
            "no remote worker image is published; set imageBundle: transformers5",
        ),
        (
            remote_lane(engine="candle"),
            "the remote adapters run only in the pytorch worker",
        ),
        (
            remote_lane(resources={"limits": {"nvidia.com/gpu": "1"}}),
            "workers.pools.remote.resources.limits: a remote lane runs no model on an accelerator",
        ),
        (
            remote_lane(resources={"requests": {"amd.com/gpu": "1"}}),
            "workers.pools.remote.resources.requests: a remote lane runs no model on an accelerator",
        ),
        (
            remote_lane(runtimeClassName="nvidia"),
            "workers.pools.remote.runtimeClassName: a remote lane runs on the cluster's default container runtime",
        ),
        (
            {"upstreams": {"team": upstream(extra=1)}},
            "upstreams.team: field 2, counting in name order, is not an upstream field",
        ),
        ({"upstreams": {"team": upstream(kind="grpc")}}, "upstreams.team.kind must be sie or openai"),
        (
            {"upstreams": {"team": upstream(base_url="https://sie.example.internal:0")}},
            "upstreams.team.base_url is not a valid URL: a port is 1 to 65535",
        ),
        (
            {"upstreams": {"team": upstream(base_url=f"http://{CREDENTIAL_CANARY}.example.internal")}},
            "upstreams.team.base_url must use https outside loopback",
        ),
        (
            {"upstreams": {"team": upstream(base_url=f"https://user:{CREDENTIAL_CANARY}@sie.example.internal")}},
            "upstreams.team.base_url must be an http or https URL that names a host and carries no credentials",
        ),
        (
            {"upstreams": {"team": upstream(base_url=f"https://sie.example.internal/v1?key={CREDENTIAL_CANARY}")}},
            "upstreams.team.base_url must not carry a query or a fragment",
        ),
        (
            {"upstreams": {"team": upstream(api_key_secret=CREDENTIAL_CANARY)}},
            "upstreams.team.api_key_secret must name a Kubernetes Secret as {name, key}",
        ),
        (
            {"upstreams": {"team": upstream(base_url="http://127.0.0.1:8080", proxy_url="http://proxy:3128")}},
            "upstreams.team.proxy_url requires an https base_url",
        ),
        (
            {"upstreams": {"Team_SIE": upstream()}},
            "upstreams: entry 1, counting in name order, is not an upstream name",
        ),
        (
            {"upstreams": {CREDENTIAL_CANARY.upper(): upstream()}},
            "upstreams: entry 1, counting in name order, is not an upstream name",
        ),
        ({"upstreams": {CREDENTIAL_CANARY: "pasted"}}, "upstreams: entry 1, counting in name order, must be a map"),
        (
            {"upstreams": {"team": {**upstream(), CREDENTIAL_CANARY: "pasted"}}},
            "upstreams.team: field 4, counting in name order, is not an upstream field",
        ),
        (
            {
                "upstreams": {
                    "team": upstream(rate_cap={"requests_per_minute": 1, "max_concurrency": 1, CREDENTIAL_CANARY: 1})
                }
            },
            "upstreams.team.rate_cap: field 3, counting in name order, is not a rate cap field",
        ),
        (
            {
                "upstreams": {
                    "team": upstream(api_key_secret={"name": "team-upstream", "key": "api-key", CREDENTIAL_CANARY: 1})
                }
            },
            "upstreams.team.api_key_secret: field 3, counting in name order, is not a field",
        ),
        ({"upstreams": {"team": upstream(rate_cap=None)}}, "upstreams.team.rate_cap is required"),
        (
            {"upstreams": {"team": upstream(rate_cap={"requests_per_minute": 0, "max_concurrency": 1})}},
            "upstreams.team.rate_cap.requests_per_minute must be a positive integer",
        ),
        (
            {"workers": {"common": {"extraEnv": [{"name": "SIE_UPSTREAMS_FILE", "value": "/tmp/upstreams.yaml"}]}}},
            "workers.common.extraEnv must not override chart-owned variable SIE_UPSTREAMS_FILE",
        ),
        (
            {"workers": {"common": {"extraEnv": [{"name": "SIE_REMOTE_SERVING", "value": "1"}]}}},
            "workers.common.extraEnv must not override chart-owned variable SIE_REMOTE_SERVING",
        ),
        (
            {"upstreams": SECRET_UPSTREAM, "workers": {"common": {"extraEnv": [reads_secret("team-upstream")]}}},
            "workers.common.extraEnv reads the upstream Secret",
        ),
        (
            {"upstreams": SECRET_UPSTREAM, "gateway": {"extraEnv": [reads_secret("team-upstream")]}},
            "gateway.extraEnv reads the upstream Secret",
        ),
        (
            {
                "upstreams": SECRET_UPSTREAM,
                "workers": {"common": {"workerSidecar": {"extraEnv": [reads_secret("team-upstream")]}}},
            },
            "workers.common.workerSidecar.extraEnv reads the upstream Secret",
        ),
        (
            {
                "upstreams": SECRET_UPSTREAM,
                "workers": {
                    "pools": {
                        "l4": {
                            "enabled": True,
                            "bundles": {
                                "default": {
                                    "minReplicas": 0,
                                    "maxReplicas": 1,
                                    "extraEnv": [reads_secret("team-upstream")],
                                }
                            },
                        }
                    }
                },
            },
            "workers.pools.l4.bundles.default.extraEnv reads the upstream Secret",
        ),
        (
            {
                "upstreams": SECRET_UPSTREAM,
                "workers": {"common": {"hfCache": {"tokenSecret": "team-upstream", "tokenSecretKey": "api-key"}}},
            },
            "upstreams.team.api_key_secret and workers.common.hfCache.tokenSecret name the same Secret key "
            "(team-upstream/api-key)",
        ),
        (
            {
                "upstreams": SECRET_UPSTREAM,
                "gateway": {
                    "auth": {"mode": "static", "tokenSecretName": "team-upstream", "tokenSecretKey": "api-key"}
                },
            },
            "upstreams.team.api_key_secret and gateway.auth.tokenSecretName name the same Secret key",
        ),
        (
            {
                "upstreams": SECRET_UPSTREAM,
                "gateway": {"auth": {"adminTokenSecretName": "team-upstream", "adminTokenSecretKey": "api-key"}},
            },
            "upstreams.team.api_key_secret and gateway.auth.adminTokenSecretName name the same Secret key",
        ),
        (
            {
                "upstreams": SECRET_UPSTREAM,
                "config": {"auth": {"readTokenSecretName": "team-upstream", "readTokenSecretKey": "api-key"}},
            },
            "upstreams.team.api_key_secret and the sie-config read token name the same Secret key",
        ),
        (
            {
                "upstreams": {
                    "team": upstream(api_key_secret={"name": GENERATED_ADMIN_TOKEN_SECRET, "key": "SIE_ADMIN_TOKEN"})
                }
            },
            "upstreams.team.api_key_secret and the sie-config admin token name the same Secret key",
        ),
        (
            {"upstreams": {"team": upstream(api_key_secret={"name": "sie-nats-auth-worker", "key": "password"})}},
            "upstreams.team.api_key_secret and the NATS worker password name the same Secret key",
        ),
    ],
)
def test_unsafe_remote_configuration_fails_the_render(tmp_path: Path, values: dict, message: str) -> None:
    result = render_workers(tmp_path, with_worker_network_policy(values))

    assert result.returncode != 0
    assert message in result.stderr
    assert CREDENTIAL_CANARY not in result.stderr.lower()


@pytest.mark.parametrize("case", upstreams_fixture()["urls"], ids=lambda case: case["url"])
def test_the_chart_judges_each_url_as_the_server_does(tmp_path: Path, case: dict) -> None:
    field = "base_url" if case["tls"] else "proxy_url"
    definition = upstream(base_url=case["url"]) if case["tls"] else upstream(proxy_url=case["url"])

    result = render_workers(tmp_path, {**remote_pool_values(), "upstreams": {"team": definition}})

    if case["accepted"]:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert f"upstreams.team.{field} " in result.stderr


DEFINITION_BASE = {
    "base_url": "https://host.example.com/v1",
    "rate_cap": {"requests_per_minute": 60, "max_concurrency": 4},
}


@pytest.mark.parametrize(
    "case", upstreams_fixture()["definitions"], ids=lambda case: str(sorted(case["upstream"].items()))[:80]
)
def test_the_chart_judges_each_definition_as_the_server_does(tmp_path: Path, case: dict) -> None:
    definition = {**DEFINITION_BASE, **case["upstream"]}

    result = render_workers(tmp_path, {**remote_pool_values(), "upstreams": {"team": definition}})

    if case["accepted"]:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "upstreams.team" in result.stderr


@pytest.mark.parametrize("key", [".dockerconfigjson", ".dockercfg"])
def test_image_pull_credentials_cannot_be_sent_to_an_upstream(tmp_path: Path, key: str) -> None:
    values = remote_pool_values()
    values["global"] = {"imagePullSecrets": ["registry-credentials"]}
    values["upstreams"] = {"team": upstream(api_key_secret={"name": "registry-credentials", "key": key})}

    result = render_workers(tmp_path, values)

    assert result.returncode != 0
    assert "global.imagePullSecrets name the same Secret key" in result.stderr


def test_an_image_pull_secret_may_hold_a_separate_upstream_key(tmp_path: Path) -> None:
    values = remote_pool_values()
    values["global"] = {"imagePullSecrets": ["registry-credentials"]}
    values["upstreams"] = {"team": upstream(api_key_secret={"name": "registry-credentials", "key": "upstream-key"})}

    result = render_workers(tmp_path, values)

    assert result.returncode == 0, result.stderr


REMOTE_NETWORK_POLICY = "sie-sie-cluster-worker-remote"
DENIED_BASE_V4 = [
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "100.64.0.0/10",
    "169.254.0.0/16",
    "168.63.129.16/32",
]
DENIED_BASE_V6 = ["fc00::/7", "fe80::/10", "64:ff9b::/96", "64:ff9b:1::/48"]
CLUSTER_DNS = {
    "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "kube-system"}},
    "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
}


def remote_network_policy(docs: list[dict]) -> dict | None:
    policies = [
        doc for doc in docs if doc["kind"] == "NetworkPolicy" and doc["metadata"]["name"] == REMOTE_NETWORK_POLICY
    ]
    return policies[0] if policies else None


def pod_labels(doc: dict) -> dict:
    if doc["kind"] == "Pod":
        return doc["metadata"].get("labels", {})
    return doc["spec"]["template"]["metadata"].get("labels", {})


def selects(selector: dict, labels: dict) -> bool:
    if any(labels.get(key) != value for key, value in (selector.get("matchLabels") or {}).items()):
        return False
    for expression in selector.get("matchExpressions") or []:
        key, operator = expression["key"], expression["operator"]
        if operator == "In" and labels.get(key) not in expression["values"]:
            return False
        if operator == "NotIn" and key in labels and labels[key] in expression["values"]:
            return False
        if operator == "Exists" and key not in labels:
            return False
        if operator == "DoesNotExist" and key in labels:
            return False
    return True


def selected_workloads(docs: list[dict], selector: dict) -> set[str]:
    return {doc["metadata"]["name"] for doc, _ in pod_specs(docs) if selects(selector, pod_labels(doc))}


def public_rule(policy: dict) -> dict:
    (rule,) = [rule for rule in policy["spec"]["egress"] if any("ipBlock" in peer for peer in rule.get("to", []))]
    return rule


def test_the_remote_network_policy_admits_only_the_gateway_and_selects_only_remote_lanes(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, remote_pool_values(l4={"enabled": True}, cpu={"enabled": True}))

    policy = remote_network_policy(docs)

    assert policy is not None
    spec = policy["spec"]
    assert selected_workloads(docs, spec["podSelector"]) == {REMOTE_WORKER[1]}
    assert spec["policyTypes"] == ["Ingress", "Egress"]
    (ingress,) = spec["ingress"]
    (peer,) = ingress["from"]
    assert selected_workloads(docs, peer["podSelector"]) == {"sie-sie-cluster-gateway"}
    assert ingress["ports"] == [{"port": 8080, "protocol": "TCP"}]


def test_no_other_policy_admits_extra_sources_to_the_remote_lane(tmp_path: Path) -> None:
    extra = {
        "from": [{"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "benchmarks"}}}],
        "ports": [{"port": 8080, "protocol": "TCP"}],
    }
    values = remote_pool_values(l4={"enabled": True})
    values["workers"]["networkPolicy"] = {"enabled": True, "extraIngress": [extra]}
    docs = rendered_documents(tmp_path, values)
    lanes = {doc["metadata"]["name"]: pod_labels(doc) for doc, _ in pod_specs(docs) if doc["kind"] == "StatefulSet"}

    def admitted(lane: str) -> list[dict]:
        return [
            rule
            for doc in docs
            if doc["kind"] == "NetworkPolicy" and selects(doc["spec"]["podSelector"], lanes[lane])
            for rule in doc["spec"].get("ingress", [])
        ]

    remote_rules = admitted(REMOTE_WORKER[1])
    assert [peer for rule in remote_rules for peer in rule["from"]] == [
        {
            "podSelector": {
                "matchLabels": remote_network_policy(docs)["spec"]["ingress"][0]["from"][0]["podSelector"][
                    "matchLabels"
                ]
            }
        }
    ]
    assert {
        workload
        for rule in remote_rules
        for peer in rule["from"]
        for workload in selected_workloads(docs, peer["podSelector"])
    } == {"sie-sie-cluster-gateway"}
    assert extra in admitted("sie-sie-cluster-worker-l4-default")


def test_the_remote_network_policy_limits_egress_to_the_resolver_the_release_and_public_upstream_ports(
    tmp_path: Path,
) -> None:
    docs = rendered_documents(tmp_path, remote_pool_values())

    egress = remote_network_policy(docs)["spec"]["egress"]

    assert egress[0] == {
        "to": [CLUSTER_DNS],
        "ports": [{"port": 53, "protocol": "UDP"}, {"port": 53, "protocol": "TCP"}],
    }
    assert all("to" in rule and rule["ports"] for rule in egress)
    in_release = {
        (workload, port["port"])
        for rule in egress[1:]
        for peer in rule["to"]
        if "podSelector" in peer
        for workload in selected_workloads(docs, peer["podSelector"])
        for port in rule["ports"]
    }
    assert in_release == {
        ("sie-nats", 4222),
        ("sie-sie-cluster-config", 8080),
        ("sie-sie-cluster-gateway", 8080),
    }
    public = public_rule(remote_network_policy(docs))
    assert public["to"] == [
        {"ipBlock": {"cidr": "0.0.0.0/0", "except": DENIED_BASE_V4}},
        {"ipBlock": {"cidr": "::/0", "except": DENIED_BASE_V6}},
    ]
    assert public["ports"] == [{"port": 443, "protocol": "TCP"}, {"port": 3128, "protocol": "TCP"}]


def test_the_cluster_resolver_is_configurable_and_must_be_scoped(tmp_path: Path) -> None:
    node_local = {"ipBlock": {"cidr": "169.254.20.10/32"}}
    values = remote_pool_values()
    values["workers"]["remote"] = {"networkPolicy": {"dnsTo": [CLUSTER_DNS, node_local]}}
    unscoped = remote_pool_values()
    unscoped["workers"]["remote"] = {"networkPolicy": {"dnsTo": [{"podSelector": {}}]}}
    empty = remote_pool_values()
    empty["workers"]["remote"] = {"networkPolicy": {"dnsTo": []}}

    egress = remote_network_policy(rendered_documents(tmp_path, values))["spec"]["egress"]

    assert egress[0]["to"] == [CLUSTER_DNS, node_local]
    assert "dnsTo[0] admits every destination" in render_error(tmp_path, unscoped)
    assert "dnsTo needs at least one peer" in render_error(tmp_path, empty)


def test_the_remote_network_policy_admits_the_collector_when_it_runs(tmp_path: Path) -> None:
    values = remote_pool_values()
    values.update(AUTOSCALING_VALUES)

    docs = rendered_documents(tmp_path, values)

    egress = remote_network_policy(docs)["spec"]["egress"]
    collector_rules = [
        rule
        for rule in egress
        for peer in rule.get("to", [])
        if "podSelector" in peer
        and peer["podSelector"]["matchLabels"].get("app.kubernetes.io/component") == "otel-collector"
    ]
    assert [rule["ports"] for rule in collector_rules] == [[{"port": 4327, "protocol": "TCP"}]]


def test_the_remote_network_policy_admits_bundled_tempo(tmp_path: Path) -> None:
    values = remote_pool_values()
    values["observability"] = {"tracing": {"enabled": True, "tempo": {"install": True}}}

    docs = rendered_documents(tmp_path, values)

    tempo_rules = [
        rule
        for rule in remote_network_policy(docs)["spec"]["egress"]
        for peer in rule.get("to", [])
        if "podSelector" in peer and selected_workloads(docs, peer["podSelector"]) == {"tempo"}
    ]
    assert [rule["ports"] for rule in tempo_rules] == [[{"port": 4317, "protocol": "TCP"}]]


def test_upstream_ports_are_the_explicit_ports_of_what_the_lane_connects_to(tmp_path: Path) -> None:
    values = remote_pool_values()
    values["upstreams"] = {
        "tls-port": upstream(base_url="https://sie.example.internal:8443/v1"),
        "default-port": upstream(base_url="https://host.example.com"),
        "loopback": upstream(base_url="http://127.0.0.1:9000"),
        "loopback-name": upstream(base_url="http://LOCALHOST:9100"),
        "ipv6": upstream(base_url="https://[2001:db8::1]:9443"),
        "proxied": upstream(base_url="https://behind.example.com:7443", proxy_url="http://proxy.example.internal:3128"),
        "proxy-no-port": upstream(base_url="https://other.example.com:6443", proxy_url="http://proxy.example.internal"),
    }

    public = public_rule(remote_network_policy(rendered_documents(tmp_path, values)))

    assert sorted(port["port"] for port in public["ports"]) == [443, 3128, 8443, 9443]


def test_allowed_ranges_carry_the_denied_ranges_inside_them(tmp_path: Path) -> None:
    values = remote_pool_values()
    values["workers"]["remote"] = {
        "networkPolicy": {
            "allowedCidrs": ["0.0.0.0/1", "128.0.0.0/1", "203.0.113.0/24", "::/0"],
            "extraDeniedCidrs": ["198.18.0.0/15", "2001:db8:1::/48"],
        }
    }

    public = public_rule(remote_network_policy(rendered_documents(tmp_path, values)))

    assert public["to"] == [
        {"ipBlock": {"cidr": "0.0.0.0/1", "except": ["10.0.0.0/8", "100.64.0.0/10"]}},
        {
            "ipBlock": {
                "cidr": "128.0.0.0/1",
                "except": ["172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "168.63.129.16/32", "198.18.0.0/15"],
            }
        },
        {"ipBlock": {"cidr": "203.0.113.0/24"}},
        {"ipBlock": {"cidr": "::/0", "except": [*DENIED_BASE_V6, "2001:db8:1::/48"]}},
    ]


def test_an_empty_allowed_list_renders_no_public_rule(tmp_path: Path) -> None:
    values = remote_pool_values()
    values["workers"]["remote"] = {"networkPolicy": {"allowedCidrs": []}}

    egress = remote_network_policy(rendered_documents(tmp_path, values))["spec"]["egress"]

    assert not [peer for rule in egress for peer in rule.get("to", []) if "ipBlock" in peer]


def test_an_extra_egress_port_range_set_on_the_command_line_renders(tmp_path: Path) -> None:
    values_file = tmp_path / "values.yaml"
    values_file.write_text(yaml.safe_dump(remote_pool_values()), encoding="utf-8")
    rule = "workers.remote.networkPolicy.extraEgress[0]"

    result = subprocess.run(
        [
            "mise",
            "exec",
            "--",
            "helm",
            "template",
            "sie",
            str(helm.CHART_DIR),
            "--namespace",
            "sie",
            "-f",
            str(values_file),
            "--set",
            f"{rule}.to[0].ipBlock.cidr=203.0.113.7/32",
            "--set",
            f"{rule}.ports[0].port=8000",
            "--set",
            f"{rule}.ports[0].endPort=9000",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    (policy,) = [
        doc
        for doc in yaml.safe_load_all(result.stdout)
        if doc and doc["kind"] == "NetworkPolicy" and doc["metadata"]["name"] == REMOTE_NETWORK_POLICY
    ]
    assert policy["spec"]["egress"][-1]["ports"] == [{"port": 8000, "endPort": 9000}]


def test_extra_egress_rules_are_appended(tmp_path: Path) -> None:
    rule = {"to": [{"ipBlock": {"cidr": "169.254.169.254/32"}}], "ports": [{"port": 80, "protocol": "TCP"}]}
    values = remote_pool_values()
    values["workers"]["remote"] = {"networkPolicy": {"extraEgress": [rule]}}

    egress = remote_network_policy(rendered_documents(tmp_path, values))["spec"]["egress"]

    assert egress[-1] == rule


@pytest.mark.parametrize(
    ("network_policy", "message"),
    [
        ({"extraEgress": [{"ports": [{"port": 443}]}]}, "extraEgress[0] needs a non-empty to and ports"),
        (
            {"extraEgress": [{"to": [{"ipBlock": {"cidr": "10.0.0.1/32"}}]}]},
            "extraEgress[0] needs a non-empty to and ports",
        ),
        ({"extraEgress": [{"to": [{}], "ports": [{"port": 443}]}]}, "extraEgress[0].to[0] admits every destination"),
        (
            {"extraEgress": [{"to": [{"podSelector": {}}], "ports": [{"port": 443}]}]},
            "extraEgress[0].to[0] admits every destination",
        ),
        (
            {"extraEgress": [{"to": [{"ipBlock": {"cidr": "0.0.0.0/0"}}], "ports": [{"port": 443}]}]},
            "extraEgress[0].to[0] admits every address",
        ),
        (
            {"extraEgress": [{"to": [{"ipBlock": {"cidr": "10.0.0.1/32"}}], "ports": [{}]}]},
            "extraEgress[0].ports[0] admits every port",
        ),
        (
            {"extraEgress": [{"to": [{"ipBlock": {"cidr": "10.0.0.1/32"}}], "ports": [{"port": 1, "endPort": 65535}]}]},
            "extraEgress[0].ports[0] spans nearly every port",
        ),
        (
            {
                "extraEgress": [
                    {"to": [{"ipBlock": {"cidr": "10.0.0.1/32"}}], "ports": [{"port": "https", "endPort": 9000}]}
                ]
            },
            "extraEgress[0].ports[0] sets endPort, which needs a numeric port",
        ),
        (
            {
                "extraEgress": [
                    {"to": [{"ipBlock": {"cidr": "10.0.0.1/32"}}], "ports": [{"port": 8000, "endPort": "9000"}]}
                ]
            },
            "extraEgress[0].ports[0].endPort must be an integer",
        ),
        ({"allowedCidrs": ["2000::/3"]}, "allowedCidrs[0]: the only IPv6 entry is ::/0"),
        ({"allowedCidrs": ["10.1.0.0/16"]}, "allowedCidrs[0] lies inside the denied range 10.0.0.0/8"),
        ({"allowedCidrs": ["0.0.0.0/33"]}, "allowedCidrs[0] must be an IPv4 CIDR"),
        ({"extraDeniedCidrs": ["not-a-range"]}, "extraDeniedCidrs[0] must be a CIDR"),
    ],
)
def test_unsafe_remote_network_policy_values_fail_the_render(
    tmp_path: Path, network_policy: dict, message: str
) -> None:
    values = remote_pool_values()
    values["workers"]["remote"] = {"networkPolicy": network_policy}

    assert message in render_error(tmp_path, values)


@pytest.mark.parametrize(
    "values",
    [
        {"upstreams": {"team": upstream()}, **L4_POOL},
        {
            "workers": {
                "networkPolicy": {"enabled": True},
                "remote": {"networkPolicy": {"enabled": False}},
                "pools": {"remote": {"enabled": True}},
            }
        },
    ],
    ids=["no-remote-lane", "disabled"],
)
def test_no_remote_network_policy_without_a_remote_lane_or_when_disabled(tmp_path: Path, values: dict) -> None:
    assert remote_network_policy(rendered_documents(tmp_path, values)) is None


INVALID_POLICY_CIDRS = [
    "2001:db8::/129",
    "2001:db8::/999",
    "2001:::1/64",
    "2001:db8::1::2/64",
    "1:2:3:4:5:6:7:8:9/64",
    "1:2:3:4:5:6:7/64",
    "::ffff:999.2.3.4/128",
    "999.2.3.4/32",
    "10.0.0.0/33",
]


@pytest.mark.parametrize("cidr", INVALID_POLICY_CIDRS)
@pytest.mark.parametrize("surface", ["extraDeniedCidrs", "extraEgress"])
def test_remote_policy_rejects_invalid_cidrs(tmp_path: Path, cidr: str, surface: str) -> None:
    values = remote_pool_values()
    policy = (
        {"extraDeniedCidrs": [cidr]}
        if surface == "extraDeniedCidrs"
        else {"extraEgress": [{"to": [{"ipBlock": {"cidr": cidr}}], "ports": [{"port": 443}]}]}
    )
    values["workers"]["remote"] = {"networkPolicy": policy}

    result = render_workers(tmp_path, values)

    assert result.returncode != 0
    assert surface in result.stderr
    assert "CIDR" in result.stderr


@pytest.mark.parametrize("cidr", ["::/128", "2001:db8::/64", "1:2:3:4:5:6:7:8/128", "2001:db8::192.0.2.1/128"])
def test_remote_policy_accepts_valid_ipv6_cidrs(tmp_path: Path, cidr: str) -> None:
    values = remote_pool_values()
    values["workers"]["remote"] = {
        "networkPolicy": {
            "extraDeniedCidrs": [cidr],
            "extraEgress": [{"to": [{"ipBlock": {"cidr": cidr}}], "ports": [{"port": 443}]}],
        }
    }

    result = render_workers(tmp_path, values)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "policy",
    [
        {"allowedCidrs": ["10.0.0.0/8"]},
        {"allowedCidrs": ["10.1.2.3/8"]},
        {"extraDeniedCidrs": ["0.0.0.0/0"]},
        {"extraDeniedCidrs": ["::/0"]},
        {"extraDeniedCidrs": ["0:0:0:0:0:0:0:0/0"]},
        {"allowedCidrs": ["192.0.2.0/24"], "extraDeniedCidrs": ["192.0.2.32/24"]},
    ],
)
def test_remote_policy_rejects_an_allowed_range_covered_by_a_denial(tmp_path: Path, policy: dict) -> None:
    values = remote_pool_values()
    values["workers"]["remote"] = {"networkPolicy": policy}
    result = render_workers(tmp_path, values)
    assert result.returncode != 0
    assert re.search(r"lies inside the denied range|CIDR with a network address", result.stderr)


@pytest.mark.parametrize(
    "cidr",
    [
        "::ffff:192.0.2.1/128",
        "::ffff:c000:201/128",
        "0:0:0:0:0:FFFF:C000:0201/128",
        "203.0.113.1/24",
        "2001:db8::1/64",
    ],
)
@pytest.mark.parametrize("surface", ["extraDeniedCidrs", "extraEgress"])
def test_remote_policy_rejects_mapped_addresses_and_host_bits(tmp_path: Path, cidr: str, surface: str) -> None:
    values = remote_pool_values()
    policy = (
        {"extraDeniedCidrs": [cidr]}
        if surface == "extraDeniedCidrs"
        else {"extraEgress": [{"to": [{"ipBlock": {"cidr": cidr}}], "ports": [{"port": 443}]}]}
    )
    values["workers"]["remote"] = {"networkPolicy": policy}
    result = render_workers(tmp_path, values)
    assert result.returncode != 0
    assert surface in result.stderr
    assert "CIDR" in result.stderr


@pytest.mark.parametrize(
    "peer",
    [
        {"ipBlock": {"cidr": "203.0.113.0/24"}, "podSelector": {"matchLabels": {"app": "proxy"}}},
        {"ipBlock": {"cidr": "203.0.113.0/24"}, "namespaceSelector": {"matchLabels": {"name": "proxy"}}},
    ],
)
def test_remote_policy_rejects_ipblock_mixed_with_selectors(tmp_path: Path, peer: dict) -> None:
    values = remote_pool_values()
    values["workers"]["remote"] = {"networkPolicy": {"extraEgress": [{"to": [peer], "ports": [{"port": 443}]}]}}
    assert "cannot combine ipBlock" in render_error(tmp_path, values)


@pytest.mark.parametrize(
    "port",
    [
        {"port": -1},
        {"port": 65536},
        {"port": 443.5},
        {"port": 443, "endPort": 0},
        {"port": 443, "endPort": 65536},
        {"port": 443, "endPort": 442},
    ],
)
def test_remote_policy_rejects_invalid_numeric_ports(tmp_path: Path, port: dict) -> None:
    values = remote_pool_values()
    values["workers"]["remote"] = {
        "networkPolicy": {"extraEgress": [{"to": [{"ipBlock": {"cidr": "203.0.113.0/24"}}], "ports": [port]}]}
    }
    assert "extraEgress[0].ports[0]" in render_error(tmp_path, values)


def test_remote_policy_accepts_combined_selectors_and_a_named_port(tmp_path: Path) -> None:
    values = remote_pool_values()
    values["workers"]["remote"] = {
        "networkPolicy": {
            "extraEgress": [
                {
                    "to": [
                        {
                            "podSelector": {"matchLabels": {"app": "proxy"}},
                            "namespaceSelector": {"matchLabels": {"name": "proxy"}},
                        }
                    ],
                    "ports": [{"port": "https"}],
                }
            ]
        }
    }
    assert render_workers(tmp_path, values).returncode == 0


def test_sie_config_receives_the_names_in_the_rendered_upstreams_file(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, remote_pool_values())

    (secret,) = [doc for doc in docs if doc["kind"] == "Secret" and doc["metadata"]["name"] == UPSTREAMS_SECRET]
    names = container_env(docs, *CONFIG_SERVICE.split("/"))["SIE_UPSTREAM_NAMES"]["value"]

    assert names.split(",") == sorted(yaml.safe_load(secret["stringData"]["upstreams.yaml"])["upstreams"])


def test_without_a_remote_lane_sie_config_receives_no_upstream_names(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, {"upstreams": upstreams_fixture()["values"], **L4_POOL})

    assert container_env(docs, *CONFIG_SERVICE.split("/"))["SIE_UPSTREAM_NAMES"]["value"] == ""


@pytest.mark.parametrize("seconds", [1, 600, 86400])
def test_remote_fallback_persistence_alert_uses_configured_threshold_and_replica_freshness(
    tmp_path: Path, seconds: int
) -> None:
    values = {
        "alertRules": {"enabled": True, "remoteFallbackPersistenceSeconds": seconds},
        "observability": AUTOSCALING_VALUES["observability"],
    }
    result = render_template(tmp_path, values, "templates/prometheusrule.yaml")
    assert result.returncode == 0, result.stderr
    document = yaml.safe_load(result.stdout)
    rule = next(
        rule
        for group in document["spec"]["groups"]
        for rule in group["rules"]
        if rule["alert"] == "SIERemoteFallbackPersistent"
    )
    assert rule["expr"].strip().endswith(f"> {seconds}")
    assert "and on (producer_instance, collector_generation, model)" in rule["expr"]
    assert 'outcome="committed"' in rule["expr"]
    assert "[5m]" in rule["expr"]
    assert "__REMOTE_FALLBACK" not in result.stdout


@pytest.mark.parametrize("seconds", [0, -1, 86401, 1.5, "600", True])
def test_remote_fallback_persistence_alert_refuses_invalid_thresholds(tmp_path: Path, seconds: object) -> None:
    result = render_template(
        tmp_path,
        {
            "alertRules": {"enabled": True, "remoteFallbackPersistenceSeconds": seconds},
            "observability": AUTOSCALING_VALUES["observability"],
        },
        "templates/prometheusrule.yaml",
    )
    assert result.returncode != 0
    assert "alertRules.remoteFallbackPersistenceSeconds" in result.stderr


THRESHOLD_VALUES = {
    "gateway": {"thresholdRouting": {"enabled": True}},
    "config": {"configStore": {"enabled": True}},
}


def test_threshold_control_is_isolated_and_gateway_only(tmp_path: Path) -> None:
    result = render_template(tmp_path, THRESHOLD_VALUES, "templates/threshold-control.yaml")
    assert result.returncode == 0, result.stderr
    docs = {doc["kind"]: doc for doc in yaml.safe_load_all(result.stdout) if doc}
    assert set(docs) == {"ConfigMap", "Deployment", "Service", "NetworkPolicy"}
    config = docs["ConfigMap"]["data"]["nats.conf"]
    fixture = (ROOT / "tools/ci/fixtures/sie-threshold-nats.conf").read_text()

    def strip_comments(text: str) -> str:
        return "\n".join(line for line in text.splitlines() if not line.startswith("#")).strip()

    assert strip_comments(config) == strip_comments(fixture)
    deployment = docs["Deployment"]["spec"]
    assert deployment["replicas"] == 1
    assert deployment["strategy"]["type"] == "Recreate"
    pod = deployment["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    (container,) = pod["containers"]
    assert container["env"] == [
        {
            "name": "SIE_NATS_AUTH_GATEWAY_PASSWORD",
            "valueFrom": {"secretKeyRef": {"name": "sie-nats-auth-gateway", "key": "password"}},
        }
    ]
    policy = docs["NetworkPolicy"]["spec"]
    assert policy["policyTypes"] == ["Ingress", "Egress"]
    assert policy["egress"] == []
    assert policy["ingress"][0]["from"][0]["podSelector"]["matchLabels"]["app.kubernetes.io/component"] == "gateway"
    assert docs["Service"]["spec"]["ports"] == [{"name": "client", "port": 4222, "targetPort": "client"}]


@pytest.mark.parametrize(
    "override",
    [
        {"nats": {"auth": {"enabled": False}}},
        {"config": {"enabled": False}},
        {"config": {"configStore": {"enabled": False}}},
        {"workers": {"common": {"workerSidecar": {"enabled": False}}}},
    ],
)
def test_threshold_rejects_incomplete_queue_authority(tmp_path: Path, override: dict) -> None:
    values = {**THRESHOLD_VALUES, **override}
    result = render_template(tmp_path, values, "templates/threshold-control.yaml")
    assert result.returncode != 0


def test_threshold_flag_is_owned_by_chart_and_propagates(tmp_path: Path) -> None:
    for template, container_name in [
        ("templates/gateway-deployment.yaml", "gateway"),
        ("templates/config-deployment.yaml", "config"),
    ]:
        result = render_template(tmp_path, THRESHOLD_VALUES, template)
        assert result.returncode == 0, result.stderr
        deployment = next(doc for doc in yaml.safe_load_all(result.stdout) if doc and doc["kind"] == "Deployment")
        container = next(c for c in deployment["spec"]["template"]["spec"]["containers"] if c["name"] == container_name)
        env = {item["name"]: item for item in container["env"]}
        assert env["SIE_THRESHOLD_ROUTING_ENABLED"]["value"] == "true"
        if container_name == "gateway":
            assert env["SIE_THRESHOLD_NATS_URL"]["value"] == "nats://sie-sie-cluster-threshold:4222"
    values = {"gateway": {"extraEnv": [{"name": "SIE_THRESHOLD_ROUTING_ENABLED", "value": "true"}]}}
    result = render_template(tmp_path, values, "templates/gateway-deployment.yaml")
    assert result.returncode != 0


def test_threshold_flag_reaches_workers_and_refuses_shared_gateway_secret(tmp_path: Path) -> None:
    docs = worker_statefulsets(tmp_path, {**THRESHOLD_VALUES, **L4_POOL})
    assert docs
    for doc in docs:
        containers = doc["spec"]["template"]["spec"]["containers"]
        for container in containers:
            env = {item["name"]: item for item in container.get("env", [])}
            if container["name"] != "worker-sidecar":
                assert env["SIE_THRESHOLD_ROUTING_ENABLED"]["value"] == "true"
                assert "SIE_THRESHOLD_NATS_URL" not in env
                assert "SIE_NATS_AUTH_GATEWAY_PASSWORD" not in env
    values = {**THRESHOLD_VALUES, "nats": {"auth": {"existingSecrets": {"gateway": "shared", "worker": "shared"}}}}
    result = render_template(tmp_path, values, "templates/threshold-control.yaml")
    assert result.returncode != 0
    assert "distinct from config and worker" in result.stderr


@pytest.mark.parametrize("fullname", ["a" * 90, "a" * 52 + "-" + "b" * 10])
def test_threshold_long_name_keeps_control_resources_and_gateway_url_aligned(tmp_path: Path, fullname: str) -> None:
    values = {**THRESHOLD_VALUES, "fullnameOverride": fullname}
    result = render_template(tmp_path, values, "templates/threshold-control.yaml")
    assert result.returncode == 0, result.stderr
    docs = [doc for doc in yaml.safe_load_all(result.stdout) if doc]
    names = {doc["metadata"]["name"] for doc in docs}
    assert len(names) == 1
    (name,) = names
    assert len(name) <= 63
    assert name == fullname[:53].rstrip("-") + "-threshold"
    deployment = next(doc for doc in docs if doc["kind"] == "Deployment")
    assert deployment["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"] == name
    gateway_result = render_template(tmp_path, values, "templates/gateway-deployment.yaml")
    assert gateway_result.returncode == 0, gateway_result.stderr
    gateway = next(doc for doc in yaml.safe_load_all(gateway_result.stdout) if doc and doc["kind"] == "Deployment")
    env = {item["name"]: item for item in gateway["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["SIE_THRESHOLD_NATS_URL"]["value"] == f"nats://{name}:4222"


GATEWAY_TOKEN_SECRET = "sie-sie-cluster-gateway-token"
WORKER_TOKEN_SECRET = "sie-sie-cluster-worker-token"
GENERATED_GATEWAY_TOKEN = {"name": GATEWAY_TOKEN_SECRET, "key": "SIE_AUTH_TOKEN"}
GENERATED_WORKER_TOKEN = {"name": WORKER_TOKEN_SECRET, "key": "SIE_AUTH_TOKEN"}
GENERATED_ADMIN_TOKEN_REF = {"tokenSecretName": GENERATED_ADMIN_TOKEN_SECRET, "tokenSecretKey": "SIE_ADMIN_TOKEN"}
GENERATED_READ_TOKEN_REF = {"tokenSecretName": GENERATED_READ_TOKEN_SECRET, "tokenSecretKey": "SIE_CONFIG_READ_TOKEN"}
L4_WORKER = "sie-sie-cluster-worker-l4-default/worker"
L4_SIDECAR = "sie-sie-cluster-worker-l4-default/worker-sidecar"
# Values as `helm upgrade --reuse-values` sees them from a release that predates
# these keys: null removes the chart default.
PREDATING_GATEWAY_AND_WORKER_TOKEN_KEYS = {
    "gateway": {"auth": {"generateToken": None}},
    "workers": {"auth": None, **L4_POOL["workers"]},
}


def secret_names(docs: list[dict]) -> set[str]:
    return {doc["metadata"]["name"] for doc in docs if doc["kind"] == "Secret"}


def generated_token(docs: list[dict], name: str, component: str) -> str:
    (secret,) = [doc for doc in docs if doc["kind"] == "Secret" and doc["metadata"]["name"] == name]
    assert secret["metadata"]["labels"]["app.kubernetes.io/component"] == component
    return secret_token(secret, "SIE_AUTH_TOKEN")


@pytest.mark.parametrize("values", [L4_POOL, PREDATING_GATEWAY_AND_WORKER_TOKEN_KEYS])
def test_gateway_and_worker_tokens_are_off_by_default(tmp_path: Path, values: dict) -> None:
    docs = rendered_documents(tmp_path, values)
    assert {GATEWAY_TOKEN_SECRET, WORKER_TOKEN_SECRET}.isdisjoint(secret_names(docs))
    assert env_entries(docs, "SIE_AUTH_MODE") == {GATEWAY: "none", L4_WORKER: "none"}
    for name in ("SIE_AUTH_TOKEN", "SIE_AUTH_TOKENS", "SIE_GATEWAY_API_KEY", "SIE_WORKER_AUTH_TOKEN"):
        assert env_entries(docs, name) == {}


def test_generated_gateway_token_with_auth_mode_none_reaches_only_the_worker_sidecars(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, {**L4_POOL, "gateway": {"auth": {"generateToken": True}}})
    generated_token(docs, GATEWAY_TOKEN_SECRET, "gateway")
    assert env_entries(docs, "SIE_AUTH_MODE") == {GATEWAY: "none", L4_WORKER: "none"}
    assert env_entries(docs, "SIE_AUTH_TOKEN") == {}
    assert env_entries(docs, "SIE_GATEWAY_API_KEY") == {L4_SIDECAR: GENERATED_GATEWAY_TOKEN}


@pytest.mark.parametrize("mode", ["token", "static"])
def test_generated_gateway_token_is_wired_to_every_consumer_with_token_auth(tmp_path: Path, mode: str) -> None:
    gateway = {"auth": {"mode": mode, "generateToken": True}}
    docs = rendered_documents(tmp_path, ingress_values(SCOPED_TLS, gateway=gateway, workers=L4_POOL["workers"]))
    generated_token(docs, GATEWAY_TOKEN_SECRET, "gateway")
    assert env_entries(docs, "SIE_AUTH_MODE") == {GATEWAY: mode, L4_WORKER: "none"}
    assert env_entries(docs, "SIE_AUTH_TOKEN") == {GATEWAY: GENERATED_GATEWAY_TOKEN}
    assert env_entries(docs, "SIE_GATEWAY_API_KEY") == {L4_SIDECAR: GENERATED_GATEWAY_TOKEN}
    assert len(gateway_ingresses(docs)) == 1


def test_operator_gateway_token_secret_keeps_its_extra_env_token_override(tmp_path: Path) -> None:
    extra_env = [{"name": "SIE_AUTH_TOKENS", "value": "override"}]
    values = {**L4_POOL, "gateway": {**GATEWAY_AUTH, "extraEnv": extra_env}}
    docs = rendered_documents(tmp_path, values)
    assert env_entries(docs, "SIE_AUTH_TOKEN") == {GATEWAY: {"name": "sie-gateway-auth", "key": "SIE_AUTH_TOKEN"}}
    assert env_entries(docs, "SIE_AUTH_TOKENS") == {GATEWAY: "override"}


@pytest.mark.parametrize("generate", [False, True])
def test_operator_gateway_token_secret_is_used_unchanged(tmp_path: Path, generate: bool) -> None:
    auth = {"mode": "static", "tokenSecretName": "sie-gateway-auth", "tokenSecretKey": "tokens"}
    docs = rendered_documents(tmp_path, {**L4_POOL, "gateway": {"auth": {**auth, "generateToken": generate}}})
    assert GATEWAY_TOKEN_SECRET not in secret_names(docs)
    operator = {"name": "sie-gateway-auth", "key": "tokens"}
    assert env_entries(docs, "SIE_AUTH_TOKEN") == {GATEWAY: operator}
    assert env_entries(docs, "SIE_GATEWAY_API_KEY") == {L4_SIDECAR: operator}


def test_worker_auth_requires_the_token_on_every_container_that_serves_the_worker_api(tmp_path: Path) -> None:
    pools = {
        "l4": {"enabled": True, "gpu": {"count": 2}},
        "cpu": {
            "enabled": True,
            "bundles": {"candle": {"engine": "candle", "minReplicas": 0, "maxReplicas": 1}},
        },
    }
    docs = rendered_documents(tmp_path, {"workers": {"auth": {"enabled": True}, "pools": pools}})
    generated_token(docs, WORKER_TOKEN_SECRET, "worker")
    servers = [
        "sie-sie-cluster-worker-l4-default/worker-0",
        "sie-sie-cluster-worker-l4-default/worker-1",
        "sie-sie-cluster-worker-cpu-default/worker",
    ]
    assert "sie-sie-cluster-worker-cpu-candle/worker" in env_entries(docs, "SIE_BUNDLE")
    assert env_entries(docs, "SIE_AUTH_TOKEN") == dict.fromkeys(servers, GENERATED_WORKER_TOKEN)
    assert env_entries(docs, "SIE_AUTH_MODE") == {GATEWAY: "none", **dict.fromkeys(servers, "token")}
    assert env_entries(docs, "SIE_WORKER_AUTH_TOKEN") == {GATEWAY: GENERATED_WORKER_TOKEN}


def test_worker_auth_covers_the_remote_lane(tmp_path: Path) -> None:
    values = remote_pool_values()
    values["workers"]["auth"] = {"enabled": True}
    tokens = env_entries(rendered_documents(tmp_path, values), "SIE_AUTH_TOKEN")
    assert tokens["sie-sie-cluster-worker-remote-remote/worker"] == GENERATED_WORKER_TOKEN


def test_worker_auth_upgrade_aid_gives_the_token_only_to_the_gateway(tmp_path: Path) -> None:
    auth = {"enabled": True, "allowUnauthenticated": True}
    docs = rendered_documents(tmp_path, {"workers": {"auth": auth, **L4_POOL["workers"]}})
    generated_token(docs, WORKER_TOKEN_SECRET, "worker")
    assert env_entries(docs, "SIE_WORKER_AUTH_TOKEN") == {GATEWAY: GENERATED_WORKER_TOKEN}
    assert env_entries(docs, "SIE_AUTH_MODE") == {GATEWAY: "none", L4_WORKER: "none"}
    assert env_entries(docs, "SIE_AUTH_TOKEN") == {}


def test_operator_worker_token_secret_is_used_unchanged(tmp_path: Path) -> None:
    auth = {"enabled": True, "tokenSecretName": "sie-worker-auth", "tokenSecretKey": "token"}
    docs = rendered_documents(tmp_path, {"workers": {"auth": auth, **L4_POOL["workers"]}})
    assert WORKER_TOKEN_SECRET not in secret_names(docs)
    operator = {"name": "sie-worker-auth", "key": "token"}
    assert env_entries(docs, "SIE_WORKER_AUTH_TOKEN") == {GATEWAY: operator}
    assert env_entries(docs, "SIE_AUTH_TOKEN") == {L4_WORKER: operator}


def test_worker_network_policy_admits_the_gateway_with_worker_auth(tmp_path: Path) -> None:
    workers = {"auth": {"enabled": True}, "networkPolicy": {"enabled": True}, **L4_POOL["workers"]}
    (policy,) = worker_network_policies(rendered_documents(tmp_path, {"workers": workers}))
    (rule,) = policy["spec"]["ingress"]
    gateway, _ = rule["from"]
    assert gateway["podSelector"]["matchLabels"]["app.kubernetes.io/component"] == "gateway"
    assert [port["port"] for port in rule["ports"]] == [8080]


@pytest.mark.parametrize(
    ("values", "error"),
    [
        (
            {"workers": {"auth": {"allowUnauthenticated": True}}},
            "workers.auth.allowUnauthenticated=true needs workers.auth.enabled=true",
        ),
        (
            {
                "gateway": {"auth": {"generateToken": True}},
                "workers": {"auth": {"enabled": True, "tokenSecretName": GATEWAY_TOKEN_SECRET}},
            },
            "The worker token and the gateway client token are the same Secret key",
        ),
        (
            {"gateway": GATEWAY_AUTH, "workers": {"auth": {"enabled": True, "tokenSecretName": "sie-gateway-auth"}}},
            "The worker token and the gateway client token are the same Secret key",
        ),
        (
            {
                "gateway": {"auth": {"adminTokenSecretName": "gateway-admin"}},
                "workers": {
                    "auth": {"enabled": True, "tokenSecretName": "gateway-admin", "tokenSecretKey": "SIE_ADMIN_TOKEN"}
                },
            },
            "The worker token and the gateway admin token are the same Secret key",
        ),
        (
            {"workers": {"auth": {"enabled": True, **GENERATED_ADMIN_TOKEN_REF}}},
            "The worker token and the sie-config admin token are the same Secret key",
        ),
        (
            {"workers": {"auth": {"enabled": True, **GENERATED_READ_TOKEN_REF}}},
            "The worker token and the sie-config read token are the same Secret key",
        ),
        (
            {"gateway": {"auth": {"mode": "token", **GENERATED_ADMIN_TOKEN_REF}}},
            "The gateway client token and the sie-config admin token are the same Secret key",
        ),
        (
            {"gateway": {"auth": {"mode": "token", **GENERATED_READ_TOKEN_REF}}},
            "The gateway client token and the sie-config read token are the same Secret key",
        ),
        (
            {
                "gateway": {
                    "auth": {
                        "generateToken": True,
                        "adminTokenSecretName": GATEWAY_TOKEN_SECRET,
                        "adminTokenSecretKey": "SIE_AUTH_TOKEN",
                    }
                }
            },
            "The gateway client token and the gateway admin token are the same Secret key",
        ),
        ({"gateway": {"auth": {"generateToken": "true"}}}, "gateway.auth.generateToken must be a boolean"),
        ({"workers": {"auth": {"enabled": "false"}}}, "workers.auth.enabled must be a boolean"),
        (
            {"workers": {"auth": {"enabled": True, "allowUnauthenticated": "true"}}},
            "workers.auth.allowUnauthenticated must be a boolean",
        ),
        ({"gateway": {"auth": {"generateToken": True, "tokenSecretKey": " "}}}, "gateway.auth.tokenSecretKey is empty"),
        ({"workers": {"auth": {"enabled": True, "tokenSecretKey": ""}}}, "workers.auth.tokenSecretKey is empty"),
        (
            {
                "gateway": {"extraEnv": [{"name": "SIE_WORKER_AUTH_TOKEN", "value": "override"}]},
                "workers": {"auth": {"enabled": True}},
            },
            "gateway.extraEnv must not override chart-owned variable SIE_WORKER_AUTH_TOKEN",
        ),
        (
            {
                "workers": {
                    "auth": {"enabled": True},
                    "common": {"extraEnv": [{"name": "SIE_AUTH_MODE", "value": "none"}]},
                }
            },
            "workers.common.extraEnv must not override chart-owned variable SIE_AUTH_MODE",
        ),
        (
            {
                "workers": {
                    "auth": {"enabled": True},
                    "common": {"extraEnv": [{"name": "SIE_AUTH_TOKENS", "value": "override"}]},
                }
            },
            "workers.common.extraEnv must not override chart-owned variable SIE_AUTH_TOKENS",
        ),
        (
            {
                "gateway": {
                    "auth": {"mode": "token", "generateToken": True},
                    "extraEnv": [{"name": "SIE_AUTH_TOKENS", "value": "override"}],
                }
            },
            "gateway.extraEnv must not override chart-owned variable SIE_AUTH_TOKENS",
        ),
        (
            {
                "gateway": {"auth": {"generateToken": True}},
                "workers": {
                    "common": {"hfCache": {"tokenSecret": GATEWAY_TOKEN_SECRET, "tokenSecretKey": "SIE_AUTH_TOKEN"}}
                },
            },
            "The gateway client token and the Hugging Face token are the same Secret key",
        ),
        (
            {
                "workers": {
                    "auth": {"enabled": True, "tokenSecretName": "sie-nats-auth-worker", "tokenSecretKey": "password"}
                }
            },
            "The worker token and the NATS worker password are the same Secret key",
        ),
    ],
)
def test_unsafe_gateway_and_worker_token_values_fail_the_render(tmp_path: Path, values: dict, error: str) -> None:
    workers = {**L4_POOL["workers"], **values.get("workers", {})}
    assert error in render_error(tmp_path, {**values, "workers": workers})


@pytest.mark.parametrize(
    ("values", "api_key_secret", "owner"),
    [
        (
            {"gateway": {"auth": {"generateToken": True}}},
            {"name": GATEWAY_TOKEN_SECRET, "key": "SIE_AUTH_TOKEN"},
            "the generated gateway client token",
        ),
        (
            {"workers": {"auth": {"enabled": True}}},
            {"name": WORKER_TOKEN_SECRET, "key": "SIE_AUTH_TOKEN"},
            "the generated worker token",
        ),
        (
            {"workers": {"auth": {"enabled": True, "tokenSecretName": "team-upstream", "tokenSecretKey": "api-key"}}},
            {"name": "team-upstream", "key": "api-key"},
            "workers.auth.tokenSecretName",
        ),
    ],
)
def test_an_upstream_key_cannot_be_the_gateway_client_or_worker_token(
    tmp_path: Path, values: dict, api_key_secret: dict, owner: str
) -> None:
    result = render_workers(tmp_path, {**values, "upstreams": {"team": upstream(api_key_secret=api_key_secret)}})
    assert result.returncode != 0
    assert f"upstreams.team.api_key_secret and {owner} name the same Secret key" in result.stderr


WORKER_AUTH_TEST_JOB = "sie-sie-cluster-worker-auth-test-1"


def worker_with(**auth: object) -> dict:
    return {"auth": {"enabled": True, **auth}, **L4_POOL["workers"]}


@pytest.mark.parametrize(
    ("values", "error"),
    [
        (
            {"workers": worker_with(tokenSecretName="oauth2-proxy", tokenSecretKey="OAUTH2_PROXY_COOKIE_SECRET")},
            "The worker token and the oauth2-proxy cookieSecretKey are the same Secret key",
        ),
        (
            {
                "gateway": {
                    "auth": {
                        "mode": "token",
                        "tokenSecretName": "oauth2-proxy",
                        "tokenSecretKey": "OAUTH2_PROXY_CLIENT_SECRET",
                    }
                }
            },
            "The gateway client token and the oauth2-proxy clientSecretKey are the same Secret key",
        ),
        (
            {
                "global": {"imagePullSecrets": ["regcred"]},
                "workers": worker_with(tokenSecretName="regcred", tokenSecretKey=".dockerconfigjson"),
            },
            "The worker token and the image pull secret are the same Secret key",
        ),
        (
            {
                "mcpEdge": {"existingSecretName": "mcp-conn"},
                "gateway": {
                    "auth": {"mode": "token", "tokenSecretName": "mcp-conn", "tokenSecretKey": "connector-secrets"}
                },
            },
            "The gateway client token and the MCP connector secrets are the same Secret key",
        ),
        (
            {
                "observability": {"otel": {"collector": {"betterStack": {"existingSecret": "better-stack"}}}},
                "workers": worker_with(tokenSecretName="better-stack", tokenSecretKey="token"),
            },
            "The worker token and the Better Stack token are the same Secret key",
        ),
        (
            {
                "mcpEdge": {"clusterApiKey": {"existingSecretName": "mcp-key"}},
                "workers": worker_with(tokenSecretName="mcp-key", tokenSecretKey="api-key"),
            },
            "The worker token and the MCP edge cluster API key are the same Secret key",
        ),
        (
            {
                "mcpEdge": {"clusterApiKey": {"existingSecretName": "hf", "secretKey": "token"}},
                "workers": {"common": {"hfCache": {"tokenSecret": "hf", "tokenSecretKey": "token"}}},
            },
            "The MCP edge cluster API key and the Hugging Face token are the same Secret key",
        ),
        (
            {"workers": worker_with(tokenSecretName="sie-sie-cluster-mcp-tls", tokenSecretKey="tls.key")},
            "The worker token and the MCP ingress TLS certificate are the same Secret key",
        ),
        (
            {
                "gateway": {
                    "auth": {"mode": "token"},
                    "extraEnv": [
                        {
                            "name": "SIE_AUTH_TOKENS",
                            "valueFrom": {"secretKeyRef": {"name": "client-tokens", "key": "tokens"}},
                        }
                    ],
                },
                "workers": worker_with(tokenSecretName="client-tokens", tokenSecretKey="tokens"),
            },
            "The worker token and the gateway client token from gateway.extraEnv SIE_AUTH_TOKENS are the same",
        ),
        (
            {
                "gateway": {
                    "auth": {"mode": "token"},
                    "extraEnv": [
                        {
                            "name": "SIE_AUTH_TOKEN",
                            "valueFrom": {"secretKeyRef": {"name": "sie-nats-auth-gateway", "key": "password"}},
                        }
                    ],
                }
            },
            "gateway.extraEnv SIE_AUTH_TOKEN and the NATS gateway password are the same Secret key",
        ),
    ],
)
def test_client_and_worker_tokens_are_kept_apart_from_every_wired_secret_key(
    tmp_path: Path, values: dict, error: str
) -> None:
    workers = {**L4_POOL["workers"], **values.get("workers", {})}
    assert error in render_error(tmp_path, {**values, "workers": workers})


@pytest.mark.parametrize("credential", ["worker", "gateway", "upstream"])
@pytest.mark.parametrize(
    ("grafana", "secret_name", "secret_key"),
    [
        ({}, "sie-grafana", "admin-password"),
        ({"admin": {"existingSecret": "grafana-admin"}}, "grafana-admin", "admin-user"),
        (
            {"admin": {"existingSecret": "{{ .Release.Name }}-admin", "passwordKey": "password"}},
            "sie-admin",
            "password",
        ),
        (
            {"fullnameOverride": "dashboard", "admin": {"existingSecret": "{{ .Values.fullnameOverride }}"}},
            "dashboard",
            "admin-password",
        ),
        (
            {"admin": {"existingSecret": "{{ .Values.admin.userKey }}-admin", "userKey": "dashboard"}},
            "dashboard-admin",
            "dashboard",
        ),
    ],
)
def test_inference_credentials_cannot_reuse_bundled_grafana_admin_keys(
    tmp_path: Path, credential: str, grafana: dict, secret_name: str, secret_key: str
) -> None:
    values = {"kube-prometheus-stack": {"install": True, "grafana": grafana}}
    if credential == "worker":
        values["workers"] = worker_with(tokenSecretName=secret_name, tokenSecretKey=secret_key)
    elif credential == "gateway":
        values["gateway"] = {"auth": {"mode": "token", "tokenSecretName": secret_name, "tokenSecretKey": secret_key}}
    else:
        values["upstreams"] = {"team": upstream(api_key_secret={"name": secret_name, "key": secret_key})}
    error = render_error(tmp_path, values)
    assert "Grafana admin" in error
    if credential == "upstream":
        assert "upstreams.team.api_key_secret" in error
    else:
        assert f"{secret_name}/{secret_key}" in error


@pytest.mark.parametrize(
    "stack",
    [
        {"install": False},
        {"install": True, "grafana": {"enabled": False}},
        {"install": True, "grafana": {"namespaceOverride": "monitoring"}},
        {"install": True, "grafana": {"env": {"GF_SECURITY_ADMIN_PASSWORD": "fixture-password"}}},
        {"install": True, "grafana": {"env": {"GF_SECURITY_ADMIN_PASSWORD__FILE": "/etc/grafana/password"}}},
        {"install": True, "grafana": {"env": {"GF_SECURITY_DISABLE_INITIAL_ADMIN_CREATION": "true"}}},
    ],
)
def test_grafana_admin_keys_only_reserve_credentials_actually_wired_in_the_worker_namespace(
    tmp_path: Path, stack: dict
) -> None:
    values = {
        "kube-prometheus-stack": stack,
        "workers": worker_with(tokenSecretName="sie-grafana", tokenSecretKey="admin-password"),
    }
    result = render_workers(tmp_path, values)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("grafana_namespace", ["sie", "workers"])
def test_grafana_admin_collision_uses_the_effective_worker_namespace(tmp_path: Path, grafana_namespace: str) -> None:
    values = {
        "global": {"namespace": "workers"},
        "kube-prometheus-stack": {"install": True, "grafana": {"namespaceOverride": grafana_namespace}},
        "workers": worker_with(tokenSecretName="sie-grafana", tokenSecretKey="admin-password"),
    }
    result = render_workers(tmp_path, values)
    if grafana_namespace == "workers":
        assert result.returncode != 0
        assert "The worker token and the Grafana admin password are the same Secret key" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


def test_the_mcp_edge_may_present_the_gateway_client_token(tmp_path: Path) -> None:
    values = {
        **L4_POOL,
        "gateway": {"auth": {"mode": "token", "generateToken": True}},
        "mcpEdge": {
            "enabled": True,
            "clusterApiKey": {"existingSecretName": GATEWAY_TOKEN_SECRET, "secretKey": "SIE_AUTH_TOKEN"},
        },
    }
    docs = rendered_documents(tmp_path, values)
    assert env_entries(docs, "SIE_API_KEY") == {"sie-sie-cluster-mcp/mcp-edge": GENERATED_GATEWAY_TOKEN}


def test_an_upstream_key_cannot_be_a_gateway_extra_env_token(tmp_path: Path) -> None:
    secret_ref = {"secretKeyRef": {"name": "team-upstream", "key": "api-key"}}
    values = {
        "gateway": {"auth": {"mode": "token"}, "extraEnv": [{"name": "SIE_AUTH_TOKENS", "valueFrom": secret_ref}]},
        "upstreams": {"team": upstream(api_key_secret={"name": "team-upstream", "key": "api-key"})},
    }
    result = render_workers(tmp_path, values)
    assert result.returncode != 0
    assert (
        "upstreams.team.api_key_secret and gateway.extraEnv SIE_AUTH_TOKENS name the same Secret key" in result.stderr
    )


@pytest.mark.parametrize(
    "workers",
    [
        L4_POOL["workers"],
        {"auth": {"enabled": True, "allowUnauthenticated": True}, **L4_POOL["workers"]},
    ],
)
def test_workers_that_do_not_require_the_token_are_explicitly_open(tmp_path: Path, workers: dict) -> None:
    pools = {
        **workers["pools"],
        "cpu": {"enabled": True, "bundles": {"candle": {"engine": "candle", "minReplicas": 0, "maxReplicas": 1}}},
    }
    docs = rendered_documents(tmp_path, {"workers": {**workers, "pools": pools}})
    assert "sie-sie-cluster-worker-cpu-candle/worker" in env_entries(docs, "SIE_BUNDLE")
    cpu_worker = "sie-sie-cluster-worker-cpu-default/worker"
    assert env_entries(docs, "SIE_AUTH_MODE") == {GATEWAY: "none", L4_WORKER: "none", cpu_worker: "none"}
    assert env_entries(docs, "SIE_AUTH_TOKEN") == {}


def test_an_operator_worker_token_in_extra_env_is_not_overridden(tmp_path: Path) -> None:
    extra_env = [{"name": "SIE_AUTH_TOKEN", "valueFrom": {"secretKeyRef": {"name": "own-worker", "key": "token"}}}]
    docs = rendered_documents(tmp_path, {"workers": {"common": {"extraEnv": extra_env}, **L4_POOL["workers"]}})
    assert env_entries(docs, "SIE_AUTH_MODE") == {GATEWAY: "none"}
    assert env_entries(docs, "SIE_AUTH_TOKEN") == {L4_WORKER: {"name": "own-worker", "key": "token"}}


@pytest.mark.parametrize("name", ["SIE_AUTH_MODE", "SIE_AUTH_TOKEN", "SIE_AUTH_TOKENS"])
def test_worker_auth_variables_are_reserved_in_the_upgrade_step(tmp_path: Path, name: str) -> None:
    workers = {
        "auth": {"enabled": True, "allowUnauthenticated": True},
        "common": {"extraEnv": [{"name": name, "value": "override"}]},
        **L4_POOL["workers"],
    }
    error = render_error(tmp_path, {"workers": workers})
    assert f"workers.common.extraEnv must not override chart-owned variable {name}" in error


def test_the_sidecar_gateway_key_is_reserved_while_the_chart_wires_it(tmp_path: Path) -> None:
    sidecar = {"workerSidecar": {"extraEnv": [{"name": "SIE_GATEWAY_API_KEY", "value": "override"}]}}
    workers = {"common": sidecar, **L4_POOL["workers"]}
    error = render_error(tmp_path, {"gateway": {"auth": {"generateToken": True}}, "workers": workers})
    assert "workers.common.workerSidecar.extraEnv must not override chart-owned variable SIE_GATEWAY_API_KEY" in error
    docs = rendered_documents(tmp_path, {"workers": workers})
    assert env_entries(docs, "SIE_GATEWAY_API_KEY") == {L4_SIDECAR: "override"}


MCP_TOKEN_AUTH = {"gateway": {"auth": {"mode": "token", "generateToken": True}}, "mcpEdge": {"enabled": True}}


def test_the_mcp_edge_needs_a_gateway_token_when_the_gateway_requires_one(tmp_path: Path) -> None:
    error = render_error(tmp_path, {**L4_POOL, **MCP_TOKEN_AUTH})
    assert "mcpEdge.enabled with gateway token auth" in error
    assert "needs mcpEdge.clusterApiKey.existingSecretName" in error
    assert f"mcpEdge.clusterApiKey.existingSecretName={GATEWAY_TOKEN_SECRET}" in error


RUNTIME_GATEWAY_MODE = {
    "auth": {"generateToken": True},
    "extraEnv": [{"name": "SIE_AUTH_MODE", "valueFrom": {"configMapKeyRef": {"name": "gateway-mode", "key": "mode"}}}],
}


def test_the_mcp_edge_needs_a_gateway_token_when_the_gateway_mode_is_known_only_at_run_time(tmp_path: Path) -> None:
    error = render_error(tmp_path, {**L4_POOL, "gateway": RUNTIME_GATEWAY_MODE, "mcpEdge": {"enabled": True}})
    assert "or a gateway auth mode known only at run time" in error
    assert f"mcpEdge.clusterApiKey.existingSecretName={GATEWAY_TOKEN_SECRET}" in error


def rendered_notes(tmp_path: Path, values: dict) -> str:
    chart = tmp_path / "notes-chart"
    shutil.copytree(helm.CHART_DIR, chart)
    notes = (chart / "templates" / "NOTES.txt").read_text(encoding="utf-8")
    (chart / "templates" / "NOTES.txt").unlink()
    (chart / "templates" / "_notes_under_test.tpl").write_text(
        '{{- define "notes.under.test" -}}\n' + notes + "\n{{- end -}}\n", encoding="utf-8"
    )
    (chart / "templates" / "notes-under-test.yaml").write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: notes-under-test\n"
        'data:\n  notes: {{ include "notes.under.test" . | quote }}\n',
        encoding="utf-8",
    )
    values_file = tmp_path / "notes-values.yaml"
    values_file.write_text(yaml.safe_dump(values), encoding="utf-8")
    result = subprocess.run(
        [
            "mise",
            "exec",
            "--",
            "helm",
            "template",
            "sie",
            str(chart),
            "--namespace",
            "sie",
            "-f",
            str(values_file),
            "--show-only",
            "templates/notes-under-test.yaml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return yaml.safe_load(result.stdout)["data"]["notes"]


def test_the_notes_say_when_the_gateway_mode_is_known_only_at_run_time(tmp_path: Path) -> None:
    notes = rendered_notes(tmp_path, {**L4_POOL, "gateway": RUNTIME_GATEWAY_MODE})
    assert "known only at run time" in notes
    assert "does not require this token yet" not in notes
    assert "does not require this token yet (gateway auth mode none)" in rendered_notes(
        tmp_path / "none", {**L4_POOL, "gateway": {"auth": {"generateToken": True}}}
    )


@pytest.mark.parametrize(
    "values",
    [
        {"gateway": {"auth": {"generateToken": True}}, "mcpEdge": {"enabled": True}},
        {**MCP_TOKEN_AUTH, "mcpEdge": {"enabled": True, "clusterBaseUrl": "https://gateway.example.com"}},
        {
            **MCP_TOKEN_AUTH,
            "mcpEdge": {"enabled": True, "clusterApiKey": {"existingSecretName": "mcp-key", "secretKey": "token"}},
        },
    ],
)
def test_the_mcp_edge_renders_when_it_has_what_its_gateway_needs(tmp_path: Path, values: dict) -> None:
    rendered_documents(tmp_path, {**L4_POOL, **values})


def test_a_gateway_mode_from_value_from_gets_the_generated_token(tmp_path: Path) -> None:
    mode = {"name": "SIE_AUTH_MODE", "valueFrom": {"configMapKeyRef": {"name": "gateway-auth", "key": "mode"}}}
    docs = rendered_documents(tmp_path, {**L4_POOL, "gateway": {"auth": {"generateToken": True}, "extraEnv": [mode]}})
    assert env_entries(docs, "SIE_AUTH_TOKEN") == {GATEWAY: GENERATED_GATEWAY_TOKEN}


def hook_jobs(docs: list[dict]) -> dict[str, dict]:
    return {doc["metadata"]["name"]: doc for doc in docs if doc["kind"] == "Job"}


def test_enforced_worker_auth_adds_a_release_test_of_every_api_lane(tmp_path: Path) -> None:
    pools = {
        "l4": {"enabled": True},
        "cpu": {"enabled": True, "bundles": {"candle": {"engine": "candle", "minReplicas": 0, "maxReplicas": 1}}},
    }
    docs = rendered_documents(tmp_path, {"workers": {"auth": {"enabled": True}, "pools": pools}})
    job = hook_jobs(docs)[WORKER_AUTH_TEST_JOB]
    assert job["metadata"]["annotations"]["helm.sh/hook"] == "post-install,post-upgrade"
    (container,) = job["spec"]["template"]["spec"]["containers"]
    script = container["args"][0]
    assert 'LANES="sie-sie-cluster-worker-cpu-default sie-sie-cluster-worker-l4-default"' in script
    assert "candle" not in script
    assert "http://$host:$PORT/v1/models" in script
    assert "401)" in script
    assert "exit 1" in script
    (role,) = [
        doc for doc in docs if doc["kind"] == "Role" and doc["metadata"]["name"] == "sie-sie-cluster-worker-auth-test"
    ]
    assert role["rules"] == [
        {
            "apiGroups": ["apps"],
            "resources": ["statefulsets"],
            "resourceNames": ["sie-sie-cluster-worker-cpu-default", "sie-sie-cluster-worker-l4-default"],
            "verbs": ["get"],
        },
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["list"]},
    ]


def test_the_worker_auth_test_job_is_kept_and_ends_well_before_its_deadline(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, {"workers": {"auth": {"enabled": True}, **L4_POOL["workers"]}})
    job = hook_jobs(docs)[WORKER_AUTH_TEST_JOB]
    assert job["metadata"]["annotations"]["helm.sh/hook-delete-policy"] == "before-hook-creation"
    assert job["spec"]["ttlSecondsAfterFinished"] >= 3600
    (wait,) = re.findall(
        r"DEADLINE=\$\(\( \$\(date \+%s\) \+ (\d+) \)\)", job["spec"]["template"]["spec"]["containers"][0]["args"][0]
    )
    assert job["spec"]["activeDeadlineSeconds"] - int(wait) >= 120


@pytest.mark.parametrize(
    "workers",
    [L4_POOL["workers"], {"auth": {"enabled": True, "allowUnauthenticated": True}, **L4_POOL["workers"]}],
)
def test_there_is_no_worker_auth_test_while_workers_do_not_require_the_token(tmp_path: Path, workers: dict) -> None:
    assert WORKER_AUTH_TEST_JOB not in hook_jobs(rendered_documents(tmp_path, {"workers": workers}))


SERVER_SOURCES = ROOT / "packages" / "sie_server" / "src"
GATEWAY_SOURCES = ROOT / "packages" / "sie_gateway" / "src"


def source_text(root: Path, suffix: str) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(root.rglob(f"*{suffix}")))


def test_the_worker_auth_variables_the_chart_renders_are_read_by_the_images(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, {"workers": {"auth": {"enabled": True}, **L4_POOL["workers"]}})
    worker_names = set(container_env(docs, "sie-sie-cluster-worker-l4-default", "worker")) & {
        "SIE_AUTH_MODE",
        "SIE_AUTH_TOKEN",
    }
    gateway_names = set(container_env(docs, "sie-sie-cluster-gateway", "gateway")) & {"SIE_WORKER_AUTH_TOKEN"}
    assert worker_names == {"SIE_AUTH_MODE", "SIE_AUTH_TOKEN"}
    assert gateway_names == {"SIE_WORKER_AUTH_TOKEN"}
    server = source_text(SERVER_SOURCES, ".py")
    gateway = source_text(GATEWAY_SOURCES, ".rs")
    for name in sorted(worker_names):
        assert f'"{name}"' in server, f"sie-server does not read {name}"
    for name in sorted(gateway_names):
        assert f'"{name}"' in gateway, f"sie-gateway does not read {name}"


AUTH_TEST_POD_LABELS = {
    "app.kubernetes.io/name": "sie-health-test",
    "app.kubernetes.io/component": "worker-auth",
    "app.kubernetes.io/instance": "sie",
}


def ingress_peers(docs: list[dict], policy_name: str) -> list[dict]:
    (policy,) = [doc for doc in docs if doc["kind"] == "NetworkPolicy" and doc["metadata"]["name"] == policy_name]
    (rule,) = policy["spec"]["ingress"]
    return [peer["podSelector"]["matchLabels"] for peer in rule["from"]]


@pytest.mark.parametrize(
    ("auth", "admitted"),
    [({"enabled": True}, True), ({"enabled": True, "allowUnauthenticated": True}, False), ({}, False)],
)
def test_worker_policies_admit_the_auth_test_only_while_workers_require_the_token(
    tmp_path: Path, auth: dict, admitted: bool
) -> None:
    values = remote_pool_values()
    values["workers"].update({"auth": auth, "networkPolicy": {"enabled": True}})
    values["workers"]["pools"]["l4"] = {"enabled": True}
    docs = rendered_documents(tmp_path, values)
    for policy in ("sie-sie-cluster-worker", "sie-sie-cluster-worker-remote"):
        assert (AUTH_TEST_POD_LABELS in ingress_peers(docs, policy)) is admitted, policy
    if admitted:
        pod_labels = hook_jobs(docs)[WORKER_AUTH_TEST_JOB]["spec"]["template"]["metadata"]["labels"]
        assert pod_labels == AUTH_TEST_POD_LABELS


def test_the_worker_auth_test_fails_closed_and_reports_what_it_checked(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, {"workers": {"auth": {"enabled": True}, **L4_POOL["workers"]}})
    script = hook_jobs(docs)[WORKER_AUTH_TEST_JOB]["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert 'echo "FAILED: StatefulSet $lane not found."' in script
    assert 'echo "FAILED: $lane has ready workers that did not answer from this pod."' in script
    assert 'echo "WARNING: no worker was checked. Every lane was scaled to zero or still starting."' in script
    assert '000|"")' in script


WORKER_AUTH_KUBECTL = """
import json, os, sys

args = sys.argv[1:]
with open(os.path.join(os.environ["STUB_DIR"], "kubectl.log"), "a") as log:
    log.write(" ".join(args) + "\\n")
output = args[args.index("-o") + 1] if "-o" in args else ""
if args[:2] == ["get", "statefulset"]:
    if os.environ["STUB_STATEFULSET"] != "present":
        sys.exit(1)
    answers = {
        "jsonpath={.spec.replicas}": os.environ["STUB_REPLICAS"],
        "jsonpath={.status.updateRevision}": "rev-2",
    }
    if output not in answers:
        sys.exit(2)
    print(answers[output], end="")
elif args[:2] == ["get", "pods"] and output == "json" and "-l" in args:
    if os.environ.get("STUB_PODS_FAIL"):
        sys.exit(1)
    key, _, value = args[args.index("-l") + 1].partition("=")
    items = json.loads(os.environ["STUB_PODS"])["items"]
    print(json.dumps({"items": [pod for pod in items if pod["metadata"].get("labels", {}).get(key) == value]}))
else:
    sys.exit(2)
"""
WORKER_AUTH_CURL = """
import json, os, sys

host = sys.argv[-1].removeprefix("http://").removesuffix("/v1/models").rpartition(":")[0]
with open(os.path.join(os.environ["STUB_DIR"], "curl.log"), "a") as log:
    log.write(host + "\\n")
print(json.loads(os.environ["STUB_HTTP_CODES"]).get(host, os.environ["STUB_HTTP_CODE"]), end="")
"""
WORKER_AUTH_SHELL_STUBS = {
    "date": """#!/bin/sh
count=$(cat "$STUB_DIR/date-calls" 2>/dev/null || echo 0)
echo $((count + 1)) > "$STUB_DIR/date-calls"
if [ "$count" -eq 0 ]; then echo 0; else echo 100000; fi
""",
    "sleep": '#!/bin/sh\necho "$@" >> "$STUB_DIR/sleep.log"\n',
}


def worker_pod(ip: str, revision: str = "rev-2", ready: bool = True, terminating: bool = False) -> dict:
    metadata: dict = {"labels": {"controller-revision-hash": revision}}
    if terminating:
        metadata["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    return {
        "metadata": metadata,
        "status": {"podIP": ip, "conditions": [{"type": "Ready", "status": "True" if ready else "False"}]},
    }


READY_POD = {"items": [worker_pod("10.0.0.7")]}
MIXED_PODS = {
    "items": [
        worker_pod("10.0.0.1"),
        worker_pod("10.0.0.2", ready=False),
        worker_pod("10.0.0.3", terminating=True),
        worker_pod("10.0.0.4", revision="rev-1"),
        worker_pod("fd00::5"),
    ]
}
MIXED_POD_CODES = {"10.0.0.1": "401", "10.0.0.2": "200", "10.0.0.3": "200", "10.0.0.4": "200", "[fd00::5]": "401"}


def pipefail_shell() -> str:
    """A POSIX shell that supports pipefail, as the hook image's BusyBox ash does."""
    for shell in ("/bin/sh", shutil.which("bash") or "/bin/bash"):
        if subprocess.run([shell, "-c", "set -o pipefail"], capture_output=True, check=False).returncode == 0:
            return shell
    pytest.fail("no POSIX shell with pipefail support")


def worker_auth_test_script(tmp_path: Path) -> str:
    docs = rendered_documents(tmp_path, {"workers": {"auth": {"enabled": True}, **L4_POOL["workers"]}})
    return hook_jobs(docs)[WORKER_AUTH_TEST_JOB]["spec"]["template"]["spec"]["containers"][0]["args"][0]


def run_worker_auth_test(
    tmp_path: Path,
    statefulset: str,
    pods: dict,
    http_code: str,
    *,
    replicas: str = "1",
    http_codes: dict | None = None,
    pods_fail: bool = False,
) -> subprocess.CompletedProcess:
    script = worker_auth_test_script(tmp_path)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, body in {
        "kubectl": f"#!{sys.executable}\n{WORKER_AUTH_KUBECTL}",
        "curl": f"#!{sys.executable}\n{WORKER_AUTH_CURL}",
        **WORKER_AUTH_SHELL_STUBS,
    }.items():
        (stubs / name).write_text(body, encoding="utf-8")
        (stubs / name).chmod(0o755)
    env = {
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "STUB_DIR": str(stubs),
        "STUB_STATEFULSET": statefulset,
        "STUB_REPLICAS": replicas,
        "STUB_PODS": json.dumps(pods),
        "STUB_HTTP_CODE": http_code,
        "STUB_HTTP_CODES": json.dumps(http_codes or {}),
        **({"STUB_PODS_FAIL": "1"} if pods_fail else {}),
    }
    script = script.replace("KUBECONFIG=/tmp/sie-in-cluster-kubeconfig", f"KUBECONFIG={tmp_path}/kubeconfig")
    return subprocess.run(
        [pipefail_shell(), "-c", script], env=env, capture_output=True, text=True, check=False, timeout=60
    )


def stub_log(tmp_path: Path, name: str) -> list[str]:
    path = tmp_path / "stubs" / f"{name}.log"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


@pytest.mark.parametrize(
    ("statefulset", "pods", "http_code", "exit_code", "message"),
    [
        ("present", READY_POD, "401", 0, "SUCCESS: 1 worker(s) refused a request without a token."),
        ("present", READY_POD, "200", 1, "answered GET /v1/models without a token (HTTP 200)"),
        ("present", READY_POD, "404", 1, "answered GET /v1/models without a token (HTTP 404)"),
        ("present", READY_POD, "000", 1, "has ready workers that did not answer from this pod"),
        ("missing", READY_POD, "401", 1, "StatefulSet sie-sie-cluster-worker-l4-default not found"),
        ("present", {"items": []}, "401", 0, "WARNING: no worker was checked."),
    ],
)
def test_the_worker_auth_test_script_decides_on_what_it_observes(
    tmp_path: Path, statefulset: str, pods: dict, http_code: str, exit_code: int, message: str
) -> None:
    result = run_worker_auth_test(tmp_path, statefulset, pods, http_code)
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert message in result.stdout


def test_the_worker_auth_test_checks_only_ready_workers_of_the_current_revision(tmp_path: Path) -> None:
    result = run_worker_auth_test(tmp_path, "present", MIXED_PODS, "000", http_codes=MIXED_POD_CODES)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SUCCESS: 2 worker(s) refused a request without a token." in result.stdout
    assert sorted(stub_log(tmp_path, "curl")) == ["10.0.0.1", "[fd00::5]"]
    assert "get pods -l controller-revision-hash=rev-2 -o json" in stub_log(tmp_path, "kubectl")


def test_the_worker_auth_test_skips_a_lane_scaled_to_zero_at_once(tmp_path: Path) -> None:
    result = run_worker_auth_test(tmp_path, "present", READY_POD, "200", replicas="0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SKIPPED: sie-sie-cluster-worker-l4-default is scaled to zero." in result.stdout
    assert "WARNING: no worker was checked." in result.stdout
    assert not [call for call in stub_log(tmp_path, "kubectl") if call.startswith("get pods")]
    assert stub_log(tmp_path, "sleep") == []
    assert stub_log(tmp_path, "curl") == []


def test_the_worker_auth_test_fails_when_it_cannot_list_the_pods(tmp_path: Path) -> None:
    result = run_worker_auth_test(tmp_path, "present", READY_POD, "401", pods_fail=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAILED: cannot list the pods of sie-sie-cluster-worker-l4-default." in result.stdout
    assert "WARNING" not in result.stdout


@pytest.mark.parametrize("name", ["SIE_AUTH_MODE", "SIE_AUTH_TOKEN", "SIE_AUTH_TOKENS"])
def test_worker_auth_variables_are_reserved_for_bundles_in_the_upgrade_step(tmp_path: Path, name: str) -> None:
    bundle = {"default": {"extraEnv": [{"name": name, "value": "override"}]}}
    workers = {
        "auth": {"enabled": True, "allowUnauthenticated": True},
        "pools": {"l4": {"enabled": True, "bundles": bundle}},
    }
    error = render_error(tmp_path, {"workers": workers})
    assert f"workers.pools.l4.bundles.default.extraEnv must not override chart-owned variable {name}" in error


def chart_app_version() -> str:
    return yaml.safe_load((helm.CHART_DIR / "Chart.yaml").read_text(encoding="utf-8"))["appVersion"]


def older_version(version: str) -> str:
    major, minor, patch = (int(part) for part in version.lstrip("v").split("."))
    return f"v{major}.{minor}.{patch - 1}" if patch else f"v{major}.{minor - 1}.0" if minor else f"v{major - 1}.0.0"


def worker_auth_requirements(docs: list[dict]) -> dict[str, list[str]]:
    return {
        f"{doc['metadata']['name']}/{container['name']}": requirements
        for doc in docs
        if doc["kind"] in {"Deployment", "StatefulSet"}
        for container in doc["spec"]["template"]["spec"]["containers"]
        if (requirements := [arg for arg in container.get("args", []) if arg.startswith("--require-auth-mode=")])
    }


@pytest.mark.parametrize(
    ("auth", "tag"),
    [
        ({"enabled": True}, None),
        ({"enabled": True}, "same"),
        ({"enabled": True}, "v99.0.0"),
        ({"enabled": True}, "main"),
        ({"enabled": True}, "older"),
        ({}, "older"),
    ],
)
def test_worker_auth_requires_runtime_capability_independently_of_image_tag(
    tmp_path: Path, auth: dict, tag: str | None
) -> None:
    version = chart_app_version()
    tags = {"same": version, "older": older_version(version)}
    common = {"image": {"tag": tags.get(tag, tag)}} if tag else {}
    docs = rendered_documents(tmp_path, {"workers": {"auth": auth, "common": common, **L4_POOL["workers"]}})
    expected = {L4_WORKER: ["--require-auth-mode=token"]} if auth.get("enabled") else {}
    assert worker_auth_requirements(docs) == expected


@pytest.mark.parametrize("tag", [None, "v0.10.1"], ids=["default-image", "explicit-v0.10.1"])
@pytest.mark.parametrize("allow_unauthenticated", [False, True], ids=["enforced", "upgrade-step"])
def test_worker_auth_requires_capability_for_default_and_legacy_image_tags(
    tmp_path: Path, tag: str | None, allow_unauthenticated: bool
) -> None:
    common = {"image": {"tag": tag}} if tag is not None else {}
    workers = {
        "auth": {"enabled": True, "allowUnauthenticated": allow_unauthenticated},
        "common": common,
        **L4_POOL["workers"],
    }
    docs = rendered_documents(tmp_path, {"workers": workers})
    mode = "none" if allow_unauthenticated else "token"
    assert worker_auth_requirements(docs) == {L4_WORKER: [f"--require-auth-mode={mode}"]}


@pytest.mark.parametrize("allow_unauthenticated", [False, True])
def test_worker_auth_capability_covers_every_python_child_even_at_zero_replicas(
    tmp_path: Path, allow_unauthenticated: bool
) -> None:
    pools = {
        "l4": {
            "enabled": True,
            "gpu": {"count": 2},
            "bundles": {"default": {"minReplicas": 0, "maxReplicas": 1}},
        },
        "cpu": {
            "enabled": True,
            "bundles": {"candle": {"engine": "candle", "minReplicas": 0, "maxReplicas": 1}},
        },
    }
    auth = {"enabled": True, "allowUnauthenticated": allow_unauthenticated}
    docs = rendered_documents(tmp_path, {"workers": {"auth": auth, "pools": pools}})
    mode = "none" if allow_unauthenticated else "token"
    assert worker_auth_requirements(docs) == {
        "sie-sie-cluster-worker-l4-default/worker-0": [f"--require-auth-mode={mode}"],
        "sie-sie-cluster-worker-l4-default/worker-1": [f"--require-auth-mode={mode}"],
        "sie-sie-cluster-worker-cpu-default/worker": [f"--require-auth-mode={mode}"],
    }
    (lane,) = [doc for doc in docs if doc["kind"] == "StatefulSet" and doc["metadata"]["name"].endswith("l4-default")]
    assert lane["spec"]["replicas"] == 0


@pytest.mark.parametrize("allow_unauthenticated", [False, True])
def test_worker_auth_capability_covers_the_remote_lane(tmp_path: Path, allow_unauthenticated: bool) -> None:
    values = remote_pool_values()
    values["workers"]["auth"] = {"enabled": True, "allowUnauthenticated": allow_unauthenticated}
    mode = "none" if allow_unauthenticated else "token"
    assert worker_auth_requirements(rendered_documents(tmp_path, values)) == {
        "sie-sie-cluster-worker-remote-remote/worker": [f"--require-auth-mode={mode}"],
    }


def test_worker_auth_hook_failure_recommends_the_required_capability(tmp_path: Path) -> None:
    docs = rendered_documents(tmp_path, {"workers": {"auth": {"enabled": True}, **L4_POOL["workers"]}})
    (job,) = [doc for doc in docs if doc["kind"] == "Job" and doc["metadata"]["name"] == WORKER_AUTH_TEST_JOB]
    script = job["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "Use worker images that support serve --require-auth-mode=token." in script
    assert "same release as the chart" not in script


@pytest.mark.parametrize("tag", [None, "v0.10.1"], ids=["default-image", "explicit-v0.10.1"])
@pytest.mark.parametrize("auth", [{}, {"enabled": False}], ids=["default-auth", "explicit-auth-off"])
def test_worker_auth_off_preserves_existing_image_compatibility(tmp_path: Path, tag: str | None, auth: dict) -> None:
    common = {"image": {"tag": tag}} if tag is not None else {}
    docs = rendered_documents(tmp_path, {"workers": {"auth": auth, "common": common, **L4_POOL["workers"]}})
    env = container_env(docs, "sie-sie-cluster-worker-l4-default", "worker")
    assert env["SIE_AUTH_MODE"] == {"name": "SIE_AUTH_MODE", "value": "none"}
    assert "SIE_AUTH_TOKEN" not in env
    assert WORKER_AUTH_TEST_JOB not in hook_jobs(docs)
    assert worker_auth_requirements(docs) == {}


def adapter_containers(statefulset: dict) -> list[dict]:
    return [
        container
        for container in statefulset["spec"]["template"]["spec"]["containers"]
        if container["name"] != "worker-sidecar"
    ]


def test_worker_liveness_catches_hangs_and_startup_covers_startup(tmp_path: Path) -> None:
    """A wedged CUDA context exits the worker; liveness only needs to catch hangs."""
    (statefulset,) = worker_statefulsets(tmp_path, L4_POOL)
    (worker,) = adapter_containers(statefulset)

    assert worker["terminationMessagePolicy"] == "FallbackToLogsOnError"
    assert worker["startupProbe"]["httpGet"]["path"] == "/healthz"
    liveness = worker["livenessProbe"]
    assert liveness["httpGet"]["path"] == "/livez"
    assert liveness["periodSeconds"] * liveness["failureThreshold"] == 60
    for container in statefulset["spec"]["template"]["spec"]["containers"]:
        names = {env["name"] for env in container.get("env", [])}
        assert "SIE_WORKER_LIVENESS_BUDGET_S" not in names


def test_model_ready_timeout_is_independent_of_liveness(tmp_path: Path) -> None:
    """Model loads do not touch the liveness probe, so a long load budget renders."""
    values = {"workers": {"common": {"modelReadyTimeoutSec": 1800}, "pools": {"l4": {"enabled": True}}}}
    (statefulset,) = worker_statefulsets(tmp_path, values)
    timeouts = {
        env["value"]
        for container in statefulset["spec"]["template"]["spec"]["containers"]
        for env in container.get("env", [])
        if env["name"] == "SIE_MODEL_READY_TIMEOUT_S"
    }
    assert timeouts == {"1800"}
