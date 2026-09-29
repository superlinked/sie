from __future__ import annotations

import base64
import json
import os
import re
import shutil
import socket
import subprocess
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


def test_only_gateway_pods_hold_a_token_that_can_write_configmaps(tmp_path: Path) -> None:
    docs = nats_documents(tmp_path, {**NATS_L4_POOL, "mcpEdge": {"enabled": True}})
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
