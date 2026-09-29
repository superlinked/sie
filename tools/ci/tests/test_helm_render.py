from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
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


def render_chart(tmp_path: Path, values: dict) -> subprocess.CompletedProcess[str]:
    values_file = tmp_path / "chart-values.yaml"
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
            "-f",
            str(values_file),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def rendered_documents(tmp_path: Path, values: dict) -> list[dict]:
    result = render_chart(tmp_path, values)
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


@pytest.mark.parametrize("values", [{}, PREDATING_TOKEN_KEYS])
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
    values = {**L4_POOL, "gateway": {"auth": {"mode": "token", "adminTokenSecretName": "gateway-admin"}}}
    docs = rendered_documents(tmp_path, values)
    assert env_entries(docs, "SIE_ADMIN_TOKEN") == {
        CONFIG_SERVICE: GENERATED_ADMIN,
        GATEWAY: {"name": "gateway-admin", "key": "SIE_ADMIN_TOKEN"},
    }
    assert env_entries(docs, "SIE_CONFIG_SERVICE_TOKEN") == dict.fromkeys(CONFIG_CONSUMERS, GENERATED_READ)


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


@pytest.mark.parametrize("admin", [{}, {"adminTokenSecretName": "operator-admin"}])
def test_admin_token_without_a_read_token_fails_the_render(tmp_path: Path, admin: dict) -> None:
    result = render_chart(tmp_path, {**L4_POOL, "config": {"auth": {**admin, "generateReadToken": False}}})
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
        f'"keySetting" "config.auth.{setting}TokenSecretKey" "nameSetting" "config.auth.{setting}TokenSecretName") }}}}\n',
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
