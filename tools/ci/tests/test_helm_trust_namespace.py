"""Keep trust-manager's TLS producer and consumer in the same namespace."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from tools.ci.tests.test_helm_render import ROOT, render_chart, rendered_documents
from tools.mise_tasks import helm


def trust_values(
    *, installed: bool = True, helm_cert: bool = True, namespace: str = "", workload_namespace: str = ""
) -> dict:
    return {
        "certManagerBundle": {"trustManager": {"install": installed}},
        "trust-manager": {"namespace": namespace, "app": {"webhook": {"tls": {"helmCert": {"enabled": helm_cert}}}}},
        "global": {"namespace": workload_namespace},
    }


@pytest.mark.parametrize("workload_namespace", ["", "monitoring", "workers"])
def test_helm_generated_trust_tls_rejects_a_foreign_child_namespace(tmp_path: Path, workload_namespace: str) -> None:
    result = render_chart(tmp_path, trust_values(namespace="monitoring", workload_namespace=workload_namespace))
    assert result.returncode != 0
    assert "trust-manager.namespace" in result.stderr
    assert "app.webhook.tls.helmCert.enabled=true" in result.stderr
    assert '"monitoring"' in result.stderr and '"sie"' in result.stderr


@pytest.mark.parametrize(
    ("namespace", "helm_cert", "workload_namespace"),
    [("", True, ""), ("sie", True, ""), ("", True, "workers"), ("monitoring", False, "")],
)
def test_supported_trust_tls_modes_keep_producer_and_consumer_together(
    tmp_path: Path, namespace: str, helm_cert: bool, workload_namespace: str
) -> None:
    docs = rendered_documents(
        tmp_path, trust_values(namespace=namespace, helm_cert=helm_cert, workload_namespace=workload_namespace)
    )
    (deployment,) = [doc for doc in docs if doc["kind"] == "Deployment" and doc["metadata"]["name"] == "trust-manager"]
    volume = next(
        volume
        for volume in deployment["spec"]["template"]["spec"]["volumes"]
        if volume.get("secret", {}).get("secretName") == "trust-manager-tls"
    )
    producer_kind = "Secret" if helm_cert else "Certificate"
    (producer,) = [
        doc
        for doc in docs
        if doc["kind"] == producer_kind
        and doc["metadata"]["name"] == ("trust-manager-tls" if helm_cert else "trust-manager")
    ]
    assert deployment["metadata"]["namespace"] == producer["metadata"]["namespace"] == (namespace or "sie")
    assert volume["secret"]["secretName"] == (
        producer["metadata"]["name"] if helm_cert else producer["spec"]["secretName"]
    )


def test_disabled_trust_manager_does_not_validate_stale_namespace_settings(tmp_path: Path) -> None:
    result = render_chart(tmp_path, trust_values(installed=False, namespace="monitoring"))
    assert result.returncode == 0, result.stderr


def test_bundle_source_namespace_does_not_relocate_webhook_tls(tmp_path: Path) -> None:
    values = trust_values()
    values["trust-manager"]["app"]["trust"] = {"namespace": "monitoring"}
    result = render_chart(tmp_path, values)
    assert result.returncode == 0, result.stderr


def test_namespace_suite_runs_in_the_prepared_helm_lane() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "--ignore tools/ci/tests/test_helm_trust_namespace.py" in workflow
    assert (
        "run: mise exec -- uv run --frozen --project . pytest -q tools/ci/tests/test_helm_trust_namespace.py"
        in workflow
    )


@pytest.mark.parametrize("child_namespace", ["system-edge", "sie"])
def test_trust_tls_uses_the_actual_helm_release_namespace(tmp_path: Path, child_namespace: str) -> None:
    values_file = tmp_path / "custom-release-values.yaml"
    values_file.write_text(yaml.safe_dump(trust_values(namespace=child_namespace, workload_namespace="workers")))
    result = subprocess.run(
        [
            "mise",
            "exec",
            "--",
            "helm",
            "template",
            "custom-release",
            str(helm.CHART_DIR),
            "--namespace",
            "system-edge",
            "-f",
            str(values_file),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if child_namespace != "system-edge":
        assert result.returncode != 0
        assert 'trust-manager.namespace="sie"' in result.stderr
        assert 'release namespace "system-edge"' in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        docs = [doc for doc in yaml.safe_load_all(result.stdout) if doc]
        tls_pair = [
            doc
            for doc in docs
            if (doc["kind"], doc["metadata"]["name"])
            in {("Secret", "trust-manager-tls"), ("Deployment", "trust-manager")}
        ]
        assert len(tls_pair) == 2
        assert all(doc["metadata"]["namespace"] == "system-edge" for doc in tls_pair)
