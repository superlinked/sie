"""The upstreams file the Helm chart renders is one the server accepts.

tools/ci/tests/test_helm_render.py checks that the chart renders exactly the
fixture's `rendered` file from its `values`. This half loads that file with the
server's own parser, so the chart and the server cannot drift apart unnoticed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sie_server.config.upstreams import Upstream, UpstreamKind, load_upstreams

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tools/ci/fixtures/helm-upstreams.yaml"


def fixture() -> dict[str, Any]:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def test_the_server_loads_the_upstreams_file_the_chart_renders(tmp_path: Path) -> None:
    contract = fixture()
    path = tmp_path / "upstreams.yaml"
    path.write_text(yaml.safe_dump(contract["rendered"]), encoding="utf-8")

    upstreams = load_upstreams(path)

    assert set(upstreams) == set(contract["values"])
    for name, upstream in upstreams.items():
        values = contract["values"][name]
        assert upstream.kind is UpstreamKind(values["kind"])
        expected_credential_env = f"SIE_UPSTREAM_KEY_{name.replace('-', '_').upper()}"
        assert upstream.api_key_secret == (expected_credential_env if "api_key_secret" in values else None)


def test_the_fixture_uses_every_upstream_field_the_server_defines() -> None:
    rendered = fixture()["rendered"]["upstreams"]

    used = set().union(*(entry.keys() for entry in rendered.values()))

    assert used == set(Upstream.model_fields)
