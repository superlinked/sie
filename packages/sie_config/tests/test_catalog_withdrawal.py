"""Catalog withdrawal for erased custom models (#1841 DPA erase).

Erasing a customer's custom model used to delete its weights Volume and stamp
``revoked_at`` — so the model 404'd — while the catalog kept ADVERTISING it. A model
the customer asked to erase stayed listed forever, which reads as a right-to-erasure
failure even though no weights remained.

Neither additive write can express withdrawal: ``POST`` is append-only and 409s on a
changed profile, and ``PUT`` cannot represent "gone" because the structure validator
requires non-empty routable profiles. ``DELETE /v1/configs/models/{id}`` is the write
that can, and #1841 consumes it rather than adding a parallel one — the erase driver
calls it through ``ConfigAuthorityClient.tombstone_model``.

The property under test throughout: **a withdrawn model must not come back** — not
from a reload, not from a restore, not through a profile variant.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_config.config_api import router as config_router
from sie_config.config_store import ConfigStore
from sie_config.model_registry import ModelRegistry

_ADAPTER = "sie_server.adapters.pytorch_embedding"
_OTHER_ADAPTER = "sie_server.adapters.bge_m3_flash"


def _write_bundle(bundles_dir: Path, name: str, adapter: str = _ADAPTER) -> None:
    (bundles_dir / f"{name}.yaml").write_text(yaml.dump({"name": name, "priority": 10, "adapters": [adapter]}))


def _model_config(sie_id: str, *, extra_profile: str | None = None, extra_adapter: str = _ADAPTER) -> dict:
    profiles: dict = {"default": {"adapter_path": f"{_ADAPTER}:PyTorchEmbeddingAdapter"}}
    if extra_profile:
        profiles[extra_profile] = {"adapter_path": f"{extra_adapter}:Adapter"}
    return {"sie_id": sie_id, "profiles": profiles}


class _Harness:
    def __init__(self, tmp: Path) -> None:
        self.bundles = tmp / "bundles"
        self.models = tmp / "models"
        self.store_dir = tmp / "store"
        for d in (self.bundles, self.models, self.store_dir):
            d.mkdir(parents=True, exist_ok=True)
        _write_bundle(self.bundles, "default")
        self.registry = ModelRegistry(self.bundles, self.models)
        self.store = ConfigStore(str(self.store_dir))
        app = FastAPI()
        app.include_router(config_router)
        app.state.model_registry = self.registry
        app.state.nats_publisher = None
        app.state.config_store = self.store
        self.client = TestClient(app)

    def add(self, sie_id: str, **kw) -> None:
        resp = self.client.post("/v1/configs/models", content=yaml.dump(_model_config(sie_id, **kw)).encode())
        assert resp.status_code in (200, 201), resp.text

    def withdraw(self, sie_id: str):
        return self.client.delete(f"/v1/configs/models/{sie_id}")

    def listed(self) -> list[str]:
        return [m["model_id"] for m in self.client.get("/v1/configs/models").json()["models"]]


def _harness(tmp_path: Path) -> _Harness:
    return _Harness(tmp_path)


def test_a_withdrawn_model_disappears_from_the_catalog(tmp_path):
    h = _harness(tmp_path)
    h.add("acme/enc")
    assert "acme/enc" in h.listed()

    resp = h.withdraw("acme/enc")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert "acme/enc" not in h.listed()
    assert h.registry.model_exists("acme/enc") is False


def test_the_export_the_gateway_reads_no_longer_carries_it(tmp_path):
    """The export IS the propagation mechanism.

    The gateway rebuilds its whole model map from /v1/configs/export
    (replace_model_configs_authoritative — a full-snapshot replace, not an overlay),
    so absence from this response is what actually stops it serving. If the model
    survived here, the withdrawal would be cosmetic.
    """
    h = _harness(tmp_path)
    h.add("acme/enc")
    h.withdraw("acme/enc")
    exported = h.client.get("/v1/configs/export").json()
    ids = {m.get("sie_id") for m in exported.get("models", [])}
    assert "acme/enc" not in ids


def test_the_epoch_bumps_so_the_gateway_notices(tmp_path):
    """Nothing is published on NATS — the delta wire has no removal verb, so the
    epoch bump is the ONLY signal that reaches the fleet.
    """
    h = _harness(tmp_path)
    h.add("acme/enc")
    before = h.client.get("/v1/configs/epoch").json()["epoch"]
    h.withdraw("acme/enc")
    assert h.client.get("/v1/configs/epoch").json()["epoch"] > before


def test_withdrawing_an_absent_model_is_a_quiet_no_op(tmp_path):
    """A retrying erase driver must not churn every gateway in the fleet.

    Each epoch bump makes every gateway re-export the entire catalog, so a retry loop
    that bumped on a no-op would turn one stuck erase into fleet-wide load.
    """
    h = _harness(tmp_path)
    h.add("acme/enc")
    h.withdraw("acme/enc")
    epoch_after_first = h.client.get("/v1/configs/epoch").json()["epoch"]

    resp = h.withdraw("acme/enc")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is False
    assert resp.json()["unchanged"] is True
    assert h.client.get("/v1/configs/epoch").json()["epoch"] == epoch_after_first


def test_the_withdrawal_survives_a_restart(tmp_path):
    """The resurrection case: the store is what a restart restores from.

    A custom model is API-persisted and has no baked filesystem baseline, so the
    withdrawal must leave NOTHING behind for the restore loop to find. If the stored
    document merely became inert, any loader that keyed off ``sie_id`` alone would
    re-register an erased model on the next restart.
    """
    h = _harness(tmp_path)
    h.add("acme/enc")
    h.withdraw("acme/enc")

    stored = h.store.load_all_models()
    assert "acme/enc" not in stored

    # Replay the restore loop exactly as app_factory does at startup.
    reloaded = ModelRegistry(h.bundles, h.models)
    for cfg in stored.values():
        reloaded.add_model_config(cfg)
    assert reloaded.model_exists("acme/enc") is False


def test_profile_variants_of_a_withdrawn_model_go_with_it(tmp_path):
    """`model:profile` is a separate registry entry. Leaving it behind keeps a route
    to erased weights resolvable under a different name.

    The variant here deliberately uses a DIFFERENT adapter, so it routes to a second
    bundle the base model does not belong to. With a shared bundle the affected-bundles
    assertion below is vacuous — the base contributes that bundle anyway — and a
    version that forgot the variant's bundles entirely still passed.
    """
    h = _harness(tmp_path)
    _write_bundle(h.bundles, "other", adapter=_OTHER_ADAPTER)
    h.registry.reload()
    h.add("acme/enc", extra_profile="fast", extra_adapter=_OTHER_ADAPTER)
    assert h.registry.model_exists("acme/enc:fast")

    resp = h.withdraw("acme/enc")
    assert h.registry.model_exists("acme/enc:fast") is False
    # The VARIANT's bundle must be in affected_bundles too. Its bundle_config_hash
    # changes when the variant leaves, and a worker on a bundle whose hash it was
    # never told about NAKs traffic on the mismatch (#1771).
    assert "other" in resp.json()["affected_bundles"]


def test_a_withdrawn_name_can_be_registered_again(tmp_path):
    """The name frees on erase, so re-registration must work — and must produce a LIVE
    model. The append-only POST merges onto whatever the store holds for the id, so a
    withdrawal that left a residue behind would merge INTO that residue.
    """
    h = _harness(tmp_path)
    h.add("acme/enc")
    h.withdraw("acme/enc")
    h.add("acme/enc")
    assert h.registry.model_exists("acme/enc")
    assert h.store.load_all_models()["acme/enc"]["profiles"]


def test_withdrawal_requires_admin_auth(monkeypatch, tmp_path):
    """It is a destructive catalog write; it must not be reachable with a read token."""
    monkeypatch.setenv("SIE_ADMIN_TOKEN", "admin-secret")
    h = _harness(tmp_path)
    assert h.client.delete("/v1/configs/models/acme/enc").status_code in (401, 403)


def test_only_the_named_model_is_withdrawn(tmp_path):
    """A withdrawal that took its bundle-mates with it would be catastrophic and silent."""
    h = _harness(tmp_path)
    h.add("acme/enc")
    h.add("acme/other")
    h.withdraw("acme/enc")
    assert "acme/other" in h.listed()
    assert h.registry.model_exists("acme/other")
