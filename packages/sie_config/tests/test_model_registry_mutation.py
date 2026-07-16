from pathlib import Path

import pytest
import yaml
from sie_config.model_registry import ModelRegistry


def _setup_registry(
    root: Path,
    bundles: dict[str, list[str]] | None = None,
    models: dict[str, str] | None = None,
) -> tuple[ModelRegistry, Path]:
    bundles_dir = root / "bundles"
    models_dir = root / "models"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if bundles is None:
        bundles = {"default": ["sie_server.adapters.bert_flash", "sie_server.adapters.sentence_transformer"]}

    for name, adapters in bundles.items():
        data = {"name": name, "priority": 10, "adapters": adapters}
        if any(adapter.startswith("sie_server_rust.adapters.candle") for adapter in adapters):
            data["engine"] = "candle"
        (bundles_dir / f"{name}.yaml").write_text(yaml.dump(data))

    if models:
        for sie_id, adapter_path in models.items():
            config = {
                "sie_id": sie_id,
                "profiles": {"default": {"adapter_path": adapter_path, "max_batch_tokens": 8192}},
            }
            filename = sie_id.replace("/", "__") + ".yaml"
            (models_dir / filename).write_text(yaml.dump(config))

    registry = ModelRegistry(bundles_dir, models_dir)
    return registry, root


class TestAddModelConfig:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._root = tmp_path

    def test_add_new_model(self) -> None:
        registry, _ = _setup_registry(self._root / "add_new")
        config = {
            "sie_id": "new/model",
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                    "max_batch_tokens": 8192,
                },
            },
        }
        created, skipped, bundles = registry.add_model_config(config)
        assert created == ["default"]
        assert skipped == []
        assert "default" in bundles

    def test_add_model_config_normalizes_pool(self) -> None:
        registry, _ = _setup_registry(self._root / "pool")
        config = {
            "sie_id": "new/model",
            "pool": " Customer-A ",
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                    "max_batch_tokens": 8192,
                },
            },
        }
        registry.add_model_config(config)
        assert config["pool"] == "customer-a"

    def test_add_model_config_rejects_invalid_pool(self) -> None:
        registry, _ = _setup_registry(self._root / "bad_pool")
        config = {
            "sie_id": "bad/model",
            "pool": "customer.a",
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                    "max_batch_tokens": 8192,
                },
            },
        }
        with pytest.raises(ValueError, match="pool"):
            registry.add_model_config(config)

    def test_add_model_config_rejects_pool_move_on_profile_append(self) -> None:
        registry, _ = _setup_registry(
            self._root / "pool_move",
            models={"existing/model": "sie_server.adapters.bert_flash:B"},
        )
        config = {
            "sie_id": "existing/model",
            "pool": "customer-a",
            "profiles": {
                "custom": {
                    "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                    "max_batch_tokens": 8192,
                },
            },
        }
        with pytest.raises(ValueError, match="Pool on model"):
            registry.add_model_config(config)

    def test_add_model_config_omitted_pool_append_preserves_existing_pool(self) -> None:
        registry, _ = _setup_registry(self._root / "pool_append_omitted")
        registry.add_model_config(
            {
                "sie_id": "tenant/model",
                "pool": "customer-a",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                        "max_batch_tokens": 1024,
                    },
                },
            }
        )
        registry.add_model_config(
            {
                "sie_id": "tenant/model",
                "profiles": {
                    "fast": {
                        "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                        "max_batch_tokens": 8192,
                    },
                },
            }
        )

        full = registry.get_full_config("tenant/model")
        assert full is not None
        assert full["pool"] == "customer-a"
        assert set(full["profiles"]) == {"default", "fast"}

    def test_add_model_config_rejects_explicit_default_pool_move_on_profile_append(self) -> None:
        registry, _ = _setup_registry(self._root / "pool_append_default")
        registry.add_model_config(
            {
                "sie_id": "tenant/model",
                "pool": "customer-a",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                        "max_batch_tokens": 1024,
                    },
                },
            }
        )
        config = {
            "sie_id": "tenant/model",
            "pool": "default",
            "profiles": {
                "fast": {
                    "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                    "max_batch_tokens": 8192,
                },
            },
        }

        with pytest.raises(ValueError, match="Pool on model"):
            registry.add_model_config(config)

    def test_model_becomes_routable(self) -> None:
        registry, _ = _setup_registry(self._root / "routable")
        registry.add_model_config(
            {
                "sie_id": "new/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )
        assert registry.model_exists("new/model")
        bundle = registry.resolve_bundle("new/model")
        assert bundle == "default"

    def test_add_profile_to_existing_model(self) -> None:
        registry, _ = _setup_registry(
            self._root / "add_prof",
            models={"existing/model": "sie_server.adapters.bert_flash:B"},
        )
        config = {
            "sie_id": "existing/model",
            "profiles": {
                "default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 8192},
                "custom": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 2},
            },
        }
        created, skipped, _ = registry.add_model_config(config)
        assert "custom" in created
        assert "default" in skipped

    def test_add_conflicting_profile_raises(self) -> None:
        registry, _ = _setup_registry(
            self._root / "conflict",
            models={"existing/model": "sie_server.adapters.bert_flash:B"},
        )
        config = {
            "sie_id": "existing/model",
            "profiles": {
                "default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1},
            },
        }
        with pytest.raises(ValueError, match="already exist with different content"):
            registry.add_model_config(config)

    def test_add_conflicting_non_hash_profile_field_raises(self) -> None:
        registry, _ = _setup_registry(
            self._root / "conflict_non_hash",
            models={"existing/model": "sie_server.adapters.bert_flash:B"},
        )
        config = {
            "sie_id": "existing/model",
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.bert_flash:B",
                    "max_batch_tokens": 8192,
                    "max_sequence_length": 128,
                },
            },
        }
        with pytest.raises(ValueError, match="already exist with different content"):
            registry.add_model_config(config)

    def test_unroutable_adapter_raises(self) -> None:
        registry, _ = _setup_registry(self._root / "unroutable")
        config = {
            "sie_id": "bad/model",
            "profiles": {"default": {"adapter_path": "sie_server.adapters.unknown:X", "max_batch_tokens": 1}},
        }
        with pytest.raises(ValueError, match="not in any known bundle"):
            registry.add_model_config(config)

    def test_missing_sie_id_raises(self) -> None:
        registry, _ = _setup_registry(self._root / "no_id")
        with pytest.raises(ValueError, match="sie_id"):
            registry.add_model_config({"profiles": {"default": {}}})

    def test_missing_profiles_raises(self) -> None:
        registry, _ = _setup_registry(self._root / "no_prof")
        with pytest.raises(ValueError, match="profiles"):
            registry.add_model_config({"sie_id": "m"})

    def test_multi_bundle_routing(self) -> None:
        registry, _ = _setup_registry(
            self._root / "multi_bundle",
            bundles={
                "default": ["sie_server.adapters.bert_flash"],
                "sglang": ["sie_server.adapters.sglang"],
            },
        )
        registry.add_model_config(
            {
                "sie_id": "sglang/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.sglang:S", "max_batch_tokens": 1}},
            }
        )
        bundle = registry.resolve_bundle("sglang/model")
        assert bundle == "sglang"

    def test_append_profile_creates_profile_variant_route(self) -> None:
        registry, _ = _setup_registry(
            self._root / "profile_variant",
            bundles={
                "default": ["sie_server.adapters.bert_flash"],
                "candle": ["sie_server_rust.adapters.candle"],
            },
            models={"existing/model": "sie_server.adapters.bert_flash:B"},
        )

        created, skipped, bundles = registry.add_model_config(
            {
                "sie_id": "existing/model",
                "profiles": {
                    "candle": {
                        "adapter_path": "sie_server_rust.adapters.candle:CandleEmbeddingAdapter",
                        "max_batch_tokens": 8192,
                    },
                },
            }
        )

        assert created == ["candle"]
        assert skipped == []
        assert set(bundles) == {"default", "candle"}
        assert registry.resolve_bundle("existing/model") == "default"
        assert registry.resolve_bundle("existing/model:candle") == "candle"
        variant = registry.get_model_info("existing/model:candle")
        assert variant is not None
        assert variant.bundles == ["candle"]


class TestGetFullConfig:
    """Covers the in-memory full-config snapshot used by `/v1/configs/export`
    in no-ConfigStore deployments. Regression guard for the case where a
    second append-profile write used to overwrite the first.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._root = tmp_path

    def test_get_full_config_returns_none_for_unknown_model(self) -> None:
        registry, _ = _setup_registry(self._root / "unknown")
        assert registry.get_full_config("nope/model") is None

    def test_get_full_config_includes_filesystem_seeded_model(self) -> None:
        registry, _ = _setup_registry(
            self._root / "fs_seed",
            models={"seed/model": "sie_server.adapters.bert_flash:B"},
        )
        full = registry.get_full_config("seed/model")
        assert full is not None
        assert full["sie_id"] == "seed/model"
        assert "default" in full["profiles"]
        assert full["profiles"]["default"]["adapter_path"] == "sie_server.adapters.bert_flash:B"

    def test_get_full_config_survives_profile_append(self) -> None:
        registry, _ = _setup_registry(self._root / "append")
        registry.add_model_config(
            {
                "sie_id": "mem/model",
                "description": "embedding model",
                "profiles": {
                    "default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1024},
                },
            }
        )
        registry.add_model_config(
            {
                "sie_id": "mem/model",
                "profiles": {
                    "fast": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 8192},
                },
            }
        )
        full = registry.get_full_config("mem/model")
        assert full is not None
        assert full["sie_id"] == "mem/model"
        assert full["description"] == "embedding model", (
            "append-only merge must preserve top-level fields from the first write"
        )
        assert set(full["profiles"].keys()) == {"default", "fast"}, (
            "append-only merge must union profiles across writes"
        )
        assert full["profiles"]["default"]["max_batch_tokens"] == 1024
        assert full["profiles"]["fast"]["max_batch_tokens"] == 8192

    def test_get_full_config_preserves_non_canonical_profile_fields_across_appends(self) -> None:
        """Regression: a later append-profile write must not overwrite the
        first profile with its canonical (hash-only) form, which would
        drop non-hash keys like `extends`, custom metadata, etc.
        """
        registry, _ = _setup_registry(self._root / "preserve")
        registry.add_model_config(
            {
                "sie_id": "mem/model",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.bert_flash:B",
                        "max_batch_tokens": 1024,
                        "model_name": "BAAI/bge-base-en",
                        "revision": "main",
                    },
                },
            }
        )
        registry.add_model_config(
            {
                "sie_id": "mem/model",
                "profiles": {
                    "fast": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 8192},
                },
            }
        )
        full = registry.get_full_config("mem/model")
        assert full is not None
        assert full["profiles"]["default"].get("model_name") == "BAAI/bge-base-en", (
            "non-hash profile field must survive a subsequent append on the same model"
        )
        assert full["profiles"]["default"].get("revision") == "main"

    def test_get_full_config_preserves_extends_only_profile_across_appends(self) -> None:
        """`extends`-only profiles must survive later appends. Before the
        fix, the second write would overwrite the first profile with the
        canonical 4-field dict (all None because `extends` is not a hash
        field), turning a valid profile into an unroutable stub.
        """
        registry, _ = _setup_registry(self._root / "extends_only")
        registry.add_model_config(
            {
                "sie_id": "ext/model",
                "profiles": {
                    "base": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1024},
                    "derived": {"extends": "base"},
                },
            }
        )
        registry.add_model_config(
            {
                "sie_id": "ext/model",
                "profiles": {
                    "fast": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 4096},
                },
            }
        )
        full = registry.get_full_config("ext/model")
        assert full is not None
        assert full["profiles"]["derived"].get("extends") == "base", (
            "`extends` field must be preserved in the full config after later appends"
        )

    def test_get_full_config_returns_deep_copy(self) -> None:
        """Caller mutations must not leak back into the registry state."""
        registry, _ = _setup_registry(self._root / "copy")
        registry.add_model_config(
            {
                "sie_id": "mem/model",
                "profiles": {
                    "default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1024},
                },
            }
        )
        snapshot = registry.get_full_config("mem/model")
        assert snapshot is not None
        snapshot["profiles"]["default"]["max_batch_tokens"] = 0
        snapshot["profiles"]["injected"] = {"adapter_path": "x:Y", "max_batch_tokens": 1}

        again = registry.get_full_config("mem/model")
        assert again is not None
        assert again["profiles"]["default"]["max_batch_tokens"] == 1024
        assert "injected" not in again["profiles"]


class TestComputeBundleConfigHash:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._root = tmp_path

    def test_empty_bundle_returns_empty(self) -> None:
        registry, _ = _setup_registry(self._root / "empty")
        assert registry.compute_bundle_config_hash("default") == ""

    def test_hash_changes_when_model_added(self) -> None:
        registry, _ = _setup_registry(self._root / "hash_change")
        hash1 = registry.compute_bundle_config_hash("default")
        registry.add_model_config(
            {
                "sie_id": "new/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )
        hash2 = registry.compute_bundle_config_hash("default")
        assert hash1 != hash2
        assert len(hash2) == 64

    def test_hash_is_deterministic(self) -> None:
        registry, _ = _setup_registry(self._root / "deterministic")
        registry.add_model_config(
            {
                "sie_id": "m1",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )
        h1 = registry.compute_bundle_config_hash("default")
        h2 = registry.compute_bundle_config_hash("default")
        assert h1 == h2

    def test_hash_preserves_falsy_adapter_options(self) -> None:
        with_false, _ = _setup_registry(self._root / "falsy_false")
        with_false.add_model_config(
            {
                "sie_id": "m1",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.bert_flash:B",
                        "max_batch_tokens": 1,
                        "adapter_options": {"runtime": {"normalize": False, "truncate_to": 0}},
                    }
                },
            }
        )
        without_options, _ = _setup_registry(self._root / "falsy_empty")
        without_options.add_model_config(
            {
                "sie_id": "m1",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.bert_flash:B",
                        "max_batch_tokens": 1,
                        "adapter_options": {"runtime": {}},
                    }
                },
            }
        )

        assert with_false.compute_bundle_config_hash("default") != without_options.compute_bundle_config_hash("default")

    def test_hash_scoped_to_bundle(self) -> None:
        registry, _ = _setup_registry(
            self._root / "scoped",
            bundles={
                "default": ["sie_server.adapters.bert_flash"],
                "sglang": ["sie_server.adapters.sglang"],
            },
        )
        registry.add_model_config(
            {
                "sie_id": "m1",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )
        assert registry.compute_bundle_config_hash("sglang") == ""
        assert registry.compute_bundle_config_hash("default") != ""

    def test_unknown_bundle_returns_empty(self) -> None:
        registry, _ = _setup_registry(self._root / "unknown")
        assert registry.compute_bundle_config_hash("nonexistent") == ""


class TestConcurrentAddModelConfig:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._root = tmp_path

    def test_concurrent_add_10_models(self) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        registry, _ = _setup_registry(self._root / "concurrent")
        exceptions: list[Exception] = []

        def add_model(i: int) -> None:
            config = {
                "sie_id": f"concurrent/model-{i}",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                        "max_batch_tokens": 8192,
                    },
                },
            }
            try:
                created, _, bundles = registry.add_model_config(config)
                assert created == ["default"]
                assert "default" in bundles
            except Exception as e:  # noqa: BLE001
                exceptions.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_model, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()
        assert exceptions == [], f"Exceptions during concurrent add: {exceptions}"
        for i in range(10):
            model_id = f"concurrent/model-{i}"
            assert registry.model_exists(model_id), f"{model_id} should exist"
            bundle = registry.resolve_bundle(model_id)
            assert bundle == "default", f"{model_id} should resolve to 'default' bundle"

    def test_concurrent_readers_see_consistent_snapshots(self) -> None:
        # Fix #12 regression: readers like `resolve_bundle`,
        # `get_model_info`, and `list_models` are lock-free; they read
        # `self._models` / `self._model_names_lower` / ... as single
        # pointer reads. If the writer in-place mutated those dicts
        # (`.update()`, `.pop()`, etc.), a reader could observe a model
        # present in `_models` but missing from `_model_names_lower`,
        # or vice versa. Our `add_model_config` rebuilds each dict and
        # reassigns the attribute, so readers always see either the
        # pre-state or the post-state — never a torn/half-migrated one.
        # This test hammers the registry with parallel readers while
        # writers are mutating and asserts every `get_model_info` hit
        # whose id came from `list_models` finds a real entry (no
        # TOCTOU where `list_models` saw the id but `get_model_info`
        # saw the pre-state).
        import threading
        from concurrent.futures import ThreadPoolExecutor

        registry, _ = _setup_registry(self._root / "reader_consistency")
        stop = threading.Event()
        inconsistencies: list[str] = []

        def writer(i: int) -> None:
            config = {
                "sie_id": f"r/model-{i}",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                        "max_batch_tokens": 8192,
                    },
                },
            }
            registry.add_model_config(config)

        def reader() -> None:
            while not stop.is_set():
                names = registry.list_models()
                for name in names:
                    info = registry.get_model_info(name)
                    if info is None:
                        inconsistencies.append(
                            f"list_models() returned {name!r} but get_model_info returned None — torn registry state!"
                        )
                    # case-insensitive path should also resolve
                    info2 = registry.get_model_info(name.lower())
                    if info2 is None and name != name.lower():
                        inconsistencies.append(
                            f"case-insensitive lookup for {name!r} failed while "
                            f"exact lookup succeeded — `_model_names_lower` is "
                            "out of sync with `_models`!"
                        )

        with ThreadPoolExecutor(max_workers=8) as executor:
            reader_futs = [executor.submit(reader) for _ in range(4)]
            writer_futs = [executor.submit(writer, i) for i in range(50)]
            for f in writer_futs:
                f.result()
            stop.set()
            for f in reader_futs:
                f.result()
        assert inconsistencies == [], f"Reader observed torn state: {inconsistencies[:5]}"


class TestReplaceModelConfig:
    """replace_model_config: the wholesale (non-append) convergence path (#1771)."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._root = tmp_path

    def _cfg(self, sie_id: str, tokens: int = 8192, profiles: tuple[str, ...] = ("default",)) -> dict:
        return {
            "sie_id": sie_id,
            "profiles": {
                p: {"adapter_path": "sie_server.adapters.bert_flash:BertAdapter", "max_batch_tokens": tokens}
                for p in profiles
            },
        }

    def test_replace_creates_new_model(self) -> None:
        registry, _ = _setup_registry(self._root / "r_new")
        bundles = registry.replace_model_config(self._cfg("new/model"))
        assert "default" in bundles
        assert registry.model_exists("new/model")

    def test_replace_converges_changed_profile(self) -> None:
        registry, _ = _setup_registry(self._root / "r_change")
        registry.add_model_config(self._cfg("m/model", tokens=1024))
        assert registry.get_full_config("m/model")["profiles"]["default"]["max_batch_tokens"] == 1024
        registry.replace_model_config(self._cfg("m/model", tokens=4096))
        assert registry.get_full_config("m/model")["profiles"]["default"]["max_batch_tokens"] == 4096

    def test_replace_drops_stale_profiles(self) -> None:
        registry, _ = _setup_registry(self._root / "r_drop")
        registry.replace_model_config(self._cfg("m/model", profiles=("default", "fast")))
        assert registry.get_model_profile_names("m/model") == {"default", "fast"}
        registry.replace_model_config(self._cfg("m/model", profiles=("default",)))
        # The dropped profile must not survive a replace (unlike append-only add).
        assert registry.get_model_profile_names("m/model") == {"default"}
        # And its profile-variant route identity must be gone too.
        assert not registry.model_exists("m/model:fast")

    def test_replace_rejects_unroutable(self) -> None:
        registry, _ = _setup_registry(self._root / "r_bad")
        cfg = {
            "sie_id": "bad/model",
            "profiles": {"default": {"adapter_path": "sie_server.adapters.nope:X", "max_batch_tokens": 8192}},
        }
        with pytest.raises(ValueError, match="not in any known bundle"):
            registry.replace_model_config(cfg)

    def test_replace_rejects_missing_profiles(self) -> None:
        registry, _ = _setup_registry(self._root / "r_empty")
        with pytest.raises(ValueError, match="profiles"):
            registry.replace_model_config({"sie_id": "x/y"})

    def test_replace_updates_bundle_config_hash(self) -> None:
        registry, _ = _setup_registry(self._root / "r_hash")
        registry.add_model_config(self._cfg("m/model", tokens=1024))
        before = registry.compute_bundle_config_hash("default")
        registry.replace_model_config(self._cfg("m/model", tokens=4096))
        after = registry.compute_bundle_config_hash("default")
        # A content change must move the worker-parity hash (the drift the sync heals).
        assert before != after

    def test_replace_rejects_dangling_extends(self) -> None:
        # A replace discards prior profiles, so `extends` resolves against ONLY
        # the incoming set: a profile that extends a parent absent from this body
        # cannot resolve an adapter_path and must be rejected (an `extends` value
        # alone no longer exempts it — that would persist an unrouteable variant).
        registry, _ = _setup_registry(self._root / "r_dangling")
        cfg = {
            "sie_id": "d/model",
            "profiles": {
                "default": {"adapter_path": "sie_server.adapters.bert_flash:BertAdapter", "max_batch_tokens": 8192},
                # extends a parent that is NOT in this body -> unresolvable.
                "orphan": {"extends": "not-in-this-body"},
            },
        }
        with pytest.raises(ValueError, match="missing adapter_path"):
            registry.replace_model_config(cfg)

    def test_replace_notifies_the_bundle_that_loses_the_model(self) -> None:
        # A replace that MOVES a model from bundle A to bundle B must notify BOTH:
        # B (the new home) AND A (which loses the model), or A keeps a stale
        # bundle_config_hash that still counts the departed model until a later
        # full refresh (#1771 lane↔catalog hash-parity).
        registry, _ = _setup_registry(
            self._root / "r_move",
            bundles={
                "a": ["sie_server.adapters.bert_flash"],
                "b": ["sie_server.adapters.sentence_transformer"],
            },
        )
        registry.replace_model_config(
            {
                "sie_id": "mv/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:BertAdapter"}},
            }
        )
        moved = registry.replace_model_config(
            {
                "sie_id": "mv/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.sentence_transformer:STAdapter"}},
            }
        )
        # Both the old bundle (a) and the new bundle (b) are notified.
        assert set(moved) >= {"a", "b"}
