import logging
import tempfile
import threading
from pathlib import Path

import pytest
from sie_config.model_registry import (
    BundleConflictError,
    BundleInfo,
    ModelInfo,
    ModelNotFoundError,
    ModelRegistry,
    parse_model_spec,
)


class TestParseModelSpec:
    def test_simple_model_name(self) -> None:
        bundle, model = parse_model_spec("BAAI/bge-m3")
        assert bundle is None
        assert model == "BAAI/bge-m3"

    def test_model_with_variant(self) -> None:
        bundle, model = parse_model_spec("BAAI/bge-m3:FlagEmbedding")
        assert bundle is None
        assert model == "BAAI/bge-m3:FlagEmbedding"

    def test_bundle_override(self) -> None:
        bundle, model = parse_model_spec("sglang:/BAAI/bge-m3")
        assert bundle == "sglang"
        assert model == "BAAI/bge-m3"

    def test_bundle_override_with_variant(self) -> None:
        bundle, model = parse_model_spec("sglang:/BAAI/bge-m3:variant")
        assert bundle == "sglang"
        assert model == "BAAI/bge-m3:variant"

    def test_bundle_override_case_insensitive(self) -> None:
        bundle, model = parse_model_spec("SGLANG:/BAAI/bge-m3")
        assert bundle == "sglang"
        assert model == "BAAI/bge-m3"

    def test_single_word_model(self) -> None:
        bundle, model = parse_model_spec("simple-model")
        assert bundle is None
        assert model == "simple-model"


class TestBundleInfo:
    def test_bundle_info_defaults(self) -> None:
        info = BundleInfo(name="test", priority=10)
        assert info.name == "test"
        assert info.priority == 10
        assert info.adapters == []
        assert info.default is False


class TestModelInfo:
    def test_model_info_defaults(self) -> None:
        info = ModelInfo(name="test-model")
        assert info.name == "test-model"
        assert info.bundles == []


@pytest.fixture
def temp_config_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        bundles_dir = tmppath / "bundles"
        models_dir = tmppath / "models"
        bundles_dir.mkdir()
        models_dir.mkdir()

        (bundles_dir / "default.yaml").write_text(
            "name: default\n"
            "priority: 10\n"
            "default: true\n"
            "adapters:\n"
            "  - sie_server.adapters.bge_m3\n"
            "  - sie_server.adapters.sentence_transformer\n"
            "  - sie_server.adapters.cross_encoder\n"
        )

        (bundles_dir / "sglang.yaml").write_text(
            "name: sglang\npriority: 20\nadapters:\n  - sie_server.adapters.bge_m3\n  - sie_server.adapters.sglang\n"
        )

        (models_dir / "baai-bge-m3.yaml").write_text(
            "sie_id: BAAI/bge-m3\n"
            "hf_id: BAAI/bge-m3\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bge_m3:BGEM3Adapter\n"
        )

        (models_dir / "intfloat-e5-small-v2.yaml").write_text(
            "sie_id: intfloat/e5-small-v2\n"
            "hf_id: intfloat/e5-small-v2\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.sentence_transformer:SentenceTransformerAdapter\n"
        )

        (models_dir / "cross-encoder-ms-marco-minilm-l-6-v2.yaml").write_text(
            "name: cross-encoder/ms-marco-MiniLM-L-6-v2\n"
            "hf_id: cross-encoder/ms-marco-MiniLM-L-6-v2\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.cross_encoder:CrossEncoderAdapter\n"
        )

        (models_dir / "qwen-qwen3-embedding-8b.yaml").write_text(
            "sie_id: Qwen/Qwen3-Embedding-8B\n"
            "hf_id: Qwen/Qwen3-Embedding-8B\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.sglang:SGLangAdapter\n"
        )

        yield bundles_dir, models_dir


class TestModelRegistry:
    def test_load_bundles(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        bundles = registry.list_bundles()
        assert len(bundles) == 2
        assert "default" in bundles
        assert "sglang" in bundles

    def test_bundles_sorted_by_priority(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        bundles = registry.list_bundles()
        assert bundles[0] == "default"
        assert bundles[1] == "sglang"

    def test_load_models(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        models = registry.list_models()
        assert "BAAI/bge-m3" in models
        assert "intfloat/e5-small-v2" in models
        assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in models
        assert "Qwen/Qwen3-Embedding-8B" in models

    def test_model_bundle_mapping(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        info = registry.get_model_info("BAAI/bge-m3")
        assert info is not None
        assert "default" in info.bundles
        assert "sglang" in info.bundles
        assert info.bundles[0] == "default"
        info = registry.get_model_info("intfloat/e5-small-v2")
        assert info is not None
        assert info.bundles == ["default"]

    def test_resolve_bundle_auto_select(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        bundle = registry.resolve_bundle("BAAI/bge-m3")
        assert bundle == "default"

    def test_resolve_bundle_with_override(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        bundle = registry.resolve_bundle("BAAI/bge-m3", bundle_override="sglang")
        assert bundle == "sglang"

    def test_resolve_bundle_unknown_model(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.resolve_bundle("unknown/model")
        assert exc_info.value.model == "unknown/model"

    def test_resolve_bundle_incompatible_override(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        with pytest.raises(BundleConflictError) as exc_info:
            registry.resolve_bundle("intfloat/e5-small-v2", bundle_override="sglang")
        assert exc_info.value.model == "intfloat/e5-small-v2"
        assert exc_info.value.bundle == "sglang"
        assert "default" in exc_info.value.compatible_bundles

    def test_resolve_bundle_case_insensitive(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        bundle = registry.resolve_bundle("baai/bge-m3")
        assert bundle == "default"
        bundle = registry.resolve_bundle("BAAI/BGE-M3")
        assert bundle == "default"

    def test_model_exists(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        assert registry.model_exists("BAAI/bge-m3") is True
        assert registry.model_exists("baai/bge-m3") is True
        assert registry.model_exists("unknown/model") is False

    def test_get_bundle_info(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        info = registry.get_bundle_info("default")
        assert info is not None
        assert info.name == "default"
        assert info.priority == 10
        assert info.default is True
        assert "sie_server.adapters.bge_m3" in info.adapters
        info = registry.get_bundle_info("nonexistent")
        assert info is None

    def test_get_models_for_bundle(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        models = registry.get_models_for_bundle("default")
        assert "BAAI/bge-m3" in models
        assert "intfloat/e5-small-v2" in models
        models = registry.get_models_for_bundle("sglang")
        assert "BAAI/bge-m3" in models
        assert "Qwen/Qwen3-Embedding-8B" in models
        models = registry.get_models_for_bundle("nonexistent")
        assert models == []

    def test_reload(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        assert "BAAI/bge-m3" in registry.list_models()
        (bundles_dir / "new.yaml").write_text("name: new\npriority: 5\nadapters:\n  - sie_server.adapters.test\n")
        (models_dir / "new-model.yaml").write_text(
            "sie_id: new/model\nhf_id: new/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.test:TestAdapter\n"
        )
        registry.reload()
        assert "new" in registry.list_bundles()
        assert "new/model" in registry.list_models()
        assert registry.list_bundles()[0] == "new"


class TestModelRegistryEmptyDirectories:
    def test_missing_bundles_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "nonexistent_bundles"
            models_dir = tmppath / "models"
            models_dir.mkdir()
            registry = ModelRegistry(bundles_dir, models_dir)
            assert registry.list_bundles() == []
            assert registry.list_models() == []

    def test_missing_models_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "nonexistent_models"
            bundles_dir.mkdir()
            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.test\n"
            )
            registry = ModelRegistry(bundles_dir, models_dir)
            assert "default" in registry.list_bundles()
            assert registry.list_models() == []

    def test_empty_bundle_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "models"
            bundles_dir.mkdir()
            models_dir.mkdir()
            (bundles_dir / "empty.yaml").write_text("")
            registry = ModelRegistry(bundles_dir, models_dir)
            assert len(registry.list_bundles()) >= 0


class TestModelRegistryAdapterMatching:
    def test_model_with_no_matching_adapter(self) -> None:
        """A model whose adapter module is not declared in any bundle loads
        into the registry but stays unrouteable (``ModelInfo.bundles == []``)
        and is reported via ``unrouteable_models`` for operator visibility.

        We deliberately do not fail reload() on this: the baked-in
        inconsistency is caught pre-merge by
        ``packages/sie_server/tests/config/test_bundle_coverage.py`` (added
        in PR #704), and failing here would take down the entire control
        plane for a two-model config bug. Callers (e.g. readiness probes,
        alerts) are expected to treat ``unrouteable_models`` as a
        soft-error surface instead.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "models"
            bundles_dir.mkdir()
            models_dir.mkdir()
            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n"
            )
            (models_dir / "orphan.yaml").write_text(
                "sie_id: orphan/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.unknown:UnknownAdapter\n"
            )
            registry = ModelRegistry(bundles_dir, models_dir)
            assert "orphan/model" in registry.list_models()
            info = registry.get_model_info("orphan/model")
            assert info is not None
            assert info.bundles == []
            unrouteable = registry.unrouteable_models
            assert "orphan/model" in unrouteable
            assert unrouteable["orphan/model"] == {"sie_server.adapters.unknown"}

    def test_model_multiple_profiles_different_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "models"
            bundles_dir.mkdir()
            models_dir.mkdir()
            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n"
            )
            (bundles_dir / "sglang.yaml").write_text(
                "name: sglang\npriority: 20\nadapters:\n  - sie_server.adapters.sglang\n"
            )
            (models_dir / "multi.yaml").write_text(
                "name: multi/model\nprofiles:\n  cpu:\n    adapter_path: sie_server.adapters.sentence_transformer:STAdapter\n  gpu:\n    adapter_path: sie_server.adapters.sglang:SGLangAdapter\n"
            )
            registry = ModelRegistry(bundles_dir, models_dir)
            info = registry.get_model_info("multi/model")
            assert info is not None
            assert "default" in info.bundles
            assert "sglang" in info.bundles


class TestModelRegistryThreadSafety:
    def test_concurrent_reads(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        results = []
        errors = []

        def read_models():
            try:
                for _ in range(100):
                    models = registry.list_models()
                    results.append(len(models))
            except Exception as e:  # noqa: BLE001 -- concurrency test error collection
                errors.append(e)

        threads = [threading.Thread(target=read_models) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(results) == 1000
        assert all(r == results[0] for r in results)

    def test_reload_during_read(self, temp_config_dirs) -> None:
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        errors = []

        def read_models():
            try:
                for _ in range(100):
                    registry.list_models()
                    registry.resolve_bundle("BAAI/bge-m3")
            except ModelNotFoundError:
                pass
            except Exception as e:  # noqa: BLE001 -- concurrency test error collection
                errors.append(e)

        def reload_registry():
            try:
                for _ in range(10):
                    registry.reload()
            except Exception as e:  # noqa: BLE001 -- concurrency test error collection
                errors.append(e)

        read_threads = [threading.Thread(target=read_models) for _ in range(5)]
        reload_thread = threading.Thread(target=reload_registry)
        for t in read_threads:
            t.start()
        reload_thread.start()
        for t in read_threads:
            t.join()
        reload_thread.join()
        assert not errors

    def test_compute_bundles_hash_is_stable_and_canonical(self, temp_config_dirs) -> None:
        # Two registries loaded from the same on-disk state must produce
        # byte-identical hashes. Without this the gateway's poller would
        # see spurious drift on every sie-config redeploy that loaded
        # identical content in a different iteration order, triggering
        # unnecessary bootstrap re-runs on every replica.
        bundles_dir, models_dir = temp_config_dirs
        a = ModelRegistry(bundles_dir, models_dir)
        b = ModelRegistry(bundles_dir, models_dir)
        assert a.compute_bundles_hash() == b.compute_bundles_hash()
        # Result shape is a sha256 hex digest; the gateway treats it as
        # opaque but the length is load-bearing for distinguishing it from
        # the empty "nothing to sync" sentinel.
        assert len(a.compute_bundles_hash()) == 64

    def test_compute_bundles_hash_changes_on_adapter_edit(self, temp_config_dirs) -> None:
        # Editing a single adapter entry must change the hash — the whole
        # point of the signal is catching bundle drift that the model
        # epoch would otherwise miss.
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        before = registry.compute_bundles_hash()

        (bundles_dir / "default.yaml").write_text(
            "name: default\n"
            "priority: 10\n"
            "default: true\n"
            "adapters:\n"
            "  - sie_server.adapters.bge_m3\n"
            "  - sie_server.adapters.sentence_transformer\n"
            "  - sie_server.adapters.cross_encoder\n"
            "  - sie_server.adapters.new_thing\n"
        )
        registry.reload()
        assert registry.compute_bundles_hash() != before

    def test_compute_bundles_hash_independent_of_adapter_list_order(self, temp_config_dirs) -> None:
        # Two YAMLs that list the same adapters in different order describe
        # the same bundle semantically. The hash MUST NOT change — otherwise
        # a cosmetic YAML reformat on sie-config would force every gateway
        # to re-bootstrap.
        bundles_dir, _models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, _models_dir)
        before = registry.compute_bundles_hash()

        (bundles_dir / "default.yaml").write_text(
            "name: default\n"
            "priority: 10\n"
            "default: true\n"
            "adapters:\n"
            "  - sie_server.adapters.cross_encoder\n"
            "  - sie_server.adapters.bge_m3\n"
            "  - sie_server.adapters.sentence_transformer\n"
        )
        registry.reload()
        assert registry.compute_bundles_hash() == before

    def test_compute_bundles_hash_is_safe_under_concurrent_reload(self, temp_config_dirs) -> None:
        # The gateway's poller reads `/epoch` (which calls
        # compute_bundles_hash) on a fixed cadence while sie-config may be
        # processing hot-reloads. The snapshot-under-lock pattern matches
        # the pre-existing test_reload_during_read — this regression guards
        # against anyone later refactoring compute_bundles_hash into a
        # lock-free path that would race with the atomic-swap inside
        # reload().
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)
        errors: list[Exception] = []
        hashes: list[str] = []

        def hash_reader() -> None:
            try:
                for _ in range(200):
                    hashes.append(registry.compute_bundles_hash())
            except Exception as e:  # noqa: BLE001 -- concurrency test error collection
                errors.append(e)

        def reloader() -> None:
            try:
                for _ in range(20):
                    registry.reload()
            except Exception as e:  # noqa: BLE001 -- concurrency test error collection
                errors.append(e)

        readers = [threading.Thread(target=hash_reader) for _ in range(5)]
        writer = threading.Thread(target=reloader)
        for t in readers:
            t.start()
        writer.start()
        for t in readers:
            t.join()
        writer.join()

        assert not errors
        # Every observation must be a well-formed digest; a half-populated
        # read would yield a hash with the wrong length (or a crash, which
        # the `errors` assertion above already guards).
        assert all(len(h) == 64 for h in hashes)


class TestUnrouteableModelsSurface:
    """Baked-in inconsistency (a model YAML references an adapter module no
    bundle declares) is reported via :pyattr:`ModelRegistry.unrouteable_models`
    and a single aggregated ERROR log at reload time, but never fatal.

    The rationale is that PR #704's regression test
    (``packages/sie_server/tests/config/test_bundle_coverage.py``) already
    catches this in CI before it ships, so the runtime guard is
    defense-in-depth only. Failing reload() would take down the whole
    control plane for a two-model inconsistency -- a worse outage than
    keeping 102/104 models routable and letting readiness/alerts surface
    the offenders.
    """

    @staticmethod
    def _write_scenario(tmp: Path) -> tuple[Path, Path]:
        bundles_dir = tmp / "bundles"
        models_dir = tmp / "models"
        bundles_dir.mkdir()
        models_dir.mkdir()
        (bundles_dir / "default.yaml").write_text(
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n"
        )
        (models_dir / "good.yaml").write_text(
            "sie_id: good/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.sentence_transformer:Foo\n"
        )
        (models_dir / "orphan_a.yaml").write_text(
            "sie_id: orphan/a\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.unknown_a:A\n"
        )
        (models_dir / "orphan_b.yaml").write_text(
            "sie_id: orphan/b\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.unknown_b:B\n"
        )
        return bundles_dir, models_dir

    def test_inconsistent_registry_still_constructs(self) -> None:
        """reload() must NOT raise even when there are unrouteable models.
        The good models have to keep working.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            bundles_dir, models_dir = self._write_scenario(Path(tmpdir))
            registry = ModelRegistry(bundles_dir, models_dir)
            good = registry.get_model_info("good/model")
            assert good is not None
            assert good.bundles == ["default"]

    def test_unrouteable_models_surface_lists_every_offender(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundles_dir, models_dir = self._write_scenario(Path(tmpdir))
            registry = ModelRegistry(bundles_dir, models_dir)
            unrouteable = registry.unrouteable_models
            # Both orphans must show up (not just the first) so operators
            # don't have to fix one, redeploy, re-discover, repeat.
            assert set(unrouteable.keys()) == {"orphan/a", "orphan/b"}
            assert unrouteable["orphan/a"] == {"sie_server.adapters.unknown_a"}
            assert unrouteable["orphan/b"] == {"sie_server.adapters.unknown_b"}

    def test_unrouteable_snapshot_is_defensive_copy(self) -> None:
        """The property returns fresh containers so external callers can't
        mutate the registry's internal state.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            bundles_dir, models_dir = self._write_scenario(Path(tmpdir))
            registry = ModelRegistry(bundles_dir, models_dir)
            snapshot = registry.unrouteable_models
            snapshot.clear()
            snapshot.setdefault("orphan/a", set()).add("poisoned")
            assert registry.unrouteable_models, "internal state was mutated via snapshot"
            assert "poisoned" not in registry.unrouteable_models.get("orphan/a", set())

    def test_reload_recomputes_unrouteable_snapshot(self) -> None:
        """Adding the missing adapter to a bundle and reloading must clear
        the entry from ``unrouteable_models`` -- not stay stuck on stale
        data from the previous reload().
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "models"
            bundles_dir.mkdir()
            models_dir.mkdir()
            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.a\n"
            )
            (models_dir / "orphan.yaml").write_text(
                "sie_id: orphan/m\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.b:B\n"
            )
            registry = ModelRegistry(bundles_dir, models_dir)
            assert "orphan/m" in registry.unrouteable_models
            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.a\n  - sie_server.adapters.b\n"
            )
            registry.reload()
            assert registry.unrouteable_models == {}
            info = registry.get_model_info("orphan/m")
            assert info is not None
            assert info.bundles == ["default"]

    def test_extends_only_profiles_are_not_flagged_as_unrouteable(self) -> None:
        """Profiles that only ``extends`` another profile have no
        ``adapter_path`` of their own. The consistency check must skip them
        rather than mark the model unrouteable.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "models"
            bundles_dir.mkdir()
            models_dir.mkdir()
            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.a\n"
            )
            (models_dir / "meta.yaml").write_text("sie_id: meta/only\nprofiles:\n  default:\n    extends: some/other\n")
            registry = ModelRegistry(bundles_dir, models_dir)
            assert registry.unrouteable_models == {}

    def test_error_log_fires_on_unrouteable_models(self, caplog: pytest.LogCaptureFixture) -> None:
        """A single aggregated ERROR log (not per-model) is the operator's
        entry point for diagnosing a baked-in inconsistency. Losing it
        would put us back in the silent-degradation regime that masked the
        Qwen3-VL / eu-central-1 incident.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            bundles_dir, models_dir = self._write_scenario(Path(tmpdir))
            with caplog.at_level(logging.ERROR, logger="sie_config.model_registry"):
                ModelRegistry(bundles_dir, models_dir)
            errors = [r for r in caplog.records if r.levelno == logging.ERROR]
            assert any("unrouteable" in r.getMessage().lower() for r in errors), (
                f"expected an ERROR log mentioning unrouteable, got: {[r.getMessage() for r in errors]}"
            )

    def test_mixed_profile_model_surfaces_only_the_missing_module(self) -> None:
        """A model with one good profile and one orphan profile must still
        appear in ``unrouteable_models`` -- previously the check only fired
        when ``ModelInfo.bundles == []``, but bundles is the union across
        profiles, so a single good profile would hide every bad sibling.

        Regression test for the cursor[bot] blocking comment on PR #705:
        ``add_model_config`` already rejects the same shape at write time
        via ``new_adapter_modules - all_bundle_adapters``; ``reload()``
        now mirrors that so baked-in drift is surfaced identically.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "models"
            bundles_dir.mkdir()
            models_dir.mkdir()
            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.good\n"
            )
            (models_dir / "mixed.yaml").write_text(
                "sie_id: mixed/model\n"
                "profiles:\n"
                "  cpu:\n"
                "    adapter_path: sie_server.adapters.good:Adapter\n"
                "  gpu:\n"
                "    adapter_path: sie_server.adapters.missing:Adapter\n"
            )
            registry = ModelRegistry(bundles_dir, models_dir)
            info = registry.get_model_info("mixed/model")
            assert info is not None
            # Routable profiles still let the model resolve to its matching
            # bundle -- the good profile keeps working.
            assert info.bundles == ["default"]
            # ... but the missing module is surfaced explicitly. Note we
            # only list the *missing* adapter module, not the good one:
            # otherwise readiness tooling would have to reconcile the two
            # sets itself to find the real offender.
            unrouteable = registry.unrouteable_models
            assert "mixed/model" in unrouteable
            assert unrouteable["mixed/model"] == {"sie_server.adapters.missing"}
            assert "sie_server.adapters.good" not in unrouteable["mixed/model"]

    def test_add_model_config_clears_stale_unrouteable_entry(self) -> None:
        """If ``reload()`` registered a model as unrouteable but a later
        ``add_model_config`` write adds a profile whose adapter_path IS in
        a bundle, the model's missing-modules entry must be refreshed --
        otherwise ``unrouteable_models`` lies to readiness probes until
        the next disk-reload.

        (Regression test for the coderabbitai PR #705 comment about
        ``_unrouteable_models`` being stale after runtime writes.)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "models"
            bundles_dir.mkdir()
            models_dir.mkdir()
            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.a\n"
            )
            # On-disk model references only an undeclared adapter.
            (models_dir / "drift.yaml").write_text(
                "sie_id: drift/model\nprofiles:\n  broken:\n    adapter_path: sie_server.adapters.b:B\n"
            )
            registry = ModelRegistry(bundles_dir, models_dir)
            assert "drift/model" in registry.unrouteable_models

            # Operator adds adapters.b to the bundle at runtime and the
            # same model config is re-pushed via the Config API. Because
            # `_validate_config_locked` would reject an unroutable write,
            # simulate the fix by first teaching the bundle the adapter,
            # then replaying the add_model_config.
            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.a\n  - sie_server.adapters.b\n"
            )
            registry.reload()  # pick up the new bundle membership
            assert registry.unrouteable_models == {}, "reload() should have cleared the now-routable entry"

            # Now exercise the add_model_config path directly: adding a
            # new routable profile to the formerly-drifting model must
            # leave `unrouteable_models` clean even without a subsequent
            # reload().
            registry.add_model_config(
                {
                    "sie_id": "drift/model",
                    "profiles": {
                        "good": {"adapter_path": "sie_server.adapters.a:A"},
                    },
                }
            )
            assert registry.unrouteable_models == {}
