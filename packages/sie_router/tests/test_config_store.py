import tempfile
from pathlib import Path

import pytest
from sie_router.config_store import ConfigStore, EpochCASError


class TestConfigStore:
    """Tests for ConfigStore local filesystem backend."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = ConfigStore(self._tmpdir.name)

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_initial_epoch_is_zero(self) -> None:
        assert self.store.read_epoch() == 0

    def test_cas_epoch_from_zero(self) -> None:
        new_epoch = self.store.cas_epoch(0)
        assert new_epoch == 1
        assert self.store.read_epoch() == 1

    def test_cas_epoch_sequential(self) -> None:
        self.store.cas_epoch(0)
        self.store.cas_epoch(1)
        assert self.store.read_epoch() == 2

    def test_cas_epoch_fails_on_mismatch(self) -> None:
        self.store.cas_epoch(0)  # epoch is now 1
        with pytest.raises(EpochCASError) as exc_info:
            self.store.cas_epoch(0)  # expected 0 but actual is 1
        assert exc_info.value.expected == 0
        assert exc_info.value.actual == 1

    def test_write_and_read_model(self) -> None:
        yaml_content = "sie_id: BAAI/bge-m3\nhf_id: BAAI/bge-m3\n"
        self.store.write_model("BAAI/bge-m3", yaml_content)
        assert self.store.read_model("BAAI/bge-m3") == yaml_content

    def test_read_missing_model_returns_none(self) -> None:
        assert self.store.read_model("nonexistent/model") is None

    def test_list_models_empty(self) -> None:
        assert self.store.list_models() == []

    def test_list_models_after_write(self) -> None:
        self.store.write_model("BAAI/bge-m3", "sie_id: BAAI/bge-m3\n")
        self.store.write_model("intfloat/e5-base", "sie_id: intfloat/e5-base\n")
        models = self.store.list_models()
        assert sorted(models) == ["BAAI/bge-m3", "intfloat/e5-base"]

    def test_model_id_with_slash_sanitized(self) -> None:
        """Model IDs with / are stored with __ in filename."""
        self.store.write_model("org/model", "test: true\n")
        # Verify via read (slash sanitization is internal)
        assert self.store.read_model("org/model") == "test: true\n"
        assert "org/model" in self.store.list_models()

    def test_load_all_models(self) -> None:
        self.store.write_model("m1", "sie_id: m1\nfield: a\n")
        self.store.write_model("m2", "sie_id: m2\nfield: b\n")
        all_models = self.store.load_all_models()
        assert len(all_models) == 2
        assert all_models["m1"]["sie_id"] == "m1"
        assert all_models["m2"]["field"] == "b"

    def test_write_overwrites_existing(self) -> None:
        self.store.write_model("m1", "version: 1\n")
        self.store.write_model("m1", "version: 2\n")
        assert self.store.read_model("m1") == "version: 2\n"

    def test_directories_created_automatically(self, tmp_path: Path) -> None:
        nested = str(tmp_path / "nested" / "config")
        store = ConfigStore(nested)
        store.write_model("test/model", "sie_id: test/model\n")
        assert store.read_model("test/model") == "sie_id: test/model\n"

    def test_load_all_models_with_corrupt_yaml(self) -> None:
        """Corrupt YAML files are skipped gracefully."""
        self.store.write_model("good/model", "sie_id: good/model\nfield: ok\n")
        # Write corrupt YAML directly via backend
        from sie_sdk.storage import join_path

        corrupt_path = join_path(self.store.base_dir, "models", "corrupt__model.yaml")
        self.store._backend.write_text(corrupt_path, "{{not valid yaml: [")

        all_models = self.store.load_all_models()
        assert len(all_models) == 1
        assert "good/model" in all_models

    def test_load_all_models_with_empty_file(self) -> None:
        """Empty YAML files are skipped."""
        self.store.write_model("empty/model", "")
        self.store.write_model("good/model", "sie_id: good/model\n")

        all_models = self.store.load_all_models()
        assert "good/model" in all_models

    def test_epoch_survives_reopen(self) -> None:
        """Epoch persists across ConfigStore instances."""
        self.store.cas_epoch(0)
        self.store.cas_epoch(1)

        # Create new store pointing to same directory
        store2 = ConfigStore(self.store.base_dir)
        assert store2.read_epoch() == 2

    def test_models_survive_reopen(self) -> None:
        """Models persist across ConfigStore instances."""
        self.store.write_model("org/model", "sie_id: org/model\n")

        store2 = ConfigStore(self.store.base_dir)
        assert store2.read_model("org/model") == "sie_id: org/model\n"

    def test_concurrent_cas_one_fails(self) -> None:
        """Simulates concurrent CAS — second writer fails."""
        # Both readers see epoch=0
        epoch_a = self.store.read_epoch()
        epoch_b = self.store.read_epoch()
        assert epoch_a == 0
        assert epoch_b == 0

        # Writer A succeeds
        self.store.cas_epoch(epoch_a)
        assert self.store.read_epoch() == 1

        # Writer B fails (expected 0, actual 1)
        with pytest.raises(EpochCASError) as exc:
            self.store.cas_epoch(epoch_b)
        assert exc.value.expected == 0
        assert exc.value.actual == 1

        # Writer B retries with correct epoch
        new_epoch = self.store.read_epoch()
        self.store.cas_epoch(new_epoch)
        assert self.store.read_epoch() == 2
