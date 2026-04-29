import tempfile
from pathlib import Path

from sie_config.config_store import ConfigStore


class TestConfigStore:
    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = ConfigStore(self._tmpdir.name)

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_initial_epoch_is_zero(self) -> None:
        assert self.store.read_epoch() == 0

    def test_increment_epoch_from_zero(self) -> None:
        new_epoch = self.store.increment_epoch()
        assert new_epoch == 1
        assert self.store.read_epoch() == 1

    def test_increment_epoch_sequential(self) -> None:
        self.store.increment_epoch()
        self.store.increment_epoch()
        assert self.store.read_epoch() == 2

    def test_increment_epoch_returns_new_value(self) -> None:
        assert self.store.increment_epoch() == 1
        assert self.store.increment_epoch() == 2
        assert self.store.increment_epoch() == 3

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
        self.store.write_model("org/model", "test: true\n")
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
        self.store.write_model("good/model", "sie_id: good/model\nfield: ok\n")
        from sie_sdk.storage import join_path

        corrupt_path = join_path(self.store.base_dir, "models", "corrupt__model.yaml")
        self.store._backend.write_text(corrupt_path, "{{not valid yaml: [")
        all_models = self.store.load_all_models()
        assert len(all_models) == 1
        assert "good/model" in all_models

    def test_load_all_models_with_empty_file(self) -> None:
        self.store.write_model("empty/model", "")
        self.store.write_model("good/model", "sie_id: good/model\n")
        all_models = self.store.load_all_models()
        assert "good/model" in all_models

    def test_epoch_survives_reopen(self) -> None:
        self.store.increment_epoch()
        self.store.increment_epoch()
        store2 = ConfigStore(self.store.base_dir)
        assert store2.read_epoch() == 2

    def test_models_survive_reopen(self) -> None:
        self.store.write_model("org/model", "sie_id: org/model\n")
        store2 = ConfigStore(self.store.base_dir)
        assert store2.read_model("org/model") == "sie_id: org/model\n"
