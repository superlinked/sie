from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from sie_config.model_schema import model_config_schema_errors
from sie_server.config.model import ModelConfig
from sie_server.core.loader import _build_adapter_kwargs, load_adapter, load_model_config
from transformers import PretrainedConfig

_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "opensearch-project__opensearch-neural-sparse-encoding-doc-v3-gte.yaml"
)
_WEIGHTS_REVISION = "1646fef40807937e8e130c66d327a26421c408d5"
_CODE_REVISION = "e244c9398934a48b4663dacf7ff4965961a6807a"


def _config() -> ModelConfig:
    declaration = yaml.safe_load(_MODEL_PATH.read_text())
    assert model_config_schema_errors(declaration) == []
    return ModelConfig.model_validate(declaration)


def test_catalog_preserves_independent_weights_and_code_revisions() -> None:
    config = _config()
    profile = config.resolve_profile("default")

    assert config.hf_revision == _WEIGHTS_REVISION
    assert profile.loadtime["code_revision"] == _CODE_REVISION
    assert profile.loadtime["trust_remote_code"] is True

    kwargs = _build_adapter_kwargs(config, "float32")
    assert kwargs["revision"] == _WEIGHTS_REVISION
    assert kwargs["code_revision"] == _CODE_REVISION


@pytest.mark.parametrize(
    ("auto_class", "module", "class_name"),
    [("AutoConfig", "configuration", "NewConfig"), ("AutoModelForMaskedLM", "modeling", "NewForMaskedLM")],
)
def test_catalog_profile_selects_pinned_external_code(auto_class: str, module: str, class_name: str) -> None:
    config = load_model_config(_MODEL_PATH)
    adapter = load_adapter(config, _MODEL_PATH.parent, device="cpu")
    remote_config = {"auto_map": {auto_class: f"Alibaba-NLP/new-impl--{module}.{class_name}"}}

    with (
        patch("transformers.AutoTokenizer.from_pretrained") as load_tokenizer,
        patch("transformers.models.auto.auto_factory.cached_file", return_value=None) as config_file,
        patch("transformers.models.auto.auto_factory.find_adapter_config_file", return_value=None),
        patch("transformers.PretrainedConfig.get_config_dict", return_value=(remote_config, {})),
        patch(
            "transformers.dynamic_module_utils.get_cached_module_file",
            side_effect=RuntimeError("dynamic module loader reached"),
        ) as module_load,
        pytest.raises(RuntimeError, match="dynamic module loader reached"),
    ):
        if auto_class == "AutoModelForMaskedLM":
            with patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=(PretrainedConfig(auto_map=remote_config["auto_map"]), {}),
            ):
                adapter.load("cpu")
        else:
            adapter.load("cpu")

    assert load_tokenizer.call_args.args == (config.hf_id,)
    assert load_tokenizer.call_args.kwargs["revision"] == _WEIGHTS_REVISION
    assert load_tokenizer.call_args.kwargs["code_revision"] == _CODE_REVISION
    assert config_file.call_args.args == (config.hf_id, "config.json")
    assert config_file.call_args.kwargs["revision"] == _WEIGHTS_REVISION
    assert module_load.call_args.args == ("Alibaba-NLP/new-impl", f"{module}.py")
    assert module_load.call_args.kwargs["revision"] == _CODE_REVISION
