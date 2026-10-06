from pathlib import Path

import pytest
import yaml
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.bundle_requirements import resolve_bundle_requirements
from sie_server.core.loader import load_model_configs, resolve_adapter_class

SERVER_ROOT = Path(__file__).resolve().parents[2]
MODELS = SERVER_ROOT / "models"
BUNDLES = SERVER_ROOT / "bundles"
GLINER_ADAPTER = "sie_server.adapters.gliner2.entities:GLiNER2EntitiesAdapter"
PRIVACY_ADAPTER = "sie_server.adapters.privacy_filter.adapter:PrivacyFilterAdapter"


@pytest.mark.parametrize(
    ("model_id", "revision", "adapter_path", "capacity"),
    [
        ("fastino/gliner2.5-base-v1", "ca906247640776a07753514055be9726f9080ead", GLINER_ADAPTER, 4096),
        ("fastino/gliner2.5-small-v1", "7132dc4561c3f94563c6147e75ffa8ef34c4964a", GLINER_ADAPTER, 4096),
        ("fastino/gliner2-privacy-filter-PII-multi", "1cb4166094dc58fa8d836429f060d6c95f62b495", GLINER_ADAPTER, 4096),
        ("openai/privacy-filter", "7ffa9a043d54d1be65afb281eddf0ffbe629385b", PRIVACY_ADAPTER, 128000),
    ],
)
def test_native_entity_profiles_are_pinned_discoverable_and_bundle_routed(model_id, revision, adapter_path, capacity):
    catalog = load_model_configs(MODELS)
    config = catalog[model_id]
    assert config.hf_id == model_id
    assert config.hf_revision == revision
    assert config.max_sequence_length == capacity
    assert config.inputs.text is True
    assert config.tasks.extract is not None
    assert config.tasks.encode is None
    assert config.tasks.score is None
    assert config.resolve_profile("default").adapter_path == adapter_path
    assert resolve_adapter_class(config, MODELS).__name__ == adapter_path.split(":")[1]
    assert model_id in match_bundle_models(BUNDLES / "transformers5.yaml", MODELS)
    assert model_id not in match_bundle_models(BUNDLES / "default.yaml", MODELS)


def test_official_privacy_runtime_is_an_immutable_bundle_dependency():
    bundle = yaml.safe_load((BUNDLES / "transformers5.yaml").read_text())
    requirements = resolve_bundle_requirements(bundle["deps"])
    assert "opf @ git+https://github.com/openai/privacy-filter@f7f00ca7fb869683eb732c010299d901457f19c3" in requirements
    assert "gliner2==2.0.0" in requirements
    assert "tiktoken==0.12.0" in requirements
    assert "gliner2[local]" not in requirements


def test_existing_gliner2_routes_keep_their_adapter_and_default_dependency():
    catalog = load_model_configs(MODELS)
    for model_id in ("fastino/gliner2-base-v1", "fastino/gliner2-large-v1"):
        assert (
            catalog[model_id].resolve_profile("default").adapter_path
            == "sie_server.adapters.gliner2.adapter:GLiNER2Adapter"
        )
        assert model_id in match_bundle_models(BUNDLES / "default.yaml", MODELS)
    for model_id in ("fastino/GLiNER2.5-Decide", "fastino/GLiNER2.5-multi-Decide", "fastino/GLiNER2.5-Decide-1B"):
        assert (
            catalog[model_id].resolve_profile("default").adapter_path
            == "sie_server.adapters.gliner2.decide:GLiNER2DecideAdapter"
        )
    default = yaml.safe_load((BUNDLES / "default.yaml").read_text())
    assert default["deps"]["gliner2"] == ">=1.3.1,<2"
