"""Decision families stay pinned, text-only and confined to their runtime bundle."""

from pathlib import Path

import pytest
import yaml
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.api.systemone import _is_decision_model
from sie_server.config.model import ModelConfig

SERVER = Path(__file__).resolve().parents[2]
MODELS = (
    ("vllm-sr/d3-ultra", "37fee01926f3de396620e794201b074fb427ac2d", "transformers5"),
    ("baselabs/elephant-1", "14537f08b0374edf9867ed0a4d2bd8df59e0400c", "transformers5"),
    ("baselabs/mammoth-1", "9d1962f193e48a68cf0f3598dc7fb4462fd23ad9", "transformers5"),
    ("perplexity-ai/pplx-decider-v1.2", "cce0f3f80033e9d89d80831d5281fd72124aacce", "transformers5"),
    ("PelaAI/KnowLine-4B-Gen5.5", "779944d2cc8ffa19cef386aee83c83bcb967ade2", "sglang-cu130"),
    ("PelaAI/KnowLine-31B-Gen1", "8f052a8130e599b53e5324c34f0f5ff90696e1c6", "sglang-cu130"),
)


@pytest.mark.parametrize(("name", "revision", "bundle_name"), MODELS)
def test_decision_family_contract_and_bundle_routing(name, revision, bundle_name):
    path = SERVER / "models" / f"{name.replace('/', '__')}.yaml"
    config = ModelConfig.model_validate(yaml.safe_load(path.read_text()))
    assert config.hf_revision == revision
    assert config.sie_id == config.hf_id == name
    assert config.inputs.text is True
    assert config.inputs.image is config.inputs.audio is config.inputs.video is False
    assert config.tasks.extract is not None
    assert config.tasks.encode is config.tasks.score is config.tasks.generate is None
    assert _is_decision_model(config)
    for bundle in (SERVER / "bundles").glob("*.yaml"):
        assert (name in match_bundle_models(bundle, SERVER / "models")) == (bundle.stem == bundle_name)


def test_cuda13_decisions_reuse_the_bundle_in_a_separate_worker_pool():
    bundle = SERVER / "bundles/sglang-cu130.yaml"
    decisions = {name for name, _, runtime in MODELS if runtime == "sglang-cu130"}
    assert set(match_bundle_models(bundle, SERVER / "models", pool_name="decision")) == decisions
    assert not decisions.intersection(match_bundle_models(bundle, SERVER / "models", pool_name="default"))
