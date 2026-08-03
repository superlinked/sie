from __future__ import annotations

from typing import Any

import pytest
from sie_server.adapters.colpali import ColPaliAdapter
from sie_server.adapters.colqwen2 import ColQwen2Adapter
from sie_server.adapters.colqwen3 import ColQwen3Adapter
from sie_server.adapters.colsmol import ColSmolAdapter
from sie_server.adapters.nemo_colembed import NemoColEmbedAdapter
from sie_server.core.postprocessor import MuveraPostprocessor

_VISUAL_ADAPTERS: tuple[tuple[type[Any], str], ...] = (
    (ColPaliAdapter, "vidore/colpali-v1.3-hf"),
    (ColQwen2Adapter, "vidore/colqwen2.5-v0.2"),
    (ColQwen3Adapter, "TomoroAI/tomoro-colqwen3-embed-4b"),
    (ColSmolAdapter, "vidore/colSmol-256M"),
    (NemoColEmbedAdapter, "nvidia/llama-nemoretriever-colembed-3b-v1"),
)


@pytest.mark.parametrize(("adapter_cls", "model_id"), _VISUAL_ADAPTERS)
def test_visual_adapter_registers_configured_muvera_postprocessor(
    adapter_cls: type[Any],
    model_id: str,
) -> None:
    muvera_config: dict[str, Any] = {
        "num_repetitions": 2,
        "num_simhash_projections": 3,
        "projection_dim": 4,
        "final_projection_dim": 16,
        "seed": 7,
        "normalize": True,
        "center_tokens": True,
    }
    adapter = adapter_cls(model_id, token_dim=17, muvera_config=muvera_config)

    postprocessors = adapter.get_postprocessors()

    assert postprocessors is not None
    assert set(postprocessors) == {"muvera"}
    postprocessor = postprocessors["muvera"]
    assert isinstance(postprocessor, MuveraPostprocessor)
    assert postprocessor.token_dim == 17
    assert postprocessor.target_dim == 16
    assert postprocessor.config.num_repetitions == 2
    assert postprocessor.config.num_simhash_projections == 3
    assert postprocessor.config.projection_dim == 4
    assert postprocessor.config.final_projection_dim == 16
    assert postprocessor.config.seed == 7
    assert postprocessor.config.normalize is True
    assert postprocessor.config.center_tokens is True
