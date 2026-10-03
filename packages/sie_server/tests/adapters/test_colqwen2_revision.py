"""The colqwen2 revision pin must scope to the adapter repo, not the base.

vidore/colqwen2.5-v0.2 is a PEFT adapter repo: transformers resolves its
adapter_config.json, swaps the load target to base_model_name_or_path
(vidore/colqwen2.5-base), and applies a TOP-LEVEL ``revision`` kwarg to that
base-weights download - where the adapter repo's sha does not exist. Run
32963263930 failed every colqwen cell exactly this way after #3753 pinned
the top-level revision. The pin must ride in ``adapter_kwargs`` instead,
which transformers threads to adapter-config resolution and
``model.load_adapter`` only. The processor is a single-repo load, so its
top-level revision stays.
"""

from unittest.mock import MagicMock, patch

from sie_server.adapters.colqwen2 import ColQwen2Adapter

_PIN = "6f6fcdfd1a114dfe365f529701b33d66b9349014"


def _load_with(revision: str | None) -> tuple[MagicMock, MagicMock]:
    """Run adapter.load with mocked model/processor; return the two mocks."""
    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = MagicMock(dim=128)
    processor = MagicMock()
    with (
        patch("sie_server.adapters.colqwen2._make_colqwen2_5_cls", return_value=model_cls),
        patch(
            "transformers.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor.from_pretrained",
            processor,
        ),
        patch("sie_server.adapters.colqwen2.rebind_vision_patch_embed"),
    ):
        adapter = ColQwen2Adapter("vidore/colqwen2.5-v0.2", revision=revision)
        adapter.load("cpu")
    return model_cls.from_pretrained, processor


def test_pinned_revision_scopes_to_adapter_kwargs() -> None:
    model_call, processor_call = _load_with(_PIN)

    model_kwargs = model_call.call_args.kwargs
    assert "revision" not in model_kwargs
    assert model_kwargs["adapter_kwargs"] == {"revision": _PIN}
    # The processor is a single-repo load on the adapter repo: top-level
    # revision is correct there (verified live against the pin in #3753).
    assert processor_call.call_args.kwargs["revision"] == _PIN


def test_unpinned_load_passes_no_revision_anywhere() -> None:
    model_call, processor_call = _load_with(None)

    model_kwargs = model_call.call_args.kwargs
    assert "revision" not in model_kwargs
    assert "adapter_kwargs" not in model_kwargs
    assert "revision" not in processor_call.call_args.kwargs
