"""Loader contract: every adapter must accept (and honor) a pinned HF ``revision``.

The model loader adds ``revision=config.hf_revision`` to the adapter constructor
kwargs whenever a model YAML pins ``hf_revision`` (version identity).
An adapter whose ``__init__`` rejects ``revision`` hard-fails at load
(``TypeError: __init__() got an unexpected keyword argument 'revision'`` — the
CI eval-loader failure this suite regresses against), and an adapter that accepts
it via ``**kwargs`` but never forwards it silently drops the pin.

This module enforces the *acceptance* half of that contract for EVERY adapter
class the package exposes, parameterized so a future adapter cannot regress
silently, plus *forwarding* spot-checks (clip / siglip / a flash adapter) that a
pinned revision actually reaches the HuggingFace ``from_pretrained`` call.

Pure and offline: no weights are downloaded and no model is loaded — HF loaders
are mocked, and signature inspection imports only the adapter modules.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from unittest.mock import MagicMock, patch

import pytest
import sie_server.adapters as adapters_pkg
import torch
from sie_server.adapters.base import ModelAdapter

# Shared/abstract bases that are never instantiated as concrete adapters. The
# ``inspect.isabstract`` filter already drops these (they leave ``load`` etc.
# unimplemented), but naming them keeps the intent explicit and stable.
_ABSTRACT_BASES = frozenset({"ModelAdapter", "BaseAdapter", "FlashBaseAdapter"})


def _discover_adapter_classes() -> list[type[ModelAdapter]]:
    """Import every adapter submodule and return the concrete adapter classes.

    Modules that cannot be imported because an optional heavy dependency is
    absent from the unit environment (e.g. ``sglang``/``mlx_lm`` off their target
    platforms) are skipped — those adapters are exercised in the image that ships
    the dependency. Real import errors surface via the count/core guards below.
    """
    discovered: dict[str, type[ModelAdapter]] = {}
    for mod_info in pkgutil.walk_packages(adapters_pkg.__path__, prefix=f"{adapters_pkg.__name__}."):
        try:
            module = importlib.import_module(mod_info.name)
        except Exception:  # noqa: BLE001, S112 — optional deps may be absent in the unit env; skip
            continue
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != mod_info.name:
                continue  # skip base classes / symbols imported into the module
            if not issubclass(obj, ModelAdapter) or name in _ABSTRACT_BASES:
                continue
            if inspect.isabstract(obj):
                continue
            discovered[f"{obj.__module__}.{name}"] = obj
    return [discovered[key] for key in sorted(discovered)]


_ADAPTER_CLASSES = _discover_adapter_classes()


def _accepts_revision(cls: type) -> bool:
    """True if ``cls.__init__`` accepts a ``revision`` kwarg (explicit or VAR_KEYWORD)."""
    params = inspect.signature(cls.__init__).parameters.values()
    return any(param.name == "revision" or param.kind is inspect.Parameter.VAR_KEYWORD for param in params)


def test_discovery_imported_the_adapter_package() -> None:
    """Guard: discovery actually walked the package (not a mass import failure)."""
    # There are dozens of adapters; a tiny count means imports broke wholesale.
    assert len(_ADAPTER_CLASSES) >= 20, f"only discovered {len(_ADAPTER_CLASSES)} adapters — imports likely broke"
    names = {cls.__name__ for cls in _ADAPTER_CLASSES}
    # These depend only on always-present core deps (transformers /
    # sentence-transformers), so they must always be discovered.
    for required in ("CLIPAdapter", "SiglipAdapter", "CrossEncoderAdapter", "PyTorchEmbeddingAdapter"):
        assert required in names, f"{required} not discovered — adapter import regressed"


@pytest.mark.parametrize("adapter_cls", _ADAPTER_CLASSES, ids=lambda cls: cls.__name__)
def test_adapter_init_accepts_revision(adapter_cls: type) -> None:
    """Every adapter ``__init__`` must accept the loader's pinned ``revision`` kwarg."""
    assert _accepts_revision(adapter_cls), (
        f"{adapter_cls.__module__}.{adapter_cls.__name__}.__init__ must accept a 'revision' keyword "
        "(explicit 'revision: str | None = None' or **kwargs) so a YAML-pinned hf_revision is not "
        "rejected by the loader (see core/loader._build_adapter_kwargs)."
    )


class TestRevisionReachesFromPretrained:
    """Forwarding spot-checks: a pinned revision reaches the HF load calls."""

    def test_clip_forwards_revision(self) -> None:
        from sie_server.adapters.clip import CLIPAdapter

        adapter = CLIPAdapter("openai/clip-vit-base-patch32", revision="deadbeef")
        assert adapter._revision == "deadbeef"

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        with (
            patch("transformers.CLIPProcessor") as mock_proc,
            patch("transformers.CLIPModel") as mock_mdl,
        ):
            mock_mdl.from_pretrained.return_value = mock_model
            adapter.load("cpu")

        assert mock_proc.from_pretrained.call_args.kwargs["revision"] == "deadbeef"
        assert mock_mdl.from_pretrained.call_args.kwargs["revision"] == "deadbeef"

    def test_clip_without_revision_omits_kwarg(self) -> None:
        from sie_server.adapters.clip import CLIPAdapter

        adapter = CLIPAdapter("openai/clip-vit-base-patch32")
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        with (
            patch("transformers.CLIPProcessor") as mock_proc,
            patch("transformers.CLIPModel") as mock_mdl,
        ):
            mock_mdl.from_pretrained.return_value = mock_model
            adapter.load("cpu")

        assert "revision" not in mock_proc.from_pretrained.call_args.kwargs
        assert "revision" not in mock_mdl.from_pretrained.call_args.kwargs

    def test_siglip_transformers_backend_forwards_revision(self) -> None:
        from sie_server.adapters.siglip.adapter import SiglipAdapter

        adapter = SiglipAdapter("google/siglip-so400m-patch14-384", revision="cafef00d")
        assert adapter._revision == "cafef00d"

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        with (
            patch("transformers.SiglipProcessor") as mock_proc,
            patch("transformers.SiglipModel") as mock_mdl,
        ):
            mock_mdl.from_pretrained.return_value = mock_model
            adapter.load("cpu")

        assert mock_proc.from_pretrained.call_args.kwargs["revision"] == "cafef00d"
        assert mock_mdl.from_pretrained.call_args.kwargs["revision"] == "cafef00d"

    def test_flash_adapter_forwards_revision(self) -> None:
        """A representative flash embedding adapter threads revision into both loaders."""
        from sie_server.adapters.bert_flash import BertFlashAdapter

        adapter = BertFlashAdapter("intfloat/e5-base-v2", revision="0ff1ce")
        assert adapter._revision == "0ff1ce"

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        with (
            patch("transformers.AutoTokenizer") as mock_tok,
            patch("transformers.AutoModel") as mock_mdl,
        ):
            mock_mdl.from_pretrained.return_value = mock_model
            # Flash adapters require a CUDA device string; the HF loaders and
            # ``.to()`` are mocked, so no real GPU is touched.
            adapter.load("cuda:0")

        assert mock_tok.from_pretrained.call_args.kwargs["revision"] == "0ff1ce"
        assert mock_mdl.from_pretrained.call_args.kwargs["revision"] == "0ff1ce"

    def test_rotary_colbert_forwards_model_and_code_revisions(self) -> None:
        """External trusted code is pinned independently from model artifacts."""
        from sie_server.adapters.colbert_rotary_flash import ColBERTRotaryFlashAdapter

        adapter = ColBERTRotaryFlashAdapter(
            "jinaai/jina-colbert-v2",
            revision="model-sha",
            code_revision="code-sha",
        )
        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token_id = None
        mock_tokenizer.vocab = {}
        mock_model = MagicMock()
        mock_model.config.hidden_size = 1024
        projection = torch.nn.Linear(1024, 128, bias=False)
        with (
            patch("transformers.AutoTokenizer") as mock_tok,
            patch("transformers.AutoModel") as mock_mdl,
            patch(
                "sie_server.adapters.colbert_rotary_flash.load_standalone_colbert_projection",
                return_value=projection,
            ) as mock_projection,
        ):
            mock_tok.from_pretrained.return_value = mock_tokenizer
            mock_mdl.from_pretrained.return_value = mock_model
            adapter.load("cuda:0")

        expected = {"revision": "model-sha", "code_revision": "code-sha"}
        for loader in (mock_tok, mock_mdl):
            kwargs = loader.from_pretrained.call_args.kwargs
            assert {key: kwargs[key] for key in expected} == expected
        mock_projection.assert_called_once_with(
            "jinaai/jina-colbert-v2",
            revision="model-sha",
            device="cuda:0",
            dtype=torch.bfloat16,
            expected_in_features=1024,
            expected_out_features=128,
            allow_bias=False,
            required=True,
        )

    def test_rotary_colbert_fallback_preserves_model_and_code_revisions(self) -> None:
        """CPU fallback must pin both weights and trusted external code."""
        from sie_server.adapters.colbert import ColBERTAdapter
        from sie_server.adapters.colbert_rotary_flash import ColBERTRotaryFlashAdapter

        with patch("sie_server.core.inference.is_flash_attention_available", return_value=False):
            adapter = ColBERTRotaryFlashAdapter.create_for_device(
                "cpu",
                model_name_or_path="jinaai/jina-colbert-v2",
                revision="model-sha",
                code_revision="code-sha",
            )

        assert isinstance(adapter, ColBERTAdapter)
        assert adapter._revision == "model-sha"
        assert adapter._code_revision == "code-sha"

        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token_id = None
        mock_tokenizer.vocab = {}
        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_model = MagicMock()
        mock_model.config.hidden_size = 1024
        mock_model.named_modules.return_value = []
        projection = torch.nn.Linear(1024, 128, bias=False)
        with (
            patch("transformers.AutoTokenizer") as mock_tok,
            patch("transformers.AutoConfig") as mock_cfg,
            patch("transformers.AutoModel") as mock_mdl,
            patch("sie_server.adapters.colbert.load_pylate_dense_chain", return_value=None),
            patch(
                "sie_server.adapters.colbert.load_standalone_colbert_projection",
                return_value=projection,
            ),
        ):
            mock_tok.from_pretrained.return_value = mock_tokenizer
            mock_cfg.from_pretrained.return_value = mock_config
            mock_mdl.from_pretrained.return_value = mock_model
            adapter.load("cpu")

        expected = {"revision": "model-sha", "code_revision": "code-sha"}
        for loader in (mock_tok, mock_cfg, mock_mdl):
            kwargs = loader.from_pretrained.call_args.kwargs
            assert {key: kwargs[key] for key in expected} == expected

    def test_colsmol_pins_lora_repo_only_not_base(self) -> None:
        """ColSmol pins the LoRA-repo loads but never the separate base-checkpoint repo.

        Regression: ``base_id`` is a *different* repo discovered from the LoRA's
        ``adapter_config``; forwarding the LoRA repo's revision SHA to it raises
        RevisionNotFoundError → 503 on the first request under a pinned YAML.
        """
        from sie_server.adapters import colsmol
        from sie_server.adapters.colsmol import ColSmolAdapter

        base_repo = "vidore/ColSmolVLM-Instruct-256M-base"
        adapter = ColSmolAdapter("vidore/colSmol-256M", revision="feedface")
        assert adapter._revision == "feedface"

        peft_cfg = MagicMock()
        peft_cfg.base_model_name_or_path = base_repo
        base_cls = MagicMock()
        with (
            patch("peft.PeftConfig") as mock_peft_cfg,
            patch("peft.PeftModel") as mock_peft_model,
            patch("transformers.Idefics3Config") as mock_config,
            patch("transformers.Idefics3Processor") as mock_proc,
            patch.object(colsmol, "_make_colidefics3_cls", return_value=base_cls),
        ):
            mock_peft_cfg.from_pretrained.return_value = peft_cfg
            adapter.load("cpu")

        # LoRA-repo loads (they read vidore/colSmol-256M) carry the pin.
        assert mock_peft_cfg.from_pretrained.call_args.kwargs["revision"] == "feedface"
        assert mock_proc.from_pretrained.call_args.kwargs["revision"] == "feedface"
        assert mock_peft_model.from_pretrained.call_args.kwargs["revision"] == "feedface"
        # base_id loads target the *base* repo and must NOT carry the LoRA revision.
        assert mock_config.from_pretrained.call_args.args[0] == base_repo
        assert "revision" not in mock_config.from_pretrained.call_args.kwargs
        assert base_cls.from_pretrained.call_args.args[0] == base_repo
        assert "revision" not in base_cls.from_pretrained.call_args.kwargs
