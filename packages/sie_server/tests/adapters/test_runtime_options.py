from __future__ import annotations

from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.types.inputs import Item


class TestAdapterEncodeAcceptsOptions:
    """Verify all adapter encode() methods accept the 'options' keyword parameter.

    The worker pipeline passes options= to adapter.encode(). If any adapter
    is missing the parameter, it will get a TypeError at runtime.
    """

    def test_all_adapters_encode_accepts_options(self) -> None:
        """Every concrete adapter's encode() accepts options kwarg."""
        import inspect

        from sie_server.adapters.base import ModelAdapter

        # Discover all adapter classes that override encode()
        adapter_modules = [
            "sie_server.adapters.sentence_transformer",
            "sie_server.adapters.bge_m3",
            "sie_server.adapters.bge_m3_flag",
            "sie_server.adapters.colbert",
            "sie_server.adapters.colpali",
            "sie_server.adapters.colqwen2",
            "sie_server.adapters.nemo_colembed",
            "sie_server.adapters.clip",
            "sie_server.adapters.siglip",
            "sie_server.adapters.florence2",
            "sie_server.adapters.donut",
            "sie_server.adapters.owlv2",
            "sie_server.adapters.grounding_dino",
            "sie_server.adapters.pytorch_embedding",
        ]

        missing = []
        for mod_name in adapter_modules:
            try:
                mod = __import__(mod_name, fromlist=["__name__"])
            except ImportError:
                continue  # Skip if module can't be imported (missing deps)

            for name in dir(mod):
                obj = getattr(mod, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, ModelAdapter)
                    and obj is not ModelAdapter
                    and hasattr(obj, "encode")
                ):
                    sig = inspect.signature(obj.encode)
                    if "options" not in sig.parameters:
                        missing.append(f"{mod_name}.{name}")

        assert missing == [], f"Adapters missing 'options' param in encode(): {missing}"


class TestRuntimeOptionsConsumption:
    """Verify adapters consume runtime options from the options dict.

    These tests exercise the wiring from options -> _format_texts/_extract_texts
    and options -> _pool_embeddings/_apply_pooling, without loading models.
    """

    # --- Template override tests (PyTorchEmbeddingAdapter) ---

    def test_format_texts_uses_query_template_from_options(self) -> None:
        """query_template from options overrides the loadtime default."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter(
            "test-model",
            query_template="default_query: {text}",
            doc_template="default_doc: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        # Use options override
        texts = adapter._format_texts(
            items,
            None,
            is_query=True,
            query_template="custom_query: {text}",
        )
        assert texts == ["custom_query: hello"]

    def test_format_texts_uses_doc_template_from_options(self) -> None:
        """doc_template from options overrides the loadtime default."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter(
            "test-model",
            query_template="default_query: {text}",
            doc_template="default_doc: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        texts = adapter._format_texts(
            items,
            None,
            is_query=False,
            doc_template="custom_doc: {text}",
        )
        assert texts == ["custom_doc: hello"]

    def test_format_texts_uses_default_instruction_from_options(self) -> None:
        """default_instruction from options overrides the loadtime default."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter(
            "test-model",
            query_template="Instruct: {instruction}\nQuery: {text}",
            default_instruction="default_instr",
        )
        items: list[Item] = [Item(text="hello")]

        # Override default_instruction via options
        texts = adapter._format_texts(
            items,
            None,
            is_query=True,
            default_instruction="custom_instr",
        )
        assert texts == ["Instruct: custom_instr\nQuery: hello"]

    def test_format_texts_falls_back_to_loadtime_defaults(self) -> None:
        """Empty options dict falls back to loadtime defaults."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter(
            "test-model",
            query_template="loadtime: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        # No options override — should use loadtime template
        texts = adapter._format_texts(items, None, is_query=True)
        assert texts == ["loadtime: hello"]

    def test_format_texts_none_options_uses_loadtime_defaults(self) -> None:
        """None passed for template params falls back to loadtime defaults."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter(
            "test-model",
            query_template="loadtime: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        # Explicitly pass None — should use loadtime template
        texts = adapter._format_texts(
            items,
            None,
            is_query=True,
            query_template=None,
        )
        assert texts == ["loadtime: hello"]

    # --- Template override tests (flash adapter _extract_texts) ---

    def test_extract_texts_uses_template_from_options(self) -> None:
        """Flash adapter _extract_texts uses template params over self._* defaults."""
        from sie_server.adapters.bert_flash import BertFlashAdapter

        adapter = BertFlashAdapter(
            "test-model",
            query_template="default_query: {text}",
            doc_template="default_doc: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        texts = adapter._extract_texts(
            items,
            None,
            is_query=True,
            query_template="custom: {text}",
        )
        assert texts == ["custom: hello"]

    def test_extract_texts_falls_back_to_loadtime(self) -> None:
        """Flash adapter _extract_texts falls back to self._* when no params given."""
        from sie_server.adapters.bert_flash import BertFlashAdapter

        adapter = BertFlashAdapter(
            "test-model",
            query_template="loadtime: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        texts = adapter._extract_texts(items, None, is_query=True)
        assert texts == ["loadtime: hello"]

    # --- Pooling override tests (PyTorchEmbeddingAdapter) ---

    def test_apply_pooling_cls_override(self) -> None:
        """Pooling can be overridden to 'cls' at runtime."""
        import torch
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter("test-model", pooling="mean")
        hidden = torch.randn(2, 5, 10)
        mask = torch.ones(2, 5)

        result = adapter._apply_pooling(hidden, mask, pooling="cls")
        # CLS pooling takes position 0
        expected = hidden[:, 0]
        assert torch.equal(result, expected)

    def test_apply_pooling_mean_override(self) -> None:
        """Pooling can be overridden to 'mean' at runtime."""
        import torch
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter("test-model", pooling="cls")
        hidden = torch.randn(2, 5, 10)
        mask = torch.ones(2, 5)

        result = adapter._apply_pooling(hidden, mask, pooling="mean")
        # Mean pooling averages all tokens
        expected = hidden.mean(dim=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_apply_pooling_falls_back_to_loadtime(self) -> None:
        """Pooling falls back to loadtime config when not overridden."""
        import torch
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter("test-model", pooling="cls")
        hidden = torch.randn(2, 5, 10)
        mask = torch.ones(2, 5)

        result = adapter._apply_pooling(hidden, mask)
        expected = hidden[:, 0]
        assert torch.equal(result, expected)

    # --- Pooling safety validation ---

    def test_last_token_pooling_rejected_at_runtime_when_not_loaded(self) -> None:
        """Requesting last_token pooling at runtime when loaded with mean raises ValueError."""
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter("test-model", pooling="mean")
        # Pretend model is loaded
        adapter._model = MagicMock()
        adapter._tokenizer = MagicMock()
        adapter._device = "cpu"
        adapter._dense_dim = 10

        items: list[Item] = [Item(text="hello")]
        with pytest.raises(ValueError, match="Cannot use 'last_token' pooling at runtime"):
            adapter.encode(
                items,
                ["dense"],
                options={"pooling": "last_token"},
            )

    # --- SGLang adapter does NOT wire pooling ---

    def test_sglang_format_texts_uses_options(self) -> None:
        """SGLang adapter _format_texts accepts template overrides."""
        from sie_server.adapters.sglang import SGLangEmbeddingAdapter

        adapter = SGLangEmbeddingAdapter(
            "test-model",
            query_template="default: {text}",
        )
        items: list[Item] = [Item(text="hello")]

        texts = adapter._format_texts(
            items,
            None,
            is_query=True,
            query_template="custom: {text}",
        )
        assert texts == ["custom: hello"]

    # --- Normalize override in encode() flow ---

    def test_encode_respects_normalize_false_override(self) -> None:
        """options={"normalize": False} disables normalization even when loadtime is True."""
        import torch
        from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter

        adapter = PyTorchEmbeddingAdapter("test-model", normalize=True, pooling="cls")

        # Set up minimal mocks for encode() to run
        mock_model = MagicMock()
        hidden = torch.randn(1, 5, 10)
        mock_model.return_value = MagicMock(last_hidden_state=hidden)
        adapter._model = mock_model
        adapter._device = "cpu"
        adapter._dense_dim = 10

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(
            __getitem__=lambda self, key: torch.ones(1, 5) if key == "attention_mask" else None,
        )
        mock_tokenizer.return_value.to = MagicMock(return_value=mock_tokenizer.return_value)
        mock_tokenizer.return_value.__getitem__ = lambda self, key: torch.ones(1, 5)
        adapter._tokenizer = mock_tokenizer
        adapter._forward_kwargs = {}

        items: list[Item] = [Item(text="hello")]
        result = adapter.encode(items, ["dense"], options={"normalize": False})

        # With normalize=False, the output should NOT be unit-length
        embeddings = result.dense
        norms = np.linalg.norm(embeddings, axis=-1)
        # CLS token from random hidden state will NOT have unit norm
        assert not np.allclose(norms, 1.0, atol=0.01)


class TestRuntimeOptionsWiringRegression:
    """Structural regression tests ensuring runtime options wiring is preserved.

    These tests inspect adapter method signatures and source code to catch
    accidental removal of options wiring during refactoring. If a test here
    fails, it means someone removed or renamed a runtime-options parameter
    that adapters must accept for per-request overrides to work.
    """

    # ---- Helper ----

    @staticmethod
    def _get_adapter_class(module_name: str, class_name: str) -> type:
        """Import and return an adapter class by module and class name."""
        import importlib

        mod = importlib.import_module(f"sie_server.adapters.{module_name}")
        return getattr(mod, class_name)

    @staticmethod
    def _get_param_names(adapter_cls: type, method_name: str) -> set[str]:
        """Return the set of parameter names for a method on a class."""
        import inspect

        method = getattr(adapter_cls, method_name)
        sig = inspect.signature(method)
        return set(sig.parameters.keys())

    # ---- Group A dense: flash adapters with _extract_texts + _pool_embeddings ----

    _DENSE_FLASH_ADAPTERS: ClassVar[list[tuple[str, str]]] = [
        ("bert_flash", "BertFlashAdapter"),
        ("nomic_flash", "NomicFlashAdapter"),
        ("qwen2_flash", "Qwen2FlashAdapter"),
        ("rope_flash", "RoPEFlashAdapter"),
        ("xlm_roberta_flash", "XLMRobertaFlashAdapter"),
    ]

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _DENSE_FLASH_ADAPTERS,
        ids=[m for m, _ in _DENSE_FLASH_ADAPTERS],
    )
    def test_dense_flash_extract_texts_accepts_template_params(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """Dense flash adapters' _extract_texts must accept query_template and doc_template."""
        cls = self._get_adapter_class(module_name, class_name)
        params = self._get_param_names(cls, "_extract_texts")
        assert "query_template" in params, f"{class_name}._extract_texts missing query_template"
        assert "doc_template" in params, f"{class_name}._extract_texts missing doc_template"

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _DENSE_FLASH_ADAPTERS,
        ids=[m for m, _ in _DENSE_FLASH_ADAPTERS],
    )
    def test_dense_flash_pool_embeddings_accepts_runtime_params(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """Dense flash adapters' _pool_embeddings must accept normalize and pooling."""
        cls = self._get_adapter_class(module_name, class_name)
        params = self._get_param_names(cls, "_pool_embeddings")
        assert "normalize" in params, f"{class_name}._pool_embeddings missing normalize"
        assert "pooling" in params, f"{class_name}._pool_embeddings missing pooling"

    # ---- Group A sparse: flash adapters with _extract_texts only ----

    _SPARSE_FLASH_ADAPTERS: ClassVar[list[tuple[str, str]]] = [
        ("splade_flash", "SPLADEFlashAdapter"),
        ("gte_sparse_flash", "GTESparseFlashAdapter"),
    ]

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _SPARSE_FLASH_ADAPTERS,
        ids=[m for m, _ in _SPARSE_FLASH_ADAPTERS],
    )
    def test_sparse_flash_extract_texts_accepts_template_params(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """Sparse flash adapters' _extract_texts must accept query_template and doc_template."""
        cls = self._get_adapter_class(module_name, class_name)
        params = self._get_param_names(cls, "_extract_texts")
        assert "query_template" in params, f"{class_name}._extract_texts missing query_template"
        assert "doc_template" in params, f"{class_name}._extract_texts missing doc_template"

    # ---- Group B: PyTorch and SGLang ----

    def test_pytorch_format_texts_accepts_all_template_params(self) -> None:
        """PyTorchEmbeddingAdapter._format_texts must accept template and instruction params."""
        cls = self._get_adapter_class("pytorch_embedding", "PyTorchEmbeddingAdapter")
        params = self._get_param_names(cls, "_format_texts")
        assert "query_template" in params
        assert "doc_template" in params
        assert "default_instruction" in params

    def test_pytorch_apply_pooling_accepts_pooling_param(self) -> None:
        """PyTorchEmbeddingAdapter._apply_pooling must accept pooling param."""
        cls = self._get_adapter_class("pytorch_embedding", "PyTorchEmbeddingAdapter")
        params = self._get_param_names(cls, "_apply_pooling")
        assert "pooling" in params

    def test_sglang_format_texts_accepts_all_template_params(self) -> None:
        """SGLangEmbeddingAdapter._format_texts must accept template and instruction params."""
        cls = self._get_adapter_class("sglang", "SGLangEmbeddingAdapter")
        params = self._get_param_names(cls, "_format_texts")
        assert "query_template" in params
        assert "doc_template" in params
        assert "default_instruction" in params

    # ---- Group C: BGE-M3 variants (normalize only) ----

    def test_bge_m3_flash_compute_embeddings_accepts_normalize(self) -> None:
        """BGEM3FlashAdapter._compute_embeddings must accept normalize param."""
        cls = self._get_adapter_class("bge_m3_flash", "BGEM3FlashAdapter")
        params = self._get_param_names(cls, "_compute_embeddings")
        assert "normalize" in params

    def test_bge_m3_compute_embeddings_accepts_normalize(self) -> None:
        """BGEM3Adapter._compute_embeddings must accept normalize param."""
        cls = self._get_adapter_class("bge_m3", "BGEM3Adapter")
        params = self._get_param_names(cls, "_compute_embeddings")
        assert "normalize" in params

    # ---- Cross-cutting: encode() must resolve options dict ----

    _ALL_ENCODE_ADAPTERS: ClassVar[list[tuple[str, str]]] = [
        ("bert_flash", "BertFlashAdapter"),
        ("nomic_flash", "NomicFlashAdapter"),
        ("qwen2_flash", "Qwen2FlashAdapter"),
        ("rope_flash", "RoPEFlashAdapter"),
        ("xlm_roberta_flash", "XLMRobertaFlashAdapter"),
        ("splade_flash", "SPLADEFlashAdapter"),
        ("gte_sparse_flash", "GTESparseFlashAdapter"),
        ("pytorch_embedding", "PyTorchEmbeddingAdapter"),
        ("sglang", "SGLangEmbeddingAdapter"),
        ("bge_m3_flash", "BGEM3FlashAdapter"),
        ("bge_m3", "BGEM3Adapter"),
    ]

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _ALL_ENCODE_ADAPTERS,
        ids=[m for m, _ in _ALL_ENCODE_ADAPTERS],
    )
    def test_encode_accepts_options_parameter(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """All encoding adapters' encode() must accept an options parameter."""
        cls = self._get_adapter_class(module_name, class_name)
        params = self._get_param_names(cls, "encode")
        assert "options" in params, f"{class_name}.encode() missing 'options' parameter"

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _ALL_ENCODE_ADAPTERS,
        ids=[m for m, _ in _ALL_ENCODE_ADAPTERS],
    )
    def test_encode_resolves_options_dict(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """All encoding adapters' encode() must resolve the options dict (not ignore it)."""
        import inspect

        cls = self._get_adapter_class(module_name, class_name)
        source = inspect.getsource(cls.encode)
        assert "options or {}" in source or "options or dict()" in source, (
            f"{class_name}.encode() does not resolve options dict — "
            "expected 'options or {{}}' pattern. Runtime options will be silently ignored."
        )


class TestScoreRuntimeOptions:
    """Tests for score runtime options wiring.

    Verifies that all cross-encoder adapters accept options in score_pairs(),
    and that flash CE adapters consume max_length from options.
    """

    # ---- Helper ----

    @staticmethod
    def _get_adapter_class(module_name: str, class_name: str) -> type:
        """Import and return an adapter class by module and class name."""
        import importlib

        mod = importlib.import_module(f"sie_server.adapters.{module_name}")
        return getattr(mod, class_name)

    @staticmethod
    def _get_param_names(adapter_cls: type, method_name: str) -> set[str]:
        """Return the set of parameter names for a method on a class."""
        import inspect

        method = getattr(adapter_cls, method_name)
        return set(inspect.signature(method).parameters.keys())

    # ---- All CE adapters accept options ----

    _ALL_CE_ADAPTERS: ClassVar[list[tuple[str, str]]] = [
        ("cross_encoder", "CrossEncoderAdapter"),
        ("bert_flash_cross_encoder", "BertFlashCrossEncoderAdapter"),
        ("jina_flash_cross_encoder", "JinaFlashCrossEncoderAdapter"),
        ("modernbert_flash_cross_encoder", "ModernBertFlashCrossEncoderAdapter"),
        ("qwen2_flash_cross_encoder", "Qwen2FlashCrossEncoderAdapter"),
    ]

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _ALL_CE_ADAPTERS,
        ids=[m for m, _ in _ALL_CE_ADAPTERS],
    )
    def test_score_pairs_accepts_options(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """All cross-encoder adapters' score_pairs() must accept options kwarg."""
        cls = self._get_adapter_class(module_name, class_name)
        params = self._get_param_names(cls, "score_pairs")
        assert "options" in params, f"{class_name}.score_pairs() missing 'options' parameter"

    # ---- Flash CE adapters resolve options and consume max_length ----

    _FLASH_CE_ADAPTERS: ClassVar[list[tuple[str, str]]] = [
        ("bert_flash_cross_encoder", "BertFlashCrossEncoderAdapter"),
        ("jina_flash_cross_encoder", "JinaFlashCrossEncoderAdapter"),
        ("modernbert_flash_cross_encoder", "ModernBertFlashCrossEncoderAdapter"),
        ("qwen2_flash_cross_encoder", "Qwen2FlashCrossEncoderAdapter"),
    ]

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _FLASH_CE_ADAPTERS,
        ids=[m for m, _ in _FLASH_CE_ADAPTERS],
    )
    def test_flash_ce_score_pairs_resolves_options(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """Flash CE adapters' score_pairs() must resolve the options dict."""
        import inspect

        cls = self._get_adapter_class(module_name, class_name)
        source = inspect.getsource(cls.score_pairs)
        assert "options or {}" in source or "options or dict()" in source, (
            f"{class_name}.score_pairs() does not resolve options dict — expected 'options or {{{{}}}}' pattern."
        )

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        _FLASH_CE_ADAPTERS,
        ids=[m for m, _ in _FLASH_CE_ADAPTERS],
    )
    def test_flash_ce_consumes_max_length_from_options(
        self,
        module_name: str,
        class_name: str,
    ) -> None:
        """Flash CE adapters' score_pairs() must read max_length from options."""
        import inspect

        cls = self._get_adapter_class(module_name, class_name)
        source = inspect.getsource(cls.score_pairs)
        assert 'opts.get("max_length"' in source or "opts.get('max_length'" in source, (
            f"{class_name}.score_pairs() does not read max_length from options — "
            "per-request max_length override will not work."
        )

    # ---- Base class score()/score_pairs() accept options ----

    def test_base_class_score_accepts_options(self) -> None:
        """ModelAdapter.score() must accept options parameter."""
        from sie_server.adapters.base import ModelAdapter

        params = self._get_param_names(ModelAdapter, "score")
        assert "options" in params, "ModelAdapter.score() missing 'options' parameter"

    def test_base_class_score_pairs_accepts_options(self) -> None:
        """ModelAdapter.score_pairs() must accept options parameter."""
        from sie_server.adapters.base import ModelAdapter

        params = self._get_param_names(ModelAdapter, "score_pairs")
        assert "options" in params, "ModelAdapter.score_pairs() missing 'options' parameter"


class TestExtractRuntimeOptions:
    """Tests for extract runtime options wiring.

    Verifies that GLiNER reads merge_adjacent_entities from options, and that
    extract adapters properly wire runtime options.
    """

    def test_gliner_reads_merge_adjacent_entities_from_options(self) -> None:
        """GLiNER adapter reads merge_adjacent_entities from options dict."""
        import inspect

        from sie_server.adapters.gliner import GLiNERAdapter

        source = inspect.getsource(GLiNERAdapter.extract)
        assert 'opts.get("merge_adjacent_entities"' in source or "opts.get('merge_adjacent_entities'" in source, (
            "GLiNERAdapter.extract() does not read merge_adjacent_entities from options — "
            "NuNER_Zero runtime override will not work."
        )

    def test_gliner_reads_threshold_from_options(self) -> None:
        """GLiNER adapter reads threshold from options dict."""
        import inspect

        from sie_server.adapters.gliner import GLiNERAdapter

        source = inspect.getsource(GLiNERAdapter.extract)
        assert 'opts.get("threshold"' in source or "opts.get('threshold'" in source, (
            "GLiNERAdapter.extract() does not read threshold from options."
        )

    def test_gliner_reads_flat_ner_from_options(self) -> None:
        """GLiNER adapter reads flat_ner from options dict."""
        import inspect

        from sie_server.adapters.gliner import GLiNERAdapter

        source = inspect.getsource(GLiNERAdapter.extract)
        assert 'opts.get("flat_ner"' in source or "opts.get('flat_ner'" in source, (
            "GLiNERAdapter.extract() does not read flat_ner from options."
        )

    def test_gliner_constructor_default_merge_adjacent_false(self) -> None:
        """GLiNER merge_adjacent_entities defaults to False (safe default)."""
        from sie_server.adapters.gliner import GLiNERAdapter

        adapter = GLiNERAdapter("test-model")
        assert adapter._merge_adjacent_entities is False

    def test_gliner_constructor_accepts_merge_adjacent(self) -> None:
        """GLiNER constructor accepts merge_adjacent_entities=True."""
        from sie_server.adapters.gliner import GLiNERAdapter

        adapter = GLiNERAdapter("test-model", merge_adjacent_entities=True)
        assert adapter._merge_adjacent_entities is True

    def test_gliclass_reads_threshold_from_options(self) -> None:
        """GLiClass adapter reads threshold from options dict."""
        import inspect

        from sie_server.adapters.gliclass import GLiClassAdapter

        source = inspect.getsource(GLiClassAdapter.extract)
        assert 'opts.get("threshold"' in source or "opts.get('threshold'" in source, (
            "GLiClassAdapter.extract() does not read threshold from options."
        )

    def test_gliclass_populates_classifications_not_entities(self) -> None:
        """GLiClass adapter returns classifications in ExtractOutput, not entities."""
        from unittest.mock import MagicMock

        from sie_server.adapters.gliclass import GLiClassAdapter

        adapter = GLiClassAdapter("test-model")

        # Mock the pipeline to return known classification results
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            [{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.1}],
        ]
        adapter._pipeline = mock_pipeline
        adapter._device = "cpu"

        items = [Item(text="Great product!")]
        output = adapter.extract(items, labels=["positive", "negative"])

        # Classifications should be populated with correct data
        assert output.classifications is not None
        assert len(output.classifications) == 1
        assert output.classifications[0][0]["label"] == "positive"
        assert output.classifications[0][0]["score"] == 0.9
        assert output.classifications[0][1]["label"] == "negative"
        assert output.classifications[0][1]["score"] == 0.1

        # Entities should be empty lists (one per item)
        assert output.entities == [[]]

    def test_nli_classification_flash_populates_classifications_not_entities(self) -> None:
        """NLI Classification Flash adapter returns classifications in ExtractOutput, not entities."""
        from unittest.mock import MagicMock

        import torch
        from sie_server.adapters.nli_classification_flash import NLIClassificationFlashAdapter

        adapter = NLIClassificationFlashAdapter("test-model")

        # Mock model and tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(2, 10, dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }

        mock_model = MagicMock()
        mock_model.config.num_labels = 3
        mock_model.config.hidden_size = 768
        mock_model.config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        # Model returns logits: 1 text x 2 labels = 2 rows, 3 classes (entail/neutral/contradict)
        mock_output = MagicMock()
        mock_output.logits = torch.tensor(
            [
                [2.0, 0.5, -1.0],  # high entailment for label 1
                [-1.0, 0.5, 2.0],  # low entailment for label 2
            ]
        )
        mock_model.return_value = mock_output

        adapter._model = mock_model
        adapter._tokenizer = mock_tokenizer
        adapter._device = "cpu"
        adapter._entailment_idx = 0

        items = [Item(text="Great product!")]
        output = adapter.extract(items, labels=["positive", "negative"])

        # Classifications should be populated
        assert output.classifications is not None
        assert len(output.classifications) == 1
        assert len(output.classifications[0]) == 2
        # Should be sorted by score descending
        assert output.classifications[0][0]["label"] == "positive"
        assert output.classifications[0][0]["score"] > output.classifications[0][1]["score"]
        # Each classification should have label and score keys
        for cls in output.classifications[0]:
            assert "label" in cls
            assert "score" in cls

        # Entities should be empty lists (one per item)
        assert output.entities == [[]]

    def test_nli_classification_populates_classifications_not_entities(self) -> None:
        """NLI Classification (non-flash) adapter returns classifications in ExtractOutput, not entities."""
        from unittest.mock import MagicMock

        from sie_server.adapters.nli_classification import NLIClassificationAdapter

        adapter = NLIClassificationAdapter("test-model")

        # Mock the pipeline to return known classification results
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"labels": ["positive", "negative"], "scores": [0.9, 0.1]},
        ]
        adapter._pipeline = mock_pipeline
        adapter._device = "cpu"

        items = [Item(text="Great product!")]
        output = adapter.extract(items, labels=["positive", "negative"])

        # Classifications should be populated
        assert output.classifications is not None
        assert len(output.classifications) == 1
        assert len(output.classifications[0]) == 2
        # Should be sorted by score descending
        assert output.classifications[0][0]["label"] == "positive"
        assert output.classifications[0][0]["score"] == 0.9
        assert output.classifications[0][1]["label"] == "negative"
        assert output.classifications[0][1]["score"] == 0.1
        # Each classification should have label and score keys
        for cls in output.classifications[0]:
            assert "label" in cls
            assert "score" in cls

        # Entities should be empty lists (one per item)
        assert output.entities == [[]]

    def test_grounding_dino_reads_thresholds_from_options(self) -> None:
        """GroundingDINO adapter reads box_threshold and text_threshold from options."""
        import inspect

        from sie_server.adapters.grounding_dino import GroundingDINOAdapter

        source = inspect.getsource(GroundingDINOAdapter.extract)
        assert 'opts.get("box_threshold"' in source, (
            "GroundingDINOAdapter.extract() does not read box_threshold from options."
        )
        assert 'opts.get("text_threshold"' in source, (
            "GroundingDINOAdapter.extract() does not read text_threshold from options."
        )

    def test_owlv2_reads_score_threshold_from_options(self) -> None:
        """OWLv2 adapter reads score_threshold from options."""
        import inspect

        from sie_server.adapters.owlv2 import Owlv2Adapter

        source = inspect.getsource(Owlv2Adapter.extract)
        assert 'opts.get("score_threshold"' in source, (
            "Owlv2Adapter.extract() does not read score_threshold from options."
        )

    def test_florence2_reads_task_from_options(self) -> None:
        """Florence2 adapter reads task, max_new_tokens, num_beams from options."""
        import inspect

        from sie_server.adapters.florence2 import Florence2Adapter

        source = inspect.getsource(Florence2Adapter.extract)
        assert 'options.get("task"' in source or "options.get('task'" in source, (
            "Florence2Adapter.extract() does not read task from options."
        )
        assert 'options.get("max_new_tokens"' in source or "options.get('max_new_tokens'" in source, (
            "Florence2Adapter.extract() does not read max_new_tokens from options."
        )
        assert 'options.get("num_beams"' in source or "options.get('num_beams'" in source, (
            "Florence2Adapter.extract() does not read num_beams from options."
        )
