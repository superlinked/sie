from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, cast

from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.adapters._utils import grouped_score_pairs
from sie_server.adapters.base import HostOffload, ModelAdapter, ModelCapabilities, ModelDims, released_bytes

if TYPE_CHECKING:
    import torch

    from sie_server.core.inference_output import ScoreOutput
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)


class BaseAdapter(ModelAdapter):
    """Concrete base with common defaults.

    Provides:
    - ``capabilities`` / ``dims`` properties derived from ``spec``.
    - Standard ``unload()`` driven by ``spec.unload_fields``.
    - Default ``get_preprocessor()`` returning ``CharCountPreprocessor``.
    - ``_resolve_dtype()`` mapping ``compute_precision`` string to dtype.
    - ``_check_loaded()`` guard for encode/score/extract entry points.

    Every concrete subclass must declare a class-level ``spec``.
    """

    spec: ClassVar[AdapterSpec]

    # Host-RAM parking (``ModelAdapter.offload_to_host``) for in-process
    # weights. The generic implementation below moves every tensor reachable
    # from the adapter's attributes (see ``_host_offload``). An adapter that
    # keeps device state that walk cannot see sets this to False.
    host_offload: ClassVar[bool] = True

    # Guards the lazy creation of each instance's ``_tokenizer_lock`` (below).
    # A HuggingFace *fast* tokenizer wraps a Rust object behind a ``RefCell``:
    # calling it from two threads at once — or mutating shared state like
    # ``pad_token_id`` mid-call — raises ``RuntimeError("Already borrowed")``.
    # On the managed worker path the single-threaded inference executor runs an
    # adapter's ``encode()`` (which tokenizes) while a *different* asyncio
    # thread-pool thread runs the §7.3 metering fallback
    # (``count_input_tokens`` → ``_token_counts_or_none``) for another
    # concurrent request against the *same* tokenizer object. Both sides take
    # ``_tokenizer_guard()`` so tokenizer access is serialised per adapter
    # instance and can never overlap. See issue #1800.
    _tokenizer_lock_init: ClassVar[threading.Lock] = threading.Lock()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Only validate classes that declare their own spec
        if "spec" not in cls.__dict__:
            return

        spec = cls.spec
        if not isinstance(spec, AdapterSpec):
            msg = f"{cls.__name__}.spec must be an AdapterSpec instance"
            raise TypeError(msg)

        if not spec.inputs:
            msg = f"{cls.__name__}.spec.inputs must be non-empty"
            raise TypeError(msg)

        if not spec.outputs:
            msg = f"{cls.__name__}.spec.outputs must be non-empty"
            raise TypeError(msg)

        # Validate output -> method consistency
        encode_outputs = {"dense", "sparse", "multivector"}
        declared_encode = encode_outputs & set(spec.outputs)
        if declared_encode and cls.encode is ModelAdapter.encode:
            msg = f"{cls.__name__} declares {declared_encode} in outputs but does not implement encode()"
            raise TypeError(msg)

        if "score" in spec.outputs:
            # BaseAdapter ships a default score_pairs() that delegates to score().
            # Treat that default as "not implemented" for validation purposes:
            # subclasses must override either score() or score_pairs() so the
            # default delegate doesn't bottom out in ModelAdapter.score().
            score_overridden = cls.score is not ModelAdapter.score
            score_pairs_overridden = cls.score_pairs not in (
                ModelAdapter.score_pairs,
                BaseAdapter.score_pairs,
            )
            if not score_overridden and not score_pairs_overridden:
                msg = f"{cls.__name__} declares 'score' in outputs but does not implement score() or score_pairs()"
                raise TypeError(msg)

        if "json" in spec.outputs and cls.extract is ModelAdapter.extract:
            msg = f"{cls.__name__} declares 'json' in outputs but does not implement extract()"
            raise TypeError(msg)

    # -- Properties derived from spec ----------------------------------------

    @property
    def capabilities(self) -> ModelCapabilities:
        # spec stores Literal tuples; cast needed because list() widens type.
        return ModelCapabilities(
            inputs=cast("Any", list(self.spec.inputs)),
            outputs=cast("Any", list(self.spec.outputs)),
        )

    @property
    def dims(self) -> ModelDims:
        return ModelDims(
            dense=self.spec.dense_dim or getattr(self, "_dense_dim", None),
            sparse=self.spec.sparse_dim or getattr(self, "_sparse_dim", None),
            multivector=self.spec.multivector_dim or getattr(self, "_multivector_dim", None),
        )

    # -- Standard lifecycle --------------------------------------------------

    def unload(self) -> None:
        """Unload model weights and free device memory.

        Iterates ``spec.unload_fields`` and sets each to ``None``, then
        runs ``gc.collect()`` and clears the device cache.
        """
        device = getattr(self, "_device", None)

        for attr in self.spec.unload_fields:
            if hasattr(self, attr):
                setattr(self, attr, None)

        self._device = None
        self._parked_tensors = None

        gc.collect()

        if device is not None:
            import torch as _torch

            if str(device).startswith("cuda"):
                _torch.cuda.empty_cache()
            elif str(device) == "mps":
                _torch.mps.empty_cache()

    # -- Host-RAM parking ------------------------------------------------------

    def _host_offload_device(self) -> torch.device | None:
        """The CUDA device this adapter's weights live on, or None if parking would free nothing.

        On CPU the weights already are in host RAM, and on Apple silicon the
        accelerator shares it.
        """
        device = getattr(self, "_device", None)
        if device is None:
            return None
        import torch as _torch

        try:
            resolved = _torch.device(device)
        except (RuntimeError, TypeError, ValueError):
            return None
        return resolved if resolved.type == "cuda" else None

    def host_offload_bytes(self) -> int | None:
        """Bytes of device tensors the generic walk would park, or None when it cannot park."""
        if not self.host_offload or getattr(self, "_parked_tensors", None) is not None:
            return None
        device = self._host_offload_device()
        if device is None:
            return None
        if any(not callable(getattr(holder, "clear", None)) for holder in self._graph_holders()):
            # A graph holder that cannot be cleared would replay against
            # addresses the park frees.
            return None
        from sie_server.adapters import _host_offload

        parkable = _host_offload.parkable_bytes(self, device)
        # The model ``memory_footprint`` measures must be among what the walk
        # reached; a shortfall means weights sit where parking cannot see them
        # and would stay on the device. The slack covers parameters that share
        # storage, which the footprint counts once each.
        if parkable is None or parkable < 0.9 * released_bytes(self.memory_footprint()):
            return None
        return parkable

    def offload_to_host(self) -> HostOffload:
        """Copy the weights to host RAM and free their device memory."""
        device = self._host_offload_device()
        if device is None or getattr(self, "_parked_tensors", None) is not None:
            msg = f"{type(self).__name__} has nothing on a CUDA device to park"
            raise RuntimeError(msg)
        from sie_server.adapters import _host_offload

        self._drop_device_caches()
        parked = _host_offload.park(self, device)
        self._parked_tensors = parked
        gc.collect()
        _host_offload.release_cached_device_memory(device)
        return HostOffload(host_bytes=parked.nbytes)

    def restore_from_host(self) -> int:
        """Copy parked weights back to the device they came from."""
        parked = getattr(self, "_parked_tensors", None)
        if parked is None:
            msg = f"{type(self).__name__} is not parked"
            raise RuntimeError(msg)
        from sie_server.adapters import _host_offload

        _host_offload.unpark(parked)
        self._parked_tensors = None
        return 0

    def _drop_device_caches(self) -> None:
        """Drop device state recorded against the current weight addresses.

        A CUDA graph replays kernels against the exact addresses it recorded,
        and parking moves every weight. Graph runners (``self._graphs``) are
        cleared, and anything else the adapter releases as optional memory
        goes with them; graphs record again on demand after the model wakes.
        """
        for holder in self._graph_holders():
            holder.clear()
        if self.has_releasable_memory():
            self.release_optional_memory()

    def _graph_holders(self) -> list[Any]:
        """Attributes holding recorded CUDA graphs: runners (``_graphs``) and dicts of them.

        Found by name, as every graph runner in this package is held. Scalars
        that merely describe graphs (``_cuda_graphs = "off"``) are skipped.
        """
        return [
            value
            for name, value in vars(self).items()
            if "graph" in name.lower() and value is not None and not isinstance(value, (bool, int, float, str))
        ]

    def get_preprocessor(self) -> Any:
        """Return ``CharCountPreprocessor`` for cost estimation."""
        from sie_server.core.preprocessor import CharCountPreprocessor

        return CharCountPreprocessor(
            model_name=getattr(self, "_model_name_or_path", ""),
        )

    # -- Default batched scoring ---------------------------------------------

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        """Default ``score_pairs()`` that batches via per-query grouping.

        Groups parallel ``(query, doc)`` pairs by ``(text, id, instruction)``
        so each unique query is encoded once and its docs are scored as a
        single ``score()`` call. Subclasses with a more efficient native
        cross-batch path (e.g. cross-encoders that pack queries and docs
        into one transformer pass) should override this.

        Per-call ``options`` are not supported by this default delegate
        (it dispatches per-query and cannot route options into ``score()``
        without subclass-specific knowledge). If ``options`` is a non-empty
        mapping, this raises ``NotImplementedError`` to surface the
        unsupported configuration; pass ``options=None`` (or ``{}``) or
        override ``score_pairs()`` with an options-aware implementation.
        """
        if options:
            msg = (
                f"{type(self).__name__}.score_pairs(): per-call options are "
                f"not supported by the default batching path "
                f"(got options={options!r}). Override score_pairs() with an "
                f"options-aware implementation."
            )
            raise NotImplementedError(msg)
        return grouped_score_pairs(self.score, queries, docs, instruction=instruction)

    # -- Tokenizer concurrency guard -----------------------------------------

    @contextmanager
    def _tokenizer_guard(self) -> Iterator[None]:
        """Serialise access to ``self._tokenizer`` across threads.

        HuggingFace fast tokenizers are not re-entrant: concurrent calls (or a
        ``pad_token_id`` swap mid-call) raise ``RuntimeError("Already
        borrowed")``. The encode/score paths tokenize on the inference-executor
        thread while the metering fallback tokenizes on an asyncio thread-pool
        thread for a *different* concurrent request; wrapping both in this lock
        prevents the collision. The lock is created lazily (adapters set
        ``self._tokenizer`` in their own ``__init__``/``load`` and don't all
        call ``super().__init__``) under a class-level lock so two threads
        cannot race to build it. See issue #1800.
        """
        lock = getattr(self, "_tokenizer_lock", None)
        if lock is None:
            with BaseAdapter._tokenizer_lock_init:
                lock = getattr(self, "_tokenizer_lock", None)
                if lock is None:
                    lock = threading.Lock()
                    self._tokenizer_lock = lock
        with lock:
            yield

    # -- Unit-meter token counting (§7.3) ------------------------------------

    def _metering_tokenizer(self) -> Any | None:
        """Return this adapter's in-process HF tokenizer for authoritative
        token counting, or ``None`` when it has none (server-backed adapters
        like SGLang, image/audio adapters).

        Every in-tree text adapter stores its tokenizer as ``self._tokenizer``;
        image-text adapters (CLIP/SigLIP) instead keep it inside their
        ``self._processor`` (a ``CLIPProcessor`` / ``SiglipProcessor``), which
        bundles the image processor and the text tokenizer. Falling back to
        ``processor.tokenizer`` here lets those adapters surface real §7.3 text
        counts through the shared hook — otherwise CLIP/SigLIP TEXT bills
        nothing, since its internal tokenization bypasses the pipeline's
        preprocessor counts. Adapters that keep the tokenizer elsewhere
        override this hook rather than re-implementing the counting logic.
        """
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        processor = getattr(self, "_processor", None)
        if processor is not None:
            return getattr(processor, "tokenizer", None)
        return None

    def _metering_max_length(self) -> int | None:
        """Resolve the truncation cap for token counting from the adapter's
        configured max sequence length (``_max_seq_length`` / ``_max_length``).

        ``None`` means "no cap" — count the full tokenizer length.
        """
        for attr in ("_max_seq_length", "_max_length"):
            value = getattr(self, attr, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        # Processor-tokenizer adapters (CLIP/SigLIP) declare no
        # ``_max_seq_length``; cap at the tokenizer's own ``model_max_length``
        # so a long text counts exactly where the forward pass truncates it
        # (CLIP 77, SigLIP 64) rather than over-billing beyond the context
        # window. Consulted ONLY when the adapter has no dedicated
        # ``_tokenizer`` (it does have a ``_processor``), so every existing
        # ``_tokenizer``-based emitter keeps its exact behaviour.
        if getattr(self, "_tokenizer", None) is None and getattr(self, "_processor", None) is not None:
            tokenizer = self._metering_tokenizer()
            model_max = getattr(tokenizer, "model_max_length", None)
            if isinstance(model_max, int) and not isinstance(model_max, bool) and 0 < model_max <= 100_000:
                return model_max
        return None

    def count_input_tokens(self, items: list[Item]) -> list[int] | None:
        """Real per-item input-token counts via this adapter's own tokenizer.

        Tokenizes each item's text with the same subword vocabulary the forward
        pass uses and returns the per-item ``len(input_ids)`` (special tokens
        included) — the identical ground-truth basis the encode preprocessor
        records for pipeline-tokenized adapters (§P3.5). Best-effort: returns
        ``None`` (no count, never an estimate presented as a count) when
        there is no in-process tokenizer, any item is non-text, or the tokenizer
        raises. Never raises — metering must not fail inference.
        """
        tokenizer = self._metering_tokenizer()
        if tokenizer is None:
            return None
        texts: list[str] = []
        for item in items:
            text = getattr(item, "text", None)
            if not isinstance(text, str):
                return None
            texts.append(text)
        if not texts:
            return []
        # Serialise against the encode/score tokenization running on the
        # inference-executor thread — a bare fast tokenizer raises ``Already
        # borrowed`` when re-tokenized concurrently (#1800).
        with self._tokenizer_guard():
            return self._token_counts_or_none(tokenizer, texts, expected_len=len(items))

    def count_pair_input_tokens(
        self,
        query: Item,
        docs: list[Item],
        *,
        instruction: str | None = None,
    ) -> list[int] | None:
        """Real per-pair input-token counts for reranker metering (§7.3).

        Tokenizes each ``(query, doc)`` pair jointly (query and doc packed into
        one sequence with the model's separators, mirroring the cross-encoder
        forward pass), applying ``instruction`` to the query the same way the
        adapter does. Best-effort with the same ``None`` contract as
        :meth:`count_input_tokens`.
        """
        tokenizer = self._metering_tokenizer()
        if tokenizer is None:
            return None
        query_text = getattr(query, "text", None)
        if not isinstance(query_text, str):
            return None
        if instruction is not None:
            query_text = f"{instruction} {query_text}"
        doc_texts: list[str] = []
        for doc in docs:
            doc_text = getattr(doc, "text", None)
            if not isinstance(doc_text, str):
                return None
            doc_texts.append(doc_text)
        if not doc_texts:
            return []
        # Serialise against the encode/score tokenization on the inference
        # thread (#1800), same as ``count_input_tokens``.
        with self._tokenizer_guard():
            return self._token_counts_or_none(
                tokenizer,
                [query_text] * len(doc_texts),
                doc_texts,
                expected_len=len(docs),
            )

    def _token_counts_or_none(
        self,
        tokenizer: Any,
        text_or_pairs: list[str],
        text_pair: list[str] | None = None,
        *,
        expected_len: int,
    ) -> list[int] | None:
        """Run ``tokenizer`` over texts (or joint text/text_pair) and return the
        per-entry ``len(input_ids)``; ``None`` on any quirk so a malformed count
        is never mis-attributed.
        """
        # NOTE: callers that already hold ``_tokenizer_guard()`` (e.g. CLIP /
        # SigLIP ``_encode_texts`` serialise the whole tokenize-then-count block
        # under their own ``_tokenizer_lock``) invoke this helper directly, so it
        # must NOT re-acquire the guard — the lock is non-reentrant and would
        # deadlock. Serialisation for the standalone metering path is applied at
        # the ``count_input_tokens`` / ``count_pair_input_tokens`` entry points
        # instead. See issue #1800.
        max_length = self._metering_max_length()
        try:
            if text_pair is not None:
                encoded = tokenizer(text_or_pairs, text_pair, truncation=max_length is not None, max_length=max_length)
            else:
                encoded = tokenizer(text_or_pairs, truncation=max_length is not None, max_length=max_length)
            counts = [len(ids) for ids in encoded["input_ids"]]
        except Exception:  # noqa: BLE001 — metering must never fail inference
            return None
        if len(counts) != expected_len:
            return None
        if not all(isinstance(count, int) and not isinstance(count, bool) for count in counts):
            return None
        return counts

    # -- Shared helpers ------------------------------------------------------

    def _check_loaded(self) -> None:
        """Raise ``RuntimeError`` if the model is not loaded."""
        if getattr(self, "_model", None) is None:
            raise RuntimeError(ERR_NOT_LOADED)

    def _resolve_dtype(self) -> torch.dtype:
        """Map ``self._compute_precision`` to a ``torch.dtype``."""
        import torch as _torch

        dtype_map: dict[str, torch.dtype] = {
            "float16": _torch.float16,
            "bfloat16": _torch.bfloat16,
            "float32": _torch.float32,
        }
        return dtype_map.get(
            getattr(self, "_compute_precision", "float16"),
            _torch.float16,
        )
