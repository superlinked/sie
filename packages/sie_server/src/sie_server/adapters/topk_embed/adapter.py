"""Multi-vector text and page-image embeddings for the TopK-Embed-V1 family.

Serves ``topk-io/topk-embed-v1-xsmall`` (Qwen3.5-0.8B backbone, 1024-dim tokens)
and ``topk-io/topk-embed-v1-small`` (Qwen3.5-2B backbone, 2048-dim tokens). Every
token becomes a vector; a query scores a document with MaxSim.

The checkpoints ship a sentence-transformers 6 module stack whose backbone packs a
batch into one sequence and runs flash-linear-attention and compiled flex-attention
kernels (CUDA only). This adapter reproduces the same recipe on the stock
``transformers.models.qwen3_5`` classes, one right-padded row per input, so it needs
neither ``trust_remote_code`` nor sentence-transformers 6:

* query: ``"Query: " + text.strip()``, capped at ``query_length`` tokens, every
  token scored;
* text document: ``("Document: " + text).strip()``, capped at ``document_length``
  tokens, tokens in ``scoring_skip_ids`` (punctuation, special tokens) not scored;
* image document: the chat template with one image placeholder, expanded to the
  image's merged patches after ``smart_resize`` to at most ``image_token_budget``
  of them; only those patch tokens are scored.

On CUDA with flash-linear-attention installed, each batch runs packed (see
``packed.py``): one sequence, no padding, variable-length kernels, as the
reference pipeline runs, with fused kernels for the per-token work (RMSNorm with
the residual add, gated RMSNorm, SwiGLU). Otherwise every input is its own
right-padded row.

Numerics follow the reference stack (transformers 5.9): rotary tables built from
bf16 ``inv_freq``, vision rotary phases in the weight dtype, pixel normalisation in
the weight dtype, and bidirectional full-attention layers. The vision tower is run
here from the checkpoint's own sub-modules, so the result does not depend on how a
transformers release builds the vision rotary table (5.17 changed it).
"""

from __future__ import annotations

import io
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch.nn import functional as F

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._multivector import maxsim_scores_batched
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters._utils import grouped_score_pairs, validate_output_types
from sie_server.adapters.topk_embed import vision_rotary
from sie_server.adapters.topk_embed.graphs import GRAPH_MODES, GraphMode, GraphRunner, default_max_tokens
from sie_server.adapters.topk_embed.packed import PackedTextModel, Packing, resolve_kernels, to_device
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.oom import is_oom_error
from sie_server.core.postprocessor import MuveraConfig, MuveraPostprocessor
from sie_server.types.inputs import ImageInput, InvalidInputError, Item, decode_image

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from sie_server.core.inference_output import ScoreOutput

logger = logging.getLogger(__name__)

_ERR_NO_INPUT = "TopkEmbedAdapter requires text or image input per item"
_ERR_IMAGE_QUERY = "TopK-Embed queries must be text; images can only be encoded as documents"
_ERR_INSTRUCTION = "TopK-Embed applies its own 'Query: ' and 'Document: ' templates; instructions are not supported"

# Files the adapter reads; the repos also carry the sentence-transformers remote code and plots.
_CHECKPOINT_FILES = [
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "sentence_bert_config.json",
]
# TopK-specific keys in config.json that Qwen3_5Config does not own.
_TOPK_CONFIG_KEYS = (
    "dim",
    "output_dim",
    "normalize",
    "query_template",
    "document_prompt",
    "image_token_budget",
    "scoring_skip_ids",
)
_REMOTE_CODE_CONFIG_KEYS = ("architectures", "auto_map", "model_type", "transformers_version")
_IMAGE_MESSAGE = [{"role": "user", "content": [{"type": "image"}]}]
# theta of transformers 5.9's Qwen3_5VisionRotaryEmbedding, the table the checkpoints were trained on.
_VISION_ROPE_THETA = 10000.0
# Threads decoding, resizing and patchifying the pages of one request.
_PREPROCESS_WORKERS = 4
# flash-linear-attention tunes some kernels again as a batch grows: its short convolution
# for every 1,024 tokens, its RMSNorm of narrow rows for every 2,048 rows. The attention
# layers' per-head norms take one row per head, so theirs moves every 2,048 / heads tokens
# (256 for TopK-Embed's 8 query heads). The warm-up runs one batch per step.
_CONV_TUNING_TOKENS = 1024
_NORM_TUNING_ROWS = 2048


@dataclass
class _ImageRow:
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    grid_thw: tuple[int, int, int]


@dataclass
class _Launched:
    """A forward whose vectors are on their way to the host.

    On CUDA the copy runs asynchronously into pinned memory and ``copied`` marks its
    end, so the host can launch the next batch before this one is unpacked.
    """

    vectors: torch.Tensor
    copied: Any
    split: Callable[[torch.Tensor], list[torch.Tensor]]

    def rows(self) -> list[torch.Tensor]:
        """One ``[len, dim]`` host tensor per input row; waits for the copy."""
        if self.copied is not None:
            self.copied.synchronize()
        return self.split(self.vectors)


class TopkEmbedAdapter(BaseAdapter):
    """Late-interaction encoder for TopK-Embed-V1: text queries; text or page-image documents."""

    spec = AdapterSpec(
        inputs=("text", "image"),
        outputs=("multivector", "score"),
        # Discovered at load: the head width, or ``token_dim`` when set.
        multivector_dim=None,
        unload_fields=("_model", "_head", "_processor", "_tokenizer"),
        default_preprocessor="image",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "bfloat16",
        revision: str | None = None,
        max_seq_length: int | None = None,
        normalize: bool = True,
        token_dim: int | None = None,
        query_max_length: int | None = None,
        image_token_budget: int | None = None,
        text_batch_tokens: int = 16384,
        image_batch_size: int = 4,
        muvera_config: dict[str, Any] | None = None,
        attn_implementation: str | None = None,
        packed: bool | None = None,
        cuda_graphs: str | bool = "off",
        cuda_graph_tokens: int | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local checkpoint directory.
            compute_precision: Weight and activation dtype. The checkpoints are bf16.
            revision: HuggingFace revision to download (pin a commit SHA).
            max_seq_length: Upper bound on the document cap (``document_length``).
            normalize: L2-normalize each token vector (MaxSim expects unit vectors).
            token_dim: Keep only the leading ``token_dim`` dimensions of each vector
                (Matryoshka), truncated before normalizing. Defaults to the head width.
            query_max_length: Query token cap. Defaults to ``query_length`` in
                ``sentence_bert_config.json``.
            image_token_budget: Most merged patches per image. Defaults to the
                checkpoint's ``image_token_budget``.
            text_batch_tokens: Padded tokens per text forward pass.
            image_batch_size: Images per vision forward pass.
            muvera_config: Optional MUVERA configuration (passed to postprocessor).
            attn_implementation: Attention kernel for the full-attention layers of
                the row-per-input path (``sdpa`` when unset).
            packed: Run each batch as one packed sequence with variable-length
                kernels. ``None`` packs on CUDA when flash-linear-attention is
                installed; ``True`` always packs (PyTorch reference kernels where
                the fast ones are missing); ``False`` never packs.
            cuda_graphs: CUDA graphs for small text batches on the packed CUDA path
                (see ``graphs.py``): ``"off"`` or ``"bucketed"``. An operator setting,
                fixed at load.
            cuda_graph_tokens: Most tokens (rows times padded length) one graph holds.
                Defaults to 2,048 at a hidden size of 768, proportionally fewer for
                wider models.
        """
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._revision = revision
        self._max_seq_length = max_seq_length
        self._normalize = normalize
        self._token_dim_opt = token_dim
        self._query_max_length_opt = query_max_length
        self._image_token_budget_opt = image_token_budget
        self._text_batch_tokens = max(1, int(text_batch_tokens))
        self._image_batch_size = max(1, int(image_batch_size))
        self._muvera_config = muvera_config
        self._attn_implementation = attn_implementation
        self._packed = packed
        self._packed_text: PackedTextModel | None = None
        mode = "off" if cuda_graphs is False else cuda_graphs
        if mode not in GRAPH_MODES:
            msg = f"cuda_graphs must be one of {', '.join(GRAPH_MODES)}, got {cuda_graphs!r}"
            raise ValueError(msg)
        self._cuda_graphs = cast("GraphMode", mode)
        self._cuda_graph_tokens = cuda_graph_tokens
        self._graphs: GraphRunner | None = None
        self._preprocess_pool: ThreadPoolExecutor | None = None
        self._preprocess_pool_lock = threading.Lock()

        self._model: Any = None
        self._head: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None
        self._dtype: torch.dtype = torch.bfloat16
        self._multivector_dim: int | None = token_dim
        self._query_template = "Query: "
        self._document_prompt = "Document: "
        self._query_max_length = 1024
        self._doc_max_length = max_seq_length or 8192
        self._image_token_id = 0
        self._attention_heads = 1
        self._skip_ids = torch.empty(0, dtype=torch.long)
        self._image_prefix = torch.empty(0, dtype=torch.long)
        self._image_suffix = torch.empty(0, dtype=torch.long)
        self._vision_inv_freq: torch.Tensor | None = None
        self._patch_size = 16
        self._merge_size = 2
        self._temporal_patch_size = 2
        self._min_pixels = 0
        self._max_pixels = 0
        self._pixel_scale = 1.0
        self._pixel_bias = 0.0
        self._num_grid_per_side = 0
        self._grid_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # transformers' output recorder patches module forwards during each call;
        # serialize forwards so concurrent requests cannot race it.
        self._forward_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, device: str) -> None:
        """Load the backbone, projection head and processor onto ``device``."""
        # transformers >= 5.2 (the transformers5 bundle): the default bundle's 4.57 has no qwen3_5.
        from transformers import Qwen3VLProcessor
        from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5Model  # ty: ignore[unresolved-import]

        self._device = device
        self._dtype = self._resolve_compute_dtype()
        path = self._resolve_checkpoint_dir()
        raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
        topk = {key: raw.pop(key) for key in _TOPK_CONFIG_KEYS if key in raw}
        for key in _REMOTE_CODE_CONFIG_KEYS:
            raw.pop(key, None)
        config = Qwen3_5Config(**raw)
        # Bidirectional full-attention layers; the linear-attention layers stay causal recurrences.
        config.text_config.is_causal = False
        self._attention_heads = int(config.text_config.num_attention_heads)
        config.text_config.use_cache = False

        attn_impl = self._attn_implementation or "sdpa"
        logger.info(
            "Loading TopK-Embed %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            self._dtype,
            attn_impl,
        )
        model, loading = Qwen3_5Model.from_pretrained(
            path,
            config=config,
            dtype=self._dtype,
            attn_implementation=attn_impl,
            device_map=device,
            output_loading_info=True,
        )
        # The checkpoint is the reference wrapper: ``model.*`` (this backbone) plus ``head.weight``.
        unexpected = set(loading.get("unexpected_keys", [])) - {"head.weight"}
        if loading.get("missing_keys") or loading.get("mismatched_keys") or unexpected:
            msg = (
                f"TopK-Embed checkpoint does not match Qwen3_5Model: missing={loading.get('missing_keys')} "
                f"mismatched={loading.get('mismatched_keys')} unexpected={sorted(unexpected)}"
            )
            raise RuntimeError(msg)
        self._model = model.eval()
        # An unpadded batch drops the key-padding mask, and sdpa then falls back to
        # module.is_causal; clear it so that fallback is bidirectional as well.
        for layer in self._model.language_model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.is_causal = False
        # Rotary tables are rebuilt in float32 at load; the weights were trained against bf16 ones.
        for module in self._model.modules():
            inv_freq = getattr(module, "inv_freq", None)
            if isinstance(inv_freq, torch.Tensor) and inv_freq.is_floating_point():
                module.inv_freq = inv_freq.to(torch.bfloat16).to(self._dtype)
        kernels = resolve_kernels(device, mode=self._packed)
        self._packed_text = PackedTextModel(self._model.language_model, kernels) if kernels else None
        logger.info("TopK-Embed text path: %s", kernels.names if kernels else "row per input (padded)")
        vision = config.vision_config
        rotary_dim = vision.hidden_size // vision.num_heads // 2
        inv_freq = 1.0 / (_VISION_ROPE_THETA ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        self._vision_inv_freq = inv_freq.to(torch.bfloat16).to(device=device, dtype=self._dtype)
        self._num_grid_per_side = int(vision.num_position_embeddings**0.5)

        self._head = self._load_head(path).to(device=device, dtype=self._dtype).eval()
        width = self._head.out_features
        token_dim = int(self._token_dim_opt or topk.get("output_dim") or width)
        if not 0 < token_dim <= width:
            msg = f"token_dim must be in 1..{width}, got {token_dim}"
            raise ValueError(msg)
        self._multivector_dim = token_dim
        self._query_template = topk.get("query_template", self._query_template)
        self._document_prompt = topk.get("document_prompt", self._document_prompt)
        self._skip_ids = torch.tensor(topk.get("scoring_skip_ids") or [], dtype=torch.long)
        self._image_token_id = int(config.image_token_id)

        st_config = self._read_json(path / "sentence_bert_config.json")
        text_cap = int(config.text_config.max_position_embeddings)
        self._query_max_length = min(int(self._query_max_length_opt or st_config.get("query_length", 1024)), text_cap)
        doc_cap = int(st_config.get("document_length", 8192))
        if self._max_seq_length:
            doc_cap = min(doc_cap, self._max_seq_length)
        self._doc_max_length = min(doc_cap, text_cap)

        # The repo's auto_map points at its remote code; the stock processor needs none of it.
        processor = cast("Any", Qwen3VLProcessor.from_pretrained(path, trust_remote_code=False))
        self._processor = processor
        self._tokenizer = processor.tokenizer
        # Pads come after real tokens: the linear-attention layers are causal recurrences.
        self._tokenizer.padding_side = "right"
        image_processor = processor.image_processor
        self._patch_size = int(image_processor.patch_size)
        self._merge_size = int(image_processor.merge_size)
        self._temporal_patch_size = int(image_processor.temporal_patch_size)
        self._min_pixels = int(image_processor.size.shortest_edge)
        budget = int(self._image_token_budget_opt or topk.get("image_token_budget", 1280))
        self._max_pixels = budget * (self._patch_size * self._merge_size) ** 2
        self._pixel_scale = image_processor.rescale_factor / image_processor.image_std[0]
        self._pixel_bias = -image_processor.image_mean[0] / image_processor.image_std[0]
        template = self._processor.apply_chat_template(_IMAGE_MESSAGE, tokenize=False, add_generation_prompt=False)
        ids = self._tokenizer(template)["input_ids"]
        split = ids.index(self._image_token_id)
        self._image_prefix = torch.tensor(ids[:split], dtype=torch.long)
        self._image_suffix = torch.tensor(ids[split + 1 :], dtype=torch.long)
        self._grid_cache.clear()
        self._graphs = self._graph_runner(device)

    def _graph_runner(self, device: str) -> GraphRunner | None:
        """The CUDA graph runner for small text batches, when configured and possible."""
        if self._cuda_graphs == "off":
            return None
        text = self._packed_text
        if not str(device).startswith("cuda"):
            reason = "they need a CUDA device"
        elif text is None or text.kernels.names.get("delta_rule") != "fla chunk":
            reason = "they need the packed path with flash-linear-attention"
        else:
            embed = self._model.get_input_embeddings()
            max_tokens = int(self._cuda_graph_tokens or default_max_tokens(int(embed.weight.shape[1])))
            logger.info("TopK-Embed CUDA graphs: bucketed, up to %d tokens per graph", max_tokens)
            return GraphRunner(
                text,
                embed,
                pad_token_id=int(self._tokenizer.pad_token_id),
                max_tokens=max_tokens,
                name=self._model_name_or_path,
            )
        logger.info("TopK-Embed CUDA graphs off: %s", reason)
        return None

    def warmup(self) -> None:
        """Compile and tune the GPU kernels before the model takes traffic.

        flash-linear-attention's Triton kernels compile and autotune on first use, and
        some tune again as a batch grows (see ``_NORM_TUNING_ROWS``). On a fresh machine
        the first requests would stall, for seconds to over a minute, at every new batch
        size. So one packed batch per tuning step, up to the largest batch the adapter
        builds, runs here, then the CUDA graphs' padded forward at each size it tunes
        for, a query and a page (the vision tower's kernels). Off CUDA or off the packed
        path there is nothing to compile.
        """
        if self._packed_text is None or not str(self._device).startswith("cuda"):
            return
        started = time.perf_counter()
        token = int(self._tokenizer.pad_token_id)
        step = max(1, min(_CONV_TUNING_TOKENS, _NORM_TUNING_ROWS // self._attention_heads))
        largest = self._largest_batch_tokens()
        for total in range(step, largest + step, step):
            full, rest = divmod(total, self._doc_max_length)
            lengths = [self._doc_max_length] * full + ([rest] if rest else [])
            rows = [torch.full((n,), token, dtype=torch.long) for n in lengths]
            try:
                self._forward_packed(rows, images=None, normalize=True, dtype=torch.float16, graphs=False).rows()
            except Exception as exc:
                if not is_oom_error(exc):
                    raise
                # A GPU shared with other models: serving splits what does not fit, so load anyway.
                torch.cuda.empty_cache()
                logger.warning("TopK-Embed warm-up stopped at %d tokens per batch: out of GPU memory", total)
                break
        if self._graphs is not None:
            with self._forward_lock, torch.inference_mode():
                self._graphs.warm_up(_CONV_TUNING_TOKENS)
        options = {"output_dtype": "float16"}
        self.encode([Item(text="warm up")], ["multivector"], is_query=True, options=options)
        self.encode([Item(images=[_warmup_page()])], ["multivector"], options=options)
        logger.info(
            "TopK-Embed warm-up: batches up to %d tokens in steps of %d, a query and a page in %.1f s",
            largest,
            step,
            time.perf_counter() - started,
        )

    def _largest_batch_tokens(self) -> int:
        """Most tokens one packed forward holds: a text batch, one capped document, or a batch of pages."""
        page = self._max_pixels // (self._patch_size * self._merge_size) ** 2
        page += len(self._image_prefix) + len(self._image_suffix)
        return max(self._text_batch_tokens, self._doc_max_length, self._image_batch_size * page)

    def _resolve_compute_dtype(self) -> torch.dtype:
        # Honoured on every device: the reference runs bf16 on CPU too.
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        return dtype_map.get(self._compute_precision, torch.bfloat16)

    def _resolve_checkpoint_dir(self) -> Path:
        local = Path(self._model_name_or_path)
        if local.is_dir():
            return local
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(self._model_name_or_path, revision=self._revision, allow_patterns=_CHECKPOINT_FILES)
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    @staticmethod
    def _load_head(path: Path) -> torch.nn.Linear:
        from safetensors import safe_open

        shard = "model.safetensors"
        index = path / "model.safetensors.index.json"
        if index.is_file():
            shard = json.loads(index.read_text(encoding="utf-8"))["weight_map"]["head.weight"]
        with safe_open(str(path / shard), framework="pt") as handle:
            weight = handle.get_tensor("head.weight")
        head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype)
        head.load_state_dict({"weight": weight})
        return head

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: Any = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        self._check_loaded()
        validate_output_types(output_types, {"multivector"}, type(self).__name__)
        if instruction:
            raise InvalidInputError(_ERR_INSTRUCTION)
        opts = options or {}
        normalize = bool(opts.get("normalize", self._normalize))
        # Vectors cross to the host in 16 bits when the response is 16-bit anyway and nothing
        # after the adapter reads them at full precision (the MUVERA postprocessor does).
        half = opts.get("output_dtype") == "float16" and opts.get("muvera") is None
        dtype = torch.float16 if half else torch.float32

        # Items route by modality: a batch of the reference model carries one modality,
        # so text and image items run as separate forward passes. An item with images
        # is a page-image document (its text is not encoded); several images on one
        # item are concatenated into one multi-vector.
        results: list[np.ndarray | None] = [None] * len(items)
        token_counts = [0] * len(items)
        text_slots: list[int] = []
        texts: list[str] = []
        image_slots: list[tuple[int, int]] = []
        images: list[PILImage.Image] = []
        for idx, item in enumerate(items):
            if item.images:
                if is_query:
                    raise InvalidInputError(_ERR_IMAGE_QUERY)
                decoded = [decode_image(image, item_index=idx, image_index=j) for j, image in enumerate(item.images)]
                image_slots.append((idx, len(decoded)))
                images.extend(decoded)
            elif item.text is not None:
                text_slots.append(idx)
                texts.append(item.text)
            else:
                raise InvalidInputError(_ERR_NO_INPUT)

        if texts:
            vectors, lengths = self._encode_texts(texts, is_query=is_query, normalize=normalize, dtype=dtype)
            for idx, vector, length in zip(text_slots, vectors, lengths, strict=True):
                results[idx] = vector
                token_counts[idx] = length
        if images:
            per_image = self._encode_images(images, normalize=normalize, dtype=dtype)
            cursor = 0
            for idx, count in image_slots:
                segment = per_image[cursor : cursor + count]
                cursor += count
                results[idx] = segment[0] if count == 1 else np.concatenate(segment, axis=0)

        multivector = [vector for vector in results if vector is not None]
        assert len(multivector) == len(items)
        output = EncodeOutput(
            multivector=multivector,
            batch_size=len(items),
            is_query=is_query,
            multivector_token_dim=self._multivector_dim,
        )
        # Unit-meter seam (§7.3): the tokens each text item actually ran, template and
        # truncation included. Image items bill through ``images`` and count 0 here.
        if texts:
            output.extra["input_token_counts"] = token_counts
        return output

    def _encode_texts(
        self, texts: list[str], *, is_query: bool, normalize: bool, dtype: torch.dtype = torch.float32
    ) -> tuple[list[np.ndarray], list[int]]:
        if is_query:
            payloads = [self._query_template + (text or "").strip() for text in texts]
            cap = self._query_max_length
        else:
            fallback = self._tokenizer.eos_token or "."
            payloads = [(self._document_prompt + (text or "")).strip() or fallback for text in texts]
            cap = self._doc_max_length
        with self._tokenizer_guard():
            encoded = self._tokenizer(payloads, truncation=True, max_length=cap)["input_ids"]
        rows = [torch.tensor(ids, dtype=torch.long) for ids in encoded]
        keeps = [
            torch.ones_like(row, dtype=torch.bool) if is_query else ~torch.isin(row, self._skip_ids) for row in rows
        ]

        vectors: list[np.ndarray | None] = [None] * len(rows)

        def deliver(batch: list[int], launched: _Launched) -> None:
            for output, i in zip(launched.rows(), batch, strict=True):
                vectors[i] = output[keeps[i]].numpy()

        batches = [
            (batch, [rows[i] for i in batch], None) for batch in self._plan_text_batches([len(row) for row in rows])
        ]
        self._run_batches(batches, normalize=normalize, dtype=dtype, deliver=deliver)
        return [vector for vector in vectors if vector is not None], [len(row) for row in rows]

    def _plan_text_batches(self, lengths: list[int]) -> list[list[int]]:
        """Group rows by length so each batch stays within ``text_batch_tokens``.

        A padded batch costs its row count times its longest row; a packed batch
        costs the sum of its rows.
        """
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        batches: list[list[int]] = []
        current: list[int] = []
        tokens = 0
        for i in order:
            # ``order`` is ascending, so row i sets the padded width of the batch.
            cost = tokens + lengths[i] if self._packed_text is not None else (len(current) + 1) * lengths[i]
            if current and cost > self._text_batch_tokens:
                batches.append(current)
                current, tokens = [], 0
            current.append(i)
            tokens += lengths[i]
        if current:
            batches.append(current)
        return batches

    def _encode_images(
        self, images: list[PILImage.Image], *, normalize: bool, dtype: torch.dtype = torch.float32
    ) -> list[np.ndarray]:
        rows = self._image_rows(images)
        order = sorted(range(len(rows)), key=lambda i: len(rows[i].input_ids))
        vectors: list[np.ndarray | None] = [None] * len(rows)

        def deliver(batch: list[int], launched: _Launched) -> None:
            for output, i in zip(launched.rows(), batch, strict=True):
                vectors[i] = output[rows[i].input_ids == self._image_token_id].numpy()

        batches = []
        for start in range(0, len(order), self._image_batch_size):
            batch = order[start : start + self._image_batch_size]
            batches.append((batch, [rows[i].input_ids for i in batch], [rows[i] for i in batch]))
        self._run_batches(batches, normalize=normalize, dtype=dtype, deliver=deliver)
        # Page batches vary in size; hand the cached blocks back once the request is done.
        if self._device and str(self._device).startswith("cuda"):
            torch.cuda.empty_cache()
        return [vector for vector in vectors if vector is not None]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _run_batches(
        self,
        batches: list[tuple[list[int], list[torch.Tensor], list[_ImageRow] | None]],
        *,
        normalize: bool,
        dtype: torch.dtype,
        deliver: Callable[[list[int], _Launched], None],
    ) -> None:
        """Run ``(indices, rows, images)`` batches back to back.

        Each batch is launched before the previous one is unpacked, so on a GPU the host
        prepares and unpacks batches while the device computes.
        """
        pending: tuple[list[int], _Launched] | None = None
        for indices, rows, images in batches:
            launched = self._launch(rows, images=images, normalize=normalize, dtype=dtype)
            if pending is not None:
                deliver(*pending)
            pending = (indices, launched)
        if pending is not None:
            deliver(*pending)

    def _launch(
        self,
        rows: list[torch.Tensor],
        *,
        images: list[_ImageRow] | None = None,
        normalize: bool,
        dtype: torch.dtype = torch.float32,
    ) -> _Launched:
        """Run rows through the backbone and head; the vectors come back as ``dtype`` on the host."""
        assert self._model is not None
        assert self._head is not None
        try:
            if self._packed_text is not None:
                return self._forward_packed(rows, images=images, normalize=normalize, dtype=dtype)
            return self._forward_padded(rows, images=images, normalize=normalize, dtype=dtype)
        except Exception as exc:
            if self._graphs is not None and is_oom_error(exc):
                # Give the graphs' memory back before the worker's out-of-memory recovery retries.
                with self._forward_lock:
                    self._graphs.clear()
                torch.cuda.empty_cache()
            raise

    def _forward_padded(
        self, rows: list[torch.Tensor], *, images: list[_ImageRow] | None, normalize: bool, dtype: torch.dtype
    ) -> _Launched:
        device = self._device or "cpu"
        width = max(len(row) for row in rows)
        pad_id = self._tokenizer.pad_token_id
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = row
            attention_mask[i, : len(row)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with self._forward_lock, torch.inference_mode():
            inputs_embeds = self._model.get_input_embeddings()(input_ids)
            position_ids = None
            if images:
                image_hidden = self._vision_forward(images)
                image_positions = input_ids == self._image_token_id
                inputs_embeds[image_positions] = image_hidden.to(inputs_embeds.dtype)
                position_ids = self._image_position_ids(images, width).to(device)
            hidden = self._model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).last_hidden_state
            vectors = self._project(hidden, normalize=normalize, dtype=dtype).cpu()
        return _Launched(vectors, None, lambda host: [host[i, : len(row)] for i, row in enumerate(rows)])

    def _forward_packed(
        self,
        rows: list[torch.Tensor],
        *,
        images: list[_ImageRow] | None,
        normalize: bool,
        dtype: torch.dtype,
        graphs: bool = True,
    ) -> _Launched:
        packed_text = self._packed_text
        assert packed_text is not None
        device = self._device or "cpu"
        lengths = [len(row) for row in rows]
        with self._forward_lock, torch.inference_mode():
            if not images and graphs and self._graphs is not None:
                hidden = self._graphs.run(rows)
                if hidden is not None:
                    # Rows come back right-padded to the graph's length.
                    vectors = self._project(hidden, normalize=normalize, dtype=dtype)
                    return self._to_host(vectors, lambda host: [host[i, :n] for i, n in enumerate(lengths)])
            ids = torch.cat(rows)
            # Host-to-device copies do not wait for the previous batch still on the GPU.
            input_ids = to_device(ids, device).unsqueeze(0)
            packing = Packing.from_lengths(lengths, device)
            if images:
                positions = torch.cat([self._image_position_ids([row], len(row.input_ids)) for row in images], dim=2)
            else:
                # Text: every packed input restarts at position 0 on all three rotary axes.
                positions = torch.cat([torch.arange(n) for n in lengths]).view(1, 1, -1).expand(3, 1, -1).contiguous()
            embeds = self._model.get_input_embeddings()(input_ids)
            if images:
                image_hidden = self._vision_forward(images)
                # Positions found on the host: a boolean mask on the device would wait for the GPU.
                slots = to_device((ids == self._image_token_id).nonzero().squeeze(1), device)
                embeds[0].index_copy_(0, slots, image_hidden.to(embeds.dtype))
            hidden = packed_text(embeds, to_device(positions, device), packing)
            vectors = self._project(hidden, normalize=normalize, dtype=dtype)[0]
            return self._to_host(vectors, lambda host: list(host.split(lengths)))

    def _project(self, hidden: torch.Tensor, *, normalize: bool, dtype: torch.dtype) -> torch.Tensor:
        """Head, truncation to the token dim, L2 normalization in float32, then ``dtype``; on the device."""
        vectors = self._head(hidden).float()
        vectors = vectors[..., : self._multivector_dim]
        if normalize:
            vectors = F.normalize(vectors, p=2, dim=-1)
        return vectors.to(dtype)

    @staticmethod
    def _to_host(vectors: torch.Tensor, split: Callable[[torch.Tensor], list[torch.Tensor]]) -> _Launched:
        """Start copying ``vectors`` to the host: asynchronously into pinned memory on CUDA."""
        if vectors.device.type != "cuda":
            return _Launched(vectors.cpu(), None, split)
        host = torch.empty(vectors.shape, dtype=vectors.dtype, pin_memory=True)
        host.copy_(vectors, non_blocking=True)
        copied = torch.cuda.Event()
        copied.record()
        return _Launched(host, copied, split)

    def _vision_forward(self, images: list[_ImageRow]) -> torch.Tensor:
        """The reference's packed vision tower: every image in one sequence, attention kept per image."""
        visual = self._model.visual
        device = self._device or "cpu"
        weight = visual.patch_embed.proj.weight
        pixel_values = to_device(torch.cat([row.pixel_values for row in images]), device)
        # The patch-embed Conv3d has kernel == stride over pre-patched input: one matmul.
        hidden = F.linear(pixel_values.to(weight.dtype), weight.view(weight.shape[0], -1), visual.patch_embed.proj.bias)

        per_grid = [self._grid_inputs(row.grid_thw) for row in images]
        pos_index = to_device(torch.cat([grid[0] for grid in per_grid], dim=1), device)
        pos_weight = to_device(torch.cat([grid[1] for grid in per_grid], dim=1), device)
        rot_pos = to_device(torch.cat([grid[2] for grid in per_grid]), device)
        hidden = hidden + (visual.pos_embed(pos_index) * pos_weight.unsqueeze(-1)).sum(0)

        inv_freq = self._vision_inv_freq
        assert inv_freq is not None
        max_hw = max(max(row.grid_thw[1:]) for row in images)
        # transformers 5.9 Qwen3_5VisionRotaryEmbedding: phases in the (bf16) inv_freq dtype.
        freqs = (torch.arange(max_hw, device=device).unsqueeze(-1) * inv_freq).flatten(1)
        freqs = freqs[rot_pos].flatten(1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        # Attention stays within each image: images are padded to the longest and padded keys
        # masked, the reference's own sdpa call. Page vectors are sensitive to the attention
        # kernel's rounding: FlashAttention's varlen kernel, exact enough on its own, moved some
        # page tokens far from the reference on an L4 (worst cosine 0.28); this call matches it.
        lengths = torch.tensor([math.prod(row.grid_thw) for row in images], dtype=torch.long)
        max_len = int(lengths.max())
        seq_idx = torch.repeat_interleave(torch.arange(len(images)), lengths)
        within = torch.arange(int(lengths.sum())) - torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
        pad_index = to_device(seq_idx * max_len + within, device)
        key_mask = to_device((torch.arange(max_len) < lengths[:, None])[:, None, None, :].contiguous(), device)
        for block in visual.blocks:
            attended = _vision_attention(
                block.attn, block.norm1(hidden), cos, sin, pad_index=pad_index, key_mask=key_mask, max_len=max_len
            )
            hidden = hidden + attended
            hidden = hidden + block.mlp(block.norm2(hidden))
        return visual.merger(hidden)

    # ------------------------------------------------------------------
    # Image preprocessing (topk_embed_st.py in the model repo)
    # ------------------------------------------------------------------

    def _image_rows(self, images: list[PILImage.Image]) -> list[_ImageRow]:
        """Preprocess the pages of a request in parallel (decode, resize, patchify)."""
        if len(images) == 1:
            return [self._image_row(images[0])]
        with self._preprocess_pool_lock:
            if self._preprocess_pool is None:
                self._preprocess_pool = ThreadPoolExecutor(
                    max_workers=_PREPROCESS_WORKERS, thread_name_prefix="topk-embed-preprocess"
                )
            pool = self._preprocess_pool
        return list(pool.map(self._image_row, images))

    def _image_row(self, image: PILImage.Image) -> _ImageRow:
        from torchvision.transforms.v2 import InterpolationMode
        from torchvision.transforms.v2.functional import pil_to_tensor, resize

        pixels = pil_to_tensor(image.convert("RGB"))
        _, height, width = pixels.shape
        height, width = smart_resize(
            height,
            width,
            factor=self._patch_size * self._merge_size,
            min_pixels=self._min_pixels,
            max_pixels=self._max_pixels,
        )
        pixels = resize(pixels, [height, width], interpolation=InterpolationMode.BICUBIC, antialias=True)
        grid = (1, height // self._patch_size, width // self._patch_size)
        count = grid[1] * grid[2] // self._merge_size**2
        input_ids = torch.cat(
            [self._image_prefix, torch.full((count,), self._image_token_id, dtype=torch.long), self._image_suffix]
        )
        return _ImageRow(input_ids=input_ids, pixel_values=self._patchify(pixels), grid_thw=grid)

    def _patchify(self, pixels: torch.Tensor) -> torch.Tensor:
        """uint8 [C, H, W] -> normalized patches [H/p * W/p, C * T * p * p] in merge-block order."""
        channel, height, width = pixels.shape
        patch, merge, temporal = self._patch_size, self._merge_size, self._temporal_patch_size
        grid_h, grid_w = height // patch, width // patch
        patches = pixels.unsqueeze(0).expand(temporal, channel, height, width)
        patches = patches.reshape(temporal, channel, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
        patches = patches.permute(2, 5, 3, 6, 1, 0, 4, 7)
        flat = patches.reshape(grid_h * grid_w, channel * temporal * patch * patch)
        # Normalized in the weight dtype, as the reference does.
        return flat.to(self._dtype).mul_(self._pixel_scale).add_(self._pixel_bias)

    def _grid_inputs(self, thw: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bilinear position-embedding indices and weights, and rotary (row, col) per patch."""
        cached = self._grid_cache.get(thw)
        if cached is not None:
            return cached
        t, h, w = thw
        merge, side = self._merge_size, self._num_grid_per_side
        h_idxs, w_idxs = torch.linspace(0, side - 1, h), torch.linspace(0, side - 1, w)
        h_floor, w_floor = h_idxs.int(), w_idxs.int()
        h_ceil, w_ceil = (h_floor + 1).clip(max=side - 1), (w_floor + 1).clip(max=side - 1)
        dh, dw = h_idxs - h_floor, w_idxs - w_floor
        index = torch.stack(
            [
                ((h_floor * side)[:, None] + w_floor[None]).flatten(),
                ((h_floor * side)[:, None] + w_ceil[None]).flatten(),
                ((h_ceil * side)[:, None] + w_floor[None]).flatten(),
                ((h_ceil * side)[:, None] + w_ceil[None]).flatten(),
            ]
        ).to(torch.int64)
        weight = torch.stack(
            [
                ((1 - dh)[:, None] * (1 - dw)[None]).flatten(),
                ((1 - dh)[:, None] * dw[None]).flatten(),
                (dh[:, None] * (1 - dw)[None]).flatten(),
                (dh[:, None] * dw[None]).flatten(),
            ]
        )
        perm = torch.arange(t * h * w).view(t, h // merge, merge, w // merge, merge).permute(0, 1, 3, 2, 4).reshape(
            -1
        ) % (h * w)
        row = (
            (torch.arange(h // merge)[:, None, None, None] * merge + torch.arange(merge)[None, None, :, None])
            .expand(h // merge, w // merge, merge, merge)
            .reshape(-1)
        )
        col = (
            (torch.arange(w // merge)[None, :, None, None] * merge + torch.arange(merge)[None, None, None, :])
            .expand(h // merge, w // merge, merge, merge)
            .reshape(-1)
        )
        entry = (index[:, perm], weight[:, perm].to(self._dtype), torch.stack((row, col), dim=-1).repeat(t, 1))
        self._grid_cache[thw] = entry
        return entry

    def _image_position_ids(self, images: list[_ImageRow], width: int) -> torch.Tensor:
        """3D (temporal, height, width) rotary positions per row, right-padded to ``width``."""
        prefix = len(self._image_prefix)
        positions = torch.zeros((3, len(images), width), dtype=torch.long)
        for i, row in enumerate(images):
            within = torch.arange(len(row.input_ids))
            grid_h, grid_w = row.grid_thw[1] // self._merge_size, row.grid_thw[2] // self._merge_size
            count = grid_h * grid_w
            image = (within >= prefix) & (within < prefix + count)
            j = within - prefix
            # Text after the image resumes at prefix + max(grid_h, grid_w).
            sequential = within + torch.where(within >= prefix + count, max(grid_h, grid_w) - count, 0)
            positions[0, i, : len(within)] = torch.where(image, torch.full_like(within, prefix), sequential)
            positions[1, i, : len(within)] = torch.where(image, prefix + j // grid_w, sequential)
            positions[2, i, : len(within)] = torch.where(image, prefix + j % grid_w, sequential)
        return positions

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score documents (text or page images) against a text query with MaxSim."""
        self._check_loaded()
        # MaxSim runs here on the host in float32, whatever the response dtype.
        options = {key: value for key, value in (options or {}).items() if key != "output_dtype"}
        query_output = self.encode([query], ["multivector"], instruction=instruction, is_query=True, options=options)
        doc_output = self.encode(items, ["multivector"], instruction=instruction, is_query=False, options=options)
        if query_output.multivector is None or doc_output.multivector is None:
            raise RuntimeError("TopkEmbedAdapter: encode returned no multivector output")
        query_tensor = torch.from_numpy(query_output.multivector[0])
        return maxsim_scores_batched(query_tensor, [torch.from_numpy(doc) for doc in doc_output.multivector])

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        def score_with_options(query: Item, items: list[Item], *, instruction: str | None = None) -> list[float]:
            return self.score(query, items, instruction=instruction, options=options)

        return grouped_score_pairs(score_with_options, queries, docs, instruction=instruction)

    # ------------------------------------------------------------------
    # Metering and pipeline hooks
    # ------------------------------------------------------------------

    def count_input_tokens(self, items: list[Item]) -> list[int] | None:
        """Per-item text-token counts that survive a mixed text and image batch.

        Image items run the vision tower (billed through ``images``) and count 0
        text tokens, like ``qwen3_vl_embedding``.
        """
        tokenizer = self._metering_tokenizer()
        if tokenizer is None:
            return None
        positions = [i for i, item in enumerate(items) if not item.images and isinstance(item.text, str)]
        counts = [0] * len(items)
        if not positions:
            return counts
        with self._tokenizer_guard():
            measured = self._token_counts_or_none(
                tokenizer, [items[i].text or "" for i in positions], expected_len=len(positions)
            )
        if measured is None:
            return None
        for i, count in zip(positions, measured, strict=True):
            counts[i] = count
        return counts

    def _metering_max_length(self) -> int | None:
        return self._doc_max_length

    def unload(self) -> None:
        with self._preprocess_pool_lock:
            pool, self._preprocess_pool = self._preprocess_pool, None
        if pool is not None:
            pool.shutdown(wait=False)
        if self._graphs is not None:
            self._graphs.clear()
        self._graphs = None
        self._packed_text = None
        super().unload()

    def get_postprocessors(self) -> dict[str, Any]:
        """Return the configured MUVERA multivector-to-dense postprocessor."""
        config = MuveraConfig(**self._muvera_config) if self._muvera_config else MuveraConfig()
        return {"muvera": MuveraPostprocessor(token_dim=self._multivector_dim or 1024, config=config)}

    def get_preprocessor(self) -> Any:
        # The adapter owns tokenization and image preprocessing. The base
        # CharCountPreprocessor (cost-only) keeps text requests on the batched
        # ModelWorker path; image batches reach the worker through the pipeline's
        # passthrough (ColQwen3 does the same).
        return super().get_preprocessor()


def smart_resize(height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen2-VL ``smart_resize`` (transformers 5.9): sides divisible by ``factor``, area in bounds."""
    if max(height, width) / min(height, width) > 200:
        msg = f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        raise InvalidInputError(msg)
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _vision_attention(
    attn: Any,
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    pad_index: torch.Tensor,
    key_mask: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """One vision attention layer over packed patches, attending only within each image."""
    seq_length = hidden.shape[0]
    query, key, value = attn.qkv(hidden).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    query, key = _apply_rotary_pos_emb_vision(query, key, cos, sin)
    n_images = key_mask.shape[0]

    def padded(x: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros((n_images * max_len, *x.shape[1:]))
        out.index_copy_(0, pad_index, x)
        return out.view(n_images, max_len, *x.shape[1:]).transpose(1, 2)

    out = F.scaled_dot_product_attention(padded(query), padded(key), padded(value), attn_mask=key_mask)
    out = out.transpose(1, 2)
    out = out.reshape(-1, *out.shape[2:]).index_select(0, pad_index)
    return attn.proj(out.reshape(seq_length, -1))


def _apply_rotary_pos_emb_vision(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate in float32 and return the input dtype, as transformers' ``apply_rotary_pos_emb_vision`` does.

    On CUDA one fused kernel does it with bit-identical results (``vision_rotary.py``).
    """
    if vision_rotary.available(query):
        return vision_rotary.rotate(query, cos, sin), vision_rotary.rotate(key, cos, sin)
    q, k = query.float(), key.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(query.dtype), k_embed.to(key.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _warmup_page() -> ImageInput:
    """A small blank page: its size does not matter to the kernels the warm-up compiles."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "white").save(buffer, "PNG")
    return ImageInput(data=buffer.getvalue(), format="png")
