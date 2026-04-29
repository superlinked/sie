from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import torch

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters._utils import extract_texts, validate_output_types
from sie_server.adapters.base import ModelAdapter
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput, SparseVector
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_Arch = Literal["bert", "roberta", "distilbert"]


def _has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


class SPLADEFlashAdapter(PEFTLoRAMixin, FlashBaseAdapter):
    """SPLADE adapter with flash-attention support for packed sequences.

    On CUDA with flash-attn installed, eliminates padding waste by packing
    sequences and using flash_attn_varlen_func.

    On non-CUDA devices (or when flash-attn is unavailable), delegates to the
    model's own forward pass for bit-exact parity with the reference
    sentence-transformers pipeline.

    SPLADE produces sparse lexical representations using masked language modeling:
    - weights = log(1 + ReLU(MLM_logits))
    - max-pool over tokens to get per-term weights
    """

    fallback_adapter_path: ClassVar[str | None] = None

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("sparse",),
        unload_fields=("_model", "_tokenizer", "_use_flash", "_arch", "_idf", "_vocab_size"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_seq_length: int = 512,
        compute_precision: ComputePrecision = "float16",
        query_template: str | None = None,
        doc_template: str | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._query_template = query_template
        self._doc_template = doc_template
        self._trust_remote_code = trust_remote_code

        self._model: Any = None  # AutoModelForMaskedLM
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._device: str | None = None
        self._vocab_size: int | None = None
        self._arch: _Arch | None = None
        self._use_flash: bool = False
        self._idf: torch.Tensor | None = None

    @classmethod
    def create_for_device(cls, device: str, **kwargs: Any) -> ModelAdapter:
        if device.startswith("cuda") and _has_flash_attn():
            logger.info("SPLADE: flash-attn available, using packed flash path")
        elif device.startswith("cuda"):
            logger.info("SPLADE: flash-attn unavailable, using native model forward")
        else:
            logger.info("SPLADE: non-CUDA device '%s', using native model forward", device)
        return cls(**kwargs)

    def load(self, device: str) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._device = device

        # Use float32 on CPU/MPS for correctness; configured precision on CUDA
        dtype = self._resolve_dtype() if device.startswith("cuda") else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # Load with eager attention — the flash path runs its own attention;
        # the native path uses the model's forward() directly.
        self._model = AutoModelForMaskedLM.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

        self._arch = self._detect_arch()
        self._vocab_size = self._model.config.vocab_size
        self._use_flash = device.startswith("cuda") and _has_flash_attn()

        # Disable packed flash path for RoBERTa until position-id parity is
        # verified — fall back to native model forward which handles this
        # correctly via its own embeddings module.
        if self._arch == "roberta" and self._use_flash:
            logger.warning("Disabling packed flash path for RoBERTa until position-id parity is verified")
            self._use_flash = False

        logger.info(
            "Loaded SPLADE: arch=%s, vocab_size=%d, hidden_size=%d, path=%s",
            self._arch,
            self._vocab_size,
            self._model.config.hidden_size,
            "flash" if self._use_flash else "native",
        )
        self._idf = self._try_load_idf_vector(self._tokenizer)

        # Clamp configured max_seq_length to whatever the tokenizer/model
        # actually support to avoid OOB position embeddings on long inputs.
        self._max_seq_length = self._resolve_tokenizer_ceiling(
            self._tokenizer,
            self._model,
            self._max_seq_length,
        )

    def _detect_arch(self) -> _Arch:
        if hasattr(self._model, "bert"):
            return "bert"
        if hasattr(self._model, "roberta"):
            return "roberta"
        if hasattr(self._model, "distilbert"):
            return "distilbert"
        msg = f"Unsupported model architecture: {type(self._model).__name__}"
        raise ValueError(msg)

    def _resolve_dtype(self) -> torch.dtype:
        dtype_map: dict[ComputePrecision, torch.dtype] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        try:
            return dtype_map[self._compute_precision]
        except KeyError as exc:
            msg = f"Unsupported compute_precision={self._compute_precision!r}; expected one of {tuple(dtype_map)}"
            raise ValueError(msg) from exc

    # ------------------------------------------------------------------
    # Encode entry point
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
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        validate_output_types(output_types, {"sparse"}, "SPLADEFlashAdapter")

        opts = options or {}
        query_template = opts.get("query_template", self._query_template)
        doc_template = opts.get("doc_template", self._doc_template)

        texts = extract_texts(
            items,
            instruction,
            is_query=is_query,
            query_template=query_template,
            doc_template=doc_template,
            err_msg="SPLADEFlashAdapter requires text input",
        )

        if not texts:
            return EncodeOutput(sparse=[], batch_size=0, is_query=is_query)

        # Inference-free query encoding via IDF lookup (doc-* checkpoint pattern)
        if is_query and self._idf is not None:
            return self._encode_query_idf(texts, is_query)

        if self._use_flash:
            sparse_list = self._encode_flash(texts)
        else:
            sparse_list = self._encode_native(texts)

        return EncodeOutput(sparse=sparse_list, batch_size=len(items), is_query=is_query)

    # ------------------------------------------------------------------
    # Native path — standard model forward (CPU / MPS / CUDA-without-flash)
    # ------------------------------------------------------------------

    def _encode_native(self, texts: list[str]) -> list[SparseVector]:
        """Encode using the model's standard forward pass.

        Uses padded batches with the model's own attention implementation for
        bit-exact parity with the reference sentence-transformers pipeline.
        """
        inputs = self._tokenizer(
            texts,
            max_length=self._max_seq_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"]

        with torch.inference_mode():
            output = self._model(**inputs)
            logits = output.logits if hasattr(output, "logits") else output[0]

            # Mask padding positions before max-pooling
            weights = torch.log1p(torch.relu_(logits))
            weights = weights * attention_mask.unsqueeze(-1)

            # Max-pool over tokens per sequence
            max_weights, _ = weights.max(dim=1)  # [batch, vocab_size]

        return self._dense_to_sparse_list(max_weights)

    # ------------------------------------------------------------------
    # Flash path — packed sequences with flash_attn_varlen_func (CUDA)
    # ------------------------------------------------------------------

    def _encode_flash(self, texts: list[str]) -> list[SparseVector]:
        """Encode using packed sequences with flash attention."""
        batch_encoding = self._tokenizer(
            texts,
            max_length=self._max_seq_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        seq_lengths = [len(ids) for ids in batch_encoding["input_ids"]]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        input_ids_packed = torch.tensor(
            [tok_id for ids in batch_encoding["input_ids"] for tok_id in ids],
            dtype=torch.long,
            device=self._device,
        )

        cu_seqlens = torch.zeros(len(texts) + 1, dtype=torch.int32, device=self._device)
        cu_seqlens[1:] = torch.tensor(seq_lengths, dtype=torch.int32, device=self._device).cumsum(0)

        with torch.inference_mode():
            position_ids_packed = self._build_position_ids(cu_seqlens)
            hidden = self._run_embeddings(input_ids_packed, position_ids_packed)
            hidden = self._run_transformer_flash(hidden, cu_seqlens, max_seqlen, total_tokens)
            logits = self._run_mlm_head(hidden)
            weights = torch.log1p(torch.relu_(logits))
            sparse_list = self._aggregate_sparse(weights, cu_seqlens, seq_lengths)

        return sparse_list

    # ------------------------------------------------------------------
    # Flash path internals
    # ------------------------------------------------------------------

    def _build_position_ids(self, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Build position IDs for packed sequences.

        BERT uses 0-based positions. RoBERTa offsets by padding_idx + 1.
        """
        total_tokens = int(cu_seqlens[-1].item())
        positions = torch.arange(total_tokens, device=self._device, dtype=torch.long)
        offsets = torch.repeat_interleave(
            cu_seqlens[:-1],
            cu_seqlens[1:] - cu_seqlens[:-1],
        )
        position_ids = positions - offsets

        if self._arch == "roberta":
            padding_idx = self._get_base_model().embeddings.padding_idx
            position_ids = position_ids + padding_idx + 1

        return position_ids

    def _get_base_model(self) -> Any:
        if self._arch == "bert":
            return self._model.bert
        if self._arch == "roberta":
            return self._model.roberta
        if self._arch == "distilbert":
            return self._model.distilbert
        raise RuntimeError("Base model requested before architecture was detected")

    def _run_mlm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._arch == "distilbert":
            hidden = self._model.vocab_transform(hidden)
            hidden = self._model.activation(hidden)
            hidden = self._model.vocab_layer_norm(hidden)
            return self._model.vocab_projector(hidden)
        if self._arch == "roberta":
            return self._model.lm_head(hidden)
        # BERT
        return self._model.cls(hidden)

    def _run_embeddings(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        base_model = self._get_base_model()
        embeddings = base_model.embeddings

        word_emb = embeddings.word_embeddings(input_ids)
        pos_emb = embeddings.position_embeddings(position_ids)

        if hasattr(embeddings, "token_type_embeddings"):
            token_type_emb = embeddings.token_type_embeddings(torch.zeros_like(input_ids))
            hidden = word_emb + pos_emb + token_type_emb
        else:
            hidden = word_emb + pos_emb

        hidden = embeddings.LayerNorm(hidden)
        return embeddings.dropout(hidden)

    def _run_transformer_flash(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func

        base_model = self._get_base_model()
        num_heads = self._model.config.num_attention_heads
        hidden_size = self._model.config.hidden_size
        head_dim = hidden_size // num_heads
        softmax_scale = 1.0 / (head_dim**0.5)

        if self._arch == "distilbert":
            layers = base_model.transformer.layer
        else:
            layers = base_model.encoder.layer

        for layer in layers:
            if self._arch == "distilbert":
                attention = layer.attention
                query = attention.q_lin(hidden).view(total_tokens, num_heads, head_dim)
                key = attention.k_lin(hidden).view(total_tokens, num_heads, head_dim)
                value = attention.v_lin(hidden).view(total_tokens, num_heads, head_dim)
            else:
                attention = layer.attention.self
                query = attention.query(hidden).view(total_tokens, num_heads, head_dim)
                key = attention.key(hidden).view(total_tokens, num_heads, head_dim)
                value = attention.value(hidden).view(total_tokens, num_heads, head_dim)

            attn_out = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
                softmax_scale=softmax_scale,
            )
            attn_out = attn_out.reshape(total_tokens, hidden_size)

            if self._arch == "distilbert":
                attn_out = attention.out_lin(attn_out)
                attn_out = attention.dropout(attn_out)
                hidden = layer.sa_layer_norm(attn_out + hidden)

                inter = layer.ffn.lin1(hidden)
                inter = layer.ffn.activation(inter)
                inter = layer.ffn.dropout(inter)
                out = layer.ffn.lin2(inter)
                out = layer.ffn.dropout(out)
                hidden = layer.output_layer_norm(out + hidden)
            else:
                attn_out = layer.attention.output.dense(attn_out)
                attn_out = layer.attention.output.dropout(attn_out)
                hidden = layer.attention.output.LayerNorm(attn_out + hidden)

                inter = layer.intermediate.dense(hidden)
                inter = layer.intermediate.intermediate_act_fn(inter)
                out = layer.output.dense(inter)
                out = layer.output.dropout(out)
                hidden = layer.output.LayerNorm(out + hidden)

        return hidden

    # ------------------------------------------------------------------
    # Sparse output helpers
    # ------------------------------------------------------------------

    def _aggregate_sparse(
        self,
        weights: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
    ) -> list[SparseVector]:
        """Max-pool packed token weights into per-sequence sparse vectors."""
        num_seqs = len(seq_lengths)
        max_weights = torch.segment_reduce(weights, "max", offsets=cu_seqlens)
        dense = max_weights.cpu().float().numpy()
        results: list[SparseVector] = []
        for i in range(num_seqs):
            row = dense[i]
            mask = row > 0
            results.append(
                SparseVector(
                    indices=np.where(mask)[0].astype(np.int32),
                    values=row[mask],
                )
            )
        return results

    @staticmethod
    def _dense_to_sparse_list(max_weights: torch.Tensor) -> list[SparseVector]:
        """Convert dense [batch, vocab_size] weights to a list of SparseVector."""
        dense = max_weights.cpu().float().numpy()
        results: list[SparseVector] = []
        for i in range(dense.shape[0]):
            row = dense[i]
            mask = row > 0
            results.append(
                SparseVector(
                    indices=np.where(mask)[0].astype(np.int32),
                    values=row[mask],
                )
            )
        return results

    # ------------------------------------------------------------------
    # IDF / query-weight utilities
    # ------------------------------------------------------------------

    def _try_load_idf_vector(self, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor | None:
        from pathlib import Path

        vocab = tokenizer.get_vocab()

        for filename, loader in (
            ("query_token_weights.txt", self._parse_query_token_weights),
            ("idf.json", self._parse_idf_json),
        ):
            resolved = self._resolve_repo_file(Path(self._model_name_or_path), filename)
            if resolved is None:
                continue
            idf = loader(resolved)
            if idf is None:
                continue

            idf_vec = torch.zeros(tokenizer.vocab_size, dtype=torch.float32)
            for tok, weight in idf.items():
                tid = vocab.get(tok)
                if tid is not None:
                    idf_vec[tid] = float(weight)

            logger.info(
                "IDF loaded from %s for %s: %d non-zero entries out of %d vocab tokens",
                filename,
                self._model_name_or_path,
                int((idf_vec > 0).sum().item()),
                tokenizer.vocab_size,
            )
            return idf_vec

        logger.info("No IDF / query weights found for %s", self._model_name_or_path)
        return None

    @staticmethod
    def _resolve_repo_file(model_path: Path, filename: str) -> str | None:
        if model_path.exists() and model_path.is_dir():
            candidate = model_path / filename
            return str(candidate) if candidate.exists() else None
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(repo_id=str(model_path), filename=filename)
            if isinstance(cached, str):
                return cached
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(repo_id=str(model_path), filename=filename)
            return downloaded if isinstance(downloaded, str) else None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _parse_query_token_weights(path: str) -> dict[str, float] | None:
        try:
            weights: dict[str, float] = {}
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 2:
                        weights[parts[0]] = float(parts[1])
            return weights or None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _parse_idf_json(path: str) -> dict[str, float] | None:
        import json

        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return None

    def _encode_query_idf(self, texts: list[str], is_query: bool) -> EncodeOutput:
        self._check_loaded()
        if self._tokenizer is None or self._idf is None:
            raise RuntimeError(ERR_NOT_LOADED)

        batch_encoding = self._tokenizer(
            texts,
            max_length=self._max_seq_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        idf = self._idf

        sparse_list: list[SparseVector] = []
        for input_ids in batch_encoding["input_ids"]:
            unique_ids = torch.tensor(sorted(set(input_ids)), dtype=torch.long)
            values = idf[unique_ids]
            keep = values > 0
            sparse_list.append(
                SparseVector(
                    indices=unique_ids[keep].numpy().astype(np.int32),
                    values=values[keep].numpy().astype(np.float32),
                )
            )

        return EncodeOutput(sparse=sparse_list, batch_size=len(texts), is_query=is_query)
