"""ColBERTModernBERTFlashAdapter tokenizes a batch in one call, with each text's ids exactly as a lone call gives them."""

from __future__ import annotations

from sie_server.adapters.colbert_modernbert_flash.adapter import ColBERTModernBERTFlashAdapter
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

_WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _tokenizer() -> PreTrainedTokenizerFast:
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[Q]", "[D]"]
    vocab = {token: index for index, token in enumerate([*specials, *_WORDS, ".", ","])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))  # noqa: S106 -- a vocabulary entry, not a secret
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",  # noqa: S106
        unk_token="[UNK]",  # noqa: S106
        cls_token="[CLS]",  # noqa: S106
        sep_token="[SEP]",  # noqa: S106
    )


class _CountingTokenizer:
    def __init__(self, inner: PreTrainedTokenizerFast) -> None:
        self.inner = inner
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.inner(*args, **kwargs)


def test_one_batched_call_gives_each_text_its_lone_ids() -> None:
    texts = [
        "[D] alpha beta gamma delta epsilon zeta eta theta alpha beta",
        "[D] gamma , delta .",
        "[D] " + " ".join(_WORDS * 20),
        "[D] unknownword alpha",
    ]
    tokenizer = _tokenizer()
    adapter = ColBERTModernBERTFlashAdapter("unused")
    counting = _CountingTokenizer(tokenizer)
    adapter._tokenizer = counting  # type: ignore[assignment]

    for max_length in (4, 16, 300):
        counting.calls = 0
        batched = adapter._tokenize_inputs(texts, max_length)
        assert counting.calls == 1
        lone = [tokenizer(text, max_length=max_length, truncation=True)["input_ids"] for text in texts]
        assert batched == lone
