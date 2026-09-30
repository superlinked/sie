"""GLiNER checkpoints on byte-level BPE encoders (RoBERTa, Longformer).

``gliner`` passes its tokenizer pre-split words. A RoBERTa-family fast tokenizer
loaded without ``add_prefix_space=True`` refuses them, so every request to such a
checkpoint (``numind/NuNER_Zero-4k``, on Longformer) failed before this fix.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters import _word_window
from sie_server.adapters._word_window import prefix_space_for_split_words

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")


def byte_level_tokenizer(add_prefix_space: bool) -> Any:
    """A Longformer fast tokenizer with a byte-level vocabulary and no merges."""
    first, padding, last, unknown = "<s>", "<pad>", "</s>", "<unk>"
    alphabet = tokenizers.pre_tokenizers.ByteLevel.alphabet()
    vocab = {token: index for index, token in enumerate([first, padding, last, unknown, *sorted(alphabet)])}
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, [], unk_token=unknown))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
    backend.decoder = tokenizers.decoders.ByteLevel()
    return transformers.LongformerTokenizerFast(
        tokenizer_object=backend,
        bos_token=first,
        eos_token=last,
        unk_token=unknown,
        pad_token=padding,
        add_prefix_space=add_prefix_space,
    )


def test_split_words_fail_without_prefix_space() -> None:
    tokenizer = byte_level_tokenizer(add_prefix_space=False)
    with pytest.raises(AssertionError, match="add_prefix_space"):
        tokenizer(["ab", "c"], is_split_into_words=True)


def test_switches_a_byte_level_tokenizer_in_place() -> None:
    tokenizer = byte_level_tokenizer(add_prefix_space=False)
    tokenizer.add_tokens(["<<ENT>>"])
    entity_token = tokenizer.convert_tokens_to_ids("<<ENT>>")

    assert prefix_space_for_split_words(tokenizer) is True

    ids = tokenizer(["ab", "c"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
    # Every word is encoded as it would be mid-sentence, after a space ("Ġ").
    space = tokenizer.convert_tokens_to_ids("Ġ")
    assert ids == [space, *tokenizer.convert_tokens_to_ids(["a", "b"]), space, tokenizer.convert_tokens_to_ids("c")]
    # The tokens gliner added after loading survive: nothing was reloaded.
    assert tokenizer.convert_tokens_to_ids("<<ENT>>") == entity_token
    assert prefix_space_for_split_words(tokenizer) is False


def test_leaves_other_tokenizers_alone() -> None:
    ready = byte_level_tokenizer(add_prefix_space=True)
    assert prefix_space_for_split_words(ready) is False
    # DeBERTa-style tokenizers have no add_prefix_space attribute at all.
    sentencepiece_like = MagicMock(spec=["backend_tokenizer"])
    assert prefix_space_for_split_words(sentencepiece_like) is False


def test_bound_gliner_words_prepares_the_tokenizer_first() -> None:
    model = MagicMock()
    model.config.max_len = 32
    model.config.encoder_config = None
    order: list[str] = []
    with (
        patch.object(_word_window, "prefix_space_for_split_words", side_effect=lambda _t: order.append("prefix")),
        patch.object(_word_window, "split_word_counter", side_effect=lambda _t: order.append("counter")),
        patch.object(_word_window, "SubwordCounter"),
        patch.object(_word_window, "WindowedSplitter"),
    ):
        _word_window.bound_gliner_words(model)
    assert order == ["prefix", "counter"]
