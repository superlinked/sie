"""A tokenizer config in the transformers-5 format loads through the server's loader.

The remote bundle pins transformers 5 so that a worker can count a hybrid
model's generation with such a tokenizer. This runs only where transformers 5
is installed, as on the image the chart runs the remote lane on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import transformers
from sie_server.core.tokenizer import load_tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers

pytestmark = pytest.mark.skipif(
    int(transformers.__version__.split(".", 1)[0]) < 5,
    reason="the transformers-5 tokenizer format needs transformers 5, which the remote bundle installs",
)


def test_a_transformers5_format_tokenizer_loads_and_counts_a_rendered_chat(tmp_path: Path) -> None:
    vocab = {"[UNK]": 0, "<|user|>": 1, "<|assistant|>": 2, "hello": 3, "world": 4}
    tokenizer = Tokenizer(models.WordLevel(vocab, "[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<|user|>", "<|assistant|>"])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"backend": "tokenizers", "tokenizer_class": "TokenizersBackend", "unk_token": "[UNK]"}),
        encoding="utf-8",
    )
    (tmp_path / "chat_template.jinja").write_text(
        "{% for message in messages %}<|{{ message.role }}|>{{ message.content }}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}",
        encoding="utf-8",
    )

    loaded = load_tokenizer(tmp_path)
    rendered = loaded.apply_chat_template(
        [{"role": "user", "content": "hello world"}], tokenize=False, add_generation_prompt=True
    )

    assert type(loaded).__name__ == "TokenizersBackend"
    assert rendered == "<|user|>hello world<|assistant|>"
    assert loaded.encode(rendered, add_special_tokens=False) == [1, 3, 4, 2]
