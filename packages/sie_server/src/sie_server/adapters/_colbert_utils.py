from __future__ import annotations

import string
from typing import Any


def punctuation_token_ids(tokenizer: Any) -> set[int]:
    """Return exact punctuation vocabulary IDs without tokenizing the strings."""
    token_ids: set[int] = set()
    for character in string.punctuation:
        token_id = tokenizer.convert_tokens_to_ids(character)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            token_ids.add(token_id)
    return token_ids
