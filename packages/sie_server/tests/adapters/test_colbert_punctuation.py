from typing import Any

from sie_server.adapters._colbert_utils import punctuation_token_ids


class _TokenizerStub:
    unk_token_id = 3

    def __init__(self, token_ids: dict[str, Any]) -> None:
        self._token_ids = token_ids

    def convert_tokens_to_ids(self, token: str) -> Any:
        return self._token_ids.get(token)

    def encode(self, _token: str, *, add_special_tokens: bool) -> list[int]:
        raise AssertionError("punctuation IDs must be resolved by exact vocabulary lookup")


def test_punctuation_skiplist_uses_exact_ids_without_encoded_boundary_tokens() -> None:
    tokenizer = _TokenizerStub({",": 4, ".": 5})

    token_ids = punctuation_token_ids(tokenizer)

    assert token_ids == {4, 5}
    assert 6 not in token_ids  # XLM-R's incidental standalone word-boundary token.


def test_punctuation_skiplist_excludes_unknown_and_invalid_ids() -> None:
    tokenizer = _TokenizerStub(
        {
            "!": _TokenizerStub.unk_token_id,
            '"': None,
            "#": [7, 8],
            ".": 5,
        },
    )

    token_ids = punctuation_token_ids(tokenizer)

    assert token_ids == {5}
