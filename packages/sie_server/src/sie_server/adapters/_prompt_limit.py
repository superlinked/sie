"""Limits on the task prompt GLiNER-family adapters encode with each document.

GLiNER, GLiNER2 and GLiREL encode a request's labels, relation types, class
labels and schema fields (its task prompt) with every document. The prompt is
not billed (input tokens count the document), so it is bounded instead, as
GLiFormer and GLiNER2.5-Decide bound theirs. A request is rejected with
``InvalidInputError`` (HTTP 400 ``INVALID_INPUT``) when

* a label, relation type, class label, task name, field name or choice has
  more than ``MAX_LABEL_CHARS`` characters (checked first, before anything is
  tokenized), or
* the prompt takes more than ``max_prompt_tokens`` tokens, counted with the
  model's tokenizer.

The defaults sit far above the label sets these models are used with: the
largest label set among this repository's examples takes about 40 tokens, a
60-type PII list about 230, and a structured-extraction schema of 50 described
fields about 1,200, while ``DEFAULT_MAX_PROMPT_TOKENS`` holds several hundred
entity types and ``DEFAULT_MAX_SCHEMA_PROMPT_TOKENS`` a schema of dozens of
described fields with their choices.

Descriptions are not limited one by one; the whole prompt is tokenized only
after its characters are checked, so counting costs at most
``MAX_PROMPT_CHARS_PER_TOKEN`` characters of tokenization per allowed token.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable
from typing import Any

from sie_server.types.inputs import InvalidInputError

# Characters a label, relation type, class label, task name, field name or choice may have.
MAX_LABEL_CHARS = 128
# Tokens a request's labels, relation types and class labels may take.
DEFAULT_MAX_PROMPT_TOKENS = 1024
# Tokens a GLiNER2 request's labels or schema (field names, descriptions and choices) may take.
DEFAULT_MAX_SCHEMA_PROMPT_TOKENS = 2048
# No token of these tokenizers covers more characters than this.
MAX_PROMPT_CHARS_PER_TOKEN = 32
_PROMPT_CACHE_SIZE = 256


def validate_max_prompt_tokens(value: object) -> int:
    """A ``max_prompt_tokens`` adapter option, checked.

    Raises:
        ValueError: The value is not a positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_prompt_tokens must be a positive integer")
    return value


def check_label_chars(model: str, kind: str, values: Iterable[str]) -> None:
    """Reject a label (or relation type, class label, field name, choice) of more than ``MAX_LABEL_CHARS`` characters.

    Raises:
        InvalidInputError: A value is not a string or is too long.
    """
    for value in values:
        if not isinstance(value, str):
            raise InvalidInputError(f"{model} {kind} must be strings")
        if len(value) > MAX_LABEL_CHARS:
            raise InvalidInputError(f"{model} {kind} may have at most {MAX_LABEL_CHARS} characters each")


class PromptLimit:
    """Checks a request's task prompt against ``max_tokens``, remembering recent prompts' sizes."""

    __slots__ = ("_counts", "max_tokens", "model")

    def __init__(self, model: str, max_tokens: int) -> None:
        self.model = model
        self.max_tokens = validate_max_prompt_tokens(max_tokens)
        self._counts: OrderedDict[bytes, int] = OrderedDict()

    def check(self, texts: Iterable[str], count: Callable[[], int], key: Hashable) -> int:
        """The prompt's tokens, from ``count()``, after checking ``texts`` (its strings) and the result.

        ``key`` identifies the prompt (its strings, in order, and anything else
        its size depends on); a prompt seen recently is not counted again.

        Raises:
            InvalidInputError: The prompt takes more than ``max_tokens`` tokens.
        """
        digest = hashlib.sha256(repr(key).encode("utf-8", "surrogatepass")).digest()
        tokens = self._counts.get(digest)
        if tokens is None:
            chars = sum(len(text) for text in texts)
            if chars > self.max_tokens * MAX_PROMPT_CHARS_PER_TOKEN:
                raise InvalidInputError(self._message(None, chars))
            tokens = int(count())
            self._counts[digest] = tokens
            if len(self._counts) > _PROMPT_CACHE_SIZE:
                self._counts.popitem(last=False)
        else:
            self._counts.move_to_end(digest)
        if tokens > self.max_tokens:
            raise InvalidInputError(self._message(tokens, None))
        return tokens

    def _message(self, tokens: int | None, chars: int | None) -> str:
        size = f"{tokens} tokens" if tokens is not None else f"{chars} characters"
        return (
            f"{self.model} labels, relation types, class labels and schema fields take {size}; "
            f"a request may use at most {self.max_tokens} tokens for them"
        )


def gliner_prompt_counter(model: Any) -> Callable[[list[str], list[str]], int]:
    """Tokens of the prompt a loaded ``gliner`` model builds for entity and relation types.

    The prompt is built by the model's own processor (``prepare_inputs``, with
    no document words) and tokenized as the processor tokenizes it.
    """
    processor = model.data_processor
    tokenizer = processor.transformer_tokenizer

    def count(entity_types: list[str], relation_types: list[str]) -> int:
        kwargs = {"relations": relation_types} if relation_types else {}
        (words,), _ = processor.prepare_inputs([[]], entity_types, **kwargs)
        if not words:
            return 0
        encoding = tokenizer(list(words), is_split_into_words=True, add_special_tokens=False)
        return len(encoding["input_ids"])

    return count
