"""Smoke tests for sie_sdk package."""

import sie_sdk
from sie_sdk.types import (
    EbnfGrammar,
    GenerateGrammar,
    GenerateImage,
    JsonSchemaGrammar,
    RegexGrammar,
)


def test_import() -> None:
    """Verify package can be imported."""
    assert sie_sdk.__version__ == "0.1.0"


def test_generate_contracts_are_importable_from_types() -> None:
    """Native generate request contracts are available from the types module."""
    assert GenerateImage is not None
    assert GenerateGrammar is not None
    assert JsonSchemaGrammar is not None
    assert RegexGrammar is not None
    assert EbnfGrammar is not None
