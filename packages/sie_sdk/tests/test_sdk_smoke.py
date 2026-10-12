"""Smoke tests for sie_sdk package."""

from importlib.metadata import version

import sie_sdk
from sie_sdk.client._shared import get_sdk_version
from sie_sdk.types import (
    EbnfGrammar,
    GenerateGrammar,
    GenerateImage,
    JsonSchemaGrammar,
    RegexGrammar,
)


def test_version_is_the_installed_distribution_version() -> None:
    """``__version__`` is the installed package's version, not a literal that goes stale."""
    assert sie_sdk.__version__ == version("sie-sdk")
    # The same value the client reports to the server as ``X-SIE-SDK-Version``.
    assert sie_sdk.__version__ == get_sdk_version()


def test_generate_contracts_are_importable_from_types() -> None:
    """Native generate request contracts are available from the types module."""
    assert GenerateImage is not None
    assert GenerateGrammar is not None
    assert JsonSchemaGrammar is not None
    assert RegexGrammar is not None
    assert EbnfGrammar is not None
