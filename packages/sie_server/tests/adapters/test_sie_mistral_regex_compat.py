"""The fix_mistral_regex tokenizer backport for Transformers 5.3 (SGLang bundles)."""

from typing import Any

import pytest
from sie_server.adapters.sglang_vision_extract._compat import sie_mistral_regex_compat as compat


def _backend() -> type:
    calls: list[dict[str, Any]] = []

    class Backend:
        """Mirrors Transformers 5.3's TokenizersBackend: the flag is passed twice."""

        seen = calls

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_kwargs = dict(kwargs)
            self._tokenizer = object()
            self._tokenizer = self._patch_mistral_regex(
                self._tokenizer,
                self.init_kwargs.get("name_or_path", None),
                init_kwargs=self.init_kwargs,
                fix_mistral_regex=kwargs.get("fix_mistral_regex"),
                **kwargs,
            )

        @classmethod
        def _patch_mistral_regex(
            cls,
            tokenizer: Any,
            pretrained_model_name_or_path: Any,
            init_kwargs: dict[str, Any] | None = None,
            fix_mistral_regex: Any = None,
            **kwargs: Any,
        ) -> Any:
            cls.seen.append({"init_kwargs": dict(init_kwargs or {}), "fix_mistral_regex": fix_mistral_regex, **kwargs})
            return tokenizer

    return Backend


def test_unpatched_backend_rejects_the_flag() -> None:
    with pytest.raises(TypeError, match="multiple values for keyword argument 'fix_mistral_regex'"):
        _backend()(name_or_path="m", fix_mistral_regex=True)


def test_patch_passes_the_flag_once_with_the_fixed_release_arguments() -> None:
    backend = _backend()
    assert compat.install_on(backend) is True
    assert compat.install_on(backend) is False
    tokenizer = backend(name_or_path="m", fix_mistral_regex=True)
    assert tokenizer.init_kwargs == {"name_or_path": "m", "fix_mistral_regex": True}
    (call,) = backend.seen
    assert call["fix_mistral_regex"] is True
    assert call["init_kwargs"] == {"name_or_path": "m", "fix_mistral_regex": True}
    assert call["name_or_path"] == "m"


def test_patch_leaves_configs_without_the_flag_unchanged() -> None:
    backend = _backend()
    compat.install_on(backend)
    tokenizer = backend(name_or_path="m")
    assert tokenizer.init_kwargs == {"name_or_path": "m"}
    assert backend.seen == [{"init_kwargs": {"name_or_path": "m"}, "fix_mistral_regex": None, "name_or_path": "m"}]


def test_fixed_backend_is_not_patched() -> None:
    class Fixed:
        def __init__(self, **kwargs: Any) -> None:
            self.flag = kwargs.pop("fix_mistral_regex", None)

        @classmethod
        def _patch_mistral_regex(cls, tokenizer: Any, *args: Any, **kwargs: Any) -> Any:
            return tokenizer

    init = Fixed.__init__
    assert compat.install_on(Fixed) is False
    assert Fixed.__init__ is init
