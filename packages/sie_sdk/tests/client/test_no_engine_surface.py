from __future__ import annotations

import inspect

from sie_sdk import SIEAsyncClient, SIEClient

PUBLIC_INFERENCE_METHODS = (
    "encode",
    "score",
    "extract",
    "generate",
    "responses",
    "chat_completions",
)


def test_public_inference_methods_do_not_expose_engine_parameter() -> None:
    for client_cls in (SIEClient, SIEAsyncClient):
        for method_name in PUBLIC_INFERENCE_METHODS:
            signature = inspect.signature(getattr(client_cls, method_name))
            assert "engine" not in signature.parameters, (
                f"{client_cls.__name__}.{method_name} must route alternate runtimes "
                "through model profile variants, not a public engine parameter"
            )
