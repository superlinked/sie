"""Render a chat request's tools into the prompt as the client sent them.

SGLang's ``/v1/chat/completions`` parses ``tools`` into its own models and
hands the chat template ``tool.model_dump()`` instead of the request's JSON.
The dump lists each function's keys in model order (``description`` first),
adds ``"strict": false``, and on 0.5.20 adds a tool-level
``"defer_loading": null``. A template that writes tools with ``tojson`` or
iterates their keys puts those bytes in the prompt; one that reads
``description`` renders a missing description as ``None``. The queued worker
renders the tools as the request carries them, and so does a direct
``apply_chat_template`` call with the model's own template.

This hook gives the template the request's ``tools`` exactly as sent, all of
them, whenever SGLang renders any tools: ``tool_choice: "none"`` still renders
none, and a named ``tool_choice`` renders every tool, as the worker does. The
sent tools are used only when each one validates to the tool SGLang parsed
from the same position, so any other body keeps SGLang's own dump.

SGLang 0.5.20 also caches the render of a multimodal model's chat request
under a key that sorts every object's keys. Two requests whose tools or
tool-call arguments differ only in key order would then share one prompt, so
the hook turns that cache off.

Only the SGLang child opts into this hook. It waits for the engine's normal
import of the chat serving module.
"""

from __future__ import annotations

import builtins
import contextvars
import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from types import ModuleType
from typing import Any, cast

_SERVING_CHAT_MODULE = "sglang.srt.entrypoints.openai.serving_chat"
_CLASS_MARKER = "_sie_chat_tools_as_sent"
_IMPORT_MARKER = "_sie_chat_tools_as_sent_deferred"
_SENT_TOOLS: contextvars.ContextVar[list[Any] | None] = contextvars.ContextVar("sie_sent_tools", default=None)


def tools_as_sent(sent: Any, parsed: Any, rendered: Any) -> Any:
    """Return ``sent`` when it is the request's tools, otherwise ``rendered``.

    ``rendered`` is what SGLang would pass to the chat template and ``parsed``
    is the request's parsed ``tools``. Nothing rendered stays nothing.
    """
    if not rendered or not isinstance(sent, list) or not isinstance(parsed, list) or len(sent) != len(parsed):
        return rendered
    try:
        for raw, tool in zip(sent, cast("list[Any]", parsed), strict=True):
            if type(tool).model_validate(raw).model_dump() != tool.model_dump():
                return rendered
    except Exception:  # noqa: BLE001 - an unexpected body keeps SGLang's own tools
        return rendered
    return sent


async def _sent_tools(raw_request: Any) -> list[Any] | None:
    if raw_request is None:
        return None
    try:
        body = await raw_request.json()
    except Exception:  # noqa: BLE001 - an unreadable body keeps SGLang's own tools
        return None
    tools = body.get("tools") if isinstance(body, dict) else None
    return tools if isinstance(tools, list) else None


def _patch_serving_chat_module(module: ModuleType) -> None:
    serving_class = getattr(module, "OpenAIServingChat", None)
    if serving_class is None:
        raise RuntimeError(f"{_SERVING_CHAT_MODULE} does not expose OpenAIServingChat")
    if getattr(serving_class, _CLASS_MARKER, False):
        return
    original_handle = serving_class.handle_request
    original_render = serving_class._apply_jinja_template

    @wraps(original_handle)
    async def handle_request(self: Any, request: Any, raw_request: Any = None, *args: Any, **kwargs: Any) -> Any:
        sent = await _sent_tools(raw_request) if getattr(request, "tools", None) else None
        token = _SENT_TOOLS.set(sent)
        try:
            return await original_handle(self, request, raw_request, *args, **kwargs)
        finally:
            _SENT_TOOLS.reset(token)

    @wraps(original_render)
    def apply_jinja_template(self: Any, request: Any, tools: Any, *args: Any, **kwargs: Any) -> Any:
        tools = tools_as_sent(_SENT_TOOLS.get(), getattr(request, "tools", None), tools)
        return original_render(self, request, tools, *args, **kwargs)

    serving_class.handle_request = handle_request
    serving_class._apply_jinja_template = apply_jinja_template
    original_encode = getattr(serving_class, "_render_and_encode_chat_template", None)
    if original_encode is not None:

        @wraps(original_encode)
        def render_and_encode(self: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs["use_cache"] = False
            return original_encode(self, *args, **kwargs)

        serving_class._render_and_encode_chat_template = render_and_encode
    setattr(serving_class, _CLASS_MARKER, True)


def install_chat_tools_as_sent() -> None:
    """Patch SGLang's chat serving class now, or as soon as it is imported."""
    loaded = sys.modules.get(_SERVING_CHAT_MODULE)
    if isinstance(loaded, ModuleType) and hasattr(loaded, "OpenAIServingChat"):
        _patch_serving_chat_module(loaded)
        return

    current_import = builtins.__import__
    if getattr(current_import, _IMPORT_MARKER, False):
        return

    def deferred_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        module = current_import(name, globals, locals, fromlist, level)
        loaded_module = sys.modules.get(_SERVING_CHAT_MODULE)
        if not isinstance(loaded_module, ModuleType) or not hasattr(loaded_module, "OpenAIServingChat"):
            return module
        try:
            _patch_serving_chat_module(loaded_module)
        finally:
            if builtins.__import__ is deferred_import:
                setattr(builtins, "__import__", current_import)  # noqa: B010
        return module

    setattr(deferred_import, _IMPORT_MARKER, True)
    setattr(builtins, "__import__", deferred_import)  # noqa: B010
