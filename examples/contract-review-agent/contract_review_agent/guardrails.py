"""Input guardrail backed by ibm-granite/granite-guardian.

Before the orchestrator sees a request, granite-guardian screens it for unsafe
content / prompt-injection. The model emits a "yes" (unsafe) / "no" (safe)
verdict; we trip the guardrail on "yes". This is the Agents SDK's guardrail
hook wired to a dedicated safety model in the SIE catalog — the same "guard
content" job the landing page describes.
"""

from __future__ import annotations

import time
from typing import Any

from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, input_guardrail

from .runtime import AppContext, chat_once


def _input_text(data: Any) -> str:
    """Flatten the guardrail input (a string, or a list of input items) to text."""
    if isinstance(data, str):
        return data
    parts: list[str] = []
    for item in data or []:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                c["text"]
                for c in content
                if isinstance(c, dict) and isinstance(c.get("text"), str)
            )
    return "\n".join(parts)


@input_guardrail
async def safety_guardrail(
    ctx: RunContextWrapper[AppContext], agent: Agent, data: Any
) -> GuardrailFunctionOutput:
    app = ctx.context
    model = app.cfg["models"]["guard"]
    t0 = time.monotonic()
    try:
        res = await chat_once(
            app,
            model,
            [{"role": "user", "content": _input_text(data)[:6000]}],
            max_tokens=8,
            timeout_s=25,
        )
    except Exception as exc:  # noqa: BLE001 - configured fail-open boundary.
        # Guard model unavailable: fail OPEN (allow the run) but make it visible.
        # A stricter deployment might fail closed — that's a policy choice.
        app.ledger.record(
            "Safety guardrail (granite-guardian)",
            model,
            "generate",
            warmup_s=time.monotonic() - t0,
            got=f"unavailable: {type(exc).__name__}",
        )
        return GuardrailFunctionOutput(
            output_info={"error": str(exc), "model": model}, tripwire_triggered=False
        )
    verdict = res.text.strip()
    app.ledger.record(
        "Safety guardrail (granite-guardian)",
        model,
        "generate",
        warmup_s=res.provision_s,
        latency_s=res.gen_s,
        sent=f"{res.prompt_tokens:,} tok" if res.prompt_tokens else "—",
        got=verdict or "—",
    )
    unsafe = verdict.lower().startswith("yes")
    return GuardrailFunctionOutput(
        output_info={"verdict": verdict, "model": model},
        tripwire_triggered=unsafe,
    )
