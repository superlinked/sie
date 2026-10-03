from typing import Any

from sie_server.adapters.sglang.generation import SGLangGenerationAdapter


class SGLangCuda13Adapter(SGLangGenerationAdapter):
    """Route generic generation through the CUDA 13 SGLang bundle."""

    _PREFILL_GRAPH_OFF_FLAG = "--disable-prefill-cuda-graph"
    _MAMBA_STRATEGY_FLAG = "--mamba-radix-cache-strategy"
    # SGLang 0.5.13 never captured a prefill graph for a multimodal model, and
    # every profile routed here was sized under that engine. 0.5.20 captures
    # one by default, which needs memory those profiles do not leave free.
    _PREFILL_GRAPH_OFF_AT_EVERY_WIDTH = True


class SGLangStrictThinkingAdapter(SGLangCuda13Adapter):
    """CUDA 13 SGLang lane that guarantees a closed private thought block.

    The CUDA 13 SGLang engine can suppress premature EOS tokens until the model emits its
    reasoning terminator.  The older shared SGLang bundle does not expose this
    launch option, so reasoning profiles that need the guarantee route through
    the existing CUDA 13 bundle while retaining the generic generation adapter.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        extra_launch_args: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        launch_args = list(extra_launch_args or [])
        if "--enable-strict-thinking" not in launch_args:
            launch_args.append("--enable-strict-thinking")
        super().__init__(
            model_name_or_path,
            extra_launch_args=launch_args,
            **kwargs,
        )
