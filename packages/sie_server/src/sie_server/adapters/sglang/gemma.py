from sie_server.adapters.sglang.generation import SGLangGenerationAdapter


class SGLangGemmaAdapter(SGLangGenerationAdapter):
    """Bundle-routing seam for the Gemma worker image.

    Behaviour is identical to :class:`SGLangGenerationAdapter` — Gemma 4 is
    served through the same ``sglang serve`` subprocess with the ``gemma4``
    reasoning/tool parsers set via the model YAML. The only reason this
    subclass exists is routing: bundle compatibility keys on the adapter
    *module* path, and the ``sglang`` bundle declares only
    ``sie_server.adapters.sglang.generation``. Giving Gemma 4 a distinct
    module lets the ``sglang-cu130`` bundle own it without making every
    Qwen3.x model compatible with that bundle too — so the Qwen serving stack
    on the ``sglang`` bundle stays untouched. The launch flags are spelled for
    that bundle's engine.
    """

    _PREFILL_GRAPH_OFF_FLAG = "--disable-prefill-cuda-graph"
    _MAMBA_STRATEGY_FLAG = "--mamba-radix-cache-strategy"
    # Gemma 4 is multimodal: SGLang 0.5.13 never captured its prefill graph,
    # and the profiles routed here were sized under that engine.
    _PREFILL_GRAPH_OFF_AT_EVERY_WIDTH = True
