import os

from sie_server.adapters.tensorrt_llm.compat import install_rc24_compatibility

if os.environ.get("SIE_TRTLLM_RC24_COMPAT") == "1":
    try:
        install_rc24_compatibility()
    except Exception as error:
        # CPython deliberately catches ordinary exceptions raised by
        # sitecustomize and continues interpreter startup. Convert every
        # compatibility failure into an uncaught BaseException so the engine
        # can never start after a missing, stale, or partially applied patch.
        raise SystemExit(f"TensorRT-LLM rc24 compatibility startup failed: {error}") from error
