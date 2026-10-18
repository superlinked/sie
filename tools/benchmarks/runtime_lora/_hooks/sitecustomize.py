"""Chain SIE's production child hooks before the benchmark-only observer."""

import importlib

importlib.import_module("sie_server.adapters.sglang._compat.sitecustomize")
importlib.import_module("tools.benchmarks.runtime_lora.observer").install()
