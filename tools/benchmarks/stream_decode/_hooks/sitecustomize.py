"""Preserve production child compatibility hooks before installing observation."""

import importlib

importlib.import_module("sie_server.adapters.sglang._compat.sitecustomize")
importlib.import_module("tools.benchmarks.stream_decode.observer").install()
