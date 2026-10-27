"""Host-RAM parking for SGLang engines.

Shared by the generation and embedding adapters. A parked engine keeps its
process, compiled kernels and captured CUDA graphs; its weights move to pinned
host memory and its KV cache and graph memory are released (see
``_server.park_engine``). Waking copies the weights back into the same device
addresses, so the engine serves again without a restart.

Parking needs the engine's memory saver, which is a launch flag, so an engine
started while the host tier was off (``SIE_HOST_CACHE_BYTES=0``), started for
a multi-device group, or whose profile sets ``host_offload: false`` never
parks and is unloaded on eviction as before.
"""

from __future__ import annotations

import subprocess

from sie_server.adapters.base import HostOffload
from sie_server.adapters.sglang import _server
from sie_server.core.host_cache import host_cache_configured


class SGLangParkingMixin:
    """``ModelAdapter`` parking hooks for an adapter that runs one SGLang child process."""

    _process: subprocess.Popen[bytes] | None
    _server_url: str | None
    _model_name_or_path: str
    _revision: str | None
    _tensor_parallel_size: int

    # Profile option (``adapter_options.loadtime.host_offload``); the adapter
    # constructor overrides it.
    _host_offload_requested: bool = True
    # Whether this engine was launched able to park.
    _memory_saver: bool = False
    _parked: bool = False
    # Host bytes the engine holds once it has parked; its backup buffer is kept
    # and reused after a wake.
    _host_copy_bytes: int = 0

    def _memory_saver_launch_args(self) -> list[str]:
        """Launch flags that let this engine park later; empty when it never will."""
        self._memory_saver = (
            self._host_offload_requested and self._tensor_parallel_size == 1 and host_cache_configured()
        )
        return list(_server.MEMORY_SAVER_LAUNCH_ARGS) if self._memory_saver else []

    def host_offload_bytes(self) -> int | None:
        if (
            not self._memory_saver
            or self._parked
            or self._server_url is None
            or _server.engine_exit_code(self._process) is not None
        ):
            return None
        if self._host_copy_bytes:
            return self._host_copy_bytes
        return _server.checkpoint_weight_bytes(self._model_name_or_path, self._revision)

    def offload_to_host(self) -> HostOffload:
        host_bytes = self.host_offload_bytes()
        if host_bytes is None or self._server_url is None:
            msg = f"SGLang engine for {self._model_name_or_path!r} cannot park"
            raise RuntimeError(msg)
        # Refuse new requests before the engine is drained, not after.
        self._parked = True
        _server.park_engine(self._server_url)
        self._host_copy_bytes = host_bytes
        # The process keeps its CUDA context and the workspaces allocated
        # outside the memory saver's regions.
        return HostOffload(host_bytes=host_bytes, retains_device_memory=True)

    def restore_from_host(self) -> int:
        if not self._parked or self._server_url is None:
            msg = f"SGLang engine for {self._model_name_or_path!r} is not parked"
            raise RuntimeError(msg)
        _server.raise_if_engine_exited(self._process, self._model_name_or_path)
        _server.wake_engine(self._server_url)
        self._parked = False
        return self._host_copy_bytes
