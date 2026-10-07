"""The numerical profile snapshot a server process reports for numerical admission.

For every model config the registry holds, the snapshot reports the local
execution identity of the ``default`` profile and the model contract. For a
model whose remote profile this process serves, it also reports that profile's
contract and its current numerical admission.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from sie_server.config.equivalence import model_contract_digest, remote_profile_contract_digest
from sie_server.config.hybrid_admission import remote_admission
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import installed_upstreams
from sie_server.core.profile_identity import local_profile_identity, runtime_instance_id, serving_code_digest
from sie_server.ipc_types import (
    NumericalAdmissionObservation,
    NumericalProfileObservation,
    NumericalProfileSnapshotResponse,
)

if TYPE_CHECKING:
    from sie_server.core.registry import ModelRegistry

logger = logging.getLogger(__name__)

_MAX_NUMERICAL_PROFILES = 1024
_MAX_ADMITTED_IDENTITIES = 8
_MAX_NUMERICAL_MODEL_ID_BYTES = 1024


async def numerical_profile_snapshot(registry: ModelRegistry) -> NumericalProfileSnapshotResponse:
    """The registry's numerical profile snapshot, taken under its execution lease.

    A failure answers an incomplete snapshot with no profiles, and the log names
    only the class of the error.
    """
    try:
        async with registry.execution_lease():
            return await asyncio.to_thread(_snapshot, registry)
    except Exception as exc:  # noqa: BLE001
        error_class = (
            "io" if isinstance(exc, OSError) else "invalid" if isinstance(exc, TypeError | ValueError) else "internal"
        )
        logger.debug("Could not collect numerical profile snapshot (error_class=%s)", error_class)
        return NumericalProfileSnapshotResponse(runtime_instance_id=runtime_instance_id())


def _snapshot(registry: ModelRegistry) -> NumericalProfileSnapshotResponse:
    configs = registry.get_configs_snapshot()
    complete = len(configs) <= _MAX_NUMERICAL_PROFILES
    observations = []
    for name in sorted(configs)[:_MAX_NUMERICAL_PROFILES]:
        if not name or len(name.encode()) > _MAX_NUMERICAL_MODEL_ID_BYTES:
            complete = False
            continue
        config = configs[name]
        identity = None
        contract = None
        if isinstance(config, ModelConfig):
            identity = local_profile_identity(
                config,
                "default",
                device=registry.profile_execution_device(name) or "",
                engine_config=registry.engine_config,
            )
            contract = model_contract_digest(config)
        else:
            complete = False
        observation = NumericalProfileObservation(name, identity, contract)
        if isinstance(config, ModelConfig) and config.synthetic_profile_variant_source is None:
            _observe_remote_profile(observation, config)
        observations.append(observation)
    return NumericalProfileSnapshotResponse(runtime_instance_id(), observations, complete)


def _observe_remote_profile(observation: NumericalProfileObservation, config: ModelConfig) -> None:
    """Report this process's contract and admission for a model's remote profile, if it serves one."""
    routing = config.routing
    profile = routing.fallback_profile if routing is not None and routing.fallback_profile else "default"
    remote_contract = remote_profile_contract_digest(config, profile, installed_upstreams())
    if remote_contract is None:
        return
    observation.remote_contract_sha256 = remote_contract
    observation.remote_execution_sha256 = serving_code_digest()
    if config.tasks.encode is None and config.tasks.score is None:
        return
    admission = remote_admission(config, wait=False)
    if (
        isinstance(admission, str)
        or not admission.outputs
        or len(admission.local_identities) > _MAX_ADMITTED_IDENTITIES
    ):
        return
    observation.admission = NumericalAdmissionObservation(
        sha256=admission.sha256,
        kind=admission.kind,
        local_identities=sorted(admission.local_identities),
        model_contract_sha256=admission.model_contract_sha256,
        outputs=sorted(admission.outputs),
        expires_at_unix_ms=int(admission.expires_at.timestamp() * 1000),
    )
