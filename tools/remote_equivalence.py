#!/usr/bin/env python3
"""Measure a configured remote profile against two local SIE runs.

Run from the locked public workspace. Credential arguments name environment
variables; no credential, request body or vector is written to the evidence.
The local model must expose a verified profile identity. The record covers
every process that reports that identity. Hybrid routing remains disabled while
the explicit remote profile is evaluated.

With --cluster the URL is a cluster gateway. Every observation then comes from
the processes that cluster status lists for the model, and the probe refuses
unless they all report one identity and stay the same throughout the run.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from numbers import Integral
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from sie_sdk import SIEClient
from sie_sdk.client.errors import RequestError, SIEError
from sie_sdk.types import DEFAULT_OUTPUT_DTYPE, Item
from sie_server.config.equivalence import (
    EquivalenceRecord,
    ProbeCase,
    canonical_digest,
    measure_values,
    model_contract_digest,
    remote_profile_contract_digest,
    upstream_contract_digest,
)
from sie_server.config.model import ModelConfig, is_immutable_revision, is_remote_adapter_path
from sie_server.config.upstreams import load_upstreams, validate_upstream_url
from sie_server.core.profile_identity import _execution_code
from sie_server.core.tokenizer import load_tokenizer

_MAX_CONFIG_BYTES = 1 << 20
_MAX_CONTEXT = 32_768
_MIN_CONTEXT = 2
_HASH_LENGTH = 64
_MATRIX_NDIM = 2
_PREFIX = "Represent this text for retrieval:"
_SCORE_QUERY = "relevant documents for search"
_IDENTITY = re.compile(r"v[12]:sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class _Case:
    category: str
    texts: tuple[str, ...]
    token_counts: tuple[int, ...]
    instruction: str | None = None
    is_query: bool = False


def _cases(tokenizer: Any, context: int, *, default_instruction: str | None = None) -> list[_Case]:
    def count(text: str, instruction: str | None = None) -> int:
        prefix = default_instruction if instruction is None else instruction
        if prefix is not None:
            text = f"{prefix} {text}"
        return len(tokenizer.encode(text, add_special_tokens=True))

    # Binary-search a deterministic synthetic text. Use the actual pinned
    # tokenizer so the suite crosses the model's real truncation boundary.
    def boundary_text(repetitions: int) -> str:
        return "HEAD astronomy. " + "retrieval medicine law programming history finance " * repetitions + "TAIL ocean."

    low, high = 1, context * 4
    if count(boundary_text(low)) > context or count(boundary_text(high)) <= context:
        raise ValueError("could not construct a truncation-boundary probe")
    while low + 1 < high:
        middle = (low + high) // 2
        if count(boundary_text(middle)) <= context:
            low = middle
        else:
            high = middle
    before, after = boundary_text(low), boundary_text(high)
    long_text = boundary_text(max(1, low // 2))
    short = "How does a search engine find relevant documents?"
    requests = [
        ("short", (short,), None, False),
        ("long", (long_text,), None, False),
        ("boundary_before", (before,), None, False),
        ("boundary_after", (after,), None, False),
        ("query_prefix", (short,), _PREFIX, True),
        ("query_default", (short,), None, True),
        ("empty_prefix", (short,), "", False),
        ("document_prefix", (f"{_PREFIX} {short}",), None, False),
        (
            "score_scale",
            (
                "Search retrieves documents related to a query.",
                "The oven temperature is two hundred degrees.",
                "Documents are ranked by relevance to a search query.",
                "A search engine finds relevant documents.",
            ),
            None,
            False,
        ),
    ]
    result = [
        _Case(category, texts, tuple(count(text, instruction) for text in texts), instruction, is_query)
        for category, texts, instruction, is_query in requests
    ]
    if result[2].token_counts[0] > context or result[3].token_counts[0] <= context:
        raise ValueError("tokenizer cannot establish the truncation boundary")
    return result


def _sparse_values(results: list[Any], dimension: int) -> list[np.ndarray]:
    maps: list[dict[int, float]] = []
    for result in results:
        sparse = result[0]["sparse"]
        indices, values = sparse["indices"], sparse["values"]
        if any(not isinstance(value, float | np.floating) or not np.isfinite(value) for value in values):
            raise ValueError("sparse output must contain finite floating values")
        if (
            np.asarray(indices).ndim != 1
            or np.asarray(values).ndim != 1
            or len(indices) != len(values)
            or len(set(indices)) != len(indices)
            or any(
                not isinstance(index, Integral) or isinstance(index, bool | np.bool_) or not 0 <= index < dimension
                for index in indices
            )
        ):
            raise ValueError("sparse output is malformed")
        maps.append(dict(zip(indices, values, strict=True)))
    keys = sorted(set().union(*(mapping.keys() for mapping in maps)))
    # Empty sparse vectors still carry a measurable zero-valued observation.
    return [np.asarray([mapping.get(key, 0.0) for key in keys] or [0.0]) for mapping in maps]


def _values(
    results: list[Any], output: str, *, dimension: int | None = None, context_length: int = 0
) -> list[np.ndarray]:
    if output == "sparse":
        if dimension is None:
            raise ValueError("sparse dimension is unidentified")
        return _sparse_values(results, dimension)
    if output == "score":
        observations = []
        ids: list[str] | None = None
        for result in results:
            scores = sorted(result["scores"], key=lambda value: value["item_id"])
            observed_ids = [value["item_id"] for value in scores]
            if len(set(observed_ids)) != len(observed_ids) or (ids is not None and observed_ids != ids):
                raise ValueError("score output identities differ")
            ids = observed_ids
            if any(not isinstance(value["score"], float | np.floating) for value in scores):
                raise ValueError("score output must contain floating values")
            observations.append(np.asarray([value["score"] for value in scores]))
        return observations
    arrays = [np.asarray(result[0][output]) for result in results]
    if dimension is None or any(
        (
            array.shape != (dimension,)
            if output == "dense"
            else array.ndim != _MATRIX_NDIM or array.shape[1] != dimension or not 0 < array.shape[0] <= context_length
        )
        for array in arrays
    ):
        raise ValueError("probe output does not match the declared dimensions")
    return arrays


@dataclass(frozen=True)
class _Provenance:
    identity: str
    revision: str
    remote_execution: str
    local_instance: str | None
    processes: tuple[tuple[str, ...], ...] = ()


def _hex_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == _HASH_LENGTH and all(char in "0123456789abcdef" for char in value)


def _server_provenance(local: SIEClient, config: ModelConfig, profile: str, expected_remote: str) -> _Provenance:
    metadata = local.get_model(config.sie_id)
    profiles = metadata.get("profiles") or {}
    identity = profiles.get("default", {}).get("identity")
    local_instance = profiles.get("default", {}).get("runtime_instance_id")
    if not _hex_digest(local_instance):
        raise ValueError("probe requires a direct worker runtime instance")
    if profiles.get(profile, {}).get("remote_contract_sha256") != expected_remote:
        raise ValueError("serving endpoint/model contract differs from the supplied files")
    remote_execution = profiles.get(profile, {}).get("remote_execution_sha256")
    if not _hex_digest(remote_execution):
        raise ValueError("serving remote execution cannot be identified")
    if not isinstance(identity, str) or not _IDENTITY.fullmatch(identity):
        raise ValueError("local execution cannot be identified; no equivalence record can authorize it")
    if (
        metadata.get("revision") != config.hf_revision
        or metadata.get("max_sequence_length") != config.max_sequence_length
    ):
        raise ValueError("local model metadata differs from the supplied model contract")
    return _Provenance(
        identity, cast("str", config.hf_revision), cast("str", remote_execution), cast("str", local_instance)
    )


def _cluster_workers(client: SIEClient) -> list[Mapping[str, Any]]:
    for message in client.watch(mode="cluster"):
        workers = message.get("workers") if isinstance(message, dict) else None
        if isinstance(workers, list) and all(isinstance(worker, dict) for worker in workers):
            return cast("list[Mapping[str, Any]]", workers)
        break
    raise ValueError("cluster status is unavailable")


def _model_observations(worker: Mapping[str, Any], model: str) -> list[tuple[object, object, Mapping[str, Any]]]:
    inventory = worker.get("numerical_process_inventory") or {}
    if not isinstance(inventory, dict):
        raise ValueError("cluster status is malformed")
    children = inventory.get("children") or []
    if not isinstance(children, list):
        raise ValueError("cluster status is malformed")
    found = []
    for child in children:
        if not isinstance(child, dict):
            raise ValueError("cluster status is malformed")
        snapshot = child.get("snapshot") or {}
        if not isinstance(snapshot, dict):
            raise ValueError("cluster status is malformed")
        profiles = snapshot.get("profiles") or []
        if not isinstance(profiles, list) or not all(isinstance(observation, dict) for observation in profiles):
            raise ValueError("cluster status is malformed")
        found.extend(
            (child.get("status"), snapshot.get("runtime_instance_id"), observation)
            for observation in profiles
            if observation.get("model_id") == model
        )
    return found


def _wake_lanes(
    local: SIEClient, remote: SIEClient, config: ModelConfig, profile: str, machine_profile: str | None
) -> None:
    """Send one local and one remote call that wait for capacity, so cold lanes start before provenance."""
    encode_outputs = sorted(set(config.outputs) & {"dense", "sparse", "multivector"})
    items: list[Item] = [{"id": "probe-wake", "text": "wake"}]
    for client, selected in ((local, "default"), (remote, profile)):
        common: dict[str, Any] = {
            "options": {**config.resolve_profile(selected).runtime, "profile": selected},
            "wait_for_capacity": True,
            "max_oom_retries": 0,
        }
        if machine_profile is not None and selected == "default":
            common["gpu"] = machine_profile
        if encode_outputs:
            client.encode(
                config.sie_id,
                items,
                output_types=cast("Any", encode_outputs),
                output_dtype=DEFAULT_OUTPUT_DTYPE,
                **common,
            )
        else:
            client.score(config.sie_id, {"text": _SCORE_QUERY}, items, **common)


def _cluster_provenance(
    client: SIEClient, config: ModelConfig, expected_remote: str, machine_profile: str | None
) -> _Provenance:
    """Attribute every probe observation to processes listed by cluster status.

    The local processes are those of non-remote workers (of one machine profile
    when given) that list the model, and they must all report one identity and
    the supplied model contract. The remote processes are those of remote
    workers that report the model's remote contract, and they must agree.
    """
    model, contract = config.sie_id, model_contract_digest(config)
    local: dict[str, object] = {}
    remote: dict[str, tuple[object, object]] = {}
    for worker in _cluster_workers(client):
        observations = _model_observations(worker, model)
        if worker.get("bundle") == "remote":
            for status, instance, observation in observations:
                if observation.get("remote_contract_sha256") is None:
                    continue
                if status != "observed" or not _hex_digest(instance):
                    raise ValueError("remote process provenance is incomplete")
                remote[cast("str", instance)] = (
                    observation.get("remote_contract_sha256"),
                    observation.get("remote_execution_sha256"),
                )
            continue
        if machine_profile is not None and worker.get("gpu") != machine_profile:
            continue
        if not observations and model in (worker.get("loaded_models") or []):
            raise ValueError("a worker serving the model reports no process provenance")
        for status, instance, observation in observations:
            if (
                status != "observed"
                or not _hex_digest(instance)
                or observation.get("model_contract_sha256") != contract
            ):
                raise ValueError("local process provenance is incomplete or differs from the supplied model")
            local[cast("str", instance)] = observation.get("local_identity")
    identities = set(local.values())
    if len(identities) != 1:
        raise ValueError("local processes do not report exactly one execution identity")
    (identity,) = identities
    if not isinstance(identity, str) or not _IDENTITY.fullmatch(identity):
        raise ValueError("local execution cannot be identified; no equivalence record can authorize it")
    contracts = set(remote.values())
    if len(contracts) != 1:
        raise ValueError("remote processes do not report exactly one remote contract")
    ((remote_contract, remote_execution),) = contracts
    if remote_contract != expected_remote:
        raise ValueError("serving endpoint/model contract differs from the supplied files")
    if not _hex_digest(remote_execution):
        raise ValueError("serving remote execution cannot be identified")
    processes = (
        *sorted((instance, cast("str", value)) for instance, value in local.items()),
        *sorted((instance, *cast("tuple[str, str]", value)) for instance, value in remote.items()),
    )
    return _Provenance(identity, cast("str", config.hf_revision), cast("str", remote_execution), None, processes)


def _request(
    client: SIEClient,
    config: ModelConfig,
    case: _Case,
    operation: str,
    outputs: set[str],
    *,
    profile: str,
    upstream: str,
    local_instance: str | None,
    machine_profile: str | None = None,
) -> tuple[str, Any]:
    common: dict[str, Any] = {
        "options": {**config.resolve_profile(profile).runtime, "profile": profile},
        "instruction": case.instruction,
        "wait_for_capacity": local_instance is None,
        "provision_timeout_s": None if local_instance is None else 60.0,
        "max_oom_retries": 0,
    }
    if machine_profile is not None and profile == "default":
        common["gpu"] = machine_profile
    try:
        items: list[Item] = [{"id": f"probe-{index}", "text": text} for index, text in enumerate(case.texts)]
        if operation == "score":
            result = client.score(config.sie_id, {"text": _SCORE_QUERY}, items, **common)
            if {value["item_id"] for value in result["scores"]} != {value["id"] for value in items} or len(
                result["scores"]
            ) != len(items):
                raise ValueError("probe score items are missing or repeated")
            rows = [result]
        else:
            result = client.encode(
                config.sie_id,
                items,
                output_types=cast("Any", sorted(outputs)),
                output_dtype=DEFAULT_OUTPUT_DTYPE,
                is_query=case.is_query,
                **common,
            )
            rows = cast("list[Any]", result)
        if not isinstance(rows, list) or len(rows) != (1 if operation == "score" else len(items)):
            raise ValueError("probe result count differs")
        for row in rows:
            evidence = row.get("request") or {}
            expected = "local" if profile == "default" else "remote"
            if evidence.get("served_by") != expected or (expected == "remote" and evidence.get("upstream") != upstream):
                raise ValueError("probe serving provenance is missing or differs")
            if local_instance is not None and evidence.get("runtime_instance_id") != local_instance:
                raise ValueError("probe serving runtime instance differs")
        return "ok", result
    except RequestError as error:
        if local_instance is not None and (error.request or {}).get("runtime_instance_id") != local_instance:
            raise ValueError("probe refusal runtime instance differs") from None
        code = (error.code or "").upper()
        if error.status_code == HTTPStatus.BAD_REQUEST and code in ("INVALID_INPUT", "INPUT_TOO_LONG"):
            return "input_too_long" if code == "INPUT_TOO_LONG" else "invalid_input", None
        raise ValueError("probe request failed without a comparable input refusal") from None


def run_probe(
    config: ModelConfig,
    upstreams: Any,
    local: SIEClient,
    remote: SIEClient,
    profile: str,
    *,
    cluster: bool = False,
    machine_profile: str | None = None,
) -> EquivalenceRecord:
    if config.weights_path is not None or not is_immutable_revision(config.hf_revision) or not config.hf_id:
        raise ValueError("probe requires immutable local Hub weights")
    if config.max_sequence_length is None or not _MIN_CONTEXT <= config.max_sequence_length <= _MAX_CONTEXT:
        raise ValueError("probe context is unsupported")
    resolved = config.resolve_profile(profile)
    if profile == "default" or not is_remote_adapter_path(resolved.adapter_path):
        raise ValueError("probe requires an explicit remote profile")
    if canonical_digest(dict(resolved.runtime)) != canonical_digest(dict(config.resolve_profile("default").runtime)):
        raise ValueError("remote profile runtime differs from the local runtime it is measured against")
    default_runtime = config.resolve_profile("default").runtime
    if default_runtime.get("output_dtype", DEFAULT_OUTPUT_DTYPE) != DEFAULT_OUTPUT_DTYPE:
        raise ValueError("probe requires the default float32 output dtype")
    upstream_name, upstream_model = resolved.loadtime["upstream"], resolved.loadtime["upstream_model"]
    upstream = upstreams[upstream_name]
    expected_remote = remote_profile_contract_digest(config, profile, upstreams)
    if expected_remote is None:
        raise ValueError("serving endpoint/model contract differs from the supplied files")

    def provenance() -> _Provenance:
        if cluster:
            return _cluster_provenance(local, config, expected_remote, machine_profile)
        return _server_provenance(local, config, profile, expected_remote)

    if cluster:
        _wake_lanes(local, remote, config, profile, machine_profile)
    before = provenance()
    outputs = set(config.outputs) & {"dense", "sparse", "multivector", "score"}
    if not outputs:
        raise ValueError("probe requires encode or score outputs")
    tokenizer = load_tokenizer(config.hf_id, revision=config.hf_revision, trust_remote_code=False)
    cases = []
    for operation in ("encode", "score"):
        selected_outputs = {output for output in outputs if (output == "score") == (operation == "score")}
        if not selected_outputs:
            continue
        for case in _cases(
            tokenizer, config.max_sequence_length, default_instruction=default_runtime.get("instruction")
        ):
            if operation == "encode" and case.category == "score_scale":
                continue
            observations = [
                _request(
                    client,
                    config,
                    case,
                    operation,
                    selected_outputs,
                    profile=selected,
                    upstream=upstream_name,
                    local_instance=before.local_instance,
                    machine_profile=machine_profile,
                )
                for client, selected in ((local, "default"), (local, "default"), (remote, profile))
            ]
            outcomes = tuple(observation[0] for observation in observations)
            measurements = {}
            if outcomes == ("ok", "ok", "ok"):
                results = [observation[1] for observation in observations]
                try:
                    measurements = {
                        output: measure_values(
                            *_values(
                                results,
                                output,
                                dimension=getattr(getattr(config.tasks.encode, output, None), "dim", None),
                                context_length=config.max_sequence_length,
                            )
                        )
                        for output in sorted(selected_outputs)
                    }
                except (KeyError, TypeError, ValueError):
                    outcomes = ("shape_mismatch",) * 3
            cases.append(
                ProbeCase(
                    operation=cast("Any", operation),
                    category=cast("Any", case.category),
                    input_sha256=canonical_digest(
                        {
                            "texts": case.texts,
                            "instruction": case.instruction,
                            "is_query": case.is_query,
                            "query": _SCORE_QUERY if operation == "score" else None,
                        }
                    ),
                    token_counts=case.token_counts,
                    outcomes=cast("Any", outcomes),
                    measurements=cast("Any", measurements),
                )
            )
    if provenance() != before:
        raise ValueError("local execution changed during the probe")
    return EquivalenceRecord(
        measured_at=datetime.now(UTC),
        upstream_name=upstream_name,
        upstream_model=upstream_model,
        upstream_contract_sha256=upstream_contract_digest(upstream),
        model_contract_sha256=model_contract_digest(config),
        probe_sources_sha256=canonical_digest(
            {"cli": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "library": _execution_code()["sources"]}
        ),
        remote_contract_sha256=expected_remote,
        remote_execution_sha256=before.remote_execution,
        local_observation_sha256=canonical_digest({"identity": before.identity, "revision": before.revision}),
        runtime_options_sha256=canonical_digest(dict(config.resolve_profile("default").runtime)),
        output_dtype="float32",
        local_identity=before.identity,
        model=config.sie_id,
        remote_profile=profile,
        context_length=config.max_sequence_length,
        outputs=cast("Any", frozenset(outputs)),
        cases=tuple(cases),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--upstreams-file", type=Path, required=True)
    parser.add_argument("--local-url", required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--remote-profile", default="remote")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster", action="store_true", help="--local-url is a cluster gateway")
    parser.add_argument("--gpu", help="with --cluster, the machine profile whose processes are measured")
    args = parser.parse_args(argv)
    if args.gpu is not None and not args.cluster:
        parser.error("--gpu requires --cluster")
    try:
        if args.output.exists():
            raise ValueError("evidence output already exists")
        base_url = validate_upstream_url(args.local_url)
        data = args.model_file.read_bytes()
        if len(data) > _MAX_CONFIG_BYTES:
            raise ValueError("model contract exceeds the byte limit")
        config = ModelConfig.model_validate(yaml.safe_load(data))
        upstreams = load_upstreams(args.upstreams_file)
        key = os.environ[args.api_key_env] if args.api_key_env else None
        with (
            SIEClient(base_url, api_key=key, remote="forbid", timeout_s=60) as local,
            SIEClient(base_url, api_key=key, timeout_s=60) as remote,
        ):
            record = run_probe(
                config, upstreams, local, remote, args.remote_profile, cluster=args.cluster, machine_profile=args.gpu
            )
        # Refuse to clobber existing evidence; a scheduled run chooses a new path.
        with args.output.open("x", encoding="utf-8") as output:
            output.write(record.model_dump_json(indent=2) + "\n")
        print("PASS" if record.passed else "FAIL")
        return 0 if record.passed else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, SIEError, yaml.YAMLError):
        # SDK errors can contain URLs/provider text. Keep failure output fixed.
        print("Probe could not produce valid equivalence evidence.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
