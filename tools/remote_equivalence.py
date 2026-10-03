#!/usr/bin/env python3
"""Measure a configured remote profile against two local SIE runs.

Run from the locked public workspace. Credential arguments name environment
variables; no credential, request body or vector is written to the evidence.
The local model must expose a verified profile identity. Hybrid routing remains
disabled while the explicit remote profile is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
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
from sie_sdk.types import Item
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
_MATRIX_NDIM = 2
_PREFIX = "Represent this text for retrieval:"
_SCORE_QUERY = "relevant documents for search"


@dataclass(frozen=True)
class _Case:
    category: str
    texts: tuple[str, ...]
    token_counts: tuple[int, ...]
    instruction: str | None = None
    is_query: bool = False


def _cases(tokenizer: Any, context: int) -> list[_Case]:
    def count(text: str) -> int:
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
        _Case(category, texts, tuple(count(text) for text in texts), instruction, is_query)
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


def _request(
    client: SIEClient,
    config: ModelConfig,
    case: _Case,
    operation: str,
    outputs: set[str],
    *,
    profile: str,
    upstream: str,
) -> tuple[str, Any]:
    common: dict[str, Any] = {
        "options": {"profile": profile},
        "instruction": case.instruction,
        "wait_for_capacity": False,
        "provision_timeout_s": 60.0,
        "max_oom_retries": 0,
    }
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
                config.sie_id, items, output_types=cast("Any", sorted(outputs)), is_query=case.is_query, **common
            )
            rows = cast("list[Any]", result)
        if not isinstance(rows, list) or len(rows) != (1 if operation == "score" else len(items)):
            raise ValueError("probe result count differs")
        for row in rows:
            evidence = row.get("request") or {}
            expected = "local" if profile == "default" else "remote"
            if evidence.get("served_by") != expected or (expected == "remote" and evidence.get("upstream") != upstream):
                raise ValueError("probe serving provenance is missing or differs")
        return "ok", result
    except RequestError as error:
        code = (error.code or "").upper()
        if error.status_code == HTTPStatus.BAD_REQUEST and code in ("INVALID_INPUT", "INPUT_TOO_LONG"):
            return "input_too_long" if code == "INPUT_TOO_LONG" else "invalid_input", None
        raise ValueError("probe request failed without a comparable input refusal") from None


def run_probe(
    config: ModelConfig, upstreams: Any, local: SIEClient, remote: SIEClient, profile: str
) -> EquivalenceRecord:
    if config.weights_path is not None or not is_immutable_revision(config.hf_revision) or not config.hf_id:
        raise ValueError("probe requires immutable local Hub weights")
    if config.max_sequence_length is None or not _MIN_CONTEXT <= config.max_sequence_length <= _MAX_CONTEXT:
        raise ValueError("probe context is unsupported")
    resolved = config.resolve_profile(profile)
    if profile == "default" or not is_remote_adapter_path(resolved.adapter_path):
        raise ValueError("probe requires an explicit remote profile")
    upstream_name, upstream_model = resolved.loadtime["upstream"], resolved.loadtime["upstream_model"]
    upstream = upstreams[upstream_name]
    before = local.get_model(config.sie_id)
    identity = (before.get("profiles") or {}).get("default", {}).get("identity")
    expected_remote = remote_profile_contract_digest(config, profile, upstreams)
    if (
        expected_remote is None
        or (before.get("profiles") or {}).get(profile, {}).get("remote_contract_sha256") != expected_remote
    ):
        raise ValueError("serving endpoint/model contract differs from the supplied files")
    if not isinstance(identity, str) or not identity.startswith("v1:sha256:"):
        raise ValueError("local execution cannot be identified; no equivalence record can authorize it")
    if before.get("revision") != config.hf_revision or before.get("max_sequence_length") != config.max_sequence_length:
        raise ValueError("local model metadata differs from the supplied model contract")
    outputs = set(config.outputs) & {"dense", "sparse", "multivector", "score"}
    if not outputs:
        raise ValueError("probe requires encode or score outputs")
    tokenizer = load_tokenizer(config.hf_id, revision=config.hf_revision, trust_remote_code=False)
    cases = []
    for operation in ("encode", "score"):
        selected_outputs = {output for output in outputs if (output == "score") == (operation == "score")}
        if not selected_outputs:
            continue
        for case in _cases(tokenizer, config.max_sequence_length):
            if operation == "encode" and case.category == "score_scale":
                continue
            observations = [
                _request(client, config, case, operation, selected_outputs, profile=selected, upstream=upstream_name)
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
    after = local.get_model(config.sie_id)
    if (after.get("profiles") or {}).get("default", {}).get("identity") != identity or (
        after.get("profiles") or {}
    ).get(profile, {}).get("remote_contract_sha256") != expected_remote:
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
        local_observation_sha256=canonical_digest({"identity": identity, "revision": before.get("revision")}),
        local_identity=identity,
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
    args = parser.parse_args(argv)
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
            record = run_probe(config, upstreams, local, remote, args.remote_profile)
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
