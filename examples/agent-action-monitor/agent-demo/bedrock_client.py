"""Common interface for mock and real Bedrock Converse clients."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


class DuskBlockedError(Exception):
    """Raised when the gate returns a non-ALLOW verdict for a proposed action.

    Carries the full decision payload (verdict, score, reasons, blast
    radius) so callers can inspect and surface why the action was stopped.
    """

    def __init__(self, verdict: dict[str, Any]) -> None:
        self.verdict = verdict
        # `or []`, not a `.get` default -- a verdict payload with an explicit
        # "reasons": None must fall back the same way a missing key does.
        reasons = ", ".join(verdict.get("reasons") or []) or "no reason given"
        super().__init__(f"blocked ({verdict.get('verdict')}): {reasons}")


class BedrockConverseClient(Protocol):
    """The subset of bedrock-runtime this wrapper depends on."""

    def converse(
        self,
        *,
        modelId: str,  # noqa: N803 -- matches boto3's actual converse() signature
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


@dataclass
class DuskBedrockClient:
    """Wrap a Bedrock-compatible client behind one Converse interface."""

    client: BedrockConverseClient
    # Bedrock requires the region-prefixed inference-profile ID for on-demand
    # invocation of this model in most regions -- the bare model ID
    # ("anthropic.claude-3-5-sonnet-20241022-v2:0") 400s with
    # ValidationException: on-demand throughput isn't supported.
    model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

    def converse(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Call the model and return its raw response.

        Args:
            messages: Bedrock Converse-API-shaped message history.

        Returns:
            The raw Bedrock (or mock) response, including any proposed
            tool-call for extract_action() (see actions.py) to parse.
        """
        return self.client.converse(modelId=self.model_id, messages=messages)


def build_real_client(region: str | None = None) -> BedrockConverseClient:
    """Return a real boto3 bedrock-runtime client.

    Requires AWS credentials (``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``,
    plus ``AWS_SESSION_TOKEN`` for temporary credentials) in the environment
    -- compose.yml passes these through from the host's own environment
    (empty by default, so the keyless path is unaffected). Only called when
    USE_REAL_BEDROCK=true; the default keyless path uses MockBedrock instead
    (see mock_bedrock.py, wired in by the harness).

    Args:
        region: AWS region for the client. Defaults to the ``AWS_REGION``
            environment variable, then ``"us-east-1"``, matching the ``us.``
            inference-profile prefix on :data:`DuskBedrockClient.model_id`.
    """
    import boto3

    region = region or os.getenv("AWS_REGION", "us-east-1")
    return boto3.client("bedrock-runtime", region_name=region)  # type: ignore[no-any-return]
