"""Common interface for mock and real Bedrock Converse clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class DuskBlockedError(Exception):
    """Raised when the gate returns a non-ALLOW verdict for a proposed action.

    Carries the full decision payload (verdict, score, reasons, blast
    radius) so callers can inspect and surface why the action was stopped.
    """

    def __init__(self, verdict: dict[str, Any]) -> None:
        self.verdict = verdict
        reasons = ", ".join(verdict.get("reasons", [])) or "no reason given"
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
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def converse(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Call the model and return its raw response.

        Args:
            messages: Bedrock Converse-API-shaped message history.

        Returns:
            The raw Bedrock (or mock) response, including any proposed
            tool-call for extract_action() (see actions.py) to parse.
        """
        return self.client.converse(modelId=self.model_id, messages=messages)


def build_real_client(region: str = "us-east-1") -> BedrockConverseClient:
    """Return a real boto3 bedrock-runtime client.

    Requires AWS credentials to be configured in the environment. Only
    called when USE_REAL_BEDROCK=true; the default keyless path uses
    MockBedrock instead (see mock_bedrock.py, wired in by the harness).

    Args:
        region: AWS region for the client.
    """
    import boto3

    return boto3.client("bedrock-runtime", region_name=region)  # type: ignore[no-any-return]
