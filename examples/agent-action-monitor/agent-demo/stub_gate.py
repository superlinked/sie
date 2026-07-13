"""Schema-compatible local gate stub with canned verdicts."""

from __future__ import annotations

import os
import uuid
from typing import Any

from flask import Flask, request

app = Flask(__name__)

#: action_type -> canned verdict. Anything else defaults to ALLOW.
CANNED_VERDICTS: dict[str, dict[str, Any]] = {
    "firewall_rule_change": {
        "verdict": "BLOCK",
        "score": 0.91,
        "blast": "high",
        "mitre_attack": ["T1562.004"],
        "mitre_atlas": ["AML.T0051"],
        "reasons": ["stub: firewall_rule_change is out of this agent's canned baseline"],
        "predicted_next": "unknown",
        "similar_decision_ids": [],
    },
}

_DEFAULT_VERDICT: dict[str, Any] = {
    "verdict": "ALLOW",
    "score": 0.05,
    "blast": "low",
    "mitre_attack": [],
    "mitre_atlas": [],
    "reasons": ["stub: no canned rule matched, defaulting to ALLOW"],
    "predicted_next": "unknown",
    "similar_decision_ids": [],
}


@app.get("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.post("/v1/gate")
def gate() -> tuple[dict[str, Any], int]:
    action = request.get_json(force=True, silent=True)
    if not isinstance(action, dict):
        return {"error": "expected an AgentAction JSON object"}, 400
    for field in ("agent_id", "timestamp", "action_type", "target", "source"):
        if not action.get(field):
            return {"error": f"missing required field: {field}"}, 400

    verdict = dict(CANNED_VERDICTS.get(action["action_type"], _DEFAULT_VERDICT))
    verdict["trace_id"] = str(uuid.uuid4())
    return verdict, 200


def run() -> None:
    port = int(os.getenv("STUB_GATE_PORT", "8000"))
    host = os.getenv("STUB_GATE_HOST", "127.0.0.1")
    app.run(host=host, port=port)


if __name__ == "__main__":
    run()
