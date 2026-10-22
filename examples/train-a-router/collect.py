"""BBEH-specific OpenAI request adapter; no reference answers or SIE credentials."""

import json
from decimal import Decimal

ARMS = {"luna": {"provider": "openai", "model": "gpt-6-luna"},
        "sol": {"provider": "openai", "model": "gpt-6.1-sol"}}
PRICES = {
    "gpt-6.1-sol": {"input": "2", "cached_input": "0.10", "cache_write": "2.50", "output": "10"},
    "gpt-6-luna": {"input": "0.10", "cached_input": "0.01", "cache_write": "0.125", "output": "0.50"},
}
SYSTEM = (
    "Solve the question independently, using the full rules and options supplied. "
    "Do not use tools or external sources. Return the requested JSON object. "
    "The answer field must contain only the complete final answer in the format "
    "requested by the source question. Preserve semantic brackets, quotes, "
    "punctuation, nesting, case and internal spaces. JSON is only the transport "
    "container; it does not replace or remove any part of the requested answer. "
)
OPENAI_SETTINGS = {"reasoning": {"effort": "medium"}, "max_output_tokens": 16384,
                   "store": False, "service_tier": "default"}


def body_for(item, arm):
    if set(item) != {"id", "request", "response_contract"}:
        raise ValueError("Only question ID, request and public answer contract may reach collector")
    if not isinstance(item["request"], str) or not item["request"].strip():
        raise ValueError("Empty question")
    if item["response_contract"] != {"type": "exact_string", "max_length": 2048}:
        raise ValueError("BBEH uses one prospective complete-string answer contract")
    return {"model": ARMS[arm]["model"], "instructions": SYSTEM,
            "input": item["request"] + "\n\nFinal-answer type: exact_string.",
            **OPENAI_SETTINGS, "text": {"format": {"type": "json_schema", "name": "answer", "strict": True,
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                           "required": ["answer"], "additionalProperties": False}}}}


def parse_answer(content, contract):
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"answer"} or not isinstance(value["answer"], str):
        return None
    answer = value["answer"].strip()
    return answer if answer and len(answer) <= contract["max_length"] else None


def openai_cost(usage, body):
    incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
    details = usage.get("input_tokens_details") or {}
    cached, written = details.get("cached_tokens", 0), details.get("cache_write_tokens", 0)
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
    values = (incoming, outgoing, cached, written, reasoning)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Invalid token usage")
    if cached + written > incoming or reasoning > outgoing or outgoing > body["max_output_tokens"]:
        raise ValueError("Inconsistent usage")
    p = {key: Decimal(value) for key, value in PRICES[body["model"]].items()}
    cost = ((incoming - cached - written) * p["input"] + cached * p["cached_input"]
            + written * p["cache_write"] + outgoing * p["output"]) / 1_000_000
    budget = ((incoming - cached) * p["cache_write"] + cached * p["cached_input"]
              + outgoing * p["output"]) / 1_000_000
    return {"usd_estimate": str(cost), "budget_usd": str(budget),
            "basis": "token-accounted standard list prices; output includes reasoning",
            "cache_write_detail_present": "cache_write_tokens" in details}


def reservation(body, arm):
    p = PRICES[body["model"]]
    input_bound = len(json.dumps(body, ensure_ascii=False).encode()) + 2048
    return (Decimal(input_bound) * Decimal(p["cache_write"])
            + Decimal(body["max_output_tokens"]) * Decimal(p["output"])) / 1_000_000
