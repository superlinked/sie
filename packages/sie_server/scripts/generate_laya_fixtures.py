#!/usr/bin/env python3
"""Generate the Laya reference fixtures used by the adapter's parity tests.

Writes ``packages/sie_server/tests/adapters/fixtures/laya/<variant>.json`` for
``convaiinnovations/laya``, ``laya-multilingual``, and ``laya-typed-decisions``.
Each fixture records what the reference implementation (the ``laya`` 0.3.11
package, CPU float32) produces for a fixed set of states and questions at the
revision pinned in the model's config (``packages/sie_server/models``):

* per (state, question) row: the length and sha256 of the exact ``input_ids``,
  the option-marker positions, the question type, and the raw decision-head
  logits (rounded to 1e-6);
* per state: the reference's answers and ``usage.input_tokens``;
* the same for a per-call ``max_len``/``head_max_len`` override, and the error
  text for options that cannot fit;
* every string the reference tokenized, with its token ids, so the tests can
  check sequence assembly without the real tokenizer.

Not run in CI: it downloads each checkpoint and runs the reference model on
CPU. Re-run it whenever a Laya model config's ``hf_revision`` changes (the
parity tests check that each fixture's revision matches its config), and commit
the regenerated JSON.

Run it in a separate environment with the reference package installed; the
committed fixtures record the torch and transformers versions they came from::

    python -m venv /tmp/laya-ref
    /tmp/laya-ref/bin/pip install laya==0.3.11 torch==2.8.0 transformers==4.57.6 \\
        --extra-index-url https://download.pytorch.org/whl/cpu
    /tmp/laya-ref/bin/python packages/sie_server/scripts/generate_laya_fixtures.py

The JSON is indented one level per line so regenerated fixtures diff line by
line. Keys are not sorted: question, criterion, and state key order determine
the rows the model reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PACKAGE_DIR / "models"
FIXTURES_DIR = PACKAGE_DIR / "tests" / "adapters" / "fixtures" / "laya"
VARIANTS = ("laya", "laya-multilingual", "laya-typed-decisions")
CHECKPOINT_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
LOGIT_DECIMALS = 6

LONG = " ".join(["The shipment was delayed again and the customer wrote in several times."] * 90)
LONG_TURNS: list[dict[str, str]] = []
for _i in range(45):
    LONG_TURNS.append(
        {"role": "user", "content": f"Message {_i}: my order #{1000 + _i} still has not arrived, where is it?"}
    )
    LONG_TURNS.append({"role": "agent", "content": f"Reply {_i}: sorry, we are checking with the courier."})
LONG_TURNS.append({"role": "user", "content": "FINAL: forget it, cancel everything and refund me now."})

CASES: list[tuple[str, Any]] = [
    ("plain_email", "Hi, I was charged twice for my subscription this month. Please refund one charge ASAP!"),
    (
        "json_ticket",
        {
            "subject": "Login broken",
            "body": "Since the update I can't log in; error 500.",
            "customer_tier": "enterprise",
            "prior_tickets": 3,
        },
    ),
    (
        "conversation_list",
        [
            {"role": "user", "content": "I want to cancel."},
            {"role": "agent", "content": "Sorry to hear that - may I ask why?"},
            {"role": "user", "content": "Too expensive, and support is slow. Cancel it now."},
        ],
    ),
    ("long_truncated", LONG),
    ("long_conversation_left_truncated", LONG_TURNS),
    ("non_ascii_masks", "Bonjour, je voudrais annuler ma commande n°1234 — merci. [MASK] Grüße <mask> ¿qué? 😀"),
    ("non_latin", "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें। 二重に請求されました。返金してください。"),
    ("empty", ""),
]

MANY_LONG = {f"dept_{i:02d}": f"requests that belong to department number {i} and nowhere else" for i in range(30)}

QUESTIONS: dict[str, Any] = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
            "refund": "wants money back",
            "cancel": "wants to cancel",
            "technical": "reports a bug",
            "other": None,
        },
    },
    "priority": {"type": "choice", "instructions": "Ticket priority", "criteria": ["low", "normal", "urgent"]},
    "many": {
        "type": "choice",
        "instructions": "Pick the department",
        "criteria": [
            "billing",
            "sales",
            "support",
            "legal",
            "security",
            "hr",
            "it",
            "ops",
            "finance",
            "marketing",
            "product",
            "design",
        ],
    },
    "refund_requested": {"type": "noul", "instructions": "The customer explicitly requests a refund."},
    "angry_labels": {
        "type": "noul",
        "instructions": "Is the customer angry?",
        "criteria": {"true": "angry or hostile tone"},
        "labels": {"false": "calm", "true": "angry"},
    },
    "frustration": {
        "type": "score",
        "instructions": {"rubric": "How frustrated is the customer?"},
        "criteria": ["calm", {"desc": "mildly annoyed"}, "frustrated", "furious"],
    },
    "crit_values": {
        "type": "choice",
        "instructions": "Which channel? [MASK] <mask>",
        "criteria": {"email": 0, "phone": ["call", "voicemail"], "chat": {"live": True}, "none": ""},
    },
    "noul_both": {
        "type": "noul",
        "instructions": "Does the text mention money?",
        "criteria": {"true": "mentions a payment, charge, price or refund", "false": "no money is mentioned"},
    },
    "budget_squeeze": {
        "type": "choice",
        "instructions": "Which of the thirty departments owns this request?",
        "criteria": MANY_LONG,
    },
    "label": {
        "type": "choice",
        "instructions": "Which category does this text belong to?",
        "criteria": ["billing", "technical", "sales", "other"],
    },
}

# Per-call token budget override (reference predict(max_len=..., head_max_len=...)).
OVERRIDE: dict[str, Any] = {
    "max_len": 96,
    "head_max_len": 40,
    "questions": ["intent", "priority", "many", "refund_requested", "frustration"],
    "cases": ["plain_email", "long_conversation_left_truncated"],
}

# Options that cannot fit: the reference's error text.
OVERFLOW_QUESTIONS = {
    "too_many": {"type": "choice", "instructions": "pick", "criteria": [f"option {i}" for i in range(80)]}
}


def _sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


class RecordingTokenizer:
    """Proxy around the reference tokenizer that records every string it tokenizes."""

    def __init__(self, tokenizer: Any) -> None:
        object.__setattr__(self, "_tok", tokenizer)
        object.__setattr__(self, "calls", {})

    def __call__(self, text: str, **kwargs: Any) -> Any:
        if kwargs != {"add_special_tokens": False} or not isinstance(text, str):
            msg = f"unexpected tokenizer call: {text!r} {kwargs!r}"
            raise AssertionError(msg)
        out = self._tok(text, **kwargs)
        self.calls[text] = list(out["input_ids"])
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tok, name)


def pinned_revision(variant: str) -> tuple[str, str]:
    """(repo, revision) from the variant's model config."""
    config = yaml.safe_load((MODELS_DIR / f"convaiinnovations__{variant}.yaml").read_text(encoding="utf-8"))
    return config["hf_id"], config["hf_revision"]


def record(model_dir: Path, repo: str, revision: str) -> dict[str, Any]:
    """Run the reference on the fixed cases and record rows, logits, answers, and tokenizations."""
    import laya
    import torch
    import transformers
    from laya.agent import Agent
    from laya.common import collate_items

    torch.manual_seed(0)
    agent = Agent(str(model_dir), device="cpu")
    recorder = RecordingTokenizer(agent.tok)
    agent.tok = recorder
    raw: dict[str, Any] = {
        "generator": f"laya reference implementation (pip laya=={laya.__version__}), CPU float32",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "repo": repo,
        "revision": revision,
        "max_len": agent.cfg.get("max_len"),
        "head_max_len": agent.cfg.get("head_max_len"),
        "temperature_applied": agent.temperature,
        "temperature_by_options_applied": agent.temperature_by_options,
        "tokenizer": {
            "mask_token": agent.tok.mask_token,
            "mask_token_id": agent.tok.mask_token_id,
            "cls_token_id": agent.tok.cls_token_id,
            "sep_token_id": agent.tok.sep_token_id,
            "pad_token_id": agent.tok.pad_token_id,
        },
        "questions": QUESTIONS,
    }

    def run(states: list[tuple[str, Any]], questions: dict[str, Any], **budget: int) -> list[dict[str, Any]]:
        ids = list(questions)
        internal = {q: agent._to_internal(questions[q]) for q in ids}
        for q in ids:
            agent._check_question(q, questions[q])
        results = agent.predict_batch([state for _, state in states], questions, **budget)
        cases = []
        for (name, state), result in zip(states, results, strict=True):
            items = agent._encode_state(state, ids, internal, **budget)
            batch = collate_items([items], agent.tok.pad_token_id)
            with torch.no_grad():
                logits, _act = agent.model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["marker_pos"],
                    batch["marker_mask"],
                    batch["qtype"],
                )
            rows = []
            for j, qid in enumerate(ids):
                k = len(items[j]["markers"])
                rows.append(
                    {
                        "qid": qid,
                        "len": len(items[j]["ids"]),
                        "sha256": _sha(items[j]["ids"]),
                        "markers": items[j]["markers"],
                        "qtype": items[j]["qtype"],
                        "logits": logits[j, :k].tolist(),
                    }
                )
            for answer in result["answers"].values():
                answer.pop("action", None)
            cases.append(
                {
                    "name": name,
                    "state": state,
                    "rows": rows,
                    "answers": result["answers"],
                    "input_tokens": result["usage"]["input_tokens"],
                }
            )
        return cases

    raw["cases"] = run(CASES, QUESTIONS)
    override_questions = {q: QUESTIONS[q] for q in OVERRIDE["questions"]}
    override_states = [(name, state) for name, state in CASES if name in OVERRIDE["cases"]]
    budget = {"max_len": OVERRIDE["max_len"], "head_max_len": OVERRIDE["head_max_len"]}
    raw["override"] = {
        **budget,
        "questions": override_questions,
        "cases": run(override_states, override_questions, **budget),
    }
    try:
        agent.predict_batch(["x"], OVERFLOW_QUESTIONS, head_max_len=64, max_len=96)
        raw["overflow_error"] = None
    except ValueError as exc:
        raw["overflow_error"] = str(exc)
    raw["tokenizations"] = recorder.calls
    return raw


def to_fixture(raw: dict[str, Any], rl_agent_config: dict[str, Any]) -> dict[str, Any]:
    """Shape a recording into the committed fixture: logits rounded, shipped temperatures alongside."""
    for section in (raw["cases"], raw["override"]["cases"]):
        for case in section:
            for row in case["rows"]:
                row["logits"] = [round(x, LOGIT_DECIMALS) for x in row["logits"]]
    return {
        "generator": raw["generator"],
        "environment": {"torch": raw["torch"], "transformers": raw["transformers"]},
        "repo": raw["repo"],
        "revision": raw["revision"],
        "max_len": raw["max_len"],
        "head_max_len": raw["head_max_len"],
        "rl_agent_config": {k: rl_agent_config[k] for k in ("temperature", "temperature_by_options")},
        "temperature_applied": raw["temperature_applied"],
        "temperature_by_options_applied": raw["temperature_by_options_applied"],
        "tokenizer": raw["tokenizer"],
        "questions": raw["questions"],
        "cases": raw["cases"],
        "override": raw["override"],
        "overflow_error": raw["overflow_error"],
        "tokenizations": raw["tokenizations"],
    }


def dump_fixture(fixture: dict[str, Any]) -> str:
    return json.dumps(fixture, ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", choices=VARIANTS, help="Variant to regenerate (default: all)")
    parser.add_argument("--out-dir", type=Path, default=FIXTURES_DIR)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variant or VARIANTS:
        repo, revision = pinned_revision(variant)
        model_dir = Path(snapshot_download(repo, revision=revision, allow_patterns=CHECKPOINT_FILES))
        rl_agent_config = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
        text = dump_fixture(to_fixture(record(model_dir, repo, revision), rl_agent_config))
        out = args.out_dir / f"{variant}.json"
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} ({len(text.encode())} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
