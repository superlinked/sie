#!/usr/bin/env python3
"""Ask every case its typed questions through each backend, or check the recorded calls.

    python3 run.py --check --calls run-output/calls.json        # offline, nothing installed
    python3 run.py --show vulnerability-triage CVE-2023-5534 gliclass short

    uv sync
    uv run python run.py --record --set vulnerability-triage --backend laya --url http://localhost:8080
    python3 run.py --merge run-output/*--*.json            # one calls.json and manifest.json

Backends, each behind the same three steps (build the requests, send them,
record what came back). Every planned backend goes through SIE `extract` with
`sie_sdk.SIEClient`, per AGENTS.md:

    laya                     convaiinnovations/laya, the English checkpoint
    laya-typed-decisions     convaiinnovations/laya-typed-decisions, fine-tuned on
                             the workflow set's training split
    gliclass                 knowledgator/gliclass-large-v3.0, one call per question
    gliclass-large-v1        knowledgator/gliclass-large-v1.0, one call per question
    gliclass-base-v1         knowledgator/gliclass-base-v1.0, one call per question
    gliclass-small-v1        knowledgator/gliclass-small-v1.0, one call per question
    gliclass-instruct-large  knowledgator/gliclass-instruct-large-v1.0, one call per
                             question, the question sent as its instruction
    gliclass-instruct-large-grouped
                             the same model, every question a label group in one call
    gliformer-large          knowledgator/gliformer-large-v1, every question a label
                             group in one call, the question as the group's name
    gliner2-base             fastino/gliner2-base-v1, one call per question
    gliner2-large            fastino/gliner2-large-v1, one call per question
    qwen3-4b-instruct        Qwen/Qwen3-4B-Instruct-2507, one structured-output
                             call returning the CVSS v3.1 base metrics and the
                             weakness class; the typed answers are read off those
                             metrics
    qwen3.5-4b               Qwen/Qwen3.5-4B, the same call, through its
                             non-speculative profile
    nli-modernbert-base      MoritzLaurer/ModernBERT-base-zeroshot-v2.0, an NLI
                             zero-shot classifier, one call per question

The Laya checkpoints are Convai Innovations', GLiClass and GLiFormer are
Knowledgator's, GLiNER2 is Fastino's, the NLI classifier is MoritzLaurer's and
Qwen3 is Qwen's.

A typed backend (Laya) receives the state and a dict of typed questions in one
call and answers all of them; it encodes the state once per question inside
that call, so one call is not one encoder pass. A grouped backend receives the
state as text and every question as a label group in one call. A per-question
backend receives the state as text and one question's labels per call, so a
three-question case is three calls. See questions.py for how each question is
rendered for each.

`laya-package` and `laya-typed-decisions-package` run the same checkpoints
through their reference `laya` package in this process, at a pinned Hugging
Face revision. They are not in PLAN; they exist to check that SIE answers the
way the reference implementation does. The import of each client is deferred
into the runner that needs it, so `--check`, `--show` and `--merge` work on a
bare `python3`.

Every recorded latency is measured on whatever machine ran the recording, one
case at a time. It is not a served-latency benchmark.

`--check` rebuilds every recorded request from the case files and compares it
with the recorded one. It is a bijection over PLAN: a call that is missing,
recorded twice, or implied by no case fails, unless `--partial` limits the
check to the backends and sets present in the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tuning
from questions import (
    CONCRETE,
    DESCRIBED,
    SHORT,
    VULNERABILITY_QUESTIONS,
    VULNERABILITY_TRIAGE,
    WORKFLOWS,
    Phrasing,
    asked,
    from_source_question,
    humanize,
    label_options,
    phrasing_of,
    state_text,
    typed_questions,
)
from tuning import TUNED

if TYPE_CHECKING:
    from sie_sdk import SIEClient

TASK = "typed-decisions"
DEFAULT_SIE_URL = "http://localhost:8080"

# The reference implementation, for checking SIE's Laya answers against it.
LAYA_PACKAGE_CHECKPOINTS = {
    "laya-package": ("convaiinnovations/laya", "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8"),
    "laya-typed-decisions-package": (
        "convaiinnovations/laya-typed-decisions",
        "bc76315b568af04bc19f133c8bca9a2b3a2d9905",
    ),
}
LAYA_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]

SIE_MODELS: dict[str, dict[str, str]] = {
    "laya": {"model": "convaiinnovations/laya", "family": "laya", "vendor": "Convai Innovations"},
    "laya-typed-decisions": {
        "model": "convaiinnovations/laya-typed-decisions",
        "family": "laya",
        "vendor": "Convai Innovations",
    },
    "gliclass": {"model": "knowledgator/gliclass-large-v3.0", "family": "gliclass", "vendor": "Knowledgator"},
    "gliclass-large-v1": {"model": "knowledgator/gliclass-large-v1.0", "family": "gliclass", "vendor": "Knowledgator"},
    # The same model, asked every question in one call: each question a label
    # group, each group encoded as its own row. It answers as the per-question
    # calls do and inherits their tuned settings (tune.INHERITS).
    "gliclass-large-v1-one-call": {
        "model": "knowledgator/gliclass-large-v1.0",
        "family": "gliclass-separate",
        "vendor": "Knowledgator",
    },
    "gliclass-base-v1": {"model": "knowledgator/gliclass-base-v1.0", "family": "gliclass", "vendor": "Knowledgator"},
    "gliclass-small-v1": {"model": "knowledgator/gliclass-small-v1.0", "family": "gliclass", "vendor": "Knowledgator"},
    "gliclass-instruct-large": {
        "model": "knowledgator/gliclass-instruct-large-v1.0",
        "family": "gliclass-instruct",
        "vendor": "Knowledgator",
    },
    "gliclass-instruct-large-grouped": {
        "model": "knowledgator/gliclass-instruct-large-v1.0",
        "family": "gliclass-grouped",
        "vendor": "Knowledgator",
    },
    "gliformer-large": {"model": "knowledgator/gliformer-large-v1", "family": "gliformer", "vendor": "Knowledgator"},
    "gliner2-base": {"model": "fastino/gliner2-base-v1", "family": "gliner2", "vendor": "Fastino"},
    "gliner2-large": {"model": "fastino/gliner2-large-v1", "family": "gliner2", "vendor": "Fastino"},
    # Fastino's typed decision models take Laya's question mapping as output_schema
    # and answer in Laya's shape, one call per record.
    "gliner2.5-decide": {"model": "fastino/GLiNER2.5-Decide", "family": "decide", "vendor": "Fastino"},
    "gliner2.5-multi-decide": {"model": "fastino/GLiNER2.5-multi-Decide", "family": "decide", "vendor": "Fastino"},
    "gliner2.5-decide-1b": {"model": "fastino/GLiNER2.5-Decide-1B", "family": "decide", "vendor": "Fastino"},
    "nli-modernbert-base": {
        "model": "MoritzLaurer/ModernBERT-base-zeroshot-v2.0",
        "family": "nli",
        "vendor": "MoritzLaurer",
    },
    "qwen3-4b-instruct": {"model": "Qwen/Qwen3-4B-Instruct-2507", "family": "llm", "vendor": "Qwen"},
    # Qwen3.5-4B's default profile decodes speculatively, which bypasses the
    # JSON-schema grammar. A schema request goes to its non-speculative profile,
    # as SIE's gateway routes it; against a bare server it is named here.
    "qwen3.5-4b": {"model": "Qwen/Qwen3.5-4B:no-spec", "family": "llm", "vendor": "Qwen"},
}

# The LLM backend scores a vulnerability report as CVSS v3.1 base metrics plus
# a weakness class, in one structured-output call; score.py reads the typed
# questions' answers off those metrics and computes the severity from them.
LLM_MAX_TOKENS = 300
LLM_SYSTEM = (
    "You score software vulnerability reports the way an NVD analyst does. Read the report and "
    "return its CVSS v3.1 base metrics and the kind of weakness it describes, as JSON matching "
    "the schema. Use only what the report says or clearly implies."
)
LLM_METRICS = {
    "attack_vector": ["network", "adjacent", "local", "physical"],
    "attack_complexity": ["low", "high"],
    "privileges_required": ["none", "low", "high"],
    "user_interaction": ["none", "required"],
    "scope": ["unchanged", "changed"],
    "confidentiality": ["none", "low", "high"],
    "integrity": ["none", "low", "high"],
    "availability": ["none", "low", "high"],
}

# An NLI classifier turns each label into a hypothesis through a template. The
# model config's template reads as a topic ("This text is about {}.") and fits
# option names; a yes-or-no question's labels are already statements, so they
# go in as they are.
NLI_TOPIC_TEMPLATE = "This text is about {}."
NLI_STATEMENT_TEMPLATE = "{}"

# How each family's answers come back, which is how score.py reads them.
TRANSPORTS = {
    "laya": "sie-extract-typed",
    "decide": "sie-extract-typed",
    "gliclass-grouped": "sie-extract-groups",
    "gliclass-separate": "sie-extract-groups",
    "gliformer": "sie-extract-group-labels",
    "gliclass": "sie-extract",
    "gliclass-instruct": "sie-extract",
    "gliner2": "sie-extract",
    "nli": "sie-extract",
    "llm": "sie-chat-cvss",
}

# Which backend answers which set, per split, in which phrasings. On the
# vulnerability set every phrasing is recorded on dev, where tune.py chooses the
# per-question mix and the decision rules; test is recorded once, in that tuned
# mix. The workflow set keeps its source's own questions, has no dev split and
# is not tuned.
# Grouped GLiClass cannot take the described phrasing: its six labelled groups
# need 573 tokens before any record text, over the model's 512-token window.
DEV_PHRASINGS = {
    "gliclass-instruct-large-grouped": (SHORT, CONCRETE),
    # GLiNER2.5-Decide reads every question's prompt and labels in at most 256 of
    # its 512 tokens; the described questions take 369.
    "gliner2.5-decide": (SHORT, CONCRETE),
    # One-call GLiClass takes the per-question backend's settings (tune.INHERITS),
    # so on dev it is recorded only in that tuned mix, to judge it by the page rule.
    "gliclass-large-v1-one-call": (),
}
# Backends that cannot take the workflow set's questions as the source writes
# them: GLiNER2.5-Decide's 256-token prompt budget holds none of the four
# workflows' question sets (305 to 425 tokens, checked with a placeholder text).
WORKFLOWS_UNFIT = ("gliner2.5-decide",)
# The LLM backend sends one fixed prompt, so it has one phrasing and nothing to
# tune; its prompt is about CVSS, so it answers only the vulnerability set.
LLM_BACKENDS = tuple(backend for backend, spec in SIE_MODELS.items() if spec["family"] == "llm")
# The LLM the page's board compares against, chosen on dev (PREREGISTRATION.md).
# Every LLM is recorded on dev; only this one is recorded on test.
PAGE_LLM = "qwen3-4b-instruct"
PLAN: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    VULNERABILITY_TRIAGE: {
        backend: (
            ({"dev": (DESCRIBED,), "test": (DESCRIBED,)} if backend == PAGE_LLM else {"dev": (DESCRIBED,)})
            if backend in LLM_BACKENDS
            else {"dev": (*DEV_PHRASINGS.get(backend, (SHORT, DESCRIBED, CONCRETE)), TUNED), "test": (TUNED,)}
        )
        for backend in SIE_MODELS
    },
    WORKFLOWS: {
        backend: {"test": (DESCRIBED,)}
        for backend in SIE_MODELS
        if backend not in LLM_BACKENDS and backend not in WORKFLOWS_UNFIT
    },
}
VARIANT_CHOICES = (SHORT, DESCRIBED, CONCRETE, TUNED)


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py (or build_inputs.py)")
    return json.loads(path.read_text(encoding="utf-8"))


def case_file(data_dir: Path, set_name: str) -> dict[str, Any]:
    return load(data_dir / f"inputs/{set_name}/cases.json")


def questions_for(cases: dict[str, Any], case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The question set one case is asked, in the shape questions.py renders."""
    if cases["set"] == VULNERABILITY_TRIAGE:
        if cases["questions"] != VULNERABILITY_QUESTIONS:
            raise SystemExit("the case file's questions differ from questions.py; rebuild or re-fetch the inputs")
        return cases["questions"]
    source = cases["questions"][case["workflow"]]
    return {qid: from_source_question(question) for qid, question in source.items()}


def is_package(backend: str) -> bool:
    return backend in LAYA_PACKAGE_CHECKPOINTS


def family_of(backend: str) -> str:
    return "laya" if is_package(backend) else SIE_MODELS[backend]["family"]


def grouped_instruction(questions: dict[str, dict[str, Any]]) -> str:
    """One instruction naming every question, for a model that takes one prompt for the whole call."""
    return " ".join(f"{humanize(qid)}: {question['instructions']}" for qid, question in questions.items())


def requests_for(
    backend: str, variant: Phrasing, state: Any, questions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every request one case sends through one backend, built from the inputs alone.

    `variant` is a phrasing, or a dict of question id to phrasing (the tuned
    mix; see `tuning.phrasing`). This is what `--check` compares with the
    recording, so it must stay a pure function of its arguments. A grouped
    request also records which group name stands for which question, outside
    the body it sends.
    """
    if is_package(backend):
        rendered, _ = typed_questions(questions, variant)
        return [{"state": state, "questions": rendered}]
    family = family_of(backend)
    if family in ("laya", "decide"):
        # SIE's Laya adapter takes a text state as `text` and a structured one as
        # `metadata.state`, and the question dict as `output_schema`.
        rendered, _ = typed_questions(questions, variant)
        item = {"text": state} if isinstance(state, str) else {"metadata": {"state": state}}
        return [{"body": {"items": [item], "params": {"output_schema": rendered}}}]
    text = state_text(state)
    if family == "llm":
        weakness = [option["name"] for option in questions["weakness"]["options"]]
        schema = {
            "type": "object",
            "properties": {
                "weakness": {"type": "string", "enum": weakness},
                **{name: {"type": "string", "enum": values} for name, values in LLM_METRICS.items()},
            },
            "required": ["weakness", *LLM_METRICS],
            "additionalProperties": False,
        }
        return [
            {
                "body": {
                    "model": SIE_MODELS[backend]["model"],
                    "messages": [
                        {"role": "system", "content": LLM_SYSTEM},
                        {"role": "user", "content": text},
                    ],
                    "max_completion_tokens": LLM_MAX_TOKENS,
                    "temperature": 0.0,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "cvss", "strict": True, "schema": schema},
                    },
                }
            }
        ]
    if family == "gliclass-separate":
        # Each question a label group named by its id, in its own phrasing. Each
        # group is encoded as its own row, so it scores as a labels request with
        # that group's labels would.
        groups = {
            qid: [label for _, label in label_options(q, phrasing_of(variant, qid))] for qid, q in questions.items()
        }
        params = {"options": {"label_groups": groups, "group_encoding": "separate", "overflow_policy": "truncate_text"}}
        return [{"groups": {qid: qid for qid in questions}, "body": {"items": [{"text": text}], "params": params}}]
    if family in ("gliclass-grouped", "gliformer"):
        # GLiClass joins a group's name to each of its labels, so the name stays
        # short and the questions travel in the instruction. GLiFormer reads a
        # group's name as part of its prompt and takes no instruction, so there
        # the name is the question itself.
        if family == "gliclass-grouped":
            names = {qid: humanize(qid) for qid in questions}
        else:
            names = {qid: question["instructions"] for qid, question in questions.items()}
        groups = {
            names[qid]: [label for _, label in label_options(q, phrasing_of(variant, qid))]
            for qid, q in questions.items()
        }
        if family == "gliclass-grouped":
            # Joint: one row per record with every group's labels, as the
            # server encoded label groups before separate rows became the default.
            params: dict[str, Any] = {
                "instruction": grouped_instruction(questions),
                "options": {"label_groups": groups, "group_encoding": "joint", "overflow_policy": "truncate_text"},
            }
        else:
            # Every label's score, not only the winner's: threshold 0 and multi-label.
            params = {"options": {"label_groups": groups, "multi_label": True, "threshold": 0.0}}
        return [
            {
                "groups": {names[qid]: qid for qid in questions},
                "body": {"items": [{"text": text}], "params": params},
            }
        ]
    built = []
    for qid, question in questions.items():
        labels = [label for _, label in label_options(question, phrasing_of(variant, qid))]
        params = {"labels": labels}
        if family == "gliner2":
            # GLiNER2 classifies against a named task; threshold 0 keeps the top
            # label even when its confidence is low, rather than returning nothing.
            params["options"] = {"classification_task": humanize(qid), "threshold": 0.0}
        if family in ("gliclass", "gliclass-instruct"):
            # Cut the record text, never the labels, when the two do not fit in
            # 512 tokens together. Without it GLiClass v1.0 base and small refuse
            # the longest customer-service threads with the full label set.
            params["options"] = {"overflow_policy": "truncate_text"}
        if family == "gliclass-instruct":
            # The instruct models take a task prompt, so the question itself goes in.
            params["instruction"] = question["instructions"]
        if family == "nli":
            template = NLI_STATEMENT_TEMPLATE if question["type"] == "noul" else NLI_TOPIC_TEMPLATE
            params["options"] = {"hypothesis_template": template}
        built.append({"question": qid, "body": {"items": [{"text": text}], "params": params}})
    return built


def model_of(backend: str) -> str:
    if is_package(backend):
        return LAYA_PACKAGE_CHECKPOINTS[backend][0]
    return SIE_MODELS[backend]["model"]


class CallFailedError(Exception):
    """A call that returned, but not with something that can be recorded as an answer."""


class LayaPackageRunner:
    """A Laya checkpoint through its reference `laya` package, in this process."""

    transport = "laya-python-package"

    def __init__(self, backend: str, device: str) -> None:
        # Deferred: optional dependencies, needed only to record.
        import laya  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        repo, revision = LAYA_PACKAGE_CHECKPOINTS[backend]
        path = snapshot_download(repo, revision=revision, allow_patterns=LAYA_FILES)
        self.agent = laya.load(path, device=device)
        self.model = repo
        self.describe = {
            "model": repo,
            "revision": revision,
            "package": f"laya=={laya.__version__}",
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": str(self.agent.device),
            "temperature_by_options": self.agent.cfg.get("temperature_by_options"),
        }

    def send(self, request: dict[str, Any]) -> dict[str, Any]:
        result = self.agent.predict(request["state"], request["questions"])
        answers = result.get("answers") or {}
        if set(answers) != set(request["questions"]):
            raise CallFailedError(f"answered {sorted(answers)}, asked {sorted(request['questions'])}")
        return result


class SIEExtractRunner:
    """A model served by SIE, called through `sie_sdk.SIEClient.extract`.

    What comes back depends on the family, and is recorded under a transport
    name that tells score.py how to read it: Laya's typed answers under `data`,
    grouped GLiClass answers under `data`, GLiFormer's grouped labels and every
    per-question model's labels under `classifications`.
    """

    def __init__(self, backend: str, base_url: str, api_key: str | None) -> None:
        # Deferred so --check, --show and --merge run on a bare python3.
        from sie_sdk import SIEClient  # noqa: PLC0415

        self.model = SIE_MODELS[backend]["model"]
        self.family = SIE_MODELS[backend]["family"]
        self.transport = TRANSPORTS[self.family]
        self.base_url = base_url
        self.client: SIEClient = SIEClient(base_url, api_key=api_key, timeout_s=900)
        self.describe = {"model": self.model, "endpoint": base_url, **served_model(base_url, self.model, api_key)}

    def send(self, request: dict[str, Any]) -> dict[str, Any]:
        body = request["body"]
        if self.family == "llm":
            completion = self.client.chat_completions(
                body["model"],
                body["messages"],
                max_completion_tokens=body["max_completion_tokens"],
                temperature=body["temperature"],
                response_format=body["response_format"],
            )
            choices = completion.get("choices") or []
            content = (choices[0].get("message") or {}).get("content") if choices else None
            if not content:
                raise CallFailedError("response carried no assistant message")
            try:
                parsed = json.loads(content)
            except ValueError as error:
                raise CallFailedError(f"assistant message is not JSON: {error}") from error
            return {"content": content, "parsed": parsed, "usage": completion.get("usage")}
        params = body["params"]
        result = self.client.extract(
            self.model,
            body["items"][0],
            labels=params.get("labels"),
            output_schema=params.get("output_schema"),
            instruction=params.get("instruction"),
            options=params.get("options"),
        )
        if result.get("error"):
            raise CallFailedError(f"item error {result['error']}")
        if self.family in ("laya", "decide"):
            data = result.get("data") or {}
            if set(data) != set(params["output_schema"]):
                raise CallFailedError(f"answered {sorted(data)}, asked {sorted(params['output_schema'])}")
            return {"data": data, "model": result.get("model", self.model)}
        if self.family in ("gliclass-grouped", "gliclass-separate"):
            data = result.get("data") or {}
            if set(data) != set(request["groups"]):
                raise CallFailedError(f"answered groups {sorted(data)}, asked {sorted(request['groups'])}")
            return {"data": data, "model": result.get("model", self.model)}
        classifications = result.get("classifications")
        if not classifications:
            raise CallFailedError("response carried no classifications")
        if self.family == "gliformer":
            sent = {f"{name}.{label}" for name, labels in params["options"]["label_groups"].items() for label in labels}
        else:
            sent = set(params["labels"])
        unknown = {entry["label"] for entry in classifications} - sent
        if unknown:
            raise CallFailedError(f"response named labels that were not sent: {sorted(unknown)[:3]}")
        return {"classifications": classifications, "model": result.get("model", self.model)}


def served_model(base_url: str, model: str, api_key: str | None) -> dict[str, Any]:
    """The revision the server reports for `model`, from its model list."""
    request = urllib.request.Request(f"{base_url}/v1/models")  # noqa: S310
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            listed = json.loads(response.read())
    except OSError as error:
        return {"served_revision": None, "model_list_error": str(error)}
    entries = (listed.get("models") or listed.get("data") or []) if isinstance(listed, dict) else []
    for entry in entries:
        if model in (entry.get("name"), entry.get("id")):
            return {"served_revision": entry.get("revision")}
    return {"served_revision": None}


Runner = LayaPackageRunner | SIEExtractRunner


def make_runner(backend: str, args: argparse.Namespace) -> Runner:
    if is_package(backend):
        return LayaPackageRunner(backend, args.device)
    api_key = os.environ.get("SIE_API_KEY", "").strip() or None
    return SIEExtractRunner(backend, args.url, api_key)


def call_id(set_name: str, backend: str, variant: str, slug: str) -> str:
    return f"{set_name}/{backend}/{variant}/{slug}"


def record_case(
    runner: Runner,
    set_name: str,
    backend: str,
    variant: str,
    case: dict[str, Any],
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": call_id(set_name, backend, variant, case["slug"]),
        "set": set_name,
        "backend": backend,
        "variant": variant,
        "case": case["slug"],
        "split": case["split"],
        "model": runner.model,
        "transport": runner.transport,
        "requests": requests,
    }
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    responses: list[dict[str, Any]] = []
    per_request_ms: list[float] = []
    try:
        for request in requests:
            started = time.perf_counter()
            responses.append(runner.send(request))
            per_request_ms.append(round((time.perf_counter() - started) * 1000, 2))
    except Exception as error:  # noqa: BLE001
        entry.update(
            {
                "status": "error",
                "error": {"type": type(error).__name__, "message": str(error)},
                "responses": None,
                "timing": {"at": requested_at, "latency_ms": None, "per_request_ms": per_request_ms},
            }
        )
        return entry
    entry.update(
        {
            "status": "ok",
            "responses": responses,
            "timing": {
                "at": requested_at,
                "latency_ms": round(sum(per_request_ms), 2),
                "per_request_ms": per_request_ms,
            },
        }
    )
    return entry


def environment() -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "hardware_note": "latencies are one record at a time on this machine, not a served-latency benchmark",
    }


def planned(set_name: str, backend: str, split: str) -> tuple[str, ...]:
    """The variants PLAN records for a backend on one split; a reference-package backend follows its SIE twin."""
    return PLAN[set_name].get(backend.removesuffix("-package"), {}).get(split, ())


def record(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    cases = case_file(data_dir, args.set)
    splits = [args.split] if args.split else sorted({case["split"] for case in cases["cases"]})
    work: list[tuple[str, dict[str, Any]]] = []
    for split in splits:
        variants = tuple(args.variant) if args.variant else planned(args.set, args.backend, split)
        selected = [case for case in cases["cases"] if case["split"] == split]
        if args.limit:
            selected = selected[: args.limit]
        work.extend((variant, case) for variant in variants for case in selected)
    if not work:
        raise SystemExit(f"nothing planned for {args.set}/{args.backend} on {', '.join(splits)}")

    def phrasing(variant: str) -> Phrasing:
        return tuning.phrasing(args.set, args.backend, variant)

    runner = make_runner(args.backend, args)
    # One unrecorded call first, so model loading is not charged to the first case.
    warm_variant, warm = work[0]
    for request in requests_for(
        args.backend, phrasing(warm_variant), warm["state"], asked(args.set, args.backend, questions_for(cases, warm))
    ):
        runner.send(request)

    calls: list[dict[str, Any]] = []
    failed: list[str] = []
    for index, (variant, case) in enumerate(work, start=1):
        requests = requests_for(
            args.backend, phrasing(variant), case["state"], asked(args.set, args.backend, questions_for(cases, case))
        )
        entry = record_case(runner, args.set, args.backend, variant, case, requests)
        calls.append(entry)
        if entry["status"] != "ok":
            failed.append(f"{entry['id']}: {entry['error']['type']}: {entry['error']['message']}")
        if index % 50 == 0 or index == len(work):
            print(f"{args.backend}: {index} of {len(work)}", file=sys.stderr, flush=True)

    out = Path(args.out or f"run-output/{args.set}--{args.backend}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": TASK,
        "set": args.set,
        "backend": args.backend,
        "recorded": {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "runner": runner.describe,
            "environment": environment(),
            "concurrency": 1,
            "warm_up": "one unrecorded case before the first recorded call",
            # As given with --server-commit: the SIE commit the server ran.
            "server_commit": args.server_commit[0] if args.server_commit else None,
        },
        "call_count": len(calls),
        "failed_calls": len(failed),
        "complete": not failed and not args.limit,
        "calls": calls,
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(calls)} calls, {len(failed)} failed")
    for line in failed:
        print(f"  FAILED {line}", file=sys.stderr)
    return 1 if failed else 0


def expected_calls(
    data_dir: Path, scope: set[tuple[str, str, str, str]] | None
) -> dict[str, tuple[str, str, str, list]]:
    """Every call PLAN implies, by id, with the requests the inputs rebuild for it.

    `scope`, when given, keeps only the (set, backend, variant, split) tuples
    present in the recording being checked.
    """
    expected: dict[str, tuple[str, str, str, list]] = {}
    tuned = tuning.load()
    for set_name, backends in PLAN.items():
        cases = case_file(data_dir, set_name) if scope is None or any(t[0] == set_name for t in scope) else None
        if cases is None:
            continue
        for backend, splits in backends.items():
            for case in cases["cases"]:
                for variant in splits.get(case["split"], ()):
                    if scope is not None and (set_name, backend, variant, case["split"]) not in scope:
                        continue
                    phrasing = tuning.phrasing(set_name, backend, variant, tuned)
                    requests = requests_for(
                        backend, phrasing, case["state"], asked(set_name, backend, questions_for(cases, case))
                    )
                    expected[call_id(set_name, backend, variant, case["slug"])] = (set_name, backend, variant, requests)
    return expected


def body_digest(body: Any) -> str:
    """SHA-256 of a request body as canonical JSON (sorted keys, no whitespace, UTF-8)."""
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def digested(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Requests with each body replaced by its digest; a request already digested is kept."""
    out = []
    for request in requests:
        if "body" in request:
            request = {**{k: v for k, v in request.items() if k != "body"}, "body_sha256": body_digest(request["body"])}
        out.append(request)
    return out


def check(data_dir: Path, calls_paths: list[Path], partial: bool) -> int:
    calls: list[dict[str, Any]] = []
    for path in calls_paths:
        calls.extend(load(path)["calls"])
    scope = {(call["set"], call["backend"], call["variant"], call["split"]) for call in calls} if partial else None
    expected = expected_calls(data_dir, scope)

    recorded: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for call in calls:
        if call["id"] in recorded:
            duplicates.append(call["id"])
            continue
        recorded[call["id"]] = call
    missing = sorted(set(expected) - set(recorded))
    unexpected = sorted(set(recorded) - set(expected))

    mismatched: list[str] = []
    matched = 0
    for identifier in sorted(set(expected) & set(recorded)):
        set_name, backend, variant, requests = expected[identifier]
        call = recorded[identifier]
        if (call["set"], call["backend"], call["variant"]) != (set_name, backend, variant):
            mismatched.append(f"{identifier}: labelled {call['set']}/{call['backend']}/{call['variant']}")
        elif call["model"] != model_of(backend):
            mismatched.append(f"{identifier}: model {call['model']} is not {model_of(backend)}")
        elif digested(call["requests"]) != digested(requests):
            mismatched.append(f"{identifier}: rebuilt requests differ from the recorded ones")
        else:
            matched += 1

    print(f"{matched} of {len(calls)} recorded calls rebuilt from the inputs and matched")
    print(f"{len(expected)} calls expected{' within the recorded scope' if partial else ''}, {len(recorded)} recorded")
    for identifier in missing:
        print(f"MISSING {identifier}", file=sys.stderr)
    for identifier in duplicates:
        print(f"DUPLICATE {identifier}", file=sys.stderr)
    for identifier in unexpected:
        print(f"UNEXPECTED {identifier}", file=sys.stderr)
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    return 1 if (missing or duplicates or unexpected or mismatched) else 0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Where the inputs come from, and the terms they are used under. Written into
# manifest.json, which travels with the published evidence.
INPUT_SOURCES = {
    VULNERABILITY_TRIAGE: {
        "source": "National Vulnerability Database (NVD), CVE API 2.0, https://nvd.nist.gov/developers/vulnerabilities",
        "what": "CVE descriptions and NVD analyst CWE and CVSS v3.1 assessments, as the case file records them",
        "nvd_notice": "This product uses data from the NVD API but is not endorsed or certified by the NVD.",
        "cve_copyright": (
            "Copyright (c) 1999-2026, The MITRE Corporation. CVE is a trademark and the CVE logo is a "
            "registered trademark of The MITRE Corporation."
        ),
        "cve_license": (
            "MITRE hereby grants you a perpetual, worldwide, non-exclusive, no-charge, royalty-free, "
            "irrevocable copyright license to reproduce, prepare derivative works of, publicly display, "
            "publicly perform, sublicense, and distribute Common Vulnerabilities and Exposures (CVE). Any "
            "copy you make for such purposes is authorized provided that you reproduce MITRE's copyright "
            "designation and this license in any such copy. (CVE Program Terms of Use, https://www.cve.org/Legal/TermsOfUse)"
        ),
    },
    WORKFLOWS: {
        "source": "LocalLLaMA/typed-decisions on Hugging Face, https://huggingface.co/datasets/LocalLLaMA/typed-decisions",
        "revision": "c76749ec58bd8c3d2ea706b31c333a9059c38f90",
        "license": "Apache-2.0, https://www.apache.org/licenses/LICENSE-2.0",
        "changes": (
            "400 of the test split's rows (100 per workflow, in salted-hash order) were reformatted into "
            "cases.json: each row's state, questions and gold parsed from JSON strings; no text was changed"
        ),
    },
}


def merge(
    data_dir: Path,
    paths: list[Path],
    out_dir: Path,
    server_commits: list[str] | None = None,
    attach: list[str] | None = None,
    hardware: str | None = None,
) -> int:
    """Combine per-backend recordings into one compact calls.json and a manifest that pins every file.

    calls.json keeps every response as recorded. Each request body is stored as
    its SHA-256 (`body_sha256`): the body is rebuilt from the inputs by
    `--check`, which compares digests, and `--show` prints it. `attach` names
    files already in `out_dir` (a scorer's summary, say) to pin as well.
    """
    recordings = [load(path) for path in paths]
    default_commit = (server_commits or [None])[0]
    calls = [
        {**call, "requests": digested(call["requests"])} for recording in recordings for call in recording["calls"]
    ]
    ids = [call["id"] for call in calls]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate call ids across the recordings; refusing to merge")
    incomplete = [str(path) for path, recording in zip(paths, recordings) if not recording.get("complete")]
    out_dir.mkdir(parents=True, exist_ok=True)
    calls_path = out_dir / "calls.json"
    calls_path.write_text(
        json.dumps(
            {
                "task": TASK,
                "call_count": len(calls),
                "failed_calls": sum(recording["failed_calls"] for recording in recordings),
                "complete": not incomplete,
                "recordings": [
                    {
                        "set": recording["set"],
                        "backend": recording["backend"],
                        **recording["recorded"],
                        "server_commit": recording["recorded"].get("server_commit") or default_commit,
                    }
                    for recording in recordings
                ],
                "calls": calls,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    inputs = {
        f"inputs/{set_name}/cases.json": sha256_file(data_dir / f"inputs/{set_name}/cases.json") for set_name in PLAN
    }
    attached = {name: sha256_file(out_dir / name) for name in attach or []}
    files = {**inputs, "calls.json": sha256_file(calls_path), **attached}
    calls_by_set: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for call in calls:
        calls_by_set[call["set"]] = calls_by_set.get(call["set"], 0) + 1
        statuses[call["status"]] = statuses.get(call["status"], 0) + 1
    manifest = {
        "task": TASK,
        "dataset": "superlinked/sie-task-evidence",
        "call_count": len(calls),
        "calls_by_set": calls_by_set,
        "statuses": statuses,
        "calls_sha256": files["calls.json"],
        "files_sha256": files,
        "endpoint": sorted({recording["recorded"]["runner"].get("endpoint") for recording in recordings} - {None}),
        "endpoint_note": "A self-hosted SIE server, with the client on the same machine, one record at a time.",
        # The SIE server commits the recordings were made against; calls.json
        # names each recording's own.
        "server_commits": sorted(
            {recording["recorded"].get("server_commit") or default_commit for recording in recordings} - {None}
        ),
        "hardware": hardware,
        "models": sorted({call["model"] for call in calls}),
        "run_dates": sorted({recording["recorded"]["at"][:10] for recording in recordings}),
        "plan": {
            set_name: {b: {split: list(v) for split, v in splits.items()} for b, splits in backends.items()}
            for set_name, backends in PLAN.items()
        },
        "tuning_sha256": sha256_file(tuning.TUNING_PATH) if tuning.TUNING_PATH.exists() else None,
        "input_sources": INPUT_SOURCES,
        "request_bodies": "each request body is stored as body_sha256, the SHA-256 of its canonical JSON; run.py --check rebuilds and compares them",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {calls_path} ({len(calls)} calls) and {out_dir / 'manifest.json'}")
    for path in incomplete:
        print(f"note: {path} is not a complete recording", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="directory holding inputs/ (default: data)")
    parser.add_argument("--check", action="store_true", help="offline check, the default")
    parser.add_argument("--calls", nargs="+", default=["data/calls.json"], help="recordings --check reads")
    parser.add_argument("--partial", action="store_true", help="check only the sets and backends recorded")
    parser.add_argument("--show", nargs=4, metavar=("SET", "CASE", "BACKEND", "VARIANT"), help="print requests")
    parser.add_argument("--record", action="store_true", help="make live calls")
    parser.add_argument("--merge", nargs="+", metavar="RECORDING", help="combine recordings into calls.json")
    parser.add_argument("--merge-out", default="run-output", help="where --merge writes (default: run-output)")
    parser.add_argument(
        "--server-commit",
        nargs="+",
        help="with --record: the SIE commit the server runs, stamped on the recording; "
        "with --merge: the commit for recordings that carry no stamp",
    )
    parser.add_argument("--hardware", help="with --merge: what the server ran on, for the manifest")
    parser.add_argument("--attach", nargs="+", help="with --merge: files in --merge-out to pin in the manifest")
    parser.add_argument("--set", choices=sorted(PLAN), default=VULNERABILITY_TRIAGE)
    parser.add_argument("--backend", choices=sorted({*LAYA_PACKAGE_CHECKPOINTS, *SIE_MODELS}), default="laya")
    parser.add_argument("--variant", nargs="+", choices=VARIANT_CHOICES, help="record these variants, not the plan's")
    parser.add_argument("--split", choices=("dev", "test"), help="record one split only")
    parser.add_argument("--limit", type=int, help="record the first N cases only (marks the recording incomplete)")
    parser.add_argument("--url", default=os.environ.get("SIE_BASE_URL", DEFAULT_SIE_URL), help="SIE endpoint")
    parser.add_argument("--device", default="cpu", help="device for the reference Laya package (default: cpu)")
    parser.add_argument("--out", help="where --record writes (default: run-output/<set>--<backend>.json)")
    args = parser.parse_args()
    data_dir = Path(args.data)

    if args.show:
        set_name, slug, backend, variant = args.show
        cases = case_file(data_dir, set_name)
        case = next((item for item in cases["cases"] if item["slug"] == slug), None)
        if case is None:
            raise SystemExit(f"no case {slug} in {set_name}")
        requests = requests_for(
            backend,
            tuning.phrasing(set_name, backend, variant),
            case["state"],
            asked(set_name, backend, questions_for(cases, case)),
        )
        print(json.dumps(requests, indent=2, ensure_ascii=False))
        return 0
    if args.merge:
        return merge(
            data_dir,
            [Path(path) for path in args.merge],
            Path(args.merge_out),
            args.server_commit,
            args.attach,
            args.hardware,
        )
    if args.record:
        return record(args)
    return check(data_dir, [Path(path) for path in args.calls], args.partial)


if __name__ == "__main__":
    sys.exit(main())
