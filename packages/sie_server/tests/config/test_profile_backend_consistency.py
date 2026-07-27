# Release-gate guard: production Qwen3.5 profiles ship with
# ``grammar_backend: outlines`` (dottxt partnership; the codebase default).
# This test keeps the YAML aligned with that backend-of-record.

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SIE_SERVER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SIE_SERVER_ROOT.parents[1]
QWEN35_PROFILE = SIE_SERVER_ROOT / "models" / "Qwen__Qwen3.5-4B.yaml"
STATUS_DOC = REPO_ROOT / "product" / "research" / "generation-primitive-status.md"

EXPECTED_BACKEND = "outlines"


def _iter_profile_backends(profile_path: Path) -> list[tuple[str, str]]:
    """Return ``[(profile_name, grammar_backend), ...]`` for every profile in the file.

    The Qwen3.5 model YAML is a single top-level mapping with a ``profiles``
    dict. Each profile carries ``adapter_options.loadtime.grammar_backend``.
    """
    data = yaml.safe_load(profile_path.read_text()) or {}
    if not isinstance(data, dict):
        raise AssertionError(f"Expected {profile_path.name} to be a YAML mapping, got {type(data).__name__}")
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise AssertionError(
            f"Expected 'profiles' in {profile_path.name} to be a mapping, got {type(profiles).__name__}"
        )
    out: list[tuple[str, str]] = []
    for name, entry in profiles.items():
        if not isinstance(entry, dict):
            continue
        loadtime = (entry.get("adapter_options") or {}).get("loadtime") or {}
        backend = loadtime.get("grammar_backend")
        if backend is not None:
            out.append((str(name), str(backend)))
    return out


def test_qwen35_profiles_all_pin_outlines() -> None:
    """Every Qwen3.5 profile must set ``grammar_backend: outlines``.

    Outlines is the backend of record (dottxt partnership) for Qwen3.5. If a
    profile drifts to ``xgrammar`` or ``llguidance`` this test fires so the
    status doc gets updated alongside.
    """
    backends = _iter_profile_backends(QWEN35_PROFILE)
    assert backends, f"No profiles with grammar_backend found in {QWEN35_PROFILE}"
    mismatches = [(name, value) for name, value in backends if value != EXPECTED_BACKEND]
    assert not mismatches, (
        f"Qwen3.5 profile(s) drifted off the backend of record "
        f"(expected '{EXPECTED_BACKEND}'): {mismatches}. "
        f"If this is intentional, update both the YAML and the §4.2 Decision "
        f"record in {STATUS_DOC.relative_to(REPO_ROOT)}."
    )


def test_status_doc_names_outlines_as_backend_of_record() -> None:
    """The status doc must explicitly call Outlines the backend of record.

    The decision record in §4.2 is the single source of truth; this test
    guards against silent regressions where someone edits the YAML but not
    the doc (or vice versa).
    """
    if not STATUS_DOC.exists():
        pytest.skip(f"Status doc not present at {STATUS_DOC}")
    text = STATUS_DOC.read_text()
    # The decision record uses the exact phrase "backend of record" alongside
    # "Outlines". Both must appear; the doc must not say profiles are "pinned"
    # to xgrammar (the contradictory framing that finding H10 reconciled).
    assert "backend of record" in text.lower(), (
        "Expected the status doc to declare a 'backend of record' for structured outputs. Did §4.2 get rewritten?"
    )
    assert "outlines" in text.lower(), "Expected the status doc to name 'outlines' explicitly."
    forbidden_phrases = [
        "pin xgrammar",
        "pinned xgrammar",
        "profiles pin xgrammar",
        "profiles currently pin",
        "xgrammar as the validated fallback",
        "until the outlines a100 smoke",
        "until the outlines smoke",
    ]
    lowered = text.lower()
    hits = [p for p in forbidden_phrases if p in lowered]
    assert not hits, (
        f"Status doc contains contradictory 'pin xgrammar' phrasing that H10 "
        f"reconciled: {hits}. Production profiles ship grammar_backend: "
        f"outlines; the doc must match the code."
    )


MODELS_DIR = SIE_SERVER_ROOT / "models"
GUARDIAN_PROFILE = MODELS_DIR / "ibm-granite__granite-guardian-3.0-2b.yaml"

# The SGLang generation adapter defaults ``grammar_backend`` to "outlines"
# (see adapters/sglang/generation.py). Only these backends actually compile
# EBNF; outlines skips/fails it.
DEFAULT_GRAMMAR_BACKEND = "outlines"
EBNF_CAPABLE_BACKENDS = {"xgrammar", "llguidance"}


def _grammar_capabilities(data: dict) -> list[str]:
    caps = ((data.get("tasks") or {}).get("generate") or {}).get("capabilities") or {}
    grammar = caps.get("grammar") or []
    return [str(k) for k in grammar] if isinstance(grammar, list) else []


def test_advertised_ebnf_requires_capable_backend() -> None:
    """A model may advertise ``ebnf`` only if every profile pins an EBNF-capable backend.

    The gateway admits grammar requests purely on the advertised
    ``capabilities.grammar`` list. If a profile advertises ``ebnf`` but runs the
    default ``outlines`` backend (which cannot compile EBNF), the gateway admits
    a request the worker then fails to serve. This guards that advertise-vs-serve
    invariant across every shipped model YAML.
    """
    offenders: list[str] = []
    for path in sorted(MODELS_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict) or "ebnf" not in _grammar_capabilities(data):
            continue
        profiles = data.get("profiles") or {}
        for name, entry in profiles.items():
            if not isinstance(entry, dict):
                continue
            loadtime = (entry.get("adapter_options") or {}).get("loadtime") or {}
            backend = str(loadtime.get("grammar_backend") or DEFAULT_GRAMMAR_BACKEND)
            if backend not in EBNF_CAPABLE_BACKENDS:
                offenders.append(f"{path.name}:{name} advertises ebnf but runs grammar_backend={backend!r}")
    assert not offenders, (
        "Model(s) advertise 'ebnf' without an EBNF-capable grammar_backend "
        f"({sorted(EBNF_CAPABLE_BACKENDS)}); the gateway would admit EBNF requests "
        f"the worker cannot serve: {offenders}"
    )


def test_guardian_profiles_carry_guard_threshold() -> None:
    """Every Granite Guardian profile must carry the guard threshold load-time dial.

    Loadtime blocks are whole-dict replaced (not deep-merged) across profiles, so
    a non-extending variant that omits ``guard`` constructs with ``self._guard={}``
    and silently skips CHECK POLICY thresholding (falls back to raw argmax).
    """
    if not GUARDIAN_PROFILE.exists():
        pytest.skip(f"Guardian profile not present at {GUARDIAN_PROFILE}")
    data = yaml.safe_load(GUARDIAN_PROFILE.read_text()) or {}
    profiles = data.get("profiles") or {}
    assert profiles, f"No profiles found in {GUARDIAN_PROFILE.name}"
    missing: list[str] = []
    for name, entry in profiles.items():
        if not isinstance(entry, dict):
            continue
        loadtime = (entry.get("adapter_options") or {}).get("loadtime") or {}
        threshold = (loadtime.get("guard") or {}).get("threshold")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            missing.append(name)
    assert not missing, (
        f"Granite Guardian profile(s) missing guard.threshold load-time config: "
        f"{missing}. Non-extending profiles do not inherit it, so the variant would "
        f"skip thresholding. Duplicate the guard block into each profile's loadtime."
    )


def test_guardian_task_pins_harm_risk() -> None:
    """Granite's launch contract must not depend on a tokenizer default."""
    if not GUARDIAN_PROFILE.exists():
        pytest.skip(f"Guardian profile not present at {GUARDIAN_PROFILE}")
    data = yaml.safe_load(GUARDIAN_PROFILE.read_text()) or {}

    generate = (data.get("tasks") or {}).get("generate") or {}
    assert generate.get("chat_template_kwargs") == {
        "guardian_config": {"risk_name": "harm"},
    }
