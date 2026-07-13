"""Known-legitimate "unused" code, per vulture's whitelist convention.

Referenced by name only, never imported or executed: this file exists to be
scanned by vulture (`vulture src/ tests/ agent-demo/ mock-prod/
scripts/vulture_whitelist.py`), not to run. Every entry here was checked
against a real cross-reference search before being added.

- ActionGate.evaluate_all(): batch evaluation, used by the root repo's
  offline `dusk gate` CLI (a separate package -- this example's live
  /v1/gate handler processes one action per request via .evaluate(), so it
  never needs the batch method). Kept for parity with the root copy's
  public API, not dead code that was meant to be removed.
- config.py's set_config(): a real public setter with no caller in this
  package yet (tests use monkeypatch/env vars instead), kept for parity
  with get_config()/reset_config() as the documented override path.
- vector.py's SimilarDecision.similarity: set at construction (the
  cosine/rerank score each match was found at) but never read back by
  api.py, which only forwards each match's .id in similar_decision_ids.
  Kept on the dataclass as diagnostic data for a caller that wants the
  actual similarity score, not just which decisions matched.
"""

from dusk.actions.verdict import ActionGate
from dusk.config import set_config
from dusk.trace.vector import SimilarDecision

ActionGate.evaluate_all
set_config
SimilarDecision.similarity
