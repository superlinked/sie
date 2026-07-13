"""Generate synthetic agent action fixtures (generic JSON format).

Writes two JSON files, each a list of raw generic-format records (the shape
the generic adapter accepts):

- ``tests/fixtures/actions_normal.json``: routine, in-scope control-plane
  work from a few agents.
- ``tests/fixtures/actions_mixed.json``: the same routine work plus a few
  clearly out-of-pattern actions (an agent that only ever edits the corporate
  segment suddenly opening a guest-to-restricted firewall rule).

These exercise ingestion only. This layer assigns no severity; later layers
decide whether the out-of-pattern actions matter.

Run directly to (re)generate both fixtures::

    python lab/actions/generate_actions.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

BASE_TIME = datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures")


def _action(
    offset: int,
    agent_id: str,
    action_type: str,
    target: str,
    before: Any = None,
    after: Any = None,
) -> dict[str, Any]:
    """Build one generic-format record with an evenly spaced timestamp."""
    return {
        "agent_id": agent_id,
        "timestamp": (BASE_TIME + timedelta(minutes=offset)).isoformat(),
        "action_type": action_type,
        "target": target,
        "change": {"before": before, "after": after},
        "source": "generic",
        "raw_ref": f"evt-{offset:04d}",
    }


def normal_actions() -> list[dict[str, Any]]:
    """Routine, in-scope actions from three agents."""
    return [
        _action(0, "netops-agent", "firewall_rule_change", "fw-corp-https", after={"port": 443}),
        _action(1, "netops-agent", "firewall_rule_change", "fw-corp-dns", after={"port": 53}),
        _action(
            2, "netops-agent", "route_change", "rt-corp-to-dmz", after={"next_hop": "10.0.0.1"}
        ),
        _action(3, "netops-agent", "route_change", "rt-corp-default"),
        _action(4, "netops-agent", "firewall_rule_change", "fw-temp-debug", before={"port": 22}),
        _action(
            5, "segment-agent", "segment_change", "seg-corporate", after={"cidr": "10.0.10.0/24"}
        ),
        _action(
            6, "segment-agent", "segment_change", "seg-corporate", after={"cidr": "10.0.10.0/23"}
        ),
        _action(7, "segment-agent", "segment_change", "seg-corp-staging"),
        _action(8, "iam-agent", "role_assignment", "ra-netops-reader", after={"role": "reader"}),
        _action(9, "iam-agent", "role_assignment", "ra-segment-contributor"),
        _action(10, "netops-agent", "firewall_rule_change", "fw-corp-https", after={"port": 443}),
        _action(11, "netops-agent", "route_change", "rt-corp-to-monitoring"),
        _action(12, "segment-agent", "segment_change", "seg-corporate"),
        _action(13, "iam-agent", "role_assignment", "ra-stale-temp", before={"role": "reader"}),
        _action(14, "netops-agent", "port_change", "fw-corp-dns", after={"port": 53}),
    ]


def out_of_pattern_actions() -> list[dict[str, Any]]:
    """Actions that break each agent's established pattern."""
    return [
        # The segment agent has only ever touched segments; now it opens a
        # firewall path from guest into the restricted segment.
        _action(
            15,
            "segment-agent",
            "firewall_rule_change",
            "fw-guest-to-restricted",
            after={"src": "10.0.40.0/24", "dst": "10.0.99.0/24", "port": 445},
        ),
        # The IAM agent grants itself a broad owner role.
        _action(16, "iam-agent", "role_assignment", "ra-iam-owner-self", after={"role": "owner"}),
        # The netops agent reassigns a restricted segment it never manages.
        _action(17, "netops-agent", "segment_change", "seg-restricted"),
    ]


def generate(directory: str = FIXTURE_DIR) -> tuple[str, str]:
    """Write both fixtures into ``directory`` and return their paths."""
    os.makedirs(os.path.abspath(directory), exist_ok=True)
    normal = normal_actions()
    mixed = normal + out_of_pattern_actions()

    normal_path = os.path.join(directory, "actions_normal.json")
    mixed_path = os.path.join(directory, "actions_mixed.json")
    with open(normal_path, "w", encoding="utf-8") as handle:
        json.dump(normal, handle, indent=2)
    with open(mixed_path, "w", encoding="utf-8") as handle:
        json.dump(mixed, handle, indent=2)
    return os.path.abspath(normal_path), os.path.abspath(mixed_path)


def main() -> int:
    try:
        normal_path, mixed_path = generate()
    except OSError as exc:
        print(f"Failed to write fixtures: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {normal_path}")
    print(f"Wrote {mixed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
