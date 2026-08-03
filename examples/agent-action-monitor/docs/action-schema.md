# AgentAction schema

`AgentAction` is the single, canonical event the action ingest layer produces.
Whatever controller an agent used to change the network, the action is
normalised into this one shape so the rest of the pipeline never sees a
vendor-specific record.

This layer normalises only. It records what happened, not how bad it is. There
is no severity, score, blast radius, or verdict here; those belong to later
layers. This document informs the OWASP threat-model work.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `agent_id` | str | Identity of the acting agent (service account, role, or agent name as the controller reports it). Required, non-empty. |
| `timestamp` | datetime | When the action occurred. Must be timezone-aware. |
| `action_type` | str | Normalised verb: `firewall_rule_change`, `route_change`, `segment_change`, `role_assignment`, `port_change`, or `unknown`. |
| `target` | str | What was acted on (resource id, rule name, segment). Required, non-empty. |
| `change` | dict | Structured delta with keys `before` and `after`; either may be `null` for a create or delete. |
| `source` | str | Originating controller, for example `azure` or `generic`. |
| `raw_ref` | str or null | Opaque reference back to the original record (an id), never the full payload. |

Validation is strict: an empty `agent_id` or `target`, a timezone-naive
`timestamp`, or an `action_type` outside the known set is rejected with a clear
error rather than silently coerced.

## Example (generic format)

A generic record uses the canonical field names directly and is accepted by the
generic adapter. The `timestamp` may be an ISO 8601 string:

```json
{
  "agent_id": "netops-agent",
  "timestamp": "2023-11-14T22:13:20+00:00",
  "action_type": "firewall_rule_change",
  "target": "fw-corp-https",
  "change": { "before": null, "after": { "port": 443 } },
  "source": "generic",
  "raw_ref": "evt-0001"
}
```

`AgentAction.to_dict()` returns this JSON-safe shape (timestamp as an ISO 8601
string), and `AgentAction.from_dict()` reconstructs the event, so events
round-trip exactly.

## Sources and adapters

Each source has an adapter that maps its native record onto the fields above.

- `generic`: records already in the canonical shape (the path the synthetic
  generator and any not-yet-adapted source use).
- `azure`: Azure Monitor activity-log records. Operation names map to an
  action_type as follows: networkSecurityGroups or securityRules to
  `firewall_rule_change`, routeTables or routes to `route_change`,
  virtualNetworks or subnets to `segment_change`, roleAssignments to
  `role_assignment`, anything else to `unknown`. The acting identity comes from
  `caller`, the target from `resourceId`, the timestamp from `eventTimestamp`,
  the before/after delta from `properties`, and the reference from
  `correlationId` or `eventDataId`.
- `bedrock`: a proposed Bedrock Converse API tool-call, read from the model's
  response before it has been applied anywhere -- this is the seam the
  example judges. Tool names map to an action_type by substring: firewall or
  securitygroup to `firewall_rule_change`, route to `route_change`, segment,
  subnet, or vpc to `segment_change`, role or permission to
  `role_assignment`, port to `port_change`, anything else to `unknown`. The
  target and before/after delta come from the tool's input; agent_id and
  timestamp are supplied by the caller (the agent harness), not the toolUse
  block itself.

New sources are added by writing an adapter and registering it; ingest itself
does not change.
