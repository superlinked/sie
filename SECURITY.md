# Security Policy

## Reporting a vulnerability

Report vulnerabilities privately. Do not open a public issue, pull request or
discussion for them.

- Use GitHub private vulnerability reporting:
  [report a vulnerability](https://github.com/superlinked/sie/security/advisories/new).
- If you cannot use it, email support@superlinked.com with "SIE security
  report" in the subject. Leave the details out of that first message: we reply
  with a private channel for them.

Include:

- the affected component: `sie-server`, the gateway, `sie-config`, `sie-mcp`, the
  Helm chart, an SDK or an integration;
- the version, image tag or commit;
- how to reproduce it, and the impact you observed.

We confirm receipt, keep you informed while we work on a fix, and credit you in
the advisory unless you ask us not to.

## Supported versions

| Version | Security fixes |
| --- | --- |
| The latest release | Yes |
| Older releases | No. Upgrade to the latest release. |

Fixes land on `main` and ship in the next release. There are no maintenance
branches. Advisories name the first fixed release.

## Deploying safely

What a default install exposes, and how to close it.

### Single server (`sie-server serve`, pip or Docker)

- By default the server binds every interface (`0.0.0.0:8080`) and accepts
  requests without authentication. Anyone who can reach the port can run
  inference, read `/ws/status`, and make the server load catalog models, which
  costs compute, disk and network.
- Turn on token authentication with `SIE_AUTH_MODE=token`, and set the token
  with `SIE_AUTH_TOKEN` (or comma-separated `SIE_AUTH_TOKENS`). Without a
  configured token, the server generates one and keeps it in `SIE_DATA_DIR`.
  Clients send it as `Authorization: Bearer <token>` or `x-api-key: <token>`;
  when both carry a token, `Authorization` is used. `/healthz`, `/livez` and
  `/readyz` stay open. Token authentication covers the HTTP port; the IPC
  socket and the engines' ports listen on this machine only.
- Keep it on this machine with `--host 127.0.0.1`. In Docker, publish the port
  on loopback only: `-p 127.0.0.1:8080:8080`.
- A loopback server without a token is still exposed to a web page that uses
  DNS rebinding. Until 0.13.0 the server only logs a warning for a `Host` that
  does not name this machine. Set a token if a browser on this machine may
  visit untrusted pages.
- Do not expose an unauthenticated server to a network you do not control.
  Turn on token authentication, or expose it through an authenticating proxy
  or through the gateway with token authentication enabled.

### Gateway

- The gateway authenticates requests with static tokens: set
  `SIE_AUTH_MODE=token` and `SIE_AUTH_TOKENS`. Mutations of pools, admin and
  config routes also need `SIE_ADMIN_TOKEN`.
- Clients send a token as `Authorization: Bearer <token>` or
  `x-api-key: <token>`. The first `Authorization` value decides when it
  carries a token, and `x-api-key` is then ignored. When that value is empty
  or blank, `x-api-key` decides. An unreadable first value is refused, and
  `Bearer` alone counts as the token `Bearer`. Later `Authorization` values
  are ignored. Until 0.13.0, a repeated `Authorization` header whose values
  carry different tokens, include one the gateway cannot read, or carry a
  token only after the first value logs a warning; from 0.13.0 it is refused.
- `/healthz`, `/readyz`, `/openapi.json` and `/docs` are always reachable.
  `/`, `/health` and `/ws/*` require a token unless
  `SIE_AUTH_EXEMPT_OPERATIONAL=true`.
- Tokens are compared in constant time. An unknown `SIE_AUTH_MODE`, or tokens
  set while auth is off, makes the gateway refuse every non-probe request.

### Helm chart

The chart README covers each of these in detail:

- The chart refuses to render an Ingress, `LoadBalancer` or `NodePort` gateway
  without authentication, unless you allow it explicitly.
- `sie-config` admin and read tokens are generated and kept across upgrades.
- NATS connections are authenticated per component by default.
- `workers.networkPolicy.enabled=true` admits only the gateway to the worker
  ports. Without it, any pod in the cluster can call a worker directly.
- Gateway token authentication (`gateway.auth`) and single sign-on (`auth`)
  are off by default. Turn one of them on before you expose the gateway.

### Telemetry

`sie-server` sends anonymous usage telemetry. The [README](README.md#telemetry)
lists every field and how to turn it off.

## Scheduled changes

Release 0.13.0 makes these defaults secure: a loopback bind for
`sie-server serve`, a required token for any exposed server, generated tokens
and a worker NetworkPolicy in the Helm chart, and telemetry off unless enabled.
Earlier releases add the options and log warnings first.
[COMPATIBILITY.md](COMPATIBILITY.md#scheduled-changes) has the details and the
migration steps.
