{{/*
NATS authentication (nats.auth).

sie-config, the gateway and the worker sidecars each connect as their own NATS
user, and the bundled server limits each user to the subjects that component
uses. Passwords live in one Secret per component, holding a `password` key.
The same names are computed here and, through `$tplYamlSpread` in the
`nats:` values, inside the NATS sub-chart, so both sides must build them from
values both can see: nats.auth and the release name.

The "sie-cluster.nats.server.*" templates run in the NATS sub-chart's context,
where .Values is the `nats:` block (so nats.auth is .Values.auth).
*/}}

{{- define "sie-cluster.nats.authComponents" -}}
{{- toJson (list "config" "gateway" "worker") -}}
{{- end }}

{{/*
"true" when NATS connections authenticate: nats.enabled, and nats.auth.enabled
is not false. An absent key counts as true, so `helm upgrade --reuse-values`
from a release that predates it keeps the default.
*/}}
{{- define "sie-cluster.nats.authEnabled" -}}
{{- if and .Values.nats.enabled (dig "auth" "enabled" true .Values.nats) -}}
true
{{- end -}}
{{- end }}

{{/*
NATS user of a component. Args: component name.
*/}}
{{- define "sie-cluster.nats.authUser" -}}
{{- printf "sie-%s" . -}}
{{- end }}

{{/*
Secret holding a component's NATS password: nats.auth.existingSecrets.<component>
when set, otherwise the chart-generated "<release>-nats-auth-<component>".
Args (dict): auth (nats.auth), release (release name), component.
*/}}
{{- define "sie-cluster.nats.authSecretName" -}}
{{- $existing := dig "existingSecrets" .component "" (.auth | default dict) -}}
{{- if $existing -}}
{{- $existing -}}
{{- else -}}
{{- printf "%s-nats-auth-%s" .release .component -}}
{{- end -}}
{{- end }}

{{/*
SIE_NATS_USER / SIE_NATS_PASSWORD for a client container, or nothing when NATS
authentication is off. Args (dict): root (top-level context), component.
*/}}
{{- define "sie-cluster.nats.clientAuthEnv" -}}
{{- if include "sie-cluster.nats.authEnabled" .root -}}
- name: SIE_NATS_USER
  value: {{ include "sie-cluster.nats.authUser" .component | quote }}
- name: SIE_NATS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "sie-cluster.nats.authSecretName" (dict "auth" .root.Values.nats.auth "release" .root.Release.Name "component" .component) }}
      key: password
{{- end -}}
{{- end }}

{{/*
Fail when a reused generated NATS password Secret has no usable password,
instead of replacing a password that running pods hold. The server reads the
password into its configuration and cluster route URLs, so it must be letters
and digits only. Args (dict): name (Secret), data (base64 value, empty when
the key is missing).
*/}}
{{- define "sie-cluster.nats.validateReusedPassword" -}}
{{- if not .data -}}
{{- fail (printf "Secret %s exists but has no password key. If it was created by hand, set nats.auth.existingSecrets to it so the chart uses it unchanged. Otherwise delete the Secret so the chart generates a new password, then restart NATS, sie-config, the gateway, and the workers." .name) -}}
{{- end -}}
{{- $password := b64dec .data -}}
{{- if lt (len $password) 32 -}}
{{- fail (printf "Secret %s holds a password shorter than 32 characters. Replace it with a random value of at least 32 letters and digits, or delete the Secret so the chart generates a new one, then restart NATS, sie-config, the gateway, and the workers." .name) -}}
{{- end -}}
{{- if not (regexMatch "^[A-Za-z0-9]+$" $password) -}}
{{- fail (printf "Secret %s holds a password with characters other than letters and digits. The NATS server reads it into its configuration and route URLs, where other characters break parsing. Replace it, or delete the Secret so the chart generates a new one." .name) -}}
{{- end -}}
{{- end }}

{{/*
Validate the NATS authentication settings.

For the bundled server, the NATS sub-chart values must still contain the
chart's wiring (the `sieNatsAuth` keys). `helm upgrade --reuse-values` from a
release that predates NATS authentication replaces the chart's default values
with the old ones, which would leave the server without authentication while
the clients send credentials. Fail instead.

For an external server, every client that connects needs an operator Secret.
*/}}
{{- define "sie-cluster.nats.validateAuth" -}}
{{- if include "sie-cluster.nats.authEnabled" . -}}
{{- $auth := .Values.nats.auth | default dict -}}
{{- if .Values.nats.install -}}
{{- $wired := and (hasKey (dig "config" "merge" (dict) .Values.nats) "sieNatsAuth") (hasKey (dig "container" "env" (dict) .Values.nats) "sieNatsAuth") -}}
{{- if and $wired (dig "config" "cluster" "enabled" false .Values.nats) -}}
{{- $wired = hasKey (dig "config" "cluster" "merge" (dict) .Values.nats) "sieNatsAuth" -}}
{{- end -}}
{{- if not $wired -}}
{{- fail "NATS authentication is on (nats.auth.enabled), but the NATS values of this release lack the chart's server wiring (the sieNatsAuth keys under nats.config.merge, nats.config.cluster.merge, and nats.container.env). This happens with `helm upgrade --reuse-values` from a release that predates NATS authentication. Upgrade with --reset-then-reuse-values (Helm 3.14 or later) or pass your values with -f instead, and do not remove those keys. To keep NATS unauthenticated, set nats.auth.enabled=false." -}}
{{- end -}}
{{- else -}}
{{- if dig "allowAnonymous" false $auth -}}
{{- fail "nats.auth.allowAnonymous applies only to the bundled NATS server (nats.install=true). Configure anonymous access on the external server instead." -}}
{{- end -}}
{{- $needed := list "gateway" -}}
{{- if .Values.config.enabled -}}
{{- $needed = append $needed "config" -}}
{{- end -}}
{{- if .Values.workers.common.workerSidecar.enabled -}}
{{- $needed = append $needed "worker" -}}
{{- end -}}
{{- range $component := $needed -}}
{{- if not (dig "existingSecrets" $component "" $auth) -}}
{{- fail (printf "nats.install=false with nats.auth.enabled=true needs nats.auth.existingSecrets.%s: a Secret holding the %s NATS password under the key password, for the user %s on the external server. Set nats.auth.enabled=false if the external server does not require authentication." $component $component (include "sie-cluster.nats.authUser" $component)) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}

{{/*
NATS container env for the bundled server (NATS sub-chart context): one
password per user, read from the component Secrets, and, when clustered, the
route password and the route URLs that embed it. Kubernetes expands
$(SIE_NATS_AUTH_ROUTE_PASSWORD) because the sub-chart renders env vars in
name order and that name sorts first.
*/}}
{{- define "sie-cluster.nats.server.env" -}}
{{- $auth := .Values.auth | default dict -}}
{{- if and .Values.enabled (dig "enabled" true $auth) -}}
{{- $components := include "sie-cluster.nats.authComponents" . | fromJsonArray -}}
{{- if .Values.config.cluster.enabled -}}
{{- $components = append $components "route" -}}
{{- end -}}
{{- range $component := $components }}
{{ printf "SIE_NATS_AUTH_%s_PASSWORD" (upper $component) }}:
  valueFrom:
    secretKeyRef:
      name: {{ include "sie-cluster.nats.authSecretName" (dict "auth" $auth "release" $.Release.Name "component" $component) }}
      key: password
{{- end }}
{{- if .Values.config.cluster.enabled }}
SIE_NATS_ROUTES: {{ include "sie-cluster.nats.server.routeUrls" . | quote }}
{{- end }}
{{- end -}}
{{- end }}

{{/*
Cluster route URLs with the route user's credentials, matching the hosts the
NATS sub-chart renders in files/config/cluster.yaml.
*/}}
{{- define "sie-cluster.nats.server.routeUrls" -}}
{{- $urls := list -}}
{{- with .Values.config.cluster -}}
{{- $domain := $.Values.headlessService.name -}}
{{- if .routeURLs.useFQDN -}}
{{- $domain = printf "%s.%s.svc.%s" $domain (include "nats.namespace" $) .routeURLs.k8sClusterDomain -}}
{{- end -}}
{{- $proto := ternary "tls" "nats" .tls.enabled -}}
{{- $port := int .port -}}
{{- range $i, $_ := until (int .replicas) -}}
{{- $urls = append $urls (printf "%s://%s:$(SIE_NATS_AUTH_ROUTE_PASSWORD)@%s-%d.%s:%d" $proto (include "sie-cluster.nats.authUser" "route") $.Values.statefulSet.name $i $domain $port) -}}
{{- end -}}
{{- end -}}
{{- printf "[%s]" (join "," $urls) -}}
{{- end }}

{{/*
Top-level NATS server configuration (NATS sub-chart context): the users and
their permissions. Subjects are listed in the chart README ("NATS
authentication"). Passwords are environment references, unquoted by the
sub-chart's << >> rule.
*/}}
{{- define "sie-cluster.nats.server.configMerge" -}}
{{- $auth := .Values.auth | default dict -}}
{{- if and .Values.enabled (dig "enabled" true $auth) -}}
authorization:
  users:
    - user: {{ include "sie-cluster.nats.authUser" "config" }}
      password: "<< $SIE_NATS_AUTH_CONFIG_PASSWORD >>"
      permissions:
        publish:
          allow:
            - "sie.config.models.>"
        subscribe:
          deny:
            - ">"
    - user: {{ include "sie-cluster.nats.authUser" "gateway" }}
      password: "<< $SIE_NATS_AUTH_GATEWAY_PASSWORD >>"
      permissions:
        publish:
          allow:
            - "sie.work.>"
            - "sie.dlq.>"
            - "cancel.>"
            - "work_cancel.>"
            - "batch_cancel.>"
            - "$JS.API.STREAM.INFO.*"
            - "$JS.API.STREAM.CREATE.*"
            - "$JS.API.STREAM.UPDATE.*"
            - "$JS.API.CONSUMER.INFO.*.*"
        subscribe:
          allow:
            - "sie.config.models._all"
            - "sie.health.>"
            - "_INBOX.>"
            - "$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>"
    - user: {{ include "sie-cluster.nats.authUser" "worker" }}
      password: "<< $SIE_NATS_AUTH_WORKER_PASSWORD >>"
      permissions:
        publish:
          allow:
            - "_INBOX.>"
            - "sie.health.>"
            - "$JS.ACK.>"
            - "$JS.API.STREAM.INFO.*"
            - "$JS.API.STREAM.CREATE.*"
            - "$JS.API.STREAM.UPDATE.*"
            - "$JS.API.CONSUMER.LIST.*"
            - "$JS.API.CONSUMER.INFO.*.*"
            - "$JS.API.CONSUMER.CREATE.*.>"
            - "$JS.API.CONSUMER.DELETE.*.*"
            - "$JS.API.CONSUMER.MSG.NEXT.*.*"
        subscribe:
          allow:
            - "sie.config.models.*"
            - "cancel.>"
            - "work_cancel.>"
            - "batch_cancel.>"
            - "_INBOX_WORKER.>"
    {{- if dig "allowAnonymous" false $auth }}
    - user: sie-anonymous
    {{- end }}
{{- if dig "allowAnonymous" false $auth }}
no_auth_user: sie-anonymous
{{- end }}
{{- end -}}
{{- end }}

{{/*
Cluster block additions (NATS sub-chart context): routes authenticate as the
route user, and the route URLs carry its password.
*/}}
{{- define "sie-cluster.nats.server.clusterMerge" -}}
{{- $auth := .Values.auth | default dict -}}
{{- if and .Values.enabled (dig "enabled" true $auth) .Values.config.cluster.enabled -}}
authorization:
  user: {{ include "sie-cluster.nats.authUser" "route" }}
  password: "<< $SIE_NATS_AUTH_ROUTE_PASSWORD >>"
routes: "<< $SIE_NATS_ROUTES >>"
{{- end -}}
{{- end }}
