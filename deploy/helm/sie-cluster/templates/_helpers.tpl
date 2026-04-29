{{/*
Expand the name of the chart.
*/}}
{{- define "sie-cluster.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "sie-cluster.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "sie-cluster.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "sie-cluster.labels" -}}
helm.sh/chart: {{ include "sie-cluster.chart" . }}
{{ include "sie-cluster.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "sie-cluster.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sie-cluster.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: sie
{{- end }}

{{/*
Gateway labels
*/}}
{{- define "sie-cluster.gateway.labels" -}}
{{ include "sie-cluster.labels" . }}
app.kubernetes.io/component: gateway
{{- end }}

{{/*
Gateway selector labels
*/}}
{{- define "sie-cluster.gateway.selectorLabels" -}}
{{ include "sie-cluster.selectorLabels" . }}
app.kubernetes.io/component: gateway
{{- end }}

{{/*
Config service labels
*/}}
{{- define "sie-cluster.config.labels" -}}
{{ include "sie-cluster.labels" . }}
app.kubernetes.io/component: config
{{- end }}

{{/*
Config service selector labels
*/}}
{{- define "sie-cluster.config.selectorLabels" -}}
{{ include "sie-cluster.selectorLabels" . }}
app.kubernetes.io/component: config
{{- end }}

{{/*
Worker labels
*/}}
{{- define "sie-cluster.worker.labels" -}}
{{ include "sie-cluster.labels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Worker selector labels for a specific pool
*/}}
{{- define "sie-cluster.worker.selectorLabels" -}}
{{ include "sie-cluster.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "sie-cluster.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "sie-cluster.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Namespace to use
*/}}
{{- define "sie-cluster.namespace" -}}
{{- default .Release.Namespace .Values.global.namespace }}
{{- end }}

{{/*
Gateway image
*/}}
{{- define "sie-cluster.gateway.image" -}}
{{- $tag := default .Chart.AppVersion .Values.gateway.image.tag }}
{{- printf "%s:%s" .Values.gateway.image.repository $tag }}
{{- end }}

{{/*
Config service image
*/}}
{{- define "sie-cluster.config.image" -}}
{{- $tag := default .Chart.AppVersion .Values.config.image.tag }}
{{- printf "%s:%s" .Values.config.image.repository $tag }}
{{- end }}

{{/*
Config service resource name (Deployment / Service)
*/}}
{{- define "sie-cluster.config.serviceName" -}}
{{- $fullname := include "sie-cluster.fullname" . }}
{{- printf "%s-config" $fullname }}
{{- end }}

{{/*
In-cluster URL used by the gateway to reach the config service for the
bootstrap GET /v1/configs/export call and the periodic GET /v1/configs/epoch
drift poll. Built from the Helm-owned Service name and port so it stays
correct on overlays.
*/}}
{{- define "sie-cluster.config.internalUrl" -}}
{{- $svc := include "sie-cluster.config.serviceName" . }}
{{- $ns := include "sie-cluster.namespace" . }}
{{- $port := .Values.config.service.port | default 8080 }}
{{- printf "http://%s.%s.svc.cluster.local:%v" $svc $ns $port }}
{{- end }}

{{/*
Worker StatefulSet name for a pool
*/}}
{{- define "sie-cluster.worker.name" -}}
{{- $fullname := include "sie-cluster.fullname" .root }}
{{- printf "%s-worker-%s" $fullname .poolName }}
{{- end }}

{{/*
Worker Service name (headless service for StatefulSet)
*/}}
{{- define "sie-cluster.worker.serviceName" -}}
{{- $fullname := include "sie-cluster.fullname" . }}
{{- printf "%s-worker" $fullname }}
{{- end }}

{{/*
Gateway service name (used for worker discovery)
*/}}
{{- define "sie-cluster.gateway.serviceName" -}}
{{- $fullname := include "sie-cluster.fullname" . }}
{{- printf "%s-gateway" $fullname }}
{{- end }}

{{/*
OAuth2 proxy service name
*/}}
{{- define "sie-cluster.oauth2Proxy.serviceName" -}}
{{- $fullname := include "sie-cluster.fullname" . }}
{{- printf "%s-oauth2-proxy" $fullname }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "sie-cluster.imagePullSecrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range . }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Health gate hook: Prometheus readiness SA name
*/}}
{{- define "sie-cluster.healthGate.prometheus.serviceAccountName" -}}
{{- printf "%s-health-prometheus" (include "sie-cluster.fullname" . | trunc 45 | trimSuffix "-") }}
{{- end }}

{{/*
Health gate hook: ScaledObject readiness SA name
*/}}
{{- define "sie-cluster.healthGate.scaledobject.serviceAccountName" -}}
{{- printf "%s-health-scaledobject" (include "sie-cluster.fullname" . | trunc 43 | trimSuffix "-") }}
{{- end }}

{{/*
Health gate hook: Gateway readiness SA name
*/}}
{{- define "sie-cluster.healthGate.gateway.serviceAccountName" -}}
{{- printf "%s-health-gateway" (include "sie-cluster.fullname" . | trunc 49 | trimSuffix "-") }}
{{- end }}

{{/*
Health gate hook: Config readiness SA name.
Budget: prefix (≤49) + "-health-config" (14) = 63 (DNS-1123 label max).
*/}}
{{- define "sie-cluster.healthGate.config.serviceAccountName" -}}
{{- printf "%s-health-config" (include "sie-cluster.fullname" . | trunc 49 | trimSuffix "-") }}
{{- end }}

{{/*
Validation: NATS install/enabled consistency.
Fails if nats.install=true but nats.enabled=false (NATS deploys but nothing connects).
*/}}
{{- define "sie-cluster.validateNats" -}}
{{- if and .Values.nats.install (not .Values.nats.enabled) }}
{{- fail "Invalid configuration: nats.install=true but nats.enabled=false. NATS will be deployed but nothing will connect to it. Set nats.enabled=true or nats.install=false." }}
{{- end }}
{{- if and (not .Values.nats.install) .Values.nats.enabled (not .Values.nats.url) }}
{{- fail "Invalid configuration: nats.enabled=true but nats.install=false and nats.url is empty. Either set nats.install=true for in-cluster NATS, or provide nats.url for an external NATS server." }}
{{- end }}
{{- end }}

{{/*
KEDA apply hook: ServiceAccount name
*/}}
{{- define "sie-cluster.keda.apply.serviceAccountName" -}}
{{- printf "%s-keda-apply" (include "sie-cluster.fullname" . | trunc 51 | trimSuffix "-") }}
{{- end }}
