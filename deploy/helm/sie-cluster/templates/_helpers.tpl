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
Router labels
*/}}
{{- define "sie-cluster.router.labels" -}}
{{ include "sie-cluster.labels" . }}
app.kubernetes.io/component: router
{{- end }}

{{/*
Router selector labels
*/}}
{{- define "sie-cluster.router.selectorLabels" -}}
{{ include "sie-cluster.selectorLabels" . }}
app.kubernetes.io/component: router
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
Router image
*/}}
{{- define "sie-cluster.router.image" -}}
{{- $tag := default .Chart.AppVersion .Values.router.image.tag }}
{{- printf "%s:%s" .Values.router.image.repository $tag }}
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
Router service name (used for worker discovery)
*/}}
{{- define "sie-cluster.router.serviceName" -}}
{{- $fullname := include "sie-cluster.fullname" . }}
{{- printf "%s-router" $fullname }}
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
Health gate hook: Router readiness SA name
*/}}
{{- define "sie-cluster.healthGate.router.serviceAccountName" -}}
{{- printf "%s-health-router" (include "sie-cluster.fullname" . | trunc 49 | trimSuffix "-") }}
{{- end }}

{{/*
KEDA apply hook: ServiceAccount name
*/}}
{{- define "sie-cluster.keda.apply.serviceAccountName" -}}
{{- printf "%s-keda-apply" (include "sie-cluster.fullname" . | trunc 51 | trimSuffix "-") }}
{{- end }}
