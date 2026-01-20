{{/*
Expand the name of the chart.
*/}}
{{- define "claude-multiagent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "claude-multiagent.fullname" -}}
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
{{- define "claude-multiagent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "claude-multiagent.labels" -}}
helm.sh/chart: {{ include "claude-multiagent.chart" . }}
{{ include "claude-multiagent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "claude-multiagent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "claude-multiagent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "claude-multiagent.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "claude-multiagent.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate the namespace
*/}}
{{- define "claude-multiagent.namespace" -}}
{{- if .Values.namespace.create }}
{{- .Values.namespace.name | default .Release.Namespace }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Option A specific labels
*/}}
{{- define "claude-multiagent.optionA.labels" -}}
{{ include "claude-multiagent.labels" . }}
app.kubernetes.io/component: coordinator-file-based
{{- end }}

{{/*
Option A selector labels
*/}}
{{- define "claude-multiagent.optionA.selectorLabels" -}}
{{ include "claude-multiagent.selectorLabels" . }}
app.kubernetes.io/component: coordinator-file-based
{{- end }}

{{/*
Option B specific labels
*/}}
{{- define "claude-multiagent.optionB.labels" -}}
{{ include "claude-multiagent.labels" . }}
app.kubernetes.io/component: mcp-server
{{- end }}

{{/*
Option B selector labels
*/}}
{{- define "claude-multiagent.optionB.selectorLabels" -}}
{{ include "claude-multiagent.selectorLabels" . }}
app.kubernetes.io/component: mcp-server
{{- end }}

{{/*
Option C specific labels
*/}}
{{- define "claude-multiagent.optionC.labels" -}}
{{ include "claude-multiagent.labels" . }}
app.kubernetes.io/component: orchestrator
{{- end }}

{{/*
Option C selector labels
*/}}
{{- define "claude-multiagent.optionC.selectorLabels" -}}
{{ include "claude-multiagent.selectorLabels" . }}
app.kubernetes.io/component: orchestrator
{{- end }}

{{/*
Generate image reference
*/}}
{{- define "claude-multiagent.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" }}
{{- $repository := .image.repository }}
{{- $tag := .image.tag | default "latest" }}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}
