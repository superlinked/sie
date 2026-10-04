{{- define "sie-cluster.threshold.config" -}}
# Isolated control broker; only gateways have credentials.
port: 4222
http_port: 8222
jetstream { max_memory_store: 16MB, max_file_store: 0 }
max_payload: 64KB
authorization {
  users: [{
    user: sie-gateway
    password: $SIE_NATS_AUTH_GATEWAY_PASSWORD
    permissions: {
      publish: { allow: [
        "$JS.API.INFO",
        "$JS.API.STREAM.CREATE.KV_SIE_THRESHOLD_COUNTS",
        "$JS.API.STREAM.CREATE.KV_SIE_THRESHOLD_DECISIONS",
        "$JS.API.STREAM.CREATE.KV_SIE_THRESHOLD_LEASE",
        "$JS.API.STREAM.INFO.KV_SIE_THRESHOLD_COUNTS",
        "$JS.API.STREAM.INFO.KV_SIE_THRESHOLD_DECISIONS",
        "$JS.API.STREAM.INFO.KV_SIE_THRESHOLD_LEASE",
        "$JS.API.STREAM.MSG.GET.KV_SIE_THRESHOLD_COUNTS",
        "$JS.API.STREAM.MSG.GET.KV_SIE_THRESHOLD_DECISIONS",
        "$JS.API.STREAM.MSG.GET.KV_SIE_THRESHOLD_LEASE",
        "$KV.SIE_THRESHOLD_COUNTS.>",
        "$KV.SIE_THRESHOLD_DECISIONS.>",
        "$KV.SIE_THRESHOLD_LEASE.sampler"
      ] }
      subscribe: { allow: ["_INBOX.>"] }
    }
  }]
}
{{- end }}
