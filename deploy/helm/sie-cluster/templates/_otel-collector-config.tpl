{{/*
The exact collector.yaml payload shared by the ConfigMap and rollout checksum.
All values are resolved and validated by the caller before this partial runs.
*/}}
{{- define "sie-cluster.otel.collectorConfig" -}}
{{- $metricsEnabled := .metricsEnabled -}}
{{- $logsEnabled := .logsEnabled -}}
{{- $tracesEnabled := .tracesEnabled -}}
{{- $prometheusEnabled := .prometheusEnabled -}}
{{- $collector := .collector -}}
{{- $traceEndpoint := .traceEndpoint -}}
{{- $logEndpoint := .logEndpoint -}}
{{- $betterStack := .betterStack -}}
{{- $deploymentEnvironment := .deploymentEnvironment -}}
{{- $cloudRegion := .cloudRegion -}}
{{- $localTraceExporters := .localTraceExporters -}}
{{- $logExporters := .logExporters -}}
extensions:
  health_check:
    endpoint: "0.0.0.0:13133"

receivers:
  # Only release gateway pods can reach this receiver. It is the sole
  # ingress allowed to claim the KEDA-trusted sie-gateway identity.
  otlp/gateway:
    protocols:
      grpc:
        endpoint: "0.0.0.0:4317"
      http:
        endpoint: "0.0.0.0:4318"
  # Config, worker and worker-sidecar telemetry uses an isolated receiver,
  # so those producers cannot inject gateway autoscaling signals.
  otlp/application:
    protocols:
      grpc:
        endpoint: "0.0.0.0:4327"
{{- if $betterStack.enabled }}
  # Isolated collector process health; applications still push OTLP and are
  # never scraped by this receiver.
  prometheus/self:
    config:
      scrape_configs:
        - job_name: sie-otel-collector
          scrape_interval: 30s
          static_configs:
            - targets: [127.0.0.1:8888]
{{- end }}

processors:
  memory_limiter:
    check_interval: 1s
    limit_percentage: 80
    spike_limit_percentage: 15
  batch:
    timeout: 2s
{{- if or $metricsEnabled $logsEnabled (and $tracesEnabled $betterStack.enabled) }}
  # The gateway receiver is a NetworkPolicy-enforced identity boundary. Do
  # not trust producer-supplied routing identity after that boundary: use the
  # receiver's fixed service and this collector deployment's environment.
  resource/gateway_identity:
    attributes:
      - key: service.name
        value: sie-gateway
        action: upsert
      - key: deployment.environment
        value: {{ $deploymentEnvironment | quote }}
        action: upsert
      - key: cloud.region
        value: {{ $cloudRegion | quote }}
        action: upsert
{{- end }}
{{- if or $metricsEnabled (and $tracesEnabled $betterStack.enabled) }}
  # Application producers share one receiver, so service.name is retained
  # only after the signal-specific allowlist accepts it. Environment and
  # region are still collector-authored routing dimensions.
  resource/application_identity:
    attributes:
      - key: deployment.environment
        value: {{ $deploymentEnvironment | quote }}
        action: upsert
      - key: cloud.region
        value: {{ $cloudRegion | quote }}
        action: upsert
{{- end }}
{{- if $metricsEnabled }}
  # Receiver and destination boundaries fail closed on the exact dotted
  # contract names before the Prometheus exporter can normalize punctuation.
  filter/prometheus_gateway_contract:
    error_mode: propagate
    metrics:
      metric:
        - 'not IsMatch(name, "^(sie[.]gateway[.]requests|sie[.]gateway[.]request[.]duration|sie[.]gateway[.]admission[.]decisions|sie[.]gateway[.]dispatches|sie[.]gateway[.]dispatch[.]duration|sie[.]gateway[.]config[.]applied_epoch|sie[.]gateway[.]config[.]operations|sie[.]gateway[.]config[.]bootstrap[.]degraded|sie[.]gateway[.]messaging[.]client[.]ready|sie[.]gateway[.]queue[.]publishes|sie[.]gateway[.]queue[.]publish[.]duration|sie[.]gateway[.]queue[.]publish[.]items|sie[.]gateway[.]queue[.]result_waits|sie[.]gateway[.]queue[.]result_wait[.]duration|sie[.]gateway[.]queue[.]result_chunks[.]received|sie[.]gateway[.]queue[.]result_chunk[.]bytes_received|sie[.]gateway[.]queue[.]result_chunk[.]rejections|sie[.]gateway[.]queue[.]result_chunk[.]transfers_completed|sie[.]gateway[.]queue[.]result_chunk[.]duplicates|sie[.]gateway[.]queue[.]result_chunk[.]retry_replacements|sie[.]gateway[.]queue[.]result_chunk[.]stale_retries|sie[.]gateway[.]queue[.]result_chunk[.]reserved_bytes|sie[.]gateway[.]queue[.]events|sie[.]gateway[.]provisioning[.]responses|sie[.]gateway[.]generation[.]events|sie[.]gateway[.]generation[.]ttft|sie[.]gateway[.]generation[.]tpot|sie[.]gateway[.]generation[.]tokens|sie[.]gateway[.]pool[.]pinned_model[.]loaded|sie[.]gateway[.]pending_demand|sie[.]gateway[.]lane[.]queue[.]depth|sie[.]gateway[.]lane[.]queue[.]snapshot[.]timestamp|sie[.]gateway[.]active_lease[.]gpus|sie[.]gateway[.]pool[.]warm_floor|sie[.]gateway[.]rejected[.]requests|sie[.]gateway[.]capacity[.]snapshot[.]timestamp)$")'
        - 'resource.attributes["service.name"] != "sie-gateway"'
  filter/prometheus_application_contract:
    error_mode: propagate
    metrics:
      metric:
        - 'not IsMatch(name, "^(sie[.]dispatcher[.]invocations|sie[.]dispatcher[.]invocation[.]duration|sie[.]dispatcher[.]inflight|sie[.]dispatcher[.]sealed[.]stage|sie[.]dispatcher[.]sealed[.]stage[.]duration|sie[.]dispatcher[.]sealed[.]resident|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds|sie[.]config[.]requests|sie[.]config[.]request[.]duration|sie[.]config[.]epoch|sie[.]config[.]models|sie[.]config[.]publish|sie[.]config[.]store[.]writes|sie[.]config[.]messaging[.]ready|sie[.]worker[.]queue[.]duration|sie[.]worker[.]scheduler[.]request_batch[.]dispatch_wait|sie[.]worker[.]scheduler[.]request_batch[.]total|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]queue[.]pending_at_dispatch|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]ipc[.]requests|sie[.]worker[.]ipc[.]request[.]duration|sie[.]worker[.]ipc[.]response[.]chunks|sie[.]worker[.]ipc[.]response[.]reconstructed[.]size|sie[.]worker[.]ipc[.]response[.]chunk[.]count|sie[.]worker[.]ipc[.]response[.]chunk[.]reserved|sie[.]worker[.]config[.]applies|sie[.]worker[.]config[.]epoch|sie[.]worker[.]config[.]degraded|sie[.]worker[.]nats[.]operations|sie[.]worker[.]nats[.]delivery[.]attempts|sie[.]worker[.]result[.]transport[.]attempts|sie[.]worker[.]result[.]chunks[.]published|sie[.]worker[.]result[.]chunk[.]size|sie[.]worker[.]payload[.]fetches|sie[.]worker[.]payload[.]fetch[.]duration|sie[.]worker[.]payload[.]size|sie[.]worker[.]gpu[.]slots|sie[.]worker[.]pending[.]items|sie[.]worker[.]pending[.]cost|sie[.]worker[.]inflight[.]batches|sie[.]worker[.]saturated|sie[.]worker[.]ipc[.]capacity|sie[.]worker[.]ipc[.]inflight|sie[.]worker[.]ipc[.]acquire[.]duration|sie[.]worker[.]generation[.]model_loading[.]responses|sie[.]worker[.]shutdown[.]drain[.]duration|sie[.]worker[.]runtime[.]batch[.]size|sie[.]worker[.]runtime[.]batch[.]subgroups|sie[.]worker[.]runtime[.]subgroup[.]size|sie[.]worker[.]requests|sie[.]worker[.]request[.]duration|sie[.]worker[.]inference[.]duration|sie[.]worker[.]units|sie[.]worker[.]model[.]loaded|sie[.]worker[.]model[.]load[.]duration|sie[.]worker[.]model[.]memory|sie[.]worker[.]oom[.]recoveries|sie[.]worker[.]model[.]evictions|sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot|sie[.]worker[.]generation[.]tokens|sie[.]worker[.]generation[.]inflight|sie[.]worker[.]generation[.]kv[.]reserved|sie[.]worker[.]generation[.]kv[.]budget|sie[.]worker[.]generation[.]admission[.]decisions|sie[.]worker[.]generation[.]duplicate_prevented|sie[.]worker[.]generation[.]grammar[.]compile[.]duration|sie[.]worker[.]generation[.]grammar[.]cache[.]lookups|sie[.]worker[.]generation[.]grammar[.]requests|sie[.]worker[.]runtime[.]forward[.]duration|sie[.]worker[.]runtime[.]forward[.]permit[.]wait|sie[.]worker[.]runtime[.]forward[.]concurrent|sie[.]worker[.]runtime[.]forward[.]limit)$")'
        - 'resource.attributes["service.name"] != "sie-config" and resource.attributes["service.name"] != "sie-dispatcher" and resource.attributes["service.name"] != "sie-worker" and resource.attributes["service.name"] != "sie-worker-sidecar"'
        - 'resource.attributes["service.name"] == "sie-dispatcher" and not IsMatch(name, "^(sie[.]dispatcher[.]invocations|sie[.]dispatcher[.]invocation[.]duration|sie[.]dispatcher[.]inflight|sie[.]dispatcher[.]sealed[.]stage|sie[.]dispatcher[.]sealed[.]stage[.]duration|sie[.]dispatcher[.]sealed[.]resident|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds)$")'
        - 'resource.attributes["service.name"] == "sie-config" and not IsMatch(name, "^(sie[.]config[.]requests|sie[.]config[.]request[.]duration|sie[.]config[.]epoch|sie[.]config[.]models|sie[.]config[.]publish|sie[.]config[.]store[.]writes|sie[.]config[.]messaging[.]ready)$")'
        - 'resource.attributes["service.name"] == "sie-worker-sidecar" and not IsMatch(name, "^(sie[.]worker[.]queue[.]duration|sie[.]worker[.]scheduler[.]request_batch[.]dispatch_wait|sie[.]worker[.]scheduler[.]request_batch[.]total|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]ipc[.]requests|sie[.]worker[.]ipc[.]request[.]duration|sie[.]worker[.]ipc[.]response[.]chunks|sie[.]worker[.]ipc[.]response[.]reconstructed[.]size|sie[.]worker[.]ipc[.]response[.]chunk[.]count|sie[.]worker[.]ipc[.]response[.]chunk[.]reserved|sie[.]worker[.]config[.]applies|sie[.]worker[.]config[.]epoch|sie[.]worker[.]config[.]degraded|sie[.]worker[.]nats[.]operations|sie[.]worker[.]nats[.]delivery[.]attempts|sie[.]worker[.]result[.]transport[.]attempts|sie[.]worker[.]result[.]chunks[.]published|sie[.]worker[.]result[.]chunk[.]size|sie[.]worker[.]payload[.]fetches|sie[.]worker[.]payload[.]fetch[.]duration|sie[.]worker[.]payload[.]size|sie[.]worker[.]gpu[.]slots|sie[.]worker[.]pending[.]items|sie[.]worker[.]pending[.]cost|sie[.]worker[.]inflight[.]batches|sie[.]worker[.]saturated|sie[.]worker[.]ipc[.]capacity|sie[.]worker[.]ipc[.]inflight|sie[.]worker[.]ipc[.]acquire[.]duration|sie[.]worker[.]generation[.]model_loading[.]responses|sie[.]worker[.]shutdown[.]drain[.]duration)$")'
        - 'resource.attributes["service.name"] == "sie-worker" and not IsMatch(name, "^(sie[.]worker[.]queue[.]duration|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]queue[.]pending_at_dispatch|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]runtime[.]batch[.]size|sie[.]worker[.]runtime[.]batch[.]subgroups|sie[.]worker[.]runtime[.]subgroup[.]size|sie[.]worker[.]requests|sie[.]worker[.]request[.]duration|sie[.]worker[.]inference[.]duration|sie[.]worker[.]units|sie[.]worker[.]model[.]loaded|sie[.]worker[.]model[.]load[.]duration|sie[.]worker[.]model[.]memory|sie[.]worker[.]oom[.]recoveries|sie[.]worker[.]model[.]evictions|sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot|sie[.]worker[.]generation[.]tokens|sie[.]worker[.]generation[.]inflight|sie[.]worker[.]generation[.]kv[.]reserved|sie[.]worker[.]generation[.]kv[.]budget|sie[.]worker[.]generation[.]admission[.]decisions|sie[.]worker[.]generation[.]duplicate_prevented|sie[.]worker[.]generation[.]grammar[.]compile[.]duration|sie[.]worker[.]generation[.]grammar[.]cache[.]lookups|sie[.]worker[.]generation[.]grammar[.]requests|sie[.]worker[.]runtime[.]forward[.]duration|sie[.]worker[.]runtime[.]forward[.]permit[.]wait|sie[.]worker[.]runtime[.]forward[.]concurrent|sie[.]worker[.]runtime[.]forward[.]limit)$")'
  # This is the metric field firewall shared by Prometheus and OTLP
  # destinations. Every descriptor and point is reduced to the checked-in
  # contract; resource and instrumentation-scope extensions are discarded.
  transform/contract_metrics:
    error_mode: propagate
    metric_statements:
      - context: resource
        statements:
          - keep_keys(attributes, ["service.name", "service.instance.id", "deployment.environment", "cloud.region", "service.version"])
          - set(schema_url, "")
      - context: scope
        statements:
          - keep_keys(attributes, [])
          - set(name, "")
          - set(version, "")
          - set(schema_url, "")
      - context: metric
        statements:
          - set(description, "")
          - 'set(unit, "{request}") where IsMatch(name, "^(sie[.]gateway[.]requests|sie[.]gateway[.]admission[.]decisions|sie[.]gateway[.]dispatches|sie[.]config[.]requests|sie[.]worker[.]ipc[.]requests|sie[.]worker[.]ipc[.]capacity|sie[.]worker[.]ipc[.]inflight|sie[.]worker[.]generation[.]inflight|sie[.]worker[.]generation[.]duplicate_prevented|sie[.]worker[.]generation[.]grammar[.]requests|sie[.]gateway[.]pending_demand|sie[.]gateway[.]rejected[.]requests)$")'
          - 'set(unit, "s") where IsMatch(name, "^(sie[.]gateway[.]request[.]duration|sie[.]gateway[.]dispatch[.]duration|sie[.]gateway[.]queue[.]publish[.]duration|sie[.]gateway[.]queue[.]result_wait[.]duration|sie[.]gateway[.]generation[.]ttft|sie[.]gateway[.]generation[.]tpot|sie[.]dispatcher[.]invocation[.]duration|sie[.]dispatcher[.]sealed[.]stage[.]duration|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds|sie[.]config[.]request[.]duration|sie[.]worker[.]queue[.]duration|sie[.]worker[.]scheduler[.]request_batch[.]dispatch_wait|sie[.]worker[.]scheduler[.]request_batch[.]total|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]ipc[.]request[.]duration|sie[.]worker[.]payload[.]fetch[.]duration|sie[.]worker[.]ipc[.]acquire[.]duration|sie[.]worker[.]shutdown[.]drain[.]duration|sie[.]worker[.]request[.]duration|sie[.]worker[.]inference[.]duration|sie[.]worker[.]model[.]load[.]duration|sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot|sie[.]worker[.]generation[.]grammar[.]compile[.]duration|sie[.]worker[.]runtime[.]forward[.]duration|sie[.]worker[.]runtime[.]forward[.]permit[.]wait|sie[.]gateway[.]lane[.]queue[.]snapshot[.]timestamp|sie[.]gateway[.]capacity[.]snapshot[.]timestamp)$")'
          - 'set(unit, "{epoch}") where IsMatch(name, "^(sie[.]gateway[.]config[.]applied_epoch|sie[.]config[.]epoch|sie[.]worker[.]config[.]epoch)$")'
          - 'set(unit, "{operation}") where IsMatch(name, "^(sie[.]gateway[.]config[.]operations|sie[.]config[.]publish|sie[.]config[.]store[.]writes|sie[.]worker[.]nats[.]operations)$")'
          - 'set(unit, "1") where IsMatch(name, "^(sie[.]gateway[.]config[.]bootstrap[.]degraded|sie[.]gateway[.]messaging[.]client[.]ready|sie[.]config[.]messaging[.]ready|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]config[.]degraded|sie[.]worker[.]saturated|sie[.]gateway[.]pool[.]pinned_model[.]loaded)$")'
          - 'set(unit, "{publish}") where name == "sie.gateway.queue.publishes"'
          - 'set(unit, "{item}") where IsMatch(name, "^(sie[.]gateway[.]queue[.]publish[.]items|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]queue[.]pending_at_dispatch|sie[.]worker[.]pending[.]items|sie[.]worker[.]runtime[.]batch[.]size|sie[.]worker[.]runtime[.]subgroup[.]size|sie[.]worker[.]requests|sie[.]gateway[.]lane[.]queue[.]depth)$")'
          - 'set(unit, "{wait}") where name == "sie.gateway.queue.result_waits"'
          - 'set(unit, "{chunk}") where IsMatch(name, "^(sie[.]gateway[.]queue[.]result_chunks[.]received|sie[.]gateway[.]queue[.]result_chunk[.]duplicates|sie[.]gateway[.]queue[.]result_chunk[.]stale_retries|sie[.]worker[.]ipc[.]response[.]chunk[.]count|sie[.]worker[.]result[.]chunks[.]published)$")'
          - 'set(unit, "{transfer}") where IsMatch(name, "^(sie[.]gateway[.]queue[.]result_chunk[.]transfers_completed|sie[.]gateway[.]queue[.]result_chunk[.]retry_replacements|sie[.]worker[.]ipc[.]response[.]chunks)$")'
          - 'set(unit, "{rejection}") where name == "sie.gateway.queue.result_chunk.rejections"'
          - 'set(unit, "{attempt}") where name == "sie.worker.result.transport.attempts"'
          - 'set(unit, "By") where IsMatch(name, "^(sie[.]gateway[.]queue[.]result_chunk[.]bytes_received|sie[.]gateway[.]queue[.]result_chunk[.]reserved_bytes|sie[.]worker[.]ipc[.]response[.]reconstructed[.]size|sie[.]worker[.]ipc[.]response[.]chunk[.]reserved|sie[.]worker[.]result[.]chunk[.]size)$")'
          - 'set(unit, "{event}") where IsMatch(name, "^(sie[.]gateway[.]queue[.]events|sie[.]gateway[.]generation[.]events)$")'
          - 'set(unit, "{response}") where IsMatch(name, "^(sie[.]gateway[.]provisioning[.]responses|sie[.]worker[.]generation[.]model_loading[.]responses)$")'
          - 'set(unit, "{token}") where IsMatch(name, "^(sie[.]gateway[.]generation[.]tokens|sie[.]worker[.]generation[.]tokens|sie[.]worker[.]generation[.]kv[.]reserved|sie[.]worker[.]generation[.]kv[.]budget)$")'
          - 'set(unit, "{invocation}") where IsMatch(name, "^(sie[.]dispatcher[.]invocations|sie[.]dispatcher[.]inflight)$")'
          - 'set(unit, "{stage}") where IsMatch(name, "^(sie[.]dispatcher[.]sealed[.]stage)$")'
          - 'set(unit, "{engine}") where IsMatch(name, "^(sie[.]dispatcher[.]sealed[.]resident)$")'
          - 'set(unit, "{model}") where IsMatch(name, "^(sie[.]config[.]models|sie[.]worker[.]model[.]loaded|sie[.]worker[.]model[.]evictions)$")'
          - 'set(unit, "{cost}") where IsMatch(name, "^(sie[.]worker[.]batch[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]pending[.]cost)$")'
          - 'set(unit, "{reset}") where name == "sie.worker.scheduler.starvation.resets"'
          - 'set(unit, "{apply}") where name == "sie.worker.config.applies"'
          - 'set(unit, "{attempt}") where name == "sie.worker.nats.delivery.attempts"'
          - 'set(unit, "{fetch}") where name == "sie.worker.payload.fetches"'
          - 'set(unit, "By") where IsMatch(name, "^(sie[.]worker[.]payload[.]size|sie[.]worker[.]model[.]memory)$")'
          - 'set(unit, "{slot}") where name == "sie.worker.gpu.slots"'
          - 'set(unit, "{batch}") where name == "sie.worker.inflight.batches"'
          - 'set(unit, "{subgroup}") where name == "sie.worker.runtime.batch.subgroups"'
          - 'set(unit, "{unit}") where name == "sie.worker.units"'
          - 'set(unit, "{recovery}") where name == "sie.worker.oom.recoveries"'
          - 'set(unit, "{decision}") where name == "sie.worker.generation.admission.decisions"'
          - 'set(unit, "{lookup}") where name == "sie.worker.generation.grammar.cache.lookups"'
          - 'set(unit, "{forward}") where IsMatch(name, "^(sie[.]worker[.]runtime[.]forward[.]concurrent|sie[.]worker[.]runtime[.]forward[.]limit)$")'
          - 'set(unit, "{gpu}") where name == "sie.gateway.active_lease.gpus"'
          - 'set(unit, "{worker}") where name == "sie.gateway.pool.warm_floor"'
      - context: datapoint
        statements:
          - 'keep_keys(attributes, ["operation", "outcome", "http.status_code", "machine_profile"]) where IsMatch(metric.name, "^(sie[.]gateway[.]requests|sie[.]gateway[.]request[.]duration)$")'
          - 'keep_keys(attributes, ["operation", "outcome"]) where IsMatch(metric.name, "^(sie[.]gateway[.]admission[.]decisions|sie[.]gateway[.]config[.]operations|sie[.]gateway[.]queue[.]publishes|sie[.]gateway[.]queue[.]publish[.]duration|sie[.]gateway[.]queue[.]publish[.]items|sie[.]gateway[.]queue[.]result_waits|sie[.]gateway[.]queue[.]result_wait[.]duration|sie[.]config[.]publish|sie[.]config[.]store[.]writes)$")'
          - 'keep_keys(attributes, []) where IsMatch(metric.name, "^(sie[.]gateway[.]queue[.]result_chunks[.]received|sie[.]gateway[.]queue[.]result_chunk[.]bytes_received|sie[.]gateway[.]queue[.]result_chunk[.]transfers_completed|sie[.]gateway[.]queue[.]result_chunk[.]duplicates|sie[.]gateway[.]queue[.]result_chunk[.]retry_replacements|sie[.]gateway[.]queue[.]result_chunk[.]stale_retries|sie[.]gateway[.]queue[.]result_chunk[.]reserved_bytes)$")'
          - 'keep_keys(attributes, ["reason"]) where metric.name == "sie.gateway.queue.result_chunk.rejections"'
          - 'keep_keys(attributes, ["operation", "dispatch.path", "outcome", "fallback.reason", "lane"]) where IsMatch(metric.name, "^(sie[.]gateway[.]dispatches|sie[.]gateway[.]dispatch[.]duration)$")'
          - 'keep_keys(attributes, []) where IsMatch(metric.name, "^(sie[.]gateway[.]config[.]applied_epoch|sie[.]gateway[.]config[.]bootstrap[.]degraded|sie[.]config[.]epoch|sie[.]gateway[.]capacity[.]snapshot[.]timestamp)$")'
          - 'keep_keys(attributes, ["transport"]) where IsMatch(metric.name, "^(sie[.]gateway[.]messaging[.]client[.]ready|sie[.]config[.]messaging[.]ready)$")'
          - 'keep_keys(attributes, ["event", "outcome"]) where metric.name == "sie.gateway.queue.events"'
          - 'keep_keys(attributes, ["surface", "http.status_code"]) where metric.name == "sie.gateway.provisioning.responses"'
          - 'keep_keys(attributes, ["event", "reason", "outcome"]) where metric.name == "sie.gateway.generation.events"'
          - 'keep_keys(attributes, ["operation"]) where IsMatch(metric.name, "^(sie[.]gateway[.]generation[.]ttft|sie[.]gateway[.]generation[.]tpot)$")'
          - 'keep_keys(attributes, ["operation", "token.kind"]) where metric.name == "sie.gateway.generation.tokens"'
          - 'keep_keys(attributes, ["operation", "dispatch.path", "outcome", "lane"]) where IsMatch(metric.name, "^(sie[.]dispatcher[.]invocations|sie[.]dispatcher[.]invocation[.]duration)$")'
          - 'keep_keys(attributes, ["operation", "dispatch.path", "lane"]) where metric.name == "sie.dispatcher.inflight"'
          - 'keep_keys(attributes, ["outcome"]) where IsMatch(metric.name, "^(sie[.]dispatcher[.]sealed[.]stage|sie[.]dispatcher[.]sealed[.]stage[.]duration)$")'
          - 'keep_keys(attributes, []) where IsMatch(metric.name, "^(sie[.]dispatcher[.]sealed[.]resident|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds)$")'
          - 'keep_keys(attributes, ["http.method", "http.route", "http.status_code"]) where IsMatch(metric.name, "^(sie[.]config[.]requests|sie[.]config[.]request[.]duration)$")'
          - 'keep_keys(attributes, ["source"]) where metric.name == "sie.config.models"'
          - 'keep_keys(attributes, ["operation", "lane", "model", "profile"]) where IsMatch(metric.name, "^(sie[.]worker[.]queue[.]duration|sie[.]worker[.]scheduler[.]request_batch[.]dispatch_wait|sie[.]worker[.]scheduler[.]request_batch[.]total|sie[.]worker[.]queue[.]depth|sie[.]worker[.]queue[.]pending_at_dispatch)$")'
          - 'keep_keys(attributes, ["operation", "lane", "model", "profile"]) where IsMatch(metric.name, "^(sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost)$")'
          - 'keep_keys(attributes, ["operation", "lane", "model", "profile", "flush.reason"]) where metric.name == "sie.worker.batch.fill_ratio"'
          - 'keep_keys(attributes, ["lane", "model", "profile"]) where IsMatch(metric.name, "^(sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]starvation[.]resets)$")'
          - 'keep_keys(attributes, ["kind", "lane", "model", "profile"]) where metric.name == "sie.worker.scheduler.adaptive.p50"'
          - 'keep_keys(attributes, ["method", "outcome", "lane"]) where IsMatch(metric.name, "^(sie[.]worker[.]ipc[.]requests|sie[.]worker[.]ipc[.]request[.]duration)$")'
          - 'keep_keys(attributes, ["outcome", "lane"]) where metric.name == "sie.worker.ipc.response.chunks"'
          - 'keep_keys(attributes, ["lane"]) where IsMatch(metric.name, "^(sie[.]worker[.]ipc[.]response[.]reconstructed[.]size|sie[.]worker[.]ipc[.]response[.]chunk[.]count|sie[.]worker[.]ipc[.]response[.]chunk[.]reserved|sie[.]worker[.]result[.]chunks[.]published|sie[.]worker[.]result[.]chunk[.]size)$")'
          - 'keep_keys(attributes, ["mode", "outcome", "lane"]) where metric.name == "sie.worker.result.transport.attempts"'
          - 'keep_keys(attributes, ["source", "operation", "outcome", "lane"]) where metric.name == "sie.worker.config.applies"'
          - 'keep_keys(attributes, ["source", "lane"]) where IsMatch(metric.name, "^(sie[.]worker[.]config[.]epoch|sie[.]worker[.]config[.]degraded)$")'
          - 'keep_keys(attributes, ["operation", "outcome", "reason", "lane"]) where metric.name == "sie.worker.nats.operations"'
          - 'keep_keys(attributes, ["redelivered", "lane"]) where metric.name == "sie.worker.nats.delivery.attempts"'
          - 'keep_keys(attributes, ["outcome", "reason", "lane"]) where IsMatch(metric.name, "^(sie[.]worker[.]payload[.]fetches|sie[.]worker[.]payload[.]fetch[.]duration|sie[.]worker[.]payload[.]size)$")'
          - 'keep_keys(attributes, ["state", "lane"]) where metric.name == "sie.worker.gpu.slots"'
          - 'keep_keys(attributes, ["lane"]) where IsMatch(metric.name, "^(sie[.]worker[.]pending[.]items|sie[.]worker[.]pending[.]cost|sie[.]worker[.]inflight[.]batches|sie[.]worker[.]saturated)$")'
          - 'keep_keys(attributes, ["transport", "lane"]) where IsMatch(metric.name, "^(sie[.]worker[.]ipc[.]capacity|sie[.]worker[.]ipc[.]inflight)$")'
          - 'keep_keys(attributes, ["transport", "outcome", "lane"]) where metric.name == "sie.worker.ipc.acquire.duration"'
          - 'keep_keys(attributes, ["state", "outcome", "lane", "model", "profile"]) where metric.name == "sie.worker.generation.model_loading.responses"'
          - 'keep_keys(attributes, ["outcome", "lane"]) where metric.name == "sie.worker.shutdown.drain.duration"'
          - 'keep_keys(attributes, ["operation", "backend", "lane", "model", "profile"]) where IsMatch(metric.name, "^(sie[.]worker[.]runtime[.]batch[.]size|sie[.]worker[.]runtime[.]batch[.]subgroups|sie[.]worker[.]runtime[.]subgroup[.]size)$")'
          - 'keep_keys(attributes, ["operation", "outcome", "backend", "lane", "model", "profile"]) where IsMatch(metric.name, "^(sie[.]worker[.]requests|sie[.]worker[.]request[.]duration)$")'
          - 'keep_keys(attributes, ["operation", "outcome", "phase", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.inference.duration"'
          - 'keep_keys(attributes, ["operation", "backend", "lane", "model", "profile", "unit.type"]) where metric.name == "sie.worker.units"'
          - 'keep_keys(attributes, ["backend", "lane", "model", "profile"]) where IsMatch(metric.name, "^(sie[.]worker[.]model[.]loaded|sie[.]worker[.]model[.]memory|sie[.]worker[.]generation[.]inflight|sie[.]worker[.]generation[.]kv[.]reserved|sie[.]worker[.]generation[.]kv[.]budget|sie[.]worker[.]runtime[.]forward[.]limit)$")'
          - 'keep_keys(attributes, ["outcome", "stage", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.model.load.duration"'
          - 'keep_keys(attributes, ["strategy", "outcome", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.oom.recoveries"'
          - 'keep_keys(attributes, ["reason", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.model.evictions"'
          - 'keep_keys(attributes, ["grammar", "backend", "lane", "model", "profile"]) where IsMatch(metric.name, "^(sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot)$")'
          - 'keep_keys(attributes, ["token.type", "grammar", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.generation.tokens"'
          - 'keep_keys(attributes, ["outcome", "reason", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.generation.admission.decisions"'
          - 'keep_keys(attributes, ["dispatch.path", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.generation.duplicate_prevented"'
          - 'keep_keys(attributes, ["grammar.backend", "grammar", "phase", "outcome", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.generation.grammar.compile.duration"'
          - 'keep_keys(attributes, ["grammar.backend", "grammar", "phase", "result", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.generation.grammar.cache.lookups"'
          - 'keep_keys(attributes, ["grammar.backend", "grammar", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.generation.grammar.requests"'
          - 'keep_keys(attributes, ["outcome", "input.source", "output.path", "stage", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.runtime.forward.duration"'
          - 'keep_keys(attributes, ["output.path", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.runtime.forward.permit.wait"'
          - 'keep_keys(attributes, ["state", "backend", "lane", "model", "profile"]) where metric.name == "sie.worker.runtime.forward.concurrent"'
          - 'keep_keys(attributes, ["pool", "model"]) where metric.name == "sie.gateway.pool.pinned_model.loaded"'
          - 'keep_keys(attributes, ["pool", "machine_profile", "bundle"]) where IsMatch(metric.name, "^(sie[.]gateway[.]pending_demand|sie[.]gateway[.]lane[.]queue[.]depth|sie[.]gateway[.]lane[.]queue[.]snapshot[.]timestamp|sie[.]gateway[.]active_lease[.]gpus|sie[.]gateway[.]pool[.]warm_floor)$")'
          - 'keep_keys(attributes, ["pool", "machine_profile", "bundle", "reason", "scaling_action"]) where metric.name == "sie.gateway.rejected.requests"'
  # Prometheus producer labels are explicit collector output, not copied
  # from exported_job/exported_instance after scrape-label conflict handling.
  transform/prometheus_gateway_identity:
    error_mode: propagate
    metric_statements:
      - context: datapoint
        statements:
          - set(attributes["producer_service"], "sie-gateway")
          - set(attributes["producer_instance"], resource.attributes["service.instance.id"])
  transform/prometheus_compatibility:
    error_mode: propagate
    metric_statements:
      - context: metric
        statements:
          - 'set(unit, "") where name == "sie.gateway.pool.pinned_model.loaded"'
  transform/prometheus_application_identity:
    error_mode: propagate
    metric_statements:
      - context: datapoint
        statements:
          - set(attributes["producer_service"], resource.attributes["service.name"])
          - set(attributes["producer_instance"], resource.attributes["service.instance.id"])
{{- if $prometheusEnabled }}
  # The Prometheus exporter accumulates application DELTA sums and histograms
  # in process memory. Give every collector process a distinct output series
  # so a restart cannot be mistaken for continuation of the prior accumulator.
  # The file provider reads one fresh kernel UUID when this config is loaded.
  resource/prometheus_generation:
    attributes:
      - key: sie.collector.generation
        value: ${file:/proc/sys/kernel/random/uuid}
        action: upsert
  transform/prometheus_generation:
    error_mode: propagate
    metric_statements:
      - context: resource
        statements:
          - 'replace_pattern(attributes["sie.collector.generation"], "\\s+$", "")'
      - context: datapoint
        statements:
          - set(attributes["collector_generation"], resource.attributes["sie.collector.generation"])
{{- end }}
{{- end }}
{{- if and $metricsEnabled $betterStack.enabled }}
  # Better Stack joins the queue value to its per-lane freshness companion on
  # this collector-authored producer key, independent of backend tag mapping.
  transform/remote_queue_identity:
    error_mode: propagate
    metric_statements:
      - context: datapoint
        statements:
          - 'set(attributes["producer_instance"], resource.attributes["service.instance.id"]) where IsMatch(metric.name, "^(sie[.]gateway[.]lane[.]queue[.]depth|sie[.]gateway[.]lane[.]queue[.]snapshot[.]timestamp)$")'
  filter/remote_gateway_contract:
    error_mode: propagate
    metrics:
      metric:
        - 'not IsMatch(name, "^(sie[.]gateway[.]requests|sie[.]gateway[.]request[.]duration|sie[.]gateway[.]admission[.]decisions|sie[.]gateway[.]dispatches|sie[.]gateway[.]dispatch[.]duration|sie[.]gateway[.]config[.]applied_epoch|sie[.]gateway[.]config[.]operations|sie[.]gateway[.]config[.]bootstrap[.]degraded|sie[.]gateway[.]messaging[.]client[.]ready|sie[.]gateway[.]queue[.]publishes|sie[.]gateway[.]queue[.]publish[.]duration|sie[.]gateway[.]queue[.]publish[.]items|sie[.]gateway[.]queue[.]result_waits|sie[.]gateway[.]queue[.]result_wait[.]duration|sie[.]gateway[.]queue[.]result_chunks[.]received|sie[.]gateway[.]queue[.]result_chunk[.]bytes_received|sie[.]gateway[.]queue[.]result_chunk[.]rejections|sie[.]gateway[.]queue[.]result_chunk[.]transfers_completed|sie[.]gateway[.]queue[.]result_chunk[.]duplicates|sie[.]gateway[.]queue[.]result_chunk[.]retry_replacements|sie[.]gateway[.]queue[.]result_chunk[.]stale_retries|sie[.]gateway[.]queue[.]result_chunk[.]reserved_bytes|sie[.]gateway[.]queue[.]events|sie[.]gateway[.]provisioning[.]responses|sie[.]gateway[.]generation[.]events|sie[.]gateway[.]generation[.]ttft|sie[.]gateway[.]generation[.]tpot|sie[.]gateway[.]generation[.]tokens|sie[.]gateway[.]pending_demand|sie[.]gateway[.]lane[.]queue[.]depth|sie[.]gateway[.]lane[.]queue[.]snapshot[.]timestamp|sie[.]gateway[.]active_lease[.]gpus|sie[.]gateway[.]pool[.]warm_floor|sie[.]gateway[.]rejected[.]requests)$")'
        - 'resource.attributes["service.name"] != "sie-gateway"'
  filter/remote_application_contract:
    error_mode: propagate
    metrics:
      metric:
        - 'not IsMatch(name, "^(sie[.]dispatcher[.]invocations|sie[.]dispatcher[.]invocation[.]duration|sie[.]dispatcher[.]inflight|sie[.]dispatcher[.]sealed[.]stage|sie[.]dispatcher[.]sealed[.]stage[.]duration|sie[.]dispatcher[.]sealed[.]resident|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds|sie[.]config[.]requests|sie[.]config[.]request[.]duration|sie[.]config[.]epoch|sie[.]config[.]models|sie[.]config[.]publish|sie[.]config[.]store[.]writes|sie[.]config[.]messaging[.]ready|sie[.]worker[.]queue[.]duration|sie[.]worker[.]scheduler[.]request_batch[.]dispatch_wait|sie[.]worker[.]scheduler[.]request_batch[.]total|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]queue[.]pending_at_dispatch|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]ipc[.]requests|sie[.]worker[.]ipc[.]request[.]duration|sie[.]worker[.]ipc[.]response[.]chunks|sie[.]worker[.]ipc[.]response[.]reconstructed[.]size|sie[.]worker[.]ipc[.]response[.]chunk[.]count|sie[.]worker[.]ipc[.]response[.]chunk[.]reserved|sie[.]worker[.]config[.]applies|sie[.]worker[.]config[.]epoch|sie[.]worker[.]config[.]degraded|sie[.]worker[.]nats[.]operations|sie[.]worker[.]nats[.]delivery[.]attempts|sie[.]worker[.]result[.]transport[.]attempts|sie[.]worker[.]result[.]chunks[.]published|sie[.]worker[.]result[.]chunk[.]size|sie[.]worker[.]payload[.]fetches|sie[.]worker[.]payload[.]fetch[.]duration|sie[.]worker[.]payload[.]size|sie[.]worker[.]gpu[.]slots|sie[.]worker[.]pending[.]items|sie[.]worker[.]pending[.]cost|sie[.]worker[.]inflight[.]batches|sie[.]worker[.]saturated|sie[.]worker[.]ipc[.]capacity|sie[.]worker[.]ipc[.]inflight|sie[.]worker[.]ipc[.]acquire[.]duration|sie[.]worker[.]generation[.]model_loading[.]responses|sie[.]worker[.]shutdown[.]drain[.]duration|sie[.]worker[.]runtime[.]batch[.]size|sie[.]worker[.]runtime[.]batch[.]subgroups|sie[.]worker[.]runtime[.]subgroup[.]size|sie[.]worker[.]requests|sie[.]worker[.]request[.]duration|sie[.]worker[.]inference[.]duration|sie[.]worker[.]units|sie[.]worker[.]model[.]loaded|sie[.]worker[.]model[.]load[.]duration|sie[.]worker[.]model[.]memory|sie[.]worker[.]oom[.]recoveries|sie[.]worker[.]model[.]evictions|sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot|sie[.]worker[.]generation[.]tokens|sie[.]worker[.]generation[.]inflight|sie[.]worker[.]generation[.]kv[.]reserved|sie[.]worker[.]generation[.]kv[.]budget|sie[.]worker[.]generation[.]admission[.]decisions|sie[.]worker[.]generation[.]duplicate_prevented|sie[.]worker[.]generation[.]grammar[.]compile[.]duration|sie[.]worker[.]generation[.]grammar[.]cache[.]lookups|sie[.]worker[.]generation[.]grammar[.]requests|sie[.]worker[.]runtime[.]forward[.]duration|sie[.]worker[.]runtime[.]forward[.]permit[.]wait|sie[.]worker[.]runtime[.]forward[.]concurrent|sie[.]worker[.]runtime[.]forward[.]limit)$")'
        - 'resource.attributes["service.name"] != "sie-config" and resource.attributes["service.name"] != "sie-dispatcher" and resource.attributes["service.name"] != "sie-worker" and resource.attributes["service.name"] != "sie-worker-sidecar"'
        - 'resource.attributes["service.name"] == "sie-dispatcher" and not IsMatch(name, "^(sie[.]dispatcher[.]invocations|sie[.]dispatcher[.]invocation[.]duration|sie[.]dispatcher[.]inflight|sie[.]dispatcher[.]sealed[.]stage|sie[.]dispatcher[.]sealed[.]stage[.]duration|sie[.]dispatcher[.]sealed[.]resident|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds)$")'
        - 'resource.attributes["service.name"] == "sie-config" and not IsMatch(name, "^(sie[.]config[.]requests|sie[.]config[.]request[.]duration|sie[.]config[.]epoch|sie[.]config[.]models|sie[.]config[.]publish|sie[.]config[.]store[.]writes|sie[.]config[.]messaging[.]ready)$")'
        - 'resource.attributes["service.name"] == "sie-worker-sidecar" and not IsMatch(name, "^(sie[.]worker[.]queue[.]duration|sie[.]worker[.]scheduler[.]request_batch[.]dispatch_wait|sie[.]worker[.]scheduler[.]request_batch[.]total|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]ipc[.]requests|sie[.]worker[.]ipc[.]request[.]duration|sie[.]worker[.]ipc[.]response[.]chunks|sie[.]worker[.]ipc[.]response[.]reconstructed[.]size|sie[.]worker[.]ipc[.]response[.]chunk[.]count|sie[.]worker[.]ipc[.]response[.]chunk[.]reserved|sie[.]worker[.]config[.]applies|sie[.]worker[.]config[.]epoch|sie[.]worker[.]config[.]degraded|sie[.]worker[.]nats[.]operations|sie[.]worker[.]nats[.]delivery[.]attempts|sie[.]worker[.]result[.]transport[.]attempts|sie[.]worker[.]result[.]chunks[.]published|sie[.]worker[.]result[.]chunk[.]size|sie[.]worker[.]payload[.]fetches|sie[.]worker[.]payload[.]fetch[.]duration|sie[.]worker[.]payload[.]size|sie[.]worker[.]gpu[.]slots|sie[.]worker[.]pending[.]items|sie[.]worker[.]pending[.]cost|sie[.]worker[.]inflight[.]batches|sie[.]worker[.]saturated|sie[.]worker[.]ipc[.]capacity|sie[.]worker[.]ipc[.]inflight|sie[.]worker[.]ipc[.]acquire[.]duration|sie[.]worker[.]generation[.]model_loading[.]responses|sie[.]worker[.]shutdown[.]drain[.]duration)$")'
        - 'resource.attributes["service.name"] == "sie-worker" and not IsMatch(name, "^(sie[.]worker[.]queue[.]duration|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]queue[.]pending_at_dispatch|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]runtime[.]batch[.]size|sie[.]worker[.]runtime[.]batch[.]subgroups|sie[.]worker[.]runtime[.]subgroup[.]size|sie[.]worker[.]requests|sie[.]worker[.]request[.]duration|sie[.]worker[.]inference[.]duration|sie[.]worker[.]units|sie[.]worker[.]model[.]loaded|sie[.]worker[.]model[.]load[.]duration|sie[.]worker[.]model[.]memory|sie[.]worker[.]oom[.]recoveries|sie[.]worker[.]model[.]evictions|sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot|sie[.]worker[.]generation[.]tokens|sie[.]worker[.]generation[.]inflight|sie[.]worker[.]generation[.]kv[.]reserved|sie[.]worker[.]generation[.]kv[.]budget|sie[.]worker[.]generation[.]admission[.]decisions|sie[.]worker[.]generation[.]duplicate_prevented|sie[.]worker[.]generation[.]grammar[.]compile[.]duration|sie[.]worker[.]generation[.]grammar[.]cache[.]lookups|sie[.]worker[.]generation[.]grammar[.]requests|sie[.]worker[.]runtime[.]forward[.]duration|sie[.]worker[.]runtime[.]forward[.]permit[.]wait|sie[.]worker[.]runtime[.]forward[.]concurrent|sie[.]worker[.]runtime[.]forward[.]limit)$")'
{{- end }}
{{- if $betterStack.enabled }}
  # Collector implementation health is isolated from the application
  # contract and reduced to nine stable families before remote export.
  filter/collector_self_contract:
    error_mode: propagate
    metrics:
      metric:
        - 'not IsMatch(name, "^(up|otelcol_exporter_queue_size|otelcol_exporter_queue_capacity|otelcol_receiver_refused_spans|otelcol_exporter_send_failed_spans|otelcol_receiver_refused_metric_points|otelcol_exporter_send_failed_metric_points|otelcol_receiver_refused_log_records|otelcol_exporter_send_failed_log_records|otelcol_receiver_refused_spans_total|otelcol_exporter_send_failed_spans_total|otelcol_receiver_refused_metric_points_total|otelcol_exporter_send_failed_metric_points_total|otelcol_receiver_refused_log_records_total|otelcol_exporter_send_failed_log_records_total)$")'
  transform/collector_self_metrics:
    error_mode: propagate
    metric_statements:
      - context: resource
        statements:
          - keep_keys(attributes, ["service.name", "service.instance.id", "deployment.environment", "cloud.region", "service.version"])
          - 'replace_pattern(attributes["service.instance.id"], "\\s+$", "")'
          - set(schema_url, "")
      - context: scope
        statements:
          - keep_keys(attributes, [])
          - set(name, "")
          - set(version, "")
          - set(schema_url, "")
      - context: metric
        statements:
          - 'set(name, "otelcol_receiver_refused_spans") where name == "otelcol_receiver_refused_spans_total"'
          - 'set(name, "otelcol_exporter_send_failed_spans") where name == "otelcol_exporter_send_failed_spans_total"'
          - 'set(name, "otelcol_receiver_refused_metric_points") where name == "otelcol_receiver_refused_metric_points_total"'
          - 'set(name, "otelcol_exporter_send_failed_metric_points") where name == "otelcol_exporter_send_failed_metric_points_total"'
          - 'set(name, "otelcol_receiver_refused_log_records") where name == "otelcol_receiver_refused_log_records_total"'
          - 'set(name, "otelcol_exporter_send_failed_log_records") where name == "otelcol_exporter_send_failed_log_records_total"'
          - set(description, "")
          - set(unit, "")
      - context: datapoint
        statements:
          - 'keep_keys(attributes, []) where metric.name == "up"'
          - 'keep_keys(attributes, ["exporter", "data_type"]) where IsMatch(metric.name, "^(otelcol_exporter_queue_size|otelcol_exporter_queue_capacity)$")'
          - 'keep_keys(attributes, ["receiver", "transport"]) where IsMatch(metric.name, "^(otelcol_receiver_refused_spans|otelcol_receiver_refused_metric_points|otelcol_receiver_refused_log_records)$")'
          - 'keep_keys(attributes, ["exporter", "transport"]) where IsMatch(metric.name, "^(otelcol_exporter_send_failed_spans|otelcol_exporter_send_failed_metric_points|otelcol_exporter_send_failed_log_records)$")'
  # The distroless image has no shell. Collector's file config provider reads a
  # fresh kernel UUID once at config load; the transform above strips the
  # procfs trailing newline. Every container/process restart therefore starts
  # a distinct cumulative series without depending on the restart-stable Pod.
  resource/collector_self:
    attributes:
      - key: service.name
        value: sie-otel-collector
        action: upsert
      - key: service.instance.id
        value: ${file:/proc/sys/kernel/random/uuid}
        action: upsert
      - key: deployment.environment
        value: {{ $deploymentEnvironment | quote }}
        action: upsert
      - key: cloud.region
        value: {{ $cloudRegion | quote }}
        action: upsert
{{- end }}
{{- if and $tracesEnabled $betterStack.enabled }}
  # The application receiver is shared by the bounded set of non-gateway SIE
  # services. Unknown services and gateway claims fail closed before the
  # collector stamps its authoritative environment and region.
  filter/remote_application_traces:
    error_mode: propagate
    traces:
      span:
        - 'resource.attributes["service.name"] != "sie-config" and resource.attributes["service.name"] != "sie-dispatcher" and resource.attributes["service.name"] != "sie-worker" and resource.attributes["service.name"] != "sie-worker-sidecar"'
  # Collector 0.119 parses but does not execute in-place link-slice clearing.
  # Fail closed by dropping the complete linked span on the remote branch;
  # the separate local trace branch remains byte-for-byte unchanged.
  filter/remote_linked_spans:
    error_mode: propagate
    traces:
      span:
        - 'Len(links) > 0'
  # Better Stack gets structural trace data only. Events can contain raw
  # exception text, so remove them before the allowlist transform.
  filter/remote_trace_events:
    error_mode: propagate
    traces:
      spanevent:
        - 'true'
  transform/remote_traces:
    error_mode: propagate
    trace_statements:
      - context: resource
        statements:
          - keep_keys(attributes, ["service.name", "service.instance.id", "deployment.environment", "cloud.region", "service.version"])
          - set(schema_url, "")
      - context: scope
        statements:
          - keep_keys(attributes, [])
          - set(name, "")
          - set(version, "")
          - set(schema_url, "")
      - context: span
        statements:
          - keep_keys(attributes, [])
          - set(status.message, "")
          - set(trace_state, "")
          - set(links, [])
          - 'set(name, "other") where name != "gateway.request" and name != "gateway.publish" and name != "gateway.proxy" and name != "gateway.proxy_chat" and name != "gateway.proxy_request" and name != "gateway.proxy_generate" and name != "sidecar.dispatch" and name != "worker.run_batch" and name != "worker.streaming_processor" and name != "encode" and name != "score" and name != "extract" and name != "generate" and name != "openai_embeddings" and name != "chat_completions" and name != "rerank" and name != "other"'
{{- end }}
{{- if $logsEnabled }}
  # Logs are allowlisted just like metrics are declared: only the fixed,
  # versioned gateway completion event may leave an application process.
  filter/contract_logs:
    error_mode: propagate
    logs:
      log_record:
        - 'resource.attributes["service.name"] != "sie-gateway"'
        - 'attributes["event.name"] != "inference.request.completed"'
        - 'attributes["event.schema.version"] != "1"'
        - 'body != "inference.request.completed"'
        - 'attributes["operation"] != "encode" and attributes["operation"] != "score" and attributes["operation"] != "extract" and attributes["operation"] != "generate" and attributes["operation"] != "embeddings" and attributes["operation"] != "moderations" and attributes["operation"] != "other"'
        - 'attributes["outcome"] != "success" and attributes["outcome"] != "redirect" and attributes["outcome"] != "client_error" and attributes["outcome"] != "server_error" and attributes["outcome"] != "other"'
        - 'attributes["http.status_code"] < 100 or attributes["http.status_code"] > 599'
  transform/contract_logs:
    error_mode: propagate
    log_statements:
      - context: resource
        statements:
          - keep_keys(attributes, ["service.name", "service.instance.id", "deployment.environment", "cloud.region", "service.version"])
          - set(schema_url, "")
      - context: scope
        statements:
          - keep_keys(attributes, [])
          - set(name, "sie-gateway.request-completion")
          - set(version, "")
          - set(schema_url, "")
      - context: log
        statements:
          - keep_keys(attributes, ["event.name", "event.schema.version", "operation", "outcome", "http.status_code"])
          - set(attributes["event.name"], "inference.request.completed")
          - set(attributes["event.schema.version"], "1")
          - set(body, "inference.request.completed")
          - set(severity_text, "INFO")
          - set(severity_number, SEVERITY_NUMBER_INFO)
{{- end }}

exporters:
{{- if $prometheusEnabled }}
  prometheus:
    endpoint: {{ printf "0.0.0.0:%v" $collector.prometheus.port | quote }}
    add_metric_suffixes: true
    enable_open_metrics: true
    metric_expiration: {{ $collector.prometheus.metricExpiration | quote }}
    resource_to_telemetry_conversion:
      enabled: false
{{- end }}
{{- if $traceEndpoint }}
  otlp/traces:
    endpoint: {{ $traceEndpoint | quote }}
    tls:
      insecure: {{ $collector.traces.insecure }}
{{- end }}
{{- if $logEndpoint }}
  otlphttp/logs:
    endpoint: {{ $logEndpoint | quote }}
{{- end }}
{{- if $betterStack.enabled }}
  otlphttp/betterstack:
    endpoint: ${env:BETTERSTACK_OTLP_ENDPOINT}
    headers:
      authorization: Bearer ${env:BETTERSTACK_SOURCE_TOKEN}
    compression: gzip
    timeout: 10s
    sending_queue:
      enabled: true
      blocking: false
      num_consumers: 4
      queue_size: 1000
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 5m
{{- end }}
{{- if and $tracesEnabled (eq (len $localTraceExporters) 1) (eq (index $localTraceExporters 0) "debug/traces") }}
  debug/traces:
    verbosity: basic
{{- end }}
{{- if and $logsEnabled (eq (len $logExporters) 1) (eq (index $logExporters 0) "debug/logs") }}
  debug/logs:
    verbosity: basic
{{- end }}

service:
  extensions: [health_check]
{{- if $betterStack.enabled }}
  telemetry:
    metrics:
      level: normal
      readers:
        - pull:
            exporter:
              prometheus:
                host: 127.0.0.1
                port: 8888
                # Preserve the raw self-metric names consumed by the exact
                # allowlist below when using the explicit reader schema.
                without_type_suffix: true
                without_units: true
{{- end }}
  pipelines:
  {{- if $betterStack.enabled }}
    metrics/self:
      receivers: [prometheus/self]
      processors: [memory_limiter, filter/collector_self_contract, resource/collector_self, transform/collector_self_metrics, batch]
      exporters: [otlphttp/betterstack]
  {{- end }}
  {{- if $tracesEnabled }}
    {{- if gt (len $localTraceExporters) 0 }}
    traces:
      receivers: [otlp/gateway, otlp/application]
      processors: [memory_limiter, batch]
      exporters: {{ toJson $localTraceExporters }}
    {{- end }}
    {{- if $betterStack.enabled }}
    traces/betterstack/gateway:
      receivers: [otlp/gateway]
      processors: [memory_limiter, resource/gateway_identity, filter/remote_linked_spans, filter/remote_trace_events, transform/remote_traces, batch]
      exporters: [otlphttp/betterstack]
    traces/betterstack/application:
      receivers: [otlp/application]
      processors: [memory_limiter, filter/remote_application_traces, resource/application_identity, filter/remote_linked_spans, filter/remote_trace_events, transform/remote_traces, batch]
      exporters: [otlphttp/betterstack]
    {{- end }}
  {{- end }}
  {{- if $metricsEnabled }}
    {{- if $prometheusEnabled }}
    metrics/prometheus/gateway:
      receivers: [otlp/gateway]
      processors: [memory_limiter, filter/prometheus_gateway_contract, resource/gateway_identity, transform/contract_metrics, transform/prometheus_compatibility, resource/prometheus_generation, transform/prometheus_gateway_identity, transform/prometheus_generation, batch]
      exporters: [prometheus]
    metrics/prometheus/application:
      receivers: [otlp/application]
      processors: [memory_limiter, filter/prometheus_application_contract, resource/application_identity, transform/contract_metrics, transform/prometheus_compatibility, resource/prometheus_generation, transform/prometheus_application_identity, transform/prometheus_generation, batch]
      exporters: [prometheus]
    {{- end }}
    {{- if $betterStack.enabled }}
    metrics/betterstack/gateway:
      receivers: [otlp/gateway]
      processors: [memory_limiter, filter/remote_gateway_contract, resource/gateway_identity, transform/contract_metrics, transform/remote_queue_identity, batch]
      exporters: [otlphttp/betterstack]
    metrics/betterstack/application:
      receivers: [otlp/application]
      processors: [memory_limiter, filter/remote_application_contract, resource/application_identity, transform/contract_metrics, batch]
      exporters: [otlphttp/betterstack]
    {{- end }}
  {{- end }}
  {{- if $logsEnabled }}
    logs:
      receivers: [otlp/gateway]
      processors: [memory_limiter, resource/gateway_identity, filter/contract_logs, transform/contract_logs, batch]
      exporters: {{ toJson $logExporters }}
  {{- end }}
{{- end -}}
