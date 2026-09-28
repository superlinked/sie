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
{{- if or $metricsEnabled $logsEnabled (and $tracesEnabled $betterStack.enabled) }}
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
        - 'not IsMatch(name, "^(sie[.]gateway[.]requests|sie[.]gateway[.]request[.]duration|sie[.]gateway[.]admission[.]decisions|sie[.]gateway[.]dispatches|sie[.]gateway[.]dispatch[.]duration|sie[.]gateway[.]config[.]applied_epoch|sie[.]gateway[.]config[.]operations|sie[.]gateway[.]config[.]bootstrap[.]degraded|sie[.]gateway[.]messaging[.]client[.]ready|sie[.]gateway[.]queue[.]publishes|sie[.]gateway[.]queue[.]publish[.]duration|sie[.]gateway[.]queue[.]publish[.]items|sie[.]gateway[.]queue[.]result_waits|sie[.]gateway[.]queue[.]result_wait[.]duration|sie[.]gateway[.]queue[.]result_chunks[.]received|sie[.]gateway[.]queue[.]result_chunk[.]bytes_received|sie[.]gateway[.]queue[.]result_chunk[.]rejections|sie[.]gateway[.]queue[.]result_chunk[.]transfers_completed|sie[.]gateway[.]queue[.]result_chunk[.]duplicates|sie[.]gateway[.]queue[.]result_chunk[.]retry_replacements|sie[.]gateway[.]queue[.]result_chunk[.]stale_retries|sie[.]gateway[.]queue[.]result_chunk[.]reserved_bytes|sie[.]gateway[.]queue[.]events|sie[.]gateway[.]queue[.]lane_admission[.]decisions|sie[.]gateway[.]queue[.]worker_pool[.]events|sie[.]gateway[.]provisioning[.]responses|sie[.]gateway[.]generation[.]events|sie[.]gateway[.]generation[.]ttft|sie[.]gateway[.]generation[.]tpot|sie[.]gateway[.]generation[.]tokens|sie[.]gateway[.]pool[.]pinned_model[.]loaded|sie[.]gateway[.]pending_demand|sie[.]gateway[.]lane[.]queue[.]depth|sie[.]gateway[.]lane[.]queue[.]snapshot[.]timestamp|sie[.]gateway[.]active_lease[.]gpus|sie[.]gateway[.]pool[.]warm_floor|sie[.]gateway[.]rejected[.]requests|sie[.]gateway[.]capacity[.]snapshot[.]timestamp|sie[.]gateway[.]key_snapshot[.]polls|sie[.]gateway[.]settlement[.]confirms|sie[.]gateway[.]settlement[.]confirm[.]duration)$")'
        - 'resource.attributes["service.name"] != "sie-gateway"'
  filter/prometheus_application_contract:
    error_mode: propagate
    metrics:
      metric:
        - 'not IsMatch(name, "^(sie[.]dispatcher[.]invocations|sie[.]dispatcher[.]invocation[.]duration|sie[.]dispatcher[.]inflight|sie[.]dispatcher[.]sealed[.]stage|sie[.]dispatcher[.]sealed[.]stage[.]duration|sie[.]dispatcher[.]sealed[.]resident|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds|sie[.]dispatcher[.]sealed[.]cold_start[.]seconds|sie[.]config[.]requests|sie[.]config[.]request[.]duration|sie[.]config[.]epoch|sie[.]config[.]models|sie[.]config[.]publish|sie[.]config[.]store[.]writes|sie[.]config[.]messaging[.]ready|sie[.]worker[.]queue[.]duration|sie[.]worker[.]scheduler[.]request_batch[.]dispatch_wait|sie[.]worker[.]scheduler[.]request_batch[.]total|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]queue[.]pending_at_dispatch|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]ipc[.]requests|sie[.]worker[.]ipc[.]request[.]duration|sie[.]worker[.]ipc[.]response[.]chunks|sie[.]worker[.]ipc[.]response[.]reconstructed[.]size|sie[.]worker[.]ipc[.]response[.]chunk[.]count|sie[.]worker[.]ipc[.]response[.]chunk[.]reserved|sie[.]worker[.]config[.]applies|sie[.]worker[.]config[.]epoch|sie[.]worker[.]config[.]degraded|sie[.]worker[.]nats[.]operations|sie[.]worker[.]nats[.]delivery[.]attempts|sie[.]worker[.]result[.]transport[.]attempts|sie[.]worker[.]result[.]chunks[.]published|sie[.]worker[.]result[.]chunk[.]size|sie[.]worker[.]payload[.]fetches|sie[.]worker[.]payload[.]fetch[.]duration|sie[.]worker[.]payload[.]size|sie[.]worker[.]gpu[.]slots|sie[.]worker[.]pending[.]items|sie[.]worker[.]pending[.]cost|sie[.]worker[.]inflight[.]batches|sie[.]worker[.]saturated|sie[.]worker[.]ipc[.]capacity|sie[.]worker[.]ipc[.]inflight|sie[.]worker[.]ipc[.]acquire[.]duration|sie[.]worker[.]generation[.]model_loading[.]responses|sie[.]worker[.]shutdown[.]drain[.]duration|sie[.]worker[.]runtime[.]batch[.]size|sie[.]worker[.]runtime[.]batch[.]subgroups|sie[.]worker[.]runtime[.]subgroup[.]size|sie[.]worker[.]requests|sie[.]worker[.]request[.]duration|sie[.]worker[.]inference[.]duration|sie[.]worker[.]units|sie[.]worker[.]model[.]loaded|sie[.]worker[.]model[.]load[.]duration|sie[.]worker[.]model[.]memory|sie[.]worker[.]oom[.]recoveries|sie[.]worker[.]model[.]evictions|sie[.]worker[.]generation[.]worker_wait|sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot|sie[.]worker[.]generation[.]tokens|sie[.]worker[.]generation[.]inflight|sie[.]worker[.]generation[.]kv[.]reserved|sie[.]worker[.]generation[.]kv[.]budget|sie[.]worker[.]generation[.]admission[.]decisions|sie[.]worker[.]generation[.]duplicate_prevented|sie[.]worker[.]generation[.]grammar[.]compile[.]duration|sie[.]worker[.]generation[.]grammar[.]cache[.]lookups|sie[.]worker[.]generation[.]grammar[.]requests|sie[.]worker[.]runtime[.]forward[.]duration|sie[.]worker[.]runtime[.]forward[.]permit[.]wait|sie[.]worker[.]runtime[.]forward[.]concurrent|sie[.]worker[.]runtime[.]forward[.]limit|sie[.]worker[.]work_item[.]age)$")'
        - 'resource.attributes["service.name"] != "sie-config" and resource.attributes["service.name"] != "sie-dispatcher" and resource.attributes["service.name"] != "sie-worker" and resource.attributes["service.name"] != "sie-worker-sidecar"'
        - 'resource.attributes["service.name"] == "sie-dispatcher" and not IsMatch(name, "^(sie[.]dispatcher[.]invocations|sie[.]dispatcher[.]invocation[.]duration|sie[.]dispatcher[.]inflight|sie[.]dispatcher[.]sealed[.]stage|sie[.]dispatcher[.]sealed[.]stage[.]duration|sie[.]dispatcher[.]sealed[.]resident|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds|sie[.]dispatcher[.]sealed[.]cold_start[.]seconds)$")'
        - 'resource.attributes["service.name"] == "sie-config" and not IsMatch(name, "^(sie[.]config[.]requests|sie[.]config[.]request[.]duration|sie[.]config[.]epoch|sie[.]config[.]models|sie[.]config[.]publish|sie[.]config[.]store[.]writes|sie[.]config[.]messaging[.]ready)$")'
        - 'resource.attributes["service.name"] == "sie-worker-sidecar" and not IsMatch(name, "^(sie[.]worker[.]queue[.]duration|sie[.]worker[.]scheduler[.]request_batch[.]dispatch_wait|sie[.]worker[.]scheduler[.]request_batch[.]total|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]ipc[.]requests|sie[.]worker[.]ipc[.]request[.]duration|sie[.]worker[.]ipc[.]response[.]chunks|sie[.]worker[.]ipc[.]response[.]reconstructed[.]size|sie[.]worker[.]ipc[.]response[.]chunk[.]count|sie[.]worker[.]ipc[.]response[.]chunk[.]reserved|sie[.]worker[.]config[.]applies|sie[.]worker[.]config[.]epoch|sie[.]worker[.]config[.]degraded|sie[.]worker[.]nats[.]operations|sie[.]worker[.]nats[.]delivery[.]attempts|sie[.]worker[.]result[.]transport[.]attempts|sie[.]worker[.]result[.]chunks[.]published|sie[.]worker[.]result[.]chunk[.]size|sie[.]worker[.]payload[.]fetches|sie[.]worker[.]payload[.]fetch[.]duration|sie[.]worker[.]payload[.]size|sie[.]worker[.]gpu[.]slots|sie[.]worker[.]pending[.]items|sie[.]worker[.]pending[.]cost|sie[.]worker[.]inflight[.]batches|sie[.]worker[.]saturated|sie[.]worker[.]ipc[.]capacity|sie[.]worker[.]ipc[.]inflight|sie[.]worker[.]ipc[.]acquire[.]duration|sie[.]worker[.]generation[.]model_loading[.]responses|sie[.]worker[.]shutdown[.]drain[.]duration|sie[.]worker[.]work_item[.]age)$")'
        - 'resource.attributes["service.name"] == "sie-worker" and not IsMatch(name, "^(sie[.]worker[.]queue[.]duration|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]queue[.]pending_at_dispatch|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]runtime[.]batch[.]size|sie[.]worker[.]runtime[.]batch[.]subgroups|sie[.]worker[.]runtime[.]subgroup[.]size|sie[.]worker[.]requests|sie[.]worker[.]request[.]duration|sie[.]worker[.]inference[.]duration|sie[.]worker[.]units|sie[.]worker[.]model[.]loaded|sie[.]worker[.]model[.]load[.]duration|sie[.]worker[.]model[.]memory|sie[.]worker[.]oom[.]recoveries|sie[.]worker[.]model[.]evictions|sie[.]worker[.]generation[.]worker_wait|sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot|sie[.]worker[.]generation[.]tokens|sie[.]worker[.]generation[.]inflight|sie[.]worker[.]generation[.]kv[.]reserved|sie[.]worker[.]generation[.]kv[.]budget|sie[.]worker[.]generation[.]admission[.]decisions|sie[.]worker[.]generation[.]duplicate_prevented|sie[.]worker[.]generation[.]grammar[.]compile[.]duration|sie[.]worker[.]generation[.]grammar[.]cache[.]lookups|sie[.]worker[.]generation[.]grammar[.]requests|sie[.]worker[.]runtime[.]forward[.]duration|sie[.]worker[.]runtime[.]forward[.]permit[.]wait|sie[.]worker[.]runtime[.]forward[.]concurrent|sie[.]worker[.]runtime[.]forward[.]limit)$")'
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
          - 'set(unit, "s") where IsMatch(name, "^(sie[.]gateway[.]request[.]duration|sie[.]gateway[.]dispatch[.]duration|sie[.]gateway[.]queue[.]publish[.]duration|sie[.]gateway[.]queue[.]result_wait[.]duration|sie[.]gateway[.]generation[.]ttft|sie[.]gateway[.]generation[.]tpot|sie[.]dispatcher[.]invocation[.]duration|sie[.]dispatcher[.]sealed[.]stage[.]duration|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds|sie[.]dispatcher[.]sealed[.]cold_start[.]seconds|sie[.]config[.]request[.]duration|sie[.]worker[.]queue[.]duration|sie[.]worker[.]scheduler[.]request_batch[.]dispatch_wait|sie[.]worker[.]scheduler[.]request_batch[.]total|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]ipc[.]request[.]duration|sie[.]worker[.]payload[.]fetch[.]duration|sie[.]worker[.]ipc[.]acquire[.]duration|sie[.]worker[.]shutdown[.]drain[.]duration|sie[.]worker[.]request[.]duration|sie[.]worker[.]inference[.]duration|sie[.]worker[.]model[.]load[.]duration|sie[.]worker[.]generation[.]worker_wait|sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot|sie[.]worker[.]generation[.]grammar[.]compile[.]duration|sie[.]worker[.]runtime[.]forward[.]duration|sie[.]worker[.]runtime[.]forward[.]permit[.]wait|sie[.]gateway[.]lane[.]queue[.]snapshot[.]timestamp|sie[.]gateway[.]capacity[.]snapshot[.]timestamp|sie[.]gateway[.]settlement[.]confirm[.]duration|sie[.]worker[.]work_item[.]age)$")'
          - 'set(unit, "{epoch}") where IsMatch(name, "^(sie[.]gateway[.]config[.]applied_epoch|sie[.]config[.]epoch|sie[.]worker[.]config[.]epoch)$")'
          - 'set(unit, "{operation}") where IsMatch(name, "^(sie[.]gateway[.]config[.]operations|sie[.]config[.]publish|sie[.]config[.]store[.]writes|sie[.]worker[.]nats[.]operations)$")'
          - 'set(unit, "1") where IsMatch(name, "^(sie[.]gateway[.]config[.]bootstrap[.]degraded|sie[.]gateway[.]messaging[.]client[.]ready|sie[.]config[.]messaging[.]ready|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]config[.]degraded|sie[.]worker[.]saturated|sie[.]gateway[.]pool[.]pinned_model[.]loaded)$")'
          - 'set(unit, "{publish}") where name == "sie.gateway.queue.publishes"'
          - 'set(unit, "{item}") where IsMatch(name, "^(sie[.]gateway[.]queue[.]publish[.]items|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]queue[.]pending_at_dispatch|sie[.]worker[.]pending[.]items|sie[.]worker[.]runtime[.]batch[.]size|sie[.]worker[.]runtime[.]subgroup[.]size|sie[.]worker[.]requests|sie[.]gateway[.]lane[.]queue[.]depth)$")'
          - 'set(unit, "{wait}") where name == "sie.gateway.queue.result_waits"'
          - 'set(unit, "{poll}") where name == "sie.gateway.key_snapshot.polls"'
          - 'set(unit, "{confirm}") where name == "sie.gateway.settlement.confirms"'
          - 'set(unit, "{chunk}") where IsMatch(name, "^(sie[.]gateway[.]queue[.]result_chunks[.]received|sie[.]gateway[.]queue[.]result_chunk[.]duplicates|sie[.]gateway[.]queue[.]result_chunk[.]stale_retries|sie[.]worker[.]ipc[.]response[.]chunk[.]count|sie[.]worker[.]result[.]chunks[.]published)$")'
          - 'set(unit, "{transfer}") where IsMatch(name, "^(sie[.]gateway[.]queue[.]result_chunk[.]transfers_completed|sie[.]gateway[.]queue[.]result_chunk[.]retry_replacements|sie[.]worker[.]ipc[.]response[.]chunks)$")'
          - 'set(unit, "{rejection}") where name == "sie.gateway.queue.result_chunk.rejections"'
          - 'set(unit, "{attempt}") where name == "sie.worker.result.transport.attempts"'
          - 'set(unit, "By") where IsMatch(name, "^(sie[.]gateway[.]queue[.]result_chunk[.]bytes_received|sie[.]gateway[.]queue[.]result_chunk[.]reserved_bytes|sie[.]worker[.]ipc[.]response[.]reconstructed[.]size|sie[.]worker[.]ipc[.]response[.]chunk[.]reserved|sie[.]worker[.]result[.]chunk[.]size)$")'
          - 'set(unit, "{event}") where IsMatch(name, "^(sie[.]gateway[.]queue[.]events|sie[.]gateway[.]queue[.]worker_pool[.]events|sie[.]gateway[.]generation[.]events)$")'
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
          - 'set(unit, "{decision}") where name == "sie.gateway.queue.lane_admission.decisions"'
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
          - 'keep_keys(attributes, ["pool", "event"]) where metric.name == "sie.gateway.queue.worker_pool.events"'
          - 'keep_keys(attributes, ["surface", "http.status_code"]) where metric.name == "sie.gateway.provisioning.responses"'
          - 'keep_keys(attributes, ["event", "reason", "outcome"]) where metric.name == "sie.gateway.generation.events"'
          - 'keep_keys(attributes, ["operation"]) where IsMatch(metric.name, "^(sie[.]gateway[.]generation[.]ttft|sie[.]gateway[.]generation[.]tpot)$")'
          - 'keep_keys(attributes, ["operation", "token.kind"]) where metric.name == "sie.gateway.generation.tokens"'
          - 'keep_keys(attributes, ["outcome"]) where IsMatch(metric.name, "^(sie[.]gateway[.]key_snapshot[.]polls|sie[.]gateway[.]settlement[.]confirms|sie[.]gateway[.]settlement[.]confirm[.]duration)$")'
          - 'keep_keys(attributes, ["operation", "lane"]) where metric.name == "sie.worker.work_item.age"'
          - 'keep_keys(attributes, ["operation", "dispatch.path", "outcome", "lane"]) where IsMatch(metric.name, "^(sie[.]dispatcher[.]invocations|sie[.]dispatcher[.]invocation[.]duration)$")'
          - 'keep_keys(attributes, ["operation", "dispatch.path", "lane"]) where metric.name == "sie.dispatcher.inflight"'
          - 'keep_keys(attributes, ["outcome"]) where IsMatch(metric.name, "^(sie[.]dispatcher[.]sealed[.]stage|sie[.]dispatcher[.]sealed[.]stage[.]duration)$")'
          - 'keep_keys(attributes, []) where IsMatch(metric.name, "^(sie[.]dispatcher[.]sealed[.]resident|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds)$")'
          - 'keep_keys(attributes, ["gpu_class"]) where IsMatch(metric.name, "^(sie[.]dispatcher[.]sealed[.]cold_start[.]seconds)$")'
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
          - 'keep_keys(attributes, ["grammar", "backend", "lane", "model", "profile"]) where IsMatch(metric.name, "^(sie[.]worker[.]generation[.]worker_wait|sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot)$")'
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
          - 'keep_keys(attributes, ["pool", "machine_profile", "bundle", "outcome"]) where metric.name == "sie.gateway.queue.lane_admission.decisions"'
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
        - 'not IsMatch(name, "^(sie[.]gateway[.]requests|sie[.]gateway[.]request[.]duration|sie[.]gateway[.]admission[.]decisions|sie[.]gateway[.]dispatches|sie[.]gateway[.]dispatch[.]duration|sie[.]gateway[.]config[.]applied_epoch|sie[.]gateway[.]config[.]operations|sie[.]gateway[.]config[.]bootstrap[.]degraded|sie[.]gateway[.]messaging[.]client[.]ready|sie[.]gateway[.]queue[.]publishes|sie[.]gateway[.]queue[.]publish[.]duration|sie[.]gateway[.]queue[.]publish[.]items|sie[.]gateway[.]queue[.]result_waits|sie[.]gateway[.]queue[.]result_wait[.]duration|sie[.]gateway[.]queue[.]result_chunks[.]received|sie[.]gateway[.]queue[.]result_chunk[.]bytes_received|sie[.]gateway[.]queue[.]result_chunk[.]rejections|sie[.]gateway[.]queue[.]result_chunk[.]transfers_completed|sie[.]gateway[.]queue[.]result_chunk[.]duplicates|sie[.]gateway[.]queue[.]result_chunk[.]retry_replacements|sie[.]gateway[.]queue[.]result_chunk[.]stale_retries|sie[.]gateway[.]queue[.]result_chunk[.]reserved_bytes|sie[.]gateway[.]queue[.]events|sie[.]gateway[.]queue[.]lane_admission[.]decisions|sie[.]gateway[.]provisioning[.]responses|sie[.]gateway[.]generation[.]events|sie[.]gateway[.]generation[.]ttft|sie[.]gateway[.]generation[.]tpot|sie[.]gateway[.]generation[.]tokens|sie[.]gateway[.]pending_demand|sie[.]gateway[.]lane[.]queue[.]depth|sie[.]gateway[.]lane[.]queue[.]snapshot[.]timestamp|sie[.]gateway[.]active_lease[.]gpus|sie[.]gateway[.]pool[.]warm_floor|sie[.]gateway[.]rejected[.]requests|sie[.]gateway[.]key_snapshot[.]polls|sie[.]gateway[.]settlement[.]confirms|sie[.]gateway[.]settlement[.]confirm[.]duration)$")'
        - 'resource.attributes["service.name"] != "sie-gateway"'
  filter/remote_application_contract:
    error_mode: propagate
    metrics:
      metric:
        - 'not IsMatch(name, "^(sie[.]dispatcher[.]invocations|sie[.]dispatcher[.]invocation[.]duration|sie[.]dispatcher[.]inflight|sie[.]dispatcher[.]sealed[.]stage|sie[.]dispatcher[.]sealed[.]stage[.]duration|sie[.]dispatcher[.]sealed[.]resident|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds|sie[.]dispatcher[.]sealed[.]cold_start[.]seconds|sie[.]config[.]requests|sie[.]config[.]request[.]duration|sie[.]config[.]epoch|sie[.]config[.]models|sie[.]config[.]publish|sie[.]config[.]store[.]writes|sie[.]config[.]messaging[.]ready|sie[.]worker[.]queue[.]duration|sie[.]worker[.]scheduler[.]request_batch[.]dispatch_wait|sie[.]worker[.]scheduler[.]request_batch[.]total|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]queue[.]pending_at_dispatch|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]ipc[.]requests|sie[.]worker[.]ipc[.]request[.]duration|sie[.]worker[.]ipc[.]response[.]chunks|sie[.]worker[.]ipc[.]response[.]reconstructed[.]size|sie[.]worker[.]ipc[.]response[.]chunk[.]count|sie[.]worker[.]ipc[.]response[.]chunk[.]reserved|sie[.]worker[.]config[.]applies|sie[.]worker[.]config[.]epoch|sie[.]worker[.]config[.]degraded|sie[.]worker[.]nats[.]operations|sie[.]worker[.]nats[.]delivery[.]attempts|sie[.]worker[.]result[.]transport[.]attempts|sie[.]worker[.]result[.]chunks[.]published|sie[.]worker[.]result[.]chunk[.]size|sie[.]worker[.]payload[.]fetches|sie[.]worker[.]payload[.]fetch[.]duration|sie[.]worker[.]payload[.]size|sie[.]worker[.]gpu[.]slots|sie[.]worker[.]pending[.]items|sie[.]worker[.]pending[.]cost|sie[.]worker[.]inflight[.]batches|sie[.]worker[.]saturated|sie[.]worker[.]ipc[.]capacity|sie[.]worker[.]ipc[.]inflight|sie[.]worker[.]ipc[.]acquire[.]duration|sie[.]worker[.]generation[.]model_loading[.]responses|sie[.]worker[.]shutdown[.]drain[.]duration|sie[.]worker[.]runtime[.]batch[.]size|sie[.]worker[.]runtime[.]batch[.]subgroups|sie[.]worker[.]runtime[.]subgroup[.]size|sie[.]worker[.]requests|sie[.]worker[.]request[.]duration|sie[.]worker[.]inference[.]duration|sie[.]worker[.]units|sie[.]worker[.]model[.]loaded|sie[.]worker[.]model[.]load[.]duration|sie[.]worker[.]model[.]memory|sie[.]worker[.]oom[.]recoveries|sie[.]worker[.]model[.]evictions|sie[.]worker[.]generation[.]worker_wait|sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot|sie[.]worker[.]generation[.]tokens|sie[.]worker[.]generation[.]inflight|sie[.]worker[.]generation[.]kv[.]reserved|sie[.]worker[.]generation[.]kv[.]budget|sie[.]worker[.]generation[.]admission[.]decisions|sie[.]worker[.]generation[.]duplicate_prevented|sie[.]worker[.]generation[.]grammar[.]compile[.]duration|sie[.]worker[.]generation[.]grammar[.]cache[.]lookups|sie[.]worker[.]generation[.]grammar[.]requests|sie[.]worker[.]runtime[.]forward[.]duration|sie[.]worker[.]runtime[.]forward[.]permit[.]wait|sie[.]worker[.]runtime[.]forward[.]concurrent|sie[.]worker[.]runtime[.]forward[.]limit|sie[.]worker[.]work_item[.]age)$")'
        - 'resource.attributes["service.name"] != "sie-config" and resource.attributes["service.name"] != "sie-dispatcher" and resource.attributes["service.name"] != "sie-worker" and resource.attributes["service.name"] != "sie-worker-sidecar"'
        - 'resource.attributes["service.name"] == "sie-dispatcher" and not IsMatch(name, "^(sie[.]dispatcher[.]invocations|sie[.]dispatcher[.]invocation[.]duration|sie[.]dispatcher[.]inflight|sie[.]dispatcher[.]sealed[.]stage|sie[.]dispatcher[.]sealed[.]stage[.]duration|sie[.]dispatcher[.]sealed[.]resident|sie[.]dispatcher[.]sealed[.]sandbox[.]seconds|sie[.]dispatcher[.]sealed[.]cold_start[.]seconds)$")'
        - 'resource.attributes["service.name"] == "sie-config" and not IsMatch(name, "^(sie[.]config[.]requests|sie[.]config[.]request[.]duration|sie[.]config[.]epoch|sie[.]config[.]models|sie[.]config[.]publish|sie[.]config[.]store[.]writes|sie[.]config[.]messaging[.]ready)$")'
        - 'resource.attributes["service.name"] == "sie-worker-sidecar" and not IsMatch(name, "^(sie[.]worker[.]queue[.]duration|sie[.]worker[.]scheduler[.]request_batch[.]dispatch_wait|sie[.]worker[.]scheduler[.]request_batch[.]total|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]ipc[.]requests|sie[.]worker[.]ipc[.]request[.]duration|sie[.]worker[.]ipc[.]response[.]chunks|sie[.]worker[.]ipc[.]response[.]reconstructed[.]size|sie[.]worker[.]ipc[.]response[.]chunk[.]count|sie[.]worker[.]ipc[.]response[.]chunk[.]reserved|sie[.]worker[.]config[.]applies|sie[.]worker[.]config[.]epoch|sie[.]worker[.]config[.]degraded|sie[.]worker[.]nats[.]operations|sie[.]worker[.]nats[.]delivery[.]attempts|sie[.]worker[.]result[.]transport[.]attempts|sie[.]worker[.]result[.]chunks[.]published|sie[.]worker[.]result[.]chunk[.]size|sie[.]worker[.]payload[.]fetches|sie[.]worker[.]payload[.]fetch[.]duration|sie[.]worker[.]payload[.]size|sie[.]worker[.]gpu[.]slots|sie[.]worker[.]pending[.]items|sie[.]worker[.]pending[.]cost|sie[.]worker[.]inflight[.]batches|sie[.]worker[.]saturated|sie[.]worker[.]ipc[.]capacity|sie[.]worker[.]ipc[.]inflight|sie[.]worker[.]ipc[.]acquire[.]duration|sie[.]worker[.]generation[.]model_loading[.]responses|sie[.]worker[.]shutdown[.]drain[.]duration|sie[.]worker[.]work_item[.]age)$")'
        - 'resource.attributes["service.name"] == "sie-worker" and not IsMatch(name, "^(sie[.]worker[.]queue[.]duration|sie[.]worker[.]queue[.]depth|sie[.]worker[.]batch[.]size|sie[.]worker[.]batch[.]cost|sie[.]worker[.]batch[.]fill_ratio|sie[.]worker[.]queue[.]pending_at_dispatch|sie[.]worker[.]scheduler[.]adaptive[.]wait|sie[.]worker[.]scheduler[.]adaptive[.]cost|sie[.]worker[.]scheduler[.]adaptive[.]p50|sie[.]worker[.]scheduler[.]starvation[.]resets|sie[.]worker[.]runtime[.]batch[.]size|sie[.]worker[.]runtime[.]batch[.]subgroups|sie[.]worker[.]runtime[.]subgroup[.]size|sie[.]worker[.]requests|sie[.]worker[.]request[.]duration|sie[.]worker[.]inference[.]duration|sie[.]worker[.]units|sie[.]worker[.]model[.]loaded|sie[.]worker[.]model[.]load[.]duration|sie[.]worker[.]model[.]memory|sie[.]worker[.]oom[.]recoveries|sie[.]worker[.]model[.]evictions|sie[.]worker[.]generation[.]worker_wait|sie[.]worker[.]generation[.]ttft|sie[.]worker[.]generation[.]tpot|sie[.]worker[.]generation[.]tokens|sie[.]worker[.]generation[.]inflight|sie[.]worker[.]generation[.]kv[.]reserved|sie[.]worker[.]generation[.]kv[.]budget|sie[.]worker[.]generation[.]admission[.]decisions|sie[.]worker[.]generation[.]duplicate_prevented|sie[.]worker[.]generation[.]grammar[.]compile[.]duration|sie[.]worker[.]generation[.]grammar[.]cache[.]lookups|sie[.]worker[.]generation[.]grammar[.]requests|sie[.]worker[.]runtime[.]forward[.]duration|sie[.]worker[.]runtime[.]forward[.]permit[.]wait|sie[.]worker[.]runtime[.]forward[.]concurrent|sie[.]worker[.]runtime[.]forward[.]limit)$")'
{{- end }}
{{- if $betterStack.enabled }}
  # Scalar lookups select one value; keep_keys alone retains duplicate OTLP
  # keys. Rebuild only on remote export after per-metric field pruning.
  transform/remote_metric_scalars:
    error_mode: propagate
    metric_statements:
      - context: resource
        statements:
          - keep_keys(cache, [])
          - 'set(cache["service.name"], attributes["service.name"]) where IsString(attributes["service.name"])'
          - 'set(cache["service.instance.id"], attributes["service.instance.id"]) where IsString(attributes["service.instance.id"])'
          - 'set(cache["deployment.environment"], attributes["deployment.environment"]) where IsString(attributes["deployment.environment"])'
          - 'set(cache["cloud.region"], attributes["cloud.region"]) where IsString(attributes["cloud.region"])'
          - 'set(cache["service.version"], attributes["service.version"]) where IsString(attributes["service.version"])'
          - keep_keys(attributes, [])
          - set(attributes["service.name"], cache["service.name"])
          - set(attributes["service.instance.id"], cache["service.instance.id"])
          - set(attributes["deployment.environment"], cache["deployment.environment"])
          - set(attributes["cloud.region"], cache["cloud.region"])
          - set(attributes["service.version"], cache["service.version"])
      - context: datapoint
        statements:
          - keep_keys(cache, [])
          - 'set(cache["backend"], attributes["backend"]) where IsString(attributes["backend"]) or IsInt(attributes["backend"]) or IsDouble(attributes["backend"]) or IsBool(attributes["backend"])'
          - 'set(cache["bundle"], attributes["bundle"]) where IsString(attributes["bundle"]) or IsInt(attributes["bundle"]) or IsDouble(attributes["bundle"]) or IsBool(attributes["bundle"])'
          - 'set(cache["dispatch.path"], attributes["dispatch.path"]) where IsString(attributes["dispatch.path"]) or IsInt(attributes["dispatch.path"]) or IsDouble(attributes["dispatch.path"]) or IsBool(attributes["dispatch.path"])'
          - 'set(cache["event"], attributes["event"]) where IsString(attributes["event"]) or IsInt(attributes["event"]) or IsDouble(attributes["event"]) or IsBool(attributes["event"])'
          - 'set(cache["fallback.reason"], attributes["fallback.reason"]) where IsString(attributes["fallback.reason"]) or IsInt(attributes["fallback.reason"]) or IsDouble(attributes["fallback.reason"]) or IsBool(attributes["fallback.reason"])'
          - 'set(cache["flush.reason"], attributes["flush.reason"]) where IsString(attributes["flush.reason"]) or IsInt(attributes["flush.reason"]) or IsDouble(attributes["flush.reason"]) or IsBool(attributes["flush.reason"])'
          - 'set(cache["gpu_class"], attributes["gpu_class"]) where IsString(attributes["gpu_class"]) or IsInt(attributes["gpu_class"]) or IsDouble(attributes["gpu_class"]) or IsBool(attributes["gpu_class"])'
          - 'set(cache["grammar"], attributes["grammar"]) where IsString(attributes["grammar"]) or IsInt(attributes["grammar"]) or IsDouble(attributes["grammar"]) or IsBool(attributes["grammar"])'
          - 'set(cache["grammar.backend"], attributes["grammar.backend"]) where IsString(attributes["grammar.backend"]) or IsInt(attributes["grammar.backend"]) or IsDouble(attributes["grammar.backend"]) or IsBool(attributes["grammar.backend"])'
          - 'set(cache["http.method"], attributes["http.method"]) where IsString(attributes["http.method"]) or IsInt(attributes["http.method"]) or IsDouble(attributes["http.method"]) or IsBool(attributes["http.method"])'
          - 'set(cache["http.route"], attributes["http.route"]) where IsString(attributes["http.route"]) or IsInt(attributes["http.route"]) or IsDouble(attributes["http.route"]) or IsBool(attributes["http.route"])'
          - 'set(cache["http.status_code"], attributes["http.status_code"]) where IsString(attributes["http.status_code"]) or IsInt(attributes["http.status_code"]) or IsDouble(attributes["http.status_code"]) or IsBool(attributes["http.status_code"])'
          - 'set(cache["input.source"], attributes["input.source"]) where IsString(attributes["input.source"]) or IsInt(attributes["input.source"]) or IsDouble(attributes["input.source"]) or IsBool(attributes["input.source"])'
          - 'set(cache["kind"], attributes["kind"]) where IsString(attributes["kind"]) or IsInt(attributes["kind"]) or IsDouble(attributes["kind"]) or IsBool(attributes["kind"])'
          - 'set(cache["lane"], attributes["lane"]) where IsString(attributes["lane"]) or IsInt(attributes["lane"]) or IsDouble(attributes["lane"]) or IsBool(attributes["lane"])'
          - 'set(cache["machine_profile"], attributes["machine_profile"]) where IsString(attributes["machine_profile"]) or IsInt(attributes["machine_profile"]) or IsDouble(attributes["machine_profile"]) or IsBool(attributes["machine_profile"])'
          - 'set(cache["method"], attributes["method"]) where IsString(attributes["method"]) or IsInt(attributes["method"]) or IsDouble(attributes["method"]) or IsBool(attributes["method"])'
          - 'set(cache["mode"], attributes["mode"]) where IsString(attributes["mode"]) or IsInt(attributes["mode"]) or IsDouble(attributes["mode"]) or IsBool(attributes["mode"])'
          - 'set(cache["model"], attributes["model"]) where IsString(attributes["model"]) or IsInt(attributes["model"]) or IsDouble(attributes["model"]) or IsBool(attributes["model"])'
          - 'set(cache["operation"], attributes["operation"]) where IsString(attributes["operation"]) or IsInt(attributes["operation"]) or IsDouble(attributes["operation"]) or IsBool(attributes["operation"])'
          - 'set(cache["outcome"], attributes["outcome"]) where IsString(attributes["outcome"]) or IsInt(attributes["outcome"]) or IsDouble(attributes["outcome"]) or IsBool(attributes["outcome"])'
          - 'set(cache["output.path"], attributes["output.path"]) where IsString(attributes["output.path"]) or IsInt(attributes["output.path"]) or IsDouble(attributes["output.path"]) or IsBool(attributes["output.path"])'
          - 'set(cache["phase"], attributes["phase"]) where IsString(attributes["phase"]) or IsInt(attributes["phase"]) or IsDouble(attributes["phase"]) or IsBool(attributes["phase"])'
          - 'set(cache["pool"], attributes["pool"]) where IsString(attributes["pool"]) or IsInt(attributes["pool"]) or IsDouble(attributes["pool"]) or IsBool(attributes["pool"])'
          - 'set(cache["profile"], attributes["profile"]) where IsString(attributes["profile"]) or IsInt(attributes["profile"]) or IsDouble(attributes["profile"]) or IsBool(attributes["profile"])'
          - 'set(cache["reason"], attributes["reason"]) where IsString(attributes["reason"]) or IsInt(attributes["reason"]) or IsDouble(attributes["reason"]) or IsBool(attributes["reason"])'
          - 'set(cache["redelivered"], attributes["redelivered"]) where IsString(attributes["redelivered"]) or IsInt(attributes["redelivered"]) or IsDouble(attributes["redelivered"]) or IsBool(attributes["redelivered"])'
          - 'set(cache["result"], attributes["result"]) where IsString(attributes["result"]) or IsInt(attributes["result"]) or IsDouble(attributes["result"]) or IsBool(attributes["result"])'
          - 'set(cache["scaling_action"], attributes["scaling_action"]) where IsString(attributes["scaling_action"]) or IsInt(attributes["scaling_action"]) or IsDouble(attributes["scaling_action"]) or IsBool(attributes["scaling_action"])'
          - 'set(cache["source"], attributes["source"]) where IsString(attributes["source"]) or IsInt(attributes["source"]) or IsDouble(attributes["source"]) or IsBool(attributes["source"])'
          - 'set(cache["stage"], attributes["stage"]) where IsString(attributes["stage"]) or IsInt(attributes["stage"]) or IsDouble(attributes["stage"]) or IsBool(attributes["stage"])'
          - 'set(cache["state"], attributes["state"]) where IsString(attributes["state"]) or IsInt(attributes["state"]) or IsDouble(attributes["state"]) or IsBool(attributes["state"])'
          - 'set(cache["strategy"], attributes["strategy"]) where IsString(attributes["strategy"]) or IsInt(attributes["strategy"]) or IsDouble(attributes["strategy"]) or IsBool(attributes["strategy"])'
          - 'set(cache["surface"], attributes["surface"]) where IsString(attributes["surface"]) or IsInt(attributes["surface"]) or IsDouble(attributes["surface"]) or IsBool(attributes["surface"])'
          - 'set(cache["token.kind"], attributes["token.kind"]) where IsString(attributes["token.kind"]) or IsInt(attributes["token.kind"]) or IsDouble(attributes["token.kind"]) or IsBool(attributes["token.kind"])'
          - 'set(cache["token.type"], attributes["token.type"]) where IsString(attributes["token.type"]) or IsInt(attributes["token.type"]) or IsDouble(attributes["token.type"]) or IsBool(attributes["token.type"])'
          - 'set(cache["transport"], attributes["transport"]) where IsString(attributes["transport"]) or IsInt(attributes["transport"]) or IsDouble(attributes["transport"]) or IsBool(attributes["transport"])'
          - 'set(cache["unit.type"], attributes["unit.type"]) where IsString(attributes["unit.type"]) or IsInt(attributes["unit.type"]) or IsDouble(attributes["unit.type"]) or IsBool(attributes["unit.type"])'
          - keep_keys(attributes, [])
          - set(attributes["backend"], cache["backend"])
          - set(attributes["bundle"], cache["bundle"])
          - set(attributes["dispatch.path"], cache["dispatch.path"])
          - set(attributes["event"], cache["event"])
          - set(attributes["fallback.reason"], cache["fallback.reason"])
          - set(attributes["flush.reason"], cache["flush.reason"])
          - set(attributes["gpu_class"], cache["gpu_class"])
          - set(attributes["grammar"], cache["grammar"])
          - set(attributes["grammar.backend"], cache["grammar.backend"])
          - set(attributes["http.method"], cache["http.method"])
          - set(attributes["http.route"], cache["http.route"])
          - set(attributes["http.status_code"], cache["http.status_code"])
          - set(attributes["input.source"], cache["input.source"])
          - set(attributes["kind"], cache["kind"])
          - set(attributes["lane"], cache["lane"])
          - set(attributes["machine_profile"], cache["machine_profile"])
          - set(attributes["method"], cache["method"])
          - set(attributes["mode"], cache["mode"])
          - set(attributes["model"], cache["model"])
          - set(attributes["operation"], cache["operation"])
          - set(attributes["outcome"], cache["outcome"])
          - set(attributes["output.path"], cache["output.path"])
          - set(attributes["phase"], cache["phase"])
          - set(attributes["pool"], cache["pool"])
          - set(attributes["profile"], cache["profile"])
          - set(attributes["reason"], cache["reason"])
          - set(attributes["redelivered"], cache["redelivered"])
          - set(attributes["result"], cache["result"])
          - set(attributes["scaling_action"], cache["scaling_action"])
          - set(attributes["source"], cache["source"])
          - set(attributes["stage"], cache["stage"])
          - set(attributes["state"], cache["state"])
          - set(attributes["strategy"], cache["strategy"])
          - set(attributes["surface"], cache["surface"])
          - set(attributes["token.kind"], cache["token.kind"])
          - set(attributes["token.type"], cache["token.type"])
          - set(attributes["transport"], cache["transport"])
          - set(attributes["unit.type"], cache["unit.type"])
  # Collector implementation health is isolated from the application
  # contract and reduced to nine stable families before remote export.
  filter/collector_self_contract:
    error_mode: propagate
    metrics:
      metric:
        - 'not IsMatch(name, "^(up|otelcol_exporter_queue_size|otelcol_exporter_queue_capacity|otelcol_receiver_refused_spans|otelcol_exporter_send_failed_spans|otelcol_receiver_refused_metric_points|otelcol_exporter_send_failed_metric_points|otelcol_receiver_refused_log_records|otelcol_exporter_send_failed_log_records|otelcol_receiver_refused_spans_total|otelcol_exporter_send_failed_spans_total|otelcol_receiver_refused_metric_points_total|otelcol_exporter_send_failed_metric_points_total|otelcol_receiver_refused_log_records_total|otelcol_exporter_send_failed_log_records_total|otelcol_processor_filter_spans_filtered|otelcol_processor_filter_spans_filtered_total)$")'
      datapoint:
        - 'IsMatch(metric.name, "^otelcol_processor_filter_spans_filtered(_total)?$") and attributes["filter"] != "filter/remote_linked_spans"'
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
          - 'set(name, "otelcol_processor_filter_spans_filtered") where name == "otelcol_processor_filter_spans_filtered_total"'
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
          - 'keep_keys(attributes, []) where metric.name == "otelcol_processor_filter_spans_filtered"'
          - 'set(attributes["filter"], "filter/remote_linked_spans") where metric.name == "otelcol_processor_filter_spans_filtered"'
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
  # the separate local trace branch retains the original. Safe per-request
  # timing leaves (.request) have no links and survive this same guard.
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
          - set(cache["service.name"], attributes["service.name"]) where IsString(attributes["service.name"])
          - set(cache["service.instance.id"], attributes["service.instance.id"]) where IsString(attributes["service.instance.id"])
          - set(cache["deployment.environment"], attributes["deployment.environment"]) where IsString(attributes["deployment.environment"])
          - set(cache["cloud.region"], attributes["cloud.region"]) where IsString(attributes["cloud.region"])
          - set(cache["service.version"], attributes["service.version"]) where IsString(attributes["service.version"])
          - keep_keys(attributes, [])
          - set(attributes["service.name"], cache["service.name"])
          - set(attributes["service.instance.id"], cache["service.instance.id"])
          - set(attributes["deployment.environment"], cache["deployment.environment"])
          - set(attributes["cloud.region"], cache["cloud.region"])
          - set(attributes["service.version"], cache["service.version"])
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
          - 'set(name, "other") where name != "gateway.request" and name != "gateway.response_body" and name != "gateway.generation_stream" and name != "gateway.dispatch" and name != "gateway.dispatch.ipc" and name != "gateway.dispatch.i6pn" and name != "dispatcher.dispatch" and name != "dispatcher.modal.remote" and name != "dispatcher.modal.stream" and name != "dispatcher.modal.spawn" and name != "dispatcher.websocket" and name != "dispatcher.i6pn" and name != "dispatcher.fallback" and name != "sidecar.local_ingest" and name != "worker.local_ingest" and name != "gateway.publish" and name != "gateway.proxy" and name != "gateway.proxy_chat" and name != "gateway.proxy_request" and name != "gateway.proxy_generate" and name != "sidecar.dispatch" and name != "worker.run_batch" and name != "worker.run_batch.request" and name != "sidecar.dispatch.request" and name != "worker.streaming_processor" and name != "encode" and name != "score" and name != "extract" and name != "generate" and name != "openai_embeddings" and name != "chat_completions" and name != "rerank" and name != "other"'
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
        # Keep v1 readable while old gateway pods drain during a rolling update.
        - 'attributes["event.schema.version"] != "1" and attributes["event.schema.version"] != "2"'
        - 'body != "inference.request.completed"'
        - 'attributes["operation"] != "encode" and attributes["operation"] != "score" and attributes["operation"] != "extract" and attributes["operation"] != "generate" and attributes["operation"] != "embeddings" and attributes["operation"] != "moderations" and attributes["operation"] != "other"'
        - 'attributes["outcome"] != "success" and attributes["outcome"] != "redirect" and attributes["outcome"] != "client_error" and attributes["outcome"] != "server_error" and attributes["outcome"] != "other"'
        - 'attributes["http.status_code"] < 100 or attributes["http.status_code"] > 599'
        - 'attributes["event.schema.version"] == "2" and attributes["model"] == nil'
        - 'attributes["event.schema.version"] == "2" and attributes["machine_profile"] == nil'
        - 'attributes["event.schema.version"] == "2" and attributes["duration_ms"] == nil'
        - 'attributes["event.schema.version"] == "2" and attributes["admission_outcome"] != "admitted" and attributes["admission_outcome"] != "unauthenticated" and attributes["admission_outcome"] != "forbidden" and attributes["admission_outcome"] != "auth_misconfigured" and attributes["admission_outcome"] != "region_mismatch" and attributes["admission_outcome"] != "license_excluded" and attributes["admission_outcome"] != "payload_too_large" and attributes["admission_outcome"] != "invalid_request" and attributes["admission_outcome"] != "insufficient_credits" and attributes["admission_outcome"] != "key_spend_limit_exceeded" and attributes["admission_outcome"] != "rate_limited"'
  transform/contract_logs:
    error_mode: propagate
    log_statements:
      - context: resource
        statements:
          - set(cache["service.name"], attributes["service.name"])
          - set(cache["service.instance.id"], attributes["service.instance.id"])
          - set(cache["deployment.environment"], attributes["deployment.environment"])
          - set(cache["cloud.region"], attributes["cloud.region"])
          - set(cache["service.version"], attributes["service.version"])
          - keep_keys(attributes, [])
          - set(attributes["service.name"], cache["service.name"]) where cache["service.name"] != nil
          - set(attributes["service.instance.id"], cache["service.instance.id"]) where cache["service.instance.id"] != nil
          - set(attributes["deployment.environment"], cache["deployment.environment"]) where cache["deployment.environment"] != nil
          - set(attributes["cloud.region"], cache["cloud.region"]) where cache["cloud.region"] != nil
          - set(attributes["service.version"], cache["service.version"]) where cache["service.version"] != nil
          - set(schema_url, "")
      - context: scope
        statements:
          - keep_keys(attributes, [])
          - set(name, "sie-gateway.request-completion")
          - set(version, "")
          - set(schema_url, "")
      - context: log
        statements:
          - set(cache["event.name"], attributes["event.name"])
          - set(cache["event.schema.version"], attributes["event.schema.version"])
          - set(cache["operation"], attributes["operation"])
          - set(cache["outcome"], attributes["outcome"])
          - set(cache["http.status_code"], attributes["http.status_code"])
          - set(cache["model"], attributes["model"])
          - set(cache["machine_profile"], attributes["machine_profile"])
          - set(cache["duration_ms"], attributes["duration_ms"])
          - set(cache["admission_outcome"], attributes["admission_outcome"])
          - keep_keys(attributes, [])
          - set(attributes["event.name"], cache["event.name"]) where cache["event.name"] != nil
          - set(attributes["event.schema.version"], cache["event.schema.version"]) where cache["event.schema.version"] != nil
          - set(attributes["operation"], cache["operation"]) where cache["operation"] != nil
          - set(attributes["outcome"], cache["outcome"]) where cache["outcome"] != nil
          - set(attributes["http.status_code"], cache["http.status_code"]) where cache["http.status_code"] != nil
          - set(attributes["model"], cache["model"]) where cache["model"] != nil
          - set(attributes["machine_profile"], cache["machine_profile"]) where cache["machine_profile"] != nil
          - set(attributes["duration_ms"], cache["duration_ms"]) where cache["duration_ms"] != nil
          - set(attributes["admission_outcome"], cache["admission_outcome"]) where cache["admission_outcome"] != nil
          # Canonical release domains are rechecked at the collector boundary
          # so malformed producer values cannot become vendor dimensions.
          - 'set(attributes["model"], "other") where attributes["event.schema.version"] == "2" and not IsMatch(attributes["model"], "^(BAAI/bge-m3|IDEA-Research/grounding-dino-base|Qwen/Qwen3-Embedding-4B|Qwen/Qwen3-Reranker-0[.]6B|Qwen/Qwen3-Reranker-4B|Qwen/Qwen3-VL-Reranker-2B|Qwen/Qwen3[.]5-4B|Qwen/Qwen3[.]6-27B|Snowflake/snowflake-arctic-embed-l-v2[.]0|docling|fastino/gliguard-LLMGuardrails-300M|fastino/gliner2-base-v1|fastino/gliner2-large-v1|google/owlv2-base-patch16-ensemble|google/siglip-so400m-patch14-384|google/siglip2-base-patch16-224|ibm-granite/granite-guardian-3[.]0-2b|knowledgator/gliclass-large-v3[.]0|lightonai/GTE-ModernColBERT-v1|lightonai/LightOnOCR-2-1B|numind/NuNER_Zero|openai/whisper-large-v3-turbo|other|prithivida/Splade_PP_en_v2|tencent/R3-embedding-0[.]6b|tencent/R3-rerank-0[.]6b|urchade/gliner_multi-v2[.]1|urchade/gliner_multi_pii-v1)$")'
          - 'set(attributes["machine_profile"], "other") where attributes["event.schema.version"] == "2" and not IsMatch(attributes["machine_profile"], "^(a10|a100-40gb|a100-80gb|cpu|h100|l4|l4-spot|other|sglang-cu130|t4)$")'
          - set(attributes["event.name"], "inference.request.completed")
          - set(body, "inference.request.completed")
          - set(severity_text, "INFO")
          - set(severity_number, SEVERITY_NUMBER_INFO)
  # Fixed structured lifecycle records only; never a stdout/logging bridge.
  filter/lifecycle_logs:
    error_mode: propagate
    logs:
      log_record:
        - 'attributes["event.name"] != "inference.lifecycle.completed"'
        - 'body != "inference.lifecycle.completed"'
        - 'attributes["event.schema.version"] != "1"'
        - 'attributes["phase"] != "request" and attributes["phase"] != "response_body" and attributes["phase"] != "generation_stream" and attributes["phase"] != "worker_generation" and attributes["phase"] != "dispatch_attempt"'
        - 'attributes["operation"] != "encode" and attributes["operation"] != "score" and attributes["operation"] != "extract" and attributes["operation"] != "generate" and attributes["operation"] != "embeddings" and attributes["operation"] != "moderations" and attributes["operation"] != "other"'
        - 'attributes["outcome"] != "success" and attributes["outcome"] != "rejected" and attributes["outcome"] != "error" and attributes["outcome"] != "cancelled"'
        - 'attributes["error_class"] != "none" and attributes["error_class"] != "client_error" and attributes["error_class"] != "server_error" and attributes["error_class"] != "transport" and attributes["error_class"] != "timeout" and attributes["error_class"] != "worker" and attributes["error_class"] != "cancelled" and attributes["error_class"] != "protocol" and attributes["error_class"] != "other"'
        - 'not IsDouble(attributes["duration_ms"]) and not IsInt(attributes["duration_ms"])'
        # Collector 0.119 has no IsFinite: x - x is zero only for finite numeric values.
        - 'attributes["duration_ms"] != attributes["duration_ms"] or attributes["duration_ms"] < 0 or attributes["duration_ms"] - attributes["duration_ms"] != 0'
        - 'attributes["first_token_ms"] != nil and not IsDouble(attributes["first_token_ms"]) and not IsInt(attributes["first_token_ms"])'
        - 'attributes["first_token_ms"] != nil and (attributes["first_token_ms"] != attributes["first_token_ms"] or attributes["first_token_ms"] < 0 or attributes["first_token_ms"] > attributes["duration_ms"])'
  filter/gateway_lifecycle_logs:
    error_mode: propagate
    logs:
      log_record:
        - 'attributes["phase"] != "request" and attributes["phase"] != "response_body" and attributes["phase"] != "generation_stream"'
  filter/application_lifecycle_logs:
    error_mode: propagate
    logs:
      log_record:
        - 'not (resource.attributes["service.name"] == "sie-worker" and attributes["phase"] == "worker_generation") and not (resource.attributes["service.name"] == "sie-dispatcher" and attributes["phase"] == "dispatch_attempt")'
  transform/lifecycle_logs:
    error_mode: propagate
    log_statements:
      - context: resource
        statements:
          - set(cache["service.name"], attributes["service.name"])
          - set(cache["service.instance.id"], attributes["service.instance.id"])
          - set(cache["deployment.environment"], attributes["deployment.environment"])
          - set(cache["cloud.region"], attributes["cloud.region"])
          - set(cache["service.version"], attributes["service.version"])
          - keep_keys(attributes, [])
          - set(attributes["service.name"], cache["service.name"]) where cache["service.name"] != nil
          - set(attributes["service.instance.id"], cache["service.instance.id"]) where cache["service.instance.id"] != nil
          - set(attributes["deployment.environment"], cache["deployment.environment"]) where cache["deployment.environment"] != nil
          - set(attributes["cloud.region"], cache["cloud.region"]) where cache["cloud.region"] != nil
          - set(attributes["service.version"], cache["service.version"]) where cache["service.version"] != nil
          - set(schema_url, "")
      - context: scope
        statements:
          - keep_keys(attributes, [])
          - set(name, "")
          - set(version, "")
          - set(schema_url, "")
      - context: log
        statements:
          - set(cache["event.name"], attributes["event.name"])
          - set(cache["event.schema.version"], attributes["event.schema.version"])
          - set(cache["phase"], attributes["phase"])
          - set(cache["operation"], attributes["operation"])
          - set(cache["outcome"], attributes["outcome"])
          - set(cache["error_class"], attributes["error_class"])
          - set(cache["duration_ms"], attributes["duration_ms"])
          - set(cache["first_token_ms"], attributes["first_token_ms"])
          - keep_keys(attributes, [])
          - set(attributes["event.name"], cache["event.name"]) where cache["event.name"] != nil
          - set(attributes["event.schema.version"], cache["event.schema.version"]) where cache["event.schema.version"] != nil
          - set(attributes["phase"], cache["phase"]) where cache["phase"] != nil
          - set(attributes["operation"], cache["operation"]) where cache["operation"] != nil
          - set(attributes["outcome"], cache["outcome"]) where cache["outcome"] != nil
          - set(attributes["error_class"], cache["error_class"]) where cache["error_class"] != nil
          - set(attributes["duration_ms"], cache["duration_ms"]) where cache["duration_ms"] != nil
          - set(attributes["first_token_ms"], cache["first_token_ms"]) where cache["first_token_ms"] != nil
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
      processors: [memory_limiter, filter/remote_gateway_contract, resource/gateway_identity, transform/contract_metrics, transform/remote_metric_scalars, transform/remote_queue_identity, batch]
      exporters: [otlphttp/betterstack]
    metrics/betterstack/application:
      receivers: [otlp/application]
      processors: [memory_limiter, filter/remote_application_contract, resource/application_identity, transform/contract_metrics, transform/remote_metric_scalars, batch]
      exporters: [otlphttp/betterstack]
    {{- end }}
  {{- end }}
  {{- if $logsEnabled }}
    logs:
      receivers: [otlp/gateway]
      processors: [memory_limiter, resource/gateway_identity, filter/contract_logs, transform/contract_logs, batch]
      exporters: {{ toJson $logExporters }}
    logs/lifecycle/gateway:
      receivers: [otlp/gateway]
      processors: [memory_limiter, resource/gateway_identity, filter/gateway_lifecycle_logs, filter/lifecycle_logs, transform/lifecycle_logs, batch]
      exporters: {{ toJson $logExporters }}
    logs/lifecycle/application:
      receivers: [otlp/application]
      processors: [memory_limiter, filter/application_lifecycle_logs, resource/application_identity, filter/lifecycle_logs, transform/lifecycle_logs, batch]
      exporters: {{ toJson $logExporters }}
  {{- end }}
{{- end -}}
