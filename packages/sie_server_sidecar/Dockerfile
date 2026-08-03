# syntax=docker/dockerfile:1
#
# SIE Server Sidecar Image
#
# Build from repo root:
#   docker build -f packages/sie_server_sidecar/Dockerfile \
#     -t sie-server-sidecar:latest .
#
# The published image is ghcr.io/superlinked/sie-server-sidecar; the runtime
# binary inside the image is /sie-server-sidecar.
#
# The build runs at the Cargo workspace root (#2339): the root
# Cargo.toml/Cargo.lock plus every member's manifest+sources must be present
# for cargo to load the workspace, and the binary is selected with
# `-p sie-server-sidecar`.
#
# The only Cargo feature today is `cloud-storage` (S3/GCS payload modes),
# which is the default for `ARG FEATURES` below. CI can override with
# `--build-arg FEATURES=` if you want a leaner build with no remote
# payload support.
#
# Run:
#   The worker expects a Unix socket to a Python `sie_server.ipc_server`.
#   The Helm chart wires the sidecar at /var/run/sie/ipc.sock; for local
#   testing mount a socket path and set SIE_IPC_SOCKET_PATH accordingly.

ARG FEATURES=cloud-storage
ARG RUST_VERSION=1.97.0

# =============================================================================
# Dependency planner: capture the Cargo graph only.
# =============================================================================
FROM rust:${RUST_VERSION}-slim-bookworm AS chef
RUN cargo install cargo-chef --locked --version 0.1.77
WORKDIR /build

FROM chef AS planner
COPY Cargo.toml Cargo.lock ./
COPY packages/sie_audio_prep/ packages/sie_audio_prep/
COPY packages/sie_gateway/ packages/sie_gateway/
COPY packages/sie_server_sidecar/ packages/sie_server_sidecar/
COPY packages/sie_telemetry/ packages/sie_telemetry/
RUN cargo chef prepare --recipe-path recipe.json

# =============================================================================
# Release builder: cook deps, then build the real binary.
# =============================================================================
FROM chef AS builder
ARG FEATURES

# Build deps.
#   * `pkg-config + libssl-dev` - in case any transitive dep uses OpenSSL
#     (async-nats itself uses rustls, but staying safe for feature toggles).
#   * `ca-certificates` - cargo needs these when the network cache misses.
#   * `build-essential + cmake` - native builds for bundled Opus, `onig_sys`
#     (tokenizers), and `safetensors`. These crates STATICALLY link C sources
#     so the distroless runtime image can stay minimal.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        ca-certificates \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# Force `onig_sys` to statically link the bundled Oniguruma instead of
# trying to dlopen a system libonig at runtime - distroless does not
# ship libonig and tokenizers would otherwise panic on first use.
ENV RUSTONIG_STATIC_LIBONIG=1

COPY --from=planner /build/recipe.json recipe.json

# Warm cargo's target cache with JUST the dependency graph. This keeps
# iteration fast when only worker source changes (deps hash unchanged).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo chef cook --release \
        --package sie-server-sidecar \
        --features "${FEATURES}" \
        --recipe-path recipe.json

# Restore real source.
#
# `cargo chef cook` mutates the workspace into dependency-only stubs. Re-copy
# the real manifests and sources before `cargo build` so the final image gets
# `sie-server-sidecar`, not a cache-warming placeholder binary.
COPY Cargo.toml Cargo.lock ./
COPY packages/sie_audio_prep/ packages/sie_audio_prep/
COPY packages/sie_gateway/ packages/sie_gateway/
COPY packages/sie_server_sidecar/ packages/sie_server_sidecar/
COPY packages/sie_telemetry/ packages/sie_telemetry/

# Real build.
#
# The `/build/target` cache carries compiled deps forward. The size guard
# catches regressions where the restore step is reordered and a stub binary
# reaches the runtime image.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --locked --release --package sie-server-sidecar --bin sie-server-sidecar --features "${FEATURES}" \
    && ls -lh target/release/sie-server-sidecar \
    && test "$(wc -c < target/release/sie-server-sidecar)" -gt 5000000 \
    && cp target/release/sie-server-sidecar /usr/local/bin/sie-server-sidecar

# =============================================================================
# Runtime image: minimal image, just the static-ish binary.
# =============================================================================
# `distroless/cc-debian12` ships glibc + common C runtime libs, matching what
# the Rust build links against. `nonroot` gives us uid/gid 65532.
FROM gcr.io/distroless/cc-debian12:nonroot AS runtime

COPY --from=builder /usr/local/bin/sie-server-sidecar /sie-server-sidecar

# Opinionated defaults; override via env in the Pod spec / Helm chart.
#
# * SIE_WORKER_PROBE_PORT   - liveness/readiness probe port (default 9095).
# * SIE_IPC_SOCKET_PATH     - Unix socket to the Python sie-server process. The Helm
#                             chart mounts a shared emptyDir at /var/run/sie.
# * RUST_LOG                - structured log level; OTel set at deploy time.
# * SIE_NATS_URL            - required, no default; must be set by the
#                             Pod / Helm chart.
ENV SIE_WORKER_PROBE_PORT=9095 \
    SIE_IPC_SOCKET_PATH=/var/run/sie/ipc.sock \
    RUST_LOG=info

EXPOSE 9095

USER nonroot:nonroot
ENTRYPOINT ["/sie-server-sidecar"]

LABEL org.opencontainers.image.title="sie-server-sidecar" \
      org.opencontainers.image.description="SIE server sidecar - NATS JetStream consumer + Python IPC forwarder." \
      org.opencontainers.image.source="https://github.com/superlinked/sie-internal"
