# syntax=docker/dockerfile:1
#
# SIE Rust Server Image
#
# Build from repo root:
#   docker build -f packages/sie_server_rust/Dockerfile -t sie-server-rust:latest .
#
# This image runs the Rust adapter process and exposes the existing UDS IPC
# contract to the worker-sidecar. Native engines such as Candle run inside this
# process rather than through a sibling Python adapter process.

ARG RUST_VERSION=1.97.0
ARG CARGO_FEATURES=""

FROM rust:${RUST_VERSION}-slim-bookworm AS chef
RUN cargo install cargo-chef --locked --version 0.1.77
WORKDIR /build

FROM chef AS planner
COPY packages/sie_server_rust/Cargo.toml packages/sie_server_rust/Cargo.lock ./
COPY packages/sie_server_rust/vendor/ vendor/
COPY packages/sie_telemetry/ /sie_telemetry/
COPY packages/sie_server_rust/src/ src/
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
ARG CARGO_FEATURES=""
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=planner /build/recipe.json recipe.json
COPY packages/sie_server_rust/vendor/ vendor/
COPY packages/sie_telemetry/ /sie_telemetry/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    if [ -n "$CARGO_FEATURES" ]; then \
        cargo chef cook --release --features "$CARGO_FEATURES" --recipe-path recipe.json; \
    else \
        cargo chef cook --release --recipe-path recipe.json; \
    fi

COPY packages/sie_server_rust/Cargo.toml packages/sie_server_rust/Cargo.lock ./
COPY packages/sie_server_rust/vendor/ vendor/
COPY packages/sie_telemetry/ /sie_telemetry/
COPY packages/sie_server_rust/src/ src/

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    if [ -n "$CARGO_FEATURES" ]; then \
        cargo build --locked --release --features "$CARGO_FEATURES" --bin sie-server-rust; \
    else \
        cargo build --locked --release --bin sie-server-rust; \
    fi \
    && ls -lh target/release/sie-server-rust \
    && test "$(wc -c < target/release/sie-server-rust)" -gt 1000000 \
    && cp target/release/sie-server-rust /usr/local/bin/sie-server-rust

FROM gcr.io/distroless/cc-debian12:nonroot AS runtime

COPY --from=builder /usr/local/bin/sie-server-rust /sie-server-rust

ENV SIE_IPC_SOCKET_PATH=/var/run/sie/ipc.sock \
    SIE_BUNDLE=candle \
    SIE_SERVER_HOST=0.0.0.0 \
    SIE_SERVER_PORT=8080 \
    RUST_LOG=info

EXPOSE 8080

USER nonroot:nonroot
ENTRYPOINT ["/sie-server-rust"]

LABEL org.opencontainers.image.title="sie-server-rust" \
      org.opencontainers.image.description="SIE Rust worker process backed by native inference engines." \
      org.opencontainers.image.source="https://github.com/superlinked/sie"
