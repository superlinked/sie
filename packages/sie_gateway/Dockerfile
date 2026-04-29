FROM rust:1.95.0-slim-bookworm AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

FROM chef AS planner
COPY packages/sie_gateway/Cargo.toml packages/sie_gateway/Cargo.lock ./
COPY packages/sie_gateway/src/ src/
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --features cloud-storage --recipe-path recipe.json
COPY packages/sie_gateway/Cargo.toml packages/sie_gateway/Cargo.lock ./
COPY packages/sie_gateway/src/ src/
RUN cargo build --release --features cloud-storage

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=builder /app/target/release/sie-gateway /sie-gateway
# No filesystem seed: bundles and models are fetched from sie-config at
# startup (see packages/sie_gateway/src/state/config_bootstrap.rs). Leaving
# SIE_BUNDLES_DIR / SIE_MODELS_DIR unset means ModelRegistry falls back to
# the relative defaults ("bundles", "models"), neither of which exists in
# this distroless image — that's fine, reload() warns and proceeds with an
# empty seed and the bootstrap fetch fills both sets before serving.
EXPOSE 8080
USER nonroot
ENTRYPOINT ["/sie-gateway", "serve"]
CMD ["--port", "8080", "--kubernetes"]
