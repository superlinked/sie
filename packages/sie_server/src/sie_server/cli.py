from __future__ import annotations

# IMPORTANT: HF timeout defaults must be installed BEFORE ``huggingface_hub``
# is imported anywhere in the process (the library reads HF_HUB_*_TIMEOUT
# at module import time). Inlined here — rather than imported from
# ``sie_server.core.hf_env`` — because importing ``sie_server.core.*``
# triggers ``core/__init__.py`` which transitively pulls in adapters and
# could one day import ``huggingface_hub`` itself, latching the stock 10 s
# defaults before our overrides land. Keep these two ``setdefault`` lines
# as the very first executable statements after ``__future__``.
import os
import sys

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

# hf-xet (HuggingFace's Xet download protocol) can hang indefinitely on Apple
# Silicon — it does NOT honour HF_HUB_DOWNLOAD_TIMEOUT, so a stuck ``xet_get``
# wedges model loads with no recovery (verified: bge-m3 load blocked forever in
# ``huggingface_hub.file_download.xet_get`` on macOS). Disable Xet on macOS so
# downloads use reliable HTTPS. CUDA/Linux keep Xet (faster, works there).
# Override by exporting HF_HUB_DISABLE_XET=0.
if sys.platform == "darwin":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import hashlib
import logging
from pathlib import Path
from typing import Annotated

import typer
from sie_sdk.bundle_utils import (
    ModelAdapterInventory,
    find_bundle_for_model_adapters,
    find_bundle_for_models,
    match_bundle_model_adapters,
    match_bundle_models,
)

import sie_server
from sie_server.api.generate import router as generate_router
from sie_server.api.openai_audio import router as openai_audio_router
from sie_server.app.app_state_config import ENV_DEVICES, ENV_POOL, AppStateConfig
from sie_server.config.model import ModelConfig
from sie_server.core.deps import collect_bundle_deps
from sie_server.core.loader import load_model_configs
from sie_server.main import run_server

logger = logging.getLogger(__name__)


# Prefer the package-internal location so a fresh `pip install sie-server`
# followed by `sie-server serve` works out of the box (issue #820).
def _resolve_default_dir(name: str) -> Path:
    pkg_dir = Path(sie_server.__file__).parent
    bundled = pkg_dir / name
    if bundled.is_dir():
        return bundled
    # Fallback: source/dev layout (packages/sie_server/<name>)
    return pkg_dir.parent.parent / name


_DEFAULT_BUNDLES_DIR = _resolve_default_dir("bundles")
DEFAULT_MODELS_DIR = str(_resolve_default_dir("models"))


def _model_adapter_inventory(configs: dict[str, ModelConfig]) -> ModelAdapterInventory:
    """Build bundle-matching inputs from local or cloud-loaded model configs."""
    inventory: ModelAdapterInventory = {}
    for name, config in configs.items():
        adapter_path = config.resolve_profile("default").adapter_path
        adapter_module = adapter_path.split(":", maxsplit=1)[0]
        inventory[name] = ({adapter_module}, config.pool)
    return inventory


def detect_device() -> str:
    """Auto-detect the best available device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str) -> str:
    """Resolve 'auto' to actual device, or return as-is."""
    if device == "auto":
        return detect_device()
    return device


def parse_devices(value: str | None) -> list[str] | None:
    """Parse comma-separated concrete devices from an env var."""
    if value is None:
        return None
    devices = [item.strip() for item in value.split(",") if item.strip()]
    return devices or None


app = typer.Typer(
    name="sie-server",
    help="Search Inference Engine - GPU inference server for search workloads",
    no_args_is_help=True,
    add_completion=False,
)


# Use callback to prevent serve from being the default command
@app.callback()
def callback() -> None:
    """Search Inference Engine - GPU inference server for search workloads."""


def load_bundle(
    bundle_name: str,
    bundles_dir: Path,
    models_dir: str | None = None,
    *,
    pool_name: str | None = None,
) -> list[str]:
    """Load model names from a bundle file using adapter-based matching.

    Args:
        bundle_name: Name of the bundle (without .yaml extension).
        bundles_dir: Path to the bundles directory.
        models_dir: Path to the models directory for adapter matching.
            Defaults to DEFAULT_MODELS_DIR.

    Returns:
        List of model names whose adapters match the bundle's adapter list.

    Raises:
        FileNotFoundError: If bundle file doesn't exist.
    """
    bundle_path = bundles_dir / f"{bundle_name}.yaml"
    if not bundle_path.exists():
        msg = f"Bundle file not found: {bundle_path}"
        raise FileNotFoundError(msg)

    resolved_models_dir = Path(models_dir) if models_dir else Path(DEFAULT_MODELS_DIR)
    return match_bundle_models(bundle_path, resolved_models_dir, pool_name=pool_name)


@app.command("resolve-deps")
def resolve_deps(
    bundle: str | None = typer.Option(None, "--bundle", "-b", help="Bundle name to resolve deps for"),
    models: str | None = typer.Option(None, "--models", "-m", help="Comma-separated model names"),
    models_dir: str = typer.Option(DEFAULT_MODELS_DIR, "--models-dir", help="Models directory"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
    cpu: bool = typer.Option(False, "--cpu", help="Exclude CUDA-only dependencies (flash-attn)"),
) -> None:
    """Resolve and print dependencies for a bundle or model list.

    Outputs requirements to stdout (one per line).
    Used by serve.bash to sync deps before starting the server.

    Use --cpu flag when building CPU-only images to exclude flash-attn
    and other CUDA-only dependencies.
    """
    from sie_server.core.deps import collect_bundle_deps

    models_path = Path(models_dir).resolve()
    bundles_dir = _DEFAULT_BUNDLES_DIR
    pool_name = os.environ.get(ENV_POOL) or None

    if bundle and models:
        typer.echo("Error: Cannot specify both --bundle and --models", err=True)
        raise typer.Exit(1)

    if not bundle and not models:
        typer.echo("Error: Either --bundle or --models must be specified", err=True)
        raise typer.Exit(1)

    # Resolve bundle name: explicit --bundle, or find best bundle for --models
    if bundle:
        resolved_bundle = bundle
    else:
        model_list = [m.strip() for m in models.split(",") if m.strip()]  # type: ignore
        matched = find_bundle_for_models(model_list, bundles_dir, models_path, pool_name=pool_name)
        if not matched:
            typer.echo(f"Error: No bundle found covering models: {', '.join(model_list)}", err=True)
            raise typer.Exit(1)
        resolved_bundle = matched

    result = collect_bundle_deps(resolved_bundle, bundles_dir, models_path, exclude_cuda=cpu, pool_name=pool_name)

    if result.conflicts:
        if output_json:
            import json

            typer.echo(json.dumps({"error": "conflicts", "conflicts": result.conflicts}), err=True)
        else:
            typer.echo("Dependency conflicts detected:", err=True)
            for conflict in result.conflicts:
                typer.echo(f"  - {conflict}", err=True)
        raise typer.Exit(1)

    if output_json:
        import json

        typer.echo(json.dumps(result.to_dict()))
    else:
        # Print requirements to stdout for shell scripts to consume
        for req in result.requirements:
            typer.echo(req)


@app.command("openapi")
def openapi_export(
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file path (default: stdout)"),  # noqa: B008
    indent: int = typer.Option(2, "--indent", help="JSON indentation level"),
) -> None:
    """Export the OpenAPI spec as a static JSON file."""
    import json
    from importlib.metadata import version as pkg_version

    from fastapi import FastAPI

    from sie_server.api.encode import router as encode_router
    from sie_server.api.extract import router as extract_router
    from sie_server.api.health import router as health_router
    from sie_server.api.models import router as models_router
    from sie_server.api.openai_compat import router as openai_router
    from sie_server.api.openapi import setup_custom_openapi_schema
    from sie_server.api.root import router as root_router
    from sie_server.api.score import router as score_router
    from sie_server.api.ws import router as ws_router

    # Build a lightweight FastAPI app for the PUBLISHED API surface — no lifespan (no GPU
    # init, model registry, telemetry, or NATS). NOTE: the local-only convenience routes are
    # intentionally excluded from the committed openapi.json — the ``openai_local`` router
    # (/v1/chat/completions + /v1/rerank) is a single-node/dev convenience surface. The
    # SIE-native ``generate`` route is part of the worker contract and is published here even
    # though the Rust gateway remains the production cluster API authority.
    app_ = FastAPI(
        title="SIE Server",
        description="Search Inference Engine - GPU inference server for search workloads",
        version="0.1.0",
    )
    app_.include_router(root_router)
    app_.include_router(health_router)
    app_.include_router(encode_router)
    app_.include_router(extract_router)
    app_.include_router(generate_router)
    app_.include_router(score_router)
    app_.include_router(models_router)
    app_.include_router(ws_router)
    app_.include_router(openai_audio_router)
    app_.include_router(openai_router)
    setup_custom_openapi_schema(app_)

    spec = app_.openapi()
    spec["info"]["version"] = pkg_version("sie-server")

    json_str = json.dumps(spec, indent=indent) + "\n"
    if output:
        output.write_text(json_str)
        typer.echo(f"OpenAPI spec written to {output}")
    else:
        typer.echo(json_str, nl=False)


@app.command()
def serve(
    port: int = typer.Option(8080, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),  # noqa: S104 — intentional bind to all interfaces for server
    device: str = typer.Option("auto", "--device", "-d", help="Device to use (auto, cuda, mps, cpu)"),
    models_dir: str = typer.Option(
        DEFAULT_MODELS_DIR,
        "--models-dir",
        help="Models directory (local path, s3://, gs://, abfs://, or abfss://)",
    ),
    bundle: str | None = typer.Option(None, "--bundle", "-b", help="Bundle name to load (from bundles/ dir)"),
    models: str | None = typer.Option(None, "--models", "-m", help="Comma-separated model names to load"),
    local_cache: str | None = typer.Option(None, "--local-cache", help="Local cache directory (default: HF_HOME)"),
    cluster_cache: str | None = typer.Option(
        None,
        "--cluster-cache",
        help="Cluster cache URL (s3://, gs://, abfs://, or abfss://)",
    ),
    hf_fallback: bool = typer.Option(True, "--hf-fallback/--no-hf-fallback", help="Enable HuggingFace Hub fallback"),
    reload: bool = typer.Option(default=False, help="Enable auto-reload for development"),
    tracing: bool = typer.Option(default=False, help="Enable OpenTelemetry tracing (exports to localhost:4317)"),
    instrumentation: bool = typer.Option(False, "--instrumentation", "-i", help="Enable batch instrumentation logging"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            "-l",
            envvar="SIE_LOG_LEVEL",
            help="Log level (DEBUG, INFO, WARNING, ERROR). Helm sets SIE_LOG_LEVEL for workers.",
        ),
    ] = "info",
    preload: str | None = typer.Option(None, "--preload", help="Comma-separated model names to preload at startup"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Enable structured JSON logging (for Loki)"),
) -> None:
    """Start the SIE inference server."""
    from sie_sdk.storage import is_cloud_path

    from sie_server.core.logging import configure_logging

    # Configure logging (supports JSON format for Loki compatibility)
    configure_logging(verbose=verbose, json_format=json_logs or None, level_name=log_level)

    # Handle models directory - cloud URLs pass through, local paths resolve
    if is_cloud_path(models_dir):
        models_dir_resolved = models_dir
    else:
        models_path = Path(models_dir).resolve()
        if not models_path.exists():
            typer.echo(f"Warning: Models directory '{models_path}' does not exist", err=True)
        models_dir_resolved = str(models_path)

    # Resolve device (auto-detect if needed)
    resolved_device = resolve_device(device)
    resolved_devices = parse_devices(os.environ.get(ENV_DEVICES))
    pool_name = os.environ.get(ENV_POOL) or None

    # Handle bundle/models filter
    model_filter: list[str] | None = None
    if bundle and models:
        typer.echo("Error: Cannot specify both --bundle and --models", err=True)
        raise typer.Exit(1)

    runtime_bundle = os.environ.get("SIE_BUNDLE")
    baked_bundle = (
        runtime_bundle if runtime_bundle and (_DEFAULT_BUNDLES_DIR / f"{runtime_bundle}.yaml").exists() else None
    )
    if bundle and baked_bundle and bundle != baked_bundle:
        typer.echo(
            f"Error: Bundle '{bundle}' is incompatible with baked bundle '{baked_bundle}'; "
            f"use an image built for bundle '{bundle}'",
            err=True,
        )
        raise typer.Exit(1)

    # Release images declare their baked dependency bundle through SIE_BUNDLE,
    # while the local ``mise run serve`` path resolves the default bundle when
    # neither selector is provided. Honour both here so the advertised catalog
    # never includes models whose adapter dependencies are absent.
    if not bundle and not models:
        bundle = baked_bundle or "default"

    all_configs: dict[str, ModelConfig] | None = None
    model_adapters: ModelAdapterInventory | None = None
    if is_cloud_path(models_dir_resolved):
        all_configs = load_model_configs(models_dir_resolved)
        model_adapters = _model_adapter_inventory(all_configs)

    if bundle:
        bundles_dir = _DEFAULT_BUNDLES_DIR
        if not bundles_dir.exists():
            typer.echo(f"Error: Bundles directory not found: {bundles_dir}", err=True)
            raise typer.Exit(1)
        try:
            if model_adapters is not None:
                model_filter = match_bundle_model_adapters(
                    bundles_dir / f"{bundle}.yaml",
                    model_adapters,
                    pool_name=pool_name,
                )
            else:
                model_filter = load_bundle(bundle, bundles_dir, models_dir_resolved, pool_name=pool_name)
            typer.echo(f"Bundle '{bundle}': {len(model_filter)} models")
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from e

    if models:
        model_filter = [m.strip() for m in models.split(",") if m.strip()]
        typer.echo(f"Model filter: {len(model_filter)} models")
        # Validate model names exist by loading configs and checking
        if all_configs is None:
            all_configs = load_model_configs(models_dir_resolved)
        unknown = [m for m in model_filter if m not in all_configs]
        if unknown:
            typer.echo(f"Error: Unknown model(s): {', '.join(unknown)}", err=True)
            typer.echo(f"Available models: {', '.join(sorted(all_configs.keys())[:10])}...", err=True)
            raise typer.Exit(1)

    # Augment the filter with explicitly-allowed extra model ids (opt-in via
    # ``SIE_EXTRA_MODELS``). A bundle's filter is built by scanning model YAML
    # files, so it only contains *base* ids — it can't see profile-variant ids
    # like ``Qwen/Qwen3.6-27B:rtx-pro-6000`` that the loader expands at runtime.
    # A worker that serves a base model's adapter can also serve that model's
    # profile variants (same adapter, same weights, different launch flags), so
    # allow naming them here. Validated against the variant-expanded config set.
    # The advertised bundle (``SIE_BUNDLE`` below) is unchanged, so gateway
    # routing by adapter→bundle still resolves these variants to this worker.
    extra_models_env = os.environ.get("SIE_EXTRA_MODELS")
    if extra_models_env and model_filter is not None:
        extras = [m.strip() for m in extra_models_env.split(",") if m.strip()]
        if all_configs is None:
            all_configs = load_model_configs(models_dir_resolved)
        unknown_extra = [m for m in extras if m not in all_configs]
        if unknown_extra:
            typer.echo(f"Error: SIE_EXTRA_MODELS unknown model(s): {', '.join(unknown_extra)}", err=True)
            raise typer.Exit(1)
        model_filter = sorted(set(model_filter) | set(extras))
        typer.echo(f"Extra models added to filter: {', '.join(extras)}")

    # Explicit selectors must not escape a release image's dependency boundary.
    # Local ``mise run serve -m ...`` has no initial known bundle and keeps the
    # ad-hoc dependency-resolution path in tools/mise_tasks/serve.bash.
    if baked_bundle and model_filter is not None:
        if model_adapters is not None:
            baked_models = set(
                match_bundle_model_adapters(
                    _DEFAULT_BUNDLES_DIR / f"{baked_bundle}.yaml",
                    model_adapters,
                    pool_name=pool_name,
                )
            )
        else:
            baked_models = set(
                load_bundle(baked_bundle, _DEFAULT_BUNDLES_DIR, models_dir_resolved, pool_name=pool_name)
            )
        incompatible: list[str] = []
        for selected_model in sorted(set(model_filter) - baked_models):
            if model_adapters is not None:
                selected_bundle = find_bundle_for_model_adapters(
                    [selected_model],
                    _DEFAULT_BUNDLES_DIR,
                    model_adapters,
                    pool_name=pool_name,
                )
            else:
                selected_bundle = find_bundle_for_models(
                    [selected_model],
                    _DEFAULT_BUNDLES_DIR,
                    Path(models_dir_resolved),
                    pool_name=pool_name,
                )
            # A dependency-free bundle adds only adapter/config selection; its
            # code already ships in the base server image. This keeps the
            # weightless Fake Engine usable in default-image Docker tests
            # without allowing a dependency-bearing bundle such as
            # transformers5 to escape the image's baked environment.
            if selected_bundle:
                selected_deps = collect_bundle_deps(
                    selected_bundle,
                    _DEFAULT_BUNDLES_DIR,
                    Path(DEFAULT_MODELS_DIR) if model_adapters is not None else Path(models_dir_resolved),
                    pool_name=pool_name,
                )
                if not selected_deps.conflicts and not selected_deps.requirements:
                    continue
            incompatible.append(selected_model)
        if incompatible:
            if model_adapters is not None:
                recommended = find_bundle_for_model_adapters(
                    model_filter,
                    _DEFAULT_BUNDLES_DIR,
                    model_adapters,
                    pool_name=pool_name,
                )
            else:
                recommended = find_bundle_for_models(
                    model_filter,
                    _DEFAULT_BUNDLES_DIR,
                    Path(models_dir_resolved),
                    pool_name=pool_name,
                )
            recommendation = (
                f"; use an image built for bundle '{recommended}'"
                if recommended
                else "; use an image whose bundle contains all selected models"
            )
            typer.echo(
                f"Error: Model(s) {', '.join(incompatible)} are incompatible with baked bundle "
                f"'{baked_bundle}'{recommendation}",
                err=True,
            )
            raise typer.Exit(1)

    os.environ["SIE_INSTRUMENTATION"] = "true" if instrumentation else "false"
    if baked_bundle:
        os.environ["SIE_BUNDLE"] = baked_bundle
    elif bundle:
        os.environ["SIE_BUNDLE"] = bundle
    elif model_filter:
        digest = hashlib.sha256(",".join(sorted(model_filter)).encode("utf-8")).hexdigest()[:12]
        os.environ["SIE_BUNDLE"] = f"adhoc-{digest}"
    else:
        os.environ["SIE_BUNDLE"] = "default"
    # Pass cache config via environment variables
    if local_cache:
        os.environ["SIE_LOCAL_CACHE"] = str(Path(local_cache).resolve())
    elif "SIE_LOCAL_CACHE" in os.environ:
        del os.environ["SIE_LOCAL_CACHE"]

    if cluster_cache:
        os.environ["SIE_CLUSTER_CACHE"] = cluster_cache
    elif "SIE_CLUSTER_CACHE" in os.environ:
        del os.environ["SIE_CLUSTER_CACHE"]

    os.environ["SIE_HF_FALLBACK"] = "true" if hf_fallback else "false"

    # Configure tracing via environment variables
    if tracing:
        os.environ["SIE_TRACING_ENABLED"] = "true"
        # Set standard OTel env vars for local Jaeger
        os.environ.setdefault("OTEL_SERVICE_NAME", "sie-server")
        os.environ.setdefault("OTEL_TRACES_EXPORTER", "otlp")
        os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        typer.echo("Tracing enabled - exporting to localhost:4317")
        typer.echo("Start Jaeger with: mise run jaeger")

    typer.echo(f"Starting SIE server on {host}:{port}")

    typer.echo(f"Models directory: {models_dir_resolved}")
    typer.echo(f"Device: {resolved_device}")
    if resolved_devices:
        typer.echo(f"Devices: {', '.join(resolved_devices)}")
    if resolved_device == "mps":
        # Apple-Silicon banner: surface the one-command local OpenAI endpoint so the
        # first-run experience points at the playground / docs / OpenAI base URL.
        typer.echo("")
        typer.echo("  Apple Silicon (Metal) — one local OpenAI-compatible endpoint:")
        typer.echo(f"    →  Playground   http://localhost:{port}/")
        typer.echo(f"    →  API docs     http://localhost:{port}/docs")
        typer.echo(
            f"    →  OpenAI base  http://localhost:{port}/v1   (embeddings · rerank; chat/completions with -b sglang)"
        )
        typer.echo("")
    if cluster_cache:
        typer.echo(f"Cluster cache: {cluster_cache}")
    if not hf_fallback:
        typer.echo("HuggingFace Hub fallback: disabled")

    # Handle preload models (CLI flag takes precedence, then env var from Helm configmap)
    preload_models: list[str] | None = None
    if preload:
        preload_models = [m.strip() for m in preload.split(",") if m.strip()]
        if model_filter:
            invalid = [m for m in preload_models if m not in model_filter]
            if invalid:
                typer.echo(f"Error: Preload model(s) not in model filter: {', '.join(invalid)}", err=True)
                raise typer.Exit(1)
        typer.echo(f"Preload: {len(preload_models)} models will be loaded at startup")
    else:
        preload_env = os.environ.get("SIE_PRELOAD_MODELS")
        if preload_env:
            preload_models = [m.strip() for m in preload_env.split(",") if m.strip()]
            if preload_models:
                if model_filter:
                    invalid = [m for m in preload_models if m not in model_filter]
                    if invalid:
                        typer.echo(f"Error: Preload model(s) not in model filter: {', '.join(invalid)}", err=True)
                        raise typer.Exit(1)
                typer.echo(f"Preload (from env): {len(preload_models)} models will be loaded at startup")

    config = AppStateConfig(
        models_dir=models_dir_resolved,
        device=resolved_device,
        devices=resolved_devices,
        model_filter=model_filter,
        preload_models=preload_models,
        pool_name=pool_name,
    )

    uvicorn_log = "debug" if verbose else log_level.strip().lower()
    if uvicorn_log not in ("critical", "error", "warning", "info", "debug", "trace"):
        uvicorn_log = "info"

    run_server(host=host, port=port, reload=reload, config=config, uvicorn_log_level=uvicorn_log)


if __name__ == "__main__":
    app()
