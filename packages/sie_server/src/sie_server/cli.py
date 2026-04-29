from __future__ import annotations

import logging
import os
from pathlib import Path

import typer
from sie_sdk.bundle_utils import find_bundle_for_models, match_bundle_models

import sie_server
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core.loader import load_model_configs
from sie_server.main import run_server

logger = logging.getLogger(__name__)

# Default bundles directory relative to sie_server package
_DEFAULT_BUNDLES_DIR = Path(__file__).parent.parent.parent / "bundles"
DEFAULT_MODELS_DIR = str(Path(str(sie_server.__file__)).parent.parent.parent / "models")


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


def load_bundle(bundle_name: str, bundles_dir: Path, models_dir: str | None = None) -> list[str]:
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
    return match_bundle_models(bundle_path, resolved_models_dir)


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
        matched = find_bundle_for_models(model_list, bundles_dir, models_path)
        if not matched:
            typer.echo(f"Error: No bundle found covering models: {', '.join(model_list)}", err=True)
            raise typer.Exit(1)
        resolved_bundle = matched

    result = collect_bundle_deps(resolved_bundle, bundles_dir, models_path, exclude_cuda=cpu)

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
    from sie_server.api.metrics import router as metrics_router
    from sie_server.api.models import router as models_router
    from sie_server.api.openai_compat import router as openai_router
    from sie_server.api.openapi import setup_custom_openapi_schema
    from sie_server.api.root import router as root_router
    from sie_server.api.score import router as score_router
    from sie_server.api.ws import router as ws_router

    # Build a lightweight FastAPI app with all routers but no lifespan
    # (no GPU init, no model registry, no telemetry, no NATS)
    app_ = FastAPI(
        title="SIE Server",
        description="Search Inference Engine - GPU inference server for search workloads",
        version="0.1.0",
    )
    app_.include_router(root_router)
    app_.include_router(health_router)
    app_.include_router(encode_router)
    app_.include_router(extract_router)
    app_.include_router(score_router)
    app_.include_router(models_router)
    app_.include_router(metrics_router)
    app_.include_router(ws_router)
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
        DEFAULT_MODELS_DIR, "--models-dir", help="Models directory (local path, s3://, or gs://)"
    ),
    bundle: str | None = typer.Option(None, "--bundle", "-b", help="Bundle name to load (from bundles/ dir)"),
    models: str | None = typer.Option(None, "--models", "-m", help="Comma-separated model names to load"),
    local_cache: str | None = typer.Option(None, "--local-cache", help="Local cache directory (default: HF_HOME)"),
    cluster_cache: str | None = typer.Option(None, "--cluster-cache", help="Cluster cache URL (s3:// or gs://)"),
    hf_fallback: bool = typer.Option(True, "--hf-fallback/--no-hf-fallback", help="Enable HuggingFace Hub fallback"),
    reload: bool = typer.Option(default=False, help="Enable auto-reload for development"),
    tracing: bool = typer.Option(default=False, help="Enable OpenTelemetry tracing (exports to localhost:4317)"),
    instrumentation: bool = typer.Option(False, "--instrumentation", "-i", help="Enable batch instrumentation logging"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    preload: str | None = typer.Option(None, "--preload", help="Comma-separated model names to preload at startup"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Enable structured JSON logging (for Loki)"),
) -> None:
    """Start the SIE inference server."""
    from sie_sdk.storage import is_cloud_path

    from sie_server.core.logging import configure_logging

    # Configure logging (supports JSON format for Loki compatibility)
    configure_logging(verbose=verbose, json_format=json_logs or None)

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

    # Handle bundle/models filter
    model_filter: list[str] | None = None
    if bundle and models:
        typer.echo("Error: Cannot specify both --bundle and --models", err=True)
        raise typer.Exit(1)

    if bundle:
        bundles_dir = _DEFAULT_BUNDLES_DIR
        if not bundles_dir.exists():
            typer.echo(f"Error: Bundles directory not found: {bundles_dir}", err=True)
            raise typer.Exit(1)
        try:
            model_filter = load_bundle(bundle, bundles_dir)
            typer.echo(f"Bundle '{bundle}': {len(model_filter)} models")
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from e

    if models:
        model_filter = [m.strip() for m in models.split(",") if m.strip()]
        typer.echo(f"Model filter: {len(model_filter)} models")
        # Validate model names exist by loading configs and checking
        all_configs = load_model_configs(models_dir_resolved)
        unknown = [m for m in model_filter if m not in all_configs]
        if unknown:
            typer.echo(f"Error: Unknown model(s): {', '.join(unknown)}", err=True)
            typer.echo(f"Available models: {', '.join(sorted(all_configs.keys())[:10])}...", err=True)
            raise typer.Exit(1)

    os.environ["SIE_INSTRUMENTATION"] = "true" if instrumentation else "false"
    os.environ["SIE_BUNDLE"] = bundle or "default"
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
        model_filter=model_filter,
        preload_models=preload_models,
    )

    run_server(host=host, port=port, reload=reload, config=config)


if __name__ == "__main__":
    app()
