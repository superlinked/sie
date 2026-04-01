"""CLI for SIE Router."""

import dataclasses
import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Annotated, Any

import typer

from sie_router import __version__
from sie_router.app.app_state_config import (
    ENV_K8S_NAMESPACE,
    ENV_K8S_PORT,
    ENV_K8S_SERVICE,
    ENV_KUBERNETES,
    AppStateConfig,
)
from sie_router.main import run_server
from sie_router.types import AuditEntry


def _parse_k8s_port_default() -> int:
    try:
        return int(os.environ.get(ENV_K8S_PORT, "8080"))
    except ValueError:
        return 8080


app = typer.Typer(
    name="sie-router",
    help="SIE Router - Stateless request router for elastic cloud deployments",
    no_args_is_help=True,
)


# Derived from AuditEntry dataclass — single source of truth for audit fields.
OPTIONAL_FIELDS = tuple(f.name for f in dataclasses.fields(AuditEntry))


class JSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs for Loki compatibility."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add optional structured fields if present
        log_data |= {field: value for field in OPTIONAL_FIELDS if (value := getattr(record, field, None)) is not None}

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, default=str)


def setup_logging(level: str, *, json_format: bool = False) -> None:
    """Configure logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_format: Use structured JSON format for Loki.
    """
    log_level = getattr(logging, level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
    root_logger.addHandler(handler)


@app.command()
def serve(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 8081,
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to")] = "0.0.0.0",  # noqa: S104 — intentional bind to all interfaces for router
    workers: Annotated[
        list[str] | None,
        typer.Option("--worker", "-w", help="Worker URLs (can specify multiple)"),
    ] = None,
    kubernetes: Annotated[
        bool,
        typer.Option("--kubernetes", "-k", help="Use Kubernetes discovery"),
    ] = os.environ.get(ENV_KUBERNETES, "").lower() == "true",
    k8s_namespace: Annotated[
        str,
        typer.Option("--k8s-namespace", help="Kubernetes namespace"),
    ] = os.environ.get(ENV_K8S_NAMESPACE, "default"),
    k8s_service: Annotated[
        str,
        typer.Option("--k8s-service", help="Kubernetes service name"),
    ] = os.environ.get(ENV_K8S_SERVICE, "sie-worker"),
    k8s_port: Annotated[
        int,
        typer.Option("--k8s-port", help="Worker port for K8s discovery"),
    ] = _parse_k8s_port_default(),
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="Log level"),
    ] = "info",
    json_logs: Annotated[
        bool,
        typer.Option("--json-logs", help="Enable structured JSON logging for Loki"),
    ] = False,
    reload: Annotated[
        bool,
        typer.Option("--reload", "-r", help="Enable auto-reload for development"),
    ] = False,
) -> None:
    """Start the SIE Router server.

    Routes requests to SIE workers based on GPU type and model affinity.

    Examples:
        # Static worker discovery
        sie-router serve -w http://worker-0:8080 -w http://worker-1:8080

        # Kubernetes discovery
        sie-router serve --kubernetes --k8s-service sie-worker

        # Development with auto-reload
        sie-router serve -w http://localhost:8080 --reload
    """
    if workers is None:
        workers = []

    # Check for SIE_LOG_JSON env var if not explicitly set
    use_json = json_logs or os.environ.get("SIE_LOG_JSON", "").lower() in ("true", "1", "yes")
    setup_logging(log_level, json_format=use_json)

    # Build configuration
    config = AppStateConfig(
        worker_urls=workers,
        use_kubernetes=kubernetes,
        k8s_namespace=k8s_namespace,
        k8s_service=k8s_service,
        k8s_port=k8s_port,
        enable_pools=False,
        enable_hot_reload=False,
    )

    # Run the server
    run_server(
        host=host,
        port=port,
        reload=reload,
        config=config,
        log_level=log_level,
    )


@app.command()
def version() -> None:
    """Show version information."""
    print(f"sie-router {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
