import dataclasses
import logging
import sys
from datetime import UTC, datetime
from typing import Annotated, Any

import orjson
import typer
import uvicorn

from sie_config.types import AuditEntry

app = typer.Typer(
    name="sie-config",
    help="SIE Config Service - Config control plane for SIE clusters",
    no_args_is_help=True,
)


# Derived from AuditEntry dataclass -- single source of truth for audit fields.
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
        return orjson.dumps(log_data, default=str).decode()


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
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 8080,
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to")] = "0.0.0.0",  # noqa: S104 -- intentional bind to all interfaces
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            "-l",
            envvar="SIE_LOG_LEVEL",
            help="Log level (DEBUG, INFO, WARNING, ERROR).",
        ),
    ] = "info",
    json_logs: Annotated[
        bool,
        typer.Option(
            "--json-logs",
            envvar="SIE_LOG_JSON",
            help="Enable structured JSON logging for Loki.",
        ),
    ] = False,
    reload: Annotated[
        bool,
        typer.Option("--reload", "-r", help="Enable auto-reload for development"),
    ] = False,
) -> None:
    """Start the SIE Config Service."""
    setup_logging(log_level, json_format=json_logs)

    uvicorn.run(
        "sie_config.app_factory:AppFactory.create_app",
        host=host,
        port=port,
        log_level=log_level.lower(),
        reload=reload,
        factory=True,
    )


@app.command()
def version() -> None:
    """Show version information."""
    from sie_config import __version__

    print(f"sie-config {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
