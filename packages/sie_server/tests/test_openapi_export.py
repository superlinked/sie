import json
from pathlib import Path

from sie_server.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_openapi_stdout() -> None:
    """CLI outputs valid OpenAPI JSON to stdout."""
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0, result.output
    spec = json.loads(result.output)
    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"] == "SIE Server"


def test_openapi_has_expected_paths() -> None:
    """Exported spec contains all core API paths."""
    result = runner.invoke(app, ["openapi"])
    spec = json.loads(result.output)
    paths = set(spec["paths"].keys())
    for expected in ["/v1/encode/{model}", "/v1/extract/{model}", "/v1/score/{model}", "/v1/models"]:
        assert expected in paths, f"Missing path: {expected}"


def test_openapi_has_request_body_schemas() -> None:
    """Custom Pydantic request body schemas are injected."""
    result = runner.invoke(app, ["openapi"])
    spec = json.loads(result.output)
    schemas = spec.get("components", {}).get("schemas", {})
    for name in ["EncodeRequestModel", "ExtractRequestModel", "ScoreRequestModel"]:
        assert name in schemas, f"Missing schema: {name}"


def test_openapi_output_file(tmp_path: Path) -> None:
    """CLI writes spec to a file when --output is given."""
    out = tmp_path / "spec.json"
    result = runner.invoke(app, ["openapi", "--output", str(out)])
    assert result.exit_code == 0, result.output
    spec = json.loads(out.read_text())
    assert spec["openapi"].startswith("3.")


def test_openapi_version_from_package() -> None:
    """Spec version matches the installed sie-server package version."""
    from importlib.metadata import version as pkg_version

    result = runner.invoke(app, ["openapi"])
    spec = json.loads(result.output)
    assert spec["info"]["version"] == pkg_version("sie-server")
