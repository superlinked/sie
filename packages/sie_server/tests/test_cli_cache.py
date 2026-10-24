"""The serve CLI preserves cache policy unless an explicit flag overrides it."""

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from sie_sdk.cache import get_cache_config
from sie_server import cli
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    saved = dict(os.environ)
    for name in tuple(os.environ):
        if name.startswith("SIE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path / "data"))
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.parametrize(
    "environment",
    [None, "true", "false", "1", "0", "yes", "no", "TRUE", "FALSE", "on", "off", "", "invalid", " true "],
)
@pytest.mark.parametrize("flag", [None, "--hf-fallback", "--no-hf-fallback"])
def test_cli_cache_policy_precedence(
    monkeypatch: pytest.MonkeyPatch, environment: str | None, flag: str | None
) -> None:
    if environment is not None:
        monkeypatch.setenv("SIE_HF_FALLBACK", environment)
    environment_policy = get_cache_config().hf_fallback
    observed = []
    run_server = Mock(side_effect=lambda **_kwargs: observed.append(get_cache_config().hf_fallback))
    monkeypatch.setattr(cli, "run_server", run_server)
    args = ["serve", "--host", "127.0.0.1"]
    if flag is not None:
        args.append(flag)

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    expected = flag == "--hf-fallback" if flag is not None else environment_policy
    assert observed == [expected]
    assert os.environ.get("SIE_HF_FALLBACK") == (environment if flag is None else str(expected).lower())
    assert ("HuggingFace Hub fallback: disabled" in result.output) is (not expected)
    run_server.assert_called_once()


@pytest.mark.parametrize(
    ("option", "environment_name", "field"),
    [
        ("--local-cache", "SIE_LOCAL_CACHE", "local_cache"),
        ("--cluster-cache", "SIE_CLUSTER_CACHE", "cluster_cache"),
    ],
)
@pytest.mark.parametrize("has_environment", [False, True])
@pytest.mark.parametrize("flag_value", [None, "override", ""])
def test_cli_cache_directory_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    option: str,
    environment_name: str,
    field: str,
    has_environment: bool,
    flag_value: str | None,
) -> None:
    default_value = getattr(get_cache_config(), field)
    environment_value = str(tmp_path / "environment-cache") if has_environment else None
    if environment_value is not None:
        monkeypatch.setenv(environment_name, environment_value)
    monkeypatch.setenv("SIE_HF_FALLBACK", "false")
    initial_value = getattr(get_cache_config(), field)
    observed = []
    monkeypatch.setattr(cli, "run_server", Mock(side_effect=lambda **_kwargs: observed.append(get_cache_config())))
    args = ["serve", "--host", "127.0.0.1"]
    if flag_value is not None:
        args.extend([option, flag_value])

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    if flag_value is None:
        expected_value = initial_value
        expected_environment = environment_value
    elif flag_value:
        expected_value = Path(flag_value).resolve() if field == "local_cache" else flag_value
        expected_environment = str(expected_value)
    else:
        expected_value = default_value
        expected_environment = None
    assert len(observed) == 1
    assert getattr(observed[0], field) == expected_value
    assert observed[0].hf_fallback is False
    assert os.environ.get(environment_name) == expected_environment
    assert ("Cluster cache:" in result.output) is bool(observed[0].cluster_cache)


@pytest.mark.parametrize("source", ["environment", "flag"])
@pytest.mark.parametrize(
    "cache_url",
    [
        "oss://example-user:synthetic-cache-marker@bucket/cache",
        "abfss://container@account.invalid/cache?sv=example&sig=synthetic-sas-marker",
    ],
)
def test_cluster_cache_banner_never_echoes_the_url(
    monkeypatch: pytest.MonkeyPatch, source: str, cache_url: str
) -> None:
    observed = []
    monkeypatch.setattr(cli, "run_server", Mock(side_effect=lambda **_kwargs: observed.append(get_cache_config())))
    args = ["serve", "--host", "127.0.0.1"]
    if source == "environment":
        monkeypatch.setenv("SIE_CLUSTER_CACHE", cache_url)
    else:
        args.extend(["--cluster-cache", cache_url])

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    assert len(observed) == 1
    assert observed[0].cluster_cache == cache_url
    assert os.environ["SIE_CLUSTER_CACHE"] == cache_url
    assert "Cluster cache: configured" in result.output
    assert cache_url not in result.output
    assert "synthetic-cache-marker" not in result.output
    assert "synthetic-sas-marker" not in result.output
