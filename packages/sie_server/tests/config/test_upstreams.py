"""Operator-defined upstreams: URL rules and credentials by reference."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from sie_server.config.upstreams import (
    Upstream,
    UpstreamConfigError,
    UpstreamCredentialError,
    UpstreamKind,
    load_upstreams,
    validate_upstream_url,
)

CANARY = "sk-canary-6f1d0c9a2b7e4f5a"
RATE_CAP = "    rate_cap: {requests_per_minute: 600, max_concurrency: 32}\n"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "upstreams.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_valid_file_defines_both_kinds(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "upstreams:\n"
        "  team-sie:\n"
        "    kind: sie\n"
        "    base_url: https://sie.example.internal/\n"
        "    api_key_secret: TEAM_SIE_KEY\n" + RATE_CAP + "  local-openai:\n"
        "    kind: openai\n"
        "    base_url: http://127.0.0.1:8000/v1\n"
        "    proxy_url: http://proxy.example.internal:3128\n" + RATE_CAP,
    )

    upstreams = load_upstreams(path)

    assert set(upstreams) == {"team-sie", "local-openai"}
    assert upstreams["team-sie"].kind is UpstreamKind.SIE
    assert upstreams["team-sie"].base_url == "https://sie.example.internal"
    assert upstreams["team-sie"].rate_cap.max_concurrency == 32
    assert upstreams["local-openai"].api_key_secret is None
    assert upstreams["local-openai"].proxy_url == "http://proxy.example.internal:3128"


def test_an_empty_file_defines_no_upstream(tmp_path: Path) -> None:
    assert load_upstreams(write(tmp_path, "")) == {}


@pytest.mark.parametrize(
    "url",
    [
        "https://sie.example.internal",
        "https://host.example.com/v1",
        "http://localhost:8080",
        "http://127.0.0.1:9000",
        "http://127.8.0.1:9000",
        "http://[::1]:8080",
    ],
)
def test_tls_or_loopback_urls_are_accepted(url: str) -> None:
    assert validate_upstream_url(url) == url


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://sie.example.internal", "https outside loopback"),
        ("http://10.0.0.5:8080", "https outside loopback"),
        ("https://user:pass@sie.example.internal", "credentials"),
        ("https://token@sie.example.internal", "credentials"),
        ("https://sie.example.internal/v1?key=abc", "query or a fragment"),
        ("https://sie.example.internal/?", "query or a fragment"),
        ("https://sie.example.internal/#top", "query or a fragment"),
        ("ftp://sie.example.internal", "http or https"),
        ("https:///v1", "name a host"),
        ("https://sie.example.internal:99999", "not a valid URL"),
    ],
)
def test_unsafe_urls_are_rejected(url: str, reason: str) -> None:
    with pytest.raises(UpstreamConfigError, match=reason):
        validate_upstream_url(url)


def test_a_proxy_may_be_plain_http_but_never_carries_credentials() -> None:
    assert validate_upstream_url("http://proxy.example:3128", field="proxy_url", require_tls=False)
    with pytest.raises(UpstreamConfigError, match="proxy_url must not carry credentials"):
        validate_upstream_url("http://user:pw@proxy.example:3128", field="proxy_url", require_tls=False)


@pytest.mark.parametrize(
    "body",
    [
        "upstreams:\n  Team_SIE:\n    kind: sie\n    base_url: https://h.example\n" + RATE_CAP,
        "upstreams:\n  team:\n    kind: grpc\n    base_url: https://h.example\n" + RATE_CAP,
        "upstreams:\n  team:\n    kind: sie\n    base_url: https://h.example\n",
        "upstreams:\n  team:\n    kind: sie\n    base_url: https://h.example\n    extra: 1\n" + RATE_CAP,
        "upstreams:\n  team:\n    kind: sie\n    base_url: https://h.example\n"
        "    rate_cap: {requests_per_minute: 0, max_concurrency: 1}\n",
        "models: {}\n",
    ],
    ids=["bad-name", "unknown-kind", "no-rate-cap", "unknown-field", "zero-rate", "unknown-top-level-key"],
)
def test_invalid_definitions_are_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(UpstreamConfigError, match="is invalid"):
        load_upstreams(write(tmp_path, body))


@pytest.mark.parametrize(
    "body",
    [
        "upstreams:\n  team:\n    kind: sie\n    base_url: https://a.example\n"
        + RATE_CAP
        + "  team:\n    kind: sie\n    base_url: https://b.example\n"
        + RATE_CAP,
        "upstreams:\n  team:\n    kind: sie\n    base_url: https://a.example\n"
        "    base_url: https://b.example\n" + RATE_CAP,
    ],
    ids=["repeated-upstream", "repeated-field"],
)
def test_a_repeated_key_is_rejected_rather_than_silently_replaced(tmp_path: Path, body: str) -> None:
    with pytest.raises(UpstreamConfigError, match="repeats a key on line"):
        load_upstreams(write(tmp_path, body))


@pytest.mark.parametrize("body", ["upstreams: {[]: {}}\n", "upstreams: {{a: 1}: {}}\n", "? [a]\n: 1\n"])
def test_an_unhashable_key_is_invalid_yaml_not_a_crash(tmp_path: Path, body: str) -> None:
    with pytest.raises(UpstreamConfigError, match="not valid YAML"):
        load_upstreams(write(tmp_path, body))


def test_unreadable_or_malformed_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(UpstreamConfigError, match="could not be read"):
        load_upstreams(tmp_path / "missing.yaml")
    with pytest.raises(UpstreamConfigError, match="not valid YAML"):
        load_upstreams(write(tmp_path, "upstreams: [unclosed\n"))


@pytest.mark.parametrize(
    "field",
    [
        f"    api_key_secret: {CANARY}\n",
        f"    api_key_secret: Bearer {CANARY}\n",
        f"    base_url: https://{CANARY}@sie.example.internal\n",
        f"    base_url: https://sie.example.internal/v1?api_key={CANARY}\n",
    ],
    ids=["literal-key", "literal-bearer", "url-userinfo", "url-query"],
)
def test_a_rejected_credential_is_never_repeated(tmp_path: Path, field: str, caplog: pytest.LogCaptureFixture) -> None:
    base_url = "" if "base_url" in field else "    base_url: https://sie.example.internal\n"
    path = write(tmp_path, "upstreams:\n  team:\n    kind: sie\n" + base_url + field + RATE_CAP)

    caplog.set_level(logging.DEBUG)
    with pytest.raises(UpstreamConfigError) as raised:
        load_upstreams(path)

    assert CANARY not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None or CANARY not in str(raised.value.__context__)
    assert CANARY not in caplog.text


def test_the_credential_is_read_by_reference_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEAM_SIE_KEY", raising=False)
    upstream = Upstream.model_validate(
        {
            "kind": "sie",
            "base_url": "https://sie.example.internal",
            "api_key_secret": "TEAM_SIE_KEY",
            "rate_cap": {"requests_per_minute": 60, "max_concurrency": 4},
        }
    )

    with pytest.raises(UpstreamCredentialError, match="TEAM_SIE_KEY is not set"):
        upstream.api_key()

    monkeypatch.setenv("TEAM_SIE_KEY", CANARY)
    assert upstream.api_key() == CANARY
    for rendering in (repr(upstream), str(upstream), upstream.model_dump_json(), str(upstream.model_dump())):
        assert CANARY not in rendering
