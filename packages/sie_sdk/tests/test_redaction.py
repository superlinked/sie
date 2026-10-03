"""Shared redaction helpers (#2339) — canonical behavior both planes rely on."""

import pytest
from sie_sdk.redaction import REDACTED, endpoint_origin_for_log, mask_token, redact_secrets


def test_mask_token_long() -> None:
    # Same shape as the Rust gateway's middleware::auth::mask_token.
    assert mask_token("secret-token-123") == "************-123"


def test_mask_token_short_fully_starred() -> None:
    assert mask_token("") == "****"
    assert mask_token("abc") == "****"
    assert mask_token("abcd") == "****"
    assert mask_token("abcde") == "*bcde"


def test_endpoint_origin_strips_credentials_path_and_query() -> None:
    assert (
        endpoint_origin_for_log("https://user:secret@collector.example:4318/v1/metrics?token=x")
        == "https://collector.example:4318"
    )
    assert endpoint_origin_for_log("http://127.0.0.1/v1/metrics") == "http://127.0.0.1"


def test_endpoint_origin_redacts_non_http_or_unparseable() -> None:
    assert endpoint_origin_for_log("not a URL with secret") == "<redacted>"
    assert endpoint_origin_for_log("ftp://host/path") == "<redacted>"
    assert endpoint_origin_for_log("https://bad:port:99999") == "<redacted>"


def test_endpoint_origin_brackets_ipv6() -> None:
    assert endpoint_origin_for_log("http://[::1]:4318/v1/metrics") == "http://[::1]:4318"


# Fixtures are concatenated at runtime: a secret-shaped literal in a committed
# file trips the repo's gitleaks pre-commit hook even when it is a test double.
_FAKE_BEARER = "abcdefghij" + "klmnop1234"
_FAKE_API_KEY = "sk-" + "sie-0123456789abcdefghij"
_FAKE_AWS_KEY_ID = "AKIA" + "IOSFODNN7EXAMPLE"
# STS temporary credentials use the ASIA prefix; still live for the session.
_FAKE_AWS_STS_KEY_ID = "ASIA" + "IOSFODNN7EXAMPLE"
_FAKE_JWT_SIGNATURE = "dBjftJeZ4CVPmB92" + "K27uhbUJU1p1r_wW1gFWFOEjXk"
_FAKE_JWT = ".".join(["eyJ" + "hbGciOiJIUzI1NiJ9", "eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0", _FAKE_JWT_SIGNATURE])


@pytest.mark.parametrize(
    ("secret", "must_not_survive"),
    [
        (f"Bearer {_FAKE_BEARER}", _FAKE_BEARER),
        (_FAKE_API_KEY, _FAKE_API_KEY),
        (_FAKE_AWS_KEY_ID, _FAKE_AWS_KEY_ID),
        (_FAKE_AWS_STS_KEY_ID, _FAKE_AWS_STS_KEY_ID),
        (_FAKE_JWT, _FAKE_JWT_SIGNATURE),
        ("api_key=hunter2andthensome", "hunter2andthensome"),
        ('refresh_token: "rt-abcdefghijklmnop"', "rt-abcdefghijklmnop"),
        ("client_secret=zzzzzzzzzzzz", "zzzzzzzzzzzz"),
        ("ops@example.com", "ops@example.com"),
    ],
)
def test_redact_secrets_scrubs_credential_and_pii_shapes(secret: str, must_not_survive: str) -> None:
    scrubbed = redact_secrets(f"upstream said: {secret} -- retry later")
    assert must_not_survive not in scrubbed
    assert REDACTED in scrubbed
    # The surrounding diagnostic survives.
    assert "upstream said" in scrubbed
    assert "retry later" in scrubbed


def test_redact_secrets_keeps_the_key_name_for_key_value_shapes() -> None:
    assert redact_secrets("api_key=abcdefgh") == f"api_key={REDACTED}"
    assert redact_secrets("access_token: abcdefgh") == f"access_token={REDACTED}"


def test_redact_secrets_leaves_ordinary_diagnostics_intact() -> None:
    message = "no healthy workers for BAAI/bge-m3 (queue depth 12, gpu l4)"
    assert redact_secrets(message) == message
