"""Shared redaction helpers (#2339) — canonical behavior both planes rely on."""

from sie_sdk.redaction import endpoint_origin_for_log, mask_token


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
