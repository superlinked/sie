import random
import time

import requests

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0"
}
DEFAULT_TIMEOUT_S = 20
DEFAULT_MAX_ATTEMPTS = 5
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_SESSION = requests.Session()
_SESSION.headers.update(DEFAULT_HEADERS)


def _compute_backoff_delay(response, attempt, backoff_base_s, jitter_s):
    retry_after = None
    if response is not None:
        retry_after = response.headers.get("Retry-After")

    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    exponential_delay = backoff_base_s * (2 ** (attempt - 1))
    return exponential_delay + random.uniform(0, jitter_s)


def get_json(
    url,
    *,
    params=None,
    headers=None,
    timeout=DEFAULT_TIMEOUT_S,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    backoff_base_s=1.0,
    jitter_s=0.25,
):
    merged_headers = {}
    if headers:
        merged_headers.update(headers)

    last_error = None
    for attempt in range(1, max_attempts + 1):
        response = None
        try:
            response = _SESSION.get(url, params=params, headers=merged_headers or None, timeout=timeout)
            if response.status_code == 200:
                return response.json()

            if response.status_code not in RETRYABLE_STATUS_CODES:
                response.raise_for_status()

            last_error = requests.HTTPError(
                f"Request failed with status {response.status_code} for {url}",
                response=response,
            )
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last_error = exc
            if isinstance(exc, requests.HTTPError) and response is not None and response.status_code not in RETRYABLE_STATUS_CODES:
                raise

        if attempt == max_attempts:
            break

        delay_s = _compute_backoff_delay(response, attempt, backoff_base_s, jitter_s)
        time.sleep(delay_s)

    raise RuntimeError(f"Failed to fetch {url} after {max_attempts} attempts.") from last_error
