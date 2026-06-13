"""Throttled, retrying HTTP GET shared by the fetch stages (00: Movie Madness
WP REST, 02: TMDB). One instance = one politeness budget."""

import time

import requests


class ThrottledGetter:
    def __init__(self, interval_s: float, user_agent: str, headers: dict | None = None):
        self.interval_s = interval_s
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        if headers:
            self.session.headers.update(headers)
        self._last_t = 0.0

    def get(self, url: str, params: dict | None = None, max_retries: int = 6, timeout: int = 60):
        """GET with politeness throttle and exponential backoff; 429/5xx are
        retryable and Retry-After is honored."""
        for attempt in range(max_retries):
            wait = self.interval_s - (time.monotonic() - self._last_t)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_t = time.monotonic()
                resp = self.session.get(url, params=params, timeout=timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    backoff = float(retry_after) if retry_after else min(2**attempt * 5, 120)
                    if attempt == max_retries - 1:
                        resp.raise_for_status()
                    print(f"  HTTP {resp.status_code}, backing off {backoff:.0f}s...")
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                backoff = min(2**attempt * 5, 120)
                print(f"  {type(e).__name__}, retrying in {backoff:.0f}s...")
                time.sleep(backoff)
        raise RuntimeError("unreachable")
