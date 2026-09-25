"""
One polite, budgeted client for every Transfermarkt request in a run.

Transfermarkt's firewall blocked this project once, after ~1,100 requests in
an hour (HTTP 405 + x-amzn-waf-action: captcha). Since then every request --
coach histories, club search, squads, injuries -- goes through `get()` here:

- spaced REQUEST_DELAY seconds apart (plus up to 1 s of jitter),
- counted against ONE budget per process (MAX_LIVE_REQUESTS_PER_RUN), so
  running several fetch steps in one command can't add up to more traffic
  than a single step was allowed -- `bundesliga backfill` relies on this to
  spend the budget in priority order,
- stopped at the first sign of a block (TransfermarktBlocked). A block is
  the site saying no; the only right response is to stop and wait, never to
  retry harder or route around it.
"""

from __future__ import annotations

import logging
import random
import time

import requests

log = logging.getLogger(__name__)

HEADERS = {
    # Transfermarkt serves a normal page to a standard browser UA; no
    # special auth or JS rendering needed.
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
}
REQUEST_DELAY = 4.0             # seconds between live requests (plus up to 1 s jitter)
MAX_LIVE_REQUESTS_PER_RUN = 400


class TransfermarktBlocked(RuntimeError):
    """Transfermarkt's firewall is blocking or rate-limiting us. Stop and
    back off -- never retry harder or try to get around it."""


class RequestBudgetReached(RuntimeError):
    """This run's MAX_LIVE_REQUESTS_PER_RUN is used up -- a planned stop, not an error."""


def raise_if_blocked(resp: requests.Response, url: str) -> None:
    """Transfermarkt signals a block in several ways: an AWS WAF action
    header (seen live: 405 + x-amzn-waf-action: captcha after ~1,100 requests
    in an hour), a 403/405/429, or a 200/202 with an empty body (the
    challenge that took transfermarkt.com offline for this project)."""
    waf = resp.headers.get("x-amzn-waf-action")
    if waf or resp.status_code in (403, 405, 429) or not resp.text.strip():
        raise TransfermarktBlocked(
            f"{url} -> HTTP {resp.status_code}" + (f", firewall action {waf!r}" if waf else "")
            + f", {len(resp.text)} bytes. Transfermarkt is blocking requests -- wait and retry later.")


class _Budget:
    def __init__(self) -> None:
        self.live_requests = 0
        self._last = 0.0

    def get(self, url: str, *, timeout: int = 30, allow_redirects: bool = True) -> requests.Response:
        if self.live_requests >= MAX_LIVE_REQUESTS_PER_RUN:
            raise RequestBudgetReached(f"{self.live_requests} live Transfermarkt requests this run")
        wait = REQUEST_DELAY + random.random() - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=allow_redirects)
        self._last = time.time()
        self.live_requests += 1
        raise_if_blocked(resp, url)
        if self.live_requests % 50 == 0:
            log.info("... %d live Transfermarkt requests so far this run", self.live_requests)
        return resp


_budget = _Budget()


def get(url: str, **kwargs) -> requests.Response:
    """Paced, budgeted GET; raises TransfermarktBlocked or RequestBudgetReached."""
    return _budget.get(url, **kwargs)


def live_requests() -> int:
    return _budget.live_requests


def remaining() -> int:
    return max(0, MAX_LIVE_REQUESTS_PER_RUN - _budget.live_requests)


def reset() -> None:
    """Start a fresh budget (tests; one process = one run otherwise)."""
    global _budget
    _budget = _Budget()
