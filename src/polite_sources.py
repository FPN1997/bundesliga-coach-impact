"""
soccerdata readers for Understat and football-data.co.uk that fetch like an
ordinary, identifiable script -- the same rule as for Transfermarkt
(src/transfermarkt_client.py) and FBref (src/fetch_fbref.py): be recognisable,
and stop when a site says no.

soccerdata's request-based readers use `tls_requests`, a client that imitates
a real browser's TLS fingerprint -- a way of not being recognised as a
script. Neither site needs it (checked 29 Sep 2026): football-data.co.uk
serves its CSVs to a plain request, and Understat's data endpoint only wants
the X-Requested-With header its own page sends, which soccerdata adds anyway.
So these readers use a plain `requests` session with a User-Agent that names
the project, and a 403 or 429 stops the fetch at once instead of going into
soccerdata's five retries.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests
import soccerdata as sd

log = logging.getLogger(__name__)

USER_AGENT = "bundesliga-coach-impact/0.1 (+https://github.com/FPN1997/bundesliga-coach-impact)"


class SourceRefused(BaseException):
    """A 403 or 429: the site said no. A BaseException, so it gets out of
    soccerdata's retry loop (which catches Exception) instead of being retried."""


def _refuse_on_block(response: requests.Response, *args, **kwargs) -> None:
    if response.status_code in (403, 429):
        raise SourceRefused(f"{response.url} -> HTTP {response.status_code}")


def plain_session(headers: dict[str, str] | None = None) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, **(headers or {})})
    session.hooks["response"].append(_refuse_on_block)
    return session


class PoliteUnderstat(sd.Understat):
    def _init_session(self, headers=None):
        return plain_session(headers)

    @classmethod
    def _all_leagues(cls) -> dict[str, str]:  # soccerdata looks leagues up by class name
        return sd.Understat._all_leagues()


class PoliteMatchHistory(sd.MatchHistory):
    def _init_session(self, headers=None):
        return plain_session(headers)

    @classmethod
    def _all_leagues(cls) -> dict[str, str]:
        return sd.MatchHistory._all_leagues()


def _strip_bom_from_match_history_cache() -> int:
    """football-data.co.uk's 2021-22 Premier League file starts with a UTF-8
    byte-order mark. soccerdata then can't find its first column ("Div"),
    fails on that season read alone, and silently drops it when reading many
    seasons at once. Strip the mark from the cached copies; returns how many."""
    from soccerdata._config import DATA_DIR

    fixed = 0
    for path in (Path(DATA_DIR) / "MatchHistory").glob("*.csv"):
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            path.write_bytes(data[3:])
            fixed += 1
    return fixed


def read_match_history(league: str, season: str) -> pd.DataFrame:
    """One league-season (cached by soccerdata). Read one at a time, so a bad
    file fails loudly instead of vanishing from a multi-season read."""
    try:
        return PoliteMatchHistory(leagues=league, seasons=[season]).read_games().reset_index()
    except KeyError:
        if not _strip_bom_from_match_history_cache():
            raise
        log.info("Stripped a byte-order mark from a cached football-data file; reading %s %s again",
                 league, season)
        return PoliteMatchHistory(leagues=league, seasons=[season]).read_games().reset_index()
