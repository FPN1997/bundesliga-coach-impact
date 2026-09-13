"""
Auto-resolve a club name to its Transfermarkt numeric id via live search,
instead of requiring a hand-added `config.CLUB_TRANSFERMARKT_ID` entry for
every team that shows up (which has already happened three times as
`config.SEASONS` widened and seasons turned over -- see README "Known rough
edges").

Uses transfermarkt.de's "schnellsuche" (quick search), same domain as
fetch_coach_history.py post-WAF-block. Verified live for several clubs: an
exact single match redirects straight to the club page; otherwise results
are returned in relevance order with the senior team FIRST, reserve/youth
teams after. The first-result ordering held in every live check done here,
but isn't a documented guarantee, so this filters out obvious reserve/
youth slugs explicitly (`-ii`, `-u15` through `-u23`, `-jugend`, `-frauen`)
rather than trusting position alone.

This only finds a CANDIDATE id -- it does not verify the candidate is
actually the right club. Callers should verify independently (e.g.
fetch_coach_history.py already checks the tenure page's <title> against
the expected club name for exactly this reason: a wrong id would otherwise
silently return some OTHER club's valid-looking page).
"""

from __future__ import annotations

import logging
import re

import requests

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
}
CLUB_LINK_RE = re.compile(r"/([a-z0-9-]+)/startseite/verein/(\d+)")
RESERVE_YOUTH_RE = re.compile(r"-(ii+|u1[5-9]|u2[0-3]|jugend|frauen)(-|$)")


def search_club_id(club_name: str, *, timeout: int = 20) -> int | None:
    """Best-guess Transfermarkt club id for `club_name`, or None if search
    returned nothing usable. Not verified against the club name -- see
    module docstring."""
    url = f"https://www.transfermarkt.de/schnellsuche/ergebnis/schnellsuche?query={club_name}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Transfermarkt search failed for %r: %s", club_name, exc)
        return None

    # A single exact match redirects straight to the club page.
    direct = CLUB_LINK_RE.search(resp.url)
    if direct:
        return int(direct.group(2))

    if not resp.text.strip():
        log.warning("Empty search response for %r (WAF challenge? see "
                    "fetch_coach_history.py's docstring)", club_name)
        return None

    # Results page: pull every club link in document order (search-relevance
    # order, verified live -- NOT the order BeautifulSoup/grep would give
    # you after any kind of sorting) and take the first one that doesn't
    # look like a reserve/youth/women's side.
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    for a in soup.find_all("a", href=CLUB_LINK_RE):
        match = CLUB_LINK_RE.search(a["href"])
        slug = match.group(1)
        if RESERVE_YOUTH_RE.search(slug):
            continue
        return int(match.group(2))

    log.warning("No usable club link found in Transfermarkt search results for %r", club_name)
    return None
