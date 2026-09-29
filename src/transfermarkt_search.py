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

from src import transfermarkt_client as tm

log = logging.getLogger(__name__)

CLUB_LINK_RE = re.compile(r"/([a-z0-9-]+)/startseite/verein/(\d+)")
RESERVE_YOUTH_RE = re.compile(r"-(ii+|u1[5-9]|u2[0-3]|jugend|frauen)(-|$)")


def search_club_id(club_name: str, *, timeout: int = 20) -> int | None:
    """Best-guess Transfermarkt club id for `club_name`, or None if search
    returned nothing usable. Not verified against the club name -- see
    module docstring."""
    url = f"https://www.transfermarkt.de/schnellsuche/ergebnis/schnellsuche?query={club_name}"
    try:
        # Through the shared client: paced, budgeted, and a block raises
        # TransfermarktBlocked (it used to read as "no result" and move on to
        # the next club -- exactly the hammering the block handling prevents).
        resp = tm.get(url, timeout=timeout, allow_redirects=True)
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

    candidates = club_results(resp.text)
    if not candidates:
        log.warning("No club section in Transfermarkt search results for %r", club_name)
        return None
    # Prefer a result whose name contains every word of the name searched for
    # ("FC Arsenal" for "Arsenal"); otherwise the first senior-team result.
    words = [_fold(w) for w in club_name.replace("-", " ").split() if len(w) >= 3]
    for name, club_id in candidates:
        if words and all(w in _fold(name).replace("-", " ") for w in words):
            return club_id
    return candidates[0][1]


def club_results(html: str) -> list[tuple[str, int]]:
    """(name, id) for each senior-team result in the page's CLUBS section, in
    relevance order. The page lists coaches first ("Suchergebnisse zu
    Trainern"), and a coach result links to that coach's club -- taking the
    first club link on the page used to return the club of whichever coach
    matched the query (Arsenal -> Atletico Mancha Real, 29 Sep 2026)."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    header = next((h for h in soup.find_all("h2") if "Vereinen" in h.get_text()), None)
    if header is None:
        return []
    box = header.find_parent("div", class_="box") or header.parent
    out = []
    for a in box.select("td.hauptlink a"):
        match = CLUB_LINK_RE.search(a.get("href", ""))
        if match and not RESERVE_YOUTH_RE.search(match.group(1)):
            out.append((a.get_text(strip=True), int(match.group(2))))
    return out


def _fold(text: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
