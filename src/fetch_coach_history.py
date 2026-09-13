"""
Build a (team, coach, start_date, end_date) tenure table by scraping each
club's "Trainerhistorie" (manager history) page on Transfermarkt.

Wikipedia was the original plan here -- English Wikipedia turned out not to
have "List of <club> managers" pages for Bundesliga clubs (verified live:
every guessed URL 404'd), so this scrapes Transfermarkt directly instead.
That worked cleanly at first (a normal browser User-Agent got a 200, no
login/JS challenge), and the URL only cares about the numeric club ID --
config.CLUB_TRANSFERMARKT_ID -- the slug text in the URL is cosmetic.

  https://www.transfermarkt.de/<any-slug>/mitarbeiterhistorie/verein/<id>/plus/1

**Domain note, found live**: `.com` started returning an AWS WAF
"challenge" response (HTTP 202, empty body, `x-amzn-waf-action: challenge`)
for every request -- including the plain homepage, so not something
rate-limiting or a smarter User-Agent fixes -- a few days after this was
first built and working. `.de` was verified live to still serve the exact
same page (same "items" table, same club IDs, dates as DD.MM.YYYY instead
of DD/MM/YYYY -- `dateutil` handles both) with no challenge, so that's what
this hits now. If `.de` ever starts getting challenged too, that's the
first thing to check again -- Transfermarkt's bot-detection is evidently
still evolving, not a one-time fix.

Verified live against Bayern Munich (id 27) and Heidenheim (id 2036): the
page's <title> is used as a sanity check that the ID actually points at the
expected club, since a wrong ID silently returns SOME club's valid page
rather than an error.

Output is written to data/processed/coach_history.csv. As before,
data/coach_history_manual.csv is consulted too and overrides/augments
whatever this scraper produces -- use it for any club missing an ID below,
or to patch a spell the table got wrong (very short caretaker spells are
the likeliest thing to look odd).

**Resilience note**: if most/all clubs fail to fetch (e.g. the domain gets
blocked again), this refuses to overwrite an existing, larger
coach_history.csv with the near-empty result -- via the shared
`src/data_guard.py` guard, also used by fetch_fbref.py and
fetch_understat.py. That happened once already, here specifically: a
fully-blocked run silently wrote a 1-row file (just the manual CSV
fallback) over what had been ~1600 real rows, which then broke
coach_impact.py's before/after windows and everything built on it. The
guard trades "always write something" for "never silently regress" --
exactly the failure mode every fetch script here needs to be paranoid
about, since it degrades gracefully-looking (a warning, not a crash) all
the way to a badly wrong downstream dataset.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

import config
from src.data_guard import existing_csv_row_count, guard_against_shrinkage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    # Transfermarkt serves a normal page to a standard browser UA; no
    # special auth or JS rendering needed for this page.
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
}
NAME_DOB_RE = re.compile(r"\s+\d{2}/\d{2}/\d{4}$")


def _parse_date(value: str) -> pd.Timestamp | None:
    value = (value or "").strip()
    if not value or value in {"-", "–", "—"}:
        # Blank "End of time in post" means still in the job -- return None
        # (NaT) and let build_dataset._assign_coach's fillna(2100-01-01)
        # treat it as open-ended. An earlier version of this function
        # substituted today's date here instead, which silently capped
        # every still-active coach's tenure at scrape time -- any future
        # fixture (postponed matches, the rest of an in-progress season)
        # then fell outside every window and got no coach assigned at all.
        return None
    try:
        return pd.Timestamp(dateparser.parse(value, dayfirst=True, fuzzy=True))
    except (ValueError, OverflowError):
        return None


def _fetch_club_table(club: str, club_id: int) -> pd.DataFrame:
    url = f"https://www.transfermarkt.de/verein/mitarbeiterhistorie/verein/{club_id}/plus/1"
    log.info("Fetching manager history for %s (Transfermarkt id %d)", club, club_id)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    if not resp.text.strip():
        # A 200/202 with an empty body is Transfermarkt's WAF challenge
        # response, not "no data" -- distinguish it from a genuinely missing
        # table so the log points at the right fix (see module docstring).
        waf = resp.headers.get("x-amzn-waf-action")
        raise ValueError(f"Empty response body from {url}"
                          + (f" (x-amzn-waf-action: {waf} -- likely bot-blocked, "
                             f"not a real 'no data' response)" if waf else ""))

    cache_path = Path(config.RAW_DIR) / f"transfermarkt_{club.replace(' ', '_')}.html"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(resp.text, encoding="utf-8")

    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    # Loose sanity check the ID resolved to the club we asked for -- compare
    # the first "significant" token of the club name against the title.
    key_token = club.split()[0].lower()
    if key_token not in title.lower():
        log.warning("Transfermarkt id %d's page title (%r) doesn't obviously "
                    "match %r -- double check config.CLUB_TRANSFERMARKT_ID[%r].",
                    club_id, title, club, club)

    tables = soup.find_all("table")
    items_tables = [t for t in tables if t.get("class") and "items" in t.get("class")]
    if not items_tables:
        raise ValueError(f"No staff-history table found on {url}")
    table = items_tables[0]
    tbody = table.find("tbody")
    if tbody is None:
        raise ValueError(f"Staff-history table on {url} has no <tbody>")

    rows = []
    for tr in tbody.find_all("tr", recursive=False):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 4:
            continue
        # The name cell is a nested inline-table: a linked name (with a
        # `title` attribute) above a date-of-birth row. Read the name from
        # the link itself rather than the cell's full text -- some coaches
        # have no recorded DOB (e.g. Transfermarkt shows "N/A" as a link
        # there instead of a date), which broke the old trailing-date-regex
        # approach and left "N/A" stuck on the name.
        name_link = cells[0].find("a", title=True)
        coach = name_link["title"].strip() if name_link else NAME_DOB_RE.sub(
            "", cells[0].get_text(" ", strip=True)
        ).strip()
        if not coach:
            continue
        start = _parse_date(cells[2].get_text(strip=True))
        end = _parse_date(cells[3].get_text(strip=True))
        if start is None:
            continue
        rows.append({"team": club, "coach": coach, "start_date": start, "end_date": end})

    return pd.DataFrame(rows)


def fetch_coach_history() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for club, club_id in config.CLUB_TRANSFERMARKT_ID.items():
        if not club_id:
            log.warning("No Transfermarkt id configured for %s -- add one to "
                        "config.CLUB_TRANSFERMARKT_ID or use the manual CSV.", club)
            continue
        try:
            frames.append(_fetch_club_table(club, club_id))
        except Exception as exc:  # best-effort scraper, log and move on
            log.error("Failed to parse manager history for %s: %s", club, exc)
            failures.append(club)

    tm_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["team", "coach", "start_date", "end_date"]
    )

    manual_path = Path(config.COACH_HISTORY_MANUAL_CSV)
    if manual_path.exists():
        manual_df = pd.read_csv(manual_path, parse_dates=["start_date", "end_date"])
        manual_df = manual_df.dropna(subset=["team", "coach"])
        log.info("Loaded %d manual override/addition rows from %s", len(manual_df), manual_path)
        # Manual rows win: drop any scraped rows for (team, coach) pairs the
        # manual CSV also defines, then concatenate.
        key = manual_df[["team", "coach"]].apply(tuple, axis=1)
        tm_df = tm_df[~tm_df[["team", "coach"]].apply(tuple, axis=1).isin(key)]
        combined = pd.concat([tm_df, manual_df], ignore_index=True)
    else:
        log.warning("%s not found -- run once to auto-create a template.", manual_path)
        manual_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["team", "coach", "start_date", "end_date", "note"]).to_csv(
            manual_path, index=False
        )
        combined = tm_df

    combined = combined.sort_values(["team", "start_date"]).reset_index(drop=True)

    out_path = Path(config.COACH_HISTORY_RESOLVED_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Refuse to silently regress -- see src/data_guard.py's docstring for
    # the incident that made this necessary (this exact file, once).
    guard_against_shrinkage(
        out_path, existing_csv_row_count(out_path), len(combined),
        context=(f"{len(failures)}/{len(config.CLUB_TRANSFERMARKT_ID)} clubs failed "
                 f"this run: {failures}. Check the ERROR log lines above.")
    )

    combined.to_csv(out_path, index=False)
    log.info("Saved %d resolved coach-tenure rows to %s", len(combined), out_path)

    if failures:
        log.warning(
            "Could not fetch/parse manager history for: %s. Add rows for "
            "these teams to %s by hand.", failures, manual_path,
        )

    return combined


if __name__ == "__main__":
    fetch_coach_history()
