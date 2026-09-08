"""
Build a (team, coach, start_date, end_date) tenure table by scraping each
club's "Trainerhistorie" (manager history) page on Transfermarkt.

Wikipedia was the original plan here -- English Wikipedia turned out not to
have "List of <club> managers" pages for Bundesliga clubs (verified live:
every guessed URL 404'd), so this scrapes Transfermarkt directly instead.
That turned out to be straightforward: a normal browser User-Agent gets a
200 (no login/JS challenge), and the URL only cares about the numeric club
ID -- config.CLUB_TRANSFERMARKT_ID -- the slug text in the URL is cosmetic.

  https://www.transfermarkt.com/<any-slug>/mitarbeiterhistorie/verein/<id>/plus/1

Verified live against Bayern Munich (id 27) and Heidenheim (id 2036): the
page's <title> is used as a sanity check that the ID actually points at the
expected club, since a wrong ID silently returns SOME club's valid page
rather than an error.

Output is written to data/processed/coach_history.csv. As before,
data/coach_history_manual.csv is consulted too and overrides/augments
whatever this scraper produces -- use it for any club missing an ID below,
or to patch a spell the table got wrong (very short caretaker spells are
the likeliest thing to look odd).
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
    url = f"https://www.transfermarkt.com/verein/mitarbeiterhistorie/verein/{club_id}/plus/1"
    log.info("Fetching manager history for %s (Transfermarkt id %d)", club, club_id)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()

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
        except Exception as exc:  # noqa: BLE001 -- best-effort scraper, log and move on
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
