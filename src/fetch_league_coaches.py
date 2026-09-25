"""
Coaching histories for the other top-5 leagues -- the first step towards
running the coaching-change study on more than the Bundesliga (65 sackings
give an interval of +0.04 to +0.30 points per game; five leagues would
roughly halve its width). Not used by the analysis yet: this only collects
the data, politely, so it's ready when the pipeline becomes multi-league.

- Clubs: every team Understat lists for the league since config.SEASONS[0]
  (full names, e.g. "Atletico Madrid"; the rows are kept in
  data/raw/understat_other_leagues.parquet, since the xG will be needed too).
  Understat isn't Transfermarkt, so this costs none of the request budget.
- Transfermarkt ids: Transfermarkt's own search, then a STRICT check that
  every word of the club's name appears in the history page's title (accents
  ignored). Transfermarkt.de uses German names for some clubs ("AS Rom" for
  Roma), listed in TM_NAMES. An id that doesn't verify is recorded as a
  failure for a hand entry (data/club_transfermarkt_ids_other_leagues.json),
  never guessed. Wikidata was checked as an alternative and rejected -- see
  docs/engineering-notes.md.
- Histories: one page per club, cached, so each club costs at most two
  requests (search + history), once.

Every request goes through src/transfermarkt_client.py (paced, one budget
per run, stop at the first block). Output:
data/processed/coach_history_other_leagues.csv and a status file saying
whether every club has been handled.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

import config
from src.fetch_coach_history import _fetch_club_table, fold
from src.transfermarkt_client import RequestBudgetReached, TransfermarktBlocked
from src.transfermarkt_search import search_club_id

log = logging.getLogger(__name__)

LEAGUES = config.OTHER_LEAGUES

# Understat name -> how transfermarkt.de names the club (German exonyms and
# names too ambiguous to search on their own, e.g. "Inter").
TM_NAMES = {
    "Roma": "AS Rom", "Napoli": "SSC Neapel", "Fiorentina": "AC Florenz", "Genoa": "CFC Genua",
    "Torino": "FC Turin", "Venezia": "FC Venedig", "Inter": "Inter Mailand", "AC Milan": "AC Mailand",
    "Nice": "OGC Nizza", "Athletic Club": "Athletic Bilbao",
}

IDS_PATH = Path("data/club_transfermarkt_ids_other_leagues.json")
OUT_PATH = Path(config.PROCESSED_DIR) / "coach_history_other_leagues.csv"
STATUS_PATH = Path(config.PROCESSED_DIR) / "coach_history_other_leagues_status.json"


def cache_path(league: str, club: str) -> Path:
    return Path(config.RAW_DIR) / "transfermarkt_coaches" / league / f"{club.replace(' ', '_')}.html"


def league_clubs() -> dict[str, list[str]]:
    """Every club per league since config.SEASONS[0], from Understat -- the
    file the weekly pipeline refreshes (src/league_data.py), fetched here
    only if it doesn't exist yet."""
    from src import league_data

    path = league_data.understat_path()
    df = pd.read_parquet(path) if path.exists() else league_data.fetch_understat_other_leagues()
    return {league: sorted(set(g["home_team"]) | set(g["away_team"])) for league, g in df.groupby("league")}


def title_matches(club: str, title: str) -> bool:
    """Every word of the club's (Transfermarkt) name appears in the page title."""
    words = [fold(w) for w in TM_NAMES.get(club, club).replace("-", " ").split() if len(w) >= 3]
    folded = fold(title).replace("-", " ")
    return bool(words) and all(w in folded for w in words)


def _load_ids() -> dict:
    return json.loads(IDS_PATH.read_text()) if IDS_PATH.exists() else {"ids": {}, "failed": {}}


def _save_ids(state: dict) -> None:
    IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDS_PATH.write_text(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _history(league: str, club: str, club_id: int) -> pd.DataFrame:
    """The club's history page (cached), checked strictly against the club's name."""
    table = _fetch_club_table(TM_NAMES.get(club, club), club_id, strict_title_check=False,
                              cache_path=cache_path(league, club), use_cache=True)
    title = _page_title(cache_path(league, club))
    if not title_matches(club, title):
        cache_path(league, club).unlink(missing_ok=True)
        raise ValueError(f"page title {title!r} doesn't match {club!r}")
    return table.assign(team=club, league=league)


def _page_title(path: Path) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    return soup.title.get_text(strip=True) if soup.title else ""


def fetch_league_coaches() -> dict:
    """Resolve and fetch as many clubs as this run's budget allows, then
    write everything collected so far. Returns the status (also saved)."""
    clubs = league_clubs()
    state = _load_ids()
    ids, failed = state.setdefault("ids", {}), state.setdefault("failed", {})
    stopped = None
    try:
        for league, names in clubs.items():
            ids.setdefault(league, {})
            failed.setdefault(league, {})
            for club in names:
                if club in failed[league]:
                    continue
                club_id = ids[league].get(club)
                if club_id is None:
                    club_id = search_club_id(TM_NAMES.get(club, club))
                    if club_id is None:
                        failed[league][club] = "no search result"
                        continue
                try:
                    _history(league, club, club_id)  # fetches and caches once, then verifies
                    ids[league][club] = club_id
                except (TransfermarktBlocked, RequestBudgetReached):
                    raise
                except Exception as exc:  # a wrong or unparseable page: record for a hand entry
                    failed[league][club] = f"id {club_id}: {exc}"
                    ids[league].pop(club, None)
    except RequestBudgetReached as exc:
        stopped = f"request budget used ({exc})"
        log.warning("Other-league coach histories: stopped at this run's request budget -- "
                    "continuing next run.")
    except TransfermarktBlocked as exc:
        stopped = f"blocked ({exc})"
        log.error("Transfermarkt blocked the scrape: %s", exc)
    finally:
        _save_ids(state)

    # everything verified so far, this run or earlier, rebuilt from the page cache (no requests)
    frames = [_history(league, club, club_id) for league, league_ids in ids.items()
              for club, club_id in league_ids.items() if cache_path(league, club).exists()]
    if frames:
        out = pd.concat(frames, ignore_index=True).sort_values(["league", "team", "start_date"])
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_PATH, index=False)

    status = {
        "leagues": {league: {"clubs": len(names), "fetched": len(ids.get(league, {})),
                             "failed": dict(failed.get(league, {}))}
                    for league, names in clubs.items()},
        "stopped": stopped,
    }
    status["complete"] = stopped is None and all(
        v["fetched"] + len(v["failed"]) == v["clubs"] for v in status["leagues"].values())
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    for league, v in status["leagues"].items():
        log.info("  %-20s %3d of %3d clubs fetched, %d need a hand entry", league, v["fetched"], v["clubs"],
                 len(v["failed"]))
    return status
