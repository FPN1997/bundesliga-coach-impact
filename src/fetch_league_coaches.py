"""
Coaching histories for the other top-5 leagues, for the multi-league
coaching-change study (src/league_data.py). Run by `bundesliga backfill`.

How a club gets its Transfermarkt id -- from Transfermarkt's LEAGUE pages,
not its search:

- For every league-season, Transfermarkt's competition page lists the ~20
  clubs with their ids (TM_COMPETITIONS; 52 pages, cached, past seasons are
  never fetched again).
- Each Understat club (full names, e.g. "Atletico Madrid") is matched to a
  club from the same league: the Transfermarkt club must have been in the
  league in every season the Understat club was, and every word of its name
  (TM_NAMES for German exonyms, "AS Rom" for Roma) must appear in the
  Transfermarkt name; ties go to the club whose seasons match exactly, and
  a club whose name doesn't match any candidate is still accepted when
  exactly one candidate has exactly its seasons. Anything else is recorded
  for a hand entry -- never guessed.
- The club's history page is then verified: its canonical link must be the
  staff history of exactly that id (a redirect to the homepage, or any other
  page, fails). Not the page title: clubs get renamed ("US Palermo" on the
  2014 league page is "Palermo FC" today). Pages are cached per club AND
  id, so a page saved under a wrong id can never be reused -- that happened
  once: pages the search-based run saved under wrong ids (Transfermarkt's
  homepage, for Chelsea) were reused for the right ids by club name.

Why not the search: on 29 Sep 2026 the first backfill run that got through
used Transfermarkt's club search, which lists matching COACHES first, each
linking their club -- so "Arsenal" returned Atletico Mancha Real. A strict
name check rejected all of those, but a check on names alone can't tell
"FC Arsenal" from "Arsenal Tula"; a league page can. Wikidata was also
checked and rejected (docs/engineering-notes.md).

Every request goes through src/transfermarkt_client.py (paced, one budget
per run, stop at the first block). Output:
data/processed/coach_history_other_leagues.csv and a status file.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

import config
from src import transfermarkt_client as tm
from src.fetch_coach_history import _fetch_club_table, fold
from src.transfermarkt_client import RequestBudgetReached, TransfermarktBlocked

log = logging.getLogger(__name__)

LEAGUES = config.OTHER_LEAGUES
TM_COMPETITIONS = {"ENG-Premier League": "GB1", "ESP-La Liga": "ES1", "ITA-Serie A": "IT1",
                   "FRA-Ligue 1": "FR1"}

# Understat name -> words transfermarkt.de uses instead (German exonyms, and
# names Understat shortens differently)
TM_NAMES = {
    "Roma": "AS Rom", "Napoli": "SSC Neapel", "Fiorentina": "AC Florenz", "Genoa": "CFC Genua",
    "Torino": "FC Turin", "Inter": "Inter Mailand", "AC Milan": "AC Mailand",
    "Nice": "OGC Nizza", "Athletic Club": "Athletic Bilbao", "Strasbourg": "Racing Straßburg",
}

IDS_PATH = Path("data/club_transfermarkt_ids_other_leagues.json")
OUT_PATH = Path(config.PROCESSED_DIR) / "coach_history_other_leagues.csv"
STATUS_PATH = Path(config.PROCESSED_DIR) / "coach_history_other_leagues_status.json"
CLUB_LINK_RE = re.compile(r"/([a-z0-9-]+)/startseite/verein/(\d+)")


def cache_path(league: str, club: str, club_id: int) -> Path:
    return (Path(config.RAW_DIR) / "transfermarkt_coaches" / league
            / f"{club.replace(' ', '_')}_{club_id}.html")


def league_page_path(league: str, year: int) -> Path:
    return Path(config.RAW_DIR) / "transfermarkt_leagues" / f"{TM_COMPETITIONS[league]}_{year}.html"


def understat_seasons() -> dict[str, dict[str, set[int]]]:
    """league -> Understat club -> the season start years it played in the league."""
    from src import league_data

    path = league_data.understat_path()
    df = pd.read_parquet(path) if path.exists() else league_data.fetch_understat_other_leagues()
    df = df.assign(year=2000 + df["season"].astype(str).str[:2].astype(int))
    out: dict[str, dict[str, set[int]]] = {}
    for (league, year), g in df.groupby(["league", "year"]):
        for club in set(g["home_team"]) | set(g["away_team"]):
            out.setdefault(league, {}).setdefault(club, set()).add(int(year))
    return out


def league_clubs() -> dict[str, list[str]]:
    return {league: sorted(clubs) for league, clubs in understat_seasons().items()}


def parse_league_page(html: str) -> dict[int, str]:
    """{id: Transfermarkt name} for every club on a competition page."""
    from bs4 import BeautifulSoup
    out: dict[int, str] = {}
    for a in BeautifulSoup(html, "html.parser").select("table.items td.hauptlink a"):
        match = CLUB_LINK_RE.search(a.get("href", ""))
        if match and a.get_text(strip=True):
            out.setdefault(int(match.group(2)), a.get_text(strip=True))
    return out


def tm_league_seasons(league: str, years: list[int]) -> tuple[dict[int, str], dict[int, set[int]]]:
    """(id -> name, id -> season start years) from the league's competition pages (cached)."""
    names: dict[int, str] = {}
    seasons: dict[int, set[int]] = {}
    for year in years:
        path = league_page_path(league, year)
        if not path.exists():
            resp = tm.get(f"https://www.transfermarkt.de/wettbewerb/startseite/wettbewerb/"
                          f"{TM_COMPETITIONS[league]}/plus/?saison_id={year}")
            resp.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(resp.text, encoding="utf-8")
        clubs = parse_league_page(path.read_text(encoding="utf-8"))
        if not 16 <= len(clubs) <= 22:
            path.unlink(missing_ok=True)
            raise ValueError(f"{path.name}: {len(clubs)} clubs listed, expected a league's worth")
        for club_id, name in clubs.items():
            names[club_id] = name
            seasons.setdefault(club_id, set()).add(year)
    return names, seasons


def _words(name: str) -> list[str]:
    return [fold(w) for w in name.replace("-", " ").split() if len(w) >= 3]


def match_clubs(us: dict[str, set[int]], tm_names: dict[int, str],
                tm_seasons: dict[int, set[int]]) -> tuple[dict[str, int], dict[str, str]]:
    """Understat club -> Transfermarkt id within one league (see module docstring)."""
    matched: dict[str, int] = {}
    failed: dict[str, str] = {}
    for club, years in sorted(us.items()):
        candidates = [i for i, ys in tm_seasons.items() if years <= ys]
        words = _words(TM_NAMES.get(club, club))
        named = [i for i in candidates
                 if words and all(w in fold(tm_names[i]).replace("-", " ") for w in words)]
        exact = [i for i in candidates if tm_seasons[i] == years]
        named_exact = [i for i in named if i in exact]
        # "AC Mailand" also matches "Inter Mailand" once the short "AC" is
        # dropped: a candidate named exactly like the club wins such a tie
        spelled = [i for i in named if fold(tm_names[i]) == fold(TM_NAMES.get(club, club))]
        if len(named) == 1:
            matched[club] = named[0]
        elif len(spelled) == 1:
            matched[club] = spelled[0]
        elif len(named) > 1 and len(named_exact) == 1:
            matched[club] = named_exact[0]
        elif not named and len(exact) == 1:
            matched[club] = exact[0]
            log.info("  %s -> %s (id %d) by its exact run of seasons; the names differ", club,
                     tm_names[exact[0]], exact[0])
        else:
            failed[club] = (f"ambiguous: {[tm_names[i] for i in named]}" if named else
                            f"no Transfermarkt club in every one of its seasons {sorted(years)} "
                            "with a matching name")
    # one Transfermarkt club can't be two Understat clubs -- unless Understat
    # renamed it (Parma, "Parma Calcio 1913" after 2015), when the seasons don't overlap
    by_id: dict[int, list[str]] = {}
    for club, i in matched.items():
        by_id.setdefault(i, []).append(club)
    for i, clubs in by_id.items():
        if any(us[a] & us[b] for a in clubs for b in clubs if a < b):
            for club in clubs:
                failed[club] = f"shares Transfermarkt id {i} ({tm_names[i]}) with {clubs}"
                matched.pop(club)
    return matched, failed


def _load_state() -> dict:
    return json.loads(IDS_PATH.read_text()) if IDS_PATH.exists() else {}


def _save_state(state: dict) -> None:
    IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDS_PATH.write_text(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def page_is_staff_history_of(html: str, club_id: int) -> bool:
    """Is this Transfermarkt page the staff history of exactly `club_id`?"""
    from bs4 import BeautifulSoup
    canonical = BeautifulSoup(html, "html.parser").find("link", rel="canonical")
    href = canonical.get("href", "") if canonical else ""
    return re.search(rf"/mitarbeiterhistorie/verein/{club_id}(/|$)", href) is not None


def _history(league: str, club: str, club_id: int, tm_name: str) -> pd.DataFrame:
    """The club's history page (cached per club and id), verified to be that id's staff history."""
    path = cache_path(league, club, club_id)
    try:
        table = _fetch_club_table(tm_name, club_id, cache_path=path, use_cache=True)
        if not page_is_staff_history_of(path.read_text(encoding="utf-8"), club_id):
            raise ValueError(f"the page fetched for id {club_id} isn't its staff history")
    except (TransfermarktBlocked, RequestBudgetReached):
        raise
    except Exception:
        path.unlink(missing_ok=True)  # never reuse a page that failed
        raise
    return table.assign(team=club, league=league)


def fetch_league_coaches() -> dict:
    """Resolve and fetch as many clubs as this run's budget allows, then write
    everything collected so far. Returns the status (also saved)."""
    us = understat_seasons()
    previous_ids = _load_state().get("ids", {})  # e.g. the search-based run's, cross-checked below
    state: dict = {"ids": {}, "failed": {}, "tm_names": {}}
    stopped = None
    try:
        for league, clubs in us.items():
            years = sorted(set().union(*clubs.values()))
            tm_names, tm_seasons = tm_league_seasons(league, years)
            matched, failed = match_clubs(clubs, tm_names, tm_seasons)
            for club, club_id in matched.items():
                old = previous_ids.get(league, {}).get(club)
                if old is not None and old != club_id:
                    log.warning("  %s: the league pages say id %d (%s), an earlier run had %d -- "
                                "using %d", club, club_id, tm_names[club_id], old, club_id)
            state["ids"][league], state["failed"][league] = matched, failed
            state["tm_names"][league] = {c: tm_names[i] for c, i in matched.items()}
        for league, matched in state["ids"].items():
            for club, club_id in list(matched.items()):
                try:
                    _history(league, club, club_id, state["tm_names"][league][club])
                except (TransfermarktBlocked, RequestBudgetReached):
                    raise
                except Exception as exc:  # a wrong or unparseable page: record for a hand entry
                    state["failed"][league][club] = f"id {club_id}: {exc}"
                    matched.pop(club)
    except RequestBudgetReached as exc:
        stopped = f"request budget used ({exc})"
        log.warning("Other-league coach histories: stopped at this run's request budget -- "
                    "continuing next run.")
    except TransfermarktBlocked as exc:
        stopped = f"blocked ({exc})"
        log.error("Transfermarkt blocked the scrape: %s", exc)
    finally:
        if state["ids"]:
            _save_state(state)

    frames = [_history(league, club, club_id, state["tm_names"][league][club])
              for league, ids in state["ids"].items() for club, club_id in ids.items()
              if cache_path(league, club, club_id).exists()]
    if frames:
        out = pd.concat(frames, ignore_index=True).sort_values(["league", "team", "start_date"])
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_PATH, index=False)

    fetched = {league: sum(cache_path(league, c, i).exists() for c, i in state["ids"].get(league, {}).items())
               for league in us}
    status = {
        "leagues": {league: {"clubs": len(clubs), "fetched": fetched[league],
                             "failed": dict(state["failed"].get(league, {}))}
                    for league, clubs in us.items()},
        "stopped": stopped,
    }
    status["complete"] = stopped is None and all(
        v["fetched"] + len(v["failed"]) == v["clubs"] for v in status["leagues"].values())
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    for league, v in status["leagues"].items():
        log.info("  %-20s %3d of %3d clubs fetched, %d need a hand entry", league, v["fetched"],
                 v["clubs"], len(v["failed"]))
    return status
