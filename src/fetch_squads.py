"""
Squad data from Transfermarkt: who was in each club's squad each season, how
valuable they were, which of them arrived in the winter transfer window,
and every player's injury history.

Used by src/squad_availability.py to measure, around every coaching change,
how much of a squad was out injured and how much arrived as new signings --
the two biggest things besides the coach that change "at the same moment."

Pages (transfermarkt.de, same as fetch_coach_history.py; the URL slug is
cosmetic, only the numeric ids matter):
  - squad:     /verein/kader/verein/<club>/saison_id/<year>/plus/1
               player ids, positions, market value that season
  - winter:    /verein/transfers/verein/<club>/saison_id/<year>/pos//detailpos/0/w_s/w
               the January window's arrivals
  - injuries:  /spieler/verletzungen/spieler/<player>[/page/N]
               injury, from, until, games missed

Scope: injury histories only for each club-season's INJURY_TOP_N most
valuable players -- a matchday squad plus rotation, nearly all of a squad's
market value -- which keeps it to roughly 2,000 players instead of every
youth-team registration. A value-weighted injury measure barely moves from
the players left out.

Transfermarkt has blocked scraping before (transfermarkt.com's firewall,
see docs/engineering-notes.md), so this is deliberately polite and
resumable: every page is cached under data/raw/transfermarkt_squads/, live
requests are spaced REQUEST_DELAY seconds apart, and an empty or 403/429
response stops the run with a clear error instead of retrying. Re-running
picks up where it stopped. Past seasons are cached for good; the current
season's pages are re-fetched once they're older than CURRENT_SEASON_MAX_AGE.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

import config
from src import transfermarkt_client as tm
from src.data_guard import existing_parquet_row_count, guard_against_shrinkage
from src.fetch_coach_history import _load_auto_resolved_ids
from src.transfermarkt_client import RequestBudgetReached, TransfermarktBlocked

log = logging.getLogger(__name__)

INJURY_TOP_N = 22
# Pace and per-run budget. The first full run (2 s apart, no budget) was
# blocked by Transfermarkt's firewall after ~1,100 requests in about an hour;
# slower requests and a budget per run spread the rest over several runs.
# pacing and the per-run request budget live in src/transfermarkt_client.py,
# shared with every other Transfermarkt fetch in the same run
CURRENT_SEASON_MAX_AGE = 6     # days before the current season's pages are re-fetched
BASE = "https://www.transfermarkt.de"


def cache_dir() -> Path:
    return Path(config.RAW_DIR) / "transfermarkt_squads"


def squads_path() -> Path:
    return Path(config.RAW_DIR) / "tm_squads.parquet"


def injuries_path() -> Path:
    return Path(config.RAW_DIR) / "tm_injuries.parquet"


# --- fetching ---------------------------------------------------------------

class _Fetcher:
    def __init__(self, current_season_start: int):
        self.current_season_start = current_season_start
        self.live_requests = 0  # this fetcher's share of the run's budget, for the log

    def get(self, url: str, cache: Path, season_start: int | None) -> str:
        """Cached GET. Pages for past seasons never expire; the current
        season's expire after CURRENT_SEASON_MAX_AGE days."""
        if cache.exists():
            fresh = season_start is None or season_start < self.current_season_start or (
                time.time() - cache.stat().st_mtime < CURRENT_SEASON_MAX_AGE * 86400)
            if fresh:
                return cache.read_text(encoding="utf-8")
        resp = tm.get(url)  # paced, budgeted; raises on a block or when the run's budget is used
        self.live_requests += 1
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text, encoding="utf-8")
        return resp.text


# --- parsing ----------------------------------------------------------------

def parse_market_value(text: str) -> float:
    """'35,00 Mio. €' -> 35e6, '150 Tsd. €' -> 150e3, '-' -> NaN."""
    m = re.search(r"([\d.,]+)\s*(Mrd|Mio|Tsd)", text)
    if not m:
        return float("nan")
    number = float(m.group(1).replace(".", "").replace(",", "."))
    return number * {"Mrd": 1e9, "Mio": 1e6, "Tsd": 1e3}[m.group(2)]


def _player_id(href: str) -> int | None:
    m = re.search(r"/spieler/(\d+)", href or "")
    return int(m.group(1)) if m else None


def parse_squad(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("table.items > tbody > tr"):
        a = tr.select_one("td.hauptlink a[href*='/profil/spieler/']")
        if not a:
            continue
        inner = tr.select("table.inline-table tr")
        cells = tr.find_all("td", recursive=False)
        rows.append({
            "player_id": _player_id(a["href"]),
            "player": a.get_text(strip=True),
            "position": inner[1].get_text(" ", strip=True) if len(inner) > 1 else "",
            "market_value": (parse_market_value(cells[-1].get_text(" ", strip=True))
                             if cells else float("nan")),
        })
    return pd.DataFrame(rows, columns=["player_id", "player", "position", "market_value"])


def parse_arrivals(html: str) -> set[int]:
    """Player ids in the 'Zugänge' (arrivals) box of a transfers page."""
    soup = BeautifulSoup(html, "html.parser")
    for box in soup.select("div.box"):
        h2 = box.select_one("h2")
        if h2 and "Zugänge" in h2.get_text():
            return {pid for a in box.select("table.items a[href*='/profil/spieler/']")
                    if (pid := _player_id(a["href"])) is not None}
    return set()


def _parse_date(text: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(datetime.strptime(text.strip(), "%d.%m.%Y"))
    except ValueError:
        return pd.NaT


def parse_injuries(html: str) -> tuple[pd.DataFrame, int]:
    """(injury rows, number of pages) for one page of a player's injury history."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    table = soup.select_one("table.items")
    for tr in (table.select("tbody > tr") if table else []):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
        if len(cells) < 6:
            continue
        missed = re.sub(r"\D", "", cells[5])
        rows.append({"injury": cells[1], "start": _parse_date(cells[2]), "end": _parse_date(cells[3]),
                     "games_missed": int(missed) if missed else None})
    pages = [int(m.group(1)) for a in soup.select("div.pager a")
             if (m := re.search(r"/page/(\d+)", a.get("href", "")))]
    return pd.DataFrame(rows, columns=["injury", "start", "end", "games_missed"]), max(pages, default=1)


# --- main -------------------------------------------------------------------

def _club_ids() -> dict[str, int]:
    return {**_load_auto_resolved_ids(), **config.CLUB_TRANSFERMARKT_ID}


def _team_seasons() -> pd.DataFrame:
    """Every (team, season) in the match data, with the season's start year."""
    m = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet", columns=["team", "season"])
    ts = m.drop_duplicates().copy()
    ts["season"] = ts["season"].astype(str)
    ts["start_year"] = 2000 + ts["season"].str[:2].astype(int)
    return ts.sort_values(["start_year", "team"]).reset_index(drop=True)


def fetch_squads() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fetch (or read from cache) everything, then write tm_squads/tm_injuries.
    Returns None -- writing nothing -- if the run stopped early: this run's
    request budget ran out, or Transfermarkt blocked. Everything fetched so far
    stays cached, so the next run continues where this one stopped."""
    try:
        return _fetch_squads()
    except RequestBudgetReached as exc:
        log.warning("Stopped after this run's request budget (%s). Progress is cached -- run "
                    "`bundesliga squad` again later to continue.", exc)
    except TransfermarktBlocked as exc:
        log.error("Transfermarkt blocked the scrape: %s Progress is cached -- wait (hours to a "
                  "day) and run `bundesliga squad` again.", exc)
    return None


def _fetch_squads() -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = _club_ids()
    team_seasons = _team_seasons()
    missing = sorted(set(team_seasons["team"]) - set(ids))
    if missing:
        raise RuntimeError(f"No Transfermarkt id for {missing} -- run the pipeline first "
                           f"(it auto-resolves new clubs) or add them to config.CLUB_TRANSFERMARKT_ID")
    fetcher = _Fetcher(current_season_start=int(team_seasons["start_year"].max()))
    root = cache_dir()

    squads = []
    for r in team_seasons.itertuples():
        club = ids[r.team]
        squad = parse_squad(fetcher.get(
            f"{BASE}/verein/kader/verein/{club}/saison_id/{r.start_year}/plus/1",
            root / "squad" / f"{club}_{r.start_year}.html", r.start_year))
        arrivals = parse_arrivals(fetcher.get(
            f"{BASE}/verein/transfers/verein/{club}/saison_id/{r.start_year}/pos//detailpos/0/w_s/w",
            root / "winter" / f"{club}_{r.start_year}.html", r.start_year))
        if squad.empty:
            log.warning("Empty Transfermarkt squad for %s %s", r.team, r.season)
        squad = squad.assign(team=r.team, season=r.season, winter_arrival=squad["player_id"].isin(arrivals))
        squad["injury_tracked"] = squad["market_value"].rank(ascending=False, method="first") <= INJURY_TOP_N
        squads.append(squad)
    squads = pd.concat(squads, ignore_index=True)
    log.info("Squads: %d player-seasons across %d club-seasons (%d live requests)",
             len(squads), len(team_seasons), fetcher.live_requests)

    # Injury histories: one player's pages cover their whole career, so each
    # player is fetched once; current-season players get refreshed.
    tracked = squads[squads["injury_tracked"]]
    latest_season = tracked.groupby("player_id")["season"].max()
    current_code = team_seasons.loc[team_seasons["start_year"].idxmax(), "season"]
    # Injuries are listed newest first, so paging can stop once a page
    # reaches back before the first season in the data -- most of the saving
    # comes from long careers.
    data_start = pd.Timestamp(f"{int(team_seasons['start_year'].min())}-07-01")
    injuries = []
    for n, (pid, season) in enumerate(latest_season.items(), 1):
        season_start = fetcher.current_season_start if season == current_code else None
        page, pages = 1, 1
        while page <= pages:
            suffix = f"/page/{page}" if page > 1 else ""
            html = fetcher.get(f"{BASE}/spieler/verletzungen/spieler/{pid}{suffix}",
                               root / "injuries" / f"{pid}_{page}.html", season_start)
            rows, pages = parse_injuries(html)
            injuries.append(rows.assign(player_id=pid))
            if rows["start"].notna().any() and rows["start"].min() < data_start:
                break  # (a player with no injuries has an empty table -- keep that case safe)
            page += 1
        if n % 200 == 0:
            log.info("Injury histories: %d / %d players", n, len(latest_season))
    injuries = pd.concat(injuries, ignore_index=True)
    log.info("Injuries: %d records for %d players (%d live requests in total)",
             len(injuries), len(latest_season), fetcher.live_requests)

    for df, path in ((squads, squads_path()), (injuries, injuries_path())):
        guard_against_shrinkage(path, existing_parquet_row_count(path), len(df))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return squads, injuries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_squads()
