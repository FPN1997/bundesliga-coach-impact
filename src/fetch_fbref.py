"""
Pull match results + formations for the Bundesliga from FBref via soccerdata.

Two things verified against live data that shape this file (see README
"Known rough edges"):

1. `read_team_match_stats(stat_type="schedule")` called WITHOUT a `team=`
   argument comes back with `team` (and `league`) entirely NaN -- a
   soccerdata quirk, not a data-availability issue. Passing `team=` per
   club fixes it, so we fetch team-by-team and concatenate.
2. Each team's schedule mixes in DFB-Pokal, DFL-Supercup, and European
   fixtures alongside league matches. League rows are identifiable by their
   `round` value ("Matchweek 1", "Matchweek 2", ...) -- everything else
   (`round` like "Round of 64", "Group stage") gets filtered out.

FBref sits behind Cloudflare, which starts challenging a browser session
after roughly 20 page loads -- each challenged page then costs a ~5-minute
timeout before soccerdata's retry gets through. A fresh browser session is
fast again, so the fetch restarts the browser every PAGES_PER_BROWSER_SESSION
downloaded pages. (Found during the 2014-15 backfill: ~10 s per page for the
first ~20 pages, then ~5 min per page; a restart brought back ~10 s.) The
weekly refresh only downloads the current season (~1 page per club), so it
rarely restarts; a multi-season backfill restarts roughly once per club.

Guarded against silently regressing (see src/data_guard.py) two ways: a
minimum fraction of requested teams must actually fetch successfully
(row count alone is a weak signal here, since FBref lists a whole season's
fixtures whether played or not), and the final row count can't drop
drastically from what's already saved.

Docs: https://soccerdata.readthedocs.io/en/latest/datasources/FBref.html
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd
import soccerdata as sd

import config
from src.data_guard import existing_parquet_row_count, guard_against_shrinkage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LEAGUE_ROUND_RE = re.compile(config.FBREF_LEAGUE_ROUND_PATTERN)
GAME_ID_RE = re.compile(r"/en/matches/([0-9a-f]+)/")
PAGES_PER_BROWSER_SESSION = 12  # comfortably under Cloudflare's ~20-page threshold


def _pages_written_since(fbref: sd.FBref, since: float) -> int:
    """Match-log pages downloaded (created or re-downloaded) since `since`."""
    return sum(1 for f in Path(fbref.data_dir).glob("matchlogs_*") if f.stat().st_mtime >= since)


def _teams_for_season(fbref: sd.FBref) -> list[str]:
    teams = fbref.read_team_season_stats(stat_type="standard").reset_index()
    return sorted(teams["team"].dropna().unique().tolist())


def fetch_fbref_matches() -> pd.DataFrame:
    """Return one row per team-match (league matches only) with formations."""
    fbref = sd.FBref(leagues=config.LEAGUE, seasons=config.SEASONS)

    teams = _teams_for_season(fbref)
    log.info("Fetching FBref team schedules for %d teams across seasons %s",
              len(teams), config.SEASONS)

    frames = []
    session_start = time.time()
    for team in teams:
        if _pages_written_since(fbref, session_start) >= PAGES_PER_BROWSER_SESSION:
            log.info("Restarting the browser (Cloudflare throttles long FBref sessions)")
            fbref._init_webdriver()  # quits the old browser, starts a fresh one
            session_start = time.time()
        try:
            df = fbref.read_team_match_stats(stat_type="schedule", team=team)
        except Exception as exc:  # one bad team shouldn't kill the run
            log.error("Failed to fetch schedule for %s: %s", team, exc)
            continue
        df = df.reset_index()
        league_rows = df[df["round"].astype(str).str.match(LEAGUE_ROUND_RE)]
        if league_rows.empty:
            log.warning("No 'Matchweek N' rows found for %s -- check "
                        "FBREF_LEAGUE_ROUND_PATTERN in config.py against "
                        "this team's actual 'round' values: %s",
                        team, df["round"].unique().tolist())
        league_rows = league_rows.copy()
        league_rows["team"] = team  # the raw column is unreliable; set it ourselves
        frames.append(league_rows)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("FBref: %d league-match rows across %d teams", len(df), len(teams))

    # Row count alone is a weak signal here -- FBref's per-team schedule
    # lists the whole season's fixtures whether played or not, so total
    # rows barely moves week to week regardless of how many teams' fetches
    # actually succeeded. Missing TEAMS is the more sensitive failure mode
    # for this particular source, so guard on that directly too.
    fetched_teams = len(frames)
    if teams and fetched_teams < len(teams) * 0.8:
        raise RuntimeError(
            f"Only {fetched_teams}/{len(teams)} teams fetched successfully -- "
            f"check the ERROR log lines above (per-team fetch failures) before "
            f"trusting this run's output."
        )

    df["fbref_game_id"] = df["match_report"].astype(str).str.extract(GAME_ID_RE)

    # GF/GA occasionally render as e.g. "2 (1)" (shoot-out) in FBref's raw
    # HTML even on rows we've otherwise filtered to normal-time league
    # matches; strip anything after the goal count so the column is a clean
    # numeric dtype (mixed str/int columns break parquet's Arrow conversion).
    for col in ("GF", "GA"):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
        )

    out_path = Path(config.RAW_DIR) / "fbref_schedule.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Secondary guard alongside the team-completeness check above: catches
    # e.g. a season silently coming back empty even with every team
    # nominally "succeeding". See src/data_guard.py's docstring.
    guard_against_shrinkage(out_path, existing_parquet_row_count(out_path), len(df))

    df.to_parquet(out_path, index=False)
    log.info("Saved %d rows to %s", len(df), out_path)
    return df


if __name__ == "__main__":
    fetch_fbref_matches()
