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

FBref sits behind Cloudflare. **A challenge is FBref saying no, and the
fetch stops at the first one** -- the same rule as for Transfermarkt
(src/transfermarkt_client.py). That needs PoliteFBref below, because
soccerdata's own FBref reader does the opposite by default (found in the
28 Sep 2026 weekly refresh log, see docs/engineering-notes.md):

- it drives Chrome in "undetected" mode (SeleniumBase uc=True), built to
  keep a site from recognising automation;
- its default is a VISIBLE browser (headless=False, although its docstring
  says True), and in that mode it answers a CAPTCHA by clicking through the
  Cloudflare challenge itself (uc_gui_handle_captcha / uc_gui_handle_cf);
- it reloads a challenged page and restarts the browser, 5 times per page.

PoliteFBref runs plain, headless Chrome and turns the first challenge into
FBrefBlocked; the weekly refresh then keeps last week's FBref data. (An
earlier version of this file also restarted the browser every 12 pages to
reset Cloudflare's per-session throttling -- also a way around a limit, and
removed for the same reason.)

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
from pathlib import Path

import pandas as pd
import soccerdata as sd

import config
from src.data_guard import existing_parquet_row_count, guard_against_shrinkage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LEAGUE_ROUND_RE = re.compile(config.FBREF_LEAGUE_ROUND_PATTERN)
GAME_ID_RE = re.compile(r"/en/matches/([0-9a-f]+)/")


class FBrefBlocked(RuntimeError):
    """FBref's Cloudflare protection challenged us. Stop and keep the previous
    data -- never solve, click through or retry around a challenge."""


class _Challenged(BaseException):
    """Raised from inside soccerdata's download loop. That loop catches
    Exception and retries (reloading, restarting the browser); a
    BaseException is the one thing that gets straight out of it."""


class PoliteFBref(sd.FBref):
    """soccerdata's FBref reader with every way around bot protection removed:
    headless (so the GUI CAPTCHA clicker can never run), plain Chrome instead
    of undetected mode, and the CAPTCHA "solver" replaced by a hard stop."""

    def __init__(self, **kwargs):
        kwargs["headless"] = True
        super().__init__(**kwargs)

    @classmethod
    def _all_leagues(cls) -> dict[str, str]:
        # soccerdata looks up supported leagues by class name; keep FBref's
        return sd.FBref._all_leagues()

    def _init_webdriver(self):
        import seleniumbase as sb
        if hasattr(self, "_driver"):
            self._driver.quit()
        return sb.Driver(uc=False, headless=True, binary_location=self.path_to_browser)

    def solve_captcha(self) -> None:
        raise _Challenged("FBref showed a CAPTCHA / Cloudflare challenge")


def _teams_for_season(fbref: sd.FBref) -> list[str]:
    teams = fbref.read_team_season_stats(stat_type="standard").reset_index()
    return sorted(teams["team"].dropna().unique().tolist())


def fetch_fbref_matches() -> pd.DataFrame:
    """Return one row per team-match (league matches only) with formations."""
    try:
        return _fetch_fbref_matches()
    except _Challenged as exc:
        raise FBrefBlocked(f"{exc} -- stopped at the first challenge, nothing solved or retried. "
                           "Previously saved FBref data is untouched.") from None


def _fetch_fbref_matches() -> pd.DataFrame:
    fbref = PoliteFBref(leagues=config.LEAGUE, seasons=config.SEASONS)

    teams = _teams_for_season(fbref)
    log.info("Fetching FBref team schedules for %d teams across seasons %s",
              len(teams), config.SEASONS)

    frames = []
    for team in teams:
        try:
            df = fbref.read_team_match_stats(stat_type="schedule", team=team)
        except Exception as exc:  # one bad team shouldn't kill the run (a challenge still stops it)
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
