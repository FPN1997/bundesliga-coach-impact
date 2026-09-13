"""
Pull match-level xG and PPDA (pressing intensity) for the Bundesliga from
Understat via soccerdata.

Verified against live data: `read_team_match_stats()` returns one row PER
MATCH (home_team/away_team, home_ppda/away_ppda, etc.), not one row per
team like FBref's schedule. This melts it into one row per team-match
(the same shape FBref's data is in) so build_dataset.py can join the two
on (date, team). Understat's team names also don't match FBref's --
config.TEAM_NAME_MAP renames them to FBref's canonical spelling.

Docs: https://soccerdata.readthedocs.io/en/latest/datasources/Understat.html

Guarded against silently regressing (see src/data_guard.py): unlike
FBref's schedule, Understat only returns matches that have actually been
played, so row count here should only ever grow -- a drop is a strong
signal the fetch broke rather than a false alarm.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import soccerdata as sd

import config
from src.data_guard import existing_parquet_row_count, guard_against_shrinkage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _melt_to_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    shared_cols = ["league", "season", "game_id", "date"]

    home = df[shared_cols + [c for c in df.columns if c.startswith("home_")] +
              ["away_team"]].copy()
    home = home.rename(columns={c: c.removeprefix("home_") for c in home.columns
                                 if c.startswith("home_")})
    home = home.rename(columns={"away_team": "opponent"})
    home["xga"] = df["away_xg"]  # opponent's xG = xG conceded, from this team's perspective
    home["venue"] = "Home"

    away = df[shared_cols + [c for c in df.columns if c.startswith("away_")] +
              ["home_team"]].copy()
    away = away.rename(columns={c: c.removeprefix("away_") for c in away.columns
                                 if c.startswith("away_")})
    away = away.rename(columns={"home_team": "opponent"})
    away["xga"] = df["home_xg"]
    away["venue"] = "Away"

    long_df = pd.concat([home, away], ignore_index=True)
    return long_df


def fetch_understat_matches() -> pd.DataFrame:
    """Return one row per team-match with xG and PPDA, team names FBref-normalized."""
    understat = sd.Understat(leagues=config.LEAGUE, seasons=config.SEASONS)

    log.info("Fetching Understat match stats (xG, PPDA) for %s, seasons %s",
              config.LEAGUE, config.SEASONS)

    wide = understat.read_team_match_stats().reset_index()
    log.info("Understat raw columns: %s", list(wide.columns))

    long_df = _melt_to_team_rows(wide)

    unmapped = sorted(set(long_df["team"]) - set(config.TEAM_NAME_MAP))
    if unmapped:
        log.warning("Understat team names with no entry in config.TEAM_NAME_MAP "
                    "(their rows will fail to join FBref data): %s", unmapped)
    long_df["team"] = long_df["team"].map(config.TEAM_NAME_MAP).fillna(long_df["team"])
    long_df["opponent"] = long_df["opponent"].map(config.TEAM_NAME_MAP).fillna(long_df["opponent"])

    out_path = Path(config.RAW_DIR) / "understat_matches.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    guard_against_shrinkage(out_path, existing_parquet_row_count(out_path), len(long_df))

    long_df.to_parquet(out_path, index=False)
    log.info("Saved %d team-match rows to %s", len(long_df), out_path)
    return long_df


if __name__ == "__main__":
    fetch_understat_matches()
