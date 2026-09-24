"""
Pull betting odds for every Bundesliga match from football-data.co.uk (via
soccerdata.MatchHistory), with team names normalized to FBref's spelling.

Used by the market benchmark (forecasts vs. the betting market) and by the
fixture-difficulty rating the coaching-change analysis adjusts for
(src/fixture_difficulty.py).

Kept columns: market-average odds collected the Friday/Tuesday before a
match (AvgH/D/A) and at kickoff (AvgCH/D/A), plus Pinnacle's closing odds
(PSCH/D/A), which football-data.co.uk stopped carrying in January 2026 and
are only used as a sensitivity check.

football-data.co.uk changed its format in 2019-20: earlier seasons carry the
market average as Betbrain's BbAvH/D/A and have no closing odds at all. The
older columns are folded into AvgH/D/A, so "pre-match market average" is
complete from 2014-15 on and closing odds from 2019-20 on.

Guarded like every other fetcher (src/data_guard.py): refuses to overwrite
a saved file with a drastically smaller one, and refuses outright if any
team name has no FBref counterpart (add it to config.FOOTBALL_DATA_NAME_MAP).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config
from src.data_guard import existing_parquet_row_count, guard_against_shrinkage

log = logging.getLogger(__name__)

ODDS_COLUMNS = ["AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA", "PSCH", "PSCD", "PSCA"]


def odds_path() -> Path:
    return Path(config.RAW_DIR) / "football_data_odds.parquet"


def load_odds() -> pd.DataFrame | None:
    path = odds_path()
    return pd.read_parquet(path) if path.exists() else None


def fetch_odds() -> pd.DataFrame:
    """One row per match with pre-match/closing odds, team names FBref-normalized."""
    import soccerdata as sd

    raw = sd.MatchHistory(leagues=config.LEAGUE, seasons=config.SEASONS).read_games().reset_index()
    for new, old in [("AvgH", "BbAvH"), ("AvgD", "BbAvD"), ("AvgA", "BbAvA")]:
        if old in raw.columns:
            raw[new] = raw[new].fillna(raw[old]) if new in raw.columns else raw[old]
    cols = ["season", "date", "home_team", "away_team", "FTR"]
    odds = raw[cols].copy()
    for c in ODDS_COLUMNS:  # a column absent from every season stays all-NaN, not missing
        odds[c] = raw[c] if c in raw.columns else float("nan")

    fbref_teams = set(pd.read_parquet(Path(config.RAW_DIR) / "fbref_schedule.parquet",
                                      columns=["team"])["team"])
    for side in ("home_team", "away_team"):
        odds[side] = odds[side].replace(config.FOOTBALL_DATA_NAME_MAP)
    unknown = sorted((set(odds["home_team"]) | set(odds["away_team"])) - fbref_teams)
    if unknown:
        raise RuntimeError(f"football-data team names with no FBref match: {unknown} -- add them "
                           f"to config.FOOTBALL_DATA_NAME_MAP")

    odds["season"] = odds["season"].astype(str)
    path = odds_path()
    guard_against_shrinkage(path, existing_parquet_row_count(path), len(odds))
    path.parent.mkdir(parents=True, exist_ok=True)
    odds.to_parquet(path, index=False)
    log.info("Saved %d matches of odds -> %s", len(odds), path)
    return odds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_odds()
