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

A name with no entry in config.TEAM_NAME_MAP (or the auto-resolved cache,
data/team_name_map_auto.json) gets fuzzy-matched against FBref's current
team list (see src/team_name_matcher.py) instead of being left to silently
fail the (date, team) join in build_dataset.py. A confident match gets
cached for future runs; anything below the matcher's confidence threshold
still falls through to the same "log a warning, leave it unmapped" path as
before this existed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import soccerdata as sd

import config
from src.data_guard import existing_parquet_row_count, guard_against_shrinkage
from src.team_name_matcher import best_match

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

AUTO_RESOLVED_PATH = Path("data/team_name_map_auto.json")


def _load_auto_resolved_map() -> dict[str, str]:
    if not AUTO_RESOLVED_PATH.exists():
        return {}
    return json.loads(AUTO_RESOLVED_PATH.read_text())


def _save_auto_resolved_map(mapping: dict[str, str]) -> None:
    AUTO_RESOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTO_RESOLVED_PATH.write_text(json.dumps(dict(sorted(mapping.items())), indent=2) + "\n")


def _fbref_team_names() -> list[str]:
    """The current, authoritative FBref team list to fuzzy-match against.
    fetch_fbref.py always runs before this step in run_pipeline.py, so its
    raw output should already exist; falls back to config.TEAM_NAME_MAP's
    own values (a strict subset -- only teams someone has already resolved)
    if run standalone before ever fetching FBref."""
    fbref_path = Path(config.RAW_DIR) / "fbref_schedule.parquet"
    if fbref_path.exists():
        return sorted(pd.read_parquet(fbref_path, columns=["team"])["team"].dropna().unique())
    log.warning("%s not found -- falling back to config.TEAM_NAME_MAP's own values as the "
                "fuzzy-match candidate pool (run fetch_fbref.py first for the current, "
                "complete list).", fbref_path)
    return sorted(set(config.TEAM_NAME_MAP.values()))


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

    auto_resolved = _load_auto_resolved_map()
    name_map = {**auto_resolved, **config.TEAM_NAME_MAP}  # config wins on conflict

    unmapped = sorted(set(long_df["team"]) - set(name_map))
    if unmapped:
        fbref_names = _fbref_team_names()
        newly_resolved = {}
        still_unmapped = []
        for name in unmapped:
            match, score, runner_up = best_match(name, fbref_names)
            if match is None:
                still_unmapped.append(name)
                continue
            newly_resolved[name] = match
            log.info("Auto-resolved Understat team name %r -> %r (score=%.3f, runner-up=%.3f, "
                      "cached in %s for future runs)",
                      name, match, score, runner_up, AUTO_RESOLVED_PATH)

        if newly_resolved:
            name_map.update(newly_resolved)
            auto_resolved.update(newly_resolved)
            _save_auto_resolved_map(auto_resolved)

        if still_unmapped:
            log.warning("Understat team names with no confident match (not in "
                        "config.TEAM_NAME_MAP, the auto-resolved cache, or a confident "
                        "fuzzy match against FBref's team list -- their rows will fail "
                        "to join FBref data): %s. Add an entry to config.TEAM_NAME_MAP "
                        "by hand.", still_unmapped)

    long_df["team"] = long_df["team"].map(name_map).fillna(long_df["team"])
    long_df["opponent"] = long_df["opponent"].map(name_map).fillna(long_df["opponent"])

    out_path = Path(config.RAW_DIR) / "understat_matches.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    guard_against_shrinkage(out_path, existing_parquet_row_count(out_path), len(long_df))

    long_df.to_parquet(out_path, index=False)
    log.info("Saved %d team-match rows to %s", len(long_df), out_path)
    return long_df


if __name__ == "__main__":
    fetch_understat_matches()
