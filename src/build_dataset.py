"""
Merge FBref (results + formations), Understat (xG + PPDA + deep completions),
and the coach
tenure table into one match-level dataset: one row per team per league
match, with the coach in charge of that team on that date attached.

Run fetch_fbref.py, fetch_understat.py, and fetch_coach_history.py first
(or just run run_pipeline.py, which does it in order).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# FBref column -> our snake_case name. Verified live against the 2023-24
# season (see README "Known rough edges") -- if a future soccerdata release
# renames these, fetch_fbref.py's INFO log prints the raw columns so the
# mismatch is easy to spot.
FBREF_RENAME = {
    "GF": "gf",
    "GA": "ga",
    "Formation": "formation",
    "Opp Formation": "opp_formation",
    "Poss": "possession",
    "Captain": "captain",
    "Referee": "referee",
}


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fbref = pd.read_parquet(Path(config.RAW_DIR) / "fbref_schedule.parquet")
    understat = pd.read_parquet(Path(config.RAW_DIR) / "understat_matches.parquet")
    coaches = pd.read_csv(
        Path(config.COACH_HISTORY_RESOLVED_CSV), parse_dates=["start_date", "end_date"]
    )
    return fbref, understat, coaches


def _assign_coach(matches: pd.DataFrame, coaches: pd.DataFrame) -> pd.DataFrame:
    """For each (team, date) row, find the coach whose tenure window covers it."""
    matches = matches.sort_values(["team", "date"]).reset_index(drop=True)
    coaches = coaches.sort_values(["team", "start_date"]).reset_index(drop=True)
    # Open-ended tenures (still in the job) get end_date = NaT -> treat as
    # "far future" so the merge_asof / between check includes today's matches.
    coaches = coaches.copy()
    coaches["end_date_filled"] = coaches["end_date"].fillna(pd.Timestamp("2100-01-01"))

    assigned_frames = []
    for team, team_matches in matches.groupby("team"):
        team_coaches = coaches[coaches["team"] == team]
        if team_coaches.empty:
            log.warning("No coach-history rows for team=%s -- coach will be NaN "
                        "for all their matches. Add rows to %s.",
                        team, config.COACH_HISTORY_MANUAL_CSV)
            team_matches = team_matches.copy()
            team_matches["coach"] = pd.NA
            assigned_frames.append(team_matches)
            continue

        def find_coach(match_date: pd.Timestamp) -> str | None:
            hit = team_coaches[
                (team_coaches["start_date"] <= match_date)
                & (match_date <= team_coaches["end_date_filled"])
            ]
            if hit.empty:
                return None
            # If tenures overlap (data error) take the most recently started one.
            return hit.sort_values("start_date").iloc[-1]["coach"]

        team_matches = team_matches.copy()
        team_matches["coach"] = team_matches["date"].apply(find_coach)
        assigned_frames.append(team_matches)

    return pd.concat(assigned_frames, ignore_index=True)


def build_dataset() -> pd.DataFrame:
    fbref, understat, coaches = _load_raw()

    fbref = fbref.rename(columns=FBREF_RENAME)
    fbref["date"] = pd.to_datetime(fbref["date"]).dt.normalize()
    # 'season' comes back as a plain int (e.g. 2324); keep the string form
    # soccerdata configured us with so it's comparable to config.SEASONS.
    fbref["match_date"] = fbref["date"]

    understat = understat.copy()
    understat["match_date"] = pd.to_datetime(understat["date"]).dt.normalize()

    understat_slim = understat[
        ["match_date", "team", "xg", "xga", "ppda", "deep_completions", "points"]
    ].rename(columns={"points": "points_understat"})

    # FBref's per-team schedule lists the full season's fixtures, played or
    # not -- including config.SEASONS' current/ongoing season pulls in every
    # remaining fixture through the end of that season, with gf/ga blank.
    # Drop those: they're not observations, and left in they'd silently
    # inflate coach_impact.py's matches_before/after window counts without
    # contributing any real points/goal data (pandas .mean() already skips
    # the NaNs, so the averages stay correct either way, but the reported
    # sample size would lie).
    unplayed = fbref["gf"].isna() | fbref["ga"].isna()
    if unplayed.any():
        log.info("Dropping %d unplayed/future fixture rows (no score yet)", unplayed.sum())
        fbref = fbref[~unplayed]

    merged = fbref.merge(understat_slim, on=["match_date", "team"], how="left")

    unmatched = merged["xg"].isna().sum()
    if unmatched:
        log.warning("%d/%d rows (%.1f%%) got no Understat match on (date, team) -- "
                    "likely a team-name mapping gap (check config.TEAM_NAME_MAP) or "
                    "a postponed/rescheduled fixture where the two sources disagree "
                    "on the match date.", unmatched, len(merged), 100 * unmatched / len(merged))

    merged = _assign_coach(merged, coaches)

    # Points for this team in this match, derived from FBref's result column.
    if "result" in merged.columns:
        merged["points"] = merged["result"].map({"W": 3, "D": 1, "L": 0})
    else:
        merged["points"] = (merged["gf"] > merged["ga"]).astype(int) * 3 + \
                            (merged["gf"] == merged["ga"]).astype(int)

    out_path = Path(config.PROCESSED_DIR) / "match_dataset.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    log.info("Saved merged dataset: %d rows, %d columns -> %s",
              len(merged), len(merged.columns), out_path)

    missing_coach = merged["coach"].isna().mean()
    if missing_coach > 0:
        log.warning("%.1f%% of rows have no coach assigned -- fill gaps in %s",
                    missing_coach * 100, config.COACH_HISTORY_MANUAL_CSV)

    return merged


if __name__ == "__main__":
    build_dataset()
