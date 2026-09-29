"""
Merge Understat (results, xG, PPDA, deep completions), FBref (formations,
possession, referee...) and the coach tenure table into one match-level
dataset: one row per team per league match, with the coach in charge of
that team on that date attached.

Understat is the base: every played match, with both sides' goals. FBref
only adds its extras, joined on (season, team, opponent, venue) -- unique
within a league season, and unlike the date it doesn't depend on the two
sources agreeing when a rescheduled match was played. Until 29 Sep 2026
FBref was the base; it now challenges plain browsers, and the fetch stops at
the first challenge (src/fetch_fbref.py), so new matches must not depend on
it. A match FBref doesn't have yet simply has no formation.

The score is FBref's wherever FBref has the match: it records the official
result, Understat the one on the pitch, and they differ when a result is
awarded (Union Berlin 1-1 Bochum on 14 Dec 2024 became a 2-0 Bochum win after
a lighter hit Bochum's goalkeeper). Every such disagreement is logged, so a
newly awarded result that FBref hasn't caught up with gets noticed.

Run fetch_fbref.py, fetch_understat.py, and fetch_coach_history.py first
(or just run `bundesliga pipeline`, which does it in order).
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


# FBref columns kept as extras, in the dataset's (historical) column order
FBREF_EXTRAS = ["game", "time", "round", "day", "possession", "Attendance", "captain", "formation",
                "opp_formation", "referee", "match_report", "Notes", "fbref_game_id"]
COLUMNS = ["league", "season", "team", "game", "date", "time", "round", "day", "venue", "result", "gf",
           "ga", "opponent", "possession", "Attendance", "captain", "formation", "opp_formation",
           "referee", "match_report", "Notes", "fbref_game_id", "match_date", "xg", "xga", "ppda",
           "deep_completions", "points_understat", "coach", "points"]
KEYS = ["season", "team", "opponent", "venue"]  # unique within a league season ...
MEETING = "meeting"  # ... but numbered in date order anyway, should two clubs ever meet twice at one venue


def understat_results(understat: pd.DataFrame) -> pd.DataFrame:
    """Understat's one-row-per-team-match table with both sides' goals."""
    u = understat.copy()
    u["season"] = u["season"].astype(str)
    u["match_date"] = pd.to_datetime(u["date"]).dt.normalize()
    against = u[["game_id", "team", "goals"]].rename(columns={"team": "opponent", "goals": "ga"})
    u = u.merge(against, on=["game_id", "opponent"], how="left").rename(
        columns={"goals": "gf", "points": "points_understat"})
    return u.dropna(subset=["gf", "ga"])


def build_dataset() -> pd.DataFrame:
    fbref, understat, coaches = _load_raw()
    results = understat_results(understat)

    fbref = fbref.rename(columns=FBREF_RENAME)
    fbref["season"] = fbref["season"].astype(str)
    fbref["date"] = pd.to_datetime(fbref["date"]).dt.normalize()

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

    # reindex: an extra FBref doesn't provide is just empty -- they're extras now
    extras = (fbref.reindex(columns=KEYS + FBREF_EXTRAS + ["date", "gf", "ga"])
              .rename(columns={"date": "fbref_date", "gf": "fbref_gf", "ga": "fbref_ga"})
              .sort_values("fbref_date"))
    extras[MEETING] = extras.groupby(KEYS).cumcount()
    results = results.sort_values("match_date")
    results[MEETING] = results.groupby(KEYS).cumcount()
    merged = results.merge(extras, on=[*KEYS, MEETING], how="left").drop(columns=MEETING)
    # FBref's date where it has the match (keeps the history exactly as it was
    # built before), Understat's otherwise
    merged["date"] = merged["fbref_date"].fillna(merged["match_date"])
    merged["match_date"] = merged["date"]
    merged["league"] = config.LEAGUE

    has_fbref = merged["fbref_gf"].notna() & merged["fbref_ga"].notna()
    differs = has_fbref & ((merged["fbref_gf"] != merged["gf"]) | (merged["fbref_ga"] != merged["ga"]))
    for r in merged[differs & (merged["venue"] == "Home")].itertuples():
        log.info("Score differs: %s v %s on %s -- FBref %d-%d (official, used), Understat %d-%d",
                 r.team, r.opponent, r.date.date(), r.fbref_gf, r.fbref_ga, r.gf, r.ga)
    merged["gf"] = merged["fbref_gf"].where(has_fbref, merged["gf"])
    merged["ga"] = merged["fbref_ga"].where(has_fbref, merged["ga"])

    no_fbref = merged["fbref_date"].isna()
    if no_fbref.any():
        log.info("%d of %d team-matches have no FBref row yet (no formation): FBref refusing or "
                 "not yet fetched -- latest %s", no_fbref.sum(), len(merged),
                 merged.loc[no_fbref, "date"].max().date())
    unknown = sorted(set(results["team"]) - set(fbref["team"]))
    if unknown:
        log.warning("Understat clubs with no FBref name match (check config.TEAM_NAME_MAP): %s", unknown)

    merged = _assign_coach(merged, coaches)
    merged["result"] = pd.Series(pd.NA, index=merged.index, dtype="string").mask(
        merged["gf"] > merged["ga"], "W").mask(merged["gf"] == merged["ga"], "D").mask(
        merged["gf"] < merged["ga"], "L")
    merged["points"] = merged["result"].map({"W": 3, "D": 1, "L": 0})
    merged = merged[COLUMNS]

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
