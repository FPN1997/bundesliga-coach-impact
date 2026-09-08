"""
Leak-safe feature engineering for the formation/outcome predictor.

Every rolling or cumulative stat here is computed with `.shift(1)` before
the window, so the features attached to a given match are built ONLY from
that team's matches strictly before it -- never the match itself or
anything after it. Skipping the shift is the single easiest way to build a
model that looks great in-sample and is useless in reality (it would be
"predicting" a match partly from its own result), so it gets called out
explicitly at every rolling computation below rather than assumed obvious.

One deliberate exception: `formation` and `opp_formation` are the formations
actually fielded IN that match, not a forecast of what a team will line up
in. That makes this a "which formation choices tend to pair with wins,
controlling for form and home advantage" model -- an explanatory tool, not
a pre-kickoff bookmaker-style predictor. See README's outcome-predictor
section for the honest framing and how you'd extend this to a true
pre-match predictor (use each team's recent modal formation instead).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config
from src.formation_utils import clean_formation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROLLING_WINDOW = config.FEATURE_ROLLING_WINDOW


def _one_team_features(d: pd.DataFrame) -> pd.DataFrame:
    """Rolling/cumulative form stats for a single team's matches, already
    sorted by date. Every stat is shifted by one match before any window
    is applied, so row i's features use only rows < i."""
    d = d.copy()
    goal_diff = (d["gf"] - d["ga"]).shift(1)
    xg_diff = (d["xg"] - d["xga"]).shift(1)
    points_prior = d["points"].shift(1)
    ppda_prior = d["ppda"].shift(1)

    d["form_ppg"] = points_prior.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    d["form_goal_diff"] = goal_diff.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    d["form_xg_diff"] = xg_diff.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    d["form_ppda"] = ppda_prior.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()

    # Season-to-date PPG: expanding mean within each season, shifted so the
    # match being featurized isn't included in its own average.
    d["season_ppg_to_date"] = (
        d.groupby("season")["points"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    return d


def _team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-team rolling/cumulative form stats, shifted to exclude the current match."""
    df = df.sort_values(["team", "date"]).copy()
    return (
        df.groupby("team", group_keys=False)[df.columns.tolist()]
        .apply(_one_team_features)
    )


def _attach_coach_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Days between the current coach's tenure start and this match."""
    coaches_path = Path(config.COACH_HISTORY_RESOLVED_CSV)
    if not coaches_path.exists():
        log.warning("%s not found -- coach_tenure_days will be all-NaN. Run "
                    "fetch_coach_history.py first.", coaches_path)
        df["coach_tenure_days"] = pd.NA
        return df

    coaches = pd.read_csv(coaches_path, parse_dates=["start_date"])
    # One start_date per (team, coach); a coach who left and came back gets
    # multiple rows, so take the start_date closest to (and before) the
    # match date rather than joining ambiguously.
    df = df.sort_values("date")
    tenure_starts = []
    for team, coach, match_date in zip(df["team"], df["coach"], df["date"]):
        candidates = coaches[(coaches["team"] == team) & (coaches["coach"] == coach)
                              & (coaches["start_date"] <= match_date)]
        tenure_starts.append(candidates["start_date"].max() if not candidates.empty else pd.NaT)
    df["coach_tenure_days"] = (df["date"] - pd.Series(tenure_starts, index=df.index)).dt.days
    return df


def build_feature_table() -> pd.DataFrame:
    """Build the full leak-safe, opponent-joined feature table for the outcome model."""
    matches = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")
    matches = matches.copy()
    matches["formation"] = matches["formation"].apply(clean_formation)
    matches["opp_formation"] = matches["opp_formation"].apply(clean_formation)

    featured = _team_rolling_features(matches)
    featured = _attach_coach_tenure(featured)

    form_cols = ["form_ppg", "form_goal_diff", "form_xg_diff", "form_ppda",
                 "season_ppg_to_date", "coach_tenure_days"]

    # Self-join: for each row, pull in the OPPONENT's own pre-match features
    # as of the same date (their "team" perspective row for this match).
    opp_side = featured[["date", "team"] + form_cols].rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in form_cols}}
    )
    featured = featured.merge(opp_side, on=["date", "opponent"], how="left")

    keep_cols = [
        "date", "season", "team", "opponent", "venue", "formation", "opp_formation",
        "result", "points",
    ] + form_cols + [f"opp_{c}" for c in form_cols]
    featured = featured[keep_cols]

    before = len(featured)
    featured = featured.dropna(subset=form_cols + [f"opp_{c}" for c in form_cols]
                                + ["formation", "opp_formation", "result"])
    log.info("Feature table: %d rows (dropped %d with insufficient rolling history "
              "or missing formation/result -- expected for each team's first "
              "%d matches per season and any future/unplayed fixtures)",
              len(featured), before - len(featured), ROLLING_WINDOW)

    out_path = Path(config.PROCESSED_DIR) / "outcome_features.parquet"
    featured.to_parquet(out_path, index=False)
    log.info("Saved feature table -> %s", out_path)
    return featured


if __name__ == "__main__":
    build_feature_table()
