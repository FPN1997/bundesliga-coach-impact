"""
For every coaching change found in the dataset, compare team performance in
the IMPACT_WINDOW matches immediately before vs. immediately after the
change: points-per-game, goal difference per game, and (where available)
xG difference per game and PPDA. Ranks changes by biggest swing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _window_stats(rows: pd.DataFrame) -> dict:
    stats = {"matches": len(rows), "ppg": rows["points"].mean()}
    if {"gf", "ga"}.issubset(rows.columns):
        stats["goal_diff_pg"] = (rows["gf"] - rows["ga"]).mean()
    if "xg" in rows.columns and "xga" in rows.columns:
        stats["xg_diff_pg"] = (rows["xg"] - rows["xga"]).mean()
    if "ppda" in rows.columns:
        stats["ppda"] = rows["ppda"].mean()
    return stats


def compute_coach_impact(window: int = config.IMPACT_WINDOW) -> pd.DataFrame:
    df = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")
    df = df.dropna(subset=["coach"]).sort_values(["team", "date"])

    results = []
    for team, team_matches in df.groupby("team"):
        team_matches = team_matches.reset_index(drop=True)
        # Boundaries where the coach in charge changes from one row to the next.
        change_idx = team_matches.index[
            team_matches["coach"] != team_matches["coach"].shift(1)
        ][1:]  # skip index 0 -- that's the first known coach, not a "change"

        for idx in change_idx:
            before = team_matches.iloc[max(0, idx - window):idx]
            after = team_matches.iloc[idx:idx + window]
            if before.empty or after.empty:
                continue

            before_stats = _window_stats(before)
            after_stats = _window_stats(after)

            row = {
                "team": team,
                "outgoing_coach": team_matches.iloc[idx - 1]["coach"],
                "incoming_coach": team_matches.iloc[idx]["coach"],
                "change_date": team_matches.iloc[idx]["date"],
            }
            for key in before_stats:
                if key == "matches":
                    row["matches_before"] = before_stats["matches"]
                    row["matches_after"] = after_stats.get("matches")
                    continue
                row[f"{key}_before"] = before_stats[key]
                row[f"{key}_after"] = after_stats.get(key)
                row[f"{key}_delta"] = after_stats.get(key, float("nan")) - before_stats[key]
            results.append(row)

    impact_df = pd.DataFrame(results)
    if impact_df.empty:
        log.warning("No coaching changes with full before/after windows found. "
                    "Widen SEASONS in config.py or check coach_history.csv coverage.")
        return impact_df

    impact_df = impact_df.sort_values("ppg_delta", ascending=False).reset_index(drop=True)

    out_path = Path(config.OUTPUT_DIR) / "coach_impact_rankings.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    impact_df.to_csv(out_path, index=False)
    log.info("Saved %d coaching-change rows to %s", len(impact_df), out_path)

    if not impact_df.empty:
        log.info("Biggest positive PPG swing:\n%s",
                  impact_df.head(3)[["team", "incoming_coach", "ppg_before", "ppg_after", "ppg_delta"]])
        log.info("Biggest negative PPG swing:\n%s",
                  impact_df.tail(3)[["team", "incoming_coach", "ppg_before", "ppg_after", "ppg_delta"]])

    return impact_df


if __name__ == "__main__":
    compute_coach_impact()
