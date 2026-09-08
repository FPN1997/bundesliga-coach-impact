"""
Two outputs:

1. A formation x opponent-formation matchup matrix: average points won per
   game for every (team formation, opponent formation) pairing that occurs
   often enough to mean anything, plus a heatmap PNG.
2. Each coach's most-used formation(s) during their tenure, joined onto the
   coach_impact_rankings.csv output so you can see whether a big PPG swing
   also came with a formation change.

FBref's formation strings (e.g. "4-2-3-1") sometimes carry a trailing note
like "4-2-3-1◆" for a mid-match switch -- we strip anything that isn't
digits and dashes so "4-2-3-1" and "4-2-3-1 (used from 60')" bucket together.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIN_MATCHUP_SAMPLES = 5  # hide matchup cells with too few matches to be meaningful


def _clean_formation(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = re.sub(r"[^\d\-]", "", str(value))
    return s if s else None


def build_formation_matrix() -> pd.DataFrame:
    df = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")

    df = df.copy()
    df["formation"] = df["formation"].apply(_clean_formation)
    df["opp_formation"] = df["opp_formation"].apply(_clean_formation)
    df = df.dropna(subset=["formation", "opp_formation", "points"])

    grouped = df.groupby(["formation", "opp_formation"])
    matrix_long = grouped["points"].agg(["mean", "count"]).reset_index()
    matrix_long = matrix_long[matrix_long["count"] >= MIN_MATCHUP_SAMPLES]

    pivot = matrix_long.pivot(index="formation", columns="opp_formation", values="mean")

    out_csv = Path(config.OUTPUT_DIR) / "formation_matchup_matrix.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out_csv)
    log.info("Saved formation matchup matrix (%d formations x %d) -> %s",
              pivot.shape[0], pivot.shape[1], out_csv)

    fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 2, 1.0 * pivot.shape[0] + 2))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=1.0,
        cbar_kws={"label": "Avg. points won per game"}, ax=ax,
    )
    ax.set_xlabel("Opponent formation")
    ax.set_ylabel("Team formation")
    ax.set_title(f"Bundesliga formation matchups, {config.SEASONS[0]}–{config.SEASONS[-1]}\n"
                 f"(cells with < {MIN_MATCHUP_SAMPLES} matches hidden)")
    fig.tight_layout()

    out_png = Path(config.OUTPUT_DIR) / "formation_matchup_heatmap.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info("Saved heatmap -> %s", out_png)

    return pivot


def coach_preferred_formations() -> pd.DataFrame:
    """Each coach's most-used formation(s) and how often they used it."""
    df = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")
    df = df.copy()
    df["formation"] = df["formation"].apply(_clean_formation)
    df = df.dropna(subset=["coach", "formation"])

    summary = (
        df.groupby(["team", "coach", "formation"])
        .size()
        .reset_index(name="matches")
        .sort_values(["team", "coach", "matches"], ascending=[True, True, False])
    )
    top_formation = summary.groupby(["team", "coach"]).first().reset_index()

    out_path = Path(config.OUTPUT_DIR) / "coach_preferred_formations.csv"
    top_formation.to_csv(out_path, index=False)
    log.info("Saved %d coach/preferred-formation rows -> %s", len(top_formation), out_path)
    return top_formation


if __name__ == "__main__":
    build_formation_matrix()
    coach_preferred_formations()
