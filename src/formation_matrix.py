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
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config
from src import viz_style as vs
from src.formation_utils import clean_formation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIN_MATCHUP_SAMPLES = 5  # hide matchup cells with too few matches to be meaningful
# The heatmap PNG shows only formations fielded at least this often --
# including every one-off tactical tweak turns the grid into mostly noise.
# The CSVs keep everything.
PLOT_MIN_FORMATION_MATCHES = 100


def common_formations(formations: pd.Series, pivot: pd.DataFrame) -> list[str]:
    """Formations used at least PLOT_MIN_FORMATION_MATCHES times that appear on
    both axes of `pivot`, most-used first. Falls back to every formation on
    both axes when fewer than two qualify (a small dataset -- e.g. the E2E
    fixture -- would otherwise leave nothing to plot)."""
    usage = formations.value_counts()
    both = [f for f in usage.index if f in pivot.index and f in pivot.columns]
    common = [f for f in both if usage[f] >= PLOT_MIN_FORMATION_MATCHES]
    return common if len(common) >= 2 else both


def build_formation_matrix() -> pd.DataFrame:
    df = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")

    df = df.copy()
    df["formation"] = df["formation"].apply(clean_formation)
    df["opp_formation"] = df["opp_formation"].apply(clean_formation)
    df = df.dropna(subset=["formation", "opp_formation", "points"])

    grouped = df.groupby(["formation", "opp_formation"])
    matrix_long = grouped["points"].agg(["mean", "count"]).reset_index()
    matrix_long = matrix_long[matrix_long["count"] >= MIN_MATCHUP_SAMPLES]

    pivot = matrix_long.pivot(index="formation", columns="opp_formation", values="mean")

    out_csv = Path(config.OUTPUT_DIR) / "formation_matchup_matrix.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out_csv)
    matrix_long.rename(columns={"mean": "ppg", "count": "matches"}).to_csv(
        Path(config.OUTPUT_DIR) / "formation_matchup_long.csv", index=False)
    log.info("Saved formation matchup matrix (%d formations x %d) -> %s",
              pivot.shape[0], pivot.shape[1], out_csv)

    # Colour is centred on the league-average points per game (~1.37: a
    # match hands out ~2.7 points between two teams), not an arbitrary
    # 1.0 -- centring low made most cells look "good" by construction.
    league_ppg = df["points"].mean()
    shown = pivot.loc[common_formations(df["formation"], pivot), common_formations(df["formation"], pivot)]

    fig, ax = plt.subplots(figsize=(0.95 * shown.shape[1] + 2.5, 0.75 * shown.shape[0] + 2),
                           facecolor=vs.SURFACE)
    sns.heatmap(
        shown, annot=True, fmt=".2f", cmap=vs.diverging_cmap(),
        norm=vs.centered_norm(shown.to_numpy(), league_ppg),
        linewidths=2, linecolor=vs.SURFACE, annot_kws={"fontsize": 9},
        cbar_kws={"label": f"Points per game (league average {league_ppg:.2f} = grey)"}, ax=ax,
    )
    ax.set_facecolor(vs.SURFACE)
    ax.tick_params(colors=vs.INK_2, labelsize=9, length=0)
    ax.set_xlabel("Opponent's formation", color=vs.INK_2)
    ax.set_ylabel("Team's formation", color=vs.INK_2)
    vs.title(ax, f"Points per game by formation matchup, {config.SEASONS[0]} to {config.SEASONS[-1]}\n"
                 f"formations used {PLOT_MIN_FORMATION_MATCHES}+ times; blank = fewer than "
                 f"{MIN_MATCHUP_SAMPLES} matches")
    fig.tight_layout()

    out_png = Path(config.OUTPUT_DIR) / "formation_matchup_heatmap.png"
    fig.savefig(out_png, dpi=150, facecolor=vs.SURFACE)
    plt.close(fig)
    log.info("Saved heatmap -> %s", out_png)

    return pivot


def coach_preferred_formations() -> pd.DataFrame:
    """Each coach's most-used formation(s) and how often they used it."""
    df = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")
    df = df.copy()
    df["formation"] = df["formation"].apply(clean_formation)
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
