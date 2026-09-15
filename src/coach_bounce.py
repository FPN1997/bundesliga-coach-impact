"""
"New coach bounce predictor": for a coaching change, can the SIZE of the
resulting PPG swing (coach_impact.py's ppg_delta) be predicted from the
team's pre-change form plus the INCOMING coach's own historical profile --
how they performed, on average, in whatever earlier matches THEY coached
(at any club) that are already in this dataset, strictly before this
change?

This is a coach-level join, not a team-level one: for each row in
coach_impact_rankings.csv, every match_dataset.parquet row where
coach == incoming_coach and date < change_date (regardless of which team
that was at) becomes that coach's "track record" going into this
appointment. Requiring date < change_date (not <=) is the leak-safety
property here, same spirit as features.py's .shift(1) -- a coach's
historical profile must never include the very matches whose outcome this
model is trying to explain.

**The honest headline finding, up front**: only 32 of 107 coaching changes
in the dataset have ANY prior match history for the incoming coach at all
(most incoming coaches are either new to management within this window, or
their previous job predates config.SEASONS, or was outside the Bundesliga
entirely -- none of which this dataset can see). Restricting further to a
usable sample size (MIN_PRIOR_MATCHES) leaves ~28 rows. That is nowhere
near enough to trust a model trained on it, whatever the reported R² says
-- reported here with a leave-one-out CV estimate (the only honest option
at this n; a fixed train/test split would put ~6 rows in the test fold)
and a trivial "always predict the training mean" baseline for comparison,
not as a claim that this is a usable predictor. Treat this as a working
scaffold for when more seasons make the sample bigger, not a finished
result -- see the README's "Extending" section.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIN_PRIOR_MATCHES = 10  # below this, "the coach's historical profile" is mostly noise

# Pre-change team form (already in coach_impact_rankings.csv) plus the
# incoming coach's own historical profile (computed here).
TEAM_FORM_COLS = ["ppg_before", "goal_diff_pg_before", "xg_diff_pg_before", "ppda_before"]
COACH_HISTORY_COLS = [
    "coach_hist_ppg", "coach_hist_goal_diff_pg", "coach_hist_xg_diff_pg",
    "coach_hist_ppda", "coach_hist_deep_completions", "coach_hist_n_clubs",
]
FEATURE_COLS = TEAM_FORM_COLS + COACH_HISTORY_COLS
TARGET_COL = "ppg_delta"


def _coach_historical_profile(
    coach: str, before_date: pd.Timestamp, matches: pd.DataFrame,
) -> dict | None:
    """This coach's track record from every match_dataset.parquet row where
    they were in charge, strictly before `before_date`, at ANY club --
    None if there's no prior history at all. The caller applies its own
    min_prior_matches threshold via the returned "coach_hist_matches" count
    (kept separate from this function so that threshold stays a single,
    caller-controlled parameter rather than duplicated here)."""
    prior = matches[(matches["coach"] == coach) & (matches["date"] < before_date)]
    if prior.empty:
        return None
    return {
        "coach_hist_ppg": prior["points"].mean(),
        "coach_hist_goal_diff_pg": (prior["gf"] - prior["ga"]).mean(),
        "coach_hist_xg_diff_pg": (prior["xg"] - prior["xga"]).mean(),
        "coach_hist_ppda": prior["ppda"].mean(),
        "coach_hist_deep_completions": prior["deep_completions"].mean(),
        "coach_hist_n_clubs": prior["team"].nunique(),
        "coach_hist_matches": len(prior),
    }


def build_bounce_dataset(min_prior_matches: int = MIN_PRIOR_MATCHES) -> pd.DataFrame:
    impact = pd.read_csv(
        Path(config.OUTPUT_DIR) / "coach_impact_rankings.csv", parse_dates=["change_date"]
    )
    matches = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])

    rows = []
    for _, change in impact.iterrows():
        profile = _coach_historical_profile(
            change["incoming_coach"], change["change_date"], matches
        )
        if profile is None or profile["coach_hist_matches"] < min_prior_matches:
            continue
        row = {col: change[col] for col in ["team", "incoming_coach", "change_date", TARGET_COL]}
        row.update({col: change[col] for col in TEAM_FORM_COLS})
        row.update(profile)
        rows.append(row)

    out = pd.DataFrame(rows)
    log.info("Coach-bounce dataset: %d/%d coaching changes have >= %d prior matches for the "
              "incoming coach elsewhere in the dataset (the rest have no usable track record "
              "to build a profile from -- new-to-management, pre-SEASONS, or non-Bundesliga).",
              len(out), len(impact), min_prior_matches)

    out_path = Path(config.PROCESSED_DIR) / "coach_bounce_features.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Saved -> %s", out_path)
    return out


def train_and_evaluate(min_prior_matches: int = MIN_PRIOR_MATCHES) -> dict:
    df = build_bounce_dataset(min_prior_matches)
    if len(df) < 10:
        raise RuntimeError(
            f"Only {len(df)} usable rows -- too few to fit or evaluate anything meaningful. "
            f"Widen config.SEASONS (more history for coaches to have a track record in) or "
            f"lower min_prior_matches (at the cost of noisier profiles)."
        )

    X = df[FEATURE_COLS].to_numpy()
    y = df[TARGET_COL].to_numpy()

    # RidgeCV rather than a hand-picked alpha: lets leave-one-out CV pick the
    # regularization strength itself, appropriate given how small this
    # dataset is -- an unregularized or under-regularized linear model would
    # just memorize ~28 points, and a tree ensemble would be even worse.
    model = RidgeCV(alphas=np.logspace(-2, 3, 30))

    # Leave-one-out, not a single held-out split: with ~28 rows a fixed
    # train/test split would put a handful of rows in "test" and report a
    # number driven almost entirely by which few rows happened to land
    # there. LOO uses every row as a test point exactly once instead.
    loo = LeaveOneOut()
    pred = cross_val_predict(model, X, y, cv=loo)
    baseline_pred = np.array([
        np.delete(y, i).mean() for i in range(len(y))
    ])  # "always predict the training-fold mean" -- the trivial comparison point

    metrics = {
        "n_rows": len(df),
        "n_coaching_changes_total": int(
            pd.read_csv(Path(config.OUTPUT_DIR) / "coach_impact_rankings.csv").shape[0]
        ),
        "min_prior_matches": min_prior_matches,
        "loo_r2": float(r2_score(y, pred)),
        "loo_mae": float(mean_absolute_error(y, pred)),
        "baseline_loo_r2": float(r2_score(y, baseline_pred)),
        "baseline_loo_mae": float(mean_absolute_error(y, baseline_pred)),
    }
    log.info(
        "Leave-one-out CV (n=%d): R2=%.3f MAE=%.3f  |  baseline (predict train-fold mean): "
        "R2=%.3f MAE=%.3f",
        metrics["n_rows"], metrics["loo_r2"], metrics["loo_mae"],
        metrics["baseline_loo_r2"], metrics["baseline_loo_mae"],
    )

    model.fit(X, y)  # final fit on everything -- LOO above was for evaluation only
    metrics["chosen_alpha"] = float(model.alpha_)
    metrics["coefficients"] = dict(zip(FEATURE_COLS, model.coef_.tolist(), strict=True))

    model_dir = Path(config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "coach_bounce_ridge.joblib")

    metrics_path = Path(config.OUTPUT_DIR) / "coach_bounce_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Saved metrics -> %s", metrics_path)

    _plot_predicted_vs_actual(y, pred, metrics)

    return metrics


def _plot_predicted_vs_actual(y: np.ndarray, pred: np.ndarray, metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, pred, alpha=0.7)
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Actual PPG delta")
    ax.set_ylabel("Leave-one-out predicted PPG delta")
    ax.set_title(f"Coach-bounce predictor (n={metrics['n_rows']}, "
                 f"LOO R2={metrics['loo_r2']:.2f} vs baseline {metrics['baseline_loo_r2']:.2f})")
    ax.legend()
    fig.tight_layout()
    out_path = Path(config.OUTPUT_DIR) / "coach_bounce_predicted_vs_actual.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved plot -> %s", out_path)


if __name__ == "__main__":
    train_and_evaluate()
