"""
Formation-aware match outcome predictor.

Trains two classifiers (Random Forest, XGBoost) to predict a team's match
result (win/draw/loss) from the formations both sides fielded plus recent
form (rolling PPG, goal difference, xG difference, PPDA, season-to-date PPG)
and how long the current coach has been in charge. See src/features.py's
docstring for why "formation used in the match" makes this an explanatory
model (which formation choices tend to pair with wins) rather than a
pre-kickoff forecaster.

Evaluation uses a TIME-based split (config.TEST_SEASONS held out entirely),
never a random split -- a random split would leak future form/results into
training via the rolling features' shared history and wildly overstate
accuracy. Compared against two baselines: always-predict-majority-class, and
always-predict-home-team-wins (the simplest signal a model needs to beat to
be worth anything).

Outputs to outputs/:
  - models/random_forest.joblib, models/xgboost.joblib
  - outcome_model_metrics.json          -- accuracy/F1 vs baselines
  - outcome_confusion_matrix.png
  - outcome_feature_importance.png
  - formation_matchup_predicted.csv/png -- model-predicted win rate per
    (formation, opp_formation), i.e. the earlier formation_matrix.py
    heatmap's numbers, but adjusted for form/home-advantage instead of raw.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config
from src.features import build_feature_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CATEGORICAL = ["formation", "opp_formation", "venue"]
NUMERIC = [
    "form_ppg", "form_goal_diff", "form_xg_diff", "form_ppda", "season_ppg_to_date",
    "coach_tenure_days", "opp_form_ppg", "opp_form_goal_diff", "opp_form_xg_diff",
    "opp_form_ppda", "opp_season_ppg_to_date", "opp_coach_tenure_days",
]


def _season_code(season_str: str) -> str:
    """"2025-2026" -> "2526" -- the compact form soccerdata/FBref uses for
    the 'season' column (verified live: e.g. 2023-2024 comes back as 2324)."""
    start, end = season_str.split("-")
    return start[2:] + end[2:]


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_mask = df["season"].astype(str).isin(
        [_season_code(s) for s in config.TEST_SEASONS]
    )
    train, test = df[~test_mask], df[test_mask]
    log.info("Train: %d rows (%s .. %s) | Test: %d rows (seasons %s)",
              len(train), train["date"].min().date(), train["date"].max().date(),
              len(test), config.TEST_SEASONS)
    if test.empty:
        raise RuntimeError(f"No rows matched TEST_SEASONS={config.TEST_SEASONS} -- check "
                            f"the 'season' column's actual values: {sorted(df['season'].unique())}")
    return train, test


def _build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def _baselines(y_train: pd.Series, y_test: pd.Series, venue_test: pd.Series) -> dict:
    majority_class = y_train.mode().iloc[0]
    majority_acc = (y_test == majority_class).mean()

    # "Home team wins" baseline: predict W when this row's team was at
    # home, L when away (a draw is never predicted -- it's the simplest
    # baseline that uses the one piece of information everyone agrees
    # matters, home advantage, and nothing else).
    home_pred = np.where(venue_test.to_numpy() == "Home", "W", "L")
    home_acc = float(np.mean(y_test.to_numpy() == home_pred))

    return {
        "majority_class_accuracy": float(majority_acc),
        "majority_class": majority_class,
        "home_advantage_only_accuracy": float(home_acc),
    }


def train_and_evaluate() -> dict:
    df = build_feature_table()
    train, test = _split(df)

    le = LabelEncoder()
    y_train = pd.Series(le.fit_transform(train["result"]), index=train.index)
    y_test_encoded = pd.Series(le.transform(test["result"]), index=test.index)

    # class_weight="balanced" on the forest: draws are ~27% of matches but
    # the hardest class to call (no result is "obviously" a draw the way a
    # blowout is obviously a win), so an unweighted forest learns it's
    # cheapest to just never predict D. Balancing trades some overall
    # accuracy for actually attempting the hardest class -- report both,
    # don't just take the higher raw-accuracy number.
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            objective="multi:softprob", num_class=3,
            random_state=42, n_jobs=-1, eval_metric="mlogloss",
        ),
    }

    metrics = {"baselines": _baselines(train["result"], test["result"], test["venue"])}
    model_dir = Path(config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    fitted = {}

    # XGBoost's sklearn wrapper has no class_weight param -- give it the
    # same "balanced" treatment via explicit sample weights so it gets a
    # fair shot at draws too, for the same reason noted above for the forest.
    sample_weight = compute_sample_weight("balanced", y_train)

    for name, model in models.items():
        pipe = _build_pipeline(model)
        fit_kwargs = {"model__sample_weight": sample_weight} if name == "xgboost" else {}
        pipe.fit(train[CATEGORICAL + NUMERIC], y_train, **fit_kwargs)
        pred = pipe.predict(test[CATEGORICAL + NUMERIC])

        acc = accuracy_score(y_test_encoded, pred)
        f1 = f1_score(y_test_encoded, pred, average="macro")
        metrics[name] = {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "classification_report": classification_report(
                y_test_encoded, pred, target_names=le.classes_, output_dict=True
            ),
        }
        log.info("%s: accuracy=%.3f macro_f1=%.3f (baselines: majority=%.3f, home-only=%.3f)",
                  name, acc, f1, metrics["baselines"]["majority_class_accuracy"],
                  metrics["baselines"]["home_advantage_only_accuracy"])

        joblib.dump(pipe, model_dir / f"{name}.joblib")
        fitted[name] = (pipe, pred)

    joblib.dump(le, model_dir / "label_encoder.joblib")

    out_path = Path(config.OUTPUT_DIR) / "outcome_model_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Saved metrics -> %s", out_path)

    _plot_confusion_matrices(fitted, y_test_encoded, le.classes_)
    _plot_feature_importance(fitted, test)
    _formation_matchup_predicted(fitted["xgboost"][0], df)

    return metrics


def _plot_confusion_matrices(fitted: dict, y_test: pd.Series, class_names) -> None:
    fig, axes = plt.subplots(1, len(fitted), figsize=(6 * len(fitted), 5))
    if len(fitted) == 1:
        axes = [axes]
    for ax, (name, (_, pred)) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_test, pred, labels=range(len(class_names)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                    yticklabels=class_names, ax=ax, cbar=False)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    out_path = Path(config.OUTPUT_DIR) / "outcome_confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix plot -> %s", out_path)


def _plot_feature_importance(fitted: dict, sample: pd.DataFrame, top_n: int = 15) -> None:
    fig, axes = plt.subplots(1, len(fitted), figsize=(8 * len(fitted), 6))
    if len(fitted) == 1:
        axes = [axes]
    for ax, (name, (pipe, _)) in zip(axes, fitted.items()):
        feature_names = pipe.named_steps["pre"].get_feature_names_out()
        importances = pipe.named_steps["model"].feature_importances_
        order = np.argsort(importances)[-top_n:]
        ax.barh(range(len(order)), importances[order])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
        ax.set_title(f"{name}: top {top_n} features")
        ax.set_xlabel("Importance")
    fig.tight_layout()
    out_path = Path(config.OUTPUT_DIR) / "outcome_feature_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved feature importance plot -> %s", out_path)


def _formation_matchup_predicted(pipe: Pipeline, df: pd.DataFrame, min_samples: int = 5) -> None:
    """Model-predicted P(win) per (formation, opp_formation), averaged over
    every row with that pairing (using each row's OWN form/coach-tenure
    values -- this isn't "holding form constant," it's averaging the
    model's form-aware prediction across however that pairing actually
    occurred, which is the honest way to summarize a form-conditioned model
    back down to a 2D formation-only view)."""
    proba = pipe.predict_proba(df[CATEGORICAL + NUMERIC])
    label_encoder = joblib.load(Path(config.MODEL_DIR) / "label_encoder.joblib")
    win_class_idx = list(label_encoder.classes_).index("W")
    df = df.copy()
    df["predicted_win_prob"] = proba[:, win_class_idx]

    grouped = df.groupby(["formation", "opp_formation"]).agg(
        predicted_win_prob=("predicted_win_prob", "mean"),
        actual_win_rate=("result", lambda s: (s == "W").mean()),
        n=("result", "size"),
    ).reset_index()
    grouped = grouped[grouped["n"] >= min_samples]

    out_csv = Path(config.OUTPUT_DIR) / "formation_matchup_predicted.csv"
    grouped.to_csv(out_csv, index=False)
    log.info("Saved form-adjusted formation matchup table (%d pairings) -> %s",
              len(grouped), out_csv)

    pivot = grouped.pivot(index="formation", columns="opp_formation", values="predicted_win_prob")
    fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 2, 1.0 * pivot.shape[0] + 2))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0.33,
                cbar_kws={"label": "Model-predicted P(win)"}, ax=ax)
    ax.set_xlabel("Opponent formation")
    ax.set_ylabel("Team formation")
    ax.set_title(f"Predicted win probability by formation matchup\n"
                 f"(form/home-advantage-adjusted; cells with < {min_samples} matches hidden)")
    fig.tight_layout()
    out_png = Path(config.OUTPUT_DIR) / "formation_matchup_predicted.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info("Saved form-adjusted heatmap -> %s", out_png)


if __name__ == "__main__":
    train_and_evaluate()
