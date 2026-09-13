"""
GridSearchCV hyperparameter tuning for the outcome predictor.

Two things this deliberately does NOT do the "default" way, both for the
same reason as the train/test split in outcome_predictor.py -- this is
temporal data, and pretending otherwise is the easiest way to quietly
overstate how good a model is:

1. **Cross-validation uses `TimeSeriesSplit`, not the default random
   K-fold.** A random K-fold would train on some matches from AFTER the
   matches it's validated against in the same fold -- exactly the leakage
   the time-based train/test split already guards against, just
   reintroduced one level down inside cross-validation. TimeSeriesSplit
   only ever validates on a slice that comes after everything it trained
   on. The training data is sorted by date before this happens.

2. **Class-balance weighting is decoupled from hyperparameter search.**
   Grid-searching XGBoost's structural hyperparameters (depth, learning
   rate, ...) together with per-row sample_weight requires opting into
   sklearn's metadata-routing machinery for not much benefit here, since
   the two concerns are mostly orthogonal: GridSearchCV selects
   hyperparameters using macro-F1 (which already penalizes a model that
   ignores draws, weighted or not), then the FINAL refit on the full
   training set applies the same "balanced" weighting used in
   outcome_predictor.py, for a fair comparison against the untuned numbers.

Results get written into outputs/outcome_model_metrics.json alongside the
untuned numbers (as "<model>_tuned" keys) so both are visible side by side,
and tuned models are saved as outputs/models/<name>_tuned.joblib -- the
untuned ones aren't overwritten.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config
from src.features import build_feature_table
from src.outcome_predictor import (
    FORMATION_COLS_ACTUAL,
    NUMERIC,
    _build_pipeline,
    _split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

N_CV_SPLITS = config.N_CV_SPLITS

RF_PARAM_GRID = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [8, 12, None],
    "model__min_samples_leaf": [1, 5],
}

XGB_PARAM_GRID = {
    "model__n_estimators": [150, 300],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.03, 0.1],
    "model__subsample": [0.8, 1.0],
}


def _grid_search(pipe, param_grid: dict, X, y, tscv: TimeSeriesSplit) -> GridSearchCV:
    search = GridSearchCV(
        pipe, param_grid, cv=tscv, scoring="f1_macro", n_jobs=-1, refit=True,
    )
    search.fit(X, y)
    log.info("Best CV macro-F1=%.3f with params: %s", search.best_score_, search.best_params_)
    return search


def tune_and_evaluate(
    formation_cols: list[str] = FORMATION_COLS_ACTUAL,
    variant: str = "",
) -> dict:
    """variant: suffix appended to every tuned-model name / metrics key,
    e.g. "_prematch" when formation_cols=FORMATION_COLS_PREMATCH. Empty
    string (default) reproduces the original actual-formation tuning run,
    same "<model>_tuned" keys as before this parameter existed."""
    categorical = [*formation_cols, "venue"]

    df = build_feature_table()
    train, test = _split(df)
    # TimeSeriesSplit needs chronological order -- the feature table already
    # comes out sorted by team then date from features.py, not by date alone.
    train = train.sort_values("date").reset_index(drop=True)

    le = LabelEncoder()
    y_train = le.fit_transform(train["result"])
    y_test = le.transform(test["result"])
    X_train, X_test = train[categorical + NUMERIC], test[categorical + NUMERIC]

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    sample_weight = compute_sample_weight("balanced", y_train)

    metrics_path = Path(config.OUTPUT_DIR) / "outcome_model_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    model_dir = Path(config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Random Forest: class_weight is a constructor arg, no routing needed ---
    log.info("Grid-searching random_forest%s (%d combos x %d CV splits)...",
              variant, np.prod([len(v) for v in RF_PARAM_GRID.values()]), N_CV_SPLITS)
    rf_pipe = _build_pipeline(RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=1,  # n_jobs=1: GridSearchCV parallelizes instead
    ), categorical)
    rf_search = _grid_search(rf_pipe, RF_PARAM_GRID, X_train, y_train, tscv)
    rf_best = rf_search.best_estimator_  # already refit on all of X_train/y_train
    rf_pred = rf_best.predict(X_test)

    # --- XGBoost: search unweighted, then refit the winning config weighted ---
    log.info("Grid-searching xgboost%s (%d combos x %d CV splits)...",
              variant, np.prod([len(v) for v in XGB_PARAM_GRID.values()]), N_CV_SPLITS)
    xgb_pipe = _build_pipeline(XGBClassifier(
        objective="multi:softprob", num_class=3, random_state=42, n_jobs=1,
        eval_metric="mlogloss",
    ), categorical)
    xgb_search = _grid_search(xgb_pipe, XGB_PARAM_GRID, X_train, y_train, tscv)
    # Refit with the best params, this time with balanced sample weights, to
    # stay comparable with the weighted untuned XGBoost run.
    xgb_best = xgb_search.best_estimator_
    xgb_best.fit(X_train, y_train, model__sample_weight=sample_weight)
    xgb_pred = xgb_best.predict(X_test)

    results = {
        f"random_forest{variant}_tuned": (rf_search, rf_best, rf_pred),
        f"xgboost{variant}_tuned": (xgb_search, xgb_best, xgb_pred),
    }

    for name, (search, best_model, pred) in results.items():
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        metrics[name] = {
            "best_params": search.best_params_,
            "cv_macro_f1": float(search.best_score_),
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "classification_report": classification_report(
                y_test, pred, target_names=le.classes_, output_dict=True
            ),
        }
        log.info("%s: test accuracy=%.3f macro_f1=%.3f (CV macro_f1 during search=%.3f)",
                  name, acc, f1, search.best_score_)
        joblib.dump(best_model, model_dir / f"{name}.joblib")

    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Updated %s with tuned results", metrics_path)

    untuned_rf = metrics.get(f"random_forest{variant}", {})
    untuned_xgb = metrics.get(f"xgboost{variant}", {})
    if untuned_rf and untuned_xgb:
        log.info(
            "Tuning delta -- random_forest%s: macro_f1 %.3f -> %.3f | xgboost%s: macro_f1 %.3f -> %.3f",
            variant, untuned_rf.get("macro_f1", float("nan")),
            metrics[f"random_forest{variant}_tuned"]["macro_f1"],
            variant, untuned_xgb.get("macro_f1", float("nan")),
            metrics[f"xgboost{variant}_tuned"]["macro_f1"],
        )

    return metrics


if __name__ == "__main__":
    tune_and_evaluate()
