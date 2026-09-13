"""
Wider hyperparameter search via Optuna, on top of src/tune_hyperparameters.py's
GridSearchCV attempt (12-16 combinations per model). This searches a much
bigger space -- more hyperparameters, continuous/log-scale ranges instead of
a handful of fixed values -- using TPE (Bayesian) sampling instead of
exhaustive enumeration, so it can cover that space in a fixed trial budget
rather than needing combinatorially more fits.

Same methodology as the GridSearchCV version, for a fair comparison against
it and against the untuned baseline:

  - `TimeSeriesSplit`, not random K-fold, for CV -- same leakage reasoning
    as tune_hyperparameters.py's docstring.
  - Class-balance weighting decoupled from the search: XGBoost is searched
    unweighted (scoring on macro-F1, which already penalizes ignoring
    draws), then the WINNING config is refit with balanced sample weights.
    Random Forest gets class_weight="balanced" baked in throughout, since
    it's a constructor arg with no routing complexity.

Prior finding (see README "Hyperparameter tuning"): a 12-16-combination
GridSearchCV made every model WORSE on the real held-out test set, with the
CV score during search consistently running below the eventual test score
-- pointing at TimeSeriesSplit's early, data-starved folds being a noisier
training signal than the final large evaluation, not at the grid being too
narrow. A wider search does not fix that gap; it searches harder against
the same noisy signal. Expect this to reproduce that finding rather than
overturn it -- and report whichever actually happens.

Results append into outputs/outcome_model_metrics.json as "<model>_optuna"
keys, alongside the untuned and GridSearchCV-tuned ones. Models save to
outputs/models/<name>_optuna.joblib.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config
from src.features import build_feature_table
from src.outcome_predictor import FORMATION_COLS_ACTUAL, NUMERIC, _build_pipeline, _split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # trial-by-trial spam off; we log the summary

N_CV_SPLITS = config.N_CV_SPLITS
N_TRIALS = 50


def _rf_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_categorical(
            "max_depth", [None, 4, 6, 8, 10, 12, 16, 20, 25]
        ),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None, 0.3, 0.5, 0.7]
        ),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
    }


def _xgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _run_study(
    model_name: str,
    param_fn,
    build_model,
    categorical: list[str],
    X, y,
    tscv: TimeSeriesSplit,
    n_trials: int = N_TRIALS,
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = param_fn(trial)
        model = build_model(params)
        pipe = _build_pipeline(model, categorical)
        # n_jobs=1, deliberately: nesting joblib parallelism here (across
        # CV folds) on top of the model's own internal parallelism has a
        # habit of silently hanging via fork-safety issues on macOS. Model
        # fits are fast enough on this dataset size that sequential CV
        # folds are still quick -- reliability over a bit of extra speed.
        scores = cross_val_score(pipe, X, y, cv=tscv, scoring="f1_macro", n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
    )
    log.info("Optuna: searching %s (%d trials x %d CV splits)...",
              model_name, n_trials, N_CV_SPLITS)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("%s: best CV macro-F1=%.3f with params: %s",
              model_name, study.best_value, study.best_params)
    return study


def tune_and_evaluate(
    formation_cols: list[str] = FORMATION_COLS_ACTUAL,
    variant: str = "",
    n_trials: int = N_TRIALS,
) -> dict:
    categorical = formation_cols + ["venue"]

    df = build_feature_table()
    train, test = _split(df)
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

    # --- Random Forest ---
    rf_study = _run_study(
        f"random_forest{variant}", _rf_params,
        lambda p: RandomForestClassifier(**p, class_weight="balanced", random_state=42, n_jobs=1),
        categorical, X_train, y_train, tscv, n_trials,
    )
    rf_best = _build_pipeline(
        RandomForestClassifier(**rf_study.best_params, class_weight="balanced",
                                random_state=42, n_jobs=-1),
        categorical,
    )
    rf_best.fit(X_train, y_train)
    rf_pred = rf_best.predict(X_test)

    # --- XGBoost: search unweighted, refit winning config weighted ---
    xgb_study = _run_study(
        f"xgboost{variant}", _xgb_params,
        lambda p: XGBClassifier(**p, objective="multi:softprob", num_class=3,
                                 random_state=42, n_jobs=1, eval_metric="mlogloss"),
        categorical, X_train, y_train, tscv, n_trials,
    )
    xgb_best = _build_pipeline(
        XGBClassifier(**xgb_study.best_params, objective="multi:softprob", num_class=3,
                       random_state=42, n_jobs=-1, eval_metric="mlogloss"),
        categorical,
    )
    xgb_best.fit(X_train, y_train, model__sample_weight=sample_weight)
    xgb_pred = xgb_best.predict(X_test)

    results = {
        f"random_forest{variant}_optuna": (rf_study, rf_best, rf_pred),
        f"xgboost{variant}_optuna": (xgb_study, xgb_best, xgb_pred),
    }

    for name, (study, best_model, pred) in results.items():
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        metrics[name] = {
            "best_params": study.best_params,
            "cv_macro_f1": float(study.best_value),
            "n_trials": len(study.trials),
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "classification_report": classification_report(
                y_test, pred, target_names=le.classes_, output_dict=True
            ),
        }
        log.info("%s: test accuracy=%.3f macro_f1=%.3f (CV macro_f1 during search=%.3f)",
                  name, acc, f1, study.best_value)
        joblib.dump(best_model, model_dir / f"{name}.joblib")

    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Updated %s with Optuna-tuned results", metrics_path)

    untuned_rf = metrics.get(f"random_forest{variant}", {})
    untuned_xgb = metrics.get(f"xgboost{variant}", {})
    gridsearch_rf = metrics.get(f"random_forest{variant}_tuned", {})
    gridsearch_xgb = metrics.get(f"xgboost{variant}_tuned", {})
    log.info(
        "random_forest%s macro_f1 -- untuned=%.3f grid=%.3f optuna=%.3f",
        variant, untuned_rf.get("macro_f1", float("nan")),
        gridsearch_rf.get("macro_f1", float("nan")),
        metrics[f"random_forest{variant}_optuna"]["macro_f1"],
    )
    log.info(
        "xgboost%s macro_f1 -- untuned=%.3f grid=%.3f optuna=%.3f",
        variant, untuned_xgb.get("macro_f1", float("nan")),
        gridsearch_xgb.get("macro_f1", float("nan")),
        metrics[f"xgboost{variant}_optuna"]["macro_f1"],
    )

    return metrics


if __name__ == "__main__":
    tune_and_evaluate()
