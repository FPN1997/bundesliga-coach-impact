"""
Experiment: does XGBoost's native categorical split support
(`enable_categorical=True` + pandas `category` dtype) beat one-hot encoding
for the formation columns?

Motivated by the tuning results in outcome_predictor.py/tune_hyperparameters.py/
tune_optuna.py: XGBoost never beat its untuned baseline under any tuning
method tried (GridSearchCV, Optuna, more CV folds), while Random Forest did.
One candidate explanation: one-hot encoding ~19 formation categories
produces a lot of sparse binary columns that a boosted tree has to
reconstruct a categorical split from across several splits, where XGBoost's
native categorical support can partition the categories directly in a
single split. This tests that hypothesis directly instead of assuming it.

Isolates exactly one variable to keep the comparison fair: same
hyperparameters as outcome_predictor.py's untuned XGBoost baseline
(n_estimators=300, max_depth=4, learning_rate=0.05), same time-based
split, same class-balanced sample weights -- only the encoding changes.

Train/test categorical columns are unioned before converting to pandas
`category` dtype, so a formation seen only in the test set doesn't
silently turn into an unmatched/NaN category at predict time (this would
otherwise be a real, easy-to-miss bug: pandas assigns category codes from
whatever categories are present when `.astype("category")` runs, and a
category absent from that set becomes NaN rather than raising).

Results append into outputs/outcome_model_metrics.json as
"xgboost[_prematch]_native_cat" keys, alongside the untuned/tuned ones.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config
from src.features import build_feature_table
from src.outcome_predictor import FORMATION_COLS_ACTUAL, NUMERIC, _split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _as_shared_categorical(train_col: pd.Series, test_col: pd.Series) -> tuple[pd.Series, pd.Series]:
    all_categories = pd.concat([train_col, test_col]).unique()
    return (
        pd.Categorical(train_col, categories=all_categories),
        pd.Categorical(test_col, categories=all_categories),
    )


def train_and_evaluate(formation_cols: list[str] = FORMATION_COLS_ACTUAL, variant: str = "") -> dict:
    categorical = [*formation_cols, "venue"]

    df = build_feature_table()
    train, test = _split(df)

    le = LabelEncoder()
    y_train = le.fit_transform(train["result"])
    y_test = le.transform(test["result"])

    X_train = train[[*categorical, *NUMERIC]].copy()
    X_test = test[[*categorical, *NUMERIC]].copy()
    for col in categorical:
        X_train[col], X_test[col] = _as_shared_categorical(X_train[col], X_test[col])

    sample_weight = compute_sample_weight("balanced", y_train)

    # Same hyperparameters as outcome_predictor.py's untuned XGBoost --
    # only the encoding differs, so any accuracy delta is attributable to
    # that, not to a coincidentally different model configuration.
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        objective="multi:softprob", num_class=3,
        enable_categorical=True, tree_method="hist",
        random_state=42, n_jobs=-1, eval_metric="mlogloss",
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    metrics_path = Path(config.OUTPUT_DIR) / "outcome_model_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    name = f"xgboost{variant}_native_cat"
    metrics[name] = {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "classification_report": classification_report(
            y_test, pred, target_names=le.classes_, output_dict=True
        ),
    }

    baseline_key = f"xgboost{variant}"
    baseline_f1 = metrics.get(baseline_key, {}).get("macro_f1")
    log.info("%s: accuracy=%.3f macro_f1=%.3f%s", name, acc, f1,
              f" (one-hot baseline: {baseline_f1:.3f})" if baseline_f1 is not None else "")

    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Saved metrics -> %s", metrics_path)

    model_dir = Path(config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{name}.joblib")
    joblib.dump(le, model_dir / f"label_encoder{variant}_native_cat.joblib")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
