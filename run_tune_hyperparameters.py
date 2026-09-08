"""
Grid-search hyperparameters for the outcome predictor and compare against
the untuned baseline from run_outcome_predictor.py.

Run run_outcome_predictor.py at least once first -- this reads and updates
outputs/outcome_model_metrics.json rather than starting from scratch, so
the untuned numbers stay there for comparison.

    python run_tune_hyperparameters.py

This can take several minutes: two GridSearchCV runs (Random Forest,
XGBoost) each fitting ~12-16 hyperparameter combinations across 5
TimeSeriesSplit folds.
"""

from __future__ import annotations

from pathlib import Path

import config
from src.tune_hyperparameters import tune_and_evaluate


def main() -> None:
    required = Path(config.PROCESSED_DIR) / "match_dataset.parquet"
    if not required.exists():
        raise SystemExit(
            f"{required} not found -- run `python run_pipeline.py` first."
        )
    tune_and_evaluate()


if __name__ == "__main__":
    main()
