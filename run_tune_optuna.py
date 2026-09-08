"""
Wider Optuna hyperparameter search for the actual-formation outcome
predictor -- see src/tune_optuna.py's docstring for how this differs from
run_tune_hyperparameters.py's GridSearchCV attempt and what happened when
that one was tried.

Run run_outcome_predictor.py at least once first for a full untuned vs.
GridSearchCV vs. Optuna comparison -- this appends "_optuna"-suffixed
entries to outputs/outcome_model_metrics.json rather than starting fresh.

    python run_tune_optuna.py

50 Optuna trials x 2 models x 5 CV folds -- takes a few minutes.
"""

from __future__ import annotations

from pathlib import Path

import config
from src.tune_optuna import tune_and_evaluate


def main() -> None:
    required = Path(config.PROCESSED_DIR) / "match_dataset.parquet"
    if not required.exists():
        raise SystemExit(
            f"{required} not found -- run `python run_pipeline.py` first."
        )
    tune_and_evaluate()


if __name__ == "__main__":
    main()
