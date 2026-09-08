"""
Grid-search hyperparameters for the pre-match (recent-formation) variant of
the outcome predictor -- run_prematch_predictor.py's models, tuned the same
way run_tune_hyperparameters.py tunes the actual-formation ones (GridSearchCV
+ TimeSeriesSplit; see src/tune_hyperparameters.py's docstring for why).

Run run_prematch_predictor.py at least once first for a side-by-side
untuned-vs-tuned comparison -- this appends "_prematch_tuned"-suffixed
entries to outputs/outcome_model_metrics.json rather than starting fresh.

    python run_tune_prematch.py
"""

from __future__ import annotations

from pathlib import Path

import config
from src.outcome_predictor import FORMATION_COLS_PREMATCH
from src.tune_hyperparameters import tune_and_evaluate


def main() -> None:
    required = Path(config.PROCESSED_DIR) / "match_dataset.parquet"
    if not required.exists():
        raise SystemExit(
            f"{required} not found -- run `python run_pipeline.py` first."
        )
    tune_and_evaluate(formation_cols=FORMATION_COLS_PREMATCH, variant="_prematch")


if __name__ == "__main__":
    main()
