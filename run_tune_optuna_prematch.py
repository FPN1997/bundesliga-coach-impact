"""
Wider Optuna hyperparameter search for the pre-match (recent-formation)
outcome predictor -- see src/tune_optuna.py's docstring.

    python run_tune_optuna_prematch.py
"""

from __future__ import annotations

from pathlib import Path

import config
from src.outcome_predictor import FORMATION_COLS_PREMATCH
from src.tune_optuna import tune_and_evaluate


def main() -> None:
    required = Path(config.PROCESSED_DIR) / "match_dataset.parquet"
    if not required.exists():
        raise SystemExit(
            f"{required} not found -- run `python run_pipeline.py` first."
        )
    tune_and_evaluate(formation_cols=FORMATION_COLS_PREMATCH, variant="_prematch")


if __name__ == "__main__":
    main()
