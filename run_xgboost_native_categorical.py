"""
Run the XGBoost native-categorical-vs-one-hot experiment for both formation
variants -- see src/xgboost_native_categorical.py's docstring for the
hypothesis being tested and why it isolates encoding as the only variable.

    python run_xgboost_native_categorical.py

Spoiler (see README "Formation-aware outcome predictor"): the hypothesis
didn't hold. One-hot wins both variants, clearly on actual-formation.
"""

from __future__ import annotations

from pathlib import Path

import config
from src.outcome_predictor import FORMATION_COLS_PREMATCH
from src.xgboost_native_categorical import train_and_evaluate


def main() -> None:
    required = Path(config.PROCESSED_DIR) / "match_dataset.parquet"
    if not required.exists():
        raise SystemExit(f"{required} not found -- run `python run_pipeline.py` first.")
    train_and_evaluate()
    train_and_evaluate(formation_cols=FORMATION_COLS_PREMATCH, variant="_prematch")


if __name__ == "__main__":
    main()
