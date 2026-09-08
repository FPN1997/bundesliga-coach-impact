"""
Train and evaluate the formation-aware match outcome predictor.

Requires match_dataset.parquet and coach_history.csv to already exist --
run run_pipeline.py first (or at least once, ever; this reuses whatever's
already in data/processed/).

    python run_outcome_predictor.py

Outputs land in outputs/: models/*.joblib, outcome_model_metrics.json,
outcome_confusion_matrix.png, outcome_feature_importance.png,
formation_matchup_predicted.csv/png.
"""

from __future__ import annotations

from pathlib import Path

import config
from src.outcome_predictor import train_and_evaluate


def main() -> None:
    required = Path(config.PROCESSED_DIR) / "match_dataset.parquet"
    if not required.exists():
        raise SystemExit(
            f"{required} not found -- run `python run_pipeline.py` first to "
            f"build the match dataset this predictor trains on."
        )
    train_and_evaluate()


if __name__ == "__main__":
    main()
