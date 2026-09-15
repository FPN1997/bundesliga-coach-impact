"""
Train and evaluate the "new coach bounce" predictor: given a team's
pre-change form and the incoming coach's own historical track record
elsewhere in the dataset, can the size of the resulting PPG swing be
predicted? See src/coach_bounce.py's module docstring for the full
rationale and the honest small-sample caveat up front.

Run run_pipeline.py first (needs outputs/coach_impact_rankings.csv and
data/processed/match_dataset.parquet).

    python run_coach_bounce.py

Outputs:
  - data/processed/coach_bounce_features.parquet -- the joined feature table
  - outputs/coach_bounce_metrics.json             -- leave-one-out R2/MAE vs
    a trivial "predict the training-fold mean" baseline
  - outputs/coach_bounce_predicted_vs_actual.png
  - outputs/models/coach_bounce_ridge.joblib
"""

from __future__ import annotations

from pathlib import Path

import config
from src.coach_bounce import train_and_evaluate


def main() -> None:
    required = [
        Path(config.PROCESSED_DIR) / "match_dataset.parquet",
        Path(config.OUTPUT_DIR) / "coach_impact_rankings.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing {[str(p) for p in missing]} -- run `python run_pipeline.py` first."
        )
    train_and_evaluate()


if __name__ == "__main__":
    main()
