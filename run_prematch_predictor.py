"""
Train the genuine pre-match variant of the outcome predictor: each team's
recent MODAL formation (most common formation over its last
config.FEATURE_ROLLING_WINDOW matches, shift-safe) instead of the actual
formation fielded in the match. See src/features.py and
src/outcome_predictor.py's docstrings for why this distinction matters --
the default run_outcome_predictor.py trains an explanatory model that
"cheats" by using information (the exact matchday formation) a real
pre-match forecaster wouldn't have.

Run run_outcome_predictor.py at least once first for a side-by-side
comparison -- this appends "_prematch"-suffixed entries to
outputs/outcome_model_metrics.json rather than replacing the originals.

    python run_prematch_predictor.py
"""

from __future__ import annotations

from pathlib import Path

import config
from src.outcome_predictor import FORMATION_COLS_PREMATCH, train_and_evaluate


def main() -> None:
    required = Path(config.PROCESSED_DIR) / "match_dataset.parquet"
    if not required.exists():
        raise SystemExit(
            f"{required} not found -- run `python run_pipeline.py` first."
        )
    train_and_evaluate(formation_cols=FORMATION_COLS_PREMATCH, variant="_prematch")


if __name__ == "__main__":
    main()
