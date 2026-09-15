"""
Re-run GridSearchCV hyperparameter tuning (both formation variants) with an
embargoed TimeSeriesSplit -- config.CV_EMBARGO_GAP rows excluded between
each fold's train and validation slice, on top of the existing
non-overlapping split. See src/tune_hyperparameters.py's tune_and_evaluate()
docstring for why: a validation fold's earliest rows are otherwise
immediately time-adjacent to the training rows just before them, so their
rolling-form features are highly serially correlated with what the model
just trained on -- optimistic validation scores that don't generalize as
well to the real, less-correlated held-out test set.

Motivated directly by the CV-vs-test gap documented in the README's
"Hyperparameter tuning" section: raising TimeSeriesSplit's fold count from
5 to 10 fixed that gap for Random Forest but not XGBoost. This is the
"untried lever ... specific to why it plateaued" flagged there.

Run run_tune_hyperparameters.py first -- this reads and updates
outputs/outcome_model_metrics.json, writing "<model>[_prematch]_tuned_embargoed"
keys alongside the existing "_tuned" (no-gap) ones so both are visible
side by side, not overwriting them.

    python run_tune_embargoed.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import config
from src.outcome_predictor import FORMATION_COLS_ACTUAL, FORMATION_COLS_PREMATCH
from src.tune_hyperparameters import tune_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    required = Path(config.PROCESSED_DIR) / "match_dataset.parquet"
    if not required.exists():
        raise SystemExit(
            f"{required} not found -- run `python run_pipeline.py` first."
        )
    log.info("Actual-formation variant (embargo=%d rows)", config.CV_EMBARGO_GAP)
    tune_and_evaluate(formation_cols=FORMATION_COLS_ACTUAL, variant="", gap=config.CV_EMBARGO_GAP)
    log.info("Pre-match variant (embargo=%d rows)", config.CV_EMBARGO_GAP)
    tune_and_evaluate(formation_cols=FORMATION_COLS_PREMATCH, variant="_prematch",
                       gap=config.CV_EMBARGO_GAP)


if __name__ == "__main__":
    main()
