"""
End-to-end pipeline: fetch -> merge -> analyze.

    python run_pipeline.py
    python run_pipeline.py --skip-fetch     # reuse whatever's already in data/raw

Re-run individual src/*.py files directly while iterating -- e.g. after
tweaking a formation-column heuristic you only need `python -m src.formation_matrix`,
not a full re-scrape.
"""

from __future__ import annotations

import argparse
import logging

from src.build_dataset import build_dataset
from src.coach_impact import compute_coach_impact
from src.fetch_coach_history import fetch_coach_history
from src.fetch_fbref import fetch_fbref_matches
from src.fetch_understat import fetch_understat_matches
from src.formation_matrix import build_formation_matrix, coach_preferred_formations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip the scraping steps and reuse data/raw/*.parquet")
    args = parser.parse_args()

    if not args.skip_fetch:
        log.info("Step 1/5: FBref (results + formations)")
        fetch_fbref_matches()
        log.info("Step 2/5: Understat (xG + PPDA)")
        fetch_understat_matches()
        log.info("Step 3/5: Coach tenure history")
        fetch_coach_history()
    else:
        log.info("Skipping fetch steps (--skip-fetch)")

    log.info("Step 4/5: Merging into match_dataset.parquet")
    build_dataset()

    log.info("Step 5/5: Analysis")
    compute_coach_impact()
    build_formation_matrix()
    coach_preferred_formations()

    log.info("Done. Check outputs/ for CSVs + the heatmap PNG.")


if __name__ == "__main__":
    main()
