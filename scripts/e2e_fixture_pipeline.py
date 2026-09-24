"""
End-to-end smoke test: runs the REAL pipeline code (build_dataset.py,
coach_impact.py, formation_matrix.py, features.py, outcome_predictor.py --
everything except the network fetch itself) against a small, deterministic,
checked-in-as-code fixture dataset, and asserts each stage produces sane,
non-empty output without raising.

Why this exists: the unit tests in tests/ all use small synthetic
DataFrames scoped to one function each (leak-safety, the time-based split,
the overwrite guards, ...) -- real, valuable, but none of them actually
wire the pipeline stages together the way `bundesliga pipeline` does. A
schema mismatch between what fetch_fbref.py/fetch_understat.py actually
save and what build_dataset.py expects to read (exactly the kind of thing
a future soccerdata release could cause -- see README "Known rough edges")
would sail through every existing test and only surface the next time
someone actually runs the real pipeline. This closes that gap without the
cost/flakiness of a live scrape: every column name and dtype below is
copied from the REAL data/raw/*.parquet and data/processed/coach_history.csv
files' actual schemas (verified live, not guessed), so the two fixture
teams below exercise the exact same rename/merge/join code paths real data
goes through.

Everything runs against temp-directory config overrides (RAW_DIR,
PROCESSED_DIR, OUTPUT_DIR, MODEL_DIR, COACH_HISTORY_RESOLVED_CSV, SEASONS,
TEST_SEASONS) -- this never touches the real data/ or outputs/ directories,
so it's also safe to run locally, not just in CI.

Run directly:
    python scripts/e2e_fixture_pipeline.py

Wired into .github/workflows/e2e-fixture.yml as a SCHEDULED (not per-push)
job -- see that file for why not per-push.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

# This lives in scripts/, not the repo root like cli.py -- put the repo
# root on sys.path so `import config` / `from src...` resolve the same way
# they do for the root-level entry points, regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Deliberately not real season strings -- unambiguous, can never collide
# with a real value in config.SEASONS as that list changes over time.
FIXTURE_SEASONS = ["2001-2002", "2002-2003"]
FIXTURE_TEST_SEASONS = ["2002-2003"]
TEAMS = ["Fixture United", "Test Town"]
FORMATIONS = ["4-2-3-1", "4-3-3", "3-4-3"]

# (home_gf, away_gf) per matchday -- fixed, not random, so a failure is
# reproducible. Deliberately varied (wins/draws/losses both directions) so
# points/results aren't degenerate.
SEASON_SCORES = {
    "2001-2002": [(2, 1), (0, 0), (1, 3), (2, 2), (3, 0), (1, 1), (0, 2), (2, 0), (1, 1), (3, 1)],
    "2002-2003": [(1, 0), (2, 2), (0, 1), (3, 3), (1, 2), (2, 0), (0, 0), (1, 3), (2, 1), (0, 2)],
}
# One coaching change: Fixture United gets a new coach for the second
# season; Test Town keeps theirs throughout (not every team changes coach
# -- exercises compute_coach_impact() actually finding a change, without
# every team being a change).
COACHES = {
    ("Fixture United", "2001-2002"): "Ann Coachley",
    ("Fixture United", "2002-2003"): "Ben Managerson",
    ("Test Town", "2001-2002"): "Cara Gaffer",
    ("Test Town", "2002-2003"): "Cara Gaffer",
}


def _season_code(season_str: str) -> str:
    start, end = season_str.split("-")
    return start[2:] + end[2:]


def _points(gf: int, ga: int) -> int:
    return 3 if gf > ga else (1 if gf == ga else 0)


def _result(gf: int, ga: int) -> str:
    return "W" if gf > ga else ("D" if gf == ga else "L")


def _build_fbref_fixture() -> pd.DataFrame:
    """Columns/dtypes match the real data/raw/fbref_schedule.parquet schema
    (verified live) -- specifically the ones build_dataset.py's
    FBREF_RENAME + downstream code actually touch."""
    rows = []
    for season in FIXTURE_SEASONS:
        code = _season_code(season)
        scores = SEASON_SCORES[season]
        base_date = pd.Timestamp(f"{season[:4]}-08-01")
        for i, (home_gf, away_gf) in enumerate(scores):
            date = base_date + pd.Timedelta(weeks=i)
            home, away = (TEAMS[0], TEAMS[1]) if i % 2 == 0 else (TEAMS[1], TEAMS[0])
            home_formation = FORMATIONS[i % len(FORMATIONS)]
            away_formation = FORMATIONS[(i + 1) % len(FORMATIONS)]
            rows.append({
                "season": code, "team": home, "date": date, "round": f"Matchweek {i + 1}",
                "venue": "Home", "result": _result(home_gf, away_gf),
                "GF": float(home_gf), "GA": float(away_gf), "opponent": away,
                "Poss": 50, "Captain": "Player A", "Formation": home_formation,
                "Opp Formation": away_formation, "Referee": "Ref One",
            })
            rows.append({
                "season": code, "team": away, "date": date, "round": f"Matchweek {i + 1}",
                "venue": "Away", "result": _result(away_gf, home_gf),
                "GF": float(away_gf), "GA": float(home_gf), "opponent": home,
                "Poss": 50, "Captain": "Player B", "Formation": away_formation,
                "Opp Formation": home_formation, "Referee": "Ref One",
            })
    return pd.DataFrame(rows)


def _build_understat_fixture() -> pd.DataFrame:
    """Columns match the real data/raw/understat_matches.parquet schema
    (verified live) -- xg/xga/ppda/deep_completions loosely track the goals
    so the numbers are plausible, not that the pipeline cares."""
    rows = []
    for season in FIXTURE_SEASONS:
        code = _season_code(season)
        scores = SEASON_SCORES[season]
        base_date = pd.Timestamp(f"{season[:4]}-08-01")
        for i, (home_gf, away_gf) in enumerate(scores):
            date = base_date + pd.Timedelta(weeks=i)
            home, away = (TEAMS[0], TEAMS[1]) if i % 2 == 0 else (TEAMS[1], TEAMS[0])
            rows.append({
                "season": code, "team": home, "date": date, "opponent": away,
                "points": _points(home_gf, away_gf),
                "xg": home_gf + 0.3, "xga": away_gf + 0.2,
                "ppda": 8.0 + i % 4, "deep_completions": 10 + i % 5,
            })
            rows.append({
                "season": code, "team": away, "date": date, "opponent": home,
                "points": _points(away_gf, home_gf),
                "xg": away_gf + 0.2, "xga": home_gf + 0.3,
                "ppda": 9.0 + i % 4, "deep_completions": 8 + i % 5,
            })
    return pd.DataFrame(rows)


def _build_coach_history_fixture() -> pd.DataFrame:
    """Columns match the real data/processed/coach_history.csv schema."""
    rows = []
    for (team, season), coach in COACHES.items():
        start = pd.Timestamp(f"{season[:4]}-07-01")
        rows.append({"team": team, "coach": coach, "start_date": start, "end_date": pd.NaT})
    return pd.DataFrame(rows)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="e2e_fixture_") as tmp:
        tmp_path = Path(tmp)
        config.SEASONS = FIXTURE_SEASONS
        config.TEST_SEASONS = FIXTURE_TEST_SEASONS
        config.RAW_DIR = str(tmp_path / "raw")
        config.PROCESSED_DIR = str(tmp_path / "processed")
        config.OUTPUT_DIR = str(tmp_path / "outputs")
        config.MODEL_DIR = str(tmp_path / "outputs" / "models")
        config.COACH_HISTORY_RESOLVED_CSV = str(tmp_path / "processed" / "coach_history.csv")

        for d in (config.RAW_DIR, config.PROCESSED_DIR, config.OUTPUT_DIR, config.MODEL_DIR):
            Path(d).mkdir(parents=True, exist_ok=True)

        _build_fbref_fixture().to_parquet(Path(config.RAW_DIR) / "fbref_schedule.parquet", index=False)
        _build_understat_fixture().to_parquet(
            Path(config.RAW_DIR) / "understat_matches.parquet", index=False
        )
        _build_coach_history_fixture().to_csv(config.COACH_HISTORY_RESOLVED_CSV, index=False)

        # Imported here, after the config overrides above -- these modules
        # read config.* at call time (verified: they do `config.RAW_DIR`
        # etc. inside function bodies, not at module import time), so
        # importing late isn't required for correctness, but keeps the
        # override block above visually self-contained regardless.
        from src.build_dataset import build_dataset
        from src.coach_change_effect import run as estimate_coach_change_effect
        from src.coach_impact import compute_coach_impact
        from src.features import build_feature_table
        from src.formation_matrix import build_formation_matrix, coach_preferred_formations
        from src.outcome_predictor import train_and_evaluate

        print("== build_dataset ==")
        matches = build_dataset()
        assert len(matches) == 40, f"expected 40 merged rows (2 teams x 20 matches), got {len(matches)}"
        assert matches["coach"].notna().all(), "every fixture row should have a coach assigned"
        assert matches["xg"].notna().all(), (
            "every fixture row should join Understat xG (team names match by construction)"
        )

        print("== coach_impact ==")
        impact = compute_coach_impact()
        assert len(impact) == 1, f"expected exactly 1 coaching change (Fixture United), got {len(impact)}"
        assert impact.iloc[0]["team"] == "Fixture United"

        print("== coach_change_effect ==")
        # The fixture's single change is a summer appointment, so the
        # mid-season estimate must come back empty rather than crash.
        effect = estimate_coach_change_effect()
        assert effect["mid_season"]["ppg"]["n_treated"] == 0

        print("== formation_matrix ==")
        matrix = build_formation_matrix()
        assert not matrix.empty, "formation matchup matrix should be non-empty"
        preferred = coach_preferred_formations()
        assert not preferred.empty, "coach_preferred_formations should be non-empty"

        print("== features.build_feature_table ==")
        features = build_feature_table()
        assert not features.empty, "feature table should be non-empty"

        # train_and_evaluate() -> _split() raises RuntimeError itself if no
        # rows match config.TEST_SEASONS, which is the real, meaningful
        # check that the season-code plumbing (SEASONS -> compact code ->
        # TEST_SEASONS match) actually works end to end -- no need to
        # duplicate that check here.
        print("== outcome_predictor.train_and_evaluate ==")
        metrics = train_and_evaluate()
        assert "random_forest" in metrics and "xgboost" in metrics
        assert 0.0 <= metrics["random_forest"]["accuracy"] <= 1.0
        assert 0.0 <= metrics["xgboost"]["accuracy"] <= 1.0

        print("\nAll pipeline stages ran successfully against the fixture dataset.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # top-level smoke test -- any failure should print clearly and exit non-zero
        print(f"E2E FIXTURE PIPELINE FAILED: {exc}", file=sys.stderr)
        raise
