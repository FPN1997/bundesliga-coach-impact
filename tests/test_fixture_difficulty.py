"""
Tests for src/fixture_difficulty.py and its use in the coaching-change
estimate: a fixture's rating must depend only on the opponent and venue,
and adjusting for it must remove a planted "easier fixtures" confound.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src import coach_change_effect as cce
from src import fixture_difficulty as fd

STRENGTH = {"Strong": 2.0, "Good": 1.0, "Average": 0.0, "Weak": -1.0}  # log-odds scale


def _synthetic_odds(season: str = "2324") -> pd.DataFrame:
    """Every pairing home and away, with odds implied by team strength plus a
    home advantage -- no bookmaker margin, so the numbers are easy to reason about."""
    rows = []
    for home, away in itertools.permutations(STRENGTH, 2):
        diff = STRENGTH[home] - STRENGTH[away] + 0.4
        p_home = 1 / (1 + np.exp(-diff)) * 0.75
        p_away = (1 - 1 / (1 + np.exp(-diff))) * 0.75
        p_draw = 1 - p_home - p_away
        rows.append({"season": season, "home_team": home, "away_team": away,
                     "AvgCH": 1 / p_home, "AvgCD": 1 / p_draw, "AvgCA": 1 / p_away})
    return pd.DataFrame(rows)


def test_ratings_order_teams_by_strength():
    ratings, _ = fd.fit_ease_model(_synthetic_odds())
    r = ratings.xs("2324")
    assert r["Strong"] > r["Good"] > r["Average"] > r["Weak"]


def test_ease_depends_on_opponent_and_venue_not_on_the_team():
    odds = _synthetic_odds()
    matches = pd.DataFrame({
        "season": "2324",
        "team": ["Good", "Weak", "Good", "Good"],
        "opponent": ["Strong", "Strong", "Weak", "Weak"],
        "venue": ["Home", "Home", "Home", "Away"],
    })
    ease = fd.fixture_ease(matches, odds)
    assert ease[0] == pytest.approx(ease[1])  # same fixture, different team -> same ease
    assert ease[2] > ease[0]                  # a weak opponent is an easier fixture
    assert ease[2] > ease[3]                  # home is easier than away


def test_opponent_without_a_rating_gives_nan():
    matches = pd.DataFrame({"season": ["2324"], "team": ["Good"], "opponent": ["Unknown FC"],
                            "venue": ["Home"]})
    assert np.isnan(fd.fixture_ease(matches, _synthetic_odds())[0])


def test_adjusting_for_fixtures_removes_an_easier_fixtures_confound():
    # Controls: pure regression to the mean, plus +1 PPG per unit of easier
    # fixtures. Sacked teams get the same rules plus a planted effect of 0.2
    # -- but also systematically easier fixtures afterwards (+0.3), which an
    # estimate that ignores fixtures would wrongly credit to the change.
    rng = np.random.default_rng(0)
    rows = []
    for t in range(12):
        for kind, n, fixture_shift, effect in [("control", 60, 0.0, 0.0), ("treated", 3, 0.3, 0.2)]:
            for _ in range(n):
                before = rng.uniform(0, 3) if kind == "control" else rng.uniform(0, 0.8)
                fixture_change = rng.normal(fixture_shift, 0.2)
                rows.append({"team": f"T{t}", "kind": kind, "ppg_before": before,
                             "xgd_before": rng.normal(0, 0.5), "fixture_change": fixture_change,
                             "ppg_after": 1.3 + fixture_change + effect + rng.normal(0, 0.02),
                             "xgd_after": 0.0})
    w = pd.DataFrame(rows)

    naive = cce._AdjustedEstimator(w, "ppg").estimate()["effect"]
    adjusted = cce._AdjustedEstimator(
        w, "ppg", (*cce._AdjustedEstimator.BASE_COVARIATES, "fixture_change")).estimate()["effect"]

    assert naive == pytest.approx(0.5, abs=0.05)     # 0.2 planted + 0.3 from easier fixtures
    assert adjusted == pytest.approx(0.2, abs=0.03)  # only the planted effect remains
