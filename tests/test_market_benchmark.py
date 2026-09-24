"""
Tests for src/market_benchmark.py's scoring and joining -- a benchmark is
only as trustworthy as its scoring rules and its match-up of forecasts to
odds, so both are checked against hand-computed values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import market_benchmark as mb


def test_rps_hand_computed_values():
    home_win = np.array([0])
    assert mb.rps(np.array([[1.0, 0.0, 0.0]]), home_win)[0] == pytest.approx(0.0)
    assert mb.rps(np.array([[0.0, 0.0, 1.0]]), home_win)[0] == pytest.approx(1.0)
    # uniform: cumulative (1/3, 2/3) vs (1, 1) -> ((2/3)^2 + (1/3)^2) / 2 = 5/18
    assert mb.rps(np.full((1, 3), 1 / 3), home_win)[0] == pytest.approx(5 / 18)


def test_rps_respects_outcome_order():
    # The reason RPS is the football standard: calling a home win a draw is
    # less wrong than calling it an away win. Log loss can't tell these apart.
    home_win = np.array([0])
    said_draw = mb.rps(np.array([[0.0, 1.0, 0.0]]), home_win)[0]
    said_away = mb.rps(np.array([[0.0, 0.0, 1.0]]), home_win)[0]
    assert said_draw < said_away


def test_implied_probabilities_remove_margin_and_leave_missing_odds_as_nan():
    odds = pd.DataFrame({
        "AvgCH": [2.0, 1.9, np.nan], "AvgCD": [4.0, 3.5, np.nan], "AvgCA": [4.0, 4.2, np.nan],
    })
    probs = mb.implied_probabilities(odds, "closing")
    np.testing.assert_allclose(probs[0], [0.5, 0.25, 0.25])  # 1/2, 1/4, 1/4 already sum to 1
    # a real bookmaker line (margin > 0): implied probabilities sum to > 1 until normalized
    assert (1 / odds.iloc[1]).sum() > 1
    np.testing.assert_allclose(probs[1].sum(), 1.0)
    assert np.isnan(probs[2]).all()  # missing odds must not silently become a forecast


def test_team_perspective_probabilities_map_to_home_draw_away():
    # LabelEncoder sorts classes alphabetically: D, L, W
    team_probs = np.array([[0.2, 0.3, 0.5]])
    hda = mb._team_probs_to_hda(team_probs, ["D", "L", "W"])
    np.testing.assert_allclose(hda, [[0.5, 0.2, 0.3]])


def _features_and_odds(ftr_for_second_match: str):
    features = pd.DataFrame({
        "season": ["2526", "2526", "2526"],
        "venue": ["Home", "Home", "Away"],
        "team": ["Alpha", "Beta", "Alpha"],
        "opponent": ["Beta", "Alpha", "Beta"],
        "result": ["W", "D", "L"],
    })
    odds = pd.DataFrame({
        "season": ["2526", "2526"],
        "home_team": ["Alpha", "Beta"],
        "away_team": ["Beta", "Alpha"],
        "FTR": ["H", ftr_for_second_match],
    })
    return features, odds


def test_build_test_frame_keeps_one_row_per_match_from_the_home_side(monkeypatch):
    monkeypatch.setattr(mb.config, "TEST_SEASONS", ["2025-2026"])
    features, odds = _features_and_odds("D")
    test = mb.build_test_frame(features, odds)
    assert len(test) == 2
    assert test["outcome"].tolist() == [0, 1]  # H, D


def test_build_test_frame_refuses_to_score_a_mismatched_join(monkeypatch):
    monkeypatch.setattr(mb.config, "TEST_SEASONS", ["2025-2026"])
    features, odds = _features_and_odds("A")  # football-data says away win, FBref says draw
    with pytest.raises(RuntimeError, match="disagree on the result"):
        mb.build_test_frame(features, odds)
