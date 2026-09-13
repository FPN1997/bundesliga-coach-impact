"""
Tests for the time-based train/test split in src/outcome_predictor.py --
the second core correctness guarantee in this project, alongside the
leak-safe features tested in test_features.py. A random split here would
leak future match results into training via the rolling features' shared
history and silently inflate every reported accuracy number.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import outcome_predictor as op


def test_season_code_matches_fbref_compact_format():
    assert op._season_code("2025-2026") == "2526"
    assert op._season_code("2019-2020") == "1920"
    assert op._season_code("2023-2024") == "2324"


def _make_league_frame() -> pd.DataFrame:
    """One row per (season, matchweek) across 4 seasons, in FBref's compact
    season-code format, with dates that increase monotonically with the
    season -- exactly the shape _split expects."""
    seasons = ["1920", "2021", "2122", "2223"]
    rows = []
    for season, year in zip(seasons, [2019, 2020, 2021, 2022], strict=True):
        for week in range(3):
            rows.append({
                "season": season,
                "date": pd.Timestamp(f"{year}-09-01") + pd.Timedelta(weeks=week),
                "result": "W",
            })
    return pd.DataFrame(rows)


def test_split_puts_only_configured_seasons_in_test(monkeypatch):
    monkeypatch.setattr(op.config, "TEST_SEASONS", ["2021-2022", "2022-2023"])
    df = _make_league_frame()

    train, test = op._split(df)

    assert set(test["season"]) == {"2122", "2223"}
    assert set(train["season"]) == {"1920", "2021"}
    assert len(train) + len(test) == len(df)


def test_split_never_puts_a_later_match_in_train_than_in_test(monkeypatch):
    # The whole reason this is a time-based split rather than a random one:
    # every training-set date must precede every test-set date. If a future
    # edit changed TEST_SEASONS to something non-contiguous with the rest
    # of the timeline, this is the check that would catch it.
    monkeypatch.setattr(op.config, "TEST_SEASONS", ["2021-2022", "2022-2023"])
    df = _make_league_frame()

    train, test = op._split(df)

    assert train["date"].max() < test["date"].min()


def test_split_raises_when_no_rows_match_test_seasons(monkeypatch):
    # Guards against a silent-empty-test-set bug: if TEST_SEASONS drifts out
    # of sync with whatever season codes are actually in the data (exactly
    # what happened during development -- see README), this must fail
    # loudly rather than quietly evaluating on zero rows.
    monkeypatch.setattr(op.config, "TEST_SEASONS", ["2099-2100"])
    df = _make_league_frame()

    with pytest.raises(RuntimeError, match="No rows matched TEST_SEASONS"):
        op._split(df)


def test_formation_column_sets_are_disjoint_and_correctly_named():
    assert op.FORMATION_COLS_ACTUAL == ["formation", "opp_formation"]
    assert op.FORMATION_COLS_PREMATCH == ["recent_formation", "opp_recent_formation"]
    assert set(op.FORMATION_COLS_ACTUAL).isdisjoint(op.FORMATION_COLS_PREMATCH)
