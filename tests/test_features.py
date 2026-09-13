"""
Tests for the leak-safety properties in src/features.py -- this is the
single most important correctness guarantee in the whole project: a match's
features must depend only on that team's matches strictly BEFORE it. Every
test here is built to fail loudly if a future edit removes a `.shift(1)`
or otherwise lets a row see its own outcome (or a later one).

All tests use small synthetic DataFrames, not the live scraped dataset --
deterministic, no network access, and independent of however much data
happens to be sitting in data/ on a given machine.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import features


def test_rolling_mode_returns_none_until_window_is_full():
    values = ["A", "A", "B", "A", "B"]
    result = features._rolling_mode(values, window=2)
    assert result[0] is None
    assert result[1] is None
    assert all(r is not None for r in result[2:])


def test_rolling_mode_excludes_the_current_value():
    # Precise trace (window=2):
    #   i=0,1: not enough history yet -> None
    #   i=2: history=[A,A] (both from i=0,1)         -> mode "A"
    #   i=3: history=[A,B] (i=1,2 -- i=2's own "B"
    #        has NOT been folded in yet)              -> tie, "A" first-seen
    #   i=4: history=[B,A] (i=2,3)                     -> tie, "B" first-seen
    # If the current row's own value leaked into its own history, i=2's
    # result would be pulled toward "B" and i=3's toward "A" a step early --
    # this test would catch that regression.
    values = ["A", "A", "B", "A", "B"]
    result = features._rolling_mode(values, window=2)
    assert result == [None, None, "A", "A", "B"]


def _make_team_frame(n: int, season: str = "2324") -> pd.DataFrame:
    """n rows of made-up but internally-consistent match stats for one team,
    sorted by date, with a distinct, easy-to-hand-check value in every
    stat-bearing column."""
    dates = pd.date_range("2023-08-01", periods=n, freq="7D")
    points = [0, 1, 3] * (n // 3 + 1)
    return pd.DataFrame({
        "date": dates,
        "season": season,
        "points": points[:n],
        "gf": list(range(n)),
        "ga": [0] * n,  # goal_diff == gf, so goal_diff == row index
        "xg": [float(i) for i in range(n)],
        "xga": [0.0] * n,  # xg_diff == xg == row index
        "ppda": [10.0 + i for i in range(n)],
        "formation": (["4-2-3-1", "4-3-3"] * (n // 2 + 1))[:n],
    })


def test_one_team_features_form_ppg_uses_only_strictly_prior_rows():
    window = features.ROLLING_WINDOW
    df = _make_team_frame(2 * window + 2)

    out = features._one_team_features(df)

    for i in range(len(df)):
        if i < window:
            assert pd.isna(out["form_ppg"].iloc[i]), f"row {i} should have no form yet"
        else:
            expected = sum(df["points"].iloc[i - window:i]) / window
            assert out["form_ppg"].iloc[i] == pytest.approx(expected), (
                f"row {i}: form_ppg must be the mean of rows {i - window}..{i - 1}, "
                f"not including row {i} itself"
            )


def test_one_team_features_goal_diff_and_xg_diff_are_shift_safe():
    window = features.ROLLING_WINDOW
    df = _make_team_frame(2 * window + 2)
    out = features._one_team_features(df)

    # gf-ga == row index by construction, so form_goal_diff at row i is the
    # mean of indices [i-window, i), i.e. NOT including i.
    for i in range(window, len(df)):
        expected_gd = sum(range(i - window, i)) / window
        assert out["form_goal_diff"].iloc[i] == pytest.approx(expected_gd)
        expected_xgd = expected_gd  # xg-xga constructed identically to gf-ga
        assert out["form_xg_diff"].iloc[i] == pytest.approx(expected_xgd)


def test_one_team_features_season_ppg_excludes_current_match():
    df = _make_team_frame(6)
    out = features._one_team_features(df)

    # Row 0 of a season has no prior matches -> NaN, not 0 and not its own points.
    assert pd.isna(out["season_ppg_to_date"].iloc[0])
    # Row i's season-to-date PPG is the mean of points[0:i], excluding i.
    for i in range(1, len(df)):
        expected = df["points"].iloc[:i].mean()
        assert out["season_ppg_to_date"].iloc[i] == pytest.approx(expected)


def test_one_team_features_season_ppg_resets_at_season_boundary():
    first = _make_team_frame(4, season="2223")
    second = _make_team_frame(4, season="2324")
    second["date"] = pd.date_range("2024-08-01", periods=4, freq="7D")
    df = pd.concat([first, second], ignore_index=True)

    out = features._one_team_features(df)

    # First match of the new season must not see the old season's points.
    new_season_start = 4
    assert pd.isna(out["season_ppg_to_date"].iloc[new_season_start])


def test_team_rolling_features_does_not_leak_across_teams():
    # Two teams with deliberately different point sequences, interleaved by
    # date, run through the full per-team groupby. A bug that failed to
    # group by team before rolling would blend the two teams' histories.
    team_a = _make_team_frame(6)
    team_a["team"] = "Alpha"
    team_a["points"] = [3, 3, 3, 3, 3, 3]  # always wins

    team_b = _make_team_frame(6)
    team_b["team"] = "Beta"
    team_b["points"] = [0, 0, 0, 0, 0, 0]  # always loses

    df = pd.concat([team_a, team_b], ignore_index=True)
    out = features._team_rolling_features(df)

    window = features.ROLLING_WINDOW
    alpha_rows = out[out["team"] == "Alpha"].reset_index(drop=True)
    beta_rows = out[out["team"] == "Beta"].reset_index(drop=True)
    if window < len(team_a):
        assert alpha_rows["form_ppg"].iloc[window] == pytest.approx(3.0)
        assert beta_rows["form_ppg"].iloc[window] == pytest.approx(0.0)


def test_attach_coach_tenure_computes_days_since_start(monkeypatch, tmp_path):
    coach_csv = tmp_path / "coach_history.csv"
    pd.DataFrame({
        "team": ["Alpha"],
        "coach": ["Coach X"],
        "start_date": ["2023-01-01"],
        "end_date": [""],
    }).to_csv(coach_csv, index=False)
    monkeypatch.setattr(features.config, "COACH_HISTORY_RESOLVED_CSV", str(coach_csv))

    df = pd.DataFrame({
        "team": ["Alpha", "Alpha"],
        "coach": ["Coach X", "Coach X"],
        "date": pd.to_datetime(["2023-01-01", "2023-01-11"]),
    })

    out = features._attach_coach_tenure(df)

    assert out["coach_tenure_days"].iloc[0] == 0
    assert out["coach_tenure_days"].iloc[1] == 10


def test_attach_coach_tenure_missing_history_file_is_all_nan(monkeypatch, tmp_path):
    monkeypatch.setattr(features.config, "COACH_HISTORY_RESOLVED_CSV",
                         str(tmp_path / "does_not_exist.csv"))
    df = pd.DataFrame({
        "team": ["Alpha"], "coach": ["Coach X"], "date": pd.to_datetime(["2023-01-01"]),
    })
    out = features._attach_coach_tenure(df)
    assert out["coach_tenure_days"].isna().all()
