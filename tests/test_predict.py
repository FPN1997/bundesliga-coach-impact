"""
Tests for predict.py's current_team_state() -- the "as of right now"
feature computation that src/features.py's own leak-safety tests don't
cover (see README "Extending"). Only the pieces it's built from
(features._rolling_mode) were tested before this; this exercises the
function itself against hand-computed values.

Also a regression test for a real bug this test coverage found: the
previous recent_formation implementation reused features._rolling_mode(...)
directly and took its LAST element, which is shift-safe FOR A PAST ROW
(excludes that row's own formation) but is therefore one match STALE for
"the team's recent tendency right now" -- it silently drops the team's
most recent played match from the mode. Verified live against the real
dataset before fixing: 5 of 28 teams' reported recent_formation disagreed
between the old (stale) and corrected computation, including cases where a
team's real formation switch in its last match was invisible in the old
output. _recent_formation() is the fix; test_recent_formation_includes_the_
most_recent_played_match is the regression test for it.
"""

from __future__ import annotations

import pandas as pd
import pytest

import predict

ROLLING_WINDOW = predict.ROLLING_WINDOW  # 5, from config.FEATURE_ROLLING_WINDOW


def _make_matches(n: int, team: str = "Alpha", season: str = "2526") -> pd.DataFrame:
    """n matches for `team`, dated weekly, with distinct hand-computable
    values in every stat column (all equal to the 1-indexed match number,
    same pattern as test_features.py's _make_team_frame)."""
    dates = pd.date_range("2025-08-01", periods=n, freq="7D")
    vals = list(range(1, n + 1))
    return pd.DataFrame({
        "team": team, "date": dates, "season": season,
        "points": vals, "gf": vals, "ga": [0] * n,
        "xg": [float(v) for v in vals], "xga": [0.0] * n,
        "ppda": [float(v) for v in vals], "deep_completions": vals,
        "formation": ["4-2-3-1"] * n,
    })


def _make_coaches(team: str = "Alpha", coach: str = "Coach X", start: str = "2025-01-01"):
    return pd.DataFrame({"team": [team], "coach": [coach], "start_date": [pd.Timestamp(start)]})


def test_form_stats_are_the_mean_of_the_last_rolling_window_played_matches():
    n = 2 * ROLLING_WINDOW
    matches = _make_matches(n)
    coaches = _make_coaches()

    state = predict.current_team_state("Alpha", matches, coaches)

    # vals == 1..n, so the last ROLLING_WINDOW values are (n-window+1)..n.
    expected_vals = list(range(n - ROLLING_WINDOW + 1, n + 1))
    expected_mean = sum(expected_vals) / ROLLING_WINDOW
    assert state["form_ppg"] == pytest.approx(expected_mean)
    assert state["form_goal_diff"] == pytest.approx(expected_mean)  # gf-ga == gf here
    assert state["form_xg_diff"] == pytest.approx(expected_mean)
    assert state["form_ppda"] == pytest.approx(expected_mean)
    assert state["form_deep_completions"] == pytest.approx(expected_mean)


def test_season_ppg_to_date_includes_the_most_recent_match():
    # Unlike features.py's shift-safe season_ppg_to_date (which excludes the
    # match being featurized), current_team_state() has no future match to
    # shift away from -- "as of right now" correctly means every match
    # played this season so far, the most recent one included.
    n = ROLLING_WINDOW  # exactly the minimum -- exercises the boundary, not just comfortably above it
    matches = _make_matches(n)  # points 1..n, all one season
    coaches = _make_coaches()

    state = predict.current_team_state("Alpha", matches, coaches)

    assert state["season_ppg_to_date"] == pytest.approx(sum(range(1, n + 1)) / n)


def test_too_few_matches_exits_rather_than_computing_garbage():
    matches = _make_matches(ROLLING_WINDOW - 1)
    coaches = _make_coaches()

    with pytest.raises(SystemExit, match="need at least"):
        predict.current_team_state("Alpha", matches, coaches)


def test_coach_tenure_days_uses_the_latest_started_tenure():
    matches = _make_matches(ROLLING_WINDOW)
    coaches = pd.concat([
        _make_coaches(coach="Old Coach", start="2020-01-01"),
        _make_coaches(coach="Coach X", start="2025-01-01"),
    ], ignore_index=True)

    state = predict.current_team_state("Alpha", matches, coaches)

    assert state["coach"] == "Coach X"
    assert state["coach_tenure_days"] == (pd.Timestamp.today() - pd.Timestamp("2025-01-01")).days


def test_recent_formation_includes_the_most_recent_played_match():
    # Regression test for the staleness bug described in the module
    # docstring: 8 matches, a formation switch on the very LAST match. The
    # old _rolling_mode(...)[-1] implementation would report "A" (the mode
    # of the 5 matches BEFORE the last one); the correct "recent tendency
    # right now" is "B", since 3 of the last 5 played matches are "B".
    matches = _make_matches(8)
    matches["formation"] = ["A", "A", "A", "A", "A", "B", "B", "B"]
    coaches = _make_coaches()

    state = predict.current_team_state("Alpha", matches, coaches)

    assert state["recent_formation"] == "B"


def test_recent_formation_helper_ignores_none_and_uses_the_tail_window():
    formations = ["A", None, "A", "B", "B", "B"]
    assert predict._recent_formation(formations, window=3) == "B"


def test_recent_formation_helper_returns_none_when_nothing_played():
    assert predict._recent_formation([None, None], window=3) is None


def _state(team: str, tenure):
    return {"team": team, "form_ppg": 1.0, "form_goal_diff": 0.5, "form_xg_diff": 0.2,
            "form_ppda": 10.0, "form_deep_completions": 6.0, "season_ppg_to_date": 1.5,
            "coach_tenure_days": tenure, "recent_formation": "4-2-3-1"}


def test_feature_row_prematch_has_every_model_column_and_no_missing_values():
    from src.outcome_predictor import FORMATION_COLS_PREMATCH, NUMERIC
    row = predict.feature_row(_state("A", None), _state("B", 30), "home")
    assert set(row) == {*NUMERIC, *FORMATION_COLS_PREMATCH, "venue"}
    assert row["coach_tenure_days"] == 0      # unknown coach -> 0, not None
    assert row["opp_coach_tenure_days"] == 30
    assert row["venue"] == "Home"


def test_feature_row_explanatory_uses_given_formations_not_recent_ones():
    row = predict.feature_row(_state("A", 1), _state("B", 1), "away", "4-3-3", "3-5-2")
    assert (row["formation"], row["opp_formation"], row["venue"]) == ("4-3-3", "3-5-2", "Away")
    assert "recent_formation" not in row
