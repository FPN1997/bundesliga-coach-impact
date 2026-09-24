"""
Tests for src/coach_change_effect.py: which windows count as a sacking vs. a
control, and whether the regression-adjusted estimator recovers an effect
planted in synthetic data (and reports ~0 when there is none -- the case
where a raw before/after comparison would wrongly show a big "bounce").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import coach_change_effect as cce

W = 3  # small window keeps the hand-traced cases readable


def _team(coaches: list[str], team: str = "Alpha", seasons: list[str] | None = None) -> pd.DataFrame:
    n = len(coaches)
    return pd.DataFrame({
        "team": team,
        "date": pd.date_range("2023-08-01", periods=n, freq="7D"),
        "season": seasons or ["2324"] * n,
        "coach": coaches,
        "points": [1.0] * n,
        "xg": [1.0] * n,
        "xga": [1.0] * n,
    })


def test_single_change_is_one_treated_window_and_nearby_windows_are_excluded():
    df = _team(["A"] * 8 + ["B"] * 8)
    w = cce.build_windows(df, window=W)

    treated = w[w["kind"] == "treated"]
    assert len(treated) == 1
    assert treated.iloc[0]["date"] == df["date"].iloc[8]
    # Any window of 2*W matches that straddles the change but isn't centered on
    # it is neither treated nor a clean control -- it must be skipped.
    controls = w[w["kind"] == "control"]
    change_date = df["date"].iloc[8]
    for d in controls["date"]:
        i = df.index[df["date"] == d][0]
        assert i + W <= 8 or i - W >= 8, f"control window at {i} overlaps the change at 8"
    assert change_date not in set(controls["date"])


def test_change_with_another_change_in_the_before_half_is_not_treated():
    # Caretaker "C" for 1 match right before "B": at the C->B boundary the
    # before-half mixes A and C, so its "before form" isn't one coach's.
    df = _team(["A"] * 8 + ["C"] + ["B"] * 8)
    w = cce.build_windows(df, window=W)
    treated_dates = set(w.loc[w["kind"] == "treated", "date"])
    assert df["date"].iloc[8] in treated_dates      # A -> C: clean before-half
    assert df["date"].iloc[9] not in treated_dates  # C -> B: before-half is A,A,C


def test_window_spanning_two_seasons_is_flagged_off_season():
    seasons = ["2223"] * 8 + ["2324"] * 8
    df = _team(["A"] * 8 + ["B"] * 8, seasons=seasons)
    w = cce.build_windows(df, window=W)
    assert not w.loc[w["kind"] == "treated", "in_season"].iloc[0]


def _synthetic_windows(effect: float, n_teams: int = 12, seed: int = 0) -> pd.DataFrame:
    """Controls follow pure regression to the mean (change = 1.3 - ppg_before,
    plus noise); treated windows follow the same rule plus `effect`, and are
    drawn from bad runs only -- which is exactly why a raw before/after
    comparison overstates the effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_teams):
        for _ in range(60):
            before = rng.uniform(0, 3)
            rows.append({"team": f"T{t}", "kind": "control", "ppg_before": before,
                         "ppg_after": before + 1.3 - before + rng.normal(0, 0.05),
                         "xgd_before": rng.normal(0, 0.5), "xgd_after": 0.0})
        for _ in range(3):
            before = rng.uniform(0, 0.8)
            rows.append({"team": f"T{t}", "kind": "treated", "ppg_before": before,
                         "ppg_after": before + 1.3 - before + effect,
                         "xgd_before": rng.normal(0, 0.5), "xgd_after": 0.0})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("planted", [0.0, 0.5])
def test_estimator_recovers_planted_effect_not_the_raw_bounce(planted):
    est = cce._AdjustedEstimator(_synthetic_windows(planted), "ppg")
    r = est.estimate()
    assert r["effect"] == pytest.approx(planted, abs=0.03)
    # the raw bounce is large even when nothing was planted -- that's the
    # confound this module exists to remove
    assert r["raw_change"] > planted + 0.5


def test_bootstrap_ci_brackets_the_estimate_and_is_deterministic():
    est = cce._AdjustedEstimator(_synthetic_windows(0.5), "ppg")
    point = est.estimate()["effect"]
    a = est.bootstrap(n=200, seed=1)
    b = est.bootstrap(n=200, seed=1)
    assert a == b
    lo, hi = a["effect_ci95"]
    assert lo <= point <= hi
