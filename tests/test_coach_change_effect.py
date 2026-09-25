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


@pytest.mark.parametrize("start,end,expected", [
    ("2023-10-07", "2023-12-16", False),  # autumn: squad frozen
    ("2023-11-25", "2024-01-20", True),   # runs into the January window
    ("2024-02-10", "2024-04-06", False),  # after the winter deadline
    ("2021-08-21", "2021-10-02", True),   # starts before the summer deadline
    ("2021-09-11", "2021-11-06", False),  # after it
    ("2020-09-19", "2020-11-07", True),   # 2020's summer window ran to 5 Oct (COVID)
])
def test_transfer_window_open(start, end, expected):
    assert cce.transfer_window_open(pd.Timestamp(start), pd.Timestamp(end)) is expected


def test_windows_record_whether_a_transfer_window_follows_the_change():
    # _team() dates are weekly from 2023-08-01; the change at index 8 falls
    # on 2023-09-26, and its 3-match after-half ends 2023-10-10: no window.
    df = _team(["A"] * 8 + ["B"] * 8)
    w = cce.build_windows(df, window=W)
    assert not w.loc[w["kind"] == "treated", "window_after"].iloc[0]


def test_windows_record_each_matchs_points_around_the_change():
    df = _team(["A"] * 8 + ["B"] * 8)
    df["points"] = [float(i % 4) for i in range(16)]
    w = cce.build_windows(df, window=W)
    t = w[w["kind"] == "treated"].iloc[0]
    # pts_-3 .. pts_-1 are the last W matches before the change (index 5..7),
    # pts_+0 .. pts_+2 the first W after it (index 8..10)
    assert [t[f"pts_{k:+d}"] for k in range(-W, W)] == list(df["points"].iloc[5:11])
    assert t["ppg_last2"] == pytest.approx(df["points"].iloc[6:8].mean())


def _event_windows(effect: float, seed: int = 0) -> pd.DataFrame:
    """Per-match points for `_synthetic_windows`-style data: controls regress
    to the mean, treated windows get `effect` on top after the change."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(12):
        for kind, n, top in (("control", 40, 3.0), ("treated", 3, 0.8)):
            for _ in range(n):
                before = rng.integers(0, 4, W).astype(float) * top / 3
                after = rng.normal(1.3, 0.3, W) + (effect if kind == "treated" else 0.0)
                rows.append({"team": f"T{t}", "kind": kind, "in_season": True,
                             "ppg_before": before.mean(), "ppg_after": after.mean(),
                             "xgd_before": rng.normal(0, 0.5), "xgd_after": 0.0,
                             **{f"pts_{k:+d}": v
                                for k, v in zip(range(-W, W), [*before, *after], strict=True)}})
    return pd.DataFrame(rows)


def test_event_study_after_gap_is_the_headline_effect_and_before_gap_is_zero():
    w = _event_windows(effect=0.4)
    cov = ("ppg_before", "xgd_before")
    es = cce.event_study(w, cov, window=W)
    gap = {k: r["actual"] - r["expected"] for k, r in ((r["match"], r) for r in es["matches"])}
    headline = cce._AdjustedEstimator(w, "ppg", cov).estimate()["effect"]

    assert [r["match"] for r in es["matches"]] == list(range(-W, W))
    assert np.mean([gap[k] for k in range(0, W)]) == pytest.approx(headline, abs=1e-9)
    assert headline == pytest.approx(0.4, abs=0.15)
    # mechanical, not evidence of a good comparison: ppg_before is a covariate
    assert np.mean([gap[k] for k in range(-W, 0)]) == pytest.approx(0.0, abs=1e-9)


def test_windows_can_measure_a_different_number_of_matches_after():
    df = _team(["A"] * 8 + ["B"] * 8)
    df["points"] = [float(i % 4) for i in range(16)]
    df["gf"], df["ga"] = [float(i % 3) for i in range(16)], [1.0] * 16
    w = cce.build_windows(df, window=3, after=5)
    t = w[w["kind"] == "treated"].iloc[0]
    assert [t[f"pts_{k:+d}"] for k in range(-3, 5)] == list(df["points"].iloc[5:13])
    assert t["ppg_after"] == pytest.approx(df["points"].iloc[8:13].mean())
    assert t["gd_before"] == pytest.approx((df["gf"] - df["ga"]).iloc[5:8].mean())
    assert t["gd_after"] == pytest.approx((df["gf"] - df["ga"]).iloc[8:13].mean())
    # windows need `after` matches left, so the last control starts 5 from the end
    assert w["date"].max() <= df["date"].iloc[-5]
