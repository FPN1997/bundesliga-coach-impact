"""
Tests for src/squad_availability.py (injury and new-signing shares per
match) and their use as covariates in the coaching-change estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import coach_change_effect as cce
from src import squad_availability as sa


def _squad(rows):
    return pd.DataFrame(rows, columns=["team", "season", "player_id", "market_value",
                                       "winter_arrival", "injury_tracked"])


def _matches(dates, team="Alpha", season="2324"):
    return pd.DataFrame({"team": team, "season": season, "date": pd.to_datetime(dates)})


def _injuries(rows):
    df = pd.DataFrame(rows, columns=["player_id", "start", "end", "games_missed"])
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def test_injured_share_is_value_weighted_and_covers_start_to_end():
    squad = _squad([("Alpha", "2324", 1, 30e6, False, True), ("Alpha", "2324", 2, 10e6, False, True)])
    injuries = _injuries([(1, "2023-11-10", "2023-11-20", 2)])
    out = sa.match_availability(_matches(["2023-11-05", "2023-11-10", "2023-11-20", "2023-11-25"]),
                                squad, injuries)
    assert out["injured_share"].tolist() == pytest.approx([0.0, 0.75, 0.75, 0.0])


def test_winter_arrival_joins_mid_january_and_earlier_injuries_do_not_count():
    squad = _squad([("Alpha", "2324", 1, 30e6, False, True), ("Alpha", "2324", 9, 10e6, True, True)])
    # the new signing was injured at his previous club in December
    injuries = _injuries([(9, "2023-12-01", "2023-12-31", 4)])
    out = sa.match_availability(_matches(["2023-12-15", "2024-01-20"]), squad, injuries)
    assert out["new_signings_share"].tolist() == pytest.approx([0.0, 0.25])
    assert out["injured_share"].tolist() == pytest.approx([0.0, 0.0])


def test_missing_end_date_lasts_a_week_per_game_missed():
    squad = _squad([("Alpha", "2324", 1, 10e6, False, True)])
    injuries = _injuries([(1, "2023-10-01", None, 2)])  # -> assumed to end 2023-10-15
    out = sa.match_availability(_matches(["2023-10-14", "2023-10-16"]), squad, injuries)
    assert out["injured_share"].tolist() == pytest.approx([1.0, 0.0])


def test_untracked_players_and_missing_club_seasons():
    squad = _squad([("Alpha", "2324", 1, 10e6, False, True), ("Alpha", "2324", 2, 90e6, False, False)])
    injuries = _injuries([(2, "2023-10-01", "2023-10-30", 4)])  # untracked -> ignored
    out = sa.match_availability(
        pd.concat([_matches(["2023-10-10"]), _matches(["2023-10-10"], team="Beta")], ignore_index=True),
        squad, injuries)
    assert out.loc[0, "injured_share"] == 0.0
    assert np.isnan(out.loc[1, "injured_share"])  # Beta has no squad data


def test_windows_carry_the_change_in_injury_share_between_halves():
    w = 3
    n = 16
    df = pd.DataFrame({
        "team": "Alpha", "date": pd.date_range("2023-09-01", periods=n, freq="7D"), "season": "2324",
        "coach": ["A"] * 8 + ["B"] * 8, "points": 1.0, "xg": 1.0, "xga": 1.0,
        # 40% of the squad injured before the change, 10% after
        "injured_share": [0.4] * 8 + [0.1] * 8, "new_signings_share": 0.0,
    })
    treated = cce.build_windows(df, window=w).query("kind == 'treated'").iloc[0]
    assert treated["injury_change"] == pytest.approx(-0.3)
    assert treated["signing_change"] == pytest.approx(0.0)


def test_squad_adjustment_is_skipped_when_coverage_is_incomplete(caplog):
    rng = np.random.default_rng(1)
    rows = []
    for t in range(10):
        for kind, n in [("control", 40), ("treated", 3)]:
            for _ in range(n):
                rows.append({"team": f"T{t}", "kind": kind, "in_season": True, "window_after": False,
                             "ppg_before": rng.uniform(0, 3), "ppg_after": rng.uniform(0, 3),
                             "xgd_before": rng.normal(), "xgd_after": rng.normal(),
                             "injury_change": rng.normal(), "signing_change": 0.0})
    windows = pd.DataFrame(rows)
    windows.loc[windows.sample(frac=0.2, random_state=0).index, "injury_change"] = np.nan  # 80% coverage
    results = cce.estimate_effects(windows)
    assert "injury_change" not in results["covariates"]
    assert "covers only" in caplog.text


def test_injury_pages_parse_for_a_player_with_no_injuries():
    # Regression: the first full scrape crashed on a player whose injury
    # table was empty (comparing "no date" with the data's start date).
    from src import fetch_squads as fs
    rows, pages = fs.parse_injuries("<html><body><p>Keine Einträge</p></body></html>")
    assert rows.empty and pages == 1
    assert not rows["start"].notna().any()  # the fetch loop's stop condition must handle this
