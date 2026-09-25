"""
Tests for src/sack_o_meter.py: the training target must only look ahead
within a season and only as far as RISK_HORIZON, and tenure must come from
the coach history (right for promoted clubs, a caretaker made permanent is
one spell).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import sack_o_meter as som


def _season(coaches: list[str], team: str = "Alpha", season: str = "2324") -> pd.DataFrame:
    n = len(coaches)
    return pd.DataFrame({
        "team": team, "season": season, "coach": coaches,
        "date": pd.date_range("2023-08-05", periods=n, freq="7D"),
        "opponent": "Beta", "venue": ["Home", "Away"] * (n // 2) + ["Home"] * (n % 2),
        "points": [0.0, 1.0, 3.0] * (n // 3) + [1.0] * (n % 3),
        "xg": 1.0, "xga": 1.2,
    })


def _odds(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[matches["venue"] == "Home"]
    return pd.DataFrame({"season": home["season"], "home_team": home["team"], "away_team": home["opponent"],
                         "AvgCH": 2.5, "AvgCD": 3.4, "AvgCA": 2.9})


def _spells(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return som.coach_spells(pd.DataFrame(rows, columns=["team", "coach", "start_date"]))


def test_back_to_back_rows_of_one_coach_are_one_spell():
    s = _spells([("Alpha", "A", "2020-07-01"), ("Alpha", "B", "2021-10-01"),  # caretaker spell ...
                 ("Alpha", "B", "2021-11-01"),                                  # ... made permanent
                 ("Alpha", "C", "2023-07-01")])
    assert list(s["coach"]) == ["A", "B", "C"]
    assert s.loc[s["coach"] == "B", "start_date"].iloc[0] == pd.Timestamp("2021-10-01")


def test_target_flags_only_the_matches_within_the_horizon_before_a_change():
    m = _season(["A"] * 10 + ["B"] * 10)
    spells = _spells([("Alpha", "A", "2022-07-01"), ("Alpha", "B", "2023-10-10")])
    st = som.team_match_states(m, _odds(m), spells)
    y = st["sacked_soon"].to_numpy()
    h = som.RISK_HORIZON
    # the last h matches under A are followed by a change; earlier ones aren't
    assert (y[10 - h:10] == 1).all()
    assert (y[:10 - h] == 0).all()
    # the season's last h matches can't be labelled: what follows is unknown
    assert np.isnan(y[-h:]).all()


def test_target_does_not_look_across_the_summer():
    m = pd.concat([_season(["A"] * 10, season="2223"), _season(["B"] * 10, season="2324")
                   .assign(date=lambda d: d["date"] + pd.Timedelta(days=365))], ignore_index=True)
    spells = _spells([("Alpha", "A", "2020-07-01"), ("Alpha", "B", "2024-07-01")])
    st = som.team_match_states(m, _odds(m), spells)
    first = st[st["season"] == "2223"]
    # a summer change isn't a mid-season sacking: the end of 2022-23 is unknown, not "sacked"
    assert first["sacked_soon"].iloc[:10 - som.RISK_HORIZON].eq(0).all()
    assert first["sacked_soon"].iloc[-som.RISK_HORIZON:].isna().all()


def test_tenure_comes_from_the_coach_history_not_the_match_count():
    m = _season(["A"] * 6)  # e.g. a promoted club: only 6 Bundesliga matches in the data
    st = som.team_match_states(m, _odds(m), _spells([("Alpha", "A", "2021-07-01")]))
    assert st["tenure_days"].iloc[0] == (pd.Timestamp("2023-08-05") - pd.Timestamp("2021-07-01")).days
    # a coach the history can't place falls back to a week per match
    st = som.team_match_states(m, _odds(m), _spells([("Alpha", "Z", "2021-07-01")]))
    assert list(st["tenure_days"]) == [7.0 * k for k in range(1, 7)]


def test_form_is_this_seasons_last_8_matches_or_all_of_them_if_fewer():
    m = _season(["A"] * 12)
    st = som.team_match_states(m, _odds(m), _spells([("Alpha", "A", "2022-07-01")]))
    assert st["ppg_form"].iloc[2] == pytest.approx(m["points"].iloc[:3].mean())
    assert st["ppg_form"].iloc[11] == pytest.approx(m["points"].iloc[4:12].mean())
    assert st["ppg_last2"].iloc[11] == pytest.approx(m["points"].iloc[10:12].mean())
