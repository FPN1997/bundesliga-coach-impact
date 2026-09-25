"""
Multi-league plumbing for the coaching-change study: football-data names are
paired with Understat's by fixtures, leagues without coach data stay out,
same-named clubs in two leagues stay apart, and pooling leagues gives each
its own baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import coach_change_effect as cce
from src import league_data as ld


def _fixtures(names: list[tuple[str, str]], n_days: int = 6,
              seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same fixtures as football-data (fd names) and Understat (us names)."""
    rng = np.random.default_rng(seed)
    fd_rows, us_rows = [], []
    for day in range(n_days):
        order = rng.permutation(len(names))
        for i in range(0, len(order) - 1, 2):
            (h_fd, h_us), (a_fd, a_us) = names[order[i]], names[order[i + 1]]
            hg, ag = rng.integers(0, 4, 2)
            date = pd.Timestamp("2023-08-05") + pd.Timedelta(days=7 * day)
            fd_rows.append({"date": date, "home_team": h_fd, "away_team": a_fd, "FTHG": hg, "FTAG": ag})
            us_rows.append({"date": date, "home_team": h_us, "away_team": a_us, "home_goals": hg,
                            "away_goals": ag})
    return pd.DataFrame(fd_rows), pd.DataFrame(us_rows)


def test_club_names_are_paired_by_fixtures_not_spelling():
    names = [("Man United", "Manchester United"), ("Man City", "Manchester City"),
             ("Ath Madrid", "Atletico Madrid"), ("Paris SG", "Paris Saint Germain"),
             ("Nott'm Forest", "Nottingham Forest"), ("Wolves", "Wolverhampton Wanderers")]
    fd, us = _fixtures(names, n_days=8)
    mapping, unmatched = ld.match_names_by_fixtures(fd, us)
    assert mapping == dict(names) and unmatched == []


def test_a_name_with_too_few_fixtures_stays_unmatched():
    fd, us = _fixtures([("A", "Alpha"), ("B", "Beta"), ("C", "Gamma"), ("D", "Delta")], n_days=2)
    mapping, unmatched = ld.match_names_by_fixtures(fd, us)
    assert mapping == {} and unmatched == ["A", "B", "C", "D"]  # 2 fixtures each < MIN_NAME_VOTES


def test_understat_matches_become_one_row_per_team():
    us = pd.DataFrame({"league": ["X"], "season": ["2324"], "date": ["2023-08-05 15:00"],
                       "home_team": ["Alpha"], "away_team": ["Beta"], "home_goals": [2], "away_goals": [2],
                       "home_xg": [1.5], "away_xg": [0.7]})
    rows = ld.understat_team_rows(us).set_index("team")
    assert rows.loc["Alpha", ["venue", "opponent", "gf", "ga", "xg", "xga", "points"]].tolist() == [
        "Home", "Beta", 2, 2, 1.5, 0.7, 1]
    assert rows.loc["Beta", "venue"] == "Away" and rows.loc["Beta", "xga"] == 1.5


def test_only_leagues_with_complete_coach_data_enter_the_study():
    m = pd.DataFrame({"league": ["A"] * 20 + ["B"] * 20,
                      "coach": ["x"] * 20 + ["y"] * 18 + [None] * 2})
    assert ld.coach_coverage(m) == {"A": 1.0, "B": 0.9}
    assert ld.included_leagues(m) == ["A"]  # 90% < LEAGUE_MIN_COACH_COVERAGE


def test_same_named_clubs_in_two_leagues_stay_apart():
    def season(league, coaches):
        n = len(coaches)
        return pd.DataFrame({"league": league, "team": "Union", "season": "2324", "coach": coaches,
                             "date": pd.date_range("2023-08-05", periods=n, freq="7D"),
                             "points": 1.0, "xg": 1.0, "xga": 1.0})
    m = pd.concat([season("A", ["p"] * 8 + ["q"] * 8), season("B", ["r"] * 16)])
    w = cce.build_windows(m, window=3)
    assert set(w["league"]) == {"A", "B"}
    assert (w.loc[w["kind"] == "treated", "league"] == "A").all()  # B's club never changed coach


def test_pooling_leagues_gives_each_its_own_baseline():
    """League B's clubs recover 0.4 PPG more than league A's at the same form,
    and it also has more sackings. With no real coaching effect anywhere, a
    pooled regression without league baselines would report B's faster
    recovery as an effect; with them it finds nothing."""
    rng = np.random.default_rng(3)
    rows = []
    for league, extra, n_treated in (("A", 0.0, 1), ("B", 0.4, 5)):
        for t in range(10):
            for kind, n in (("control", 60), ("treated", n_treated)):
                for _ in range(n):
                    before = rng.uniform(0, 3) if kind == "control" else rng.uniform(0, 0.8)
                    rows.append({"league": league, "team": f"T{t}", "kind": kind, "in_season": True,
                                 "ppg_before": before, "xgd_before": rng.normal(0, 0.5),
                                 "ppg_after": 1.3 + extra + rng.normal(0, 0.05), "xgd_after": 0.0})
    w = pd.DataFrame(rows)
    naive = cce._AdjustedEstimator(w, "ppg").estimate()["effect"]
    w, dummies = cce.league_dummies(w)
    pooled = cce._AdjustedEstimator(w, "ppg", (*cce._AdjustedEstimator.BASE_COVARIATES, *dummies))
    assert dummies == ["league_B"]
    assert naive > 0.1                                  # the confound, if ignored
    assert pooled.estimate()["effect"] == pytest.approx(0.0, abs=0.03)


def test_one_league_adds_no_intercepts():
    w = pd.DataFrame({"league": ["A", "A"], "x": [1, 2]})
    assert cce.league_dummies(w) == (w, [])
