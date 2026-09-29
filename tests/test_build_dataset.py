"""
The Bundesliga dataset is built on Understat (results, xG) with FBref's
extras joined on: a match FBref doesn't have yet still counts, an awarded
result keeps FBref's official score, and repeat meetings join in order.
Plus the "polite" readers: identifiable, and a refusal stops them.
"""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from src import build_dataset as bd
from src import polite_sources as ps


def _understat(matches):
    """matches: (game_id, date, home, away, home_goals, away_goals)."""
    rows = []
    for gid, date, home, away, hg, ag in matches:
        for team, opp, venue, gf in ((home, away, "Home", hg), (away, home, "Away", ag)):
            rows.append({"season": "2627", "game_id": gid, "date": pd.Timestamp(date), "team": team,
                         "opponent": opp, "venue": venue, "goals": gf, "points": 0, "xg": 1.0,
                         "xga": 1.0, "ppda": 9.0, "deep_completions": 5})
    return pd.DataFrame(rows)


def _fbref(matches):
    """matches: (date, home, away, home_goals, away_goals, home_formation)."""
    rows = []
    for date, home, away, hg, ag, formation in matches:
        for team, opp, venue, gf, ga in ((home, away, "Home", hg, ag), (away, home, "Away", ag, hg)):
            rows.append({"season": "2627", "date": pd.Timestamp(date), "team": team, "opponent": opp,
                         "venue": venue, "GF": float(gf), "GA": float(ga),
                         "Formation": formation if venue == "Home" else "4-4-2", "Opp Formation": "x"})
    return pd.DataFrame(rows)


@pytest.fixture
def build(monkeypatch, tmp_path):
    coaches = pd.DataFrame({"team": ["A", "B"], "coach": ["ca", "cb"],
                            "start_date": pd.Timestamp("2020-01-01"), "end_date": pd.NaT})

    def run(understat, fbref):
        monkeypatch.setattr(bd, "_load_raw", lambda: (fbref, understat, coaches))
        monkeypatch.setattr(bd.config, "PROCESSED_DIR", str(tmp_path))
        return bd.build_dataset().set_index(["team", "date"]).sort_index()
    return run


def test_a_match_fbref_doesnt_have_yet_still_counts(build):
    out = build(_understat([(1, "2026-08-22", "A", "B", 2, 0), (2, "2026-08-29", "B", "A", 1, 1)]),
                _fbref([("2026-08-22", "A", "B", 2, 0, "4-2-3-1")]))       # FBref stops after match 1
    new = out.loc[("B", pd.Timestamp("2026-08-29"))]
    assert (new["gf"], new["ga"], new["result"], new["points"]) == (1, 1, "D", 1)
    assert pd.isna(new["formation"]) and new["coach"] == "cb"
    assert out.loc[("A", pd.Timestamp("2026-08-22")), "formation"] == "4-2-3-1"


def test_an_awarded_result_keeps_fbrefs_official_score(build, caplog):
    # on the pitch 1-1; awarded 0-2 (as Union Berlin v Bochum, Dec 2024)
    out = build(_understat([(1, "2024-12-14", "A", "B", 1, 1)]),
                _fbref([("2024-12-14", "A", "B", 0, 2, "3-5-2")]))
    assert out.loc[("B", pd.Timestamp("2024-12-14")), "points"] == 3
    assert out.loc[("A", pd.Timestamp("2024-12-14")), "points"] == 0
    assert "Score differs" in caplog.text


def test_repeat_meetings_join_in_date_order(build):
    out = build(_understat([(1, "2026-08-22", "A", "B", 1, 0), (2, "2026-09-12", "A", "B", 0, 3)]),
                _fbref([("2026-09-12", "A", "B", 0, 3, "second"), ("2026-08-22", "A", "B", 1, 0, "first")]))
    assert out.loc[("A", pd.Timestamp("2026-08-22")), "formation"] == "first"
    assert out.loc[("A", pd.Timestamp("2026-09-12")), "formation"] == "second"


def test_polite_session_names_the_project():
    session = ps.plain_session({"X-Requested-With": "XMLHttpRequest"})
    assert "bundesliga-coach-impact" in session.headers["User-Agent"]
    assert session.headers["X-Requested-With"] == "XMLHttpRequest"


@pytest.mark.parametrize("status,refused", [(403, True), (429, True), (404, False), (200, False)])
def test_a_refusal_stops_the_reader(status, refused):
    response = requests.Response()
    response.status_code, response.url = status, "https://example"
    if refused:
        with pytest.raises(ps.SourceRefused):
            ps._refuse_on_block(response)
    else:
        ps._refuse_on_block(response)
    assert not issubclass(ps.SourceRefused, Exception)  # escapes soccerdata's retry loop
