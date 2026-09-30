"""
The Transfermarkt backfill: other leagues' coaching histories come before
squads and injuries, everything stops at the first block, and a club id is
only accepted when the page it points at names that club.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

import cli
from src import fetch_league_coaches as flc
from src import transfermarkt_client as tm

ALL = set(range(2014, 2027))


def test_clubs_match_on_their_league_seasons_and_name():
    us = {"Manchester United": ALL, "Manchester City": ALL, "Roma": ALL,
          "Paris Saint Germain": ALL, "Paris FC": {2025, 2026}}
    tm_names = {985: "Manchester United", 281: "Manchester City", 12: "AS Rom",
                583: "FC Paris Saint-Germain", 10004: "Paris FC", 99: "FC Lorient"}
    tm_seasons = {985: ALL, 281: ALL, 12: ALL, 583: ALL, 10004: {2025, 2026}, 99: {2014, 2015}}
    matched, failed = flc.match_clubs(us, tm_names, tm_seasons)
    assert matched == {"Manchester United": 985, "Manchester City": 281, "Roma": 12,
                       "Paris Saint Germain": 583, "Paris FC": 10004} and failed == {}


def test_a_club_with_an_unmatched_name_is_found_by_its_exact_run_of_seasons():
    us = {"Leganes": {2016, 2017, 2018, 2019}}
    matched, _ = flc.match_clubs(us, {7: "CD Leganés", 8: "Getafe"}, {7: {2016, 2017, 2018, 2019}, 8: ALL})
    assert matched == {"Leganes": 7}          # accents folded: matched by name anyway
    matched, _ = flc.match_clubs({"Xyz": {2016, 2017}}, {7: "Totally Other", 8: "Getafe"},
                                 {7: {2016, 2017}, 8: ALL})
    assert matched == {"Xyz": 7}              # no name match, but the only exact run of seasons


def test_a_renamed_club_keeps_one_transfermarkt_id_for_both_eras():
    us = {"Parma": {2014}, "Parma Calcio 1913": {2018, 2019, 2020}}
    matched, failed = flc.match_clubs(us, {130: "Parma Calcio 1913"}, {130: {2014, 2018, 2019, 2020}})
    assert matched == {"Parma": 130, "Parma Calcio 1913": 130} and failed == {}


def test_the_exact_transfermarkt_name_breaks_a_tie():
    """"AC" is too short to count as a word, so "AC Mailand" also matches
    "Inter Mailand"; the exactly spelled name wins (as for AS Rom / Lazio Rom)."""
    us = {"AC Milan": ALL, "Inter": ALL}
    matched, failed = flc.match_clubs(us, {5: "AC Mailand", 46: "Inter Mailand"}, {5: ALL, 46: ALL})
    assert matched == {"AC Milan": 5, "Inter": 46} and failed == {}


def test_only_the_staff_history_of_the_requested_id_counts():
    page = '<link rel="canonical" href="https://www.transfermarkt.de/fc-chelsea/mitarbeiterhistorie/verein/631">'
    homepage = '<link rel="canonical" href="https://www.transfermarkt.de/">'
    assert flc.page_is_staff_history_of(page, 631)
    assert not flc.page_is_staff_history_of(page, 63)       # a prefix of the id isn't the id
    assert not flc.page_is_staff_history_of(homepage, 631)  # a redirect to the homepage (seen live)


def test_an_ambiguous_club_is_left_for_a_hand_entry():
    us = {"Real": ALL}
    matched, failed = flc.match_clubs(us, {1: "Real Madrid", 2: "Real Sociedad"}, {1: ALL, 2: ALL})
    assert matched == {} and "ambiguous" in failed["Real"]


@pytest.fixture
def league_env(monkeypatch, tmp_path):
    monkeypatch.setattr(flc.config, "RAW_DIR", str(tmp_path / "raw"))
    for name in ("IDS_PATH", "OUT_PATH", "STATUS_PATH"):
        monkeypatch.setattr(flc, name, tmp_path / f"{name}.json")
    # the page served for each id: id 13 answers with some other club's staff history
    serves = {12: 12, 13: 159}
    fetched = []

    def fake_table(tm_name, club_id, *, cache_path, use_cache, **kw):
        if not cache_path.exists():
            if club_id == 99:
                raise tm.RequestBudgetReached("budget")
            fetched.append(club_id)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text('<html><head><link rel="canonical" href="https://www.transfermarkt.de/x/'
                                  f'mitarbeiterhistorie/verein/{serves[club_id]}"></head></html>')
        return pd.DataFrame({"team": [tm_name], "coach": ["X"], "start_date": ["2020-07-01"],
                             "end_date": [None]})

    monkeypatch.setattr(flc, "_fetch_club_table", fake_table)
    return fetched


def test_verified_clubs_are_kept_and_the_rest_recorded_for_a_hand_entry(monkeypatch, league_env):
    monkeypatch.setattr(flc, "understat_seasons", lambda: {"ITA-Serie A": {"Roma": ALL, "Napoli": ALL,
                                                                           "Genoa": ALL}})
    # the league page lists AS Rom and, wrongly for this test, id 13 under "SSC Neapel"
    monkeypatch.setattr(flc, "tm_league_seasons", lambda league, years: (
        {12: "AS Rom", 13: "SSC Neapel"}, {12: ALL, 13: ALL}))
    status = flc.fetch_league_coaches()
    league = status["leagues"]["ITA-Serie A"]
    assert league["fetched"] == 1 and set(league["failed"]) == {"Genoa", "Napoli"}
    assert "isn't its staff history" in league["failed"]["Napoli"]  # id 13's page is another club's
    assert status["complete"] is True
    assert list(pd.read_csv(flc.OUT_PATH)["team"]) == ["Roma"]

    assert not flc.cache_path("ITA-Serie A", "Napoli", 13).exists()   # a failed page isn't kept
    league_env.clear()
    flc.fetch_league_coaches()
    assert league_env == [13]  # Roma's page is cached; only the rejected one is retried


def test_running_out_of_budget_keeps_progress_and_is_not_complete(monkeypatch, league_env):
    monkeypatch.setattr(flc, "understat_seasons", lambda: {"ITA-Serie A": {"Roma": ALL, "Torino": ALL}})
    monkeypatch.setattr(flc, "tm_league_seasons", lambda league, years: (
        {12: "AS Rom", 99: "FC Turin"}, {12: ALL, 99: ALL}))
    status = flc.fetch_league_coaches()
    assert status["stopped"].startswith("request budget") and status["complete"] is False
    assert json.loads(flc.IDS_PATH.read_text())["ids"]["ITA-Serie A"]["Roma"] == 12


@pytest.fixture
def backfill_env(monkeypatch, tmp_path):
    import src.fetch_coach_history as fch
    import src.fetch_squads as fs
    monkeypatch.setattr(cli, "_require", lambda *a, **k: None)
    monkeypatch.setattr(cli.config, "COACH_HISTORY_RESOLVED_CSV", str(tmp_path / "coach_history.csv"))
    monkeypatch.setattr(cli, "BACKFILL_DONE", tmp_path / "backfill_complete")
    monkeypatch.setattr(fs, "injuries_path", lambda: tmp_path / "tm_injuries.parquet")
    tm.reset()
    calls = []
    monkeypatch.setattr(fch, "fetch_coach_history", lambda: calls.append("bundesliga coaches"))
    monkeypatch.setattr(fs, "fetch_squads", lambda: calls.append("squads"))
    return calls, monkeypatch, tmp_path


def test_backfill_puts_coach_histories_before_squads(backfill_env):
    calls, monkeypatch, tmp_path = backfill_env
    monkeypatch.setattr(flc, "fetch_league_coaches",
                        lambda: calls.append("other leagues") or {"complete": True, "stopped": None})
    cli.cmd_backfill(argparse.Namespace())
    assert calls == ["bundesliga coaches", "other leagues", "squads"]
    assert not (tmp_path / "backfill_complete").exists()  # squads not finished yet


def test_backfill_leaves_squads_for_the_next_run_when_the_coaches_used_the_budget(backfill_env):
    calls, monkeypatch, _ = backfill_env
    monkeypatch.setattr(flc, "fetch_league_coaches",
                        lambda: calls.append("other leagues")
                        or {"complete": False, "stopped": "request budget"})
    cli.cmd_backfill(argparse.Namespace())
    assert calls == ["bundesliga coaches", "other leagues"]


def test_backfill_stops_everything_at_a_block(backfill_env):
    calls, monkeypatch, _ = backfill_env
    import src.fetch_coach_history as fch

    def blocked():
        raise tm.TransfermarktBlocked("blocked")

    monkeypatch.setattr(fch, "fetch_coach_history", blocked)
    monkeypatch.setattr(flc, "fetch_league_coaches", lambda: pytest.fail("kept going after a block"))
    cli.cmd_backfill(argparse.Namespace())
    assert calls == []


SEARCH_PAGE = """
<div class="box"><h2>Suchergebnisse zu Trainern &amp; Funktionären - 1</h2>
  <table class="items"><tr class="odd"><td><table class="inline-table"><tr>
    <td class="hauptlink"><a href="/some-coach/profil/trainer/1">Some Coach</a></td></tr><tr>
    <td><a href="/atletico-mancha-real/startseite/verein/29270">Atlético Mancha Real</a></td>
  </tr></table></td></tr></table></div>
<div class="box"><h2>Suchergebnisse zu Vereinen - 77 Treffer</h2>
  <table class="items">
    <tr class="odd"><td><table class="inline-table"><tr>
      <td class="hauptlink"><a href="/fc-arsenal-u23/startseite/verein/9249">FC Arsenal U21</a></td>
      </tr></table></td></tr>
    <tr class="even"><td><table class="inline-table"><tr>
      <td class="hauptlink"><a href="/arsenal-tula/startseite/verein/3729">Arsenal Tula</a></td>
      </tr></table></td></tr>
    <tr class="odd"><td><table class="inline-table"><tr>
      <td class="hauptlink"><a href="/fc-arsenal/startseite/verein/11">FC Arsenal</a></td>
      </tr></table></td></tr>
  </table></div>
"""


def test_search_reads_the_clubs_section_not_a_coachs_club():
    """The page lists coaches first, each linking their club: the first club
    link on the page was Atletico Mancha Real for "Arsenal" (29 Sep 2026)."""
    from src import transfermarkt_search as tms
    assert tms.club_results(SEARCH_PAGE) == [("Arsenal Tula", 3729), ("FC Arsenal", 11)]  # U21 skipped


def test_search_takes_the_first_club_named_like_the_query(monkeypatch):
    """What the search can and can't do: it skips non-matching and youth
    results, but "Arsenal Tula" and "FC Arsenal" both match "Arsenal", so
    the first one listed wins. That's why the other leagues get their ids
    from league pages instead (fetch_league_coaches)."""
    from src import transfermarkt_search as tms

    class Resp:
        url, text = "https://www.transfermarkt.de/schnellsuche/ergebnis/schnellsuche?query=x", SEARCH_PAGE

        def raise_for_status(self):
            pass

    monkeypatch.setattr(tms.tm, "get", lambda *a, **k: Resp())
    assert tms.search_club_id("Arsenal") == 3729
    assert tms.search_club_id("Tula") == 3729
    assert tms.search_club_id("Chelsea") == 3729  # no name match: the first senior result
