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


@pytest.mark.parametrize("club,title,ok", [
    ("Roma", "AS Rom - Trainer & Funktionäre | Transfermarkt", True),         # German name via TM_NAMES
    ("Atletico Madrid", "Atlético Madrid - Trainerhistorie", True),            # accents ignored
    ("Paris Saint Germain", "FC Paris Saint-Germain - Trainerhistorie", True),
    ("Paris Saint Germain", "Paris FC - Trainerhistorie", False),              # first word isn't enough
    ("Manchester United", "Manchester City - Trainerhistorie", False),
])
def test_an_id_is_only_accepted_when_the_page_names_the_club(club, title, ok):
    assert flc.title_matches(club, title) is ok


@pytest.fixture
def league_env(monkeypatch, tmp_path):
    monkeypatch.setattr(flc.config, "RAW_DIR", str(tmp_path / "raw"))
    for name in ("IDS_PATH", "OUT_PATH", "STATUS_PATH"):
        monkeypatch.setattr(flc, name, tmp_path / f"{name}.json")
    titles = {12: "AS Rom - Trainerhistorie", 13: "Paris FC - Trainerhistorie"}
    fetched = []

    def fake_table(club, club_id, *, cache_path, use_cache, **kw):
        if not cache_path.exists():
            if club_id == 99:
                raise tm.RequestBudgetReached("budget")
            fetched.append(club_id)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(f"<html><title>{titles[club_id]}</title></html>")
        return pd.DataFrame({"team": [club], "coach": ["X"], "start_date": ["2020-07-01"],
                             "end_date": [None]})

    monkeypatch.setattr(flc, "_fetch_club_table", fake_table)
    return fetched


def test_verified_clubs_are_kept_and_the_rest_recorded_for_a_hand_entry(monkeypatch, league_env):
    monkeypatch.setattr(flc, "league_clubs", lambda: {"ITA-Serie A": ["Genoa", "Napoli", "Roma"]})
    monkeypatch.setattr(flc, "search_club_id", lambda name: {"AS Rom": 12, "SSC Neapel": 13}.get(name))
    status = flc.fetch_league_coaches()
    league = status["leagues"]["ITA-Serie A"]
    assert league["fetched"] == 1 and set(league["failed"]) == {"Genoa", "Napoli"}
    assert status["complete"] is True
    assert list(pd.read_csv(flc.OUT_PATH)["team"]) == ["Roma"]

    # a second run makes no requests: ids and pages are cached, failures aren't retried
    league_env.clear()
    monkeypatch.setattr(flc, "search_club_id", lambda name: pytest.fail("searched again"))
    flc.fetch_league_coaches()
    assert league_env == []


def test_running_out_of_budget_keeps_progress_and_is_not_complete(monkeypatch, league_env):
    monkeypatch.setattr(flc, "league_clubs", lambda: {"ITA-Serie A": ["Roma", "Torino"]})
    monkeypatch.setattr(flc, "search_club_id", lambda name: {"AS Rom": 12, "FC Turin": 99}[name])
    status = flc.fetch_league_coaches()
    assert status["stopped"].startswith("request budget") and status["complete"] is False
    assert json.loads(flc.IDS_PATH.read_text())["ids"]["ITA-Serie A"] == {"Roma": 12}


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
