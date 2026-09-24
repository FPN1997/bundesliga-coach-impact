"""
Transfermarkt's firewall blocked the squad scrape after ~1,100 requests
(HTTP 405 with x-amzn-waf-action: captcha). These tests pin down the
response to that: recognize every form of block, stop at the first one, cap
each squad-scrape run, and keep the weekly refresh going on the previous
coach history instead of failing it.
"""

from __future__ import annotations

import pandas as pd
import pytest
import requests

import cli
from src import fetch_coach_history as fch
from src import fetch_squads as fs


def _response(status: int, body: str = "<html>ok</html>", headers: dict | None = None) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r._content = body.encode()
    r.headers.update(headers or {})
    return r


@pytest.mark.parametrize("status,body,headers", [
    (405, "<html>captcha</html>", {"x-amzn-waf-action": "captcha"}),  # the block seen live
    (202, "", {}),                                                      # the old .com challenge
    (403, "<html>denied</html>", {}),
    (429, "<html>slow down</html>", {}),
])
def test_every_form_of_block_is_recognized(status, body, headers):
    with pytest.raises(fch.TransfermarktBlocked):
        fch.raise_if_blocked(_response(status, body, headers), "https://example")


def test_a_normal_page_is_not_a_block():
    fch.raise_if_blocked(_response(200), "https://example")


def test_coach_history_stops_at_the_first_block(monkeypatch, tmp_path):
    monkeypatch.setattr(fch.config, "RAW_DIR", str(tmp_path))  # no FBref file -> config club list
    monkeypatch.setattr(fch, "AUTO_RESOLVED_PATH", tmp_path / "auto.json")
    monkeypatch.setattr(fch.config, "CLUB_TRANSFERMARKT_ID", {"Alpha": 1, "Beta": 2, "Gamma": 3})
    calls = []

    def blocked(club, club_id, **kwargs):
        calls.append(club)
        raise fch.TransfermarktBlocked("blocked")

    monkeypatch.setattr(fch, "_fetch_club_table", blocked)
    with pytest.raises(fch.TransfermarktBlocked):
        fch.fetch_coach_history()
    assert calls == ["Alpha"]  # didn't go on to hammer the other clubs


def test_weekly_refresh_keeps_previous_coach_history_when_blocked(monkeypatch, tmp_path, caplog):
    def blocked():
        raise fch.TransfermarktBlocked("blocked")

    existing = tmp_path / "coach_history.csv"
    monkeypatch.setattr(cli.config, "COACH_HISTORY_RESOLVED_CSV", str(existing))
    with pytest.raises(fch.TransfermarktBlocked):
        cli.refresh_coach_history(blocked)  # nothing to fall back on -> still an error

    pd.DataFrame({"team": ["Alpha"]}).to_csv(existing, index=False)
    cli.refresh_coach_history(blocked)      # falls back, with a warning
    assert "keeping the previous coach history" in caplog.text


def test_squad_scrape_stops_at_its_request_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(fs, "REQUEST_DELAY", 0.0)
    monkeypatch.setattr(fs, "MAX_LIVE_REQUESTS_PER_RUN", 2)
    monkeypatch.setattr(fs.requests, "get", lambda *a, **k: _response(200))
    fetcher = fs._Fetcher(current_season_start=2026)
    fetcher.get("https://example/1", tmp_path / "1.html", 2020)
    fetcher.get("https://example/2", tmp_path / "2.html", 2020)
    fetcher.get("https://example/1", tmp_path / "1.html", 2020)  # cached: doesn't count
    with pytest.raises(fs.RequestBudgetReached):
        fetcher.get("https://example/3", tmp_path / "3.html", 2020)
