"""
FBref sits behind Cloudflare. The fetch must stop at the first challenge --
never solve it, click through it or retry around it -- and the weekly
refresh must then keep the previous FBref data. soccerdata's own reader does
the opposite by default (undetected Chrome, a visible browser, a GUI CAPTCHA
clicker, reloads and browser restarts), so these tests pin down that
PoliteFBref removes all of it.
"""

from __future__ import annotations

import pandas as pd
import pytest

import cli
from src import fetch_fbref as ff


@pytest.fixture
def fake_driver(monkeypatch):
    """Record how the browser would be started, without starting one."""
    import seleniumbase

    calls = []

    class Driver:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def quit(self):
            pass

    monkeypatch.setattr(seleniumbase, "Driver", Driver)
    return calls


def test_the_browser_is_plain_headless_chrome(fake_driver, tmp_path):
    reader = ff.PoliteFBref(leagues="GER-Bundesliga", seasons=["2024-2025"],
                            headless=False)  # even when asked for a visible browser
    assert reader.headless is True
    assert fake_driver[-1]["uc"] is False and fake_driver[-1]["headless"] is True


def test_a_captcha_is_never_solved(fake_driver, tmp_path):
    reader = ff.PoliteFBref(leagues="GER-Bundesliga", seasons=["2024-2025"])
    with pytest.raises(ff._Challenged):
        reader.solve_captcha()


def test_the_challenge_escapes_soccerdatas_retry_loop():
    # soccerdata's download loop retries on `except Exception`; the stop signal must not be one
    assert not issubclass(ff._Challenged, Exception)


class FakeReader:
    def __init__(self, challenge_at: int | None = None):
        self.challenge_at, self.calls = challenge_at, 0

    def read_team_match_stats(self, stat_type: str, team: str) -> pd.DataFrame:
        self.calls += 1
        if self.calls == self.challenge_at:
            raise ff._Challenged("challenge")
        return pd.DataFrame({"round": ["Matchweek 1"], "GF": ["1"], "GA": ["0"],
                             "match_report": ["/en/matches/abc123/"]})


def _patch(monkeypatch, tmp_path, reader, clubs=5):
    monkeypatch.setattr(ff, "PoliteFBref", lambda **kwargs: reader)
    monkeypatch.setattr(ff, "_teams_for_season", lambda fbref: [f"Club {i}" for i in range(clubs)])
    monkeypatch.setattr(ff.config, "RAW_DIR", str(tmp_path / "raw"))


def test_the_fetch_stops_at_the_first_challenge(monkeypatch, tmp_path):
    reader = FakeReader(challenge_at=2)
    _patch(monkeypatch, tmp_path, reader)
    with pytest.raises(ff.FBrefBlocked):
        ff.fetch_fbref_matches()
    assert reader.calls == 2                                    # didn't go on to the other clubs
    assert not (tmp_path / "raw" / "fbref_schedule.parquet").exists()  # nothing half-written


def test_an_unchallenged_fetch_still_saves_every_club(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, FakeReader())
    assert len(ff.fetch_fbref_matches()) == 5


def test_weekly_refresh_keeps_previous_fbref_data_when_challenged(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(cli.config, "RAW_DIR", str(tmp_path))

    def blocked():
        raise ff.FBrefBlocked("challenge")

    with pytest.raises(RuntimeError, match="no previous data"):
        cli.refresh_fbref(blocked)  # nothing to fall back on -> still an error
    pd.DataFrame({"team": ["A"]}).to_parquet(tmp_path / "fbref_schedule.parquet")
    cli.refresh_fbref(blocked)
    assert "keeping the previous fbref_schedule.parquet" in caplog.text
