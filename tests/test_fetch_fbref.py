"""
Test for fetch_fbref.py's browser-session restarts. FBref's Cloudflare
protection throttles a browser session after ~20 page loads (found during
the 2014-15 backfill), so the fetch must switch to a fresh browser every
PAGES_PER_BROWSER_SESSION downloaded pages -- whether those pages come from
a few clubs with many seasons (a backfill) or many clubs with one page each
(the weekly refresh).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import fetch_fbref as ff


class FakeFBref:
    """Stands in for soccerdata.FBref: each club "downloads" a fixed number
    of match-log pages into the cache dir, like the real reader does."""

    def __init__(self, data_dir, pages_per_club: int):
        self.data_dir = data_dir
        self.pages_per_club = pages_per_club
        self.restarts = 0

    def read_team_match_stats(self, stat_type: str, team: str) -> pd.DataFrame:
        for season in range(self.pages_per_club):
            (self.data_dir / f"matchlogs_{team}_{season}_schedule.html").write_text("x")
        return pd.DataFrame({"round": ["Matchweek 1"], "GF": ["1"], "GA": ["0"],
                             "match_report": ["/en/matches/abc123/"]})

    def _init_webdriver(self) -> None:
        self.restarts += 1


@pytest.mark.parametrize("clubs,pages_per_club,expected_restarts", [
    (5, 5, 1),   # backfill-like: restart after 15 pages (>= 12), before club 4
    (20, 1, 1),  # weekly-like: restart after 12 one-page clubs
    (3, 2, 0),   # a small run never restarts
])
def test_browser_restarts_every_n_downloaded_pages(monkeypatch, tmp_path, clubs, pages_per_club,
                                                   expected_restarts):
    cache = tmp_path / "cache"
    cache.mkdir()
    fake = FakeFBref(cache, pages_per_club)
    monkeypatch.setattr(ff.sd, "FBref", lambda **kwargs: fake)
    monkeypatch.setattr(ff, "_teams_for_season", lambda fbref: [f"Club {i}" for i in range(clubs)])
    monkeypatch.setattr(ff.config, "RAW_DIR", str(tmp_path / "raw"))

    out = ff.fetch_fbref_matches()

    assert fake.restarts == expected_restarts
    assert len(out) == clubs  # every club's rows still came through
