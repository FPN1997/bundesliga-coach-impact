"""
Leak-safety test for src/coach_bounce.py's coach-level historical-profile
join: an incoming coach's "track record" must be built ONLY from their
matches strictly BEFORE the change_date being predicted -- same spirit as
features.py's .shift(1) tests, just for a coach-level join instead of a
per-team rolling window. Also covers the min-prior-matches threshold that
drops a coaching change with too thin a track record to build a profile
from, rather than silently keeping it with a noisy 1-2-match average.

All synthetic data, no network access -- see test_features.py's module
docstring for why that matters here too.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import coach_bounce as cb


def _write_fixtures(tmp_path, monkeypatch, *, matches_df: pd.DataFrame, impact_df: pd.DataFrame):
    monkeypatch.setattr(cb.config, "PROCESSED_DIR", str(tmp_path))
    monkeypatch.setattr(cb.config, "OUTPUT_DIR", str(tmp_path))
    matches_df.to_parquet(tmp_path / "match_dataset.parquet", index=False)
    impact_df.to_csv(tmp_path / "coach_impact_rankings.csv", index=False)


def _prior_matches(coach: str, team: str, n: int, *, last_date: str) -> pd.DataFrame:
    """n matches for `coach` at `team`, dated weekly, ending strictly before
    `last_date` (the (n+1)th weekly slot before it) -- deterministic,
    hand-computable stat values (points/gf/ga/xg/xga/ppda/deep_completions
    all equal to the match's 1-indexed position, so the mean of n matches
    is exactly (n+1)/2)."""
    end = pd.Timestamp(last_date) - pd.Timedelta(weeks=1)
    dates = pd.date_range(end=end, periods=n, freq="7D")
    vals = list(range(1, n + 1))
    return pd.DataFrame({
        "coach": coach, "team": team, "date": dates,
        "points": vals, "gf": vals, "ga": [0] * n,
        "xg": [float(v) for v in vals], "xga": [0.0] * n,
        "ppda": [float(v) for v in vals], "deep_completions": vals,
    })


def _impact_row(team: str, coach: str, change_date: str, ppg_delta: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "team": team, "incoming_coach": coach, "change_date": change_date,
        "ppg_delta": ppg_delta, "ppg_before": 1.0, "goal_diff_pg_before": 0.0,
        "xg_diff_pg_before": 0.0, "ppda_before": 10.0,
    }])


def test_historical_profile_uses_only_strictly_prior_matches(tmp_path, monkeypatch):
    # 12 prior matches (values 1..12, mean 6.5) for "Coach X" at "OldClub",
    # all safely before the change -- plus one match dated exactly ON
    # change_date with an extreme value that must NOT be folded in.
    prior = _prior_matches("Coach X", "OldClub", n=12, last_date="2023-01-01")
    on_boundary = pd.DataFrame([{
        "coach": "Coach X", "team": "OldClub", "date": pd.Timestamp("2023-01-01"),
        "points": 999, "gf": 999, "ga": 0, "xg": 999.0, "xga": 0.0,
        "ppda": 999.0, "deep_completions": 999,
    }])
    matches = pd.concat([prior, on_boundary], ignore_index=True)
    impact = _impact_row("NewClub", "Coach X", "2023-01-01")

    _write_fixtures(tmp_path, monkeypatch, matches_df=matches, impact_df=impact)
    out = cb.build_bounce_dataset(min_prior_matches=10)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["coach_hist_ppg"] == pytest.approx(6.5)  # mean(1..12), NOT pulled toward 999
    assert row["coach_hist_goal_diff_pg"] == pytest.approx(6.5)  # gf-ga == gf here
    assert row["coach_hist_ppda"] == pytest.approx(6.5)
    assert row["coach_hist_deep_completions"] == pytest.approx(6.5)
    assert row["coach_hist_n_clubs"] == 1


def test_coaching_change_with_too_thin_a_track_record_is_dropped(tmp_path, monkeypatch):
    thin = _prior_matches("Coach Y", "OldClub", n=3, last_date="2023-01-01")  # below threshold
    impact = _impact_row("NewClub", "Coach Y", "2023-01-01")

    _write_fixtures(tmp_path, monkeypatch, matches_df=thin, impact_df=impact)
    out = cb.build_bounce_dataset(min_prior_matches=10)

    assert out.empty


def test_coach_with_zero_prior_matches_is_dropped_not_crashed(tmp_path, monkeypatch):
    # A coach who has never appeared in match_dataset.parquet before --
    # e.g. new to management within the window, or pre-SEASONS/non-Bundesliga.
    matches = _prior_matches("Someone Else", "OldClub", n=12, last_date="2023-01-01")
    impact = _impact_row("NewClub", "Coach Z", "2023-01-01")

    _write_fixtures(tmp_path, monkeypatch, matches_df=matches, impact_df=impact)
    out = cb.build_bounce_dataset(min_prior_matches=10)

    assert out.empty


def test_multiple_prior_clubs_counted_correctly(tmp_path, monkeypatch):
    # A coach with a track record spanning two different clubs -- exercises
    # coach_hist_n_clubs and confirms matches from BOTH clubs are pooled
    # into one profile (a coach's history isn't scoped to a single team).
    club_a = _prior_matches("Coach W", "ClubA", n=6, last_date="2022-06-01")
    club_b = _prior_matches("Coach W", "ClubB", n=6, last_date="2023-01-01")
    matches = pd.concat([club_a, club_b], ignore_index=True)
    impact = _impact_row("NewClub", "Coach W", "2023-01-01")

    _write_fixtures(tmp_path, monkeypatch, matches_df=matches, impact_df=impact)
    out = cb.build_bounce_dataset(min_prior_matches=10)

    assert len(out) == 1
    assert out.iloc[0]["coach_hist_n_clubs"] == 2
    assert out.iloc[0]["coach_hist_ppg"] == pytest.approx(3.5)  # mean(1..6) twice over -> mean(1..6)
