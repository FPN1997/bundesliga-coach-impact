"""
Regression test for a real incident: Transfermarkt started blocking every
request with an AWS WAF challenge, every club's fetch failed, and
fetch_coach_history() silently overwrote a ~1600-row coach_history.csv with
a 1-row file (just the manual-CSV fallback) -- which then made 98% of
match_dataset.parquet's rows coach-less and broke coach_impact.py's
before/after windows entirely, without a single crash anywhere in the
chain. This checks the guard added to prevent exactly that from happening
silently again.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import fetch_coach_history as fch


def _write_existing_history(path, n_rows: int):
    pd.DataFrame({
        "team": ["Alpha"] * n_rows,
        "coach": [f"Coach {i}" for i in range(n_rows)],
        "start_date": pd.date_range("2000-01-01", periods=n_rows, freq="365D"),
        "end_date": pd.date_range("2000-06-01", periods=n_rows, freq="365D"),
    }).to_csv(path, index=False)


def _isolate_from_real_project_data(monkeypatch, tmp_path):
    """_teams_needing_coach_data() falls back to config.CLUB_TRANSFERMARKT_ID
    only when data/raw/fbref_schedule.parquet doesn't exist -- on this
    machine it does, so without this, these tests would silently read the
    real project's team list instead of the test's. Point RAW_DIR at an
    empty tmp dir so the intended fallback path actually gets exercised,
    and keep the auto-resolved-id cache out of the real project file too.
    """
    monkeypatch.setattr(fch.config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fch, "AUTO_RESOLVED_PATH", tmp_path / "auto_ids.json")


def test_refuses_to_overwrite_a_large_history_with_a_tiny_one(monkeypatch, tmp_path):
    resolved_path = tmp_path / "coach_history.csv"
    manual_path = tmp_path / "coach_history_manual.csv"
    _write_existing_history(resolved_path, n_rows=100)
    pd.DataFrame(columns=["team", "coach", "start_date", "end_date"]).to_csv(
        manual_path, index=False
    )

    _isolate_from_real_project_data(monkeypatch, tmp_path)
    monkeypatch.setattr(fch.config, "COACH_HISTORY_RESOLVED_CSV", str(resolved_path))
    monkeypatch.setattr(fch.config, "COACH_HISTORY_MANUAL_CSV", str(manual_path))
    monkeypatch.setattr(fch.config, "CLUB_TRANSFERMARKT_ID", {"Alpha": 1, "Beta": 2})

    # Simulate every club's live fetch failing (the WAF-block scenario).
    monkeypatch.setattr(fch, "_fetch_club_table",
                         lambda club, club_id, **kwargs: (_ for _ in ()).throw(ValueError("blocked")))

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        fch.fetch_coach_history()

    # The pre-existing (good) file must be untouched, not clobbered.
    survived = pd.read_csv(resolved_path)
    assert len(survived) == 100


def test_allows_writing_when_the_result_is_not_a_big_drop(monkeypatch, tmp_path):
    resolved_path = tmp_path / "coach_history.csv"
    manual_path = tmp_path / "coach_history_manual.csv"
    _write_existing_history(resolved_path, n_rows=10)  # below the 20-row guard threshold
    pd.DataFrame(columns=["team", "coach", "start_date", "end_date"]).to_csv(
        manual_path, index=False
    )

    _isolate_from_real_project_data(monkeypatch, tmp_path)
    monkeypatch.setattr(fch.config, "COACH_HISTORY_RESOLVED_CSV", str(resolved_path))
    monkeypatch.setattr(fch.config, "COACH_HISTORY_MANUAL_CSV", str(manual_path))
    monkeypatch.setattr(fch.config, "CLUB_TRANSFERMARKT_ID", {"Alpha": 1})
    monkeypatch.setattr(fch, "_fetch_club_table",
                         lambda club, club_id, **kwargs: (_ for _ in ()).throw(ValueError("blocked")))

    # Small existing files (<=20 rows) aren't worth guarding -- this should
    # complete (writing 0 rows) rather than raise.
    result = fch.fetch_coach_history()
    assert len(result) == 0


def test_teams_needing_coach_data_falls_back_when_fbref_data_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(fch.config, "RAW_DIR", str(tmp_path))  # empty -- no fbref_schedule.parquet
    monkeypatch.setattr(fch.config, "CLUB_TRANSFERMARKT_ID", {"Zeta": 1, "Alpha": 2})
    assert fch._teams_needing_coach_data() == ["Alpha", "Zeta"]


def test_teams_needing_coach_data_prefers_live_fbref_team_list(monkeypatch, tmp_path):
    # When fbref_schedule.parquet exists, its team list wins even over a
    # differently-shaped config dict -- this is what lets a brand-new,
    # not-yet-configured team actually get attempted (and auto-resolved)
    # rather than silently skipped.
    monkeypatch.setattr(fch.config, "RAW_DIR", str(tmp_path))
    pd.DataFrame({"team": ["NewTeam", "Alpha", "Alpha"]}).to_parquet(
        tmp_path / "fbref_schedule.parquet", index=False
    )
    monkeypatch.setattr(fch.config, "CLUB_TRANSFERMARKT_ID", {"Alpha": 1})
    assert fch._teams_needing_coach_data() == ["Alpha", "NewTeam"]


def _stub_fetch_club_table(monkeypatch, *, reject_id=None):
    """Replaces _fetch_club_table with a stub that returns one fake row per
    call, unless called with the id in `reject_id` (simulates a failed
    strict title-check on a bad auto-resolved guess) -- and records every
    call's (club, club_id, strict_title_check) for assertions."""
    calls = []

    def stub(club, club_id, *, strict_title_check=False):
        calls.append((club, club_id, strict_title_check))
        if club_id == reject_id:
            raise ValueError("title mismatch")
        return pd.DataFrame([{
            "team": club, "coach": "Some Coach",
            "start_date": pd.Timestamp("2020-01-01"), "end_date": None,
        }])

    monkeypatch.setattr(fch, "_fetch_club_table", stub)
    return calls


def test_new_team_gets_auto_resolved_verified_strictly_and_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(fch.config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fch, "AUTO_RESOLVED_PATH", tmp_path / "auto_ids.json")
    monkeypatch.setattr(fch.config, "COACH_HISTORY_RESOLVED_CSV", str(tmp_path / "resolved.csv"))
    monkeypatch.setattr(fch.config, "COACH_HISTORY_MANUAL_CSV", str(tmp_path / "manual.csv"))
    monkeypatch.setattr(fch.config, "CLUB_TRANSFERMARKT_ID", {})  # nothing configured
    monkeypatch.setattr(fch, "search_club_id", lambda club: 999)

    calls = _stub_fetch_club_table(monkeypatch)
    pd.DataFrame({"team": ["NewTeam"]}).to_parquet(tmp_path / "fbref_schedule.parquet", index=False)

    result = fch.fetch_coach_history()

    assert len(result) == 1
    assert calls == [("NewTeam", 999, True)]  # strict_title_check=True for an auto-resolved id
    cached = json.loads((tmp_path / "auto_ids.json").read_text())
    assert cached == {"NewTeam": 999}


def test_cached_auto_resolved_id_skips_search_on_later_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(fch.config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fch, "AUTO_RESOLVED_PATH", tmp_path / "auto_ids.json")
    (tmp_path / "auto_ids.json").write_text(json.dumps({"NewTeam": 999}))
    monkeypatch.setattr(fch.config, "COACH_HISTORY_RESOLVED_CSV", str(tmp_path / "resolved.csv"))
    monkeypatch.setattr(fch.config, "COACH_HISTORY_MANUAL_CSV", str(tmp_path / "manual.csv"))
    monkeypatch.setattr(fch.config, "CLUB_TRANSFERMARKT_ID", {})

    def fail_if_called(club):
        raise AssertionError("search_club_id should not be called for an already-cached team")
    monkeypatch.setattr(fch, "search_club_id", fail_if_called)

    calls = _stub_fetch_club_table(monkeypatch)
    pd.DataFrame({"team": ["NewTeam"]}).to_parquet(tmp_path / "fbref_schedule.parquet", index=False)

    fch.fetch_coach_history()

    # Cached id is trusted like a configured one -- not re-verified strictly.
    assert calls == [("NewTeam", 999, False)]


def test_rejected_auto_resolution_falls_back_to_failure_not_a_bad_id(monkeypatch, tmp_path):
    monkeypatch.setattr(fch.config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fch, "AUTO_RESOLVED_PATH", tmp_path / "auto_ids.json")
    monkeypatch.setattr(fch.config, "COACH_HISTORY_RESOLVED_CSV", str(tmp_path / "resolved.csv"))
    monkeypatch.setattr(fch.config, "COACH_HISTORY_MANUAL_CSV", str(tmp_path / "manual.csv"))
    monkeypatch.setattr(fch.config, "CLUB_TRANSFERMARKT_ID", {})
    monkeypatch.setattr(fch, "search_club_id", lambda club: 999)  # a wrong guess

    _stub_fetch_club_table(monkeypatch, reject_id=999)  # simulates a strict title-check failure
    pd.DataFrame({"team": ["NewTeam"]}).to_parquet(tmp_path / "fbref_schedule.parquet", index=False)

    result = fch.fetch_coach_history()

    assert len(result) == 0
    # A rejected guess must NOT be cached -- caching a wrong id would be
    # worse than not resolving at all (silently wrong forever vs. a
    # visible gap that still needs a manual entry).
    assert not (tmp_path / "auto_ids.json").exists()
