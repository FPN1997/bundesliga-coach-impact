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


def test_refuses_to_overwrite_a_large_history_with_a_tiny_one(monkeypatch, tmp_path):
    resolved_path = tmp_path / "coach_history.csv"
    manual_path = tmp_path / "coach_history_manual.csv"
    _write_existing_history(resolved_path, n_rows=100)
    pd.DataFrame(columns=["team", "coach", "start_date", "end_date"]).to_csv(
        manual_path, index=False
    )

    monkeypatch.setattr(fch.config, "COACH_HISTORY_RESOLVED_CSV", str(resolved_path))
    monkeypatch.setattr(fch.config, "COACH_HISTORY_MANUAL_CSV", str(manual_path))
    monkeypatch.setattr(fch.config, "CLUB_TRANSFERMARKT_ID", {"Alpha": 1, "Beta": 2})

    # Simulate every club's live fetch failing (the WAF-block scenario).
    monkeypatch.setattr(fch, "_fetch_club_table",
                         lambda club, club_id: (_ for _ in ()).throw(ValueError("blocked")))

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

    monkeypatch.setattr(fch.config, "COACH_HISTORY_RESOLVED_CSV", str(resolved_path))
    monkeypatch.setattr(fch.config, "COACH_HISTORY_MANUAL_CSV", str(manual_path))
    monkeypatch.setattr(fch.config, "CLUB_TRANSFERMARKT_ID", {"Alpha": 1})
    monkeypatch.setattr(fch, "_fetch_club_table",
                         lambda club, club_id: (_ for _ in ()).throw(ValueError("blocked")))

    # Small existing files (<=20 rows) aren't worth guarding -- this should
    # complete (writing 0 rows) rather than raise.
    result = fch.fetch_coach_history()
    assert len(result) == 0
