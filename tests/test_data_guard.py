"""
Tests for the shared shrinkage guard (src/data_guard.py) used by all three
fetch scripts. See test_fetch_coach_history.py for the guard exercised
through fetch_coach_history.py specifically, and its docstring for the
real incident that made this necessary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_guard import (
    existing_csv_row_count,
    existing_parquet_row_count,
    guard_against_shrinkage,
)


def test_guard_raises_on_a_big_drop():
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        guard_against_shrinkage(Path("dummy.csv"), existing_row_count=1000, new_row_count=100)


def test_guard_allows_growth_or_a_small_drop():
    guard_against_shrinkage(Path("dummy.csv"), existing_row_count=1000, new_row_count=1200)
    guard_against_shrinkage(Path("dummy.csv"), existing_row_count=1000, new_row_count=900)


def test_guard_exempts_small_existing_files():
    # Below min_existing_rows -- a fresh checkout or small test file, not
    # worth guarding (would just make early iteration annoying).
    guard_against_shrinkage(Path("dummy.csv"), existing_row_count=5, new_row_count=0)


def test_guard_boundary_at_exactly_half():
    # max_drop_fraction=0.5 means new_row_count must be >= 50% of existing
    # to pass -- exactly 50% should NOT raise (the check is a strict "<").
    guard_against_shrinkage(Path("dummy.csv"), existing_row_count=100, new_row_count=50)
    with pytest.raises(RuntimeError):
        guard_against_shrinkage(Path("dummy.csv"), existing_row_count=100, new_row_count=49)


def test_guard_error_message_includes_context():
    with pytest.raises(RuntimeError, match="3/10 teams failed"):
        guard_against_shrinkage(
            Path("dummy.csv"), existing_row_count=1000, new_row_count=100,
            context="3/10 teams failed this run.",
        )


def test_existing_csv_row_count_missing_file_is_zero(tmp_path):
    assert existing_csv_row_count(tmp_path / "does_not_exist.csv") == 0


def test_existing_csv_row_count_counts_data_rows_not_header(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    assert existing_csv_row_count(path) == 3


def test_existing_parquet_row_count_missing_file_is_zero(tmp_path):
    assert existing_parquet_row_count(tmp_path / "does_not_exist.parquet") == 0


def test_existing_parquet_row_count_matches_dataframe_length(tmp_path):
    path = tmp_path / "data.parquet"
    pd.DataFrame({"a": range(42)}).to_parquet(path, index=False)
    assert existing_parquet_row_count(path) == 42
