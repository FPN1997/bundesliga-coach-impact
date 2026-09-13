"""
Shared "refuse to silently regress" guard for every fetch script
(fetch_coach_history.py, fetch_fbref.py, fetch_understat.py).

If a fresh scrape/fetch produces far fewer rows than what's already on
disk, that's much more likely to mean the fetch broke somehow -- a site
blocking requests, an API/HTML shape change, a bad or partial response --
than the underlying data actually shrinking. None of what this project
pulls (match results, coach tenure, xG history) loses rows over time in
reality, so a big drop is a red flag, not a valid new value.

This exists because that exact failure mode happened for real: Transfermarkt
started blocking every request with an AWS WAF challenge, every club's
coach-history fetch failed, and the pipeline silently wrote a 1-row file
over what had been ~1600 real rows -- a warning log, not a crash, all the
way down to coach_impact.py quietly finding zero valid coaching changes.
See fetch_coach_history.py's docstring for the full incident.
"""

from __future__ import annotations

from pathlib import Path


def guard_against_shrinkage(
    out_path: Path,
    existing_row_count: int,
    new_row_count: int,
    *,
    min_existing_rows: int = 20,
    max_drop_fraction: float = 0.5,
    context: str = "",
) -> None:
    """Raise RuntimeError rather than let a caller overwrite `out_path` if
    `new_row_count` is a suspiciously large drop from `existing_row_count`.

    Files with `existing_row_count <= min_existing_rows` are exempt --
    not worth guarding (a fresh checkout, a deliberately small test
    dataset), and guarding them would just make early iteration annoying.
    `context` is appended to the error message -- use it for anything that
    would help someone reading the failure understand what to check first
    (e.g. how many sub-fetches failed, out of how many attempted).
    """
    if existing_row_count <= min_existing_rows:
        return
    if new_row_count < existing_row_count * (1 - max_drop_fraction):
        raise RuntimeError(
            f"Refusing to overwrite {out_path} ({existing_row_count} rows) with "
            f"this run's result ({new_row_count} rows) -- that's a >"
            f"{max_drop_fraction:.0%} drop, which almost certainly means the fetch "
            f"broke rather than the underlying data actually shrinking."
            + (f" {context}" if context else "")
            + f" Fix the underlying failure before re-running -- the existing file "
              f"has been left untouched."
        )


def existing_csv_row_count(path: Path) -> int:
    """Row count of an existing CSV, or 0 if it doesn't exist yet -- minus
    one for the header, so it's directly comparable to a DataFrame's len()."""
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f) - 1


def existing_parquet_row_count(path: Path) -> int:
    """Row count of an existing parquet file, or 0 if it doesn't exist yet.
    Reads only the file's metadata (via pyarrow), not the actual data --
    cheap even for a large file."""
    if not path.exists():
        return 0
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows
