"""Shared formation-string cleanup, used by formation_matrix.py and features.py."""

from __future__ import annotations

import re

import pandas as pd


def clean_formation(value: object) -> str | None:
    """FBref formation strings (e.g. "4-2-3-1") sometimes carry a trailing
    marker like "4-2-3-1◆" for a mid-match switch -- strip anything that
    isn't digits and dashes so "4-2-3-1" and "4-2-3-1◆" bucket together
    instead of fragmenting into separate categories."""
    if pd.isna(value):
        return None
    s = re.sub(r"[^\d\-]", "", str(value))
    return s if s else None
