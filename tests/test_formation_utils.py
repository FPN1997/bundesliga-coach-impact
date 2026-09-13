import math

from src.formation_utils import clean_formation


def test_clean_formation_passes_plain_formation_through():
    assert clean_formation("4-2-3-1") == "4-2-3-1"


def test_clean_formation_strips_midmatch_switch_marker():
    # FBref appends a marker (e.g. "◆") to flag a formation change during
    # the match. Without stripping it, "4-4-2" and "4-4-2◆" would be
    # treated as two different formations everywhere downstream.
    assert clean_formation("4-4-2◆") == "4-4-2"


def test_clean_formation_strips_arbitrary_non_digit_dash_characters():
    # Verified live against every formation string FBref actually returns
    # (data/raw/fbref_schedule.parquet): the only marker seen in practice is
    # "◆". This checks the broader "strip anything but digits/dashes" rule
    # holds for letters/spaces too, not just that one specific character --
    # but note it's a blunt rule: a hypothetical note containing its own
    # digits (never observed in real data) would leak those digits into the
    # result rather than being cleanly removed.
    assert clean_formation("4-2-3-1x") == "4-2-3-1"
    assert clean_formation("4-2-3-1 extra") == "4-2-3-1"


def test_clean_formation_none_and_nan_return_none():
    assert clean_formation(None) is None
    assert clean_formation(math.nan) is None


def test_clean_formation_empty_string_returns_none():
    assert clean_formation("") is None
    assert clean_formation("◆◆◆") is None  # nothing but the marker left
