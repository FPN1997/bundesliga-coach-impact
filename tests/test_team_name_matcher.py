"""
Validates src/team_name_matcher.py against every (Understat name -> FBref
name) pair already known correct in config.TEAM_NAME_MAP -- the actual
empirical check this module's approach was chosen from (see its docstring
for why difflib.get_close_matches was rejected in favor of comparing
.ratio() values directly).
"""

from __future__ import annotations

import config
from src.team_name_matcher import best_match


def test_resolves_every_known_team_name_pair_correctly():
    fbref_names = sorted(set(config.TEAM_NAME_MAP.values()))
    failures = []
    for understat_name, expected_fbref_name in config.TEAM_NAME_MAP.items():
        got, score, runner_up = best_match(understat_name, fbref_names)
        if got != expected_fbref_name:
            failures.append((understat_name, expected_fbref_name, got, score, runner_up))
    assert not failures, (
        f"{len(failures)}/{len(config.TEAM_NAME_MAP)} pairs failed to auto-resolve: {failures}"
    )


def test_translation_pair_with_no_shared_substring_still_resolves():
    # The single hardest case in the validated set: "FC Cologne" and "Köln"
    # share no obvious characters (an English-name-vs-German-name
    # translation, not a spelling/transliteration difference like the
    # other pairs). Called out explicitly so a future threshold change
    # that breaks this specific case fails loudly, not just as one entry
    # in the aggregate test above.
    candidates = ["Köln", "Wolfsburg", "Holstein Kiel", "Dortmund", "Hoffenheim"]
    match, score, runner_up = best_match("FC Cologne", candidates)
    assert match == "Köln"
    assert score > runner_up


def test_no_match_below_min_score_returns_none():
    # Confirmed by hand this scores 0.083 against both candidates, well
    # below MIN_SCORE -- unlike a superficially-similar-looking string
    # ("Completely Unrelated Name Xyz" vs "Dortmund" actually scores 0.27
    # from incidental shared letters, above threshold -- checked, not
    # assumed, which is exactly why this uses a string with no letters in
    # common with either candidate instead).
    match, _, _ = best_match("Zzzzz Qqqqq", ["Bayern Munich", "Dortmund"])
    assert match is None


def test_ambiguous_tied_candidates_return_none_rather_than_guess():
    # Constructed to be genuinely symmetric: "AAAA XXXX" and "YYYY BBBB"
    # each share exactly one word with "AAAA BBBB", so they score
    # identically -- confirmed by hand (both 0.556) before writing this
    # assertion, not assumed.
    match, score, runner_up = best_match("AAAA BBBB", ["AAAA XXXX", "YYYY BBBB"])
    assert match is None
    assert score == runner_up
