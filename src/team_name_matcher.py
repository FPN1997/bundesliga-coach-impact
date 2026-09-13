"""
Auto-resolve an Understat team name to its canonical FBref name via fuzzy
string matching, instead of requiring a hand-added `config.TEAM_NAME_MAP`
entry for every team -- the same recurring cost `fetch_coach_history.py`'s
Transfermarkt id resolution had (see that module's docstring for the
fuller rationale; the two are independent but address the same underlying
problem of "a new team shows up, something needs a hand-edit").

Validated against all 28 (Understat name -> FBref name) pairs already
known correct in `config.TEAM_NAME_MAP`: 28/28 resolve correctly using
`difflib.SequenceMatcher.ratio()` on a lightly normalized (lowercased,
umlaut-transliterated) string, picking the single best-scoring FBref name
by comparing `.ratio()` values directly -- deliberately NOT
`difflib.get_close_matches`, whose result order does not reliably match
descending `.ratio()`. Confirmed empirically, not a hypothetical
concern: for "FC Cologne", `get_close_matches` ranked "Wolfsburg" above
"Köln" despite Köln's actual ratio being higher (0.400 vs 0.316) -- an
real bug in trusting that function's ordering, caught by checking rather
than assuming stdlib behavior matched what the docs implied.

That same validation run is also why "FC Cologne" -> "Köln" -- an
English-name-vs-German-name translation with literally zero shared
characters in the obvious places -- turned out to be resolvable at all:
it scores 0.400, comfortably clear of the next-best candidate at 0.348.
Nothing here is hardcoded for that pair specifically; it just happens to
clear the same bar as everything else.
"""

from __future__ import annotations

import difflib
import re

_UMLAUT_TRANSLITERATIONS = [("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")]

# Thresholds picked from the validation run: the smallest winning score
# among all 28 known-correct pairs was 0.400 (FC Cologne -> Köln) and the
# smallest winner-to-runner-up margin was 0.052 (same pair). Both
# thresholds sit comfortably below those, so every previously-verified
# pair still resolves -- see tests/test_team_name_matcher.py.
MIN_SCORE = 0.25
MIN_MARGIN = 0.03


def _normalize(name: str) -> str:
    name = name.lower()
    for umlaut, ascii_form in _UMLAUT_TRANSLITERATIONS:
        name = name.replace(umlaut, ascii_form)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def best_match(name: str, candidates: list[str]) -> tuple[str | None, float, float]:
    """Best-scoring candidate for `name`, or None if no candidate clears
    MIN_SCORE or the top two are too close to call confidently (their
    margin is below MIN_MARGIN -- an ambiguous case, safer to leave
    unresolved than to guess). Returns (match_or_None, best_score,
    runner_up_score) -- the scores are included so a caller can log them
    for a human to sanity-check occasionally, not just the final verdict.
    """
    target = _normalize(name)
    scored = sorted(
        ((difflib.SequenceMatcher(None, target, _normalize(c)).ratio(), c) for c in candidates),
        reverse=True,
    )
    if not scored:
        return None, 0.0, 0.0
    best_score, best_name = scored[0]
    runner_up_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < MIN_SCORE or (best_score - runner_up_score) < MIN_MARGIN:
        return None, best_score, runner_up_score
    return best_name, best_score, runner_up_score
