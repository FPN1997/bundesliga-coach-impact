"""
Central configuration for the Bundesliga Coach Impact Analyzer.

Adjust SEASONS to control how far back the pipeline pulls data. FBref and
Understat both have reliable formation / PPDA coverage from roughly the
2014-15 season onward; going back further is possible for goals/results but
formations and PPDA get sparse.
"""

from __future__ import annotations

# soccerdata's standardized league code for the top German flight.
# Verify with: soccerdata.FBref.available_leagues() if this ever errors.
LEAGUE = "GER-Bundesliga"

# Seasons to pull, soccerdata format ("2019-2020" style strings).
# Keep this list short while you're developing the pipeline -- each season
# is several dozen HTTP requests against FBref/Understat.
SEASONS = [
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
    "2026-2027",  # ongoing season -- partial data, early-season rows will
                  # be thin (few matches played), which coach_impact.py's
                  # matches_before/matches_after columns already surface
]

# Where raw scraped tables and the merged analysis dataset land.
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs"

# Manual coach-tenure fallback file. fetch_coach_history.py scrapes each
# club's Transfermarkt "Trainerhistorie" (manager history) page first (see
# CLUB_TRANSFERMARKT_ID below) and writes anything it can't resolve into
# this file as blank rows for you to fill in by hand. (Wikipedia was tried
# first -- English Wikipedia turned out not to have "List of <club>
# managers" pages for Bundesliga clubs; every guessed URL 404'd live.)
COACH_HISTORY_MANUAL_CSV = "data/coach_history_manual.csv"
COACH_HISTORY_RESOLVED_CSV = "data/processed/coach_history.csv"

# FBref canonical team name -> Transfermarkt numeric club id (the slug text
# in a Transfermarkt URL is cosmetic; only this id is looked up). Verified
# live for Bayern Munich (27) and Heidenheim (2036); the rest are Transfer-
# markt's well-known stable ids for these clubs but weren't individually
# re-verified here -- fetch_coach_history.py checks the fetched page's
# <title> against the club name and logs a loud warning if an id looks like
# it resolved to the wrong club, so a bad id won't fail silently.
# Look up a fresh/missing id via Transfermarkt's search:
#   https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query=<club name>
# then take the numeric id from the non-reserve, non-youth result's URL
# (.../startseite/verein/<id>).
# Keys MUST match the team names FBref/soccerdata actually returns (verified
# live against the 2023-24 season -- see README "Known rough edges"). These
# differ from Understat's naming, hence TEAM_NAME_MAP below.
CLUB_TRANSFERMARKT_ID = {
    "Bayern Munich": 27,
    "Dortmund": 16,
    "RB Leipzig": 23826,
    "Leverkusen": 15,
    "Frankfurt": 24,
    "Wolfsburg": 82,
    "Gladbach": 18,
    "Union Berlin": 89,
    "Freiburg": 60,
    "Mainz 05": 39,
    "Hoffenheim": 533,
    "Werder Bremen": 86,
    "Stuttgart": 79,
    "Köln": 3,
    "Augsburg": 167,
    "Hertha BSC": 44,          # FBref calls this "Hertha BSC", not "Hertha Berlin"
    "Schalke 04": 33,
    "Bochum": 80,
    "Darmstadt 98": 105,
    "Heidenheim": 2036,
    "St Pauli": 35,            # FBref: no period after "St"
    "Holstein Kiel": 269,
    # Added after running the full 6-season pull surfaced these (relegated
    # before the 2023-24 season used for initial testing, so missed the
    # first pass). All four ids verified live against the page <title>.
    "Arminia": 10,             # Arminia Bielefeld
    "Düsseldorf": 38,          # Fortuna Düsseldorf
    "Greuther Fürth": 65,      # SpVgg Greuther Fürth (an initial guess, id 565,
                               # turned out to resolve to a different club --
                               # this is the verified correct one)
    "Paderborn 07": 127,       # SC Paderborn 07
    # Added when SEASONS grew to include 2025-26/2026-27 and these two
    # newly-promoted clubs showed up. Both ids verified live.
    "Hamburger SV": 41,
    "Elversberg": 64,          # SV 07 Elversberg
}

# Understat team name -> canonical FBref team name (the canonical form used
# everywhere else in this project, including CLUB_TRANSFERMARKT_ID above). Verified
# live for the 2023-24 season; if you widen SEASONS and a newly-promoted
# club is missing, build_dataset.py will log its unmapped Understat name --
# add it here.
TEAM_NAME_MAP = {
    "Augsburg": "Augsburg",
    "Bayer Leverkusen": "Leverkusen",
    "Bayern Munich": "Bayern Munich",
    "Bochum": "Bochum",
    "Borussia Dortmund": "Dortmund",
    "Borussia M.Gladbach": "Gladbach",
    "Darmstadt": "Darmstadt 98",
    "Eintracht Frankfurt": "Frankfurt",
    "FC Cologne": "Köln",
    "FC Heidenheim": "Heidenheim",
    "Freiburg": "Freiburg",
    "Hoffenheim": "Hoffenheim",
    "Mainz 05": "Mainz 05",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Union Berlin": "Union Berlin",
    "VfB Stuttgart": "Stuttgart",
    "Werder Bremen": "Werder Bremen",
    "Wolfsburg": "Wolfsburg",
    "Hertha Berlin": "Hertha BSC",       # fixed: FBref uses "Hertha BSC"
    "Schalke 04": "Schalke 04",
    "St. Pauli": "St Pauli",             # fixed: FBref has no period after "St"
    "Holstein Kiel": "Holstein Kiel",
    # Added after the full 6-season pull surfaced these (relegated before
    # the 2023-24 season used for initial testing).
    "Arminia Bielefeld": "Arminia",
    "Fortuna Duesseldorf": "Düsseldorf",
    "Greuther Fuerth": "Greuther Fürth",
    "Paderborn": "Paderborn 07",
}

# football-data.co.uk team name -> canonical FBref team name, for the
# betting-market benchmark (src/market_benchmark.py). Only names that differ
# are listed. Resolved with src/team_name_matcher.py and checked by hand
# against all 8 seasons: the matcher got 27/28 on its own and correctly
# refused to guess "Bielefeld" (FBref: "Arminia", no shared substring).
FOOTBALL_DATA_NAME_MAP = {
    "Bielefeld": "Arminia",
    "Darmstadt": "Darmstadt 98",
    "Ein Frankfurt": "Frankfurt",
    "FC Koln": "Köln",
    "Fortuna Dusseldorf": "Düsseldorf",
    "Greuther Furth": "Greuther Fürth",
    "Hamburg": "Hamburger SV",
    "Hertha": "Hertha BSC",
    "M'gladbach": "Gladbach",
    "Mainz": "Mainz 05",
    "Paderborn": "Paderborn 07",
}

# FBref's team-schedule pages mix in cup/European fixtures. League matches
# are the ones whose 'round' column matches this pattern.
FBREF_LEAGUE_ROUND_PATTERN = r"^Matchweek\s+\d+$"

# Rolling-window size (in matches) used to compare "before" vs "after" a
# coaching change in coach_impact.py.
IMPACT_WINDOW = 8

# --- Formation/outcome predictor (src/features.py, src/outcome_predictor.py) ---

# Rolling-form window size (in matches) for form_ppg/form_goal_diff/etc.
# Also means each team's first FEATURE_ROLLING_WINDOW matches of the whole
# dataset (not just per-season) have no features and get dropped.
FEATURE_ROLLING_WINDOW = 5

# Seasons held out as the test set -- kept in date order, trained on
# everything strictly before the first one. 2025-2026 is a full completed
# season (fair test); 2026-2027 is included too since it's what you'd
# actually want predictions for, but has very few matches so treat its
# individual metrics as low-confidence.
TEST_SEASONS = ["2025-2026", "2026-2027"]

MODEL_DIR = "outputs/models"

# CV folds for src/tune_hyperparameters.py and src/tune_optuna.py's
# TimeSeriesSplit. Verified live (see README "Hyperparameter tuning"):
# sklearn's TimeSeriesSplit(n_splits=N) divides the training set into N+1
# equal-size chunks, so RAISING N shrinks every fold's chunk size,
# including the earliest, already-data-starved training fold -- it is NOT
# simply "more folds at the same size." Whether more, smaller folds average
# out to a better or worse (noisier) CV signal than fewer, larger ones is
# an empirical question, not something to assume either way -- which is
# exactly why this got tested rather than just bumped and left undocumented.
N_CV_SPLITS = 10

# Embargo (in rows) between each TimeSeriesSplit fold's train and validation
# slice, for src/tune_hyperparameters.py's gap= option (`bundesliga tune --method embargoed`).
# The training feature table has one row per team per match (~2x a raw
# fixture list, home + away), and a Bundesliga matchday is 9 fixtures ->
# 18 rows -- verified live against the actual training set: 3388 train rows
# over ~190 matchdays is ~17.8 rows/matchday, matching that arithmetic. One
# full matchday's worth of embargo, not an arbitrary round number.
CV_EMBARGO_GAP = 18
