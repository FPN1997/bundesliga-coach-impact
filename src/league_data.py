"""
Match data for the coaching-change study across several leagues:
data/processed/league_matches.parquet, one row per team-match with league,
season, team, opponent, venue, date, goals, points, xG and the coach.

- Bundesliga rows come straight from match_dataset.parquet (the full
  FBref + Understat + Transfermarkt pipeline), so its numbers don't move.
- Other leagues (config.OTHER_LEAGUES) use Understat for results and xG --
  the FBref scrape is far too slow to repeat for four more leagues, and the
  study doesn't need formations -- and coaches from
  src/fetch_league_coaches.py (Transfermarkt, same club names as Understat).
- Odds for fixture difficulty come from football-data.co.uk. Its club names
  ("Man United", "Ath Madrid") differ from Understat's, so they're matched
  by fixtures -- the same date and the same score -- rather than by spelling:
  within each league-season, every football-data name must pair with
  exactly one Understat name, backed by at least MIN_NAME_VOTES fixtures and
  MIN_VOTE_LEAD times as many as any other candidate.
  Per season, because Understat renames clubs over time (Parma became
  "Parma Calcio 1913" after its 2015 re-founding) while football-data
  doesn't.

A league joins the study only once LEAGUE_MIN_COACH_COVERAGE of its
team-matches have a coach (`included_leagues`); until then it's listed as
excluded, never half-included -- a missing sacking would turn into a
"kept their coach" comparison window.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.build_dataset import _assign_coach
from src.data_guard import existing_parquet_row_count, guard_against_shrinkage
from src.fetch_odds import ODDS_COLUMNS, load_odds
from src.fixture_difficulty import fixture_ease

log = logging.getLogger(__name__)

# paths resolve at call time, so a test (or the end-to-end fixture run) that
# points config's directories elsewhere never touches the real files


def understat_path() -> Path:
    return Path(config.RAW_DIR) / "understat_other_leagues.parquet"


def odds_path() -> Path:
    return Path(config.RAW_DIR) / "football_data_odds_other_leagues.parquet"


def out_path() -> Path:
    return Path(config.PROCESSED_DIR) / "league_matches.parquet"


def coaches_path() -> Path:
    return Path(config.PROCESSED_DIR) / "coach_history_other_leagues.csv"
MIN_NAME_VOTES = 3      # fixtures backing a name pairing (a club has ~4 by matchday 4)
MIN_VOTE_LEAD = 2.0     # ... and at least this many times the runner-up's (several matches on one
                        # day often share a score, so every name collects a few chance votes)


# --- fetching (no Transfermarkt involved) ------------------------------------

def fetch_understat_other_leagues() -> pd.DataFrame:
    """Match-level results and xG for config.OTHER_LEAGUES (soccerdata caches past seasons)."""
    import soccerdata as sd

    df = sd.Understat(leagues=config.OTHER_LEAGUES, seasons=config.SEASONS).read_team_match_stats()
    df = df.reset_index()
    df["season"] = df["season"].astype(str)
    guard_against_shrinkage(understat_path(), existing_parquet_row_count(understat_path()), len(df))
    understat_path().parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(understat_path(), index=False)
    log.info("Saved %d Understat matches for %s -> %s", len(df), ", ".join(config.OTHER_LEAGUES),
             understat_path())
    return df


def match_names_by_fixtures(fd: pd.DataFrame, us: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """football-data -> Understat club names within one league, from matches
    both sources agree on (date, home goals, away goals). Returns the mapping
    and the football-data names that couldn't be paired unambiguously."""
    keys = ["date", "FTHG", "FTAG"]
    joined = fd.merge(us.rename(columns={"home_team": "us_home", "away_team": "us_away",
                                         "home_goals": "FTHG", "away_goals": "FTAG"}), on=keys)
    votes = pd.concat([joined[["home_team", "us_home"]].set_axis(["fd", "us"], axis=1),
                       joined[["away_team", "us_away"]].set_axis(["fd", "us"], axis=1)])
    counts = votes.value_counts().rename("n").reset_index().sort_values("n", ascending=False)
    counts["runner_up"] = counts.groupby("fd")["n"].transform(lambda n: n.iloc[1] if len(n) > 1 else 0)
    best = counts.drop_duplicates("fd")
    best = best[(best["n"] >= MIN_NAME_VOTES) & (best["n"] >= MIN_VOTE_LEAD * best["runner_up"])]
    best = best[~best["us"].duplicated(keep=False)]  # two fd names claiming one club: neither
    mapping = dict(zip(best["fd"], best["us"], strict=True))
    unmatched = sorted((set(fd["home_team"]) | set(fd["away_team"])) - set(mapping))
    return mapping, unmatched


def _strip_bom_from_match_history_cache() -> int:
    """football-data.co.uk's 2021-22 Premier League file starts with a UTF-8
    byte-order mark. soccerdata then can't find its first column ("Div"),
    fails on that season read alone, and silently drops it when reading many
    seasons at once. Strip the mark from the cached copies; returns how many."""
    from soccerdata._config import DATA_DIR

    fixed = 0
    for path in (Path(DATA_DIR) / "MatchHistory").glob("*.csv"):
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            path.write_bytes(data[3:])
            fixed += 1
    return fixed


def _read_match_history(league: str, season: str) -> pd.DataFrame:
    """One league-season (cached by soccerdata). Read one at a time, so a bad
    file fails loudly instead of vanishing from a multi-season read."""
    import soccerdata as sd

    try:
        return sd.MatchHistory(leagues=league, seasons=[season]).read_games().reset_index()
    except KeyError:
        if not _strip_bom_from_match_history_cache():
            raise
        log.info("Stripped a byte-order mark from a cached football-data file; reading %s %s again",
                 league, season)
        return sd.MatchHistory(leagues=league, seasons=[season]).read_games().reset_index()


def fetch_odds_other_leagues() -> pd.DataFrame:
    """Odds for config.OTHER_LEAGUES, club names converted to Understat's."""
    raw = pd.concat([_read_match_history(league, season)
                     for league in config.OTHER_LEAGUES for season in config.SEASONS], ignore_index=True)
    for new, old in [("AvgH", "BbAvH"), ("AvgD", "BbAvD"), ("AvgA", "BbAvA")]:
        if old in raw.columns:
            raw[new] = raw[new].fillna(raw[old]) if new in raw.columns else raw[old]
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    us = pd.read_parquet(understat_path())
    us["date"] = pd.to_datetime(us["date"]).dt.normalize()

    raw["season"] = raw["season"].astype(str)
    us["season"] = us["season"].astype(str)
    frames = []
    for (league, season), fd in raw.groupby(["league", "season"]):
        mapping, unmatched = match_names_by_fixtures(
            fd, us[(us["league"] == league) & (us["season"] == season)])
        if unmatched:
            log.warning("%s %s: no confident Understat match for football-data names %s -- their "
                        "matches get no fixture rating", league, season, unmatched)
        out = fd[["league", "season", "date", "home_team", "away_team", "FTR"]].copy()
        for c in ODDS_COLUMNS:
            out[c] = fd[c] if c in fd.columns else np.nan
        out["home_team"] = out["home_team"].map(mapping)
        out["away_team"] = out["away_team"].map(mapping)
        frames.append(out.dropna(subset=["home_team", "away_team"]))
    odds = pd.concat(frames, ignore_index=True)
    odds["season"] = odds["season"].astype(str)
    guard_against_shrinkage(odds_path(), existing_parquet_row_count(odds_path()), len(odds))
    odds.to_parquet(odds_path(), index=False)
    log.info("Saved %d matches of odds for %s -> %s", len(odds), ", ".join(config.OTHER_LEAGUES), odds_path())
    return odds


def fetch_other_leagues() -> None:
    """The weekly refresh's data step for the other leagues: results, xG and odds."""
    fetch_understat_other_leagues()
    fetch_odds_other_leagues()


# --- building ----------------------------------------------------------------

def understat_team_rows(us: pd.DataFrame) -> pd.DataFrame:
    """Understat's one-row-per-match -> one row per team-match."""
    us = us.assign(date=pd.to_datetime(us["date"]).dt.normalize(), season=us["season"].astype(str))
    sides = []
    for side, other, venue in (("home", "away", "Home"), ("away", "home", "Away")):
        sides.append(pd.DataFrame({
            "league": us["league"], "season": us["season"], "date": us["date"],
            "team": us[f"{side}_team"], "opponent": us[f"{other}_team"], "venue": venue,
            "gf": us[f"{side}_goals"], "ga": us[f"{other}_goals"],
            "xg": us[f"{side}_xg"], "xga": us[f"{other}_xg"],
        }))
    rows = pd.concat(sides, ignore_index=True).dropna(subset=["gf", "ga"])
    rows["points"] = np.select([rows["gf"] > rows["ga"], rows["gf"] == rows["ga"]], [3, 1], 0)
    return rows.sort_values(["league", "team", "date"]).reset_index(drop=True)


def all_odds() -> pd.DataFrame | None:
    """Bundesliga and other-league odds in one frame, with a league column."""
    frames = []
    bl = load_odds()
    if bl is not None:
        frames.append(bl.assign(league=config.LEAGUE))
    if odds_path().exists():
        frames.append(pd.read_parquet(odds_path()))
    return pd.concat(frames, ignore_index=True) if frames else None


def fixture_ease_by_league(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.Series:
    """src/fixture_difficulty.py's rating, fit separately within each league."""
    ease = pd.Series(np.nan, index=matches.index, name="fixture_ease")
    for league, m in matches.groupby("league"):
        o = odds[odds["league"] == league]
        if len(o):
            ease[m.index] = fixture_ease(m, o)
    return ease


def build_league_matches() -> pd.DataFrame:
    bl = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet").assign(league=config.LEAGUE)
    frames = [bl]
    if understat_path().exists():
        other = understat_team_rows(pd.read_parquet(understat_path()))
        coaches = (pd.read_csv(coaches_path(), parse_dates=["start_date", "end_date"])
                   if coaches_path().exists() else pd.DataFrame(columns=["league", "team", "coach",
                                                                        "start_date", "end_date"]))
        for league, m in other.groupby("league"):
            c = coaches[coaches["league"] == league]
            if c.empty:
                frames.append(m.assign(coach=pd.NA))
                continue
            frames.append(_assign_coach(m.copy(), c.drop(columns="league")))
    out = pd.concat(frames, ignore_index=True)
    out["season"] = out["season"].astype(str)
    out_path().parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path(), index=False)
    cov = coach_coverage(out)
    log.info("Saved %d team-matches across %d leagues -> %s", len(out), out["league"].nunique(), out_path())
    for league, share in cov.items():
        log.info("  %-20s coach known for %5.1f%% of team-matches%s", league, 100 * share,
                 "" if share >= config.LEAGUE_MIN_COACH_COVERAGE else " -- not in the study yet")
    return out


def coach_coverage(matches: pd.DataFrame) -> dict[str, float]:
    return {league: float(m["coach"].notna().mean()) for league, m in matches.groupby("league")}


def included_leagues(matches: pd.DataFrame) -> list[str]:
    """Leagues whose coach data is complete enough for the study."""
    return sorted(league for league, share in coach_coverage(matches).items()
                  if share >= config.LEAGUE_MIN_COACH_COVERAGE)
