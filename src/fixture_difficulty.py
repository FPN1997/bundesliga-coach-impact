"""
How hard was a run of fixtures?

The coaching-change analysis (coach_change_effect.py) compares the 8 matches
before a change with the 8 after it. Who a team plays, and where, differs
between those two stretches -- for sackings and for the comparison windows
alike. A club that sacks its coach just before four home games against
bottom-half sides will "improve" for reasons that have nothing to do with
the coach. This module rates every fixture so that can be adjusted for.

A fixture's rating ("ease") is the points an AVERAGE team would expect from
it: it depends only on the opponent's strength and on home/away, never on
the team itself. That separation matters. The betting odds for a sacked
team's own matches already price in the new coach, so adjusting for them
would subtract part of the very effect being measured. The opponent's
strength is a property of another club, and the venue is fixed by the
schedule long before any sacking.

- Opponent strength: each club's season-average market rating -- the
  expected points the closing odds gave it across all its matches that
  season (market average across bookmakers, margin removed). A market
  rating rather than results, so a club's luck doesn't make it look
  stronger or weaker than it was.
- Ease: fit, over every team-match with odds,
      market expected points = a + b*own rating + c*opponent rating + d*home
  and evaluate it with the team's own rating set to the league average:
      ease = a + b*mean rating + c*opponent rating + d*home.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOME, DRAW, AWAY = ["AvgCH", "AvgCD", "AvgCA"]
FALLBACK = ["AvgH", "AvgD", "AvgA"]  # pre-match market average, if closing is missing


def team_match_expected_points(odds: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (season, team, opponent, venue) with the
    market's expected points for that team (3*P(win) + P(draw), margin removed)."""
    closing = odds[[HOME, DRAW, AWAY]].to_numpy(dtype=float)
    early = odds.reindex(columns=FALLBACK).to_numpy(dtype=float)
    decimal = np.where(np.isnan(closing).any(axis=1, keepdims=True), early, closing)
    inv = 1.0 / decimal
    p = inv / inv.sum(axis=1, keepdims=True)  # columns: home win, draw, away win
    season = odds["season"].astype(str).to_numpy()
    home = pd.DataFrame({"season": season, "team": odds["home_team"].to_numpy(),
                         "opponent": odds["away_team"].to_numpy(), "home": 1.0,
                         "xpts": 3 * p[:, 0] + p[:, 1]})
    away = pd.DataFrame({"season": season, "team": odds["away_team"].to_numpy(),
                         "opponent": odds["home_team"].to_numpy(), "home": 0.0,
                         "xpts": 3 * p[:, 2] + p[:, 1]})
    return pd.concat([home, away], ignore_index=True).dropna(subset=["xpts"])


def fit_ease_model(odds: pd.DataFrame) -> tuple[pd.Series, dict]:
    """(season-team ratings, fitted coefficients) from the odds."""
    long = team_match_expected_points(odds)
    ratings = long.groupby(["season", "team"])["xpts"].mean().rename("rating")
    long = long.join(ratings, on=["season", "team"]).join(
        ratings.rename("opp_rating"), on=["season", "opponent"])
    X = np.column_stack([np.ones(len(long)), long["rating"], long["opp_rating"], long["home"]])
    a, b, c, d = np.linalg.lstsq(X, long["xpts"].to_numpy(), rcond=None)[0]
    return ratings, {"intercept": a, "own": b, "opponent": c, "home": d,
                     "mean_rating": float(ratings.mean())}


def fixture_ease(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.Series:
    """Ease of each row of `matches` (columns season, opponent, venue),
    aligned to its index; NaN where the opponent has no rating that season."""
    ratings, fit = fit_ease_model(odds)
    opp = pd.MultiIndex.from_arrays([matches["season"].astype(str), matches["opponent"]])
    opp_rating = ratings.reindex(opp).to_numpy()
    home = (matches["venue"] == "Home").to_numpy(dtype=float)
    ease = (fit["intercept"] + fit["own"] * fit["mean_rating"]
            + fit["opponent"] * opp_rating + fit["home"] * home)
    return pd.Series(ease, index=matches.index, name="fixture_ease")
