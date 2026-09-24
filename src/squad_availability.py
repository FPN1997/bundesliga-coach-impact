"""
How much of a squad was available at each match: injuries and new signings.

Two measures per team-match, both as a share of the squad's market value
(Transfermarkt, via src/fetch_squads.py), so a missing star counts for
more than a missing squad player:

  - injured_share: the value of squad members injured on the match date.
  - new_signings_share: the value of that season's January arrivals.

"Squad members" are each club-season's injury-tracked players (the
INJURY_TOP_N most valuable), with one refinement: a January arrival only
joins the squad from WINTER_ARRIVALS_AVAILABLE (mid-January) on. The
transfers page doesn't give arrival dates; the window opens 1 January, the
Bundesliga restarts mid-month and the deadline is around 1 February, so
mid-January is the middle estimate. It also means an arrival's injuries at
their previous club, before they arrived, don't count against the new one.

An injury covers every match from its start date to its end date
inclusive. Transfermarkt occasionally has no end date; then the injury is
assumed to last one week per game missed (two weeks if that's unknown too).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINTER_ARRIVALS_AVAILABLE = "01-15"  # month-day in the season's second calendar year


def _injury_spans(injuries: pd.DataFrame) -> pd.DataFrame:
    inj = injuries.dropna(subset=["start"]).copy()
    weeks = inj["games_missed"].fillna(2).clip(lower=1)
    fallback_end = inj["start"] + pd.to_timedelta(weeks * 7, unit="D")
    inj["end"] = inj["end"].fillna(fallback_end)
    return inj[["player_id", "start", "end"]]


def match_availability(matches: pd.DataFrame, squads: pd.DataFrame,
                       injuries: pd.DataFrame) -> pd.DataFrame:
    """injured_share and new_signings_share for each row of `matches`
    (columns team, season, date), aligned to its index. NaN where the
    club-season has no squad data."""
    out = pd.DataFrame(index=matches.index, columns=["injured_share", "new_signings_share"], dtype=float)
    tracked = squads[squads["injury_tracked"] & squads["market_value"].notna()]
    spans = _injury_spans(injuries)
    spans_by_player = {pid: g[["start", "end"]].to_numpy() for pid, g in spans.groupby("player_id")}

    m = matches.assign(season=matches["season"].astype(str), date=pd.to_datetime(matches["date"]))
    for (team, season), games in m.groupby(["team", "season"]):
        squad = tracked[(tracked["team"] == team) & (tracked["season"] == season)]
        if squad.empty:
            continue
        dates = games["date"].to_numpy()
        arrive = np.datetime64(f"{2000 + int(season[2:])}-{WINTER_ARRIVALS_AVAILABLE}")
        value = np.zeros(len(dates))
        injured = np.zeros(len(dates))
        signings = np.zeros(len(dates))
        for p in squad.itertuples():
            member = dates >= arrive if p.winter_arrival else np.ones(len(dates), dtype=bool)
            value += p.market_value * member
            if p.winter_arrival:
                signings += p.market_value * member
            hurt = np.zeros(len(dates), dtype=bool)
            for start, end in spans_by_player.get(p.player_id, ()):
                hurt |= (dates >= start) & (dates <= end)
            injured += p.market_value * (hurt & member)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.loc[games.index, "injured_share"] = injured / value
            out.loc[games.index, "new_signings_share"] = signings / value
    return out
