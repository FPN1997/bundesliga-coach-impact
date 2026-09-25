"""
The sack-o-meter: for every current Bundesliga club, refreshed weekly --

1. Sack risk. How likely a coaching change is within the next
   RISK_HORIZON matches, from a logistic regression fit on every team-match
   since 2014: recent form against what the betting market expected, the
   last two results, xG difference, the coach's tenure and how far into the
   season it is. Validated leave-one-season-out (each season predicted by a
   model that never saw it), and reported next to the base rate so a "12%"
   can be read against "4% for a typical club".
2. Recovery anyway. Expected points per game over the next
   RECOVERY_HORIZON matches if the coach stays: the same comparison-window
   regression as coach_change_effect.py (control windows only; form, xG
   and the change in fixture difficulty), fit with a before-window as long
   as the current season allows -- 4 to 8 matches.
3. What a change adds. The study's mid-season estimate, shown only for
   clubs whose form is within the range where sackings actually happened;
   applying it to a club top of the table would be extrapolation.

Everything is computed from data already in the pipeline; `bundesliga
sack-o-meter` writes outputs/sack_o_meter.json for the results page.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

import config
from src import coach_change_effect as cce
from src.fetch_odds import load_odds
from src.fixture_difficulty import fixture_ease, team_match_expected_points

log = logging.getLogger(__name__)

RISK_HORIZON = 4          # matches ahead the sack risk covers
RECOVERY_HORIZON = 8      # matches ahead the recovery is measured over (as in the study)
FORM_WINDOW = cce.WINDOW  # form = the last 8 matches this season, or all of them if fewer
MIN_SEASON_MATCHES = 4    # before this, form is too thin to say anything
RISK_FEATURES = ["ppg_vs_market", "market_xpts", "ppg_last2", "xgd_form", "log_tenure_days", "season_matches"]
BASELINE_FEATURES = ["ppg_form"]  # "just look at the points" -- what the full model has to beat


def coach_spells(coaches: pd.DataFrame) -> pd.DataFrame:
    """Coach history with back-to-back rows of the same coach at the same
    club merged (a caretaker made permanent is one spell), sorted by start."""
    c = coaches.assign(start_date=pd.to_datetime(coaches["start_date"])).sort_values(["team", "start_date"])
    new = c["coach"].ne(c.groupby("team")["coach"].shift())
    c["spell"] = new.cumsum()
    return (c.groupby(["team", "spell"], as_index=False)
             .agg(coach=("coach", "first"), start_date=("start_date", "first"))
             .drop(columns="spell").sort_values("start_date").reset_index(drop=True))


def _tenure_days(m: pd.DataFrame, spells: pd.DataFrame) -> pd.Series:
    """Days since the match's coach took charge, from the coach history --
    unlike a count of Bundesliga matches, right for promoted clubs too."""
    left = m[["team", "date", "coach"]].reset_index().sort_values("date")
    joined = pd.merge_asof(left, spells.rename(columns={"coach": "spell_coach"}), left_on="date",
                           right_on="start_date", by="team", direction="backward")
    days = (joined["date"] - joined["start_date"]).dt.days.where(joined["spell_coach"] == joined["coach"])
    return days.set_axis(joined["index"]).reindex(m.index)


def team_match_states(matches: pd.DataFrame, odds: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """One row per team-match: the club's state right after that match, and
    whether its coach was gone within the next RISK_HORIZON matches of the
    same season (NaN when fewer matches are left, so it can't be known)."""
    xp = team_match_expected_points(odds).assign(venue=lambda d: np.where(d["home"] == 1.0, "Home", "Away"))
    xp = xp.groupby(["season", "team", "opponent", "venue"], as_index=False)["xpts"].mean()
    m = (matches.dropna(subset=["points"])
         .assign(season=lambda d: d["season"].astype(str))
         .merge(xp, on=["season", "team", "opponent", "venue"], how="left")
         .sort_values(["team", "date"]).reset_index(drop=True))
    m["xgd"] = m["xg"] - m["xga"]

    by_season = m.groupby(["team", "season"], sort=False)
    roll = lambda col, n: by_season[col].transform(lambda s: s.rolling(n, min_periods=1).mean())  # noqa: E731
    m["season_matches"] = by_season.cumcount() + 1
    m["ppg_form"] = roll("points", FORM_WINDOW)
    m["market_xpts"] = roll("xpts", FORM_WINDOW)
    m["ppg_vs_market"] = m["ppg_form"] - m["market_xpts"]
    m["xgd_form"] = roll("xgd", FORM_WINDOW)
    m["ppg_last2"] = roll("points", 2)

    # tenure: days in the job; if the history can't place the coach, fall back
    # to a week per consecutive match under him in this data
    by_team = m.groupby("team", sort=False)
    new_spell = m["coach"].ne(by_team["coach"].shift())
    matches_in_charge = m.groupby([m["team"], new_spell.cumsum()]).cumcount() + 1
    m["tenure_days"] = _tenure_days(m, spells).fillna(7 * matches_in_charge).clip(lower=0)
    m["log_tenure_days"] = np.log1p(m["tenure_days"])

    # target: a different coach in any of the next RISK_HORIZON matches of this season
    changed = pd.Series(False, index=m.index)
    for k in range(1, RISK_HORIZON + 1):
        nxt = by_season["coach"].shift(-k)
        changed |= nxt.notna() & m["coach"].notna() & nxt.ne(m["coach"])
    known = by_season["coach"].shift(-RISK_HORIZON).notna() | changed
    m["sacked_soon"] = np.where(known, changed.astype(float), np.nan)
    return m


def _training_rows(states: pd.DataFrame) -> pd.DataFrame:
    return states[(states["season_matches"] >= MIN_SEASON_MATCHES)
                  & states["coach"].notna()].dropna(subset=[*RISK_FEATURES, "sacked_soon"])


def _risk_model():
    # Splines let each factor bend (a bad run matters more at 0.5 PPG than at
    # 1.5). A plain logistic regression ranked clubs almost as well but was
    # overconfident at the top: forecasts averaging 42% were followed by a
    # change 22% of the time. Splines plus shrinkage (C=0.1) fixed that and
    # scored best on AUC, Brier and log loss, leave-one-season-out.
    return make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=2),
                         LogisticRegression(C=0.1, max_iter=2000))


def evaluate_risk_model(states: pd.DataFrame) -> dict:
    """Leave-one-season-out: every season is predicted by a model fit on the
    others, so the scores are what the meter would have achieved live."""
    rows = _training_rows(states)
    y = rows["sacked_soon"].to_numpy(dtype=int)
    oof = {name: np.full(len(rows), np.nan) for name in ("model", "baseline", "base_rate")}
    for season in rows["season"].unique():
        test = (rows["season"] == season).to_numpy()
        train = ~test
        if y[train].sum() == 0:
            continue
        for name, feats in (("model", RISK_FEATURES), ("baseline", BASELINE_FEATURES)):
            fit = _risk_model().fit(rows.loc[train, feats], y[train])
            oof[name][test] = fit.predict_proba(rows.loc[test, feats])[:, 1]
        oof["base_rate"][test] = y[train].mean()

    ok = ~np.isnan(oof["model"])
    scores = {
        name: {"auc": float(roc_auc_score(y[ok], p[ok])) if name != "base_rate" else 0.5,
               "brier": float(brier_score_loss(y[ok], p[ok])),
               "log_loss": float(log_loss(y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6)))}
        for name, p in oof.items()
    }
    # calibration: forecasts grouped into bands, vs. how often a change followed
    p, yy = oof["model"][ok], y[ok]
    bands = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 1.0]
    which = np.digitize(p, bands[1:-1])
    calibration = [{"lo": bands[b], "hi": bands[b + 1], "forecast": float(p[which == b].mean()),
                    "observed": float(yy[which == b].mean()), "n": int((which == b).sum())}
                   for b in range(len(bands) - 1) if (which == b).sum() >= 20]
    changes = rows[ok & (y == 1)].groupby(["team", "season"]).ngroups
    return {"n_rows": int(ok.sum()), "n_positive_rows": int(yy.sum()), "n_team_seasons_with_change": changes,
            "base_rate": float(yy.mean()),
            "scores": scores, "calibration": calibration, "features": RISK_FEATURES,
            "validation": "leave-one-season-out"}


def recovery_fit(matches: pd.DataFrame, before: int) -> dict:
    """Control-window regression (as in coach_change_effect.py) of PPG over
    the next RECOVERY_HORIZON matches on form over the last `before`: what a
    club in this position that keeps its coach takes, on average."""
    w = cce.build_windows(matches, window=before, after=RECOVERY_HORIZON)
    ctrl = w[w["in_season"] & (w["kind"] == "control")].dropna(
        subset=["ppg_before", "xgd_before", "fixture_change", "ppg_after"])
    X = np.column_stack([np.ones(len(ctrl)), ctrl[["ppg_before", "xgd_before", "fixture_change"]].to_numpy()])
    beta = np.linalg.lstsq(X, ctrl["ppg_after"].to_numpy(), rcond=None)[0]
    resid = ctrl["ppg_after"].to_numpy() - X @ beta
    return {"before": before, "intercept": float(beta[0]), "ppg_before": float(beta[1]),
            "xgd_before": float(beta[2]), "fixture_change": float(beta[3]),
            "residual_sd": float(resid.std(ddof=4)), "n_windows": len(ctrl)}


def _upcoming(fbref: pd.DataFrame, last_played: pd.Timestamp) -> pd.DataFrame:
    up = fbref[fbref["GF"].isna() & (pd.to_datetime(fbref["date"]) > last_played)].copy()
    up["date"] = pd.to_datetime(up["date"])
    up["season"] = up["season"].astype(str)
    return up.sort_values(["team", "date"])


def current_meter(matches: pd.DataFrame, states: pd.DataFrame, odds: pd.DataFrame, fbref: pd.DataFrame,
                  spells: pd.DataFrame, risk_model, effect: dict, sacked_form_max: float,
                  today: pd.Timestamp | None = None) -> list[dict]:
    season = states.loc[states["date"].idxmax(), "season"]
    latest = states[states["season"] == season].groupby("team").tail(1)
    upcoming = _upcoming(fbref, matches["date"].max())
    upcoming["fixture_ease"] = fixture_ease(upcoming, odds)
    played = states[states["season"] == season]
    fits: dict[int, dict] = {}
    clubs = []
    today = pd.Timestamp.today().normalize() if today is None else today
    for r in latest.itertuples():
        spell = spells[(spells["team"] == r.team) & (spells["start_date"] <= today)].tail(1)
        coach_now = spell["coach"].iloc[0] if len(spell) else r.coach
        row = {"team": r.team, "coach": coach_now, "tenure_days": int(r.tenure_days),
               "season_matches": int(r.season_matches), "ppg_form": float(r.ppg_form),
               "market_xpts": None if np.isnan(r.market_xpts) else float(r.market_xpts),
               "xgd_form": None if np.isnan(r.xgd_form) else float(r.xgd_form),
               "ppg_last2": float(r.ppg_last2)}
        if coach_now != r.coach:
            # the change the meter is about has already happened, after the last match in the data
            row.update({"changed_since_last_match": True, "previous_coach": r.coach, "tenure_days": None,
                        "coach_since": spell["start_date"].iloc[0].strftime("%Y-%m-%d")})
        elif len(spell):
            row["coach_since"] = spell["start_date"].iloc[0].strftime("%Y-%m-%d")
        if r.season_matches < MIN_SEASON_MATCHES or any(pd.isna(getattr(r, f)) for f in RISK_FEATURES):
            row["skipped"] = f"needs {MIN_SEASON_MATCHES} matches with xG and odds this season"
            clubs.append(row)
            continue
        if not row.get("changed_since_last_match"):
            state = pd.DataFrame([{f: getattr(r, f) for f in RISK_FEATURES}])
            row["sack_risk"] = float(risk_model.predict_proba(state)[0, 1])

        before = int(min(FORM_WINDOW, r.season_matches))
        fit = fits.setdefault(before, recovery_fit(matches, before))
        nxt = upcoming[upcoming["team"] == r.team].head(RECOVERY_HORIZON)
        past = played[played["team"] == r.team].tail(before)
        ease_before = past["fixture_ease"]
        if len(nxt) == RECOVERY_HORIZON and nxt["fixture_ease"].notna().all() and ease_before.notna().all():
            fixture_change = float(nxt["fixture_ease"].mean() - ease_before.mean())
            ppg_before = float(past["points"].mean())
            xgd_before = float(past["xgd"].mean())
            kept = (fit["intercept"] + fit["ppg_before"] * ppg_before + fit["xgd_before"] * xgd_before
                    + fit["fixture_change"] * fixture_change)
            row.update({"form_matches": before, "ppg_before": ppg_before, "fixture_change": fixture_change,
                        "expected_if_kept": float(kept), "expected_if_kept_sd": fit["residual_sd"],
                        "next_opponents": [f"{o} ({'H' if v == 'Home' else 'A'})"
                                           for o, v in zip(nxt["opponent"], nxt["venue"], strict=True)]})
            # a coach who took over within the form window IS the change: the
            # "with a new coach" line then describes what's expected already
            row["changed_in_form_window"] = bool((past["coach"] != coach_now).any())
            if ppg_before <= sacked_form_max:
                row["expected_with_change"] = float(kept + effect["effect"])
                row["expected_with_change_ci95"] = [float(kept + e) for e in effect["effect_ci95"]]
        clubs.append(row)
    just_changed = lambda c: bool(c.get("changed_since_last_match"))  # noqa: E731
    return sorted(clubs, key=lambda c: (not just_changed(c), -c.get("sack_risk", -1)))


def run() -> dict:
    matches = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")
    odds = load_odds()
    if odds is None:
        log.warning("No odds file -- the sack-o-meter needs market expectations (run `bundesliga pipeline`).")
        return {}
    matches["fixture_ease"] = fixture_ease(matches, odds)
    spells = coach_spells(pd.read_csv(config.COACH_HISTORY_RESOLVED_CSV))
    states = team_match_states(matches, odds, spells)
    evaluation = evaluate_risk_model(states)
    rows = _training_rows(states)
    model = _risk_model().fit(rows[RISK_FEATURES], rows["sacked_soon"].astype(int))

    effect_json = json.loads((Path(config.OUTPUT_DIR) / "coach_change_effect.json").read_text())
    effect = effect_json["mid_season"]["ppg"]
    windows = pd.read_parquet(Path(config.OUTPUT_DIR) / "coach_change_effect_windows.parquet")
    sacked = windows[(windows["kind"] == "treated") & windows["in_season"]]
    sacked_form_max = float(sacked["ppg_before"].quantile(0.95))

    fbref = pd.read_parquet(Path(config.RAW_DIR) / "fbref_schedule.parquet")
    clubs = current_meter(matches, states, odds, fbref, spells, model, effect, sacked_form_max)
    coaches_file = Path(config.COACH_HISTORY_RESOLVED_CSV)
    result = {
        "data_through": matches["date"].max().strftime("%Y-%m-%d"),
        "coach_list_updated": (datetime.fromtimestamp(os.path.getmtime(coaches_file), timezone.utc)
                               .strftime("%Y-%m-%d") if coaches_file.exists() else None),
        "risk_horizon": RISK_HORIZON, "recovery_horizon": RECOVERY_HORIZON,
        "sacked_form_max": sacked_form_max,
        "effect": {k: effect[k] for k in ("effect", "effect_ci95", "n_treated")},
        "risk_model": evaluation,
        "clubs": clubs,
    }
    out = Path(config.OUTPUT_DIR) / "sack_o_meter.json"
    out.write_text(json.dumps(result, indent=2))
    s = evaluation["scores"]
    log.info("Sack risk, leave-one-season-out: AUC %.3f (points only %.3f), Brier %.4f vs base rate %.4f",
             s["model"]["auc"], s["baseline"]["auc"], s["model"]["brier"], s["base_rate"]["brier"])
    for c in clubs[:6]:
        log.info("  %-16s %-22s %s", c["team"], c["coach"],
                 "changed since the last match" if c.get("changed_since_last_match")
                 else f"risk {c['sack_risk']:.1%}" if "sack_risk" in c else "-")
    log.info("Saved -> %s", out)
    return result
