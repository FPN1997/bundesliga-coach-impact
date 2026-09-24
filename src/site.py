"""
Build the published results page: site/index.html, a single self-contained
file (data embedded, charts drawn in the browser, no external requests)
served by GitHub Pages via .github/workflows/pages.yml.

Everything on the page comes from files the pipeline already writes to
outputs/ -- this module only collects and reshapes them, plus one live
piece: forecasts for the next matchday's fixtures, from the same code path
as `bundesliga predict` (predict.current_team_state + predict.forecast).

Rebuild with `bundesliga site` after `bundesliga pipeline` /
`bundesliga benchmark`, then commit site/ to publish.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from src.formation_matrix import MIN_MATCHUP_SAMPLES, common_formations
from src.formation_utils import clean_formation
from src.market_benchmark import MARKET, OUTCOMES, PROBABILITY_MODEL

log = logging.getLogger(__name__)

SITE_DIR = Path("site")
TEMPLATE = Path(__file__).with_name("site_template.html")
REPO_URL = "https://github.com/FPN1997/bundesliga-coach-impact"


def _out(name: str) -> Path:
    return Path(config.OUTPUT_DIR) / name


def _coach_effect() -> dict:
    results = json.loads(_out("coach_change_effect.json").read_text())
    w = pd.read_parquet(_out("coach_change_effect_windows.parquet"))
    mid = w[w["in_season"]]
    ctrl, tr = mid[mid["kind"] == "control"], mid[mid["kind"] == "treated"].copy()

    edges = np.arange(0.0, 3.01, 0.25)
    ctrl = ctrl.assign(delta=ctrl["ppg_after"] - ctrl["ppg_before"],
                       bin=pd.cut(ctrl["ppg_before"], edges, include_lowest=True, labels=False))
    curve = [
        {"lo": float(edges[b]), "hi": float(edges[b + 1]), "x": float(g["ppg_before"].mean()),
         "delta": float(g["delta"].mean()), "n": len(g)}
        for b, g in ctrl.groupby("bin") if len(g) >= 10
    ]

    tr["delta"] = tr["ppg_after"] - tr["ppg_before"]
    tr["excess"] = tr["delta"] - tr["ppg_expected_change"]
    sackings = [
        {"team": r.team, "date": pd.Timestamp(r.date).strftime("%Y-%m-%d"),
         "coach_out": r.coach_out, "coach_in": r.coach_in,
         "ppg_before": round(r.ppg_before, 3), "ppg_after": round(r.ppg_after, 3),
         "delta": round(r.delta, 3), "expected": round(r.ppg_expected_change, 3),
         "excess": round(r.excess, 3)}
        for r in tr.sort_values("date").itertuples()
    ]
    return {"results": results, "control_curve": curve, "sackings": sackings,
            "window": results["window_matches"]}


def _benchmark() -> dict:
    results = json.loads(_out("market_benchmark.json").read_text())
    preds = pd.read_parquet(_out("market_benchmark_predictions.parquet"))
    outcome = preds["FTR"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    onehot = np.eye(3)[outcome].ravel()
    edges = np.linspace(0, 1, 11)
    calibration = {}
    for name in results["forecasters"]:
        if name.startswith("Base rates") or name == "Market, pre-match odds":
            continue
        p = preds[[f"{name} | P({o})" for o in OUTCOMES]].to_numpy().ravel()
        which = np.digitize(p, edges[1:-1])
        calibration[name] = [
            {"p": float(p[which == b].mean()), "freq": float(onehot[which == b].mean()),
             "n": int((which == b).sum())}
            for b in range(10) if (which == b).sum() >= 15
        ]
    return {**results, "calibration": calibration, "market_name": MARKET}


def _formations() -> dict:
    m = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")
    m["formation"] = m["formation"].apply(clean_formation)
    m["opp_formation"] = m["opp_formation"].apply(clean_formation)
    m = m.dropna(subset=["formation", "opp_formation", "points"])
    long = pd.read_csv(_out("formation_matchup_long.csv"))
    pivot = long.pivot(index="formation", columns="opp_formation", values="ppg")
    common = common_formations(m["formation"], pivot)
    long = long[long["formation"].isin(common) & long["opp_formation"].isin(common)]
    return {
        "order": common,
        "league_ppg": float(m["points"].mean()),
        "min_matches": MIN_MATCHUP_SAMPLES,
        "cells": [{"f": r.formation, "o": r.opp_formation, "ppg": round(r.ppg, 3), "n": int(r.matches)}
                  for r in long.itertuples()],
    }


def _next_matchday() -> dict:
    """Forecasts for the next round of fixtures, from the pre-match default model."""
    import predict

    fbref = pd.read_parquet(Path(config.RAW_DIR) / "fbref_schedule.parquet")
    matches, coaches = predict._load_matches(), predict._load_coach_history()
    upcoming = fbref[fbref["GF"].isna() & (fbref["venue"] == "Home")
                     & (pd.to_datetime(fbref["date"]) > matches["date"].max())].copy()
    if upcoming.empty:
        return {"round": None, "fixtures": []}
    upcoming["md"] = upcoming["round"].str.extract(r"(\d+)", expand=False).astype(int)
    nxt = upcoming[upcoming["md"] == upcoming["md"].min()].sort_values(["date", "team"])

    pipe = joblib.load(Path(config.MODEL_DIR) / f"{PROBABILITY_MODEL}.joblib")
    le = joblib.load(Path(config.MODEL_DIR) / "label_encoder_prematch.joblib")
    fixtures = []
    for r in nxt.itertuples():
        row = {"date": pd.Timestamp(r.date).strftime("%Y-%m-%d"), "home": r.team, "away": r.opponent}
        try:
            home = predict.current_team_state(r.team, matches, coaches)
            away = predict.current_team_state(r.opponent, matches, coaches)
        except SystemExit as exc:  # a promoted club without enough Bundesliga matches yet
            row["skipped"] = str(exc)
        else:
            probs, _ = predict.forecast(pipe, le, home, away, "home")
            row.update({"H": probs["W"], "D": probs["D"], "A": probs["L"]})
        fixtures.append(row)
    return {"round": int(nxt["md"].iloc[0]), "fixtures": fixtures}


def collect() -> dict:
    m = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")
    return {
        "meta": {
            "data_through": m["date"].max().strftime("%Y-%m-%d"),
            "matches": int(len(m) // 2),
            "seasons": [config.SEASONS[0], config.SEASONS[-1]],
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "repo": REPO_URL,
        },
        "coach_effect": _coach_effect(),
        "benchmark": _benchmark(),
        "formations": _formations(),
        "next_matchday": _next_matchday(),
    }


def build_site() -> Path:
    data = collect()
    html = TEMPLATE.read_text().replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    SITE_DIR.mkdir(exist_ok=True)
    out = SITE_DIR / "index.html"
    out.write_text(html)
    (SITE_DIR / ".nojekyll").write_text("")
    log.info("Built %s (%.0f KB; data through %s, next matchday: %d fixtures)", out,
             out.stat().st_size / 1024, data["meta"]["data_through"],
             len(data["next_matchday"]["fixtures"]))
    return out
