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

It also draws two images, redrawn each build so they always show the
current numbers: og.png, the card link previews (chat apps, LinkedIn, X,
Reddit) show for the page, and sack-o-meter.png, this week's meter as a
16:9 image to attach to the weekly post.
"""

from __future__ import annotations

import html
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
SITE_URL = "https://fpn1997.github.io/bundesliga-coach-impact/"  # link previews need absolute URLs


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
        {"team": r.team, "league": getattr(r, "league", config.LEAGUE),
         "date": pd.Timestamp(r.date).strftime("%Y-%m-%d"),
         "coach_out": r.coach_out, "coach_in": r.coach_in,
         "ppg_before": round(r.ppg_before, 3), "ppg_after": round(r.ppg_after, 3),
         "delta": round(r.delta, 3), "expected": round(r.ppg_expected_change, 3),
         "excess": round(r.excess, 3)}
        for r in tr.sort_values("date").itertuples()
    ]
    return {"results": results, "control_curve": curve, "sackings": sackings,
            "window": results["window_matches"], "league_labels": config.LEAGUE_LABELS}


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


def _sack_o_meter() -> dict | None:
    """This week's meter (src/sack_o_meter.py), if it has been computed."""
    path = _out("sack_o_meter.json")
    return json.loads(path.read_text()) if path.exists() else None


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
        "sack_o_meter": _sack_o_meter(),
        "benchmark": _benchmark(),
        "formations": _formations(),
        "next_matchday": _next_matchday(),
    }


def _sign(v: float) -> str:
    return ("+" if v >= 0 else "\u2212") + f"{abs(v):.2f}"  # true minus sign, as on the page


def preview_text(data: dict) -> str:
    """The one-sentence summary link previews show, from the current numbers."""
    ppg = data["coach_effect"]["results"]["mid_season"]["ppg"]
    raw, anyway, effect = (_sign(ppg[k]) for k in ("raw_change", "counterfactual_change", "effect"))
    return (f"Bundesliga teams that sack their coach mid-season improve by {raw} points per game, "
            f"but similar teams that kept theirs improve {anyway} anyway. "
            f"What's left for the change itself: {effect}.")


def draw_preview_image(data: dict, out_path: Path) -> None:
    """1200x630 share card: the three headline numbers and the event study."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import viz_style as vs

    ce = data["coach_effect"]
    ppg, es = ce["results"]["mid_season"]["ppg"], ce["results"]["event_study"]
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=vs.SURFACE)
    first = data["meta"]["seasons"][0]  # "2014-2015" -> "2014/15"
    fig.text(0.05, 0.87, f"BUNDESLIGA SINCE {first[:4]}/{first[-2:]}  \u00b7  {ppg['n_treated']} MID-SEASON "
             "SACKINGS", fontsize=14, color=vs.INK_2, fontweight="bold")
    fig.text(0.05, 0.75, "Does sacking the coach work?", fontsize=40, color=vs.INK, fontweight="bold")
    stats = [(_sign(ppg["raw_change"]), "more points per game after a sacking", vs.INK),
             (_sign(ppg["counterfactual_change"]), "expected anyway (teams that kept their coach)",
              vs.SERIES[1]),
             (_sign(ppg["effect"]), "left for the change itself", vs.SERIES[0])]
    for i, (value, label, color) in enumerate(stats):
        y = 0.56 - i * 0.155
        fig.text(0.05, y, value, fontsize=34, color=color, fontweight="bold", va="center")
        fig.text(0.05, y - 0.065, label, fontsize=13.5, color=vs.INK_2, va="center")
    fig.text(0.05, 0.05, SITE_URL.removeprefix("https://").rstrip("/"), fontsize=13, color=vs.MUTED)

    ax = fig.add_axes((0.53, 0.12, 0.44, 0.47))
    vs.style_axes(ax)
    rows = es["matches"]
    k = np.arange(len(rows))
    half = len(rows) / 2 - 0.5
    ax.axvspan(half, len(rows) - 0.5, color=vs.GRID, alpha=0.45, linewidth=0)
    ax.axvline(half, color=vs.INK_2, linewidth=1)
    ax.plot(k, [r["expected"] for r in rows], color=vs.SERIES[1], linewidth=3)
    ax.plot(k, [r["actual"] for r in rows], color=vs.SERIES[0], linewidth=3.5, marker="o", markersize=7,
            markeredgecolor=vs.SURFACE, markeredgewidth=1.5)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(0, 1.8)
    ax.set_xticks([0, half - 0.5, half + 0.5, len(rows) - 1],
                  [str(rows[0]["match"]), "-1", "+1", f"+{rows[-1]['match'] + 1}"])
    ax.tick_params(labelsize=12)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_title("Points per game, match by match", loc="left", fontsize=15, color=vs.INK,
                 fontweight="bold", pad=10)
    ax.text(half + 0.3, 1.72, "new coach", fontsize=12, color=vs.INK_2, va="top")
    ax.text(0, 1.72, "sacked teams", fontsize=12, color=vs.SERIES[0], fontweight="semibold", va="top")
    ax.text(0, 1.54, "expected anyway", fontsize=12, color=vs.SERIES[1], fontweight="semibold", va="top")
    fig.savefig(out_path, dpi=100, facecolor=vs.SURFACE)
    plt.close(fig)


METER_IMAGE = "sack-o-meter.png"
METER_IMAGE_ROWS = 8


def draw_meter_image(meter: dict, out_path: Path) -> None:
    """1200x675 share image of this week's sack-o-meter: the clubs that just
    changed coach, then the highest risks, each against the base rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import viz_style as vs

    clubs = meter["clubs"]
    changed = [c for c in clubs if c.get("changed_since_last_match")]
    ranked = [c for c in clubs if c.get("sack_risk") is not None]
    rows = (changed + ranked)[:METER_IMAGE_ROWS]
    base = meter["risk_model"]["base_rate"]
    matchday = max(c["season_matches"] for c in clubs)
    through = pd.Timestamp(meter["data_through"]).strftime("%-d %b %Y")

    fig = plt.figure(figsize=(12, 6.75), dpi=100, facecolor=vs.SURFACE)
    fig.text(0.05, 0.9, f"BUNDESLIGA SACK-O-METER  \u00b7  AFTER MATCHDAY {matchday}", fontsize=14,
             color=vs.INK_2, fontweight="bold")
    fig.text(0.05, 0.815, f"Chance of a coaching change in the next {meter['risk_horizon']} matches",
             fontsize=25, color=vs.INK, fontweight="bold")

    top, row_h = 0.71, 0.072
    coach_x, bar_x0, bar_x1 = 0.25, 0.47, 0.87  # club names up to ~"Werder Bremen" fit before coach_x
    scale_max = max(0.4, *(c.get("sack_risk") or 0 for c in rows))
    to_x = lambda v: bar_x0 + (bar_x1 - bar_x0) * v / scale_max  # noqa: E731
    for i, c in enumerate(rows):
        y = top - i * row_h
        fig.text(0.05, y, c["team"], fontsize=17, color=vs.INK, fontweight="bold", va="center")
        fig.text(coach_x, y, c["coach"], fontsize=14, color=vs.INK_2, va="center")
        if c.get("changed_since_last_match"):
            since = pd.Timestamp(c["coach_since"]).strftime("%-d %b")
            fig.text(bar_x0, y, f"just changed: new coach since {since}", fontsize=14, color=vs.INK_2,
                     va="center", style="italic")
            continue
        track = bar_x1 - bar_x0
        fill = max(0.003, to_x(c["sack_risk"]) - bar_x0)
        for width, color in ((track, vs.GRID), (fill, vs.DIVERGING[0])):
            fig.patches.append(plt.Rectangle((bar_x0, y - 0.017), width, 0.034, transform=fig.transFigure,
                                             color=color, linewidth=0))
        fig.lines.append(plt.Line2D([to_x(base)] * 2, [y - 0.028, y + 0.028], transform=fig.transFigure,
                                    color=vs.INK_2, linewidth=1.5))
        fig.text(bar_x1 + 0.015, y, f"{c['sack_risk']:.0%}", fontsize=18, color=vs.INK, fontweight="bold",
                 va="center")
    # the base-rate line, labelled once under the last row
    y_last = top - (len(rows) - 1) * row_h
    fig.text(to_x(base), y_last - 0.055, f"| typical club-week: {base:.0%}", fontsize=12, color=vs.INK_2,
             ha="left", va="center")

    fig.text(0.05, 0.06, f"Data through {through}. Model trained on every Bundesliga club-week since 2014, "
             "tested on seasons it never saw.", fontsize=12, color=vs.MUTED)
    fig.text(0.05, 0.025, SITE_URL.removeprefix("https://").rstrip("/"), fontsize=12, color=vs.INK_2)
    fig.savefig(out_path, dpi=100, facecolor=vs.SURFACE)
    plt.close(fig)


def build_site() -> Path:
    data = collect()
    page = (TEMPLATE.read_text()
            .replace("__OG_DESCRIPTION__", html.escape(preview_text(data)))
            .replace("__SITE_URL__", SITE_URL)
            .replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    SITE_DIR.mkdir(exist_ok=True)
    out = SITE_DIR / "index.html"
    out.write_text(page)
    (SITE_DIR / ".nojekyll").write_text("")
    draw_preview_image(data, SITE_DIR / "og.png")
    if data.get("sack_o_meter"):
        draw_meter_image(data["sack_o_meter"], SITE_DIR / METER_IMAGE)
    log.info("Built %s (%.0f KB; data through %s, next matchday: %d fixtures)", out,
             out.stat().st_size / 1024, data["meta"]["data_through"],
             len(data["next_matchday"]["fixtures"]))
    return out
