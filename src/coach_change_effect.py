"""
Does changing coach actually improve results -- or do teams that sack their
coach just bounce back the way any team in a bad run does?

coach_impact.py measures the raw before/after swing around each coaching
change. That number is badly confounded: clubs sack coaches after bad runs,
and bad runs regress toward the mean on their own. This module estimates
how much of the swing is left once that is accounted for, by comparing
each coaching change against "control" windows -- the same kind of
before/after comparison at moments when a team did NOT change coach --
drawn from teams in a similar position.

Method (regression adjustment on control windows + a team-level cluster bootstrap):

  - Event windows: for every team and every match i with WINDOW matches
    of history before it and WINDOW matches after it, compare the WINDOW
    matches before i with the WINDOW matches from i on. A window is
    "treated" if the coach changes exactly at i, and a "control" if the
    same coach is in charge for all 2*WINDOW matches. Windows with a
    change anywhere else inside them are neither, and are skipped.
  - In-season vs. off-season: a window whose before and after halves fall
    in different seasons spans a summer break (new signings, pre-season,
    a different league table). Mid-season sackings are the "new coach
    bounce" question people actually argue about, so they are the
    headline; summer appointments are reported separately, each compared
    only against controls of the same kind.
  - Adjustment: the expected ("counterfactual") change for each treated
    window is predicted from control windows by a linear regression of
    change on (PPG before, xG difference before), fit on controls only.
    PPG says how bad the run was; xG difference says how much of it was
    bad luck rather than bad play -- a team with poor results but a decent
    xG difference was likely to recover anyway, sacking or not. Regression
    to the mean is linear in the before-value, which is why a linear model
    is enough here (docs/coach_change_effect.png shows the binned control
    means sitting on the fitted line). This replaced a first attempt at
    coarsened exact matching on the same two variables, which had to
    discard 11 of 39 mid-season sackings -- the worst runs, where control
    windows are rare -- i.e. exactly the cases the question is about.
    The effect is the average of (actual change - expected change).
  - Uncertainty: control windows from the same team overlap and are
    strongly correlated, so a naive bootstrap over windows would
    understate the uncertainty badly. The bootstrap resamples whole
    TEAMS instead (the independent unit here), refitting the control
    regression each time.

Outputs: outputs/coach_change_effect.json / .png, outputs/coach_change_effect_windows.parquet
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WINDOW = config.IMPACT_WINDOW
N_BOOTSTRAP = 2000


def build_windows(matches: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """One row per treated or control window (see module docstring)."""
    matches = matches.dropna(subset=["points"]).sort_values(["team", "date"])
    rows = []
    for team, g in matches.groupby("team"):
        points = g["points"].to_numpy(dtype=float)
        xgd = (g["xg"] - g["xga"]).to_numpy(dtype=float)
        coach = g["coach"].to_numpy(dtype=object)
        season = g["season"].astype(str).to_numpy()
        dates = g["date"].to_numpy()
        for i in range(window, len(g) - window + 1):
            span = coach[i - window:i + window]
            if pd.isna(span).any():
                continue
            if coach[i] != coach[i - 1]:
                kind = "treated"
                # the change must be the ONLY one in the before half, or the
                # "before" form belongs to more than one coach
                if len(set(span[:window])) != 1:
                    continue
            elif len(set(span)) == 1:
                kind = "control"
            else:
                continue
            rows.append({
                "team": team,
                "date": dates[i],
                "kind": kind,
                "in_season": season[i - window] == season[i + window - 1],
                "ppg_before": points[i - window:i].mean(),
                "ppg_after": points[i:i + window].mean(),
                "xgd_before": np.nanmean(xgd[i - window:i]),
                "xgd_after": np.nanmean(xgd[i:i + window]),
            })
    return pd.DataFrame(rows)


class _AdjustedEstimator:
    """Regression-adjusted effect for one outcome ("ppg" or "xgd"),
    re-estimable under any team-level bootstrap weighting via weighted
    least squares -- a closed-form solve per resample, so 2000 resamples
    take well under a second."""

    COVARIATES = ("ppg_before", "xgd_before")

    def __init__(self, windows: pd.DataFrame, outcome: str):
        w = windows.dropna(subset=[f"{outcome}_before", f"{outcome}_after", *self.COVARIATES])
        self.teams = np.array(sorted(w["team"].unique()))
        team_idx = w["team"].map({t: k for k, t in enumerate(self.teams)}).to_numpy()
        delta = (w[f"{outcome}_after"] - w[f"{outcome}_before"]).to_numpy()
        X = np.column_stack([np.ones(len(w)), w[list(self.COVARIATES)].to_numpy()])
        is_ctrl = (w["kind"] == "control").to_numpy()

        self.X_ctrl, self.y_ctrl, self.ctrl_team = X[is_ctrl], delta[is_ctrl], team_idx[is_ctrl]
        self.X_tr, self.y_tr, self.tr_team = X[~is_ctrl], delta[~is_ctrl], team_idx[~is_ctrl]
        self.tr_before = w.loc[~is_ctrl, f"{outcome}_before"].to_numpy()
        self.n_control = int(is_ctrl.sum())

    def _fit(self, team_weights: np.ndarray) -> np.ndarray:
        rw = team_weights[self.ctrl_team]
        Xw = self.X_ctrl * rw[:, None]
        return np.linalg.lstsq(Xw.T @ self.X_ctrl, Xw.T @ self.y_ctrl, rcond=None)[0]

    def estimate(self, team_weights: np.ndarray | None = None) -> dict:
        wts = np.ones(len(self.teams)) if team_weights is None else team_weights
        tw = wts[self.tr_team]
        if tw.sum() == 0 or wts[self.ctrl_team].sum() == 0:
            return {"n_treated": 0}
        beta = self._fit(wts)
        expected = self.X_tr @ beta
        raw = float(np.average(self.y_tr, weights=tw))
        cf = float(np.average(expected, weights=tw))
        return {
            "n_treated": len(self.y_tr),
            "mean_before": float(np.average(self.tr_before, weights=tw)),
            "raw_change": raw,
            "counterfactual_change": cf,
            "effect": raw - cf,
            "control_fit": {"intercept": float(beta[0]),
                            **{c: float(b) for c, b in zip(self.COVARIATES, beta[1:], strict=True)}},
        }

    def bootstrap(self, n: int = N_BOOTSTRAP, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n):
            picks = rng.integers(0, len(self.teams), len(self.teams))
            est = self.estimate(np.bincount(picks, minlength=len(self.teams)).astype(float))
            if est["n_treated"]:
                draws.append(est["effect"])
        draws = np.array(draws)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return {
            "effect_ci95": [float(lo), float(hi)],
            # one-sided: share of resamples showing no positive effect at all
            "share_of_resamples_effect_le_0": float((draws <= 0).mean()),
        }


def estimate_effects(windows: pd.DataFrame) -> dict:
    results: dict = {"window_matches": WINDOW, "method": "regression adjustment on controls",
                     "bootstrap_resamples": N_BOOTSTRAP, "bootstrap_unit": "team"}
    for label, in_season in [("mid_season", True), ("off_season", False)]:
        subset = windows[windows["in_season"] == in_season]
        results[label] = {}
        if not ((subset["kind"] == "treated").any() and (subset["kind"] == "control").any()):
            results[label] = {"ppg": {"n_treated": 0}, "xgd": {"n_treated": 0}}
            continue
        for outcome in ("ppg", "xgd"):
            est = _AdjustedEstimator(subset, outcome)
            summary = est.estimate()
            if summary["n_treated"]:
                summary.update(est.bootstrap())
                summary["n_control_windows"] = est.n_control
            results[label][outcome] = summary
    return results


# Validated categorical palette (light mode) -- slot 1
# for the sackings, slot 2 for the teams that kept their coach, text tokens
# for everything that is text.
_SACKED, _KEPT = "#2a78d6", "#eb6834"
_INK, _INK_2, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot(windows: pd.DataFrame, results: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    mid = windows[windows["in_season"]]
    ctrl, tr = mid[mid["kind"] == "control"], mid[mid["kind"] == "treated"]
    bins = np.arange(0.0, 3.01, 0.25)
    centers = (bins[:-1] + bins[1:]) / 2
    ctrl_delta = (ctrl["ppg_after"] - ctrl["ppg_before"]).groupby(
        pd.cut(ctrl["ppg_before"], bins, include_lowest=True), observed=False).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), width_ratios=[1.55, 1],
                                   facecolor=_SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(_SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(colors=_INK_2, labelsize=9)
        ax.grid(axis="y", color=_GRID, linewidth=1)
        ax.set_axisbelow(True)

    # Left: every mid-season sacking vs. what teams in the same spot did anyway
    ax1.axhline(0, color=_INK_2, linewidth=1)
    ax1.plot(centers, ctrl_delta.to_numpy(), color=_KEPT, linewidth=2, marker="o", markersize=6,
             markeredgecolor=_SURFACE, markeredgewidth=2, zorder=3,
             label=f"Kept their coach (avg of {len(ctrl):,} windows)")
    # 8-match PPG only takes values in steps of 1/8, so sackings stack on
    # identical coordinates -- a small fixed-seed horizontal jitter keeps
    # every one of them visible (within +/-0.04, a third of a step).
    jitter = np.random.default_rng(0).uniform(-0.04, 0.04, len(tr))
    ax1.scatter(tr["ppg_before"] + jitter, tr["ppg_after"] - tr["ppg_before"], s=55, color=_SACKED,
                edgecolor=_SURFACE, linewidth=2, zorder=4,
                label=f"Sacked their coach mid-season (n={len(tr)})")
    ax1.set_xlabel("Points per game over the 8 matches before", color=_INK_2, fontsize=10)
    ax1.set_ylabel("Change in PPG over the next 8 matches", color=_INK_2, fontsize=10)
    ax1.set_xlim(-0.05, 3.0)
    ax1.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=_INK)
    r = results["mid_season"]["ppg"]
    ax1.set_title(f"Teams that sack their coach improve {r['raw_change']:+.2f} PPG \u2014 "
                  f"but {r['counterfactual_change']:+.2f} of that happens anyway",
                  loc="left", fontsize=11.5, color=_INK, fontweight="semibold")

    # Right: the effect left after adjustment, with team-bootstrap 95% CIs
    rows = [("Points per game\nmid-season", results["mid_season"]["ppg"]),
            ("xG difference per game\nmid-season", results["mid_season"]["xgd"]),
            ("Points per game\nsummer appointments", results["off_season"]["ppg"]),
            ("xG difference per game\nsummer appointments", results["off_season"]["xgd"])]
    ys = np.arange(len(rows))[::-1]
    ax2.axvline(0, color=_INK_2, linewidth=1)
    for y, (_, res) in zip(ys, rows, strict=True):
        lo, hi = res["effect_ci95"]
        ax2.plot([lo, hi], [y, y], color=_INK_2, linewidth=2, solid_capstyle="round")
        ax2.scatter([res["effect"]], [y], s=60, color=_INK, edgecolor=_SURFACE, linewidth=2, zorder=3)
        ax2.annotate(f"{res['effect']:+.2f}", (res["effect"], y), xytext=(0, 9),
                     textcoords="offset points", ha="center", fontsize=9, color=_INK)
    ax2.set_yticks(ys, [label for label, _ in rows], fontsize=9, color=_INK)
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", color=_GRID, linewidth=1)
    ax2.set_xlabel("Effect of the change beyond what was expected\n(95% CI, teams resampled)",
                   color=_INK_2, fontsize=10)
    ax2.set_title("What's left after adjusting", loc="left", fontsize=11.5,
                  color=_INK, fontweight="semibold")
    ax2.set_ylim(-0.6, len(rows) - 0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    log.info("Saved plot -> %s", out_path)


def run() -> dict:
    matches = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")
    windows = build_windows(matches)
    if windows.empty or not (windows["kind"] == "treated").any():
        log.warning("No coaching changes with full %d-match windows either side -- nothing to estimate.",
                    WINDOW)
        return {}
    results = estimate_effects(windows)

    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_dir / "coach_change_effect_windows.parquet", index=False)
    (out_dir / "coach_change_effect.json").write_text(json.dumps(results, indent=2))
    if all(results[k][o].get("n_treated") for k in ("mid_season", "off_season") for o in ("ppg", "xgd")):
        plot(windows, results, out_dir / "coach_change_effect.png")
    else:
        log.warning("Not enough coaching changes of both kinds to plot (tiny dataset?) -- skipping plot.")

    for label in ("mid_season", "off_season"):
        r = results[label]["ppg"]
        if not r.get("n_treated"):
            continue
        log.info(
            "%s coaching changes (n=%d): PPG %+.2f raw, %+.2f expected anyway -> "
            "effect %+.2f PPG, 95%% CI [%+.2f, %+.2f]",
            label, r["n_treated"], r["raw_change"], r["counterfactual_change"],
            r["effect"], *r["effect_ci95"],
        )
    return results


if __name__ == "__main__":
    run()
