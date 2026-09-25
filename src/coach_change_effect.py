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
  - Fixture difficulty: the 8 matches after a change can simply be easier
    than the 8 before -- more home games, weaker opponents. Each match is
    rated by the points an AVERAGE team would expect from it (opponent's
    season-average betting-market rating and venue; src/fixture_difficulty.py),
    and the change in average rating between the two halves is a third
    covariate in the adjustment. Only the opponent and venue are used, never
    the team's own odds, which already price in the new coach.
  - Squad changes: injuries and new signings (Transfermarkt, via
    src/fetch_squads.py and src/squad_availability.py). For each half, the
    share of squad market value out injured and the share that arrived in
    January; their changes between the halves are two more covariates, used
    only when squad data covers at least SQUAD_MIN_COVERAGE of the windows.
    "The change" then means mostly the coach, including his team selection.
  - Transfer windows: "the change" is everything that happens at that
    moment, including new signings. Each window records whether a
    registration period (config.TRANSFER_WINDOWS) is open during the
    matches after the change; the mid-season effect is also estimated
    separately for changes with and without one, each against controls of
    the same kind. Where no window follows the change, the squad was
    frozen, so signings can't be what drives the effect there.
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
from src import viz_style as vs
from src.fetch_odds import load_odds
from src.fixture_difficulty import fixture_ease
from src.squad_availability import match_availability

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WINDOW = config.IMPACT_WINDOW
N_BOOTSTRAP = 2000
# Per-match measures whose change between the two halves is a covariate.
PER_MATCH_CHANGES = {"fixture_ease": "fixture_change", "injured_share": "injury_change",
                     "new_signings_share": "signing_change"}
SQUAD_COVARIATES = ("injury_change", "signing_change")
SQUAD_MIN_COVERAGE = 0.95
_TRANSFER_WINDOWS = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in config.TRANSFER_WINDOWS]


def transfer_window_open(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True if any registration period overlaps [start, end]."""
    return any(a <= end and start <= b for a, b in _TRANSFER_WINDOWS)


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
        per_match = {col: g[col].to_numpy(dtype=float) for col in PER_MATCH_CHANGES if col in g}
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
            changes = {}
            for col, values in per_match.items():
                before, after = values[i - window:i], values[i:i + window]
                # a missing value (no odds / no squad data for a match) leaves the change undefined
                changes[PER_MATCH_CHANGES[col]] = (np.nan if np.isnan(before).any() or np.isnan(after).any()
                                                   else after.mean() - before.mean())
            rows.append({
                **changes,
                "team": team,
                "date": dates[i],
                "kind": kind,
                "coach_out": coach[i - 1] if kind == "treated" else None,
                "coach_in": coach[i] if kind == "treated" else None,
                "in_season": season[i - window] == season[i + window - 1],
                # could new signings arrive during the matches after the change?
                "window_after": transfer_window_open(pd.Timestamp(dates[i]),
                                                     pd.Timestamp(dates[i + window - 1])),
                "ppg_before": points[i - window:i].mean(),
                "ppg_after": points[i:i + window].mean(),
                # sackings usually follow a collapse in the last couple of results;
                # a robustness check adjusts for that too (see estimate_effects)
                "ppg_last2": points[i - 2:i].mean(),
                # every match's points, for the match-by-match event study
                **{f"pts_{k:+d}": points[i + k] for k in range(-window, window)},
                "xgd_before": np.nanmean(xgd[i - window:i]),
                "xgd_after": np.nanmean(xgd[i:i + window]),
            })
    return pd.DataFrame(rows)


class _AdjustedEstimator:
    """Regression-adjusted effect for one outcome ("ppg" or "xgd"),
    re-estimable under any team-level bootstrap weighting via weighted
    least squares -- a closed-form solve per resample, so 2000 resamples
    take well under a second."""

    BASE_COVARIATES = ("ppg_before", "xgd_before")

    def __init__(self, windows: pd.DataFrame, outcome: str, covariates: tuple[str, ...] = BASE_COVARIATES):
        self.COVARIATES = covariates
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
    base = _AdjustedEstimator.BASE_COVARIATES
    extra = ["fixture_change"] if "fixture_change" in windows else []
    squad = [c for c in SQUAD_COVARIATES if c in windows]
    if squad:
        coverage = float(windows[squad].notna().all(axis=1).mean())
        if coverage >= SQUAD_MIN_COVERAGE:
            extra += squad
        else:
            log.warning("Squad data covers only %.0f%% of windows (need %.0f%%) -- estimating "
                        "without the injury/new-signing adjustment. Run `bundesliga squad`.",
                        100 * coverage, 100 * SQUAD_MIN_COVERAGE)
    covariates = (*base, *extra)
    results: dict = {"window_matches": WINDOW, "method": "regression adjustment on controls",
                     "covariates": list(covariates),
                     "bootstrap_resamples": N_BOOTSTRAP, "bootstrap_unit": "team"}
    groups = [
        ("mid_season", windows["in_season"]),
        ("off_season", ~windows["in_season"]),
        # mid-season only, split by whether signings were possible after the change
        ("mid_season_window_after", windows["in_season"] & windows["window_after"]),
        ("mid_season_no_window_after", windows["in_season"] & ~windows["window_after"]),
    ]
    for label, mask in groups:
        subset = windows[mask]
        results[label] = {}
        if not ((subset["kind"] == "treated").any() and (subset["kind"] == "control").any()):
            results[label] = {"ppg": {"n_treated": 0}, "xgd": {"n_treated": 0}}
            continue
        for outcome in ("ppg", "xgd"):
            est = _AdjustedEstimator(subset, outcome, covariates)
            summary = est.estimate()
            if summary["n_treated"]:
                summary.update(est.bootstrap())
                summary["n_control_windows"] = est.n_control
            results[label][outcome] = summary
        # Puts the two outcomes on one scale: how many points per game one
        # unit of xG difference per game is worth over a window (fit on
        # control windows), so the xG effect can be read in points.
        ctrl = subset[subset["kind"] == "control"].dropna(subset=["xgd_after", "ppg_after"])
        if len(ctrl) >= 30 and results[label]["xgd"].get("n_treated"):
            slope = float(np.polyfit(ctrl["xgd_after"], ctrl["ppg_after"], 1)[0])
            xgd = results[label]["xgd"]
            results[label]["ppg_per_xgd"] = slope
            results[label]["ppg_implied_by_xgd_effect"] = {
                "effect": slope * xgd["effect"],
                "ci95": [slope * x for x in xgd["effect_ci95"]],
            }

    if extra:
        mid = windows[windows["in_season"]].dropna(subset=extra)
        # How different were the fixtures / injuries / signings after a
        # sacking, compared with after nothing?
        results["change_means"] = {
            c: {kind: float(mid.loc[mid["kind"] == kind, c].mean()) for kind in ("treated", "control")}
            for c in extra
        }
        # The mid-season estimate with each layer of adjustment, all on the
        # same windows so the comparison is like for like.
        specs = {"base": base}
        if "fixture_change" in extra:
            specs["fixtures"] = (*base, "fixture_change")
        if any(c in extra for c in SQUAD_COVARIATES):
            specs["fixtures_and_squad"] = covariates
        # Robustness: clubs sack right after a couple of bad results (see the
        # event study). Does that late collapse predict an extra bounce beyond
        # the 8-match form? Checked, not assumed: add the last 2 matches' PPG.
        specs["plus_last_2_matches"] = (*covariates, "ppg_last2")
        results["mid_season_specifications"] = {}
        for name, covs in specs.items():
            results["mid_season_specifications"][name] = {"covariates": list(covs)}
            for outcome in ("ppg", "xgd"):
                est = _AdjustedEstimator(mid, outcome, covs)
                summary = est.estimate()
                summary.update(est.bootstrap())
                results["mid_season_specifications"][name][outcome] = summary
    return results




def event_study(windows: pd.DataFrame, covariates: tuple[str, ...], window: int = WINDOW) -> dict:
    """Points per game at each match from `window` before to `window` after a
    mid-season change: what sacked teams actually took, and what similar
    teams that kept their coach took at the same position -- the same
    control regression as the headline estimate, fit on each match's points
    instead of the 8-match average. Across the matches before the change the
    two lines average the same by construction (the before-form is a
    covariate, so that's no test of the comparison) -- only their shape
    there is informative. After the change the gap averages to exactly the
    headline effect, since least squares is linear in the outcome."""
    mid = windows[windows["in_season"]].dropna(subset=list(covariates))
    ctrl, tr = mid[mid["kind"] == "control"], mid[mid["kind"] == "treated"]
    X_ctrl = np.column_stack([np.ones(len(ctrl)), ctrl[list(covariates)].to_numpy()])
    X_tr = np.column_stack([np.ones(len(tr)), tr[list(covariates)].to_numpy()])
    rows = []
    for k in range(-window, window):
        col = f"pts_{k:+d}"
        beta = np.linalg.lstsq(X_ctrl, ctrl[col].to_numpy(dtype=float), rcond=None)[0]
        actual = tr[col].to_numpy(dtype=float)
        rows.append({
            "match": k,
            "actual": float(actual.mean()),
            "actual_ci95": [float(actual.mean() - 1.96 * actual.std(ddof=1) / np.sqrt(len(actual))),
                            float(actual.mean() + 1.96 * actual.std(ddof=1) / np.sqrt(len(actual)))],
            "expected": float((X_tr @ beta).mean()),
        })
    return {"n_changes": len(tr), "covariates": list(covariates), "matches": rows}


def plot_event_study(es: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    k = [r["match"] for r in es["matches"]]
    actual = [r["actual"] for r in es["matches"]]
    expected = [r["expected"] for r in es["matches"]]
    lo = [r["actual_ci95"][0] for r in es["matches"]]
    hi = [r["actual_ci95"][1] for r in es["matches"]]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=vs.SURFACE)
    vs.style_axes(ax)
    ax.axvspan(-0.5, max(k) + 0.5, color=vs.GRID, alpha=0.35, linewidth=0)
    ax.fill_between(k, lo, hi, color=vs.SERIES[0], alpha=0.12, linewidth=0)
    ax.plot(k, expected, color=vs.SERIES[1], linewidth=2, marker="o", markersize=5,
            markeredgecolor=vs.SURFACE, markeredgewidth=1.5,
            label="Expected anyway (similar teams that kept their coach)")
    ax.plot(k, actual, color=vs.SERIES[0], linewidth=2, marker="o", markersize=6,
            markeredgecolor=vs.SURFACE, markeredgewidth=1.5,
            label=f"Teams that sacked their coach (n={es['n_changes']}, 95% band)")
    ax.axvline(-0.5, color=vs.INK_2, linewidth=1)
    ax.annotate("new coach", (-0.4, ax.get_ylim()[1]), xytext=(4, -14), textcoords="offset points",
                fontsize=9, color=vs.INK_2)
    ax.set_xticks(k, [f"{x}" if x < 0 else f"+{x + 1}" for x in k])
    ax.set_xlabel("Matches before and after the coaching change", color=vs.INK_2)
    ax.set_ylabel("Points per game", color=vs.INK_2)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=vs.INK)
    vs.title(ax, "Around a mid-season sacking, match by match")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=vs.SURFACE)
    plt.close(fig)
    log.info("Saved plot -> %s", out_path)


def plot(windows: pd.DataFrame, results: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    mid = windows[windows["in_season"]]
    ctrl, tr = mid[mid["kind"] == "control"], mid[mid["kind"] == "treated"]
    bins = np.arange(0.0, 3.01, 0.25)
    centers = (bins[:-1] + bins[1:]) / 2
    ctrl_delta = (ctrl["ppg_after"] - ctrl["ppg_before"]).groupby(
        pd.cut(ctrl["ppg_before"], bins, include_lowest=True), observed=False).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), width_ratios=[1.55, 1],
                                   facecolor=vs.SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(vs.SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(vs.GRID)
        ax.tick_params(colors=vs.INK_2, labelsize=9)
        ax.grid(axis="y", color=vs.GRID, linewidth=1)
        ax.set_axisbelow(True)

    # Left: every mid-season sacking vs. what teams in the same spot did anyway
    ax1.axhline(0, color=vs.INK_2, linewidth=1)
    ax1.plot(centers, ctrl_delta.to_numpy(), color=vs.SERIES[1], linewidth=2, marker="o", markersize=6,
             markeredgecolor=vs.SURFACE, markeredgewidth=2, zorder=3,
             label=f"Kept their coach (avg of {len(ctrl):,} windows)")
    # 8-match PPG only takes values in steps of 1/8, so sackings stack on
    # identical coordinates -- a small fixed-seed horizontal jitter keeps
    # every one of them visible (within +/-0.04, a third of a step).
    jitter = np.random.default_rng(0).uniform(-0.04, 0.04, len(tr))
    ax1.scatter(tr["ppg_before"] + jitter, tr["ppg_after"] - tr["ppg_before"], s=55, color=vs.SERIES[0],
                edgecolor=vs.SURFACE, linewidth=2, zorder=4,
                label=f"Sacked their coach mid-season (n={len(tr)})")
    ax1.set_xlabel("Points per game over the 8 matches before", color=vs.INK_2, fontsize=10)
    ax1.set_ylabel("Change in PPG over the next 8 matches", color=vs.INK_2, fontsize=10)
    ax1.set_xlim(-0.05, 3.0)
    ax1.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=vs.INK)
    r = results["mid_season"]["ppg"]
    ax1.set_title(f"Teams that sack their coach improve {r['raw_change']:+.2f} PPG \u2014 "
                  f"but {r['counterfactual_change']:+.2f} of that happens anyway",
                  loc="left", fontsize=11.5, color=vs.INK, fontweight="semibold")

    # Right: the effect left after adjustment, with team-bootstrap 95% CIs
    rows = [("Points per game\nmid-season", results["mid_season"]["ppg"]),
            ("xG difference per game\nmid-season", results["mid_season"]["xgd"]),
            ("Points per game\nsummer appointments", results["off_season"]["ppg"]),
            ("xG difference per game\nsummer appointments", results["off_season"]["xgd"])]
    ys = np.arange(len(rows))[::-1]
    ax2.axvline(0, color=vs.INK_2, linewidth=1)
    for y, (_, res) in zip(ys, rows, strict=True):
        lo, hi = res["effect_ci95"]
        ax2.plot([lo, hi], [y, y], color=vs.INK_2, linewidth=2, solid_capstyle="round")
        ax2.scatter([res["effect"]], [y], s=60, color=vs.INK, edgecolor=vs.SURFACE, linewidth=2, zorder=3)
        ax2.annotate(f"{res['effect']:+.2f}", (res["effect"], y), xytext=(0, 9),
                     textcoords="offset points", ha="center", fontsize=9, color=vs.INK)
    ax2.set_yticks(ys, [label for label, _ in rows], fontsize=9, color=vs.INK)
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", color=vs.GRID, linewidth=1)
    ax2.set_xlabel("Effect of the change beyond what was expected\n(95% CI, teams resampled)",
                   color=vs.INK_2, fontsize=10)
    ax2.set_title("What's left after adjusting", loc="left", fontsize=11.5,
                  color=vs.INK, fontweight="semibold")
    ax2.set_ylim(-0.6, len(rows) - 0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=vs.SURFACE)
    plt.close(fig)
    log.info("Saved plot -> %s", out_path)


def run() -> dict:
    matches = pd.read_parquet(Path(config.PROCESSED_DIR) / "match_dataset.parquet")
    odds = load_odds()
    if odds is not None:
        matches["fixture_ease"] = fixture_ease(matches, odds)
        log.info("Fixture ratings for %.1f%% of matches", 100 * matches["fixture_ease"].notna().mean())
    else:
        log.warning("No odds file -- estimating without the fixture-difficulty adjustment "
                    "(run `bundesliga pipeline` to fetch odds).")
    squads_file = Path(config.RAW_DIR) / "tm_squads.parquet"
    injuries_file = Path(config.RAW_DIR) / "tm_injuries.parquet"
    if squads_file.exists() and injuries_file.exists():
        availability = match_availability(matches, pd.read_parquet(squads_file),
                                          pd.read_parquet(injuries_file))
        matches[["injured_share", "new_signings_share"]] = availability
        log.info("Squad data for %.1f%% of matches", 100 * availability["injured_share"].notna().mean())
    windows = build_windows(matches)
    if windows.empty or not (windows["kind"] == "treated").any():
        log.warning("No coaching changes with full %d-match windows either side -- nothing to estimate.",
                    WINDOW)
        return {}
    results = estimate_effects(windows)
    results["event_study"] = event_study(windows, tuple(results["covariates"]))

    # Per-change expected PPG change, from the control fit for its kind --
    # "which sackings beat what was expected anyway" (the published results
    # page ranks them).
    for label, in_season in [("mid_season", True), ("off_season", False)]:
        fit = results[label]["ppg"].get("control_fit")
        rows = (windows["kind"] == "treated") & (windows["in_season"] == in_season)
        if fit:
            windows.loc[rows, "ppg_expected_change"] = fit["intercept"] + sum(
                coef * windows.loc[rows, name] for name, coef in fit.items() if name != "intercept")

    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_dir / "coach_change_effect_windows.parquet", index=False)
    (out_dir / "coach_change_effect.json").write_text(json.dumps(results, indent=2))
    if all(results[k][o].get("n_treated") for k in ("mid_season", "off_season") for o in ("ppg", "xgd")):
        plot(windows, results, out_dir / "coach_change_effect.png")
        plot_event_study(results["event_study"], out_dir / "coach_change_event_study.png")
    else:
        log.warning("Not enough coaching changes of both kinds to plot (tiny dataset?) -- skipping plot.")

    for label in ("mid_season", "off_season", "mid_season_window_after", "mid_season_no_window_after"):
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
