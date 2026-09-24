"""
How good are the pre-match forecasts, measured against the one benchmark
that matters in football prediction: the betting market?

Beating "always pick the home team" (outcome_predictor.py's baselines) says
little. Bookmaker odds aggregate everything public -- injuries, lineups,
form, money -- and Pinnacle's in particular are the standard reference for
the "true" probability of a football result. This module scores the
project's pre-match forecasts against the market on the held-out test
seasons (config.TEST_SEASONS), using proper scoring rules rather than
accuracy/F1:

  - RPS (ranked probability score) -- the standard for football, because
    home win / draw / away win are ordered: calling a 2-1 home win a draw
    is less wrong than calling it an away win. Lower is better.
  - Log loss and Brier score, for comparability with general ML practice.

Odds come from football-data.co.uk (via soccerdata.MatchHistory), as the
market average across bookmakers: "pre-match" odds collected on the Friday
(Tuesday for midweek rounds) before the match -- roughly the information a
pre-match model has -- and closing odds at kickoff, after lineups are out,
the strictest bar there is. Pinnacle, the sharpest single bookmaker, would
be the ideal reference, but verified in the data: football-data.co.uk's
Pinnacle columns stop in mid-January 2026, leaving them on only ~half of
the test matches. Mixing sources match by match would muddy the benchmark,
so the market average is used throughout (complete in every season) and
Pinnacle closing odds are reported separately as a sensitivity check on
the subset of matches that have them. The bookmaker margin is removed by
normalizing the implied probabilities to sum to 1.

Two forecasters are scored:

  - predict.py's default model (Optuna-tuned Random Forest, pre-match
    variant), as-is. It was trained with class-balanced weights to get
    draw RECALL up (outcome_predictor.py), which deliberately inflates
    draw probabilities -- good for macro-F1, bad for probability quality.
  - A multinomial logistic regression on the same pre-match features,
    trained WITHOUT class balancing and with its regularization chosen by
    time-ordered CV on log loss -- i.e. trained for probabilities from the
    start, which is what a probabilistic benchmark actually rewards.

Headline: "share of the gap closed" -- how far each forecaster gets from a
no-skill baseline (the training-set home/draw/away frequencies) to the
closing market odds, in RPS. 0% = no better than base rates, 100% = as good as
the market. Paired bootstrap CIs over matches accompany the differences.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src.data_guard import existing_parquet_row_count, guard_against_shrinkage
from src.features import build_feature_table
from src.outcome_predictor import FORMATION_COLS_PREMATCH, NUMERIC, _season_code, _split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ODDS_PATH = Path(config.RAW_DIR) / "football_data_odds.parquet"
OUTCOMES = ["H", "D", "A"]  # ordered, as RPS requires
ODDS_SETS = {
    "pre-match": ["AvgH", "AvgD", "AvgA"],          # market average, collected Fri/Tue before
    "closing": ["AvgCH", "AvgCD", "AvgCA"],         # market average at kickoff
    "pinnacle closing": ["PSCH", "PSCD", "PSCA"],   # sensitivity check only (ends Jan 2026)
}
MARKET = "Market, closing odds"
DEFAULT_MODEL = "random_forest_prematch_optuna"
PROBABILITY_MODEL = "logistic_regression_prematch"
N_BOOTSTRAP = 2000


# --- odds ---------------------------------------------------------------

def fetch_odds() -> pd.DataFrame:
    """One row per match with pre-match/closing odds, team names FBref-normalized."""
    import soccerdata as sd

    raw = sd.MatchHistory(leagues=config.LEAGUE, seasons=config.SEASONS).read_games().reset_index()
    cols = ["season", "date", "home_team", "away_team", "FTR"]
    for odds_cols in ODDS_SETS.values():
        cols += [c for c in odds_cols if c in raw.columns]
    odds = raw[cols].copy()

    fbref_teams = set(pd.read_parquet(Path(config.RAW_DIR) / "fbref_schedule.parquet",
                                      columns=["team"])["team"])
    for side in ("home_team", "away_team"):
        odds[side] = odds[side].replace(config.FOOTBALL_DATA_NAME_MAP)
    unknown = sorted((set(odds["home_team"]) | set(odds["away_team"])) - fbref_teams)
    if unknown:
        raise RuntimeError(f"football-data team names with no FBref match: {unknown} -- add them "
                           f"to config.FOOTBALL_DATA_NAME_MAP")

    odds["season"] = odds["season"].astype(str)
    guard_against_shrinkage(ODDS_PATH, existing_parquet_row_count(ODDS_PATH), len(odds))
    ODDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    odds.to_parquet(ODDS_PATH, index=False)
    log.info("Saved %d matches of odds -> %s", len(odds), ODDS_PATH)
    return odds


def implied_probabilities(odds: pd.DataFrame, which: str) -> np.ndarray:
    """(n, 3) margin-free H/D/A probabilities; all-NaN rows where the odds are missing."""
    decimal = odds.reindex(columns=ODDS_SETS[which]).to_numpy(dtype=float)
    inv = 1.0 / decimal
    return inv / inv.sum(axis=1, keepdims=True)


# --- scoring ------------------------------------------------------------

def rps(probs: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Per-match ranked probability score for ordered outcomes (0=H, 1=D, 2=A)."""
    onehot = np.eye(3)[outcome]
    cum_diff = np.cumsum(probs, axis=1)[:, :2] - np.cumsum(onehot, axis=1)[:, :2]
    return (cum_diff ** 2).sum(axis=1) / 2


def log_loss(probs: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(len(outcome)), outcome], 1e-15, 1))


def brier(probs: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    return ((probs - np.eye(3)[outcome]) ** 2).sum(axis=1)


def _paired_ci(a: np.ndarray, b: np.ndarray, seed: int = 42) -> list[float]:
    """95% bootstrap CI (over matches) for mean(a - b)."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (N_BOOTSTRAP, len(d)))
    return [float(x) for x in np.percentile(d[idx].mean(axis=1), [2.5, 97.5])]


# --- forecasters --------------------------------------------------------

def _team_probs_to_hda(probs: np.ndarray, classes) -> np.ndarray:
    """Team-perspective W/D/L columns (in `classes` order) -> H/D/A for a
    home-perspective row."""
    idx = {c: i for i, c in enumerate(classes)}
    return probs[:, [idx["W"], idx["D"], idx["L"]]]


def train_probability_model(train: pd.DataFrame) -> Pipeline:
    categorical = [*FORMATION_COLS_PREMATCH, "venue"]
    train = train.sort_values("date")  # TimeSeriesSplit needs time order
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), NUMERIC),
    ])
    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 12), cv=TimeSeriesSplit(n_splits=config.N_CV_SPLITS),
        scoring="neg_log_loss", l1_ratios=(0.0,), use_legacy_attributes=False, max_iter=5000,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(train[categorical + NUMERIC], train["result"])
    log.info("Logistic regression: C=%.4g chosen by time-ordered CV on log loss",
             pipe.named_steps["model"].C_)
    return pipe


def build_test_frame(features: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Home-perspective test rows joined to odds on (season, home, away) --
    unique within a season, and unlike the date it survives postponements."""
    test_codes = [_season_code(s) for s in config.TEST_SEASONS]
    home = features[(features["venue"] == "Home") & features["season"].astype(str).isin(test_codes)]
    home = home.assign(season=home["season"].astype(str))
    merged = home.merge(odds.drop(columns=["date"], errors="ignore"),
                        left_on=["season", "team", "opponent"],
                        right_on=["season", "home_team", "away_team"], how="inner")
    if len(merged) < len(home):
        log.warning("%d/%d test matches had no odds row", len(home) - len(merged), len(home))
    ftr = merged["result"].map({"W": "H", "D": "D", "L": "A"})
    mismatch = (ftr != merged["FTR"]).sum()
    if mismatch:
        raise RuntimeError(f"{mismatch} matches disagree on the result between FBref and "
                           f"football-data -- the join is wrong, refusing to score it")
    merged["outcome"] = ftr.map({o: i for i, o in enumerate(OUTCOMES)})
    return merged.reset_index(drop=True)


# --- main ---------------------------------------------------------------

def _score(forecasts: dict[str, np.ndarray], y: np.ndarray, base_name: str, market_name: str) -> dict:
    scores = {name: {"rps": rps(p, y), "log_loss": log_loss(p, y), "brier": brier(p, y),
                     "accuracy": (p.argmax(axis=1) == y).astype(float)}
              for name, p in forecasts.items()}
    full_gap = scores[market_name]["rps"].mean() - scores[base_name]["rps"].mean()
    return {
        name: {
            **{m: float(v.mean()) for m, v in s.items()},
            "share_of_gap_to_market": float((s["rps"].mean() - scores[base_name]["rps"].mean()) / full_gap),
            "rps_minus_base_rates_ci95": _paired_ci(s["rps"], scores[base_name]["rps"]),
            "rps_minus_market_ci95": _paired_ci(s["rps"], scores[market_name]["rps"]),
        }
        for name, s in scores.items()
    }


def run(refresh_odds: bool = False) -> dict:
    odds = fetch_odds() if refresh_odds or not ODDS_PATH.exists() else pd.read_parquet(ODDS_PATH)
    features = build_feature_table()
    train, _ = _split(features)
    test = build_test_frame(features, odds)
    y = test["outcome"].to_numpy()
    X_cols = [*FORMATION_COLS_PREMATCH, "venue", *NUMERIC]

    forecasts: dict[str, np.ndarray] = {}
    base = train.loc[train["venue"] == "Home", "result"].map({"W": "H", "D": "D", "L": "A"})
    base_rates = base.value_counts(normalize=True).reindex(OUTCOMES).to_numpy()
    forecasts["Base rates (training-set H/D/A frequency)"] = np.tile(base_rates, (len(test), 1))

    default_path = Path(config.MODEL_DIR) / f"{DEFAULT_MODEL}.joblib"
    if default_path.exists():
        rf = joblib.load(default_path)
        le = joblib.load(Path(config.MODEL_DIR) / "label_encoder_prematch.joblib")
        forecasts["Random Forest (predict.py default, class-balanced)"] = _team_probs_to_hda(
            rf.predict_proba(test[X_cols]), le.classes_)
    else:
        log.warning("%s not found -- skipping it (run the Optuna pre-match tuning first)", default_path)

    lr = train_probability_model(train)
    model_dir = Path(config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, model_dir / f"{PROBABILITY_MODEL}.joblib")  # predict.py's pre-match default
    forecasts["Logistic regression (trained for probabilities)"] = _team_probs_to_hda(
        lr.predict_proba(test[X_cols]), lr.classes_)

    for which in ("pre-match", "closing"):
        probs = implied_probabilities(test, which)
        if np.isnan(probs).any():
            raise RuntimeError(f"{np.isnan(probs).any(axis=1).sum()} test matches lack "
                               f"market-average {which} odds")
        forecasts[f"Market, {which} odds"] = probs

    base_name = next(iter(forecasts))
    results = {
        "n_matches": len(test),
        "test_seasons": config.TEST_SEASONS,
        "market_odds": "average across bookmakers (football-data.co.uk), margin removed",
        "forecasters": _score(forecasts, y, base_name, MARKET),
    }

    # Sensitivity check: the same comparison against Pinnacle's closing
    # line, on the subset of matches that have it.
    pinnacle = implied_probabilities(test, "pinnacle closing")
    has_pinnacle = ~np.isnan(pinnacle).any(axis=1)
    if has_pinnacle.sum() >= 50:
        subset = {name: p[has_pinnacle] for name, p in forecasts.items()}
        subset["Pinnacle, closing odds"] = pinnacle[has_pinnacle]
        results["pinnacle_subset"] = {
            "n_matches": int(has_pinnacle.sum()),
            "forecasters": _score(subset, y[has_pinnacle], base_name, "Pinnacle, closing odds"),
        }

    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "market_benchmark.json").write_text(json.dumps(results, indent=2))
    per_match = test[["date", "season", "team", "opponent", "FTR"]].rename(
        columns={"team": "home", "opponent": "away"})
    for name, p in forecasts.items():
        for k, o in enumerate(OUTCOMES):
            per_match[f"{name} | P({o})"] = p[:, k]
    per_match.to_parquet(out_dir / "market_benchmark_predictions.parquet", index=False)
    plot(forecasts, y, results, out_dir / "market_benchmark.png")

    for name, r in results["forecasters"].items():
        log.info("%-52s RPS=%.4f  logloss=%.4f  acc=%.3f  gap closed=%5.1f%%",
                 name, r["rps"], r["log_loss"], r["accuracy"], 100 * r["share_of_gap_to_market"])
    if "pinnacle_subset" in results:
        ps = results["pinnacle_subset"]
        log.info("Pinnacle sensitivity check (%d matches): gap closed vs Pinnacle closing -- %s",
                 ps["n_matches"], ", ".join(f"{n.split(' (')[0]}: {100 * r['share_of_gap_to_market']:.0f}%"
                                           for n, r in ps["forecasters"].items()
                                           if n.startswith(("Logistic", "Random"))))
    return results


# --- plot ---------------------------------------------------------------

_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (all-pairs safe)
_INK, _INK_2, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot(forecasts: dict[str, np.ndarray], y: np.ndarray, results: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), width_ratios=[1, 1.15],
                                   facecolor=_SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(_SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(colors=_INK_2, labelsize=9)
        ax.set_axisbelow(True)

    # Left: how far each forecaster gets from base rates to the closing market
    names = [n for n in results["forecasters"] if not n.startswith("Base rates")]
    shares = [100 * results["forecasters"][n]["share_of_gap_to_market"] for n in names]
    ys = np.arange(len(names))[::-1]
    ax1.barh(ys, shares, height=0.55, color=_SLOTS[0])
    for yy, v in zip(ys, shares, strict=True):
        ax1.annotate(f"{v:.0f}%", (max(v, 0), yy), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=9, color=_INK)
    ax1.set_yticks(ys, [n.replace(" (", "\n(") for n in names], fontsize=9, color=_INK)
    ax1.axvline(0, color=_INK_2, linewidth=1)
    ax1.axvline(100, color=_INK_2, linewidth=1, linestyle=(0, (1, 2)))
    ax1.grid(axis="x", color=_GRID, linewidth=1)
    ax1.set_xlim(min(0, min(shares)) - 5, 115)
    ax1.set_xlabel("% of the way from base rates to the closing odds (by RPS)",
                   color=_INK_2, fontsize=10)
    ax1.set_title(f"Forecast skill vs. the betting market\n{results['n_matches']} test matches, "
                  f"{' + '.join(results['test_seasons'])}",
                  loc="left", fontsize=11.5, color=_INK, fontweight="semibold")

    # Right: reliability -- every (match, outcome) probability, binned
    ax2.plot([0, 1], [0, 1], color=_INK_2, linewidth=1, linestyle=(0, (1, 2)))
    series = [n for n in forecasts if n.startswith(("Logistic", "Random Forest", MARKET))]
    bins = np.linspace(0, 1, 11)
    onehot = np.eye(3)[y].ravel()
    for color, name in zip(_SLOTS, series, strict=False):
        p = forecasts[name].ravel()
        which = np.digitize(p, bins[1:-1])
        xs, fs = [], []
        for b in range(len(bins) - 1):
            m = which == b
            if m.sum() >= 15:
                xs.append(p[m].mean())
                fs.append(onehot[m].mean())
        ax2.plot(xs, fs, color=color, linewidth=2, marker="o", markersize=6,
                 markeredgecolor=_SURFACE, markeredgewidth=2, label=name)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.grid(color=_GRID, linewidth=1)
    ax2.set_xlabel("Forecast probability (bins of 0.1, >= 15 forecasts each)", color=_INK_2, fontsize=10)
    ax2.set_ylabel("How often it actually happened", color=_INK_2, fontsize=10)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=_INK)
    ax2.set_title("Calibration\non the diagonal = forecasts mean what they say",
                  loc="left", fontsize=11.5, color=_INK, fontweight="semibold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    log.info("Saved plot -> %s", out_path)


if __name__ == "__main__":
    import sys
    run(refresh_odds="--refresh-odds" in sys.argv)
