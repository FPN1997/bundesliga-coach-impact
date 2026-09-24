"""
Predict a Bundesliga match outcome using one of the trained models.

Computes each team's CURRENT form (rolling PPG/goal-diff/xG-diff/PPDA/deep-
completions over their last FEATURE_ROLLING_WINDOW matches, season-to-date
PPG, days into their current coach's tenure, recent-modal-formation)
directly from data/processed/match_dataset.parquet and
data/processed/coach_history.csv -- the same feature definitions
src/features.py uses, just evaluated as of now instead of as of a past
match, since there's no "next match" row in the historical data to attach
features to.

Two modes:

  Pre-match (no --formation/--opp-formation given): uses each team's
  recent-formation tendency. This is the mode to use for an actual
  upcoming match -- everything it needs is knowable before kickoff.
  Defaults to the model with the best-calibrated probabilities against
  the betting-market benchmark (src/market_benchmark.py): a logistic
  regression trained for probabilities. The Optuna-tuned Random Forest
  wins on macro-F1 but was trained with class-balanced weights, which
  inflates draw probabilities -- the wrong trade-off for a tool whose
  whole output IS the probabilities.

  Explanatory (--formation and --opp-formation given): uses the formations
  you supply directly. This answers "what does the model think about this
  matchup GIVEN these formations" -- useful for exploring the formation
  matrix, not a genuine forecast, since you don't know the opponent's
  actual matchday formation in advance. Defaults to untuned XGBoost; the
  tuned alternatives score within noise of it (docs/results-in-depth.md).

Usage:
    python predict.py --list-teams
    python predict.py --team "Bayern Munich" --opponent Dortmund --venue home
    python predict.py --team "Bayern Munich" --opponent Dortmund --venue home \\
        --formation 4-2-3-1 --opp-formation 4-3-3
    python predict.py --team "Bayern Munich" --opponent Dortmund --venue home \\
        --model xgboost --tuning grid
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd

import config
from src.formation_utils import clean_formation

ROLLING_WINDOW = config.FEATURE_ROLLING_WINDOW


def _recent_formation(formations: list[str | None], window: int) -> str | None:
    """The mode of this team's last `window` PLAYED (non-null) formations,
    right up to and including their most recent match -- the "as of right
    now" analogue of features.py's shift-safe recent_formation, but not
    itself the same function: build_feature_table()'s recent_formation is
    attached to a past match and therefore correctly EXCLUDES that match's
    own formation (see features._rolling_mode); here there's no future
    match to shift away from, so the correct window for "recent tendency
    going into a hypothetical next match" is the last `window` matches
    actually played, the most recent one included -- the direct analogue
    of how current_team_state()'s numeric form stats use
    `team_matches.tail(window)` below, not features._rolling_mode()'s
    shifted definition. Calling features._rolling_mode(...)[-1] here would
    silently be one match stale (verified live: 5/28 teams currently on
    record disagree between the two, e.g. a team that just switched
    formation in its most recent match)."""
    played = [f for f in formations if f is not None][-window:]
    if not played:
        return None
    return Counter(played).most_common(1)[0][0]


def _load_matches() -> pd.DataFrame:
    path = Path(config.PROCESSED_DIR) / "match_dataset.parquet"
    if not path.exists():
        sys.exit(f"{path} not found -- run `bundesliga pipeline` first.")
    df = pd.read_parquet(path)
    df["formation"] = df["formation"].apply(clean_formation)
    df["opp_formation"] = df["opp_formation"].apply(clean_formation)
    return df


def _load_coach_history() -> pd.DataFrame:
    path = Path(config.COACH_HISTORY_RESOLVED_CSV)
    if not path.exists():
        sys.exit(f"{path} not found -- run `bundesliga pipeline` first.")
    return pd.read_csv(path, parse_dates=["start_date"])


def _resolve_team(name: str, known_teams: list[str]) -> str:
    if name in known_teams:
        return name
    close = difflib.get_close_matches(name, known_teams, n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    sys.exit(f"Unknown team '{name}'.{hint} Run --list-teams to see all options.")


def current_team_state(team: str, matches: pd.DataFrame, coaches: pd.DataFrame) -> dict:
    """This team's form as of right now -- the same feature definitions as
    src/features.py, evaluated over the team's most recent matches rather
    than shifted away from a specific past match (there is no "next match"
    row to shift relative to)."""
    team_matches = matches[matches["team"] == team].sort_values("date")
    if len(team_matches) < ROLLING_WINDOW:
        sys.exit(f"{team} only has {len(team_matches)} matches in the dataset -- "
                 f"need at least {ROLLING_WINDOW} for a form-based prediction.")

    recent = team_matches.tail(ROLLING_WINDOW)
    current_season = team_matches["season"].iloc[-1]
    season_matches = team_matches[team_matches["season"] == current_season]

    coach_rows = coaches[
        (coaches["team"] == team) & (coaches["start_date"] <= pd.Timestamp.today())
    ]
    if coach_rows.empty:
        coach_tenure_days = None
        coach_name = None
    else:
        latest = coach_rows.sort_values("start_date").iloc[-1]
        coach_tenure_days = (pd.Timestamp.today() - latest["start_date"]).days
        coach_name = latest["coach"]

    return {
        "team": team,
        "coach": coach_name,
        "coach_tenure_days": coach_tenure_days,
        "form_ppg": recent["points"].mean(),
        "form_goal_diff": (recent["gf"] - recent["ga"]).mean(),
        "form_xg_diff": (recent["xg"] - recent["xga"]).mean(),
        "form_ppda": recent["ppda"].mean(),
        "form_deep_completions": recent["deep_completions"].mean(),
        "season_ppg_to_date": season_matches["points"].mean(),
        "recent_formation": _recent_formation(team_matches["formation"].tolist(), ROLLING_WINDOW)
                             or team_matches["formation"].iloc[-1],
        "last_match_date": team_matches["date"].iloc[-1].date(),
    }


def feature_row(team_state: dict, opp_state: dict, venue: str,
                formation: str | None = None, opp_formation: str | None = None) -> dict:
    """The model input row for one match, from both teams' current_team_state().
    Explanatory mode if formations are given, pre-match (recent formations) otherwise."""
    row = {"venue": "Home" if venue == "home" else "Away"}
    for key in ("form_ppg", "form_goal_diff", "form_xg_diff", "form_ppda", "form_deep_completions",
                "season_ppg_to_date", "coach_tenure_days"):
        row[key] = team_state[key]
        row[f"opp_{key}"] = opp_state[key]
    # The models can't take a missing numeric value -- fall back to 0 rather
    # than crash if a coach lookup came up empty (e.g. a club with no
    # coach-history rows yet). Shown as "None" in the printed form line.
    for key in ("coach_tenure_days", "opp_coach_tenure_days"):
        if row[key] is None:
            row[key] = 0
    if formation is not None:
        row["formation"] = clean_formation(formation)
        row["opp_formation"] = clean_formation(opp_formation)
    else:
        row["recent_formation"] = team_state["recent_formation"]
        row["opp_recent_formation"] = opp_state["recent_formation"]
    return row


def forecast(pipe, le, team_state: dict, opp_state: dict, venue: str,
             formation: str | None = None, opp_formation: str | None = None) -> tuple[dict, dict]:
    """({"W": p, "D": p, "L": p} from team_state's side, the model input row)."""
    row = feature_row(team_state, opp_state, venue, formation, opp_formation)
    proba = pipe.predict_proba(pd.DataFrame([row]))[0]
    # Label-encoded models (RF/XGBoost) predict classes 0/1/2; the logistic
    # regression was fit on the "D"/"L"/"W" strings directly.
    classes = pipe.classes_ if isinstance(pipe.classes_[0], str) else le.classes_
    return {str(c): float(p) for c, p in zip(classes, proba, strict=True)}, row


def _model_paths(variant: str, model: str, tuning: str) -> tuple[Path, Path]:
    suffix = {"none": "", "grid": "_tuned", "grid-embargoed": "_tuned_embargoed",
              "optuna": "_optuna"}[tuning]
    variant_tag = "_prematch" if variant == "prematch" else ""
    model_dir = Path(config.MODEL_DIR)
    model_path = model_dir / f"{model}{variant_tag}{suffix}.joblib"
    encoder_path = model_dir / f"label_encoder{variant_tag}.joblib"
    if not model_path.exists():
        sys.exit(f"{model_path} not found -- train it first "
                 f"(`bundesliga train`, `bundesliga tune --method ...`, or "
                 f"`bundesliga benchmark` for logistic_regression).")
    return model_path, encoder_path


def predict(
    team: str, opponent: str, venue: str,
    formation: str | None, opp_formation: str | None,
    model_name: str, tuning: str,
) -> None:
    matches = _load_matches()
    coaches = _load_coach_history()
    known_teams = sorted(matches["team"].unique())

    team = _resolve_team(team, known_teams)
    opponent = _resolve_team(opponent, known_teams)

    team_state = current_team_state(team, matches, coaches)
    opp_state = current_team_state(opponent, matches, coaches)

    explanatory = formation is not None or opp_formation is not None
    if explanatory and not (formation and opp_formation):
        sys.exit("Give both --formation and --opp-formation, or neither.")

    variant = "actual" if explanatory else "prematch"
    # Pre-match default: the logistic regression, which scores best on
    # probability quality against the betting market (see the module
    # docstring). It has no --tuning variants: its regularization is
    # already chosen by time-ordered CV on log loss when it's trained.
    model_name = model_name or ("xgboost" if variant == "actual" else "logistic_regression")
    tuning = tuning or "none"

    model_path, encoder_path = _model_paths(variant, model_name, tuning)
    pipe = joblib.load(model_path)
    le = joblib.load(encoder_path)
    probs, row = forecast(pipe, le, team_state, opp_state, venue, formation, opp_formation)

    print(f"\n{team} (home={venue=='home'}) vs {opponent}")
    if model_name == "logistic_regression":
        how = "regularization chosen by time-ordered CV"
    else:
        how = f"{tuning}-tuned" if tuning != "none" else "untuned"
    print(f"Model: {model_name} ({variant}, {how})\n")
    for label, name in [("W", f"{team} win"), ("D", "Draw"), ("L", f"{opponent} win")]:
        bar = "#" * round(probs.get(label, 0) * 40)
        print(f"  {name:16s} {probs.get(label, 0):5.1%}  {bar}")

    print(f"\n{team} form: PPG(last {ROLLING_WINDOW})={team_state['form_ppg']:.2f}  "
          f"season PPG={team_state['season_ppg_to_date']:.2f}  "
          f"coach={team_state['coach']} ({team_state['coach_tenure_days']}d)  "
          f"recent formation={team_state['recent_formation']}  "
          f"(as of {team_state['last_match_date']})")
    print(f"{opponent} form: PPG(last {ROLLING_WINDOW})={opp_state['form_ppg']:.2f}  "
          f"season PPG={opp_state['season_ppg_to_date']:.2f}  "
          f"coach={opp_state['coach']} ({opp_state['coach_tenure_days']}d)  "
          f"recent formation={opp_state['recent_formation']}  "
          f"(as of {opp_state['last_match_date']})")
    if explanatory:
        print(f"\nFormations given: {team}={row['formation']}, {opponent}={row['opp_formation']} "
              f"-- explanatory mode: this is 'what if these formations are used', "
              f"not a pre-kickoff forecast (see README).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", help="Team to predict the result FOR")
    parser.add_argument("--opponent", help="Opponent")
    parser.add_argument("--venue", choices=["home", "away"], default="home",
                         help="Whether --team is playing at home (default: home)")
    parser.add_argument("--formation", help="--team's formation (switches to explanatory mode)")
    parser.add_argument("--opp-formation", help="--opponent's formation")
    parser.add_argument("--model", choices=["logistic_regression", "random_forest", "xgboost"],
                         default=None,
                         help="Default: xgboost for explanatory, logistic_regression for pre-match "
                              "(logistic_regression is pre-match only)")
    parser.add_argument("--tuning", choices=["none", "grid", "grid-embargoed", "optuna"],
                         default=None,
                         help="Default: none. Applies to random_forest/xgboost. "
                              "grid-embargoed uses `bundesliga tune --method embargoed`'s models "
                              "(see README Hyperparameter tuning)")
    parser.add_argument("--list-teams", action="store_true",
                         help="Print every team name in the dataset and exit")
    args = parser.parse_args()

    if args.list_teams:
        matches = _load_matches()
        for t in sorted(matches["team"].unique()):
            print(t)
        return

    if not args.team or not args.opponent:
        parser.error("--team and --opponent are required (or use --list-teams)")

    predict(args.team, args.opponent, args.venue, args.formation, args.opp_formation,
            args.model, args.tuning)


if __name__ == "__main__":
    main()
