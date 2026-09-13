"""
Predict a Bundesliga match outcome using one of the trained models.

Computes each team's CURRENT form (rolling PPG/goal-diff/xG-diff/PPDA over
their last FEATURE_ROLLING_WINDOW matches, season-to-date PPG, days into
their current coach's tenure, recent-modal-formation) directly from
data/processed/match_dataset.parquet and data/processed/coach_history.csv
-- the same feature definitions src/features.py uses, just evaluated as of
now instead of as of a past match, since there's no "next match" row in
the historical data to attach features to.

Two modes:

  Pre-match (no --formation/--opp-formation given): uses each team's
  recent-formation tendency. This is the mode to use for an actual
  upcoming match -- everything it needs is knowable before kickoff.
  Defaults to the strongest pre-match-legal model documented in the
  README: GridSearchCV-tuned Random Forest.

  Explanatory (--formation and --opp-formation given): uses the formations
  you supply directly. This answers "what does the model think about this
  matchup GIVEN these formations" -- useful for exploring the formation
  matrix, not a genuine forecast, since you don't know the opponent's
  actual matchday formation in advance. Defaults to the best explanatory
  model: untuned XGBoost.

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
from pathlib import Path

import joblib
import pandas as pd

import config
from src.formation_utils import clean_formation
from src.features import _rolling_mode

ROLLING_WINDOW = config.FEATURE_ROLLING_WINDOW


def _load_matches() -> pd.DataFrame:
    path = Path(config.PROCESSED_DIR) / "match_dataset.parquet"
    if not path.exists():
        sys.exit(f"{path} not found -- run `python run_pipeline.py` first.")
    df = pd.read_parquet(path)
    df["formation"] = df["formation"].apply(clean_formation)
    df["opp_formation"] = df["opp_formation"].apply(clean_formation)
    return df


def _load_coach_history() -> pd.DataFrame:
    path = Path(config.COACH_HISTORY_RESOLVED_CSV)
    if not path.exists():
        sys.exit(f"{path} not found -- run `python run_pipeline.py` first.")
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
        "season_ppg_to_date": season_matches["points"].mean(),
        "recent_formation": _rolling_mode(team_matches["formation"].tolist(), ROLLING_WINDOW)[-1]
                             or team_matches["formation"].iloc[-1],
        "last_match_date": team_matches["date"].iloc[-1].date(),
    }


def _model_paths(variant: str, model: str, tuning: str) -> tuple[Path, Path]:
    suffix = {"none": "", "grid": "_tuned", "optuna": "_optuna"}[tuning]
    variant_tag = "_prematch" if variant == "prematch" else ""
    model_dir = Path(config.MODEL_DIR)
    model_path = model_dir / f"{model}{variant_tag}{suffix}.joblib"
    encoder_path = model_dir / f"label_encoder{variant_tag}.joblib"
    if not model_path.exists():
        sys.exit(f"{model_path} not found -- train it first "
                 f"(run_outcome_predictor.py / run_prematch_predictor.py / "
                 f"run_tune_hyperparameters.py / run_tune_optuna.py, as appropriate).")
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
    default_model = "xgboost" if variant == "actual" else "random_forest"
    default_tuning = "none" if variant == "actual" else "grid"
    model_name = model_name or default_model
    tuning = tuning or default_tuning

    model_path, encoder_path = _model_paths(variant, model_name, tuning)
    pipe = joblib.load(model_path)
    le = joblib.load(encoder_path)

    row = {
        "venue": "Home" if venue == "home" else "Away",
        "form_ppg": team_state["form_ppg"],
        "form_goal_diff": team_state["form_goal_diff"],
        "form_xg_diff": team_state["form_xg_diff"],
        "form_ppda": team_state["form_ppda"],
        "season_ppg_to_date": team_state["season_ppg_to_date"],
        "coach_tenure_days": team_state["coach_tenure_days"],
        "opp_form_ppg": opp_state["form_ppg"],
        "opp_form_goal_diff": opp_state["form_goal_diff"],
        "opp_form_xg_diff": opp_state["form_xg_diff"],
        "opp_form_ppda": opp_state["form_ppda"],
        "opp_season_ppg_to_date": opp_state["season_ppg_to_date"],
        "opp_coach_tenure_days": opp_state["coach_tenure_days"],
    }
    # Random Forest (unlike XGBoost) can't handle a missing numeric value --
    # fall back to 0 rather than crash if a coach lookup came up empty (e.g.
    # a club not yet in config.CLUB_TRANSFERMARKT_ID). Flagged in the
    # printed output, not silently swallowed.
    for key in ("coach_tenure_days", "opp_coach_tenure_days"):
        if row[key] is None:
            row[key] = 0
    if explanatory:
        row["formation"] = clean_formation(formation)
        row["opp_formation"] = clean_formation(opp_formation)
    else:
        row["recent_formation"] = team_state["recent_formation"]
        row["opp_recent_formation"] = opp_state["recent_formation"]

    X = pd.DataFrame([row])
    proba = pipe.predict_proba(X)[0]
    probs = dict(zip(le.classes_, proba))

    print(f"\n{team} (home={venue=='home'}) vs {opponent}")
    print(f"Model: {model_name} ({variant}"
          f"{', ' + tuning + '-tuned' if tuning != 'none' else ', untuned'})\n")
    for label, name in [("W", f"{team} win"), ("D", "Draw"), ("L", f"{opponent} win")]:
        bar = "#" * int(round(probs.get(label, 0) * 40))
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
    parser.add_argument("--model", choices=["random_forest", "xgboost"], default=None,
                         help="Default: xgboost for explanatory, random_forest for pre-match")
    parser.add_argument("--tuning", choices=["none", "grid", "optuna"], default=None,
                         help="Default: none for explanatory, grid for pre-match")
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
