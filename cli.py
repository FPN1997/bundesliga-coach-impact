"""
One entry point for the whole project.

    bundesliga pipeline [--skip-fetch]      fetch (incl. odds) -> merge -> coach impact/effect -> formations
    bundesliga train [--variant ...]        outcome predictor (actual-formation and/or pre-match)
    bundesliga tune --method ...            hyperparameter tuning: grid | embargoed | optuna
    bundesliga native-categorical           XGBoost native-categorical-split experiment
    bundesliga coach-effect                 does a coaching change help, beyond regression to the mean?
    bundesliga bounce                       coach-bounce predictor (small-sample, see README)
    bundesliga sack-o-meter                 this week's sack risk, recovery and change effect per club
    bundesliga squad                        squads, injuries, winter signings (Transfermarkt, cached)
    bundesliga backfill                     one polite Transfermarkt run, coach histories first (see below)
    bundesliga benchmark [--refresh-odds]   pre-match forecasts vs. betting-market odds
    bundesliga predict ...                  forecast a match (see `bundesliga predict --help`)
    bundesliga site                         rebuild the published results page (site/)
    bundesliga reproduce                    every modelling step above, in order (~20 min)

`bundesliga` is installed by `pip install -e .`; `python cli.py ...` works the
same without installing. Every step reads and writes config.py's directories,
so steps are run one at a time, in order -- never in parallel (two tuning
runs writing outputs/outcome_model_metrics.json at once is how results got
silently lost once).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import config

log = logging.getLogger("bundesliga")

VARIANTS = ("actual", "prematch")


def _require(*paths: Path, hint: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        sys.exit(f"Missing {missing} -- run `{hint}` first.")


def _match_dataset() -> Path:
    return Path(config.PROCESSED_DIR) / "match_dataset.parquet"


def _variants(choice: str) -> list[str]:
    return list(VARIANTS) if choice == "both" else [choice]


def _formation_cols(variant: str):
    from src.outcome_predictor import FORMATION_COLS_ACTUAL, FORMATION_COLS_PREMATCH
    return FORMATION_COLS_PREMATCH if variant == "prematch" else FORMATION_COLS_ACTUAL


def _suffix(variant: str) -> str:
    return "_prematch" if variant == "prematch" else ""


def refresh_coach_history(fetch) -> None:
    """Run the coach-history fetch, but if Transfermarkt is blocking us, keep
    the previous coach history rather than failing the whole refresh --
    coaching changes are rare, so a stale week costs far less than losing the
    week's match data. Without a previous file there's nothing to fall back on."""
    from src.fetch_coach_history import TransfermarktBlocked
    try:
        fetch()
    except TransfermarktBlocked as exc:
        if not Path(config.COACH_HISTORY_RESOLVED_CSV).exists():
            raise
        log.warning("Transfermarkt is blocking requests (%s) -- keeping the previous coach history. "
                    "Coaching changes since its last successful fetch are missing until it unblocks.",
                    exc)


# --- commands -------------------------------------------------------------

def cmd_pipeline(args) -> None:
    from src.build_dataset import build_dataset
    from src.coach_change_effect import run as estimate_coach_change_effect
    from src.coach_impact import compute_coach_impact
    from src.formation_matrix import build_formation_matrix, coach_preferred_formations
    from src.sack_o_meter import run as update_sack_o_meter

    if args.skip_fetch:
        log.info("Skipping fetch steps (--skip-fetch)")
    else:
        from src.fetch_coach_history import fetch_coach_history
        from src.fetch_fbref import fetch_fbref_matches
        from src.fetch_odds import fetch_odds
        from src.fetch_understat import fetch_understat_matches
        log.info("Step 1/6: FBref (results + formations)")
        fetch_fbref_matches()
        log.info("Step 2/6: Understat (xG, PPDA, deep completions)")
        fetch_understat_matches()
        log.info("Step 3/6: Coach tenure history (Transfermarkt)")
        refresh_coach_history(fetch_coach_history)
        log.info("Step 4/6: Betting odds (football-data.co.uk, for fixture difficulty)")
        fetch_odds()

    log.info("Step 5/6: Merging into match_dataset.parquet")
    build_dataset()
    log.info("Step 6/6: Analysis")
    compute_coach_impact()
    estimate_coach_change_effect()
    update_sack_o_meter()  # weekly: runs with every refresh
    build_formation_matrix()
    coach_preferred_formations()
    log.info("Done. Outputs are in %s/", config.OUTPUT_DIR)


def cmd_train(args) -> None:
    from src.outcome_predictor import train_and_evaluate
    _require(_match_dataset(), hint="bundesliga pipeline")
    for v in _variants(args.variant):
        train_and_evaluate(formation_cols=_formation_cols(v), variant=_suffix(v))


def cmd_tune(args) -> None:
    _require(_match_dataset(), hint="bundesliga pipeline")
    if args.method == "optuna":
        from src.tune_optuna import tune_and_evaluate
        for v in _variants(args.variant):
            tune_and_evaluate(formation_cols=_formation_cols(v), variant=_suffix(v))
        return
    from src.tune_hyperparameters import tune_and_evaluate
    gap = config.CV_EMBARGO_GAP if args.method == "embargoed" else 0
    for v in _variants(args.variant):
        tune_and_evaluate(formation_cols=_formation_cols(v), variant=_suffix(v), gap=gap)


def cmd_native_categorical(args) -> None:
    from src.xgboost_native_categorical import train_and_evaluate
    _require(_match_dataset(), hint="bundesliga pipeline")
    for v in VARIANTS:
        train_and_evaluate(formation_cols=_formation_cols(v), variant=_suffix(v))


def cmd_coach_effect(args) -> None:
    from src.coach_change_effect import run
    _require(_match_dataset(), hint="bundesliga pipeline")
    run()


def cmd_bounce(args) -> None:
    from src.coach_bounce import train_and_evaluate
    _require(_match_dataset(), Path(config.OUTPUT_DIR) / "coach_impact_rankings.csv",
             hint="bundesliga pipeline")
    train_and_evaluate()


def cmd_sack_o_meter(args) -> None:
    from src.sack_o_meter import run
    _require(_match_dataset(), Path(config.OUTPUT_DIR) / "coach_change_effect.json",
             hint="bundesliga pipeline")
    run()


def cmd_squad(args) -> None:
    from src.fetch_squads import fetch_squads
    _require(_match_dataset(), hint="bundesliga pipeline")
    fetch_squads()


COACH_LIST_MAX_AGE_DAYS = 7
BACKFILL_DONE = Path(config.PROCESSED_DIR) / "backfill_complete"


def cmd_backfill(args) -> None:
    """One run's Transfermarkt budget (src/transfermarkt_client.py), spent in
    priority order, stopping everything at the first block:

    1. the Bundesliga coach list, if more than a week old -- the sack-o-meter
       names coaches from it, and the weekly refresh keeps a stale one while
       Transfermarkt blocks;
    2. coaching histories for the other top-5 leagues -- more leagues narrow
       the study's interval far more than injury data would;
    3. squads and injuries, with whatever budget is left.

    Writes data/processed/backfill_complete once 2 and 3 are both done."""
    import time

    from src import transfermarkt_client as tm
    from src.fetch_coach_history import fetch_coach_history
    from src.fetch_league_coaches import fetch_league_coaches
    from src.fetch_squads import fetch_squads, injuries_path
    _require(_match_dataset(), hint="bundesliga pipeline")

    coach_list = Path(config.COACH_HISTORY_RESOLVED_CSV)
    age = (time.time() - coach_list.stat().st_mtime) / 86400 if coach_list.exists() else float("inf")
    if age > COACH_LIST_MAX_AGE_DAYS:
        log.info("Backfill 1/3: the Bundesliga coach list is %.0f days old -- refreshing it", age)
        try:
            fetch_coach_history()
        except tm.TransfermarktBlocked as exc:
            log.error("Transfermarkt blocked the scrape: %s", exc)
            return
        except tm.RequestBudgetReached as exc:
            log.warning("Stopped at this run's request budget (%s)", exc)
            return
    else:
        log.info("Backfill 1/3: the Bundesliga coach list is current (%.1f days old)", age)

    log.info("Backfill 2/3: coaching histories for the other top-5 leagues")
    try:
        leagues = fetch_league_coaches()
    except Exception as exc:  # e.g. Understat unreachable: don't let it hold up the rest
        log.error("Other-league coach histories failed (%s) -- moving on to squads this run", exc)
        leagues = {"complete": False, "stopped": None}
    if leagues["stopped"]:
        return  # blocked or out of budget: squads wait for the next run

    log.info("Backfill 3/3: squads and injuries, %d requests left this run", tm.remaining())
    if tm.remaining() == 0:
        log.warning("Stopped at this run's request budget -- squads continue next run")
        return
    fetch_squads()

    if leagues["complete"] and injuries_path().exists():
        BACKFILL_DONE.parent.mkdir(parents=True, exist_ok=True)
        BACKFILL_DONE.write_text(f"{time.strftime('%Y-%m-%d %H:%M')}\n")
        log.info("Backfill complete: every coach history and all squad data are in.")
    log.info("Transfermarkt requests this run: %d", tm.live_requests())


def cmd_benchmark(args) -> None:
    from src.market_benchmark import run
    _require(_match_dataset(), hint="bundesliga pipeline")
    run(refresh_odds=args.refresh_odds)


def cmd_predict(args) -> None:
    # Normally short-circuited in main() -- argparse subcommands can't pass
    # through "--flag" arguments (a known argparse limitation), so predict's
    # own parser has to see them directly.
    import predict
    sys.argv = ["bundesliga predict", *args.predict_args]
    predict.main()


def cmd_site(args) -> None:
    from src.site import build_site
    build_site()


def cmd_reproduce(args) -> None:
    """Every modelling result in the README, regenerated in dependency order."""
    _require(_match_dataset(), hint="bundesliga pipeline")
    ns = argparse.Namespace
    steps = [
        ("train (both variants)", cmd_train, ns(variant="both")),
        ("tune: GridSearchCV", cmd_tune, ns(method="grid", variant="both")),
        ("tune: embargoed GridSearchCV", cmd_tune, ns(method="embargoed", variant="both")),
        ("tune: Optuna", cmd_tune, ns(method="optuna", variant="both")),
        ("XGBoost native-categorical experiment", cmd_native_categorical, ns()),
        ("coaching-change effect", cmd_coach_effect, ns()),
        ("coach-bounce predictor", cmd_bounce, ns()),
        ("sack-o-meter", cmd_sack_o_meter, ns()),
        ("betting-market benchmark", cmd_benchmark, ns(refresh_odds=args.refresh_odds)),
        ("results page", cmd_site, ns()),
    ]
    for i, (label, fn, step_args) in enumerate(steps, 1):
        log.info("=== reproduce %d/%d: %s ===", i, len(steps), label)
        fn(step_args)


# --- parser ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bundesliga", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("pipeline", help="fetch, merge, and run the coach/formation analyses")
    p.add_argument("--skip-fetch", action="store_true", help="reuse data/raw instead of scraping")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("train", help="train the outcome predictor")
    p.add_argument("--variant", choices=[*VARIANTS, "both"], default="both")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("tune", help="tune the outcome predictor's hyperparameters")
    p.add_argument("--method", choices=["grid", "embargoed", "optuna"], required=True)
    p.add_argument("--variant", choices=[*VARIANTS, "both"], default="both")
    p.set_defaults(func=cmd_tune)

    sub.add_parser("native-categorical", help="XGBoost native-categorical experiment") \
        .set_defaults(func=cmd_native_categorical)
    sub.add_parser("coach-effect", help="effect of a coaching change beyond regression to the mean") \
        .set_defaults(func=cmd_coach_effect)
    sub.add_parser("bounce", help="coach-bounce predictor").set_defaults(func=cmd_bounce)
    sub.add_parser("sack-o-meter", help="this week's sack risk, recovery and change effect per club") \
        .set_defaults(func=cmd_sack_o_meter)
    sub.add_parser("squad", help="squads, market values, winter signings and injuries "
                                 "(Transfermarkt; slow the first time, then cached)") \
        .set_defaults(func=cmd_squad)

    sub.add_parser("backfill", help="one polite Transfermarkt run: coach histories first, then squads") \
        .set_defaults(func=cmd_backfill)

    p = sub.add_parser("benchmark", help="score pre-match forecasts against betting odds")
    p.add_argument("--refresh-odds", action="store_true", help="re-download odds first")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("predict", help="forecast a match (args pass through to predict.py)",
                       add_help=False)
    p.add_argument("predict_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_predict)

    sub.add_parser("site", help="rebuild the results page in site/").set_defaults(func=cmd_site)

    p = sub.add_parser("reproduce", help="rerun every modelling step in order (~20 min)")
    p.add_argument("--refresh-odds", action="store_true")
    p.set_defaults(func=cmd_reproduce)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    # config.py's directories are relative to the project root; an installed
    # `bundesliga` command can be run from anywhere, so anchor it there.
    os.chdir(Path(__file__).resolve().parent)
    if argv[:1] == ["predict"]:
        cmd_predict(argparse.Namespace(predict_args=argv[1:]))
        return
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
