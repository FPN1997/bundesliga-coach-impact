# Does sacking the coach work?

A Bundesliga analytics project: coaching changes measured against regression to the
mean, and match forecasts benchmarked against the betting market. Thirteen seasons
(2014-15 to 2026-27), 3,708 matches, four scraped sources.

[![CI](https://github.com/FPN1997/bundesliga-coach-impact/actions/workflows/ci.yml/badge.svg)](https://github.com/FPN1997/bundesliga-coach-impact/actions/workflows/ci.yml)

**→ [Interactive results page](https://fpn1997.github.io/bundesliga-coach-impact/)** —
every mid-season sacking since 2014, the forecast benchmark, and forecasts for the next
matchday.

## Findings

### A new coach is worth about +0.2 points per game — a third of the "bounce"

Teams that sack their coach mid-season take **+0.54 points per game** more over the next
8 matches than over the previous 8. But teams in the same slump that *kept* their coach
improved **+0.35** anyway, once the run of fixtures is accounted for too — bad runs end on
their own. That leaves **+0.19 points per game** for the change itself (95% interval
+0.04 to +0.30), across 65 mid-season sackings from 2014-15 to 2026-27. Two independent
measures agree:

- **Points:** +0.19 per game beyond expectation (+0.04 to +0.30).
- **Chance quality** (xG difference: the quality of chances created minus conceded, which
  doesn't depend on whether shots happen to go in): **+0.31 per game** beyond expectation
  (+0.11 to +0.46) — worth about +0.18 points per game.

It isn't just an easier run of fixtures (sacked teams' next 8 fixtures were barely easier
than usual, and adjusting for them changes the estimate from +0.16 to +0.19), and it isn't
mainly new signings: for the 39 sackings with no transfer window open afterwards — a frozen
squad — points still improved +0.20 per game beyond expectation. Summer appointments
show no clear effect (+0.05, −0.07 to +0.15). "The change" still means everything that
changes at that moment, not the new coach alone
([details](docs/results-in-depth.md#the-effect-of-a-coaching-change)).

![Sackings vs. teams that kept their coach](docs/coach_change_effect.png)

*Each blue dot is a mid-season sacking; the orange line is what teams in the same
position that kept their coach did. The comparison adjusts for points and xG before the
change and for the difficulty of the fixtures before and after, and the intervals
resample whole teams. Method:
[results-in-depth.md](docs/results-in-depth.md#the-effect-of-a-coaching-change).*

### The match forecasts get 72% of the way to the betting market

On 321 held-out matches, a logistic regression on rolling form, xG, pressing and coach
tenure closes **72% of the gap** between naive base rates and the closing betting odds,
measured by ranked probability score (the standard for football forecasts). A tuned
Random Forest does much worse, at 46%: it was trained to catch draws, and overestimates
them (33% on average, against a real draw rate of 24%).

![Forecast skill vs. the betting market, and calibration](docs/market_benchmark.png)

### Most tuning "wins" were noise

An earlier run ranked Optuna-tuned Random Forest as the clear best pre-match model
(macro-F1 0.522). After a week's data refresh — the tuning itself is deterministic — it
scored 0.491, and with five more seasons of training data 0.492; the other tuning
conclusions reshuffled each time. At this sample size, differences of a few hundredths
between tuning methods aren't real; the untuned models are as good as anything. The
robust comparison is the probabilistic benchmark above.
([details](docs/results-in-depth.md#hyperparameter-tuning))

### A coach's track record doesn't predict the size of the bounce

For 67 changes where the incoming coach had coached at least 10 earlier matches in the
data, a model can predict how big the points swing will be (leave-one-out R² +0.28) — but
all of that comes from how bad the run was beforehand, i.e. regression to the mean again
(team form alone: +0.33). The incoming coach's own record adds nothing (on its own: −0.04).
([details](docs/results-in-depth.md#coach-bounce-predictor))

### Formation matchups

![Points per game by formation matchup](docs/formation_matchup_heatmap.png)

*Grey is the league average (1.38 points per game). Raw averages mix up "this formation
works" with "strong teams use this formation"; the outcome model's
[form-adjusted version](docs/results-in-depth.md#formation-matchups) separates them.*

## How it works

```mermaid
flowchart LR
    FB[FBref<br/>results, formations] --> DS[Match dataset<br/>coach attached to every match]
    US[Understat<br/>xG, pressing] --> DS
    TM[Transfermarkt<br/>coach tenures] --> DS
    DS --> CE[Coaching-change effect]
    DS --> FM[Formation matchups]
    DS --> FC[Match forecasts]
    FD[football-data.co.uk<br/>betting odds] --> MB[Market benchmark]
    FD -->|fixture difficulty| CE
    FC --> MB
    CE & FM & FC & MB --> WEB[Results page]
```

- **Leak-safe features.** Every rolling statistic uses only matches strictly before the
  one being predicted, and evaluation is a strict time split. Both properties are tested.
- **Proper scoring.** Forecasts are scored with RPS, log loss and calibration against
  bookmaker odds, with bootstrap intervals — not just accuracy.
- **Self-maintaining data.** New clubs' Transfermarkt ids and name spellings resolve
  automatically; every scraper refuses to overwrite good data with a suspiciously small
  scrape; a weekly job refreshes everything, with a watchdog, retries and a failure
  notification.

## Quickstart

```bash
git clone https://github.com/FPN1997/bundesliga-coach-impact.git
cd bundesliga-coach-impact
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # on macOS also: brew install libomp (XGBoost needs it)
bundesliga pipeline            # scrape, merge, run the analyses (~5 min; needs Chrome for FBref)
bundesliga reproduce           # every model and number in this README, in order (~20 min)
```

Forecast a match from each team's current form:

```bash
bundesliga predict --team "Bayern Munich" --opponent Dortmund --venue home
```

```
Bayern Munich (home=True) vs Dortmund
Model: logistic_regression (prematch, regularization chosen by time-ordered CV)

  Bayern Munich win 66.1%  ##########################
  Draw             18.3%  #######
  Dortmund win     15.6%  ######
```

| Command | What it does |
|---|---|
| `bundesliga pipeline [--skip-fetch]` | Scrape (or reuse) the data, merge it, run the coach-impact and formation analyses |
| `bundesliga coach-effect` | Effect of a coaching change beyond regression to the mean |
| `bundesliga benchmark [--refresh-odds]` | Score the forecasts against betting odds |
| `bundesliga train` / `tune --method grid\|embargoed\|optuna` | Train or tune the outcome classifiers |
| `bundesliga predict ...` | Forecast a match (`--help` for options) |
| `bundesliga site` | Rebuild the results page in `site/` (published on push) |
| `bundesliga reproduce` | Every modelling step, in dependency order |

Requires Python 3.11+. `pip install -r requirements-lock.txt && pip install -e . --no-deps`
reproduces the exact tested environment.

## Engineering

- **81 tests** on synthetic data, covering leak-safety, the time split, the coaching-change
  estimator (a planted effect must be recovered, and zero reported when there is none),
  the scoring rules, the scraper guards and name resolution. CI runs lint and tests on
  Python 3.11 and 3.13, plus a weekly end-to-end run of the real pipeline on a fixture
  dataset.
- **Bugs caught by checking rather than assuming**, each of which produced plausible
  output rather than an error — a formation feature one match stale, class weights
  distorting probabilities, a heatmap centred on the wrong average, results silently lost
  to a race between two tuning runs. Written up in
  [engineering-notes.md](docs/engineering-notes.md#bugs-found-and-fixed).

## Limitations

- **Small samples.** 65 mid-season sackings; 321 test matches. The intervals are wide, and
  the write-up says so wherever it matters.
- **Observational data.** Clubs don't sack at random. The adjustment covers recent points,
  xG and fixture difficulty, not everything a club's board knows.
- **Data freshness depends on the scrapers.** FBref, Understat and Transfermarkt can
  change or block access without notice; that has already happened once
  (Transfermarkt's `.com` domain).

## More

- [docs/results-in-depth.md](docs/results-in-depth.md) — methods and every result, including the
  ones that didn't hold up
- [docs/engineering-notes.md](docs/engineering-notes.md) — data sources, scraping, automation,
  tests, configuration, bugs found

```
cli.py                 the `bundesliga` command
predict.py             match forecasts from current form
config.py              seasons, windows, name/id mappings
src/                   pipeline stages, analyses, models, results-page builder
tests/                 pytest suite
scripts/               weekly refresh (launchd) and the end-to-end fixture run
site/                  the published results page
docs/                  write-ups and the charts in this README
```

MIT licensed.
