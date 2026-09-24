# Engineering notes

The data pipeline, scraping, automation, tests and the bugs found along the way.
For the modelling write-up see [results-in-depth.md](results-in-depth.md); for the
headline findings, the [README](../README.md).

- [Data sources](#data-sources)
- [Keeping team names in sync across four sources](#keeping-team-names-in-sync-across-four-sources)
- [Guarding against a bad scrape](#guarding-against-a-bad-scrape)
- [Weekly refresh](#weekly-refresh)
- [Tests and CI](#tests-and-ci)
- [Configuration](#configuration)
- [Bugs found and fixed](#bugs-found-and-fixed)
- [Setup details](#setup-details)

## Data sources

| Data | Source | How |
|---|---|---|
| Results, formations | [FBref](https://fbref.com) | `soccerdata.FBref` (headless Chrome via seleniumbase) |
| xG, PPDA (pressing), deep completions | [Understat](https://understat.com) | `soccerdata.Understat` |
| Coach tenure dates | Transfermarkt "Trainerhistorie" pages | `src/fetch_coach_history.py` (custom scraper) + manual CSV fallback |
| Betting odds | [football-data.co.uk](https://www.football-data.co.uk) | `soccerdata.MatchHistory` |

`soccerdata` covers no manager-history source, so coach tenures are the one piece
scraped by hand. English Wikipedia was tried first, but Bundesliga clubs mostly don't
have a "List of `<club>` managers" page (every guessed URL 404'd). Transfermarkt's
manager-history table turned out to be the reliable source: a plain `requests` call
with a browser User-Agent gets a normal 200, and the URL only depends on the club's
numeric id. `data/coach_history_manual.csv` overrides or adds to whatever the scraper
produces.

**A source can change under you.** A few days after the Transfermarkt scraper was
working, `transfermarkt.com` began answering every request with an AWS WAF challenge
(HTTP 202, empty body) — even the homepage, so no header or delay would fix it.
`transfermarkt.de` still serves the identical page, so the scraper now uses `.de`. The
first time this happened, the scraper silently wrote a 1-row file over ~1,600 real
rows, which removed the coach from 98% of matches without a single crash anywhere in
the chain. That incident is why every fetcher now has an overwrite guard (below).

**Odds: Pinnacle stops in January 2026.** football-data.co.uk's Pinnacle columns end
mid-January 2026, leaving them on only about half of the test set. The market
benchmark therefore uses the market-average odds, which are complete in every season,
and reports Pinnacle as a sensitivity check on the matches that have it.

## Keeping team names in sync across four sources

FBref, Understat, Transfermarkt and football-data.co.uk each spell clubs differently
("Borussia M.Gladbach", "M'gladbach", "Gladbach"; "FC Cologne", "FC Koln", "Köln").
FBref's spelling is canonical everywhere in the project.

New clubs used to need a hand-added entry every time `SEASONS` widened or a club got
promoted — that happened three times before it was automated. Now a club missing from
the config is resolved live:

- **Transfermarkt ids** — `src/transfermarkt_search.py` searches Transfermarkt and
  takes the first non-reserve, non-youth club link. An auto-resolved id is verified
  strictly against the fetched page's `<title>`: a mismatch rejects the guess rather
  than trusting it. Validated against all 12 ids already known correct.
- **Understat names** — `src/team_name_matcher.py` fuzzy-matches against FBref's team
  list, refusing to guess when the best match is weak or too close to the runner-up.
  Validated 28/28 against the known pairs, including "FC Cologne" → "Köln". One
  detail found by checking rather than assuming: `difflib.get_close_matches` does not
  reliably return results in order of similarity (it ranked "Wolfsburg" above "Köln"
  for "FC Cologne"), so the matcher compares `.ratio()` scores directly.
- Successful resolutions are cached in `data/*_auto.json` (gitignored), so each club
  costs one lookup, ever. Anything the resolver isn't confident about falls back to the
  manual config.
- **football-data.co.uk** names live in `config.FOOTBALL_DATA_NAME_MAP`, checked
  against all 8 seasons. The fuzzy matcher got 27/28 on its own and correctly refused
  to guess "Bielefeld" (FBref: "Arminia", no shared substring). The benchmark also
  refuses to score if the two sources disagree on any match result, which would mean
  a bad join.

## Guarding against a bad scrape

`src/data_guard.py` refuses to overwrite a saved raw file with one that is drastically
smaller (more than a 50% drop, for files above 20 rows), raising a `RuntimeError`
instead. It is used by the FBref, Understat, Transfermarkt and odds fetchers. FBref has
a second, more targeted check: at least 80% of teams must fetch successfully, because
FBref lists a whole season's fixtures whether played or not, so the row count barely
moves when a few teams fail.

What is still unguarded: values that are wrong without the file shrinking (a source
quietly corrupting numbers). The fetchers are where bad external data enters, so
guarding there covers the likely failures, but it isn't a guarantee.

## Weekly refresh

`scripts/weekly_refresh.sh` runs `bundesliga pipeline` every Monday at 06:00 via a
macOS LaunchAgent. It uses `launchd` rather than `cron` deliberately: `cron` skips a
job if the Mac is asleep, while `launchd` runs a missed job on the next wake or login.

The first two scheduled runs both failed, and neither said so:

- **Sep 13:** hung for 29 hours inside the FBref/Selenium fetch, with nothing to stop it.
- **Sep 21:** launchd fired the moment the Mac woke, before Wi-Fi was back, and every
  request died on a DNS error.

The script now waits up to 15 minutes for the data sources to be reachable, runs each
attempt under a watchdog that kills the whole process group (Python and the Chrome it
spawned) after 45 minutes, retries up to three times, and raises a macOS notification
on failure. It writes `logs/last_refresh_status` either way and keeps the last ~12
logs. The watchdog is plain bash, since macOS has no `timeout` command. It was tested
against a fake pipeline that hangs while holding a child process, one that fails once
and then succeeds, and an unreachable network.

Install (edit `Weekday`/`Hour`/`Minute` in the plist first for a different time):

```bash
cp scripts/com.felixnitschke.bundesliga-coach-impact.weeklyrefresh.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.felixnitschke.bundesliga-coach-impact.weeklyrefresh.plist
```

Check status: `launchctl print gui/$(id -u)/com.felixnitschke.bundesliga-coach-impact.weeklyrefresh`.
Uninstall: `launchctl bootout gui/$(id -u)/com.felixnitschke.bundesliga-coach-impact.weeklyrefresh`.

Limitations: a Mac that stays off still misses the run until it's next on; and the
refresh updates the data and analyses, but not the trained models or the published
page. Those are rebuilt with `bundesliga reproduce` and a push.

## Tests and CI

`python -m pytest` runs the suite: small synthetic data only, no network, independent
of whatever is in `data/`. What it checks is the stuff a wrong answer wouldn't
visibly *look* wrong for:

- **Leak-safety** (`test_features.py`) — every rolling feature uses strictly earlier
  matches only, checked against hand-computed values. To confirm the tests have teeth,
  a `.shift(1)` was removed during development and the right test failed.
- **Time-based splitting** (`test_outcome_predictor.py`) — no test match ever precedes
  a training match, and a misconfigured `TEST_SEASONS` fails loudly instead of
  evaluating on nothing.
- **The coaching-change estimator** (`test_coach_change_effect.py`) — a known effect
  planted in synthetic data is recovered, and zero is reported when the raw "bounce"
  is pure regression to the mean.
- **The market benchmark** (`test_market_benchmark.py`) — RPS against hand-computed
  values, margin removal, and the refusal to score a mismatched join.
- **Overwrite guards and auto-resolution** (`test_data_guard.py`,
  `test_fetch_coach_history.py`, `test_team_name_matcher.py`) — including a direct
  regression test for the Transfermarkt blocking incident.
- **`predict.py`** (`test_predict.py`) and the coach-bounce join (`test_coach_bounce.py`).
- **The results page** (`test_site.py`) — scraped names embedded in the page can't
  break out of its `<script>` block (checked by removing the escaping and watching the
  test fail).

GitHub Actions:

- `ci.yml` — ruff + pytest on Python 3.11 and 3.13 for every push and PR, plus a macOS
  job that installs `requirements-lock.txt` exactly (macOS because the lockfile
  contains macOS-only packages from seleniumbase, and macOS runners need
  `brew install libomp` for XGBoost).
- `e2e-fixture.yml` — weekly and on demand: `scripts/e2e_fixture_pipeline.py` runs the
  real `build_dataset` → `coach_impact` → `coach_change_effect` → `formation_matrix` →
  `features` → `outcome_predictor` chain on a small fixture dataset whose schema is
  copied from the real scraped files. It catches the one thing the unit tests can't: a
  schema change from a new `soccerdata` release breaking the stages' hand-offs.
- `pages.yml` — publishes `site/` to GitHub Pages when a rebuilt page is pushed.

## Configuration

Everything tunable is in `config.py`:

- `SEASONS` — how far back to pull (default 2019-20 through the ongoing 2026-27).
  Formation and PPDA coverage is reliable from about 2014-15.
- `IMPACT_WINDOW` — matches before and after a coaching change (default 8).
- `FEATURE_ROLLING_WINDOW`, `TEST_SEASONS`, `MODEL_DIR` — rolling-form window, held-out
  seasons, model location.
- `N_CV_SPLITS`, `CV_EMBARGO_GAP` — cross-validation folds and embargo for tuning (see
  [results-in-depth.md](results-in-depth.md#hyperparameter-tuning)).
- `CLUB_TRANSFERMARKT_ID`, `TEAM_NAME_MAP`, `FOOTBALL_DATA_NAME_MAP` — verified name/id
  mappings; anything missing is auto-resolved where possible.

## Bugs found and fixed

Kept here because each one produced plausible-looking output rather than an error.

- **A stale "recent formation."** `predict.py` computed a team's current formation
  tendency with the same function training uses, which (correctly, for training)
  excludes the latest match. For "right now" that made it one match out of date: 5 of
  28 teams showed the wrong formation, including a team that had just changed. Found
  by writing the function's first tests.
- **Class balancing distorted the probabilities.** Training with balanced classes lifted
  macro-F1 by getting draws predicted, but inflated draw probabilities: 33% on average
  across the test set, against a real draw rate of 24%. Found by the betting-market
  benchmark;
  `bundesliga predict` now defaults to a model trained for probabilities.
- **A misleading hero chart.** The formation heatmap was coloured on a red-yellow-green
  scale centred at 1.0 points per game. The league average is ~1.38, so most cells
  looked "good" by construction. It's now a two-colour scale centred on the real
  average.
- **Lost results from a race.** Two tuning runs executed in parallel both read, updated
  and rewrote `outputs/outcome_model_metrics.json`; the one that finished last silently
  dropped the other's results. `bundesliga reproduce` runs every step sequentially.
- **An incumbent coach's tenure capped at scrape time.** An early version mapped a blank
  "still in charge" end date to *today*, so every future fixture ended up with no coach
  once `SEASONS` included the ongoing season. Now left open-ended.
- **pandas 3 + pyarrow.** Comparing a pyarrow-backed column's `.values` to a numpy array
  raises `AttributeError: 'ArrowExtensionArray' object has no attribute 'mean'`. Use
  `.to_numpy()`.

## Setup details

- **Python 3.11+.** scikit-learn 1.9 and pandas 3 need it; CI tests 3.11 and 3.13.
- **macOS: `brew install libomp`.** XGBoost needs the OpenMP runtime, which isn't
  installed by default. Without it the failure (`Library not loaded: @rpath/libomp.dylib`)
  only appears the first time something imports `xgboost`.
- **Exact versions:** `pip install -r requirements-lock.txt && pip install -e . --no-deps`
  reproduces a known-working environment (generated on Python 3.13).
- **FBref needs Chrome.** `soccerdata` drives a headless browser; the first run
  downloads a matching chromedriver. A full 8-season pull takes a few minutes, and
  responses are cached under `~/soccerdata/data/`.
