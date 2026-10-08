# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A golf scoring-average predictor: given a PGA Tour player's performance stats (driving distance,
accuracy, greens in regulation, putting, scrambling), predict their `Average Score` for the season
and show which stats drive that prediction most.

This is a learning project for a first-year software engineering student building their first ML
project. Prioritize teaching over shipping: explain what each step does and why in plain language,
keep code simple and heavily commented over clever or abstracted, and pause after each phase below
for the user to review before continuing to the next.

## ⚠️ Open decisions — raise these FIRST in every new conversation

**At the start of a new conversation, before any other work, show the user this list (briefly) and
ask which they want to resolve now.** When one is resolved: apply it, record the outcome in
FINDINGS.md, and delete it from this list. Don't resolve any of these yourself. Each needs either
the user's choice or an official source they provide. Also run `git status` and mention any
uncommitted work.

**Needs checking against official sources** (never fill in from memory):
1. WHS differential formula `113/slope × (adjusted − rating − PCC)` and the Rule 5.2a table
   (`WHS_TABLE` in `handicap.py`): confirm against the official WHS text.
2. Course handicap's `(rating − par)` term: does the user's national association use it, and since
   when? (`course_handicap()` in `handicap.py`)
3. Plus-handicap rounding: `round_half_up()` rounds −0.5 to −1 (away from zero). Confirm the
   association's rule.
4. Benchmark values to check against the Shot Scope eBook (`data/handicap_benchmarks.csv`):
   fairways % rises to 49 at 10 hcp, putts tie at 15/20 hcp (33.1), and GIR drops 17 points from
   0 to 5 hcp.
5. Benchmark stat definitions in the eBook: scrambling denominator (missed greens?), whether putts
   from the fringe count, and how fairways % is counted. These decide whether the Tour comparison
   in Phase 7 is fair.
6. Is the benchmark "handicap" a Handicap Index or a course handicap? (Currently assumed: index.)

**Needs the user's choice:**
7. `GIR_CONFOUND_RATIO = 0.7` (`strokes_config.py`): as a ratio it triggers with ~3.3 greens of
   margin at scratch but only ~0.5 at 25 hcp. Keep it, or switch to an absolute gap?
8. Blank PCC is saved as 0.0, so "unknown" and "zero" can't be told apart later. OK?
9. Import is all-or-nothing and skips rows that exactly match an existing round. OK?
10. `FAIRWAY_MISS_STROKES = 0.0`: revisit once enough of the user's own rounds exist?
11. 3-putts are logged but not scored, because there's no benchmark. Find a source, or leave it?
12. Overshoot uses a straight-line fit and "typical round" is labelled ±2 strokes. Accept, or
    try something else?
13. Phase 10 is not defined yet. The earlier idea was a personal model learned from the user's own
    rounds. Agree on its scope and the minimum number of rounds before starting.
14. Older open questions at the bottom of FINDINGS.md: tune the Random Forest, try a season-based
    split, add a `StandardScaler`, and fix the `web/predictor.js` coefficient drift.

## Tech stack

- Python, in a venv at `./venv` (`source venv/bin/activate`)
- pandas, scikit-learn, Streamlit — no other ML/data libraries
- No test framework, linter, or build step is set up (none needed at this project's current size)

## Commands

```
source venv/bin/activate      # activate the venv (do this before running anything)
python 01_explore_data.py     # run a phase script
streamlit run app.py          # run the PGA predictor app (Phase 5)
streamlit run 08_round_tracker.py   # run the personal round tracker (Phase 8)
```

## Dataset

`data/pgaTourData.csv` — PGA Tour player-season stats, 2010–2018, 2312 rows, 18 columns. One row
per player per season.

Exact column names (note `gir` is lowercase and the SG columns use a `SG:` prefix — no `%` in any
name despite several being percentages):

```
Player Name, Rounds, Fairway Percentage, Year, Avg Distance, gir, Average Putts,
Average Scrambling, Average Score, Points, Wins, Top 10, Average SG Putts,
Average SG Total, SG:OTT, SG:APR, SG:ARG, Money
```

`Average Score` is the prediction target. The core feature candidates are `Avg Distance`,
`Fairway Percentage`, `gir`, `Average Putts`, `Average Scrambling`.

Known data quirks:
- 634 rows are missing `Rounds`, `Fairway Percentage`, `Avg Distance`, `gir`, `Average Putts`,
  `Average Scrambling`, `Average Score`, and the SG columns together — this is one group of rows
  (players who didn't play enough rounds for the Tour to report full stats), not scattered random
  missingness.
- `Wins` is null for 2019 of 2312 rows; null here means 0 wins, not unknown, since the source data
  omits the field rather than writing 0.
- `Points` and `Money` load as strings (`Money` looks like `"$2,680,487"`) because of `$` and comma
  formatting. Neither is used as a model feature, so they don't need cleaning for this project.
- The five SG columns leak the target and must not be used as features — Phase 6 established that
  `Average Score = 71.07 - Average SG Total` to within ±0.19 strokes, and the four components sum
  to that total. See FINDINGS.md Phase 6 before reaching for them again.

`data/handicap_benchmarks.csv` — amateur average stats by handicap (0, 5, 10, 15, 20, 25), from
the Shot Scope Strokes Gained eBook 4th ed (2021). It holds score *to par* (not raw score) and has
no three-putt data. Never fill in or "fix" its numbers from memory. They come from the source only.

`data/my_rounds.csv` (Phase 8+, gitignored) — the user's own logged rounds. This is personal data,
so never commit it, and never create test rounds in it: run checks on a scratch copy.
`data/my_rounds.example.csv` is the header-only template. Round fields are listed in
`ROUND_COLUMNS` in `rounds.py`. `length_yards`, `adjusted_score` (blank = use gross score) and
`pcc` (blank = 0, range −1.0…+3.0) are optional, and older CSVs without them must still load.

WHS constants and formulas come only from official sources the user provides. Never recall or
substitute them from memory.

## Amateur feature: design principle

From Phase 7 on, the **handicap benchmarks and the user's own rounds are the main engine**. The
PGA data and its models are a secondary "tour reference" only. The tour model must never produce
an amateur score prediction, and must never be used on inputs outside its training range (see
`inside_tour_range()` in `benchmarks.py`). Almost every amateur falls outside that range.

## Conventions

- Phase scripts are numbered at the project root (`01_explore_data.py`, `02_...`, etc.) so the
  build-up is visible in the file listing itself — don't reorganize into a package/src layout.
- Shared helpers are unnumbered root files: `benchmarks.py` (load/validate the benchmark CSV,
  `expected_stat()` interpolation with clamping, tour range guard), `rounds.py` (round schema,
  validation), `storage.py` (load/save/export/import), `handicap.py`, `strokes_lost.py`
  (per-area breakdown, putting confound, ranking), `strokes_config.py` (named constants).
- `handicap.py` behaviours: everything is kept unrounded and rounded half-up (`round_half_up`,
  not Python's `round`) only for display or where WHS says to. There are two expected scores:
  `plays_to_handicap` (headline) and `typical_round` (soft, ±2, with a fitted overshoot clamped
  to 0–25). The index uses the WHS Rule 5.2a dict on the newest 20 rounds, needs at least 3
  rounds, and has no 0.96 multiplier. `python handicap.py` runs hand-checked asserts, which must
  keep passing.
- The handicap index is ALWAYS labelled "ESTIMATED, not official", along with what isn't modelled.
- Expected scores are never fed into the weakness engine, which compares stats with the
  benchmark at the player's index. Confounded putting is never ranked above penalties or short
  game.
- Strokes-lost constants live only in `strokes_config.py`. Ask the user before changing any of
  them. GIR/fairway/putt benchmarks are never scaled by slope or length.
- Don't touch `web/` as part of the amateur feature.
- Comment code heavily; explain the "why" of each pandas/sklearn call, not just what it does.

## Status

- [x] Phase 1: Set up venv, install packages, load CSV, show summary
- [x] Phase 2: Clean data, select features and target
- [x] Phase 3: Train baseline Linear Regression, report R² and RMSE
- [x] Phase 4: Train Random Forest, compare to baseline, print feature importances
- [x] Phase 4.5: Cross-validate both models on the full dataset to confirm the comparison is robust
- [x] Phase 5: Build Streamlit app (sliders per stat, live prediction, feature-importance chart)
- [x] Phase 6: Test the Strokes Gained columns as features — rejected, they leak the target
- [x] Phase 7: Amateur handicap benchmarks (Shot Scope), interpolation helper, tour reference
      column, 4-feature tour model with range guard
- [x] Phase 8: Course-aware round tracker + strokes-lost weakness report
      (`streamlit run 08_round_tracker.py`)
- [x] Phase 9: WHS-based maths (two expected scores, Rule 5.2a index, adjusted score/PCC),
      putting/GIR confound flag, storage.py + Export/Import (`09_course_aware_checks.py`)

See `FINDINGS.md` for the detailed results and interpretation from each phase, including a
noteworthy result: the Random Forest overfits (train R² 0.945 vs test R² 0.697) and is actually
beaten by the Linear Regression baseline (test R² 0.770) on this dataset. Phase 4.5's 5-fold
cross-validation (R² 0.678 ± 0.056 for Linear Regression vs 0.619 ± 0.049 for Random Forest,
Linear Regression winning all 5 folds) confirms this isn't a fluke of one split. The Streamlit app
(`app.py`) uses Linear Regression for the live prediction (more accurate) and the Random Forest's
feature importances for the "what matters most" chart (not confounded by feature scale). Phase 6
tested the Strokes Gained columns as extra features and rejected them: they push R² to 0.927, but
that's data leakage, not skill — see the Dataset note above.
