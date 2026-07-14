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

## Tech stack

- Python, in a venv at `./venv` (`source venv/bin/activate`)
- pandas, scikit-learn, Streamlit — no other ML/data libraries
- No test framework, linter, or build step is set up (none needed at this project's current size)

## Commands

```
source venv/bin/activate      # activate the venv (do this before running anything)
python 01_explore_data.py     # run a phase script
streamlit run app.py          # run the Streamlit app (once it exists, see Phase 5)
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

## Conventions

- Phase scripts are numbered at the project root (`01_explore_data.py`, `02_...`, etc.) so the
  build-up is visible in the file listing itself — don't reorganize into a package/src layout.
- Comment code heavily; explain the "why" of each pandas/sklearn call, not just what it does.

## Status

- [x] Phase 1: Set up venv, install packages, load CSV, show summary
- [x] Phase 2: Clean data, select features and target
- [x] Phase 3: Train baseline Linear Regression, report R² and RMSE
- [x] Phase 4: Train Random Forest, compare to baseline, print feature importances
- [x] Phase 4.5: Cross-validate both models on the full dataset to confirm the comparison is robust
- [x] Phase 5: Build Streamlit app (sliders per stat, live prediction, feature-importance chart)

See `FINDINGS.md` for the detailed results and interpretation from each phase, including a
noteworthy result: the Random Forest overfits (train R² 0.945 vs test R² 0.697) and is actually
beaten by the Linear Regression baseline (test R² 0.770) on this dataset. Phase 4.5's 5-fold
cross-validation (R² 0.678 ± 0.056 for Linear Regression vs 0.619 ± 0.049 for Random Forest,
Linear Regression winning all 5 folds) confirms this isn't a fluke of one split. The Streamlit app
(`app.py`) uses Linear Regression for the live prediction (more accurate) and the Random Forest's
feature importances for the "what matters most" chart (not confounded by feature scale).
