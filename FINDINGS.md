# Findings

Notes from building the golf scoring-average predictor, phase by phase. Each script referenced
here can be re-run with `source venv/bin/activate && python <script>.py` to reproduce these numbers.

## Phase 1 — Data (`01_explore_data.py`)

`data/pgaTourData.csv`: 2,312 rows (one per player-season, 2010–2018), 18 columns.

- 634 rows are missing `Rounds`, `Fairway Percentage`, `Avg Distance`, `gir`, `Average Putts`,
  `Average Scrambling`, `Average Score`, and all SG columns **together** — one group of rows
  (players who didn't play enough rounds for the Tour to report full stats), not scattered
  random missingness.
- `Wins` is null for 2,019 of 2,312 rows; null means 0, not unknown.
- `Points` and `Money` load as strings (e.g. `"$2,680,487"`) due to `$`/comma formatting. Unused
  as features, so left uncleaned.

## Phase 2 — Cleaning and feature selection (`02_clean_data.py`)

- **Target:** `Average Score`.
- **Features:** `Avg Distance`, `Fairway Percentage`, `gir`, `Average Putts`,
  `Average Scrambling` — inputs to a golfer's game.
- **Deliberately excluded:** `Wins`, `Points`, `Top 10`, `Money` — these are *outcomes* of scoring
  well, not causes of it; including them would be circular.
- Dropped the 634 rows missing the columns above (`dropna`), rather than imputing — they're
  missing nearly every stat at once, so there's nothing to impute from.
- Result: 1,678 clean rows.

## Phase 3 — Linear Regression baseline (`03_train_baseline.py`)

80/20 train/test split (`random_state=42`, reused in every later phase for a fair comparison).

| Metric | Test set |
|---|---|
| R² | 0.770 |
| RMSE | 0.366 strokes |

Explains ~77% of the variance in scoring average from 5 stats alone; typical prediction is off by
about a third of a stroke.

Raw coefficients aren't directly comparable to rank feature importance — they're confounded by
each feature's scale (e.g. `Average Putts` ranges ~28–31, `Avg Distance` ranges ~275–320), so a
bigger coefficient doesn't necessarily mean a more important feature. Random Forest importances
(Phase 4) don't have this problem.

## Phase 4 — Random Forest + overfitting diagnostic (`04_train_random_forest.py`)

| Model | Train R² | Test R² | Train RMSE | Test RMSE | Overfit gap (train R² − test R²) |
|---|---|---|---|---|---|
| Linear Regression | 0.663 | 0.770 | 0.394 | 0.366 | −0.107 |
| Random Forest (100 trees, no depth limit) | 0.945 | 0.697 | 0.159 | 0.420 | **+0.248** |

**Finding: the Random Forest underperforms the simpler Linear Regression baseline on unseen data,
despite fitting the training data far better.** The 0.248 train/test gap is a textbook overfitting
signature — with no `max_depth` limit, the trees grew deep enough to memorize quirks of the 1,342
training rows that don't hold up on the 336 test rows.

Two reasons this makes sense here, not just noise:
1. **Golf scoring is close to linear/additive by construction.** Professional "Strokes Gained"
   analytics literally defines `SG:Total = SG:OTT + SG:APR + SG:ARG + SG:Putts` — a straight sum.
   A model that assumes straight-line relationships (Linear Regression) is a natural fit for a
   target that really is close to additive; the Random Forest's extra flexibility (modeling
   curves/interactions) buys nothing here and instead gives it room to overfit.
2. **The dataset is small** (1,342 training rows) relative to the forest's unconstrained
   capacity. Linear Regression only has 5 coefficients + an intercept to fit, so it has almost no
   room to overfit — which is also why its train R² (0.663) is *lower* than its test R² (0.770):
   plain train/test split variance, not overfitting, since a 6-parameter model can't memorize
   much.

**Takeaway:** a more complex/flexible model is not automatically better — it should earn its extra
complexity by beating a simpler baseline on the held-out test set. Here it doesn't, so
Linear Regression is the better model *for prediction accuracy*. We still use the Random Forest's
feature importances below, since that diagnostic is useful independent of which model "wins."

**Feature importances (Random Forest), most to least important:**

| Feature | Importance |
|---|---|
| `gir` (greens in regulation) | 0.344 |
| `Average Putts` | 0.237 |
| `Average Scrambling` | 0.233 |
| `Avg Distance` | 0.117 |
| `Fairway Percentage` | 0.069 |

Matches the well-known golf-analytics finding "drive for show, putt/approach for dough": approach
play (`gir`) and short game (putting, scrambling) dominate; driving distance and especially
driving accuracy matter comparatively little.

## Phase 4.5 — Cross-validation robustness check (`05_cross_validation.py`)

Phase 3/4's comparison rested on a single 80/20 split, which looked suspicious — Linear
Regression's single-split test R² (0.770) was *higher* than its train R² (0.663), a sign of
lucky-split variance rather than a real effect. 5-fold cross-validation on the full 1,678 clean
rows (`KFold(n_splits=5, shuffle=True, random_state=42)`) checks this by testing on 5 different
held-out chunks instead of one, so no single lucky/unlucky split can skew the result.

| Model | R² (mean ± std) | RMSE (mean ± std) |
|---|---|---|
| Linear Regression | 0.678 ± 0.056 | 0.391 ± 0.014 strokes |
| Random Forest | 0.619 ± 0.049 | 0.427 ± 0.010 strokes |

Per-fold R² (Linear Regression vs Random Forest, same 5 folds): `[0.770, 0.684, 0.656, 0.595,
0.687]` vs `[0.701, 0.626, 0.585, 0.557, 0.623]` — **Linear Regression wins every one of the 5
folds**, for both R² and RMSE. RMSE's ± bands don't overlap at all (LR: 0.377–0.405 vs RF:
0.417–0.437); R²'s bands overlap only slightly, but the fold-by-fold sweep is stronger evidence
than the overlap alone would suggest, since it's a matched comparison rather than two independent
averages.

This also revealed that the Phase 3/4 single-split numbers were optimistic for both models — LR's
single-split test R² (0.770) turns out to be the *best* of its 5 fold scores, not typical; the
honest average is 0.678.

**Conclusion:** Linear Regression's advantage over Random Forest is real and consistent, not a
fluke of one split. This strengthens (doesn't reverse) Phase 4's conclusion.

## Open question for later phases

Which model should the Streamlit app (Phase 5) actually use for its live prediction — the more accurate Linear Regression, or the Random Forest (whose importances we're using for the chart  anyway)? Worth deciding explicitly before Phase 5 rather than defaulting silently.
