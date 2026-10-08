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

## Phase 6 — Strokes Gained features, rejected as leakage (`06_strokes_gained.py`)

The CSV has five unused Strokes Gained columns (`SG:OTT`, `SG:APR`, `SG:ARG`, `Average SG Putts`,
`Average SG Total`). They're missing on exactly the same 634 rows as everything else, so adding
them costs no extra data. Adding the four components to the core 5 looks like a triumph:

| Feature set | R² | RMSE |
|---|---|---|
| Core 5 (Phase 3–5 model) | 0.678 | 0.391 |
| Core 5 + 4 SG components | **0.927** | **0.187** |

**The improvement is entirely fake.** Three tests, all 5-fold CV on the same 1,678 rows:

| Test | Feature set | R² |
|---|---|---|
| 1 | `Average SG Total` **alone** — zero golf stats | 0.924 |
| 2 | The 4 SG components alone | 0.924 |
| 3 | Core + components + `SG:Total` | 0.926 |

Test 1 is decisive: one column that knows nothing about distance, accuracy, greens, putting or
scrambling matches the full 9-feature model. Test 2 shows the components aren't a safer subset —
they sum to the total (mean gap 0.0043), so splitting a leaky number four ways doesn't stop it
leaking. Test 3 is a side-lesson in collinearity: adding `SG:Total` to the four components changes
nothing, because a linear model is a weighted sum and could already construct that total itself.

**The mechanism.** Strokes Gained is *defined* as score relative to the field, so
`SG:Total = field average − player's score`. Adding `Average Score` and `SG:Total` back together
should therefore recover the field average — and it does, identically in every season:

| Year | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|---|---|---|---|
| mean | 71.092 | 71.015 | 71.034 | 71.053 | 71.034 | 71.099 | 71.154 | 71.084 | 71.064 |
| std | 0.161 | 0.204 | 0.196 | 0.166 | 0.180 | 0.192 | 0.167 | 0.193 | 0.219 |

Overall 71.070 ± 0.191. The entire "model" collapses to one line of arithmetic:

```
Average Score = 71.07 - SG:Total
```

The clincher: the spread of that constant (0.191) matches the SG model's measured RMSE (0.187).
The model's whole remaining error *is* the year-to-year drift in the field average — it learned the
identity and nothing else.

**Decision: all five SG columns rejected.** Not because 0.927 is a bad score, but because it isn't
a prediction. Computing SG for a season requires already knowing that season's scores. Forecasting
2019 before 2019 is played, SG doesn't exist yet, so a model depending on it can never be run. This
is the difference between *explaining* a finished season (SG is excellent) and *predicting* an
unplayed one (SG is useless). R² 0.678 that survives contact with reality beats R² 0.927 that can
never be computed when you need it.

Phase 3–5's model stands unchanged and the Streamlit app keeps its 5 sliders. The finding is a
negative one: there's no free lunch in this CSV — the only columns that would raise our score are
derived from the answer, which is why they were left out in Phase 2 in the first place.

## Phase 7 — Amateur handicap benchmarks + tour reference (`07_handicap_benchmarks.py`, `benchmarks.py`)

**Change of direction.** The goal is now helping an *amateur* find which part of their game costs
the most strokes. The main reference is `data/handicap_benchmarks.csv` (Shot Scope Strokes Gained
eBook, 4th ed, 2021). The PGA data is now a secondary "tour reference" and is never used to
predict an amateur's score.

**The benchmark table** (blank = not in source; Tour = mean of the 1,678 clean PGA rows):

| Stat | 0 | 5 | 10 | 15 | 20 | 25 | Tour |
|---|---|---|---|---|---|---|---|
| Score to par | 0.83 | 6.33 | 10.88 | 17.38 | 21.69 | 28.97 | — |
| Fairways % | 50 | 48 | 49 | 48 | 46 | 46 | 61.44 |
| GIR % | 61 | 44 | 36 | 24 | 17 | 10 | 65.66 |
| Putts / round | 29.4 | 30.2 | 31.2 | 33.1 | 33.1 | 33.8 | 29.16 |
| Scrambling % | 47 | 41 | 31 | 21 | 20 | 18 | 58.12 |
| Penalties / round | 0.56 | 0.91 | 1.62 | 2.45 | 3.03 | 4.67 | — |
| 3-putts / round | — | — | — | — | — | — | — |

Schema notes: the source reports **score to par** (not raw score), so the column is
`score_to_par` instead of the planned `avg_score`. Converting would mean assuming a par of 72.
The source also gives single handicaps, not bands, so `hcp_low = hcp_high = hcp_mid`.

**Trend check** (reported, not "fixed"):
- Score to par, GIR, scrambling and penalties all move strictly in the expected direction.
- **Fairways % is not monotonic** (50, 48, **49**, 48, 46, 46). It's nearly flat across the whole
  range: a 25-handicapper hits only 4 points fewer fairways than a scratch golfer. That's
  plausibly real (amateur accuracy doesn't vary much with skill) and matches Phase 4's finding
  that fairway % was the least important stat. It does mean fairways % alone can't tell
  handicaps apart.
- **Putts / round ties at 15 and 20** (33.1, 33.1).
- **GIR drops 17 points from 0 to 5 hcp**, the biggest step in the table. Worth checking against
  the source.
- Putts per round **rise** with handicap partly *because* GIR falls. A player who misses the green
  and chips close takes fewer putts on that hole, so raw putts mix putting skill with approach
  play. This is a known weakness of putts per round as a stat.

**Interpolation.** `expected_stat(benchmarks, stat, handicap)` draws straight lines between the
six anchors (e.g. GIR at 12 hcp = 31.2%) and **clamps** outside 0–25. Extrapolating would be
nonsense: extending the 20→25 GIR slope out to a 40 handicap gives negative GIR.

**4-feature tour model** (no `Avg Distance`, since amateurs rarely know their real distance).
Same 5-fold CV setup as Phase 4.5:

| Model | R² (mean ± std) | RMSE (mean ± std) |
|---|---|---|
| 5 features (Phase 4.5) | 0.678 ± 0.056 | 0.391 ± 0.014 |
| 4 features, no distance | 0.635 ± 0.062 | 0.416 ± 0.015 |

Dropping distance costs about 0.04 R² and 0.025 strokes of RMSE. Distance does carry some
information that the other four stats don't, but most of the model survives without it.

**Range guard.** The tour model trained on GIR 53.5–73.5%, putts 27.5–31.0, scrambling
44.0–69.3% and fairways 43.0–76.9%. `inside_tour_range()` refuses any input outside those ranges.
Run against the benchmark table, **only the 0-handicap row is inside**. Every other handicap has
2–3 of the 4 features out of range, so for real amateurs the tour model will almost always (and
correctly) refuse to answer. This confirms the PGA model can't serve as an amateur predictor.

**Weakest assumptions, to keep in mind for Phase 8+:**
- **Benchmarks are ballpark.** They come from one source and one year, with no spread given. A
  single average per handicap hides how much golfers at the same handicap differ. "You're 5% below
  your band's GIR" may well be within normal variation.
- **Tracker-user bias.** Shot Scope's data comes from golfers who buy and wear a shot-tracking
  device. They're likely more engaged (and maybe more consistent) than the average golfer at the
  same handicap.
- **Definition mismatches with the tour data.** Shot Scope doesn't state most of its definitions,
  so these are unverified. Scrambling needs the same denominator (missed greens) on both sides.
  Fairways % may handle par 3s or fairway edges differently. (Putts were later confirmed to match:
  the eBook counts only strokes on the green, the same as the Tour. See Phase 9 corrections.) The 0-hcp fairway figure (50%)
  vs the Tour's 61% is a bigger gap than GIR's (61% vs 66%), which hints at a definition gap and
  not only a skill gap.
- **Handicap index vs course handicap.** The table is keyed by "handicap", but it's not stated
  whether that's the WHS Handicap Index or the course handicap for the day. On a hard course these
  can differ by several strokes. Phase 8 should ask for the Handicap Index and say so.
- **Score to par isn't course-adjusted.** A 0-hcp averages +0.83 because handicaps are measured
  against course rating, not par. Comparisons across courses of different difficulty will be
  noisy.

## Phase 8 — Course-aware round tracker + weakness report (`08_round_tracker.py`)

Run with `streamlit run 08_round_tracker.py`. It's a separate app from `app.py`. Rounds are saved
to `data/my_rounds.csv` (gitignored), with an Export/Import CSV option in the app and a
header-only `data/my_rounds.example.csv`. 18-hole rounds only.

**Shared helpers:**
- `rounds.py`: schema and validation.
- `storage.py`: load, save, export and import. Import is all-or-nothing, skips duplicates, and
  loads older CSVs that are missing newer columns.
- `handicap.py`: course maths.
- `strokes_lost.py`: per-area breakdown and ranking.
- `strokes_config.py`: every conversion constant, named and labelled by confidence.

### Phase 9 refinements (`09_course_aware_checks.py`, `python handicap.py`)

Phase 9 tightened the maths using official WHS formulas. `09_course_aware_checks.py` prints the
numbers behind each change, and `python handicap.py` runs the hand-checked asserts.

**Overshoot noise.** Raw overshoot (`score_to_par − handicap`) at 0/5/10/15/20/25 hcp is 0.83,
1.33, **0.88**, 2.38, **1.69**, 3.97. It goes up and down, which is sampling noise and not golf.
A straight line (`np.polyfit`, degree 1) gives `overshoot = 0.1045 × handicap + 0.541`: 0.54,
1.06, 1.59, 2.11, 2.63, 3.15. The largest correction is −0.94 at 20 hcp. The line is clamped to
0–25.

**Why two expected scores.**
- **Plays to handicap** = rating + index × slope/113 is the headline number. It comes straight
  from WHS definitions.
- **Typical round** = that plus the fitted overshoot, roughly ±2 strokes. It answers "what does an
  average day look like?", since an index is built from your *best* rounds. It's soft for two
  reasons. The benchmark overshoot was measured on real courses, which probably average a slope
  above 113, so it already contains some of the slope effect and adding it on top double counts.
  And the benchmark courses' ratings are unknown, so score to par is treated as score to rating.
- On an example course (71.2/128) the two differ by 0.5 strokes at scratch and 3.2 at 25 hcp.
- Neither expected score is fed into the weakness engine, which still compares stats with the
  benchmark at the player's index.

**Course handicap** = index × slope/113 + (rating − par) (USGA WHS FAQ, Rules 6.1/6.2). It's kept
unrounded and rounded half-up only for display.

**Differential** = 113/slope × (adjusted score − rating − PCC). Adjusted score and PCC are new
optional fields. Without an adjusted score the gross score is used. PCC defaults to 0 and is
validated to −1.0…+3.0.

**Estimated index.** Uses the WHS Rule 5.2a table (stored as a plain dict, one line per count, in
`handicap.py`) on the newest 20 rounds. Each differential is rounded to one decimal, sorted, the
lowest N are averaged, the adjustment is added, and the result is rounded to one decimal. There's
no 0.96 multiplier (that was the pre-WHS system). Under 3 rounds the app shows "need 3 rounds".
The index is always labelled **"ESTIMATED, not official"**. Not modelled:
- the net double bogey cap when no adjusted score is entered, so the estimate runs high after a
  blow-up hole;
- exceptional score reductions;
- soft and hard caps.

**Rounding.** Python's `round()` uses banker's rounding (`round(18.5) == 18`), and floats store
12.55 as 12.5499…, so `round(12.55, 1)` gives 12.5. `round_half_up()` (using the `Decimal`
module) rounds halves up as golf expects. One assert checks a 12.55 → 12.6 index.

**Putts/GIR confound.** Golfers who miss many greens chip close and take fewer putts, so putting
can look better than it is. Putting is in the low-confidence tier. It's flagged **"confounded by
GIR"** when the player's *average* GIR over at least 3 rounds is below `GIR_CONFOUND_RATIO`
(0.7, a judgement call, not calibrated) × the benchmark GIR. When flagged, it's never ranked
above penalties or short game. Trigger points per 18 holes: 7.69 greens at 0 hcp, 5.54 at 5,
4.54 at 10, 3.02 at 15, 2.14 at 20, 1.26 at 25. Because it's a ratio, the margin shrinks from
~3.3 greens at scratch to ~0.5 at 25 hcp. 3-putts are logged but not scored, because the
benchmark file has no 3-putt column.

**Strokes-lost conversions** (positive = lost):

| Area | Conversion | Confidence |
|---|---|---|
| Penalties | (yours − typical) × 1.0 | high |
| Short game | (typical scramble rate × *your* attempts − your saves) × 1.0 | medium |
| Approach | (typical GIR − yours) × (1 − typical scramble rate) | low |
| Putting | (yours − typical putts) × 1.0; flagged when confounded | low |
| Driving | gap shown, cost 0 | low, off on purpose |

Per-stat gaps are labelled "not course-adjusted". GIR, fairway and putt benchmarks are never
scaled by slope or length.

**Hand-verified:**
- course handicap 15 / 130 / 73.5 / par 72 = 18.76 (19);
- differential 90 / 72.0 / 130 / PCC 0 = 15.6;
- index from [15.3, 15.2, 16.6] = 13.2;
- index from [12…17] = 11.5;
- 4-round index from [15.6, 16.5, 16.5, 22.6] = 14.6;
- differential with adjusted score 92 and PCC 1.0 on 72.0/130 = 16.52.

All tests (old-format CSV, save, import with bad/good/duplicate rows, form submit, flagged
putting ranking, empty state) ran headless on a scratch copy. The real `data/my_rounds.csv` was
never created.

**Phase 9 corrections, checked against official sources on 2026-10-08:**
- **Confirmed:** the Rule 5.2a table (USGA Rule 5.2a page, including its 15.3/15.2/16.6 → 13.2
  example), the differential formula and its rounding to the nearest tenth with .5 up (USGA FAQ
  "What is a Score Differential"), and the course handicap formula (USGA FAQ "Calculate Course
  Handicap").
- **Bug fixed:** `round_half_up` rounded plus handicaps (stored as negatives) *away* from zero,
  −0.5 → −1. USGA Appendix C's example rounds +0.5 to 0 and +1.5 to +1, i.e. toward zero.
  Negatives now use `ROUND_HALF_DOWN`, with asserts for −0.5, −1.5 and −2.25.
- **Added:** the WHS maximum index of 54.0 (Rule 5.2a), as a cap with an assert.
- **Tightened:** PCC can only be −1, 0, +1, +2 or +3 (USGA FAQ). Validation and the form now
  reject decimals.
- **New asserts** from the USGA Handicap Manual example: 95 on 71.5/125 → 21.2, and 69 → −2.3.
- **Shot Scope eBook** (shotscope.com/ebook/Strokes_Gained.pdf), checked:
  - Every benchmark value matches the CSV, including the odd fairway 49 at 10 hcp and the putts
    tie at 15/20. They are not typos.
  - The book's "Difference v Handicap" row equals our raw overshoot.
  - Putts count only on the green (a fringe stroke isn't a putt), the same as the Tour. The form
    now says so.
  - Not defined in the book: the up & down denominator, fairway rules for par 3s, and whether
    "handicap" means a WHS index. The data predates WHS.

**Phase 9 decisions (2026-10-08)**, settled by Claude after the user asked it to go ahead:
- **#7 Putting confound rule: kept the 0.7 ratio.** At high handicaps the rule is sensitive. At
  25 hcp it flags an average below 1.26 greens against a benchmark of 1.8, a gap of only ~0.5
  greens, and that small a gap barely changes putt counts. But a false flag only *re-orders*
  putting below penalties and short game and adds a label; no strokes change. High handicappers
  usually lose more to those two anyway. An absolute gap (e.g. 2 greens) would follow the
  mechanism more closely, so it's the fallback if the flag proves noisy in real rounds.
- **#8 Blank PCC: now stored blank.** It's used as 0 in calculations, so "unknown" and "0" stay
  different in the CSV, and old CSVs load with blank PCC. Previously a blank was saved as 0.0.
- **#9 Import: kept all-or-nothing, skipping exact duplicates.** A partial import is harder to
  undo than a rejected file that you fix and re-import.
- **#10 Fairway cost: kept at 0.** Revisit in Phase 10 by checking whether the user's own
  fairway % relates to their score.
- **#11 3-putts: still logged, still not scored.** There's no benchmark, and the eBook has no
  3-putt data. A source can be added later.
- **#12 Overshoot: accepted the straight-line fit and the "roughly ±2" label.** The fit's
  residuals are at most 0.94 strokes, and the double counting with the slope term is documented,
  so ±2 is an honest, rough band.

**Unverified, for the user to check:**
- whether your national association uses the (rating − par) term in course handicap, and since
  when;
- the differential formula and the Rule 5.2a table, against the official WHS text;
- (plus-handicap rounding: resolved, see Phase 9 corrections above).

**Other weak assumptions:**
- Per-stat gaps ignore course difficulty, so on a hard course every area looks worse.
- One round is noisy. The average over rounds is the real signal and needs ~5+ rounds.

## Open questions for later phases

- Phase 4's "Random Forest loses" verdict was measured against an **untuned** forest with no
  `max_depth`. The overfitting was diagnosed and never treated. Tuning it (`GridSearchCV` over
  `max_depth`/`min_samples_leaf`) would make that comparison a fair fight.
- The train/test split is random across 2010–2018, so the model trains on 2018 rows to predict
  other 2018 rows. A season-based split (train 2010–2016, test 2017–2018) asks the honest question:
  can it predict a season it has never seen? Expect the score to drop.
- Phase 3's caveat that raw linear coefficients aren't comparable across features is still
  unresolved — a `StandardScaler` pass would fix it.
- `web/predictor.js` has hand-transcribed coefficients and importances that have already drifted
  from the numbers in this file (it ranks scrambling above average putts; Phase 4 has them the
  other way). Generating that file from a script would end the drift.
