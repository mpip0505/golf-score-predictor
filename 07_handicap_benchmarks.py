# Phase 7: Amateur handicap benchmarks (+ a small "tour reference").
#
# Phases 1-6 were all about PGA Tour pros. The new goal is to help an
# AMATEUR golfer see which part of their game costs them the most strokes.
# For that, the main reference is no longer the PGA data - it's a table of
# what golfers at each handicap typically do (data/handicap_benchmarks.csv,
# from Shot Scope's Strokes Gained eBook, 4th ed, 2021).
#
# What this script does:
#   1. Loads and validates the benchmark table, and checks each stat
#      trends in a consistent direction across handicaps.
#   2. Shows how to look up the expected stat for ANY handicap (e.g. 12),
#      not just the 6 handicaps in the table, using interpolation.
#   3. Prints the table, with simple text bar charts.
#   4. Adds a "tour" column (PGA averages) next to it for reference.
#   5. Trains a 4-feature tour model and cross-validates it.
#   6. Builds a range guard so the tour model is never used on inputs it
#      wasn't trained on (which, it turns out, is every amateur).
#
# The PGA model is SECONDARY from here on. The benchmarks (and, from
# Phase 8, your own logged rounds) are the main engine.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

from benchmarks import (
    STAT_COLUMNS,
    TOUR_FEATURES,
    expected_stat,
    inside_tour_range,
    load_benchmarks,
    tour_feature_ranges,
)

# =====================================================================
# Step 1: load and validate the benchmarks
# =====================================================================
# load_benchmarks() (in benchmarks.py) raises a clear error if a column is
# missing, a value isn't a number, or the rows are out of order. If this
# line runs without crashing, the table passed all those checks.
bench = load_benchmarks()
print(f"Loaded {len(bench)} benchmark rows from {bench['source'].iloc[0]} "
      f"({bench['source_year'].iloc[0]}). Validation passed.\n")

# --- Trend check ---
# Golf common sense says that as handicap goes UP, some stats should only
# go up (score, putts, penalties) and others should only go down (GIR,
# fairways, scrambling). If a stat breaks that pattern, it might be a typo
# - or it might be real. We REPORT it, we don't "fix" it: changing source
# numbers because they look odd to us would be inventing data.
expected_direction = {
    "score_to_par": "up",
    "fairways_pct": "down",
    "gir_pct": "down",
    "putts_per_round": "up",
    "three_putts_per_round": "up",
    "scramble_pct": "down",
    "penalties_per_round": "up",
}

print("Trend check (as handicap rises):")
for stat, direction in expected_direction.items():
    values = bench[stat].dropna()
    if len(values) == 0:
        print(f"  {stat:22s} no data in source")
        continue

    # .diff() gives the change from each row to the next, e.g. GIR
    # 61 -> 44 -> 36 gives changes of -17, -8. We flip the sign for "down"
    # stats so that, either way, a positive change means "expected direction".
    changes = values.diff().dropna()
    if direction == "down":
        changes = -changes

    if (changes > 0).all():
        verdict = "OK, strictly " + direction
    elif (changes >= 0).all():
        verdict = "FLAT in places (a tie between neighbouring handicaps)"
    else:
        verdict = "NOT MONOTONIC (moves the wrong way somewhere)"
    print(f"  {stat:22s} {verdict}: {values.tolist()}")
print()

# =====================================================================
# Step 2: interpolation - expected stat for any handicap
# =====================================================================
# The table only has handicaps 0, 5, 10, 15, 20, 25. expected_stat()
# fills the gaps with straight lines between them, and clamps outside
# 0-25 (see its docstring in benchmarks.py for why).
print("Interpolation examples (GIR %):")
for hcp in [-2, 0, 7, 12, 25, 32]:
    gir = expected_stat(bench, "gir_pct", hcp)
    note = "  <- clamped, outside the table" if hcp < 0 or hcp > 25 else ""
    print(f"  handicap {hcp:>3}: {gir:5.1f}%{note}")
print(f"  three_putts_per_round at 12 hcp: "
      f"{expected_stat(bench, 'three_putts_per_round', 12)} (source has no data)\n")

# =====================================================================
# Step 3: the average-stats-by-handicap table and a text chart
# =====================================================================
# set_index("band_label") makes each handicap a row label, so `table` has
# one ROW per handicap and one COLUMN per stat. That's the shape Streamlit's
# st.dataframe / st.line_chart accept, so Phase 8 can reuse it as-is.
table = bench.set_index("band_label")[STAT_COLUMNS]

# =====================================================================
# Step 4: add a "tour" column from the PGA data
# =====================================================================
# Same 1,678 clean rows as every earlier phase (dropping the 634 rows that
# are missing the stats together).
tour = pd.read_csv("data/pgaTourData.csv")
tour_clean = tour.dropna(
    subset=["Avg Distance", *TOUR_FEATURES, "Average Score"]
)

# Map each PGA column to the benchmark stat it corresponds to. Where the
# definitions might not match exactly, the comment says so - these are
# the "apples vs. slightly different apples" caveats:
tour_means = {
    # Same idea (% of tee shots on par 4s/5s finishing in the fairway), but
    # the source doesn't spell out its par-3 / fairway-edge rules.
    "fairways_pct": tour_clean["Fairway Percentage"].mean(),
    # Standard definition on both sides: on the green in (par - 2) shots.
    "gir_pct": tour_clean["gir"].mean(),
    # Per round on both sides, and both count only strokes ON the green:
    # the Shot Scope eBook says a stroke from the fringe is not a putt
    # ("the shot from the fringe is classed as 1 putt" when the golfer
    # took 2 strokes from there). Same definition as the Tour.
    "putts_per_round": tour_clean["Average Putts"].mean(),
    # Tour definition: % of MISSED greens where the player still made par
    # or better. If Shot Scope's denominator differs, this isn't a fair
    # comparison - unverified.
    "scramble_pct": tour_clean["Average Scrambling"].mean(),
    # Left blank: the CSV has raw Average Score (~70.9), not score to par.
    # Tour par varies (70-72) and isn't in the data, so converting it would
    # mean guessing the par.
    "score_to_par": np.nan,
    # Not in the PGA CSV at all.
    "three_putts_per_round": np.nan,
    "penalties_per_round": np.nan,
}
# .loc[new_label] = values adds one more row at the bottom. Passing a
# Series lets pandas line each value up with its column by name, so the
# dict's order doesn't matter.
table.loc["Tour (PGA)"] = pd.Series(tour_means)

pd.set_option("display.width", 120)
print("Average stats by handicap (Shot Scope 2021), with PGA Tour reference:")
# For PRINTING we flip it with .T (transpose) so each stat is a row -
# 7 stats x 7 columns reads better in a terminal than the other way round.
# This doesn't change `table` itself, just what we print.
print(table.T.round(2).to_string())
print(f"\n(Tour row = mean of {len(tour_clean)} PGA player-seasons, 2010-2018. "
      f"Raw tour Average Score: {tour_clean['Average Score'].mean():.2f}, "
      f"not comparable to score_to_par.)\n")


def text_bars(stat, width=40):
    """A tiny text bar chart: one '#' bar per row, scaled to the largest value."""
    column = table[stat].dropna()
    biggest = column.max()
    print(f"{stat}:")
    for label, value in column.items():
        bar = "#" * round(value / biggest * width)
        print(f"  {label:>10s} | {bar} {value:.1f}")
    print()


# Two charts that tell the main story: GIR collapses as handicap rises,
# and score to par climbs with it.
text_bars("gir_pct")
text_bars("score_to_par")

# =====================================================================
# Step 5: a separate 4-feature tour model, 5-fold cross-validated
# =====================================================================
# Same setup as Phase 4.5 so the numbers are directly comparable, just
# without Avg Distance. If R^2 barely drops, distance wasn't adding much
# on top of the other four - good news for amateurs who can't measure it.
X = tour_clean[TOUR_FEATURES]
y = tour_clean["Average Score"]
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

results = cross_validate(
    LinearRegression(), X, y, cv=kfold,
    scoring=["r2", "neg_root_mean_squared_error"],
)
r2 = results["test_r2"]
rmse = -results["test_neg_root_mean_squared_error"]  # sklearn reports it negative

print("Tour model, 4 features (no Avg Distance), Linear Regression, 5-fold CV:")
print(f"  Per-fold R^2:  {np.round(r2, 3)}")
print(f"  R^2:  {r2.mean():.3f} +/- {r2.std():.3f}   (5-feature, Phase 4.5: 0.678 +/- 0.056)")
print(f"  RMSE: {rmse.mean():.3f} +/- {rmse.std():.3f} strokes "
      f"(5-feature, Phase 4.5: 0.391 +/- 0.014)\n")

# Fit once on all rows for the range-guard demo below (CV only measured it).
tour_model = LinearRegression().fit(X, y)

# =====================================================================
# Step 6: range guard
# =====================================================================
ranges = tour_feature_ranges(tour_clean)
print("Tour model training range (anything outside is off-limits):")
print(ranges.round(2).to_string())
print()

# Demo 1: a typical tour player - inside the range, so the model may answer.
tour_like = {"Fairway Percentage": 61, "gir": 66, "Average Putts": 29.2,
             "Average Scrambling": 58}
# Demo 2: a 15-handicapper, built from the benchmark table.
amateur_15 = {
    "Fairway Percentage": expected_stat(bench, "fairways_pct", 15),
    "gir": expected_stat(bench, "gir_pct", 15),
    "Average Putts": expected_stat(bench, "putts_per_round", 15),
    "Average Scrambling": expected_stat(bench, "scramble_pct", 15),
}

for name, inputs in [("Tour-like player", tour_like), ("15-handicapper", amateur_15)]:
    ok, problems = inside_tour_range(ranges, inputs)
    print(f"{name}: inside tour range? {ok}")
    if ok:
        row = pd.DataFrame([inputs], columns=TOUR_FEATURES)
        print(f"  Tour-reference score: {tour_model.predict(row)[0]:.2f} "
              f"(reference only, NOT an amateur prediction)")
    else:
        for problem in problems:
            print(f"  - {problem}")
        print("  -> tour model refuses to answer")
    print()

# Check every benchmark handicap, to see where the amateur table and the
# tour data overlap at all.
print("Which benchmark handicaps fall inside the tour range?")
for _, row in bench.iterrows():
    inputs = {
        "Fairway Percentage": row["fairways_pct"],
        "gir": row["gir_pct"],
        "Average Putts": row["putts_per_round"],
        "Average Scrambling": row["scramble_pct"],
    }
    ok, problems = inside_tour_range(ranges, inputs)
    print(f"  {row['band_label']:>6s}: {'inside' if ok else 'OUTSIDE'}"
          f"{'' if ok else ' (' + str(len(problems)) + ' of 4 features out)'}")
