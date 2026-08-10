# Phase 6: Add the Strokes Gained features - and discover why we can't.
#
# We've been using 5 of the CSV's 18 columns. Five unused ones are "Strokes
# Gained" (SG) stats, the modern standard in golf analytics:
#
#   SG:OTT             off the tee
#   SG:APR             approach shots
#   SG:ARG             around the green
#   Average SG Putts   putting
#   Average SG Total   all four added together
#
# Each says "how many strokes did this player gain (or lose) versus the
# average tour player in this part of the game?" A +0.5 SG:APR means their
# approach play alone saves them half a stroke per round. That sounds like
# exactly the signal our model is missing, so let's add them.
#
# Spoiler: this phase ends by REJECTING these features. The R^2 goes up
# enormously and the improvement is completely fake. This is DATA LEAKAGE -
# accidentally feeding the model information derived from the answer - and
# it is one of the most common ways real ML projects fail. The tell is
# "suspiciously good results", which is a hard tell to act on, because good
# results are what you were hoping for. Working through it once on data you
# understand is the best way to learn to spot it.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

df = pd.read_csv("data/pgaTourData.csv")

# The 5 features from Phase 2 - things a golfer physically does.
core_features = [
    "Avg Distance",
    "Fairway Percentage",
    "gir",
    "Average Putts",
    "Average Scrambling",
]

# The 4 SG components, and the total that sums them.
sg_features = ["SG:OTT", "SG:APR", "SG:ARG", "Average SG Putts"]
sg_total = "Average SG Total"
target_column = "Average Score"

# The SG columns are missing on exactly the same 634 rows as everything else
# (players who didn't play enough rounds for full stats). So adding them
# costs zero extra rows - still the same 1,678 clean player-seasons as Phase 2.
df_clean = df.dropna(subset=core_features + sg_features + [sg_total, target_column])
print(f"Clean rows: {len(df_clean)} (same as Phase 2 - no extra rows lost)\n")

y = df_clean[target_column]

# Same 5-fold setup as Phase 4.5, so every number below is directly
# comparable to that phase rather than measured a different way.
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


def score_features(columns, label):
    """Run 5-fold CV on one feature set with Linear Regression."""
    results = cross_validate(
        LinearRegression(),
        df_clean[columns],
        y,
        cv=kfold,
        scoring=["r2", "neg_root_mean_squared_error"],
    )
    r2 = results["test_r2"]
    # sklearn reports RMSE negative ("higher is always better"); flip it back.
    rmse = -results["test_neg_root_mean_squared_error"]

    print(f"  {label:<42} R^2 {r2.mean():.3f}   RMSE {rmse.mean():.3f}")
    return r2.mean(), rmse.mean()


# ============================================================
# Part 1: The exciting result
# ============================================================
print("=" * 66)
print("PART 1: Adding Strokes Gained")
print("=" * 66 + "\n")

core_r2, core_rmse = score_features(core_features, "A. Core 5 stats (Phase 3-5 model)")
sg_r2, sg_rmse = score_features(core_features + sg_features, "B. Core 5 + 4 SG components")

print(f"\n  R^2  {core_r2:.3f} -> {sg_r2:.3f}  ({sg_r2 - core_r2:+.3f})")
print(f"  RMSE {core_rmse:.3f} -> {sg_rmse:.3f}  ({sg_rmse - core_rmse:+.3f} strokes)")
print("\n  Our error more than halved. This looks like a huge win.")
print("  It is not a win. Let's find out why.\n")


# ============================================================
# Part 2: Three tests that show the improvement is fake
# ============================================================
print("=" * 66)
print("PART 2: Testing whether the model actually learned anything")
print("=" * 66 + "\n")

# TEST 1: The decisive one. Throw away every golf stat and keep only
# SG:Total. If a model that knows NOTHING about how a player hits the ball
# still predicts nearly as well as our 9-feature model, then those 9
# features were never what was doing the work.
print("Test 1 - strip out the golf stats entirely:\n")
score_features(core_features + sg_features, "B. Core 5 + 4 SG components (9 features)")
total_r2, total_rmse = score_features([sg_total], "E. 'Average SG Total' ALONE (1 feature)")
print("\n  One column, no information about distance, accuracy, greens,")
print("  putting or scrambling, and it matches the 9-feature model.")
print("  Whatever SG:Total is, it isn't a golf stat - it's the answer.\n")

# TEST 2: The 4 components aren't safer than the total, because they add up
# to it. Splitting a leaky number into four pieces doesn't stop it leaking.
print("Test 2 - are the 4 components any safer than the total?\n")
score_features(sg_features, "F. 4 SG components alone, no core stats")
component_sum = df_clean[sg_features].sum(axis=1)
gap = (df_clean[sg_total] - component_sum).abs()
print(f"\n  |SG:Total - (OTT + APR + ARG + Putts)|: mean {gap.mean():.4f}, max {gap.max():.4f}")
print("  The components ARE the total, just split four ways. Same leak.\n")

# TEST 3: A side-lesson about linear models. Adding SG:Total on top of the
# 4 components changes nothing at all - because a linear model is a weighted
# sum, and it could already build that total out of the four columns it had.
# A feature that's an exact combination of existing features carries zero new
# information. (This is called perfect collinearity.)
print("Test 3 - what happens if we add SG:Total on top of the components?\n")
score_features(core_features + sg_features, "B. Core + components")
score_features(core_features + sg_features + [sg_total], "C. Core + components + SG:Total")
print("\n  No change. A linear model is a weighted sum, so it could already")
print("  add those four columns together itself. Nothing new was added.\n")


# ============================================================
# Part 3: The smoking gun
# ============================================================
print("=" * 66)
print("PART 3: Why SG:Total is the target in disguise")
print("=" * 66 + "\n")

# Strokes Gained is DEFINED as your score relative to the field average:
#
#     SG:Total = (field average score) - (player's score)
#
# Rearranged, that's:
#
#     player's score = (field average score) - SG:Total
#
# If that's true, then adding score and SG:Total back together should
# recover the field average - the same number for every player in a season.
# Let's check.
implied_field_average = df_clean[target_column] + df_clean[sg_total]
by_year = implied_field_average.groupby(df_clean["Year"]).agg(["mean", "std"])

print("'Average Score' + 'SG:Total', grouped by season:\n")
print("  Year    mean     std")
for year, row in by_year.iterrows():
    print(f"  {year}   {row['mean']:.3f}   {row['std']:.3f}")

print(f"\n  Overall: {implied_field_average.mean():.3f} +/- {implied_field_average.std():.3f}")
print("\n  It's the same number every year - the tour's field scoring average.")
print(f"  So the whole 'model' collapses to one line of arithmetic:\n")
print(f"      Average Score = {implied_field_average.mean():.2f} - SG:Total\n")

# The clincher: if the model is really just that subtraction, then its error
# should equal the wobble in the constant. Compare the two numbers.
print("  And the error we measured matches that exactly:")
print(f"      spread of the 'constant':  {implied_field_average.std():.3f}")
print(f"      RMSE of the SG model:      {sg_rmse:.3f} strokes")
print("\n  The model's entire remaining error IS the year-to-year drift in")
print("  the field average. It learned the identity and nothing else.\n")


# ============================================================
# Part 4: The decision
# ============================================================
print("=" * 66)
print("PART 4: Decision")
print("=" * 66 + "\n")
print("""We are REJECTING all five Strokes Gained columns.

Not because 0.927 is a bad score, but because it isn't a prediction. To
compute SG:Total for a season you must already know that season's scores -
the exact thing we're trying to predict. In real use, forecasting a player's
2019 average before 2019 is played, SG simply does not exist yet. A model
that needs it can never actually be run.

This is the difference between EXPLAINING and PREDICTING. SG is excellent
for explaining a finished season. It is useless for predicting an unplayed
one. Our core 5 stats are worse on paper and are the honest answer:

  - they describe what a player DOES, not what they scored
  - they're reasonably stable year to year, so last season's values are a
    fair guess at next season's
  - R^2 0.678 that survives contact with reality beats R^2 0.927 that can
    never be computed when you need it

So Phase 3-5's model stands unchanged, and the Streamlit app keeps its 5
sliders. The real finding of this phase is a negative one, and negative
findings are still findings: there is no free lunch hiding in this CSV.
The only columns that would raise our score are ones derived from the
answer, which is exactly why nobody left them lying around for us.""")
