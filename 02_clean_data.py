# Phase 2: Clean the data and pick our features and target.
#
# A model needs two things:
#   - Features (X): the input stats it learns from.
#   - Target (y): the number it's trying to predict.
#
# It also can't work with missing values (NaN), so we have to decide what
# to do about the 634 rows from Phase 1 that were missing most stats.

import pandas as pd

df = pd.read_csv("data/pgaTourData.csv")

# --- Choosing the features ---
# We pick stats that plausibly *cause* scoring average to be low or high,
# and that a golfer could look at and understand:
#   - Avg Distance:          how far they hit the ball off the tee
#   - Fairway Percentage:    driving accuracy
#   - gir:                   % of greens hit in regulation (approach quality)
#   - Average Putts:         putts per round (putting quality)
#   - Average Scrambling:    % of the time they save par after a bad shot
#
# We're deliberately leaving out Points, Wins, Top 10, and Money. Those are
# *results* of scoring well (you win money and points because you score
# well), not inputs that explain the score - including them would be
# circular reasoning, not prediction.
feature_columns = [
    "Avg Distance",
    "Fairway Percentage",
    "gir",
    "Average Putts",
    "Average Scrambling",
]
target_column = "Average Score"

# --- Handling missing values ---
# From Phase 1, we know 634 rows are missing these columns together (players
# who didn't play enough rounds for full stats to be reported). There's no
# good way to guess what an unreported stat "should" be, so instead of
# inventing values, we drop those rows. We only look at the columns we
# actually use - a row missing "Wins" (which we don't use) is fine.
columns_we_need = feature_columns + [target_column]
df_clean = df.dropna(subset=columns_we_need)

print(f"Rows before cleaning: {len(df)}")
print(f"Rows after cleaning:  {len(df_clean)}")
print(f"Rows dropped:         {len(df) - len(df_clean)}")

# Build our final X (features) and y (target) tables.
X = df_clean[feature_columns]
y = df_clean[target_column]

print("\nFeature columns (X):")
print(X.head())

print("\nTarget column (y):")
print(y.head())

print("\nAny missing values left in X?", X.isnull().values.any())
print("Any missing values left in y?", y.isnull().values.any())
