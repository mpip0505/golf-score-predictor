# Phase 4.5: Cross-validation robustness check.
#
# Phase 3/4 compared models using ONE 80/20 train/test split. That's risky:
# whichever 336 rows happened to land in the test set could make a model
# look better or worse than it really is, just by chance. We even saw a
# symptom of this - Linear Regression's test R^2 (0.770) came out HIGHER
# than its train R^2 (0.663), which only makes sense if that particular
# test slice happened to be a bit "easier."
#
# Cross-validation (CV) fixes this by not relying on a single split:
#
#   1. Split all the data into 5 equal chunks ("folds").
#   2. Run 5 rounds. In each round, 4 folds are the training data and the
#      1 remaining fold is the test data - a different fold each round.
#   3. Every row gets used as test data in exactly one round, and as
#      training data in the other four.
#
# This gives us 5 independent R^2/RMSE scores per model instead of 1.
# Averaging them is a much more reliable estimate of real-world
# performance, and the SPREAD (standard deviation) across the 5 scores
# tells us how much to trust that average: a small spread means the model
# performs consistently no matter which rows are held out; a large spread
# means a single-split result (like Phase 3/4's) could easily have gone
# either way by luck.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

# --- Same loading and cleaning as before ---
# Note: no train_test_split here. Cross-validation uses the FULL cleaned
# dataset (all 1,678 rows) - the folding process below does the splitting
# for us, 5 different ways.
df = pd.read_csv("data/pgaTourData.csv")

feature_columns = [
    "Avg Distance",
    "Fairway Percentage",
    "gir",
    "Average Putts",
    "Average Scrambling",
]
target_column = "Average Score"

df_clean = df.dropna(subset=feature_columns + [target_column])
X = df_clean[feature_columns]
y = df_clean[target_column]

# KFold defines exactly how the 5 folds are built. shuffle=True mixes the
# rows up randomly before folding - important here because the CSV's rows
# aren't in random order (they're grouped by year), so folding them
# without shuffling could give each fold a skewed mix of years. random_state=42
# keeps that shuffle reproducible, consistent with every earlier phase.
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


def cross_validate_model(model, name):
    """Run 5-fold CV for one model and print mean +/- std R^2 and RMSE."""
    results = cross_validate(
        model,
        X,
        y,
        cv=kfold,
        scoring=["r2", "neg_root_mean_squared_error"],
    )

    r2_scores = results["test_r2"]
    # scikit-learn reports RMSE as NEGATIVE (its convention: "higher score
    # is always better", and error should always get worse/lower as a
    # score). We flip the sign back to get normal, positive RMSE values.
    rmse_scores = -results["test_neg_root_mean_squared_error"]

    print(f"{name}")
    print(f"  Per-fold R^2:   {np.round(r2_scores, 3)}")
    print(f"  Per-fold RMSE:  {np.round(rmse_scores, 3)}")
    print(f"  R^2:  {r2_scores.mean():.3f} +/- {r2_scores.std():.3f}")
    print(f"  RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f} strokes")
    print()

    return r2_scores, rmse_scores


linreg_r2, linreg_rmse = cross_validate_model(LinearRegression(), "Linear Regression")
forest_r2, forest_rmse = cross_validate_model(
    RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest"
)

# --- Plain comparison ---
print("Comparison:")
print(
    f"  Linear Regression R^2: {linreg_r2.mean():.3f} +/- {linreg_r2.std():.3f}"
)
print(f"  Random Forest     R^2: {forest_r2.mean():.3f} +/- {forest_r2.std():.3f}")
