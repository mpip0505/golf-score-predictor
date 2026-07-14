# Phase 4: Train a Random Forest and compare it to the Linear Regression
# baseline from Phase 3.
#
# A Random Forest is many decision trees trained on slightly different
# random subsets of the data. Each tree learns its own set of "if this
# stat is above/below X, go this way" rules. To predict, every tree makes
# a guess and the forest averages them. Averaging many imperfect trees
# tends to cancel out their individual mistakes, and - unlike Linear
# Regression - it can learn curves and interactions between features, not
# just straight-line relationships.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Same loading, cleaning, and split as Phase 3 ---
# Using the same random_state=42 and test_size=0.2 means both models are
# trained and tested on the exact same rows, which is what makes the
# comparison below fair.
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def evaluate(model, name):
    """Fit a model, then print R^2 and RMSE on both train and test data.

    We print train AND test to check for overfitting: a model that scores
    much higher on the training data it memorized than on the test data it
    never saw has learned noise/quirks specific to the training rows,
    rather than a pattern that generalizes to new players.
    """
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"{name}")
    print(f"  Train  R^2 = {train_r2:.3f}   RMSE = {train_rmse:.3f} strokes")
    print(f"  Test   R^2 = {test_r2:.3f}   RMSE = {test_rmse:.3f} strokes")
    print(f"  Overfitting gap (train R^2 - test R^2): {train_r2 - test_r2:.3f}")
    return model


# --- Re-run the baseline here so both numbers are printed side by side ---
linreg = evaluate(LinearRegression(), "Linear Regression")

# --- Train the Random Forest ---
# n_estimators=100: build 100 trees and average their predictions.
# random_state=42: makes the "randomness" in how trees are built
# reproducible, so rerunning this script gives the same result.
forest = evaluate(
    RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest"
)

# --- Feature importances ---
# For each split a tree makes (e.g. "is gir > 65?"), scikit-learn tracks
# how much that split reduced prediction error. A feature's importance is
# the total error-reduction credited to it, averaged across all 100 trees
# and all splits, then normalized so all features' importances add up to 1.
# Unlike Linear Regression's coefficients, this isn't affected by each
# feature's units or scale, so importances ARE directly comparable to
# each other.
importances = pd.Series(forest.feature_importances_, index=feature_columns)
importances = importances.sort_values(ascending=False)

print("\nFeature importances (Random Forest), most to least important:")
for name, importance in importances.items():
    print(f"  {name:22s} {importance:.3f}")
