# Phase 3: Train a baseline Linear Regression model.
#
# "Baseline" means: the simplest reasonable model, used as a reference
# point. If a fancier model later can't beat this, the fancier model isn't
# worth the extra complexity.
#
# Linear Regression works by finding one weight (coefficient) per feature,
# plus a starting offset (intercept), such that:
#
#   predicted_score = intercept
#                    + weight_1 * Avg_Distance
#                    + weight_2 * Fairway_Percentage
#                    + weight_3 * gir
#                    + weight_4 * Average_Putts
#                    + weight_5 * Average_Scrambling
#
# "Training" means: choosing the weights that make this formula's
# predictions match the real Average Score as closely as possible, across
# all the training rows at once.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Same loading and cleaning as Phase 2 ---
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

# --- Train/test split ---
# We hide 20% of the rows from the model during training (test_size=0.2).
# random_state=42 just makes the "random" split reproducible - rerunning
# this script gives the exact same split every time, so results are
# comparable run to run.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training rows: {len(X_train)}")
print(f"Test rows:     {len(X_test)}")

# --- Train the model ---
# .fit() is where the actual learning happens: scikit-learn searches for
# the weights that best match y_train given X_train.
model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate on the test set ---
# .predict() runs the learned formula on data. Crucially, X_test was never
# used in .fit(), so this tells us how the model does on "new" players.
y_pred = model.predict(X_test)

# R² (R-squared): fraction of the variation in Average Score that the
# model explains, from 0 (no better than guessing the average every time)
# to 1 (perfect prediction). Can go negative if the model is worse than
# just guessing the average.
r2 = r2_score(y_test, y_pred)

# RMSE (Root Mean Squared Error): the typical size of the model's
# prediction error, in the same units as the target - here, strokes.
# It's computed by squaring every error (to make them all positive and
# punish big misses more), averaging those, then square-rooting back to
# the original units.
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nR^2 on test set:   {r2:.3f}")
print(f"RMSE on test set:  {rmse:.3f} strokes")

# Bonus: Linear Regression's weights are a first, rough look at "what
# matters." A bigger weight (ignoring sign) means that feature moves the
# prediction more per unit of change. Phase 4's Random Forest will give a
# more reliable version of this same idea.
print("\nLearned weights (coefficients):")
for name, weight in zip(feature_columns, model.coef_):
    print(f"  {name:22s} {weight:+.4f}")
print(f"  {'intercept':22s} {model.intercept_:+.4f}")
