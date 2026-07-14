# Phase 5: Streamlit app.
#
# Streamlit turns a plain Python script into a web app. The key mental
# model: Streamlit reruns this ENTIRE file top-to-bottom every time you
# move a slider or interact with a widget. That's why we "cache" the slow
# parts (loading the CSV, training the models) below - without caching,
# we'd reload the CSV and retrain both models on every single slider drag.
#
# Model choice, decided in Phase 4.5: Linear Regression is the more
# accurate model (it won all 5 cross-validation folds), so it powers the
# live prediction. The Random Forest is still trained here too, purely to
# get its feature importances for the chart - those aren't confounded by
# each feature's scale the way Linear Regression's raw coefficients are,
# so they're the more trustworthy answer to "what matters most." See
# FINDINGS.md for the full comparison.

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

feature_columns = [
    "Avg Distance",
    "Fairway Percentage",
    "gir",
    "Average Putts",
    "Average Scrambling",
]
target_column = "Average Score"


# @st.cache_data tells Streamlit: "run this function once, remember its
# result, and reuse that result on future reruns instead of redoing the
# work." Here that means the CSV is only read and cleaned once per app
# session, not on every slider move.
@st.cache_data
def load_clean_data():
    df = pd.read_csv("data/pgaTourData.csv")
    return df.dropna(subset=feature_columns + [target_column])


# @st.cache_resource is the same idea, but for things like trained models
# (as opposed to plain data). Training happens once; every later rerun
# reuses the already-fitted models instead of calling .fit() again.
#
# Note we train on ALL the clean data here, not an 80/20 split. Phases 3-4.5
# already answered "how good is this model on unseen data?" using
# train/test splits and cross-validation - that question is settled. Now
# that we're shipping the model for real predictions, there's no reason to
# hold back 20% of the data anymore; using every row we have gives the
# best-fitted final model.
@st.cache_resource
def train_models(df_clean):
    X = df_clean[feature_columns]
    y = df_clean[target_column]

    linreg = LinearRegression().fit(X, y)
    forest = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    return linreg, forest


df_clean = load_clean_data()
linreg, forest = train_models(df_clean)

st.title("Golf Scoring Average Predictor")
st.write(
    "Move the sliders to describe a player's season stats and see the "
    "predicted PGA Tour scoring average update live. Remember: in golf, "
    "**lower is better**."
)

# --- Sliders ---
# One slider per feature. Bounds come from the real range of values seen
# in the training data (data/pgaTourData.csv, 2010-2018) - sliding a
# player's stats way outside that range would ask the model to guess in
# territory it never learned from.
st.sidebar.header("Player stats")

avg_distance = st.sidebar.slider(
    "Avg Distance (yards)", min_value=260.0, max_value=320.0, value=290.8, step=0.5
)
fairway_percentage = st.sidebar.slider(
    "Fairway Percentage (%)", min_value=40.0, max_value=80.0, value=61.4, step=0.5
)
gir = st.sidebar.slider(
    "Greens in Regulation - gir (%)",
    min_value=50.0,
    max_value=75.0,
    value=65.7,
    step=0.5,
)
average_putts = st.sidebar.slider(
    "Average Putts (per round)", min_value=27.0, max_value=31.5, value=29.2, step=0.05
)
average_scrambling = st.sidebar.slider(
    "Average Scrambling (%)", min_value=40.0, max_value=70.0, value=58.1, step=0.5
)

# --- Live prediction ---
# scikit-learn's .predict() expects a table shaped like the training data:
# same column names, same order. We build a one-row DataFrame from the
# slider values so it matches X from train_models() above.
input_df = pd.DataFrame(
    [[avg_distance, fairway_percentage, gir, average_putts, average_scrambling]],
    columns=feature_columns,
)
predicted_score = linreg.predict(input_df)[0]

st.metric("Predicted Average Score", f"{predicted_score:.2f} strokes")

# --- Feature importance chart ---
st.subheader("Which stats matter most?")
st.write(
    "From the Random Forest's feature importances (Phase 4) - how much "
    "each stat contributed to predicting scoring average, across every "
    "player-season in the data."
)

importances = pd.Series(forest.feature_importances_, index=feature_columns)
importances = importances.sort_values(ascending=False)  # most important first

# sort=False tells Streamlit to plot the bars in the DataFrame's own row
# order instead of re-sorting them itself (its default behavior sorts
# categories alphabetically, which would undo the sort_values() above).
st.bar_chart(importances, horizontal=True, sort=False)
