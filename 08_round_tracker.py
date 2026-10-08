# Phase 8: Round tracker + "which part of my game needs fixing" report.
#
# Run with:   streamlit run 08_round_tracker.py
#
# This is a separate Streamlit app from app.py (the PGA predictor). The two
# answer different questions, and keeping them apart follows the Phase 7
# rule that the PGA model is never used for amateur predictions.
#
# What it does:
#   1. Log a round: course info from the scorecard + your stats. Saved to
#      data/my_rounds.csv (gitignored - personal data).
#   2. Round report, in two separate parts:
#        a) COURSE-ADJUSTED: your score vs what a golfer with your index
#           typically shoots on THIS course from THESE tees.
#        b) NOT course-adjusted: each part of your game vs a typical golfer
#           at your index, converted to strokes lost, so the biggest leak
#           stands out.
#   3. Your game over time: estimated index and average strokes lost per area.
#   4. The benchmark table from Phase 7.
#
# The logic lives in the shared helpers (rounds.py, handicap.py,
# strokes_lost.py, benchmarks.py). This file only builds the page, which
# keeps the maths testable without clicking through a web page.

import pandas as pd
import streamlit as st

from benchmarks import STAT_COLUMNS, load_benchmarks
from handicap import (
    add_overshoot,
    course_handicap,
    estimated_index,
    expected_score,
    score_differential,
)
from rounds import load_rounds, save_round
from strokes_lost import biggest_leak, strokes_lost


# Benchmarks never change while the app runs, so load them once (see the
# caching notes in app.py). add_overshoot() adds the column expected_score()
# needs.
@st.cache_data
def get_benchmarks():
    return add_overshoot(load_benchmarks())


bench = get_benchmarks()

st.title("My Golf Round Tracker")
st.write(
    "Log your 18-hole rounds to see where you're losing strokes compared with "
    "a typical golfer at your handicap index. The more rounds you add, the "
    "clearer the pattern gets."
)

# =====================================================================
# 1. Log a round
# =====================================================================
# st.form groups the inputs so the page DOESN'T rerun on every keystroke -
# nothing happens until you press "Save round". value=None leaves a box
# empty instead of pre-filling a made-up number, so you can't accidentally
# save a fake round; validate_round() says what's missing.
with st.form("log_round", clear_on_submit=True):
    st.subheader("Log a round")

    col1, col2 = st.columns(2)
    date = col1.date_input("Date")
    course_name = col2.text_input("Course name")
    tee = col1.text_input("Tees played (e.g. White)")
    handicap_index = col2.number_input(
        "Your handicap index today (plus handicap: enter as negative)",
        value=None, step=0.1, format="%.1f",
    )

    st.caption("From the scorecard, for the tees you played:")
    col1, col2, col3, col4 = st.columns(4)
    course_rating = col1.number_input("Course rating", value=None, step=0.1, format="%.1f")
    slope_rating = col2.number_input("Slope rating", value=None, step=1)
    par = col3.number_input("Par", value=None, step=1)
    length_yards = col4.number_input("Length, yds (optional)", value=None, step=1)

    st.caption("Your round:")
    col1, col2, col3 = st.columns(3)
    score = col1.number_input("Score", value=None, step=1)
    fairways_hit = col2.number_input("Fairways hit", value=None, step=1)
    fairways_possible = col3.number_input(
        "Fairways possible (par 4s + 5s)", value=None, step=1
    )
    gir = col1.number_input("Greens in regulation", value=None, step=1)
    putts = col2.number_input("Total putts", value=None, step=1)
    three_putts = col3.number_input("3-putts", value=None, step=1)
    up_down_attempts = col1.number_input("Up-and-down attempts", value=None, step=1)
    up_down_saves = col2.number_input("Up-and-down saves", value=None, step=1)
    penalties = col3.number_input("Penalty strokes", value=None, step=1)

    submitted = st.form_submit_button("Save round")

if submitted:
    new_round = {
        "date": date,
        "course_name": course_name,
        "tee": tee,
        "course_rating": course_rating,
        "slope_rating": slope_rating,
        "par": par,
        "length_yards": length_yards,
        "handicap_index": handicap_index,
        "score": score,
        "fairways_hit": fairways_hit,
        "fairways_possible": fairways_possible,
        "gir": gir,
        "putts": putts,
        "three_putts": three_putts,
        "up_down_attempts": up_down_attempts,
        "up_down_saves": up_down_saves,
        "penalties": penalties,
    }
    problems = save_round(new_round)
    if problems:
        st.error("Round not saved:\n\n" + "\n".join(f"- {p}" for p in problems))
    else:
        st.success("Round saved.")

# Loaded AFTER the form, so a round saved just now is included below.
rounds = load_rounds()

if len(rounds) == 0:
    st.info("No rounds logged yet. Add your first round above.")
    st.stop()  # nothing more to show until there's at least one round

# =====================================================================
# 2. Round report
# =====================================================================
st.header("Round report")

# A readable label per round for the dropdown. Newest first, so the
# default selection is the round you just logged.
labels = [
    f"{row.date:%Y-%m-%d} - {row.course_name} ({row.tee}) - {row.score:g}"
    for row in rounds.itertuples()
]
choice = st.selectbox("Round", options=list(reversed(range(len(rounds)))),
                      format_func=lambda i: labels[i])
r = rounds.iloc[choice]

# --- a) Course-adjusted ---
expected = expected_score(r["handicap_index"], r["course_rating"], r["slope_rating"], bench)
diff = score_differential(r["score"], r["course_rating"], r["slope_rating"])

st.subheader("On this course (course-adjusted)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Course handicap",
            course_handicap(r["handicap_index"], r["slope_rating"], r["course_rating"], r["par"]))
col2.metric("Expected score", f"{expected:.1f}")
# delta_color="inverse": in golf a NEGATIVE delta (fewer strokes) is good,
# so we tell Streamlit to show negative in green and positive in red.
col3.metric("Your score", f"{r['score']:g}",
            delta=f"{r['score'] - expected:+.1f} vs expected", delta_color="inverse")
col4.metric("Score differential", f"{diff:.1f}")
st.caption(
    "Expected score = course rating + index x slope/113 + the typical amount "
    "golfers at your index score above their handicap (from the benchmark file). "
    "Differential uses your gross score (no net-double-bogey cap) and no "
    "conditions adjustment, so it's an estimate."
)

# --- b) Per-area breakdown (NOT course-adjusted) ---
st.subheader("Where the strokes went (not course-adjusted)")
breakdown = strokes_lost(r, bench)

leak = biggest_leak(breakdown)
if leak is not None:
    st.warning(f"Biggest leak this round: **{leak['area']}**, "
               f"about {leak['strokes_lost']:.1f} strokes vs a typical "
               f"{r['handicap_index']:g} handicap.")
else:
    st.success("You beat a typical golfer at your index in every area this round.")

st.dataframe(
    breakdown.assign(strokes_lost=breakdown["strokes_lost"].round(2)),
    hide_index=True,
)
st.caption(
    "Positive = strokes lost, negative = strokes gained, vs a typical golfer "
    "at your handicap index (Shot Scope 2021 benchmarks). These are NOT "
    "adjusted for course difficulty: on a hard course every area will look "
    "worse. Rows with counted = False are shown for information only."
)

# Bar chart of the counted areas only. sort=False keeps our row order.
counted = breakdown[breakdown["counted"]].set_index("area")["strokes_lost"]
st.bar_chart(counted, horizontal=True, sort=False)

# =====================================================================
# 3. Your game over time
# =====================================================================
st.header("Your game over time")

# Differentials oldest first, as estimated_index() expects.
rounds["differential"] = [
    score_differential(row.score, row.course_rating, row.slope_rating)
    for row in rounds.itertuples()
]
value, explanation = estimated_index(rounds["differential"])
col1, col2 = st.columns(2)
col1.metric("Rounds logged", len(rounds))
col2.metric("Estimated index", "-" if value is None else f"{value:.1f}")
st.caption(explanation + " Not an official WHS handicap index.")

# Average strokes lost per area across ALL your rounds. One round is
# noisy (a single bad hole can dominate it); averaging over many rounds
# shows your real pattern - this is the "learning your game" part.
all_breakdowns = pd.concat(
    [strokes_lost(row, bench) for _, row in rounds.iterrows()]
)
average_lost = (
    all_breakdowns[all_breakdowns["counted"]]
    .groupby("area", sort=False)["strokes_lost"]
    .mean()
)
st.subheader(f"Average strokes lost per round ({len(rounds)} rounds)")
st.bar_chart(average_lost, horizontal=True, sort=False)
if len(rounds) < 5:
    st.caption("Fewer than 5 rounds: treat this as a first impression, not a pattern.")

st.subheader("All rounds")
st.dataframe(
    rounds[["date", "course_name", "tee", "handicap_index", "score", "differential"]]
    .assign(differential=rounds["differential"].round(1)),
    hide_index=True,
)

# =====================================================================
# 4. Benchmarks by handicap (from Phase 7)
# =====================================================================
with st.expander("Average stats by handicap (Shot Scope 2021)"):
    table = bench.set_index("band_label")[STAT_COLUMNS]
    st.dataframe(table.T)
    st.line_chart(table[["fairways_pct", "gir_pct", "scramble_pct"]])
