# Phase 8: Round tracker + "which part of my game needs fixing" report.
# (Phase 9 refined its maths - see 09_course_aware_checks.py.)
#
# Run with:   streamlit run 08_round_tracker.py
#
# This is a separate Streamlit app from app.py (the PGA predictor). The two
# answer different questions, and keeping them apart follows the Phase 7
# rule that the PGA model is never used for amateur predictions.
#
# What it does:
#   1. Log a round: course info from the scorecard + your stats. Saved to
#      data/my_rounds.csv (gitignored - personal data). Export / Import CSV.
#   2. Round report, in two separate parts:
#        a) COURSE-ADJUSTED: your score vs two expected scores on THIS
#           course from THESE tees (headline + a softer "typical round").
#        b) NOT course-adjusted: each part of your game vs a typical golfer
#           at your index, converted to strokes lost and ranked.
#   3. Your game over time: ESTIMATED index and average strokes lost per area.
#   4. The benchmark table from Phase 7.
#
# The logic lives in the shared helpers (rounds.py, storage.py,
# handicap.py, strokes_lost.py, benchmarks.py). This file only builds the
# page, which keeps the maths testable without clicking through a web page.

import pandas as pd
import streamlit as st

from benchmarks import STAT_COLUMNS, load_benchmarks
from handicap import (
    ESTIMATE_LABEL,
    ESTIMATE_NOT_MODELLED,
    course_handicap,
    estimated_index,
    expected_scores,
    fit_overshoot,
    round_differential,
    round_half_up,
)
from storage import export_csv, import_rounds, load_rounds, save_round
from strokes_lost import biggest_leak, putting_confound, rank_areas, strokes_lost


# Benchmarks never change while the app runs, so load them (and fit the
# overshoot line) once. See the caching notes in app.py.
@st.cache_data
def get_benchmarks():
    bench = load_benchmarks()
    return bench, fit_overshoot(bench)


bench, overshoot_fit = get_benchmarks()

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
    score = col1.number_input("Score (gross)", value=None, step=1)
    fairways_hit = col2.number_input("Fairways hit", value=None, step=1)
    fairways_possible = col3.number_input(
        "Fairways possible (par 4s + 5s)", value=None, step=1
    )
    gir = col1.number_input("Greens in regulation", value=None, step=1)
    # Same definition as the benchmark (Shot Scope) and the Tour: a stroke
    # with the putter from the fringe is NOT a putt.
    putts = col2.number_input("Total putts", value=None, step=1,
                              help="Only strokes taken on the green. "
                                   "A putter from the fringe doesn't count.")
    three_putts = col3.number_input("3-putts", value=None, step=1)
    up_down_attempts = col1.number_input("Up-and-down attempts", value=None, step=1)
    up_down_saves = col2.number_input("Up-and-down saves", value=None, step=1)
    penalties = col3.number_input("Penalty strokes", value=None, step=1)

    st.caption("Optional, for a more accurate differential (from your handicap app):")
    col1, col2 = st.columns(2)
    adjusted_score = col1.number_input(
        "Adjusted score (after net double bogey cap)", value=None, step=1,
        help="Leave blank to use your gross score.",
    )
    pcc = col2.number_input(
        "PCC (playing conditions: -1, 0, +1, +2 or +3)", value=None, step=1,
        help="Leave blank if you don't know it - it's saved as unknown and "
             "counted as 0 in the differential. Fill it in later if you find it.",
    )

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
        "adjusted_score": adjusted_score,
        "pcc": pcc,
    }
    problems = save_round(new_round)
    if problems:
        st.error("Round not saved:\n\n" + "\n".join(f"- {p}" for p in problems))
    else:
        st.success("Round saved.")

# --- Export / Import ---
with st.expander("Export / Import rounds (CSV)"):
    # Export: hands you a copy of data/my_rounds.csv - a backup, or a way
    # to move your rounds to another computer.
    st.download_button("Download my rounds", data=export_csv(),
                       file_name="my_rounds.csv", mime="text/csv")

    # Import: the file is only read when you press the button. Without the
    # button, Streamlit would re-import on every rerun of the page while
    # the file sits in the uploader.
    uploaded = st.file_uploader("Import rounds from a CSV", type="csv")
    st.caption("Same columns as data/my_rounds.example.csv. Every row is checked; "
               "if any row is invalid, nothing is imported. Duplicates are skipped.")
    if uploaded is not None and st.button("Import"):
        added, problems = import_rounds(uploaded)
        if problems:
            st.error("Nothing imported:\n\n" + "\n".join(f"- {p}" for p in problems))
        else:
            st.success(f"Imported {added} new round(s).")

# Loaded AFTER the form and import, so new rounds are included below.
rounds = load_rounds()

if len(rounds) == 0:
    st.info("No rounds logged yet. Add your first round above.")
    st.stop()  # nothing more to show until there's at least one round

# The putting/GIR check is judged on ALL your rounds, never just one (a
# single round's GIR is too noisy), so it's worked out once up here.
confounded, confound_message = putting_confound(rounds, bench)

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
expected = expected_scores(r["handicap_index"], r["course_rating"], r["slope_rating"],
                           overshoot_fit)
diff = round_differential(r)
ch = course_handicap(r["handicap_index"], r["slope_rating"], r["course_rating"], r["par"])

st.subheader("Strokes vs expected on this course (course-adjusted)")
col1, col2, col3 = st.columns(3)
# delta_color="inverse": in golf a NEGATIVE delta (fewer strokes) is good,
# so we tell Streamlit to show negative in green and positive in red.
col1.metric("Plays to handicap (headline)", f"{expected['plays_to_handicap']:.1f}",
            delta=f"{r['score'] - expected['plays_to_handicap']:+.1f} your score vs this",
            delta_color="inverse")
col2.metric("Typical round at your handicap (roughly +/-2)",
            f"{expected['typical_round']:.1f}",
            delta=f"{r['score'] - expected['typical_round']:+.1f} your score vs this",
            delta_color="inverse")
col3.metric("Your score", f"{r['score']:g}")

col1, col2 = st.columns(2)
col1.metric("Course handicap", round_half_up(ch), help=f"Unrounded: {ch:.2f}")
col2.metric("Score differential", f"{round_half_up(diff, 1):.1f}")
st.caption(
    "**Plays to handicap** = course rating + index x slope / 113: what you'd "
    "shoot playing exactly to your handicap. Your index comes from your best "
    "rounds, so most rounds land above it. **Typical round** adds how far "
    "above their handicap golfers usually score (a line fitted to the "
    "benchmark data). It's softer: that benchmark figure already includes "
    "some of the effect of real courses being harder than the standard slope "
    "of 113, so adding it on top of the slope term double counts a little, "
    "and the benchmark courses' ratings aren't known. Course handicap uses "
    "index x slope/113 + (rating - par); check your association uses that "
    "version."
)

# --- b) Per-area breakdown (NOT course-adjusted) ---
st.subheader("Where the strokes went (not course-adjusted)")
ranked = rank_areas(strokes_lost(r, bench), confounded)

leak = biggest_leak(ranked)
if leak is not None:
    flag = f" ({leak['flag']})" if leak["flag"] else ""
    st.warning(f"Biggest leak this round: **{leak['area']}**{flag}, "
               f"about {leak['strokes_lost']:.1f} strokes vs a typical "
               f"{r['handicap_index']:g} handicap.")
else:
    st.success("You beat a typical golfer at your index in every area this round.")

st.dataframe(ranked.assign(strokes_lost=ranked["strokes_lost"].round(2)),
             hide_index=True)
# Driving is shown as a gap only (its cost is 0), so it's listed separately.
driving = strokes_lost(r, bench).iloc[0]
st.caption(
    f"Driving: {driving['yours']} fairways vs typical {driving['typical']} "
    f"(shown only; not converted to strokes). 3-putts: logged, but not scored "
    f"- the benchmark file has no 3-putt data yet."
)
st.caption(
    "Ranked most to fewest strokes lost vs a typical golfer at your handicap "
    "index (Shot Scope 2021 benchmarks). Positive = lost, negative = gained. "
    "NOT adjusted for course difficulty: on a hard course every area looks worse. "
    + confound_message
    + (" Putting is flagged and ranked below penalties and short game."
       if confounded else "")
)

# Bar chart in ranked order. sort=False keeps our order.
st.bar_chart(ranked.set_index("area")["strokes_lost"], horizontal=True, sort=False)

# =====================================================================
# 3. Your game over time
# =====================================================================
st.header("Your game over time")

value, explanation = estimated_index(rounds)
col1, col2 = st.columns(2)
col1.metric("Rounds logged", len(rounds))
col2.metric(f"Handicap index ({ESTIMATE_LABEL})",
            "need 3 rounds" if value is None else f"{value:.1f}")
st.caption(
    f"{ESTIMATE_LABEL}. {explanation} WHS Rule 5.2a table, newest 20 rounds. "
    "Not modelled: " + "; ".join(ESTIMATE_NOT_MODELLED) + "."
)

# Average strokes lost per area across ALL your rounds. One round is
# noisy (a single bad hole can dominate it); averaging over many rounds
# shows your real pattern - this is the "learning your game" part.
all_breakdowns = pd.concat(
    [strokes_lost(row, bench) for _, row in rounds.iterrows()]
)
average = (
    all_breakdowns[all_breakdowns["counted"]]
    .groupby("area", sort=False)
    .agg(strokes_lost=("strokes_lost", "mean"))
    .reset_index()
)
average["counted"] = True
average_ranked = rank_areas(average, confounded)

st.subheader(f"Average strokes lost per round ({len(rounds)} rounds)")
st.bar_chart(average_ranked.set_index("area")["strokes_lost"],
             horizontal=True, sort=False)
if confounded:
    st.caption("Putting: confounded by GIR - ranked below penalties and short game.")
if len(rounds) < 5:
    st.caption("Fewer than 5 rounds: treat this as a first impression, not a pattern.")

st.subheader("All rounds")
history = rounds[["date", "course_name", "tee", "handicap_index", "score",
                  "adjusted_score", "pcc"]].copy()
history["differential"] = [round_half_up(round_differential(row), 1)
                           for _, row in rounds.iterrows()]
st.dataframe(history, hide_index=True)

# =====================================================================
# 4. Benchmarks by handicap (from Phase 7)
# =====================================================================
with st.expander("Average stats by handicap (Shot Scope 2021)"):
    table = bench.set_index("band_label")[STAT_COLUMNS]
    st.dataframe(table.T)
    st.line_chart(table[["fairways_pct", "gir_pct", "scramble_pct"]])
