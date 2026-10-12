# Phase 8: Round tracker + "which part of my game needs fixing" report.
# (Phase 9 refined its maths - see 09_course_aware_checks.py. The testing
# phase added data safety, field-by-field errors and a first-run guide.)
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
# page, which keeps the maths testable without clicking through a web page
# (see selfcheck.py).
#
# How a Streamlit page works, briefly: the WHOLE script re-runs from top to
# bottom every time you click something. Anything that must survive between
# runs (an error message, a round waiting for "save anyway?") goes in
# st.session_state, a dictionary Streamlit keeps for your browser tab.

from datetime import date

import pandas as pd
import streamlit as st

from benchmarks import STAT_COLUMNS, load_benchmarks
from handicap import (
    ESTIMATE_LABEL,
    ESTIMATE_NOT_MODELLED,
    MIN_ROUNDS_FOR_INDEX,
    course_handicap,
    estimated_index,
    expected_scores,
    fit_overshoot,
    round_differential,
    round_half_up,
)
from rounds import HOLES, full_rounds, validate_round
from storage import (
    SchemaTooNewError,
    export_csv,
    find_duplicates,
    import_rounds,
    load_rounds,
    rounds_path,
    save_round,
)
from strokes_lost import (
    average_strokes_lost,
    biggest_leak,
    putting_confound,
    rank_areas,
    strokes_lost,
)

# Below this many 18-hole rounds, the weakness report says it can't say much.
FEW_ROUNDS = 3

# Every form box has a key (its name in st.session_state) equal to the
# round column it fills. That lets us (a) show an error under the right box
# and (b) clear the form after a successful save.
FORM_KEYS = [
    "date", "course_name", "tee", "handicap_index", "holes", "course_rating",
    "slope_rating", "par", "length_yards", "score", "fairways_hit",
    "fairways_possible", "gir", "putts", "three_putts", "up_down_attempts",
    "up_down_saves", "penalties", "adjusted_score", "pcc",
]


# Benchmarks never change while the app runs, so load them (and fit the
# overshoot line) once. See the caching notes in app.py.
@st.cache_data
def get_benchmarks():
    bench = load_benchmarks()
    return bench, fit_overshoot(bench)


def field_error(container, field):
    """Show the saved error for this field (if any) right under its box."""
    message = st.session_state.get("form_errors", {}).get(field)
    if message:
        container.error(message)


def save_and_rerun(new_round):
    """
    Save a round, remember the outcome, and re-run the page so everything
    below (reports, index) includes it.
    """
    try:
        problems = save_round(new_round)
    except (OSError, SchemaTooNewError) as e:
        # Atomic writes mean a failed save leaves the old file as it was.
        st.session_state["flash"] = ("error", f"Round NOT saved: {e}. "
                                              f"Your rounds file was not changed.")
        st.rerun()

    if problems:  # shouldn't happen (we validated first), but never lose them
        st.session_state["form_errors"] = dict(problems)
    else:
        st.session_state["flash"] = ("success", "Round saved.")
        st.session_state["clear_form"] = True
        st.session_state.pop("form_errors", None)
    st.session_state.pop("pending_round", None)
    st.rerun()


def show_stat_guide():
    """What each stat means - shown in the first-run guide and on request."""
    st.markdown(
        "- **Fairways hit / possible**: count **par 4s and par 5s only** (a par 3 "
        "has no fairway). Possible = how many par 4s and 5s you played; hit = how "
        "many tee shots finished on the fairway.\n"
        "- **Greens in regulation (GIR)**: your ball is on the green with two "
        "strokes to spare for par: in 1 shot on a par 3, 2 on a par 4, 3 on a par 5.\n"
        "- **Total putts**: strokes taken **on the green** only. A putter from the "
        "fringe doesn't count.\n"
        "- **3-putts**: holes where you took 3 or more putts.\n"
        "- **Up-and-down attempts / saves**: an attempt is a hole where you "
        "**missed the green** and tried to get the ball in the hole in two shots "
        "from off the green (e.g. a chip and one putt). A save is when you did it. "
        "(That's this app's definition; the benchmark's own isn't published - see "
        "Known limitations.)\n"
        "- **Penalty strokes**: only the penalty strokes you added (e.g. 1 for a "
        "ball in a penalty area), not the shots themselves.\n"
        "- **Course rating / slope / par**: printed on the scorecard for the tees "
        "you played."
    )


bench, overshoot_fit = get_benchmarks()

# --- Things left over from the previous run (see the note at the top) ---
# Clearing the form must happen BEFORE the boxes are drawn: deleting a
# box's key resets it to its default value.
if st.session_state.pop("clear_form", False):
    for key in FORM_KEYS:
        st.session_state.pop(key, None)

st.title("My Golf Round Tracker")
st.write(
    "Log your rounds to see where you're losing strokes compared with a typical "
    "golfer at your handicap index. The more rounds you add, the clearer the "
    "pattern gets."
)

# Load first, so the page knows whether this is your first visit. If the
# file is from a NEWER version of the tracker, stop: guessing at a format
# we don't understand is how data gets corrupted.
try:
    rounds, skipped = load_rounds()
except SchemaTooNewError as e:
    st.error(str(e))
    st.stop()

st.caption(f"Your rounds are saved on this computer only, in `{rounds_path()}`. "
           f"The app runs locally and never uploads your rounds anywhere.")

# Rows that failed the checks are NOT used below, but they're still in the
# file - we never delete or rewrite them. Show them so you can fix them.
if skipped:
    st.warning(
        f"{len(skipped)} row(s) in your rounds file have problems, so they're left "
        f"out of everything below. They have NOT been deleted - they're still in "
        f"the file. Fix them there (or delete the row and log the round again)."
    )
    with st.expander("Rows with problems"):
        for item in skipped:
            st.markdown(f"**Row {item['row']}**: " + " ".join(item["problems"]))
        st.dataframe(pd.DataFrame([item["values"] for item in skipped],
                                  index=[f"row {item['row']}" for item in skipped]))

# Known limitations: always available, whatever state the app is in.
with st.expander("Known limitations"):
    st.markdown(
        f"- **The handicap index is {ESTIMATE_LABEL}.** Not modelled: "
        + "; ".join(ESTIMATE_NOT_MODELLED) + ".\n"
        "- **Per-stat gaps are not course-adjusted.** Fairways, greens, putts, "
        "short game and penalties are compared with a typical golfer at your index "
        "on an average course. On a hard course every area looks worse.\n"
        "- **Benchmark age and bias.** The benchmarks come from one source (Shot "
        "Scope Strokes Gained eBook, 4th ed, 2021). It predates the World Handicap "
        "System, and it's based on golfers who track their rounds with Shot Scope, "
        "who may not be typical of all golfers.\n"
        "- **Low-confidence stroke conversions.** Turning a gap into strokes is "
        "solid for penalties, reasonable for short game, and rough for approach "
        "and putting (see the confidence column). Driving is shown as a gap only.\n"
        "- **Putting is confounded by GIR**: missing greens means chipping close "
        "and taking fewer putts, so putting can look better than it is.\n"
        "- **Unresolved #5:** the benchmark source doesn't define its Up & Down % "
        "(what counts as an attempt) or whether its fairway % leaves out par 3s. "
        "This app counts par 4s and 5s only, so the fairway and short-game gaps may "
        "not compare like with like.\n"
        "- **Unresolved #6:** the source doesn't say whether its \"handicap\" is a "
        "WHS Handicap Index (its data predates WHS). The app treats it as roughly "
        "your index.\n"
        "- **9-hole rounds** are saved and listed, but left out of the index and the "
        "strokes-lost report (the benchmarks are per 18 holes).\n"
        "- **Tour reference**: the PGA Tour model (the separate `app.py`) is only "
        "meaningful inside its training range, which almost no amateur is in. This "
        "tracker never uses it."
    )

# --- First-run guide ---
if len(rounds) == 0 and not skipped:
    st.info(
        "**How to log your first round**\n\n"
        "1. Have your scorecard handy: you need the course rating, slope rating and "
        "par for the tees you played.\n"
        "2. Fill in the form below. Every box is required except the ones marked "
        "optional.\n"
        "3. Press **Save round**. If something doesn't add up, a message appears "
        "under that box and nothing is saved.\n\n"
        "The estimated handicap index appears once you have 3 rounds of 18 holes."
    )
    show_stat_guide()
else:
    with st.expander("What each stat means"):
        show_stat_guide()

# =====================================================================
# 1. Log a round
# =====================================================================
# One-off message from the last run ("Round saved." etc.).
flash = st.session_state.pop("flash", None)
if flash:
    kind, message = flash
    {"success": st.success, "error": st.error, "info": st.info}[kind](message)

# st.form groups the inputs so the page DOESN'T rerun on every keystroke -
# nothing happens until you press "Save round". value=None leaves a box
# empty instead of pre-filling a made-up number, so you can't accidentally
# save a fake round. clear_on_submit is OFF, so if a box has a problem your
# other entries stay put; the form is cleared only after a successful save.
with st.form("log_round"):
    st.subheader("Log a round")
    errors = st.session_state.get("form_errors", {})
    if errors:
        st.error(f"Round not saved - {len(errors)} thing(s) to fix, shown under "
                 f"each box below.")

    col1, col2 = st.columns(2)
    date_played = col1.date_input("Date", key="date")
    field_error(col1, "date")
    course_name = col2.text_input("Course name", key="course_name")
    field_error(col2, "course_name")
    tee = col1.text_input("Tees played (e.g. White)", key="tee")
    field_error(col1, "tee")
    handicap_index = col2.number_input(
        "Your handicap index today (plus handicap: enter as negative)",
        value=None, step=0.1, format="%.1f", key="handicap_index",
    )
    field_error(col2, "handicap_index")
    holes = col1.radio("Holes played", [18, 9], horizontal=True, key="holes",
                       help="9-hole rounds are saved, but not counted in the "
                            "index or the strokes-lost report.")
    field_error(col1, "holes")

    st.caption("From the scorecard, for the tees you played:")
    col1, col2, col3, col4 = st.columns(4)
    course_rating = col1.number_input("Course rating", value=None, step=0.1,
                                      format="%.1f", key="course_rating")
    field_error(col1, "course_rating")
    slope_rating = col2.number_input("Slope rating", value=None, step=1, key="slope_rating")
    field_error(col2, "slope_rating")
    par = col3.number_input("Par", value=None, step=1, key="par")
    field_error(col3, "par")
    length_yards = col4.number_input("Length, yds (optional)", value=None, step=1,
                                     key="length_yards")
    field_error(col4, "length_yards")

    st.caption("Your round:")
    col1, col2, col3 = st.columns(3)
    score = col1.number_input("Score (gross)", value=None, step=1, key="score")
    field_error(col1, "score")
    fairways_hit = col2.number_input("Fairways hit", value=None, step=1, key="fairways_hit")
    field_error(col2, "fairways_hit")
    fairways_possible = col3.number_input(
        "Fairways possible (par 4s + 5s)", value=None, step=1, key="fairways_possible",
    )
    field_error(col3, "fairways_possible")
    gir = col1.number_input("Greens in regulation", value=None, step=1, key="gir")
    field_error(col1, "gir")
    # Same definition as the benchmark (Shot Scope) and the Tour: a stroke
    # with the putter from the fringe is NOT a putt.
    putts = col2.number_input("Total putts", value=None, step=1, key="putts",
                              help="Only strokes taken on the green. "
                                   "A putter from the fringe doesn't count.")
    field_error(col2, "putts")
    three_putts = col3.number_input("3-putts", value=None, step=1, key="three_putts")
    field_error(col3, "three_putts")
    up_down_attempts = col1.number_input("Up-and-down attempts", value=None, step=1,
                                         key="up_down_attempts")
    field_error(col1, "up_down_attempts")
    up_down_saves = col2.number_input("Up-and-down saves", value=None, step=1,
                                      key="up_down_saves")
    field_error(col2, "up_down_saves")
    penalties = col3.number_input("Penalty strokes", value=None, step=1, key="penalties")
    field_error(col3, "penalties")

    st.caption("Optional, for a more accurate differential (from your handicap app):")
    col1, col2 = st.columns(2)
    adjusted_score = col1.number_input(
        "Adjusted score (after net double bogey cap)", value=None, step=1,
        key="adjusted_score", help="Leave blank to use your gross score.",
    )
    field_error(col1, "adjusted_score")
    pcc = col2.number_input(
        "PCC (playing conditions: -1, 0, +1, +2 or +3)", value=None, step=1, key="pcc",
        help="Leave blank if you don't know it - it's saved as unknown and "
             "counted as 0 in the differential. Fill it in later if you find it.",
    )
    field_error(col2, "pcc")

    submitted = st.form_submit_button("Save round", key="save_round")

if submitted:
    new_round = {
        "date": date_played,
        "course_name": course_name,
        "tee": tee,
        "course_rating": course_rating,
        "slope_rating": slope_rating,
        "par": par,
        "holes": holes,
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
    problems = validate_round(new_round)
    st.session_state.pop("pending_round", None)
    if problems:
        # dict(): [("score", "Score is required."), ...] -> {"score": "..."}
        # Re-run so the messages appear under their boxes (the form has
        # already been drawn in this run).
        st.session_state["form_errors"] = dict(problems)
        st.rerun()
    st.session_state.pop("form_errors", None)
    if len(find_duplicates(new_round, rounds)) > 0:
        # Probably the same round twice - ask before saving.
        st.session_state["pending_round"] = new_round
        st.rerun()
    save_and_rerun(new_round)

# --- "Save anyway?" for a possible duplicate ---
pending = st.session_state.get("pending_round")
if pending:
    st.warning(
        f"You already have a round on {pending['date']:%Y-%m-%d} at "
        f"**{pending['course_name']}** ({pending['tee']} tees). Is this a second "
        f"round that day, or the same round entered twice?"
    )
    col1, col2 = st.columns(2)
    if col1.button("It's a different round - save it", key="confirm_duplicate"):
        save_and_rerun(pending)
    if col2.button("Same round - don't save", key="cancel_duplicate"):
        st.session_state.pop("pending_round")
        st.session_state["flash"] = ("info", "Not saved.")
        st.rerun()

# --- Export / Import ---
with st.expander("Export / Import rounds (CSV)"):
    # Export: hands you an exact copy of your rounds file - a backup, or a
    # way to move your rounds to another computer.
    st.download_button("Download my rounds", data=export_csv(),
                       file_name="my_rounds.csv", mime="text/csv", key="export")

    # Import: the file is only read when you press the button. Without the
    # button, Streamlit would re-import on every rerun of the page while
    # the file sits in the uploader.
    uploaded = st.file_uploader("Import rounds from a CSV", type="csv", key="import_file")
    st.caption("Same columns as data/my_rounds.example.csv. Every row is checked; "
               "if any row is invalid, nothing is imported. Exact duplicates are "
               "skipped. A backup of your file is made before anything is added.")
    if uploaded is not None and st.button("Import", key="import"):
        try:
            added, problems, warnings = import_rounds(uploaded)
        except (OSError, SchemaTooNewError) as e:
            added, problems, warnings = 0, [f"{e}. Your rounds file was not changed."], []
        if problems:
            st.error("Nothing imported:\n\n" + "\n".join(f"- {p}" for p in problems))
        else:
            st.success(f"Imported {added} new round(s). Scroll down to see them.")
            for w in warnings:
                st.warning(w)
            # Reload so the reports below include the imported rounds.
            rounds, skipped = load_rounds()

if len(rounds) == 0:
    st.stop()  # nothing more to show until there's at least one round

# Only 18-hole rounds can be compared with the (per-18-hole) benchmarks.
full = full_rounds(rounds)
n_full = len(full)

# The putting/GIR check is judged on ALL your rounds, never just one (a
# single round's GIR is too noisy), so it's worked out once up here.
confounded, confound_message = putting_confound(full, bench)

# =====================================================================
# 2. Round report (18-hole rounds only)
# =====================================================================
st.header("Round report")

if n_full == 0:
    st.info("Round reports need an 18-hole round. Your 9-hole rounds are listed "
            "under All rounds below.")
else:
    # A readable label per round for the dropdown. Newest first, so the
    # default selection is the round you just logged.
    labels = [
        f"{row.date:%Y-%m-%d} - {row.course_name} ({row.tee}) - {row.score:g}"
        for row in full.itertuples()
    ]
    choice = st.selectbox("Round", options=list(reversed(range(n_full))),
                          format_func=lambda i: labels[i])
    r = full.iloc[choice]

    # --- a) Course-adjusted ---
    expected = expected_scores(r["handicap_index"], r["course_rating"],
                               r["slope_rating"], overshoot_fit)
    diff = round_differential(r)
    ch = course_handicap(r["handicap_index"], r["slope_rating"], r["course_rating"],
                         r["par"])

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
        "index x slope/113 + (rating - par) (WHS Rule 6.1a)."
    )

    # --- b) Per-area breakdown (NOT course-adjusted) ---
    st.subheader("Where the strokes went (not course-adjusted)")
    st.caption("Based on 1 round: the one selected above.")
    if n_full < FEW_ROUNDS:
        st.info(f"Too few rounds to say much yet: you have {n_full} 18-hole round(s). "
                f"One round is noisy - a single bad hole can top this list. Wait for "
                f"at least {FEW_ROUNDS} before reading anything into it.")
    breakdown = strokes_lost(r, bench)
    ranked = rank_areas(breakdown, confounded)

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
    driving = breakdown.iloc[0]
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
col1.metric("Rounds logged", len(rounds),
            help=f"{n_full} of 18 holes, {len(rounds) - n_full} of 9 holes.")
col2.metric(f"Handicap index ({ESTIMATE_LABEL})",
            f"need {MIN_ROUNDS_FOR_INDEX} rounds" if value is None else f"{value:.1f}")
st.markdown(f"**{ESTIMATE_LABEL}.** This is not your official handicap index.")
st.caption(
    f"{explanation} WHS Rule 5.2a table, newest 20 18-hole rounds. "
    "Not modelled: " + "; ".join(ESTIMATE_NOT_MODELLED) + "."
)

if n_full > 0:
    # Average over many rounds = your real pattern (one round is noisy).
    average_ranked = rank_areas(average_strokes_lost(full, bench), confounded)
    st.subheader(f"Average strokes lost per round ({n_full} rounds)")
    st.caption(f"Based on {n_full} 18-hole round(s)"
               + (f"; {len(rounds) - n_full} 9-hole round(s) not included."
                  if len(rounds) > n_full else "."))
    if n_full < FEW_ROUNDS:
        st.info(f"Too few rounds to say much yet ({n_full} of the {FEW_ROUNDS} "
                f"needed for a first look).")
    elif n_full < 5:
        st.caption("Fewer than 5 rounds: treat this as a first impression, not a pattern.")
    st.bar_chart(average_ranked.set_index("area")["strokes_lost"],
                 horizontal=True, sort=False)
    if confounded:
        st.caption("Putting: confounded by GIR - ranked below penalties and short game.")

st.subheader("All rounds")
history = rounds[["date", "course_name", "tee", "holes", "handicap_index", "score",
                  "adjusted_score", "pcc"]].copy()
# A differential only means something for an 18-hole round here.
history["differential"] = [
    round_half_up(round_differential(row), 1) if row["holes"] == HOLES else None
    for _, row in rounds.iterrows()
]
st.dataframe(history, hide_index=True)

# =====================================================================
# 4. Benchmarks by handicap (from Phase 7)
# =====================================================================
with st.expander("Average stats by handicap (Shot Scope 2021)"):
    table = bench.set_index("band_label")[STAT_COLUMNS]
    st.dataframe(table.T)
    st.line_chart(table[["fairways_pct", "gir_pct", "scramble_pct"]])
