# Shared helpers: what a logged round looks like, and how we check it.
#
# Phase 8 had saving/loading in here too; that now lives in storage.py, so
# this file only answers "what columns does a round have?" and "is this
# round possible?". Keeping the RULES separate from the FILE HANDLING means
# the import feature (storage.py) can reuse exactly the same checks as the
# form in the app.
#
# 18-hole rounds only for now: the per-hole limits below (e.g. at most 18
# greens) assume 18 holes, and there's no "holes" column yet.

import pandas as pd

HOLES = 18

# The columns of a saved round, in order. Grouped as:
#   - when / where:   date, course_name, tee
#   - course info:    from the scorecard, for this course and these tees
#   - you:            your handicap index on the day you played
#   - the round:      score and stats
#   - WHS extras:     optional, for a more accurate score differential
ROUND_COLUMNS = [
    "date",
    "course_name",
    "tee",
    "course_rating",
    "slope_rating",
    "par",
    "length_yards",       # optional; stored only, not used in any calculation
    "handicap_index",
    "score",              # gross score: every stroke you actually took
    "fairways_hit",
    "fairways_possible",  # par 4s and par 5s - par 3s have no fairway to hit
    "gir",                # greens in regulation, out of 18
    "putts",
    "three_putts",        # logged, but not scored yet (no benchmark - see strokes_lost.py)
    "up_down_attempts",   # missed greens where you tried to get up and down
    "up_down_saves",      # ...and how many of those you did
    "penalties",
    "adjusted_score",     # optional: score after the WHS net-double-bogey cap
    "pcc",                # optional: WHS Playing Conditions Calculation (blank = unknown)
]

# Columns you may leave blank, and what a blank means. None = "stays
# blank"; a number = "blank means this value".
OPTIONAL_DEFAULTS = {
    "length_yards": None,
    "adjusted_score": None,  # blank -> differential uses the gross score
    # pcc stays BLANK when you don't know it, so "unknown" and "it was 0"
    # stay different in your records (you can look it up and fill it in
    # later). Calculations treat blank as 0 - see score_differential().
    "pcc": None,
}


def is_blank(value):
    """True for None, NaN (how pandas stores an empty CSV cell) or empty text."""
    if isinstance(value, str):
        return value.strip() == ""
    return value is None or pd.isna(value)


def add_missing_columns(rounds):
    """
    Make an older rounds table fit the current schema.

    A CSV saved before adjusted_score and pcc existed simply doesn't have
    those columns. Instead of failing, we add them, filled with their
    default (blank). Then every later step can assume all
    ROUND_COLUMNS are there.
    """
    rounds = rounds.copy()
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in rounds.columns:
            rounds[col] = default
        elif default is not None:
            # Column exists but some cells are blank: fill those too.
            rounds[col] = rounds[col].fillna(default)
    return rounds


def validate_round(r):
    """
    Check one round (a dict, or a pandas row, with the ROUND_COLUMNS keys)
    for impossible or out-of-range values. Returns a list of plain-English
    problems; an empty list means the round is OK to save.

    Catching mistakes here matters more than it looks: one typo (putts = 330
    instead of 33) saved to the CSV would quietly skew every average built
    from your rounds afterwards.
    """
    problems = []

    # --- Missing values ---
    for col in ROUND_COLUMNS:
        if col in OPTIONAL_DEFAULTS:
            continue
        if is_blank(r.get(col)):
            problems.append(f"{col} is required.")
    if problems:
        return problems  # the range checks below need these values

    # --- Course info ranges ---
    # Slope 55-155 is the full range WHS allows. The rating and par limits
    # are generous sanity bounds for 18-hole courses.
    if not 55 <= r["slope_rating"] <= 155:
        problems.append("Slope rating must be between 55 and 155.")
    if not 60 <= r["course_rating"] <= 80:
        problems.append("Course rating must be between 60 and 80.")
    if not 60 <= r["par"] <= 75:
        problems.append("Par must be between 60 and 75.")
    if not is_blank(r.get("length_yards")) and r["length_yards"] <= 0:
        problems.append("Length must be positive (or leave it blank).")

    # --- The round itself ---
    if r["score"] < r["par"] - 15 or r["score"] > 200:
        problems.append("Score looks wrong for 18 holes - please check it.")

    if not 0 <= r["fairways_possible"] <= HOLES:
        problems.append(f"Fairways possible must be 0-{HOLES}.")
    if not 0 <= r["fairways_hit"] <= r["fairways_possible"]:
        problems.append("Fairways hit can't be more than fairways possible.")

    if not 0 <= r["gir"] <= HOLES:
        problems.append(f"Greens in regulation must be 0-{HOLES}.")

    if r["putts"] < 0:
        problems.append("Putts can't be negative.")
    # Every 3-putt uses 3 of your putts, so 3-putts x 3 can't exceed putts.
    if r["three_putts"] < 0 or r["three_putts"] * 3 > r["putts"]:
        problems.append("3-putts don't fit with total putts (each 3-putt is 3 putts).")

    # You can only try an up-and-down on a green you MISSED.
    missed_greens = HOLES - r["gir"]
    if not 0 <= r["up_down_attempts"] <= missed_greens:
        problems.append(
            f"Up-and-down attempts must be 0-{missed_greens} "
            f"(you missed {missed_greens} greens)."
        )
    if not 0 <= r["up_down_saves"] <= r["up_down_attempts"]:
        problems.append("Up-and-down saves can't be more than attempts.")

    if r["penalties"] < 0:
        problems.append("Penalties can't be negative.")

    # Handicap index range allowed by WHS: +10 (written -10 here) to 54.
    if not -10 <= r["handicap_index"] <= 54:
        problems.append("Handicap index must be between +10 (enter -10) and 54.")

    # --- Optional WHS extras ---
    # The net-double-bogey cap can only LOWER a hole score, never raise it,
    # so an adjusted score above the gross score must be a typo.
    adjusted = r.get("adjusted_score")
    if not is_blank(adjusted):
        if adjusted > r["score"]:
            problems.append("Adjusted score can't be higher than your gross score.")
        if adjusted < r["par"] - 15:
            problems.append("Adjusted score looks wrong for 18 holes - please check it.")

    # PCC only ever takes the values -1, 0, +1, +2 or +3 (USGA FAQ "What
    # is a Score Differential"), so a decimal like 1.5 must be a typo.
    pcc = r.get("pcc")
    if not is_blank(pcc) and pcc not in (-1, 0, 1, 2, 3):
        problems.append("PCC must be one of -1, 0, +1, +2 or +3.")

    return problems
