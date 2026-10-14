# Shared helpers: what a logged round looks like, and how we check it.
#
# Phase 8 had saving/loading in here too; that now lives in storage.py, so
# this file only answers "what columns does a round have?" and "is this
# round possible?". Keeping the RULES separate from the FILE HANDLING means
# the import feature (storage.py) can reuse exactly the same checks as the
# form in the app.
#
# Rounds can be 18 or 9 holes (the `holes` column; blank = 18, so files from
# before it existed still load). The strokes-lost report and the estimated
# index only use 18-hole rounds: the benchmarks are "per 18-hole round", and
# the WHS rules for 9-hole scores haven't been provided from an official
# source yet. 9-hole rounds are still saved and listed.

import math
from datetime import date, datetime
from numbers import Number

import pandas as pd

import strokes_config as cfg

# Holes in a full round. The benchmarks (strokes_lost.py) are per 18 holes.
HOLES = 18
ALLOWED_HOLES = (9, 18)

# The columns of a saved round, in order. Grouped as:
#   - when / where:   date, course_name, tee
#   - course info:    from the scorecard, for this course and these tees
#   - you:            your handicap index on the day you played
#   - the round:      score and stats
#   - WHS extras:     optional, for a more accurate score differential
# (The file itself also has a schema_version column - that's about the FILE,
# not the round, so it lives in storage.py.)
ROUND_COLUMNS = [
    "date",
    "course_name",
    "tee",
    "course_rating",
    "slope_rating",
    "par",
    "holes",              # 18 or 9 (blank in older files = 18)
    "length_yards",       # optional; stored only, not used in any calculation
    "handicap_index",
    "score",              # gross score: every stroke you actually took
    "fairways_hit",
    "fairways_possible",  # par 4s and par 5s - par 3s have no fairway to hit
    "gir",                # greens in regulation
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
    "holes": HOLES,          # blank -> an 18-hole round (older files have no holes column)
    # Rating and slope may be blank if the scorecard doesn't show them. The
    # round is still saved and used in the strokes-lost report (which never
    # uses them), but it is NOT used for the estimated index or the
    # course-adjusted comparison - we never guess a rating. See
    # handicap.index_exclusion_reason().
    "course_rating": None,
    "slope_rating": None,
    "length_yards": None,
    "adjusted_score": None,  # blank -> differential uses the gross score
    # pcc stays BLANK when you don't know it, so "unknown" and "it was 0"
    # stay different in your records (you can look it up and fill it in
    # later). Calculations treat blank as 0 - see score_differential().
    "pcc": None,
}
REQUIRED_COLUMNS = [c for c in ROUND_COLUMNS if c not in OPTIONAL_DEFAULTS]

# Which columns hold text, which a date, and which numbers.
TEXT_COLUMNS = ["course_name", "tee"]
NUMBER_COLUMNS = [c for c in ROUND_COLUMNS if c not in TEXT_COLUMNS and c != "date"]

# Numbers that can only be whole (you can't take 85.5 strokes). Course
# rating and handicap index have decimals, so they're not in this list.
WHOLE_NUMBER_COLUMNS = [
    "slope_rating", "par", "holes", "length_yards", "score", "fairways_hit",
    "fairways_possible", "gir", "putts", "three_putts", "up_down_attempts",
    "up_down_saves", "penalties", "adjusted_score", "pcc",
]

# Numbers that can never be negative. (handicap_index CAN be: a plus
# handicap is stored as a negative number. pcc can be -1.)
NEVER_NEGATIVE_COLUMNS = [
    c for c in NUMBER_COLUMNS if c not in ("handicap_index", "pcc")
]

# Plain-English names, used in every error message so they match the
# labels you see in the app.
LABELS = {
    "date": "Date",
    "course_name": "Course name",
    "tee": "Tees played",
    "course_rating": "Course rating",
    "slope_rating": "Slope rating",
    "par": "Par",
    "holes": "Holes played",
    "length_yards": "Length",
    "handicap_index": "Handicap index",
    "score": "Score",
    "fairways_hit": "Fairways hit",
    "fairways_possible": "Fairways possible",
    "gir": "Greens in regulation",
    "putts": "Total putts",
    "three_putts": "3-putts",
    "up_down_attempts": "Up-and-down attempts",
    "up_down_saves": "Up-and-down saves",
    "penalties": "Penalty strokes",
    "adjusted_score": "Adjusted score",
    "pcc": "PCC",
}

# The typo-catching limits that depend on how many holes were played
# (course rating, par and score ranges) are judgement calls, so they live in
# strokes_config.py as SANITY_LIMITS, with every other judgement call.

# Dates in the CSV must be written like 2026-10-12. Why so strict? In
# Malaysia "12/10/2026" means 12 October, but a lot of software reads it as
# December 10. Accepting only one unambiguous format means a date can never
# be silently swapped.
DATE_FORMAT = "%Y-%m-%d"


def is_blank(value):
    """True for None, NaN (how pandas stores an empty CSV cell) or empty text."""
    if isinstance(value, str):
        return value.strip() == ""
    return value is None or pd.isna(value)


def parse_round(raw):
    """
    Turn one row of TEXT (as read from a CSV file) into a round with real
    numbers and a real date, ready for validate_round().

    `raw` is a dict like {"score": "92", "date": "2026-10-01", ...}.
    Returns (round_dict, problems). problems is a list of (field, message)
    for cells that can't be read at all, e.g. "abc" in the score column.

    Why not let pandas guess the types? If one cell in the score column
    holds "abc", pandas makes the WHOLE column text, and every comparison
    later (score > 200?) would crash. Reading cell by cell lets us point at
    the exact bad cell instead.
    """
    r, problems = {}, []

    # --- date ---
    text = str(raw.get("date", "")).strip()
    if text == "":
        r["date"] = None
    else:
        try:
            # strptime = "string parse time": read text using an exact format.
            r["date"] = datetime.strptime(text, DATE_FORMAT).date()
        except ValueError:
            r["date"] = None
            problems.append(("date", f"Date must be written as YYYY-MM-DD "
                                     f"(e.g. 2026-10-12), not '{text}'."))

    # --- text ---
    for col in TEXT_COLUMNS:
        r[col] = str(raw.get(col, "")).strip()

    # --- numbers ---
    for col in NUMBER_COLUMNS:
        text = str(raw.get(col, "")).strip()
        if text == "":
            r[col] = OPTIONAL_DEFAULTS.get(col)  # None for required columns
            continue
        try:
            value = float(text)
        except ValueError:
            value = None
        # float() happily reads "nan" and "inf", which aren't real numbers
        # of strokes, so treat them as unreadable too.
        if value is None or not math.isfinite(value):
            r[col] = None
            problems.append((col, f"{LABELS[col]} must be a number (found '{text}')."))
        else:
            r[col] = value

    return r, problems


def validate_round(r, today=None):
    """
    Check one round (a dict, or a pandas row, with the ROUND_COLUMNS keys)
    for impossible or out-of-range values.

    Returns a list of (field, message) pairs, at most ONE per field (the
    most basic problem), so the app can show each message right under the
    box it's about. An empty list means the round is OK to save.

    Catching mistakes here matters more than it looks: one typo (putts = 330
    instead of 33) saved to the CSV would quietly skew every average built
    from your rounds afterwards.

    `today` is only there so the self-check can pretend it's a different day;
    normally it's left out and means "today".
    """
    problems = []

    def add(field, message):
        # One message per field: if a field already has a problem, the
        # later (usually knock-on) ones would only add noise.
        if field not in {f for f, _ in problems}:
            problems.append((field, message))

    # --- 1. Missing values ---
    for col in REQUIRED_COLUMNS:
        if is_blank(r.get(col)):
            add(col, f"{LABELS[col]} is required.")
    if problems:
        return problems  # the checks below need these values

    # --- 2. Whole numbers, never negative ---
    for col in NUMBER_COLUMNS:
        value = r.get(col)
        if is_blank(value):
            continue  # an optional field left blank
        if col in NEVER_NEGATIVE_COLUMNS and value < 0:
            add(col, f"{LABELS[col]} can't be negative.")
        elif col in WHOLE_NUMBER_COLUMNS and value != int(value):
            add(col, f"{LABELS[col]} must be a whole number.")
    if problems:
        return problems  # the range checks below assume sensible numbers

    # --- 3. Date ---
    # pd.Timestamp(...).date() works whether the date arrived as a
    # datetime.date (from the form) or a pandas Timestamp (from the file).
    today = today or date.today()
    if pd.Timestamp(r["date"]).date() > today:
        add("date", "Date is in the future - you can only log a round you've played.")

    # --- 4. Holes ---
    holes = OPTIONAL_DEFAULTS["holes"] if is_blank(r.get("holes")) else int(r["holes"])
    if holes not in ALLOWED_HOLES:
        add("holes", "Holes played must be 9 or 18.")
        return problems  # every limit below depends on the number of holes
    limits = cfg.SANITY_LIMITS[holes]

    # --- 5. Course info ---
    # Slope 55-155 is the full range WHS allows. The rating and par limits
    # are generous sanity bounds (SANITY_LIMITS in strokes_config.py).
    # Rating and slope are optional, so they're only checked when filled in.
    if not is_blank(r.get("slope_rating")) and not 55 <= r["slope_rating"] <= 155:
        add("slope_rating", "Slope rating must be between 55 and 155.")
    low, high = limits["rating"]
    if not is_blank(r.get("course_rating")) and not low <= r["course_rating"] <= high:
        add("course_rating", f"Course rating must be between {low} and {high} "
                             f"for a {holes}-hole round.")
    low, high = limits["par"]
    if not low <= r["par"] <= high:
        add("par", f"Par must be between {low} and {high} for a {holes}-hole round.")
    if not is_blank(r.get("length_yards")) and r["length_yards"] == 0:
        add("length_yards", "Length must be more than 0 (or leave it blank).")

    # --- 6. The round itself ---
    score = r["score"]
    if score < r["par"] - limits["below_par"] or score > limits["max_score"]:
        add("score", f"Score looks wrong for {holes} holes on a par {r['par']:g} "
                     f"- please check it.")

    if r["fairways_possible"] > holes:
        add("fairways_possible", f"Fairways possible can't be more than {holes} "
                                 f"(the number of holes).")
    elif r["fairways_hit"] > r["fairways_possible"]:
        add("fairways_hit", f"Fairways hit ({r['fairways_hit']:g}) can't be more "
                            f"than fairways possible ({r['fairways_possible']:g}).")

    if r["gir"] > holes:
        add("gir", f"Greens in regulation can't be more than {holes} (one per hole).")

    # Every hole needs at least one stroke that ISN'T a putt (the tee shot),
    # so putts can be at most score - holes. This is stricter than "putts
    # can't be more than score", and still always true.
    if r["putts"] > score - holes:
        add("putts", f"Putts ({r['putts']:g}) can't be more than your score minus "
                     f"{holes}: every hole needs at least one shot that isn't a putt.")
    else:
        # A 3-putt uses at least 3 putts, and you can 3-putt each hole at
        # most once. So the most 3-putts possible is putts / 3 rounded DOWN
        # (// is whole-number division), and never more than the holes.
        most = min(int(r["putts"]) // 3, holes)
        if r["three_putts"] > most:
            add("three_putts", f"{r['three_putts']:g} 3-putts don't fit: each uses at "
                               f"least 3 putts, so {r['putts']:g} putts allow at most "
                               f"{most}.")

    # You can only try an up-and-down on a green you MISSED.
    if r["gir"] <= holes:
        missed_greens = holes - r["gir"]
        if r["up_down_attempts"] > missed_greens:
            add("up_down_attempts",
                f"Up-and-down attempts can be at most {missed_greens:g} "
                f"(you missed {missed_greens:g} of {holes} greens).")
    if r["up_down_saves"] > r["up_down_attempts"]:
        add("up_down_saves", f"Up-and-down saves ({r['up_down_saves']:g}) can't be "
                             f"more than attempts ({r['up_down_attempts']:g}).")

    # Penalty strokes are part of your score, and each hole still needs one
    # real shot, so penalties can be at most score - holes.
    if r["penalties"] > score - holes:
        add("penalties", f"Penalty strokes ({r['penalties']:g}) don't fit with a "
                         f"score of {score:g} over {holes} holes.")

    # Handicap index range allowed by WHS: +10 (written -10 here) to 54.
    if not -10 <= r["handicap_index"] <= 54:
        add("handicap_index", "Handicap index must be between +10 (enter -10) and 54.")

    # --- 7. Optional WHS extras ---
    # The net-double-bogey cap can only LOWER a hole score, never raise it,
    # so an adjusted score above the gross score must be a typo.
    adjusted = r.get("adjusted_score")
    if not is_blank(adjusted):
        if adjusted > score:
            add("adjusted_score", "Adjusted score can't be higher than your gross score.")
        elif adjusted < r["par"] - limits["below_par"]:
            add("adjusted_score", f"Adjusted score looks wrong for {holes} holes "
                                  f"- please check it.")

    # PCC only ever takes the values -1, 0, +1, +2 or +3 (USGA FAQ "What
    # is a Score Differential"), so a decimal like 1.5 must be a typo.
    pcc = r.get("pcc")
    if not is_blank(pcc) and pcc not in (-1, 0, 1, 2, 3):
        add("pcc", "PCC must be one of -1, 0, +1, +2 or +3.")

    return problems


def format_value(value):
    """
    One value as the text we write into the CSV:
    blank -> "", a date -> "2026-10-12", 72.0 -> "72", 71.3 -> "71.3".

    Writing 72 instead of 72.0 keeps the file tidy and means the same round
    always produces exactly the same text (handy for spotting duplicates).
    """
    if is_blank(value):
        return ""
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.strftime(DATE_FORMAT)
    if isinstance(value, Number):
        value = float(value)
        return str(int(value)) if value.is_integer() else str(value)
    return str(value).strip()


def full_rounds(rounds):
    """
    Only the 18-hole rounds from a table of saved rounds.

    The estimated index and the strokes-lost report use these alone: the
    benchmarks are "per 18-hole round", and the WHS rules for 9-hole scores
    haven't been checked against an official source yet. 9-hole rounds are
    still saved and listed in the history table.
    """
    return rounds[rounds["holes"] == HOLES]
