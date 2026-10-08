# Shared helpers for YOUR logged rounds (Phase 8).
#
# Every round you enter is saved as one row in data/my_rounds.csv. That
# file is gitignored on purpose: it's personal data and shouldn't end up on
# GitHub. A plain CSV keeps it simple, and you can open it in any
# spreadsheet app to check or fix a typo.
#
# 18-hole rounds only for now: the per-hole limits below (e.g. at most 18
# greens) assume 18 holes.

import os

import pandas as pd

ROUNDS_PATH = "data/my_rounds.csv"
HOLES = 18

# The columns of my_rounds.csv, in order. Grouped as:
#   - when / where:   date, course_name, tee
#   - course info:    from the scorecard, for this course and these tees
#   - you:            your handicap index on the day you played
#   - the round:      score and stats
ROUND_COLUMNS = [
    "date",
    "course_name",
    "tee",
    "course_rating",
    "slope_rating",
    "par",
    "length_yards",       # optional; stored only, not used in any calculation
    "handicap_index",
    "score",
    "fairways_hit",
    "fairways_possible",  # par 4s and par 5s - par 3s have no fairway to hit
    "gir",                # greens in regulation, out of 18
    "putts",
    "three_putts",
    "up_down_attempts",   # missed greens where you tried to get up and down
    "up_down_saves",      # ...and how many of those you did
    "penalties",
]

# Every column except length_yards must be filled in.
OPTIONAL_COLUMNS = ["length_yards"]


def validate_round(r):
    """
    Check one round (a dict with the ROUND_COLUMNS keys) for impossible or
    out-of-range values. Returns a list of plain-English problems; an empty
    list means the round is OK to save.

    Catching mistakes here matters more than it looks: one typo (putts = 330
    instead of 33) saved to the CSV would quietly skew every average built
    from your rounds afterwards.
    """
    problems = []

    # --- Missing values ---
    for col in ROUND_COLUMNS:
        if col in OPTIONAL_COLUMNS:
            continue
        if r.get(col) is None or (isinstance(r.get(col), str) and not r[col].strip()):
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
    if r.get("length_yards") is not None and r["length_yards"] <= 0:
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

    return problems


def load_rounds(path=ROUNDS_PATH):
    """All saved rounds, oldest first. An empty table if none saved yet."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=ROUND_COLUMNS)
    rounds = pd.read_csv(path, parse_dates=["date"])
    # Sort by date so "last 20 rounds" really means the most recent 20.
    # kind="stable" keeps two rounds on the same date in the order entered.
    return rounds.sort_values("date", kind="stable").reset_index(drop=True)


def save_round(r, path=ROUNDS_PATH):
    """Validate one round and append it to the CSV. Returns the problem list."""
    problems = validate_round(r)
    if problems:
        return problems

    row = pd.DataFrame([r], columns=ROUND_COLUMNS)
    # mode="a" APPENDS a row instead of overwriting the file. We only write
    # the header line the very first time, when the file doesn't exist yet.
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return []
