# Shared handicap / course helpers (Phase 8).
#
# These make the tracker "course-aware". A round of 85 means something
# very different on an easy par-70 municipal course than on a hard
# championship course. The World Handicap System (WHS) handles this with
# two numbers printed on every scorecard, for each set of tees:
#
#   - Course Rating: what a scratch (0 handicap) golfer should shoot
#     there on a normal day, e.g. 71.8. Usually close to par, but not equal.
#   - Slope Rating: how much HARDER the course gets for a higher-handicap
#     golfer compared with a scratch one. 113 is "average"; 55 is the
#     easiest possible and 155 the hardest.
#
# Every function here is plain arithmetic, no machine learning.

import pandas as pd

from benchmarks import expected_stat

WHS_TABLE_PATH = "data/whs_differentials_table.csv"

# 113 is the WHS "standard" slope: the slope of a course of average
# difficulty. Dividing slope by 113 says how much harder (>1) or easier
# (<1) than average this course plays for a handicap golfer.
STANDARD_SLOPE = 113

# How many recent differentials the full WHS calculation looks at, and how
# many of the best ones it averages.
RECENT_ROUNDS = 20
BEST_OF = 8


def course_handicap(index, slope, rating, par):
    """
    How many strokes this golfer "gets" on THIS course from THESE tees.

    WHS formula: index x (slope / 113) + (course rating - par), rounded to a
    whole number. The first part scales your index up on harder courses and
    down on easier ones. The second part adjusts for courses where even a
    scratch golfer is expected to score above or below par.
    (Formula as commonly published; check it against your national golf
    association's version.)
    """
    return round(index * slope / STANDARD_SLOPE + (rating - par))


def add_overshoot(benchmarks):
    """
    Add an "overshoot" column: how many strokes ABOVE their handicap
    golfers typically score in an average round.

    Why this exists: a handicap index is built from your BEST 8 of your last
    20 rounds, so it describes your good days, not your average day. Most
    rounds come in a few strokes higher. The benchmark table shows this
    directly: a 10-handicapper averages +10.88 to par, so their overshoot is
    10.88 - 10 = 0.88.

    Assumption: this treats the benchmark's "score to par" as "score to
    course rating", i.e. it assumes the benchmark courses were rated about
    the same as their par. The source doesn't say.
    """
    benchmarks = benchmarks.copy()  # don't change the caller's DataFrame
    benchmarks["overshoot"] = benchmarks["score_to_par"] - benchmarks["hcp_mid"]
    return benchmarks


def expected_score(index, rating, slope, benchmarks):
    """
    What a golfer with this index would typically shoot on this course.

        course rating            (what a scratch golfer shoots here)
      + index x slope / 113      (extra strokes for your handicap, scaled
                                  to how hard this course is for you)
      + typical overshoot        (an average round, not one of your best 8;
                                  interpolated from the benchmark file and
                                  clamped to its 0-25 range)

    `benchmarks` must already have the overshoot column (add_overshoot).
    """
    overshoot = expected_stat(benchmarks, "overshoot", index)
    return rating + index * slope / STANDARD_SLOPE + overshoot


def score_differential(score, rating, slope):
    """
    A round's score converted to "handicap units", so rounds on different
    courses can be compared:  (113 / slope) x (score - course rating).

    Simplified compared with official WHS:
      - We use your gross score. Official WHS first caps each hole at
        net double bogey (the "adjusted gross score"). We only log 18-hole
        totals, not hole-by-hole scores, so we can't apply that cap. One
        disaster hole will push the differential higher than WHS would.
      - No Playing Conditions Calculation (the WHS weather/conditions
        adjustment), which needs every golfer's scores that day.
    """
    return STANDARD_SLOPE / slope * (score - rating)


def load_whs_table(path=WHS_TABLE_PATH):
    """Load the WHS 'rounds played -> differentials used' table (may be empty)."""
    return pd.read_csv(path)


def estimated_index(differentials, whs_table=None):
    """
    ESTIMATE a handicap index from score differentials, oldest first.
    This is an estimate, not an official index (see score_differential for
    what's simplified, and there are no WHS soft/hard caps here either).

    Returns (value, explanation). value is None when it can't be computed.

    - 20+ rounds: average of the best 8 of the most recent 20.
    - Fewer than 20: the official WHS table says how many differentials to
      use and what adjustment to apply. That table comes from the source
      (data/whs_differentials_table.csv), never from memory. Until it's
      filled in, there's no estimate under 20 rounds.
    """
    diffs = list(differentials)
    n = len(diffs)

    if n >= RECENT_ROUNDS:
        recent = diffs[-RECENT_ROUNDS:]
        best = sorted(recent)[:BEST_OF]  # lowest differentials = best rounds
        value = round(sum(best) / BEST_OF, 1)
        return value, f"Estimate: best {BEST_OF} of your last {RECENT_ROUNDS} rounds."

    if whs_table is None:
        whs_table = load_whs_table()

    if len(whs_table) == 0:
        return None, (
            f"Only {n} round(s) logged. Under {RECENT_ROUNDS} rounds the estimate "
            f"needs the official WHS table: paste it into {WHS_TABLE_PATH}."
        )

    # Find the table row whose "rounds played" range contains n.
    row = whs_table[(whs_table["rounds_low"] <= n) & (whs_table["rounds_high"] >= n)]
    if len(row) == 0:
        return None, f"The WHS table has no index for {n} round(s)."

    use = int(row["differentials_used"].iloc[0])
    adjustment = float(row["adjustment"].fillna(0).iloc[0])
    best = sorted(diffs)[:use]
    value = round(sum(best) / use + adjustment, 1)
    return value, (
        f"Estimate: best {use} of {n} rounds"
        + (f", adjustment {adjustment:+.1f}" if adjustment else "")
        + " (WHS table)."
    )
