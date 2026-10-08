# Shared handicap / course helpers.
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
#
# Rounding rule used throughout: keep the full unrounded value for every
# calculation, and round only when DISPLAYING a number - except where WHS
# itself says to round (differentials are rounded to one decimal before
# they're averaged into an index, see index_from_differentials).

from decimal import ROUND_HALF_DOWN, ROUND_HALF_UP, Decimal

import numpy as np

from rounds import is_blank

# 113 is the WHS "standard" slope: the slope of a course of average
# difficulty. slope / 113 says how much harder (>1) or easier (<1) than
# average this course plays for a handicap golfer.
STANDARD_SLOPE = 113

# How many recent rounds an index looks at.
RECENT_ROUNDS = 20

# WHS Rule 5.2a: how an index is worked out from the differentials you have.
#   number of differentials -> (how many of the LOWEST to average, adjustment)
# Written out one line per count (rather than as ranges) so each line can
# be checked against the rule's table directly. Fewer than 3 = no index.
# Note: there is NO 0.96 multiplier - that belonged to the old (pre-2020)
# USGA system and isn't part of WHS.
WHS_TABLE = {
    3: (1, -2.0),
    4: (1, -1.0),
    5: (1, 0.0),
    6: (2, -1.0),
    7: (2, 0.0),
    8: (2, 0.0),
    9: (3, 0.0),
    10: (3, 0.0),
    11: (3, 0.0),
    12: (4, 0.0),
    13: (4, 0.0),
    14: (4, 0.0),
    15: (5, 0.0),
    16: (5, 0.0),
    17: (6, 0.0),
    18: (6, 0.0),
    19: (7, 0.0),
    20: (8, 0.0),
}
MIN_ROUNDS_FOR_INDEX = min(WHS_TABLE)  # 3

# WHS maximum Handicap Index (USGA Rule 5.2a: an initial index calculated
# above 54.0 is allocated 54.0).
MAX_INDEX = 54.0

ESTIMATE_LABEL = "ESTIMATED, not official"


def round_half_up(value, decimals=0):
    """
    Round the way people (and golf scorecards) do: halves go UP.

    Why not Python's round()? It uses "banker's rounding" (halves go to the
    nearest EVEN number, so round(18.5) == 18), and floats can't store most
    decimals exactly (12.55 is really 12.5499999...), so round(12.55, 1)
    gives 12.5. Decimal(str(value)) works with the number as written, so
    12.55 -> 12.6 and 18.5 -> 19.

    Plus handicaps (stored as NEGATIVE numbers here) round TOWARD ZERO on a
    half: -0.5 -> 0 and -1.5 -> -1. That follows the USGA's own example
    (Rules of Handicapping, Appendix C: "+0.5 ... rounds to ... 0") - i.e.
    ".5 rounded upwards" means toward the higher number, which for a
    negative value is toward zero. Decimal's ROUND_HALF_UP would go AWAY
    from zero (-0.5 -> -1), so negatives use ROUND_HALF_DOWN instead.
    """
    step = Decimal(1).scaleb(-decimals)  # 0 -> 1, 1 -> 0.1
    mode = ROUND_HALF_UP if value >= 0 else ROUND_HALF_DOWN
    result = float(Decimal(str(value)).quantize(step, rounding=mode))
    return int(result) if decimals == 0 else result


# =====================================================================
# Course handicap
# =====================================================================
def course_handicap(index, slope, rating, par):
    """
    How many strokes this golfer "gets" on THIS course from THESE tees.
    Returns the UNROUNDED value; round it only for display.

    Formula, from the USGA's WHS FAQ (Rules of Handicapping 6.1 / 6.2):
        course handicap = index x (slope / 113) + (course rating - par)

    The first part scales your index up on harder courses and down on
    easier ones. The second part adjusts for courses where even a scratch
    golfer is expected to score above or below par.

    CHECK: national associations adopted the (course rating - par) term at
    different times, so make sure YOUR association uses this version.
    expected_scores() below doesn't depend on that term at all - it uses
    rating + index x slope / 113 directly.
    """
    return index * (slope / STANDARD_SLOPE) + (rating - par)


# =====================================================================
# Expected score
# =====================================================================
def fit_overshoot(benchmarks):
    """
    Fit a straight line to "how far above their handicap golfers typically
    score", using the six rows of the benchmark table.

    Raw overshoot at each handicap = score_to_par - handicap. Those six raw
    values wiggle (e.g. 0.88 at 10 hcp is lower than 1.33 at 5 hcp), which
    is almost certainly sampling noise, not a real effect: there's no golf
    reason 10-handicappers should overshoot less than 5-handicappers. A
    straight line through all six points keeps the overall trend (higher
    handicaps overshoot more) and smooths out the noise.

    np.polyfit(x, y, 1) finds the straight line y = slope * x + intercept
    that sits closest to all the points (least squares - the same idea
    LinearRegression used back in Phase 3, with one feature).

    Returns a dict with the line and the range it's valid over.
    """
    x = benchmarks["hcp_mid"].to_numpy()
    raw = benchmarks["score_to_par"].to_numpy() - x
    slope, intercept = np.polyfit(x, raw, 1)
    return {"slope": slope, "intercept": intercept, "low": x.min(), "high": x.max()}


def fitted_overshoot(fit, index):
    """
    The fitted overshoot for this index. The index is clamped to the
    table's 0-25 range first - we don't extend the line past the data we
    fitted it on (same reasoning as expected_stat in benchmarks.py).
    """
    clamped = min(max(index, fit["low"]), fit["high"])
    return fit["slope"] * clamped + fit["intercept"]


def expected_scores(index, rating, slope, fit):
    """
    Two expected scores for a golfer with this index on this course. Always
    both - they answer different questions.

    a) plays_to_handicap = course rating + index x slope / 113
       What you'd shoot if you played exactly to your handicap. This is the
       HEADLINE number: it comes straight from WHS definitions.

    b) typical_round = plays_to_handicap + fitted overshoot
       What golfers at your index typically shoot on an average day (an
       index is built from your BEST rounds, so a typical round is higher).
       This one is SOFT - roughly +/- 2 strokes - because:
         - The benchmark overshoot was measured on real courses, whose
           average slope is probably above 113. So part of that overshoot
           is "courses are harder than standard", which the slope term in
           (a) already covers. Adding it on top double counts a little.
         - We don't know the benchmark courses' ratings: score_to_par
           assumes rating ~ par there.
    """
    plays_to_handicap = rating + index * slope / STANDARD_SLOPE
    typical_round = plays_to_handicap + fitted_overshoot(fit, index)
    return {"plays_to_handicap": plays_to_handicap, "typical_round": typical_round}


# =====================================================================
# Score differential and estimated index
# =====================================================================
def score_differential(score, rating, slope, pcc=0.0, adjusted_score=None):
    """
    A round's score converted to "handicap units", so rounds on different
    courses can be compared:

        differential = (113 / slope) x (adjusted score - course rating - pcc)

    - adjusted_score: your score after the WHS net-double-bogey cap on
      each hole. If you didn't enter it, we fall back to the gross score -
      which runs HIGH after a blow-up hole.
    - pcc: Playing Conditions Calculation, the WHS daily adjustment for
      unusually hard/easy conditions (-1.0 to +3.0). Defaults to 0.

    Returns the UNROUNDED value.
    """
    used_score = score if is_blank(adjusted_score) else adjusted_score
    pcc = 0.0 if is_blank(pcc) else pcc
    return (STANDARD_SLOPE / slope) * (used_score - rating - pcc)


def round_differential(r):
    """score_differential() for one saved round (a dict or pandas row)."""
    return score_differential(
        r["score"], r["course_rating"], r["slope_rating"],
        pcc=r.get("pcc"), adjusted_score=r.get("adjusted_score"),
    )


def index_from_differentials(differentials):
    """
    Apply the WHS Rule 5.2a table to up to 20 differentials (any order).
    Returns (value, explanation); value is None with fewer than 3.

    Steps, exactly as the rule describes:
      1. round each differential to one decimal,
      2. sort them lowest first,
      3. average the lowest N (N from WHS_TABLE) and add the adjustment,
      4. round the result to one decimal.
    """
    diffs = sorted(round_half_up(d, 1) for d in differentials)
    n = len(diffs)
    if n < MIN_ROUNDS_FOR_INDEX:
        return None, f"Need {MIN_ROUNDS_FOR_INDEX} rounds for an index (you have {n})."
    if n > RECENT_ROUNDS:
        raise ValueError(f"Pass at most {RECENT_ROUNDS} differentials.")

    use, adjustment = WHS_TABLE[n]
    lowest = diffs[:use]
    value = round_half_up(sum(lowest) / use + adjustment, 1)

    explanation = f"Lowest {use} of {n} differentials"
    if adjustment:
        explanation += f", adjustment {adjustment:+.1f}"
    if value > MAX_INDEX:
        value = MAX_INDEX
        explanation += f", capped at the WHS maximum {MAX_INDEX}"
    return value, explanation + "."


def estimated_index(rounds):
    """
    ESTIMATED index from saved rounds (a DataFrame from storage.load_rounds).

    Uses only 18-hole rounds - currently every saved round, since the
    tracker only accepts 18-hole rounds. Takes the newest 20 by date and
    applies the WHS table. See ESTIMATE_NOT_MODELLED for what's missing
    compared with an official index.
    """
    newest = rounds.sort_values("date", kind="stable").tail(RECENT_ROUNDS)
    differentials = [round_differential(row) for _, row in newest.iterrows()]
    return index_from_differentials(differentials)


# Shown in the app next to every estimated index.
ESTIMATE_NOT_MODELLED = [
    "net double bogey cap, for rounds with no adjusted score entered "
    "(the estimate runs HIGH after a blow-up hole)",
    "exceptional score reductions",
    "soft and hard caps (limits on how fast an index can rise)",
]


# =====================================================================
# Hand-checked examples. Run:  python handicap.py
# =====================================================================
# Each expected answer was worked out by hand (shown in the comment), so
# if a future edit breaks the maths, this fails loudly.
if __name__ == "__main__":
    # 15 x 130/113 = 17.257...; + (73.5 - 72) = 18.757... -> 18.76, displays as 19
    ch = course_handicap(15, 130, 73.5, 72)
    assert round_half_up(ch, 2) == 18.76, ch
    assert round_half_up(ch) == 19, ch

    # 113/130 x (90 - 72.0 - 0) = 0.8692 x 18 = 15.646 -> 15.6
    d = score_differential(90, 72.0, 130, pcc=0)
    assert round_half_up(d, 1) == 15.6, d

    # 3 differentials: lowest 1 (15.2) + adjustment -2.0 = 13.2
    assert index_from_differentials([15.3, 15.2, 16.6])[0] == 13.2

    # 6 differentials: avg of lowest 2 = (12.0 + 13.0) / 2 = 12.5; - 1.0 = 11.5
    assert index_from_differentials([12.0, 13.0, 14.0, 15.0, 16.0, 17.0])[0] == 11.5

    # Halves round UP: (12.0 + 13.1) / 2 = 12.55 -> 12.6 (Python's round gives 12.5)
    assert index_from_differentials([12.0, 13.1, 14.0, 15.0, 16.0, 17.0, 18.0])[0] == 12.6
    assert round_half_up(18.5) == 19

    # Plus handicaps round toward zero on a half (USGA Appendix C example:
    # +0.5 -> 0; 50% of +3 = +1.5 -> +1). Stored as negatives here.
    assert round_half_up(-0.5) == 0
    assert round_half_up(-1.5) == -1
    assert round_half_up(-2.25, 1) == -2.2

    # USGA Handicap Manual example: AGS 95, rating 71.5, slope 125 ->
    # 23.5 x 113 / 125 = 21.24 -> 21.2; and 69 on the same course -> -2.3.
    assert round_half_up(score_differential(95, 71.5, 125), 1) == 21.2
    assert round_half_up(score_differential(69, 71.5, 125), 1) == -2.3

    # Index above 54.0 is capped at 54.0.
    assert index_from_differentials([60.0, 61.0, 62.0])[0] == 54.0

    # Fewer than 3 -> no index
    assert index_from_differentials([10.0, 11.0])[0] is None

    print("All hand-checked handicap examples pass.")
