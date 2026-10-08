# Shared helper: "where did this round lose strokes?"
#
# For one round, compare each part of your game with a typical golfer at
# YOUR handicap index (from the benchmark table, interpolated), then turn
# every gap into strokes using strokes_config.py so they can be ranked.
#
# Sign convention: POSITIVE = strokes LOST compared with the benchmark,
# negative = strokes gained (better than a typical golfer at your index).
#
# Important: these per-stat gaps are NOT course-adjusted. Benchmarks like
# "a 15-handicapper hits 24% of greens" are averages over all sorts of
# courses. On a long, hard course you'll naturally hit fewer greens than
# that, and on a short easy one more. We deliberately DON'T scale GIR /
# fairway / putt benchmarks by slope or length: there's no data saying how
# much each stat changes with course difficulty, so any scaling would be
# invented. The course-adjusted numbers are the expected scores in
# handicap.py, shown separately - and they are NOT fed in here.

import pandas as pd

import strokes_config as cfg
from benchmarks import expected_stat
from rounds import HOLES

PUTTING = "Putting"
# The areas putting may never outrank when it's flagged as confounded.
NEVER_BELOW_CONFOUNDED_PUTTING = ["Penalties", "Short game (up & down)"]


def strokes_lost(r, benchmarks):
    """
    Break one round (a dict / row with the round columns) into strokes lost
    per area. Returns a DataFrame with one row per area:
    area, yours, typical (benchmark at your index), strokes_lost, counted,
    confidence, note.
    """
    index = r["handicap_index"]

    # Benchmarks at your index (interpolated, clamped to the 0-25 range).
    b_fairways = expected_stat(benchmarks, "fairways_pct", index) / 100
    b_gir = expected_stat(benchmarks, "gir_pct", index) / 100
    b_scramble = expected_stat(benchmarks, "scramble_pct", index) / 100
    b_putts = expected_stat(benchmarks, "putts_per_round", index)
    b_penalties = expected_stat(benchmarks, "penalties_per_round", index)

    rows = []

    # --- Off the tee: fairways ---
    # Compare as counts on YOUR course's number of fairways.
    typical_fw = b_fairways * r["fairways_possible"]
    fw_gap = typical_fw - r["fairways_hit"]  # extra fairways missed
    rows.append({
        "area": "Driving (fairways)",
        "yours": f"{r['fairways_hit']:g}/{r['fairways_possible']:g}",
        "typical": f"{typical_fw:.1f}/{r['fairways_possible']:g}",
        # + 0.0 turns the "-0.0" you get from multiplying by 0 into a plain 0.0
        "strokes_lost": fw_gap * cfg.FAIRWAY_MISS_STROKES + 0.0,
        "counted": cfg.FAIRWAY_MISS_STROKES != 0,
        "confidence": "low",
        "note": "Gap shown only (cost set to 0 in strokes_config.py)"
        if cfg.FAIRWAY_MISS_STROKES == 0 else "",
    })

    # --- Approach: greens in regulation ---
    # Each extra missed green costs (1 - typical scramble rate): a golfer
    # like you saves par from off the green only that often.
    typical_gir = b_gir * HOLES
    gir_gap = typical_gir - r["gir"]
    rows.append({
        "area": "Approach (greens)",
        "yours": f"{r['gir']:g}/{HOLES}",
        "typical": f"{typical_gir:.1f}/{HOLES}",
        "strokes_lost": gir_gap * (1 - b_scramble),
        "counted": True,
        "confidence": "low",
        "note": f"Each missed green ~{1 - b_scramble:.2f} strokes",
    })

    # --- Short game: up-and-downs ---
    # Compare your saves with what a typical golfer at your index would
    # save from the SAME number of attempts. Using your own attempts (not
    # the typical number) is what keeps this separate from the GIR row.
    attempts = r["up_down_attempts"]
    expected_saves = b_scramble * attempts
    rows.append({
        "area": "Short game (up & down)",
        "yours": f"{r['up_down_saves']:g}/{attempts:g}",
        "typical": f"{expected_saves:.1f}/{attempts:g}",
        "strokes_lost": (expected_saves - r["up_down_saves"])
        * cfg.FAILED_UP_AND_DOWN_STROKES,
        "counted": True,
        "confidence": "medium",
        "note": "",
    })

    # --- Putting ---
    # 3-putts would go here, but the benchmark file has no 3-putt column,
    # so that category is skipped (the count is still logged per round).
    rows.append({
        "area": PUTTING,
        "yours": f"{r['putts']:g}",
        "typical": f"{b_putts:.1f}",
        "strokes_lost": (r["putts"] - b_putts) * cfg.PUTT_STROKES,
        "counted": True,
        "confidence": "low",
        "note": "Fewer greens hit usually means fewer putts",
    })

    # --- Penalties ---
    rows.append({
        "area": "Penalties",
        "yours": f"{r['penalties']:g}",
        "typical": f"{b_penalties:.2f}",
        "strokes_lost": (r["penalties"] - b_penalties) * cfg.PENALTY_STROKES,
        "counted": True,
        "confidence": "high",
        "note": "",
    })

    return pd.DataFrame(rows)


def putting_confound(rounds, benchmarks):
    """
    Is putting confounded by GIR for this player? Judged on the AVERAGE over
    `rounds` (a DataFrame of saved rounds), never one round.

    Compares your average GIR with the average benchmark GIR at the index
    you had for each round (your index can change between rounds).

    Returns (flag, message). flag is True / False, or None when there are
    too few rounds to judge.
    """
    n = len(rounds)
    if n < cfg.MIN_ROUNDS_FOR_CONFOUND_CHECK:
        return None, (f"GIR/putting check needs {cfg.MIN_ROUNDS_FOR_CONFOUND_CHECK} "
                      f"rounds (you have {n}).")

    your_gir = rounds["gir"].mean()
    typical_gir = sum(
        expected_stat(benchmarks, "gir_pct", idx) / 100 * HOLES
        for idx in rounds["handicap_index"]
    ) / n
    trigger = cfg.GIR_CONFOUND_RATIO * typical_gir

    flag = bool(your_gir < trigger)  # bool(): plain True/False, not numpy's np.True_
    message = (f"Your average GIR {your_gir:.1f}/18 vs typical {typical_gir:.1f}/18 "
               f"(flag below {trigger:.1f}, i.e. {cfg.GIR_CONFOUND_RATIO:.0%} of typical).")
    return flag, message


def rank_areas(breakdown, confounded):
    """
    Order the COUNTED areas from most to fewest strokes lost.

    When putting is confounded, it's moved (if needed) so it sits below both
    penalties and short game, whatever its number says - its number isn't
    trustworthy enough to be called your biggest problem ahead of those.
    It also gets a "confounded by GIR" flag.
    """
    ranked = breakdown[breakdown["counted"]].copy()
    ranked["flag"] = ""
    ranked = ranked.sort_values("strokes_lost", ascending=False).reset_index(drop=True)

    if confounded:
        is_putting = ranked["area"] == PUTTING
        ranked.loc[is_putting, "flag"] = "confounded by GIR"

        order = list(ranked["area"])
        putting_pos = order.index(PUTTING)
        lowest_protected = max(order.index(a) for a in NEVER_BELOW_CONFOUNDED_PUTTING)
        if putting_pos < lowest_protected:
            # Take putting out and put it straight after the lower of the two.
            order.remove(PUTTING)
            order.insert(lowest_protected, PUTTING)
            ranked = ranked.set_index("area").loc[order].reset_index()

    return ranked


def biggest_leak(ranked):
    """The top-ranked area that actually lost strokes (None if none did)."""
    losing = ranked[ranked["strokes_lost"] > 0]
    return None if len(losing) == 0 else losing.iloc[0]
