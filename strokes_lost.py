# Shared helper: "where did this round lose strokes?" (Phase 8).
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
# invented. The course-adjusted number is "strokes vs expected on this
# course" (handicap.py), shown separately.

import pandas as pd

import strokes_config as cfg
from benchmarks import expected_stat
from rounds import HOLES


def strokes_lost(r, benchmarks):
    """
    Break one round (a dict / row with the rounds.py columns) into strokes
    lost per area. Returns a DataFrame with one row per area:
    area, yours, typical (benchmark at your index), strokes_lost, counted, note.
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
        "yours": f"{r['fairways_hit']}/{r['fairways_possible']}",
        "typical": f"{typical_fw:.1f}/{r['fairways_possible']}",
        "strokes_lost": fw_gap * cfg.FAIRWAY_MISS_STROKES,
        "counted": cfg.FAIRWAY_MISS_STROKES != 0,
        "note": "Gap shown only (cost set to 0 in strokes_config.py)"
        if cfg.FAIRWAY_MISS_STROKES == 0 else "Low-confidence constant",
    })

    # --- Approach: greens in regulation ---
    # Each extra missed green costs (1 - typical scramble rate): a golfer
    # like you saves par from off the green only that often.
    typical_gir = b_gir * HOLES
    gir_gap = typical_gir - r["gir"]
    rows.append({
        "area": "Approach (greens)",
        "yours": f"{r['gir']}/{HOLES}",
        "typical": f"{typical_gir:.1f}/{HOLES}",
        "strokes_lost": gir_gap * (1 - b_scramble),
        "counted": True,
        "note": f"Each missed green ~{1 - b_scramble:.2f} strokes (low confidence)",
    })

    # --- Short game: up-and-downs ---
    # Compare your saves with what a typical golfer at your index would
    # save from the SAME number of attempts. Using your own attempts (not
    # the typical number) is what keeps this separate from the GIR row.
    attempts = r["up_down_attempts"]
    expected_saves = b_scramble * attempts
    rows.append({
        "area": "Short game (up & down)",
        "yours": f"{r['up_down_saves']}/{attempts}",
        "typical": f"{expected_saves:.1f}/{attempts}",
        "strokes_lost": (expected_saves - r["up_down_saves"])
        * cfg.FAILED_UP_AND_DOWN_STROKES,
        "counted": True,
        "note": "",
    })

    # --- Putting ---
    rows.append({
        "area": "Putting",
        "yours": f"{r['putts']}",
        "typical": f"{b_putts:.1f}",
        "strokes_lost": (r["putts"] - b_putts) * cfg.PUTT_STROKES,
        "counted": True,
        "note": "Fewer greens hit usually means fewer putts",
    })
    # 3-putts: shown, never added - they're already inside total putts.
    rows.append({
        "area": "  of which 3-putts",
        "yours": f"{r['three_putts']}",
        "typical": "no benchmark",
        "strokes_lost": r["three_putts"] * cfg.THREE_PUTT_STROKES,
        "counted": False,
        "note": "Info only - already included in putts",
    })

    # --- Penalties ---
    rows.append({
        "area": "Penalties",
        "yours": f"{r['penalties']}",
        "typical": f"{b_penalties:.2f}",
        "strokes_lost": (r["penalties"] - b_penalties) * cfg.PENALTY_STROKES,
        "counted": True,
        "note": "",
    })

    return pd.DataFrame(rows)


def biggest_leak(breakdown):
    """The counted area with the most strokes lost (None if all are gains)."""
    counted = breakdown[breakdown["counted"]]
    worst = counted.loc[counted["strokes_lost"].idxmax()]
    return worst if worst["strokes_lost"] > 0 else None
