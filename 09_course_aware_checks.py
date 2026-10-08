# Phase 9: Course-aware checks.
#
# Phase 8 built the round tracker. Phase 9 tightened its maths, following
# official WHS formulas, and this script prints the numbers behind each
# change so they can be checked by eye:
#
#   1. Overshoot: the raw per-handicap values wiggled; a straight-line fit
#      smooths them. Raw vs fitted, side by side.
#   2. Two expected scores (headline + soft "typical round") for an example
#      course, to show how far apart they are.
#   3. The GIR threshold that flags putting as "confounded by GIR", at
#      every benchmark handicap.
#   4. The WHS Rule 5.2a table and the hand-checked examples.
#
# Run with:  python 09_course_aware_checks.py
# (The same hand-checked examples also run as asserts in: python handicap.py)

import pandas as pd

import strokes_config as cfg
from benchmarks import expected_stat, load_benchmarks
from handicap import (
    WHS_TABLE,
    course_handicap,
    expected_scores,
    fit_overshoot,
    fitted_overshoot,
    index_from_differentials,
    round_half_up,
    score_differential,
)
from rounds import HOLES

bench = load_benchmarks()

# =====================================================================
# 1. Overshoot: raw vs fitted
# =====================================================================
# Overshoot = how many strokes ABOVE their handicap golfers typically
# score. Raw = straight from the table (score_to_par - handicap).
fit = fit_overshoot(bench)
overshoot = pd.DataFrame({
    "handicap": bench["hcp_mid"],
    "raw": bench["score_to_par"] - bench["hcp_mid"],
    "fitted": [fitted_overshoot(fit, h) for h in bench["hcp_mid"]],
})
overshoot["raw - fitted"] = overshoot["raw"] - overshoot["fitted"]

print("1. Overshoot (strokes above handicap in a typical round)")
print(f"   Fitted line: overshoot = {fit['slope']:.4f} x handicap + {fit['intercept']:.3f}")
print(overshoot.round(2).to_string(index=False))
print("   Raw goes up AND down; fitted rises steadily. 'raw - fitted' is the")
print("   noise the fit removes.")
print(f"   Clamped outside {fit['low']:g}-{fit['high']:g}: "
      f"index -3 -> {fitted_overshoot(fit, -3):.2f}, index 36 -> {fitted_overshoot(fit, 36):.2f}\n")

# =====================================================================
# 2. Two expected scores on an example course
# =====================================================================
# Example course: rating 71.2, slope 128, par 72. Not a real course - just
# typical-looking numbers to show the formulas side by side.
rating, slope, par = 71.2, 128, 72
print(f"2. Expected scores on an example course (rating {rating}, slope {slope}, par {par})")
rows = []
for index in [0, 5, 10, 15, 20, 25]:
    e = expected_scores(index, rating, slope, fit)
    rows.append({
        "index": index,
        "course hcp": round_half_up(course_handicap(index, slope, rating, par)),
        "plays to handicap": e["plays_to_handicap"],
        "typical round (+/-2)": e["typical_round"],
    })
print(pd.DataFrame(rows).round(1).to_string(index=False))
print("   'Plays to handicap' is the headline. 'Typical round' adds the fitted")
print("   overshoot and is soft: the overshoot partly double counts the slope")
print("   term, and the benchmark courses' ratings are unknown.\n")

# =====================================================================
# 3. When does putting get flagged "confounded by GIR"?
# =====================================================================
# Flag when your AVERAGE GIR (over 3+ rounds) is below
# GIR_CONFOUND_RATIO x the benchmark GIR at your index.
print(f"3. Putting confound threshold (ratio {cfg.GIR_CONFOUND_RATIO}, "
      f"needs {cfg.MIN_ROUNDS_FOR_CONFOUND_CHECK}+ rounds)")
rows = []
for h in bench["hcp_mid"]:
    typical = expected_stat(bench, "gir_pct", h) / 100 * HOLES
    rows.append({
        "handicap": h,
        "benchmark greens/18": typical,
        "flag if your avg below": cfg.GIR_CONFOUND_RATIO * typical,
    })
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("   Because it's a ratio, the gap that triggers it shrinks at higher")
print("   handicaps: ~3.3 greens at scratch, but only ~0.5 greens at 25 hcp.\n")

# =====================================================================
# 4. WHS Rule 5.2a table + hand-checked examples
# =====================================================================
print("4. WHS Rule 5.2a: differentials -> (lowest N averaged, adjustment)")
for n, (use, adj) in WHS_TABLE.items():
    print(f"   {n:>2}: lowest {use}, adjustment {adj:+.1f}")

print("\n   Hand-checked examples (worked out by hand first):")
ch = course_handicap(15, 130, 73.5, 72)
print(f"   Course handicap, index 15, slope 130, rating 73.5, par 72: "
      f"{ch:.2f} (displays as {round_half_up(ch)})   expected 18.76 / 19")
d = score_differential(90, 72.0, 130, pcc=0)
print(f"   Differential, score 90, rating 72.0, slope 130, pcc 0: "
      f"{d:.1f}   expected 15.6")
print(f"   Index from [15.3, 15.2, 16.6]: "
      f"{index_from_differentials([15.3, 15.2, 16.6])[0]}   expected 13.2")
print(f"   Index from [12, 13, 14, 15, 16, 17]: "
      f"{index_from_differentials([12.0, 13.0, 14.0, 15.0, 16.0, 17.0])[0]}   expected 11.5")
print(f"   Index from 2 rounds: {index_from_differentials([10.0, 11.0])[1]}")
