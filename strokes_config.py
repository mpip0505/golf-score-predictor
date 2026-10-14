# Judgement-call settings: every number in the tracker that is a CHOICE
# rather than a fact from a rule book or a data file.
#
# It started (Phase 8) as the strokes-lost settings, which turn stat gaps
# ("you hit 3 fewer greens than a typical golfer at your handicap") into
# STROKES so every part of your game can be compared in the same unit. Since
# the testing phase it holds every other judgement call too (the confound
# rule, the expected-score fit, the "too few rounds" thresholds and the
# typo-catching limits), so they're all in one place, each with a name and
# a comment. Ask before changing any of them.
#
# What is NOT here: official numbers (WHS slope range 55-155, index range
# +10 to 54, the Rule 5.2a table, standard slope 113) and numbers from the
# benchmark CSV. Those are facts with a source, not choices.
#
# JUDGEMENT_CALLS at the bottom lists every setting with what it assumes.
# The app's "Known limitations" box shows that list, and it's built from the
# variables themselves, so it can't drift from the code.
#
# Confidence tiers (shown in the report's "confidence" column):
#   high   - follows directly from the rules of golf (a penalty IS a stroke).
#   medium - a reasonable model with a clear mechanism.
#   low    - a modelling choice, or a stat known to be confounded.

# =====================================================================
# Strokes-lost conversions
# =====================================================================

# --- high confidence ---

# Each penalty stroke is, by definition, one extra stroke on the card.
PENALTY_STROKES = 1.0

# --- medium confidence ---

# A failed up-and-down usually means one stroke more than a save.
FAILED_UP_AND_DOWN_STROKES = 1.0

# --- low confidence ---

# Each putt is a stroke, so every putt above the benchmark costs one.
# LOW confidence because putts per round is confounded by GIR: golfers who
# miss lots of greens chip close and take FEWER putts, so their putting
# can look better than it really is. See GIR_CONFOUND_GREENS below.
PUTT_STROKES = 1.0

# Missed greens (GIR): NO fixed constant. A missed green costs about
# (1 - scramble rate) strokes, using the benchmark scramble rate at your
# handicap - how often a golfer like you fails to save par from off the
# green (0.79 at 15 hcp, where the benchmark scramble rate is 21%). Worked
# out in strokes_lost.py. Your OWN scrambling is scored separately (failed
# up-and-downs, above), so the two don't overlap.

# Missed fairways: set to 0 on purpose. A missed fairway costs strokes
# mostly LATER - through missed greens and penalties, which are already
# counted. The benchmark fairway % also barely changes from 0 hcp to 25 hcp
# (50% -> 46%), so the gap tells us little. The report still shows the
# fairway gap; raise this value if your own rounds show fairways matter.
FAIRWAY_MISS_STROKES = 0.0

# 3-putts: NOT scored (None = "no conversion"). The benchmark file has no
# 3-putt column, so there's nothing to compare against. They're still
# logged with each round, ready for when a benchmark source is added.
THREE_PUTT_STROKES = None

# =====================================================================
# When to distrust the putting number
# =====================================================================

# Putting gets flagged "confounded by GIR" when your AVERAGE GIR is at
# least this many greens (per 18 holes) BELOW the benchmark GIR at your
# index. When flagged, putting is never ranked above penalties or short game.
#
# Why an absolute gap (decided by the user on 2026-10-12) instead of the
# earlier "below 70% of the benchmark" ratio? The mechanism is about
# greens: each extra missed green is one more hole where you chip close and
# need fewer putts. Two extra missed greens is enough to move a putt count;
# a ratio of a small number isn't. At 25 hcp, 70% of 1.8 greens is 1.26,
# so a gap of just 0.5 greens would have triggered the flag.
#
# Intended consequence: the rule CAN'T fire at index 25 (benchmark 1.8
# greens: you'd need -0.2 greens) and only rarely at 20 (benchmark 3.06
# greens: you'd need to average 1.06 or fewer). That's fine. At those
# handicaps there are only ~2-3 benchmark greens to miss, so the putting
# distortion from "extra missed greens" is capped at about that size.
#
# A judgement call, NOT calibrated on data.
GIR_CONFOUND_GREENS = 2

# The check uses your average over several rounds, never a single round:
# one round's GIR is too noisy (2 greens either way is normal variation).
# Below this many rounds, the check isn't run at all.
MIN_ROUNDS_FOR_CONFOUND_CHECK = 3

# =====================================================================
# Expected score ("typical round")
# =====================================================================

# The six raw "how far above their handicap golfers score" values wiggle,
# so handicap.fit_overshoot() smooths them with a polynomial of this degree
# (1 = a straight line), and clamps it to the table's 0-25 range.
OVERSHOOT_FIT_DEGREE = 1

# How rough the "typical round" expected score is, in strokes either way.
# Shown in the app as "roughly +/-2". The fit's residuals are at most 0.94
# strokes; the rest of the band covers the double counting with the slope
# term and the unknown ratings of the benchmark courses (FINDINGS.md #12).
TYPICAL_ROUND_BAND = 2

# =====================================================================
# How many rounds before the report says anything
# =====================================================================

# Below this many 18-hole rounds the weakness report shows "too few rounds
# to say much".
FEW_ROUNDS_NOTICE = 3

# Below this many it adds "a first impression, not a pattern".
FIRST_IMPRESSION_ROUNDS = 5

# =====================================================================
# Typo-catching limits used by validation (rounds.py)
# =====================================================================

# Generous "does this look like a typo?" bounds, NOT WHS rules. The 9-hole
# ones are roughly half the 18-hole ones, and nothing official backs them.
#   rating:    allowed course rating range
#   par:       allowed par range
#   below_par: a score more than this many strokes under par is flagged
#   max_score: a score above this is flagged
SANITY_LIMITS = {
    18: {"rating": (60, 80), "par": (60, 75), "below_par": 15, "max_score": 200},
    9: {"rating": (25, 40), "par": (27, 38), "below_par": 8, "max_score": 100},
}

# =====================================================================
# The list shown in the app (built from the variables above)
# =====================================================================
JUDGEMENT_CALLS = [
    {"name": "PENALTY_STROKES", "value": PENALTY_STROKES,
     "assumes": "Each penalty stroke over the benchmark costs one stroke.",
     "why": "High confidence: a penalty is a stroke by definition."},
    {"name": "FAILED_UP_AND_DOWN_STROKES", "value": FAILED_UP_AND_DOWN_STROKES,
     "assumes": "A failed up-and-down costs one stroke more than a save.",
     "why": "Medium: usually true, but a duff can cost two."},
    {"name": "missed green (formula)", "value": "1 - benchmark scramble rate",
     "assumes": "An extra missed green costs what a golfer at your index fails to save.",
     "why": "Low: uses the benchmark's scramble rate, whose definition is unresolved (#5)."},
    {"name": "PUTT_STROKES", "value": PUTT_STROKES,
     "assumes": "Each putt over the benchmark costs one stroke.",
     "why": "Low: putts are confounded by GIR (see GIR_CONFOUND_GREENS)."},
    {"name": "FAIRWAY_MISS_STROKES", "value": FAIRWAY_MISS_STROKES,
     "assumes": "A missed fairway costs nothing extra by itself.",
     "why": "Low: its cost shows up later as missed greens and penalties, "
            "and the benchmark barely changes with handicap."},
    {"name": "THREE_PUTT_STROKES", "value": THREE_PUTT_STROKES,
     "assumes": "3-putts are logged but not scored.",
     "why": "No benchmark to compare against."},
    {"name": "GIR_CONFOUND_GREENS", "value": GIR_CONFOUND_GREENS,
     "assumes": "Putting is distorted once you miss 2+ more greens than typical.",
     "why": "Not calibrated. Can't fire at 25 hcp, rarely at 20 (intended)."},
    {"name": "MIN_ROUNDS_FOR_CONFOUND_CHECK", "value": MIN_ROUNDS_FOR_CONFOUND_CHECK,
     "assumes": "3 rounds give a usable average GIR.",
     "why": "Still noisy at 3; a choice, not tested."},
    {"name": "OVERSHOOT_FIT_DEGREE", "value": OVERSHOOT_FIT_DEGREE,
     "assumes": "Overshoot above handicap rises in a straight line, 0-25 hcp.",
     "why": "Fitted to only 6 points; the wiggles are assumed to be noise."},
    {"name": "TYPICAL_ROUND_BAND", "value": TYPICAL_ROUND_BAND,
     "assumes": "The typical-round score is good to about this many strokes.",
     "why": "A rough band, not a measured interval."},
    {"name": "FEW_ROUNDS_NOTICE", "value": FEW_ROUNDS_NOTICE,
     "assumes": "Below this many rounds the report says little.",
     "why": "A choice about noise, not tested."},
    {"name": "FIRST_IMPRESSION_ROUNDS", "value": FIRST_IMPRESSION_ROUNDS,
     "assumes": "Below this many rounds the average is a first impression.",
     "why": "A choice about noise, not tested."},
    {"name": "SANITY_LIMITS", "value": SANITY_LIMITS,
     "assumes": "Ratings, pars and scores outside these ranges are typos.",
     "why": "Guesses, especially the 9-hole ones; may refuse an unusual real course."},
]
