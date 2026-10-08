# Strokes-lost settings for the weakness report.
#
# The weakness report turns stat gaps ("you hit 3 fewer greens than a
# typical golfer at your handicap") into STROKES, so every part of your
# game can be compared in the same unit. This file holds every number used
# for that, each with a name, so none of them are hidden "magic numbers"
# in the middle of the code. Ask before changing any of them.
#
# Confidence tiers (shown in the report's "confidence" column):
#   high   - follows directly from the rules of golf (a penalty IS a stroke).
#   medium - a reasonable model with a clear mechanism.
#   low    - a modelling choice, or a stat known to be confounded.

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
# can look better than it really is. See GIR_CONFOUND_RATIO below.
PUTT_STROKES = 1.0

# Missed greens (GIR): NO fixed constant. A missed green costs about
# (1 - scramble rate) strokes, using the benchmark scramble rate at your
# handicap - how often a golfer like you fails to save par from off the
# green. Worked out in strokes_lost.py. Your OWN scrambling is scored
# separately (failed up-and-downs, above), so the two don't overlap.

# Missed fairways: set to 0 on purpose. A missed fairway costs strokes
# mostly LATER - through missed greens and penalties, which are already
# counted. The benchmark fairway % also barely changes from 0 hcp to 25 hcp
# (50% -> 46%), so the gap tells us little. The report still shows the
# fairway gap; raise this value if your own rounds show fairways matter.
FAIRWAY_MISS_STROKES = 0.0

# 3-putts: NOT scored. The benchmark file has no 3-putt column, so there's
# nothing to compare against. They're still logged with each round, ready
# for when a benchmark source is added.

# --- When to distrust the putting number ---

# Putting gets flagged "confounded by GIR" when your AVERAGE GIR is below
# this fraction of the benchmark GIR at your index. 0.7 means "you hit
# fewer than 70% as many greens as a typical golfer at your handicap".
# A judgement call, NOT calibrated on data.
# When flagged, putting is never ranked above penalties or short game.
GIR_CONFOUND_RATIO = 0.7

# The check uses your average over several rounds, never a single round:
# one round's GIR is too noisy (2 greens either way is normal variation).
# Below this many rounds, the check isn't run at all.
MIN_ROUNDS_FOR_CONFOUND_CHECK = 3
