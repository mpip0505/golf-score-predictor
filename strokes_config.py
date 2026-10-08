# Strokes-lost conversion settings (Phase 8).
#
# The weakness report turns stat gaps ("you hit 3 fewer greens than a
# typical golfer at your handicap") into STROKES, so every part of your
# game can be compared in the same unit. This file holds every number used
# for that conversion, each with a name, so none of them are hidden "magic
# numbers" in the middle of the code.
#
# Confidence levels:
#   HIGH - follows directly from the rules of golf (a penalty IS a stroke).
#   LOW  - a modelling choice. Fine for a ballpark, worth revisiting once
#          your own rounds can test it (Phase 9).

# --- HIGH confidence ---

# Each penalty stroke is, by definition, one extra stroke on the card.
PENALTY_STROKES = 1.0

# Each putt is a stroke, so every putt above the benchmark costs one stroke.
# (Caveat: missing greens and chipping close LOWERS putt counts, so putts
# partly reflect approach play, not just putting.)
PUTT_STROKES = 1.0

# A failed up-and-down usually means one stroke more than a save.
FAILED_UP_AND_DOWN_STROKES = 1.0

# --- Shown, but NOT added to the total ---

# A 3-putt costs ~1 stroke compared with a 2-putt. But every 3-putt is
# already included in total putts above, so adding it again would count the
# same stroke twice. The report shows it as information only.
THREE_PUTT_STROKES = 1.0

# --- LOW confidence ---

# Missed greens (GIR): NO fixed constant. A missed green costs about
# (1 - scramble rate) strokes, using the benchmark scramble rate at your
# handicap. That's how often a golfer like you fails to save par from off
# the green. Example: at 10 hcp the benchmark scramble rate is 31%, so each
# extra missed green costs ~0.69 strokes. This is worked out in
# strokes_lost.py. Your OWN scrambling is scored separately (failed
# up-and-downs, above), so the two don't overlap.

# Missed fairways: set to 0 on purpose. A missed fairway costs strokes
# mostly LATER - through missed greens and penalties, which are already
# counted. The benchmark fairway % also barely changes from 0 hcp to 25 hcp
# (50% -> 46%), so the gap tells us little. The report still shows the
# fairway gap; raise this value if your own rounds show fairways matter.
FAIRWAY_MISS_STROKES = 0.0
