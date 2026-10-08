# Shared helpers for the amateur handicap benchmarks (Phase 7 onwards).
#
# This file is NOT a numbered phase script - it doesn't do anything when
# you run it. It just holds functions that several scripts (and later the
# Streamlit app) need, so we write them once here and import them:
#
#     from benchmarks import load_benchmarks, expected_stat
#
# Keeping them in one place means that if we fix a bug in, say, the
# interpolation, every script that uses it gets the fix automatically.

import numpy as np
import pandas as pd

BENCHMARKS_PATH = "data/handicap_benchmarks.csv"

# The stat columns in the benchmark CSV that hold numbers we can look up.
# (band_label, source, notes, etc. are descriptive text, not stats.)
STAT_COLUMNS = [
    "score_to_par",
    "fairways_pct",
    "gir_pct",
    "putts_per_round",
    "three_putts_per_round",
    "scramble_pct",
    "penalties_per_round",
]

# Every column the CSV must have. If one is missing, we want to find out
# immediately with a clear error, not 50 lines later with a confusing one.
REQUIRED_COLUMNS = [
    "band_label",
    "hcp_low",
    "hcp_high",
    "hcp_mid",
    *STAT_COLUMNS,
    "source",
    "source_year",
    "notes",
]

# The 4 tour-model features (Phase 7, step 5). Avg Distance is left out on
# purpose: amateurs rarely know their real average driving distance, so a
# model that needs it would be asking for a number they can't give.
TOUR_FEATURES = ["Fairway Percentage", "gir", "Average Putts", "Average Scrambling"]


def load_benchmarks(path=BENCHMARKS_PATH):
    """Load the benchmark CSV and check it's usable. Returns a DataFrame."""
    df = pd.read_csv(path)

    # 1. Required columns: compare the list we need against what's there.
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Benchmark CSV is missing columns: {missing}")

    if len(df) == 0:
        raise ValueError("Benchmark CSV has a header but no data rows yet.")

    # 2. Numeric values: pd.to_numeric(errors="coerce") turns anything that
    #    isn't a number (like a stray "n/a" or "12%") into NaN. If a cell
    #    had text in it that ISN'T blank in the original, that's a typo we
    #    want to catch. Truly blank cells are fine - they mean "the source
    #    doesn't report this stat" (e.g. three_putts_per_round).
    for col in ["hcp_low", "hcp_high", "hcp_mid", *STAT_COLUMNS]:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad_rows = converted.isna() & df[col].notna()
        if bad_rows.any():
            raise ValueError(
                f"Column {col!r} has non-numeric values: "
                f"{df.loc[bad_rows, col].tolist()}"
            )
        df[col] = converted

    # The handicap columns are the "anchors" for interpolation, so unlike
    # the stats they can never be blank.
    if df[["hcp_low", "hcp_high", "hcp_mid"]].isna().any().any():
        raise ValueError("hcp_low / hcp_high / hcp_mid must be filled in every row.")

    # 3. Bands in order: interpolation needs the anchor points sorted from
    #    lowest handicap to highest, with no duplicates. .is_monotonic_increasing
    #    only checks "never goes down", so we also check for repeats.
    if not df["hcp_mid"].is_monotonic_increasing or df["hcp_mid"].duplicated().any():
        raise ValueError("Rows must be sorted by hcp_mid, lowest first, no repeats.")

    # Each row's midpoint should sit inside its own band.
    outside = (df["hcp_mid"] < df["hcp_low"]) | (df["hcp_mid"] > df["hcp_high"])
    if outside.any():
        raise ValueError(f"hcp_mid outside its band in rows: {df.index[outside].tolist()}")

    return df


def expected_stat(benchmarks, stat, handicap):
    """
    Return the expected value of `stat` for a golfer with this `handicap`,
    by drawing a straight line between the two nearest benchmark rows.

    Example: the table has 10 hcp -> 36% GIR and 15 hcp -> 24% GIR.
    A 12 handicap is 2/5 of the way from 10 to 15, so we go 2/5 of the way
    from 36 to 24: 36 + 0.4 * (24 - 36) = 31.2% GIR.

    Outside the table's range we CLAMP: a +2 handicap gets the 0-hcp value,
    a 32 handicap gets the 25-hcp value. We don't extend the line past the
    ends (extrapolate), because:
      - We have no data out there, so any trend is a guess. GIR drops by
        7 points from 20 to 25 hcp; extending that line to a 40 handicap
        would predict negative GIR, which is impossible.
      - Clamping gives a slightly-off but sensible answer; extrapolating
        can give a confidently-wrong nonsense one.
    This is the same lesson as the tour model's range guard below: a
    model or table is only trustworthy inside the data it was built from.

    Returns NaN if the source has no numbers for this stat at all.
    """
    if stat not in STAT_COLUMNS:
        raise ValueError(f"Unknown stat {stat!r}. Choose from {STAT_COLUMNS}")

    # Only use rows where this stat is actually filled in.
    rows = benchmarks.dropna(subset=[stat])
    if len(rows) == 0:
        return float("nan")

    anchors_x = rows["hcp_mid"].to_numpy()
    anchors_y = rows[stat].to_numpy()

    # Clamp first, explicitly, so the behaviour is visible right here
    # rather than hidden inside numpy. (np.interp also happens to clamp
    # at the ends, but we don't want to rely on a detail you'd have to
    # look up to understand this function.)
    clamped = min(max(handicap, anchors_x.min()), anchors_x.max())

    # np.interp(x, xs, ys) does the "straight line between the two nearest
    # points" calculation from the docstring example.
    return float(np.interp(clamped, anchors_x, anchors_y))


def tour_feature_ranges(tour_df):
    """
    Min and max of each tour-model feature, from the real PGA data.
    Returns a DataFrame with one row per feature and columns min / max.
    """
    return tour_df[TOUR_FEATURES].agg(["min", "max"]).T


def inside_tour_range(ranges, inputs):
    """
    Check whether `inputs` (a dict like {"gir": 64, ...}) sits inside the
    range the tour model was trained on.

    Returns (ok, problems): ok is True only if EVERY feature is in range;
    problems is a list of plain-English messages for the ones that aren't.

    Why this matters: Linear Regression will happily produce a number for
    ANY input, even absurd ones. Feed it an 18-handicapper's 25% GIR (the
    tour's lowest is ~54%) and it extends its straight line far past
    anything it has seen and returns a score - but that score means
    nothing. So the rule is: tour model inside this range only, and never
    as an amateur score prediction.
    """
    problems = []
    for feature in TOUR_FEATURES:
        value = inputs[feature]
        low, high = ranges.loc[feature, "min"], ranges.loc[feature, "max"]
        if not (low <= value <= high):
            problems.append(
                f"{feature} = {value} is outside the tour range {low:.2f}-{high:.2f}"
            )
    return len(problems) == 0, problems
