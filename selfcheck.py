# Self-check: one command that tests everything the testing phase relies on.
#
# Run with:   python selfcheck.py
#
# It prints PASS or FAIL for each check and exits with a non-zero code if
# anything failed (so it could be run automatically one day). Every check
# runs on a SCRATCH folder made with tempfile - it never reads or writes
# data/my_rounds.csv. The very last check proves that by comparing the real
# file (or its absence) with how it was before we started.
#
# Why a hand-rolled checker instead of a test framework (pytest)? The
# project rule is pandas / scikit-learn / Streamlit only, and at this size
# a list of small functions that raise AssertionError does the same job.

import os
import sys
import tempfile

# --- Safety first: point the app at a scratch file BEFORE importing anything
# that might read GOLF_ROUNDS_PATH. storage.rounds_path() reads it on every
# call, so the Streamlit app run below (AppTest) can only ever see scratch.
PROJECT = os.path.dirname(os.path.abspath(__file__))
SCRATCH = tempfile.mkdtemp(prefix="golf_selfcheck_")
os.environ["GOLF_ROUNDS_PATH"] = os.path.join(SCRATCH, "unused", "my_rounds.csv")
os.chdir(PROJECT)  # so "data/..." paths work wherever you run this from

REAL_ROUNDS = os.path.join(PROJECT, "data", "my_rounds.csv")
REAL_BACKUPS = os.path.join(PROJECT, "data", "backups")


def snapshot(path):
    """The file's exact bytes, or None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def list_dir(path):
    return sorted(os.listdir(path)) if os.path.isdir(path) else None


REAL_BEFORE = snapshot(REAL_ROUNDS)
REAL_BACKUPS_BEFORE = list_dir(REAL_BACKUPS)

import io  # noqa: E402  (imports after the safety setup on purpose)
import shutil  # noqa: E402
import subprocess  # noqa: E402
import traceback  # noqa: E402
from datetime import date  # noqa: E402

import pandas as pd  # noqa: E402

import storage  # noqa: E402
from benchmarks import (  # noqa: E402
    REQUIRED_COLUMNS as BENCH_COLUMNS,
    STAT_COLUMNS,
    TOUR_FEATURES,
    expected_stat,
    inside_tour_range,
    load_benchmarks,
    tour_feature_ranges,
)
from handicap import estimated_index  # noqa: E402
from rounds import parse_round, validate_round  # noqa: E402
from strokes_lost import (  # noqa: E402
    average_strokes_lost,
    biggest_leak,
    putting_confound,
    rank_areas,
)

APP_FILE = os.path.join(PROJECT, "08_round_tracker.py")
CHECKS = []  # (name, function), filled in by the @check decorator
INFO = []    # extra lines to print under a check (e.g. trend flags)


def check(name):
    """Decorator: register a function as a named check."""
    def register(func):
        CHECKS.append((name, func))
        return func
    return register


def scratch_path(name):
    """A fresh rounds-file path in its own scratch folder."""
    folder = os.path.join(SCRATCH, name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "my_rounds.csv")


def data_rows(path):
    """How many rounds (lines after the header) are in a file."""
    return len(pd.read_csv(path, dtype=str, keep_default_na=False))


# A valid 18-hole round to start from; checks change one thing at a time.
BASE = {
    "date": date(2026, 9, 1), "course_name": "Test Course", "tee": "White",
    "course_rating": 72.0, "slope_rating": 113, "par": 72, "holes": 18,
    "length_yards": 6400, "handicap_index": 15.0, "score": 92,
    "fairways_hit": 6, "fairways_possible": 14, "gir": 2, "putts": 36,
    "three_putts": 3, "up_down_attempts": 14, "up_down_saves": 2,
    "penalties": 3, "adjusted_score": None, "pcc": None,
}


def round_with(**changes):
    r = dict(BASE)
    r.update(changes)
    return r


# =====================================================================
# The hand-built 5-round set (+ one 9-hole round that must be ignored)
# =====================================================================
# All at index 15, so every benchmark is read straight off the 15-hcp row
# of data/handicap_benchmarks.csv - no interpolation to redo by hand:
#   GIR 24% -> 0.24 x 18 = 4.32 greens, scramble 21%, putts 33.1,
#   penalties 2.45.
# Course: rating 72.0, slope 113, so differential = score - 72 exactly.
#
#   round  date        score  gir  putts  attempts  saves  penalties
#   R1     2026-09-01   92     2    36      14        2       3
#   R2     2026-09-02   90     3    35      13        3       2
#   R3     2026-09-03   95     1    37      15        2       4
#   R4     2026-09-04   91     2    34      14        3       3
#   R5     2026-09-05   89     2    36      12        2       2
#
# Averages over the 5 rounds:
#   GIR       (2+3+1+2+2)/5     = 10/5  = 2.0
#   putts     (36+35+37+34+36)/5 = 178/5 = 35.6
#   attempts  (14+13+15+14+12)/5 = 68/5  = 13.6
#   saves     (2+3+2+3+2)/5      = 12/5  = 2.4
#   penalties (3+2+4+3+2)/5      = 14/5  = 2.8
#
# Strokes lost (all formulas are straight lines, so the average of the
# per-round numbers equals the formula applied to the averages):
#   Approach   (4.32 - 2.0) x (1 - 0.21) = 2.32 x 0.79 = 1.8328
#   Short game  0.21 x 13.6 - 2.4        = 2.856 - 2.4 = 0.456
#   Putting     35.6 - 33.1                            = 2.5
#   Penalties   2.8 - 2.45                             = 0.35
#
# Confound check: flag if average GIR < 0.7 x 4.32 = 3.024. 2.0 < 3.024,
# so putting is flagged. By the raw numbers putting (2.5) would rank first,
# but a flagged putting must sit below penalties AND short game, so:
#   expected order: Approach, Short game, Penalties, Putting (flagged)
#   biggest leak:   Approach, 1.8328 strokes
#
# Estimated index: differentials 20, 18, 23, 19, 17. With 5 rounds WHS
# Rule 5.2a uses the lowest 1, adjustment 0 -> 17.0.
#
# The 9-hole round (R6) would change both answers if it were counted:
# its differential 40 - 36 = 4.0 would become the lowest (index 9.5 with 6
# rounds: (4.0 + 17.0) / 2 - 1.0), and strokes_lost() refuses 9-hole rounds.
FIVE_ROUNDS = [
    round_with(date=date(2026, 9, 1), score=92, gir=2, putts=36, up_down_attempts=14,
               up_down_saves=2, penalties=3),
    round_with(date=date(2026, 9, 2), score=90, gir=3, putts=35, up_down_attempts=13,
               up_down_saves=3, penalties=2),
    round_with(date=date(2026, 9, 3), score=95, gir=1, putts=37, up_down_attempts=15,
               up_down_saves=2, penalties=4),
    round_with(date=date(2026, 9, 4), score=91, gir=2, putts=34, up_down_attempts=14,
               up_down_saves=3, penalties=3),
    round_with(date=date(2026, 9, 5), score=89, gir=2, putts=36, up_down_attempts=12,
               up_down_saves=2, penalties=2),
]
NINE_HOLE = round_with(date=date(2026, 9, 6), holes=9, course_rating=36.0, par=36,
                       score=40, fairways_possible=7, fairways_hit=3, gir=1, putts=15,
                       three_putts=1, up_down_attempts=5, up_down_saves=1, penalties=0,
                       length_yards=None)


def save_all(path, rounds):
    for r in rounds:
        problems = storage.save_round(r, path)
        assert problems == [], f"setup round refused: {problems}"


# =====================================================================
# 1. handicap.py asserts
# =====================================================================
@check("handicap.py hand-checked asserts")
def check_handicap():
    # Run it as its own process, exactly like `python handicap.py`.
    result = subprocess.run([sys.executable, "handicap.py"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr.strip().splitlines()[-1]


# =====================================================================
# 2. Benchmark CSV validation
# =====================================================================
@check("benchmark CSV: columns, numeric values, trend flags")
def check_benchmarks():
    bench = load_benchmarks()
    assert all(c in bench.columns for c in BENCH_COLUMNS)
    for col in ["hcp_mid", *STAT_COLUMNS]:
        assert pd.api.types.is_numeric_dtype(bench[col]), f"{col} isn't numeric"

    # The validator must REJECT a broken file (tested on scratch copies).
    broken = bench.copy()
    broken["gir_pct"] = broken["gir_pct"].astype(object)
    broken.loc[0, "gir_pct"] = "61%"
    path = os.path.join(SCRATCH, "bench_text.csv")
    broken.to_csv(path, index=False)
    try:
        load_benchmarks(path)
        raise AssertionError("a text value in gir_pct was accepted")
    except ValueError:
        pass
    path = os.path.join(SCRATCH, "bench_missing.csv")
    bench.drop(columns=["scramble_pct"]).to_csv(path, index=False)
    try:
        load_benchmarks(path)
        raise AssertionError("a missing column was accepted")
    except ValueError:
        pass

    # Trend flags: REPORTED, never fixed (the numbers come from the source).
    directions = {"score_to_par": "up", "fairways_pct": "down", "gir_pct": "down",
                  "putts_per_round": "up", "scramble_pct": "down",
                  "penalties_per_round": "up"}
    for stat, direction in directions.items():
        values = bench[stat].dropna()
        changes = values.diff().dropna() * (1 if direction == "up" else -1)
        if (changes < 0).any():
            INFO.append(f"flag: {stat} moves the wrong way somewhere: {values.tolist()}")
        elif (changes == 0).any():
            INFO.append(f"flag: {stat} is flat in places: {values.tolist()}")
    if bench["three_putts_per_round"].isna().all():
        INFO.append("flag: three_putts_per_round has no data in the source")


# =====================================================================
# 3. Tour model range guard
# =====================================================================
@check("tour model range guard")
def check_tour_guard():
    tour = pd.read_csv("data/pgaTourData.csv")
    tour = tour.dropna(subset=["Avg Distance", *TOUR_FEATURES, "Average Score"])
    ranges = tour_feature_ranges(tour)
    bench = load_benchmarks()

    tour_like = {"Fairway Percentage": 61, "gir": 66, "Average Putts": 29.2,
                 "Average Scrambling": 58}
    assert inside_tour_range(ranges, tour_like)[0], "tour-like player refused"

    # Exactly on the edge counts as inside; just past it doesn't.
    edge = {f: ranges.loc[f, "min"] for f in TOUR_FEATURES}
    assert inside_tour_range(ranges, edge)[0], "edge of range refused"
    edge["gir"] -= 0.01
    assert not inside_tour_range(ranges, edge)[0], "just below the range accepted"

    amateur_15 = {"Fairway Percentage": expected_stat(bench, "fairways_pct", 15),
                  "gir": expected_stat(bench, "gir_pct", 15),
                  "Average Putts": expected_stat(bench, "putts_per_round", 15),
                  "Average Scrambling": expected_stat(bench, "scramble_pct", 15)}
    ok, problems = inside_tour_range(ranges, amateur_15)
    assert not ok and problems, "a 15-handicapper was let through the tour guard"


# =====================================================================
# 4. Validation rules
# =====================================================================
@check("validation: each bad input caught, on the right field")
def check_validation():
    assert validate_round(BASE) == [], validate_round(BASE)
    assert validate_round(NINE_HOLE) == [], validate_round(NINE_HOLE)

    # (description, the round, field that must be flagged)
    cases = [
        ("future date", round_with(date=date(2026, 10, 13)), "date"),
        ("missing score", round_with(score=None), "score"),
        ("fairways hit > possible", round_with(fairways_hit=15), "fairways_hit"),
        ("fairways possible > holes", round_with(fairways_possible=19), "fairways_possible"),
        ("GIR 19", round_with(gir=19), "gir"),
        ("GIR 10 on 9 holes", dict(NINE_HOLE, gir=10), "gir"),
        ("putts > score - holes", round_with(putts=75), "putts"),
        ("4 three-putts from 11 putts", round_with(putts=11, three_putts=4), "three_putts"),
        ("saves > attempts", round_with(up_down_saves=15), "up_down_saves"),
        ("attempts > missed greens", round_with(up_down_attempts=17), "up_down_attempts"),
        ("negative penalties", round_with(penalties=-1), "penalties"),
        ("negative putts", round_with(putts=-3), "putts"),
        ("half a stroke", round_with(score=91.5), "score"),
        ("holes = 12", round_with(holes=12), "holes"),
        ("PCC 1.5", round_with(pcc=1.5), "pcc"),
        ("adjusted > gross", round_with(adjusted_score=95), "adjusted_score"),
    ]
    for description, r, field in cases:
        problems = dict(validate_round(r, today=date(2026, 10, 12)))
        assert field in problems, f"{description}: not caught ({problems})"
        INFO.append(f"{description:28s} -> {problems[field]}")

    # 3-putt limit is putts // 3 (rounded DOWN): 11 putts allow 3, not 4.
    assert validate_round(round_with(putts=11, three_putts=3)) == []

    # One message per field, even when several rules fail at once.
    fields = [f for f, _ in validate_round(round_with(score=None, putts=None))]
    assert len(fields) == len(set(fields)), "a field got two messages"

    # Unreadable cells in a CSV are reported, not crashed on.
    _, problems = parse_round({"date": "12/10/2026", "score": "abc"})
    assert {f for f, _ in problems} == {"date", "score"}, problems


# =====================================================================
# 5. Storage
# =====================================================================
@check("storage: save, reload, export, import, backups")
def check_storage_round_trip():
    path = scratch_path("roundtrip")
    backups = os.path.join(os.path.dirname(path), "backups")

    assert storage.save_round(BASE, path) == []
    assert list_dir(backups) is None, "a backup was made of a file that didn't exist"
    before_second = snapshot(path)
    assert storage.save_round(round_with(date=date(2026, 9, 2), score=90), path) == []
    made = list_dir(backups)
    assert len(made) == 1, made
    assert snapshot(os.path.join(backups, made[0])) == before_second, "backup isn't the old file"

    rounds, skipped = storage.load_rounds(path)
    assert len(rounds) == 2 and skipped == []
    assert list(rounds["score"]) == [92, 90]
    # Blank optional fields: stored as empty cells, loaded as blank.
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert (raw["pcc"] == "").all() and (raw["adjusted_score"] == "").all()
    assert rounds["pcc"].isna().all()
    assert (raw["schema_version"] == str(storage.SCHEMA_VERSION)).all()

    # Export = an exact copy of the file.
    exported = storage.export_csv(path)
    assert exported.encode() == snapshot(path)

    # Import the export into a fresh scratch copy -> same rounds.
    fresh = scratch_path("import_target")
    added, problems, _ = storage.import_rounds(io.StringIO(exported), fresh)
    assert (added, problems) == (2, []), (added, problems)
    assert storage.load_rounds(fresh)[0].equals(rounds)
    # Re-importing the same file adds nothing and makes no backup.
    fresh_backups = os.path.join(os.path.dirname(fresh), "backups")
    added, problems, _ = storage.import_rounds(io.StringIO(exported), fresh)
    assert (added, problems) == (0, []) and list_dir(fresh_backups) is None

    # A file with one bad row: nothing imported, file untouched.
    bad = exported + "2026-09-03,X,White,72,113,72,18,,15,92,6,14,19,36,3,14,2,3,,,2\n"
    before = snapshot(fresh)
    added, problems, _ = storage.import_rounds(io.StringIO(bad), fresh)
    assert added == 0 and any("Row 4" in p for p in problems), problems
    assert snapshot(fresh) == before

    # Same date + course + tee, different score: imported, with a warning.
    near = pd.DataFrame([storage._as_raw_row(round_with(score=93))]).to_csv(index=False)
    added, problems, warnings = storage.import_rounds(io.StringIO(near), fresh)
    assert added == 1 and len(warnings) == 1, (added, problems, warnings)


@check("storage: bad rows reported, never dropped from the file")
def check_storage_bad_rows():
    path = scratch_path("bad_rows")
    save_all(path, FIVE_ROUNDS[:2])
    # Someone typed "abc" as a score in a spreadsheet.
    with open(path, "a") as f:
        f.write("2026-09-03,Test Course,White,72,113,72,18,,15,abc,6,14,2,36,3,14,2,3,,,2\n")
    before = snapshot(path)

    rounds, skipped = storage.load_rounds(path)
    assert len(rounds) == 2 and len(skipped) == 1, (len(rounds), skipped)
    assert skipped[0]["row"] == 4 and "Score" in skipped[0]["problems"][0], skipped
    assert snapshot(path) == before, "loading changed the file"

    # Saving another round keeps the bad row in the file.
    storage.save_round(FIVE_ROUNDS[2], path)
    assert ",abc," in open(path).read(), "saving dropped the bad row"
    assert data_rows(path) == 4


@check("storage: old (version 1) and newer-version files")
def check_storage_versions():
    # A Phase 9 file: no holes, no schema_version.
    path = scratch_path("v1")
    v1_columns = [c for c in storage.ROUND_COLUMNS if c != "holes"]
    v1 = pd.DataFrame([storage._as_raw_row(BASE)])[v1_columns]
    v1.to_csv(path, index=False)
    rounds, skipped = storage.load_rounds(path)
    assert len(rounds) == 1 and skipped == [] and rounds["holes"].iloc[0] == 18
    assert storage.save_round(FIVE_ROUNDS[1], path) == []
    assert storage.load_rounds(path)[0]["score"].tolist() == [92, 90]

    # A file from a NEWER tracker: refused for loading, saving and importing.
    path = scratch_path("v3")
    row = storage._as_raw_row(BASE)
    row["schema_version"] = "3"
    pd.DataFrame([row]).to_csv(path, index=False)
    before = snapshot(path)
    for action in (lambda: storage.load_rounds(path),
                   lambda: storage.save_round(FIVE_ROUNDS[1], path)):
        try:
            action()
            raise AssertionError("a newer-version file was accepted")
        except storage.SchemaTooNewError:
            pass
    assert snapshot(path) == before
    added, problems, _ = storage.import_rounds(path, scratch_path("v3_import"))
    assert added == 0 and "newer version" in problems[0]


@check("storage: atomic write survives a crash mid-write")
def check_atomic():
    path = scratch_path("atomic")
    save_all(path, FIVE_ROUNDS[:2])
    before = snapshot(path)
    folder = os.path.dirname(path)

    def crash_halfway(f, text):
        f.write(text[: len(text) // 2])  # half the file...
        f.flush()
        raise OSError("simulated crash mid-write")  # ...then the "crash"

    def crash_on_swap(src, dst):
        raise OSError("simulated crash before the swap")

    real_write, real_replace = storage._write_and_sync, storage.os.replace
    for name, patch in [("half-written", ("_write_and_sync", crash_halfway)),
                        ("before swap", ("replace", crash_on_swap))]:
        target = storage if patch[0] == "_write_and_sync" else storage.os
        setattr(target, patch[0], patch[1])
        try:
            storage.save_round(FIVE_ROUNDS[2], path)
            raise AssertionError(f"{name}: the simulated crash didn't happen")
        except OSError:
            pass
        finally:
            storage._write_and_sync, storage.os.replace = real_write, real_replace
        assert snapshot(path) == before, f"{name}: the real file changed"
        leftovers = [f for f in os.listdir(folder) if f.endswith(".tmp")]
        assert leftovers == [], f"{name}: temp files left: {leftovers}"
    assert len(storage.load_rounds(path)[0]) == 2


@check("storage: only the newest 30 backups are kept")
def check_backup_pruning():
    path = scratch_path("pruning")
    for day in range(1, 36):  # 35 saves -> 34 backups made, 30 kept
        storage.save_round(round_with(date=date(2026, 8, day % 28 + 1), score=80 + day), path)
    backups = list_dir(os.path.join(os.path.dirname(path), "backups"))
    assert len(backups) == 30, len(backups)
    # The newest backup = the file just before the last save (34 rounds).
    newest = os.path.join(os.path.dirname(path), "backups", backups[-1])
    assert data_rows(newest) == 34, data_rows(newest)


# =====================================================================
# 6. Weakness engine on the hand-built set
# =====================================================================
@check("weakness engine: hand-computed 5-round answer")
def check_weakness_engine():
    path = scratch_path("weakness")
    save_all(path, FIVE_ROUNDS + [NINE_HOLE])
    rounds, _ = storage.load_rounds(path)
    bench = load_benchmarks()

    flagged, _ = putting_confound(rounds, bench)
    assert flagged is True, "putting should be flagged (avg GIR 2.0 < 3.024)"

    ranked = rank_areas(average_strokes_lost(rounds, bench), flagged)
    expected = [("Approach (greens)", 1.8328), ("Short game (up & down)", 0.456),
                ("Penalties", 0.35), ("Putting", 2.5)]
    got = list(zip(ranked["area"], ranked["strokes_lost"]))
    assert [a for a, _ in got] == [a for a, _ in expected], f"order: {got}"
    for (area, value), (_, want) in zip(got, expected):
        assert abs(value - want) < 1e-9, f"{area}: {value} vs hand-computed {want}"
    assert ranked.loc[ranked["area"] == "Putting", "flag"].item() == "confounded by GIR"
    assert biggest_leak(ranked)["area"] == "Approach (greens)"

    value, _ = estimated_index(rounds)
    assert value == 17.0, f"index {value}, hand-computed 17.0 (9-hole round counted?)"
    for area, value in got:
        INFO.append(f"{area:24s} {value:.4f}")


# =====================================================================
# 7. The app itself, run headless with Streamlit's AppTest
# =====================================================================
from streamlit.testing.v1 import AppTest  # noqa: E402


def run_app(path):
    """Start the tracker app pointed at a scratch rounds file."""
    assert os.path.abspath(path).startswith(SCRATCH), "refusing a non-scratch path"
    os.environ["GOLF_ROUNDS_PATH"] = path
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    assert not at.exception, f"app crashed: {at.exception[0].value}"
    return at


def page_text(at):
    """All the text on the page, as one string, to search in."""
    parts = []
    for kind in ("title", "header", "subheader", "markdown", "caption",
                 "info", "warning", "error", "success"):
        parts += [str(e.value) for e in getattr(at, kind)]
    return "\n".join(parts)


def metric(at, label_start):
    for m in at.metric:
        if m.label.startswith(label_start):
            return m.value
    raise AssertionError(f"no metric starting {label_start!r}")


def fill_and_save(at, r):
    """Type a round into the form and press Save round."""
    for key, value in r.items():
        if value is None:
            continue
        if key == "date":
            at.date_input(key="date").set_value(value)
        elif key in ("course_name", "tee"):
            at.text_input(key=key).set_value(value)
        elif key == "holes":
            at.radio(key="holes").set_value(value)
        else:
            at.number_input(key=key).set_value(value)
    at.button(key="save_round").click().run()
    assert not at.exception, f"app crashed: {at.exception[0].value}"


@check("app: no rounds (first-run guide)")
def check_app_empty():
    at = run_app(scratch_path("app_empty"))
    text = page_text(at)
    for phrase in ("How to log your first round", "par 4s and par 5s only",
                   "runs locally", "Known limitations", "Unresolved #5", "Unresolved #6"):
        assert phrase in text or any(phrase in e.label for e in at.expander), phrase


@check("app: incomplete or impossible form refused")
def check_app_refuses():
    path = scratch_path("app_refuse")
    at = run_app(path)
    fill_and_save(at, {"course_name": "Test Course"})
    errors = [e.value for e in at.error]
    assert "Score is required." in errors, errors
    assert not os.path.exists(path), "an incomplete round was saved"

    fill_and_save(at, round_with(putts=80))
    errors = [e.value for e in at.error]
    assert any(e.startswith("Putts (80)") for e in errors), errors
    assert not os.path.exists(path), "an impossible round was saved"


@check("app: complete form saved, then duplicate warning")
def check_app_save_and_duplicate():
    path = scratch_path("app_save")
    at = run_app(path)
    fill_and_save(at, round_with(date=date(2025, 6, 1)))
    assert "Round saved." in [e.value for e in at.success]
    assert data_rows(path) == 1
    assert at.number_input(key="score").value is None, "form wasn't cleared"

    # Same date + course + tee again -> asked, not saved.
    fill_and_save(at, round_with(date=date(2025, 6, 1), score=99))
    assert "You already have a round" in page_text(at)
    assert data_rows(path) == 1, "duplicate saved without asking"
    at.button(key="cancel_duplicate").click().run()
    assert data_rows(path) == 1 and "Not saved." in page_text(at)

    # And "it's a different round" does save it.
    fill_and_save(at, round_with(date=date(2025, 6, 1), score=99))
    at.button(key="confirm_duplicate").click().run()
    assert data_rows(path) == 2


@check("app: saved rounds (report, index, bad row shown)")
def check_app_saved_rounds():
    path = scratch_path("app_saved")
    save_all(path, FIVE_ROUNDS + [NINE_HOLE])
    with open(path, "a") as f:
        f.write("2026-09-07,Test Course,White,72,113,72,18,,15,abc,6,14,2,36,3,14,2,3,,,2\n")
    before = snapshot(path)

    at = run_app(path)
    text = page_text(at)
    assert metric(at, "Handicap index") == "17.0", metric(at, "Handicap index")
    assert "ESTIMATED, not official" in text
    assert "Based on 5 18-hole round(s); 1 9-hole round(s) not included." in text
    assert "1 row(s) in your rounds file have problems" in text
    assert "Biggest leak this round: **Approach (greens)**" in text
    assert snapshot(path) == before, "viewing the app changed the file"


@check("app: fewer than 3 rounds")
def check_app_few_rounds():
    path = scratch_path("app_few")
    save_all(path, FIVE_ROUNDS[:2])
    at = run_app(path)
    assert metric(at, "Handicap index") == "need 3 rounds"
    assert "Too few rounds to say much" in page_text(at)


@check("app: Export / Import")
def check_app_export_import():
    source = scratch_path("app_export")
    save_all(source, FIVE_ROUNDS + [NINE_HOLE])
    at = run_app(source)
    assert len(at.download_button) == 1, "no export button"

    target = scratch_path("app_import")
    at = run_app(target)
    at.file_uploader(key="import_file").set_value(
        ("my_rounds.csv", storage.export_csv(source).encode(), "text/csv")).run()
    at.button(key="import").click().run()
    assert not at.exception, at.exception
    assert "Imported 6 new round(s)." in [e.value for e in at.success][0]
    assert storage.load_rounds(target)[0].equals(storage.load_rounds(source)[0])
    assert metric(at, "Handicap index") == "17.0"


# =====================================================================
# 8. The real rounds file was never touched
# =====================================================================
@check("data/my_rounds.csv untouched")
def check_real_file():
    assert snapshot(REAL_ROUNDS) == REAL_BEFORE, "data/my_rounds.csv changed!"
    assert list_dir(REAL_BACKUPS) == REAL_BACKUPS_BEFORE, "data/backups changed!"
    INFO.append("data/my_rounds.csv " + ("does not exist (as before)" if REAL_BEFORE is None
                                         else "is byte-identical to before"))


if __name__ == "__main__":
    failed = 0
    for name, func in CHECKS:
        INFO.clear()
        try:
            func()
            print(f"PASS  {name}")
        except Exception as e:  # any error = a failed check, keep going
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                print("      " + traceback.format_exc().strip().splitlines()[-2].strip())
        for line in INFO:
            print(f"        {line}")

    print(f"\n{len(CHECKS) - failed} of {len(CHECKS)} checks passed.")
    if failed:
        # Keep the scratch files so you can look at what went wrong.
        print(f"Scratch files kept for inspection: {SCRATCH}")
    else:
        shutil.rmtree(SCRATCH)
    sys.exit(1 if failed else 0)
