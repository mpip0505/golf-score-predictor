# Shared helpers: reading and writing YOUR rounds (data/my_rounds.csv).
#
# That file is gitignored on purpose: it's personal data and shouldn't end
# up on GitHub. data/my_rounds.example.csv (header only, safe to commit)
# shows the expected columns, e.g. if you want to type rounds into a
# spreadsheet and import them.
#
# Everything that touches the file lives here; the rules for what counts
# as a valid round live in rounds.py. Every write goes through
# validate_round() first, whether it comes from the form or an import.
#
# DATA SAFETY RULES (the testing phase depends on these):
#
#   1. Atomic writes. We never write into my_rounds.csv directly. We write
#      the whole new file to a temporary file in the SAME folder, then
#      os.replace() swaps it in. On the same disk, that swap is "atomic":
#      it either fully happens or doesn't happen at all. So if the app
#      crashes halfway through writing, you're left with the old file
#      (plus a stray .tmp file), never a half-written one.
#
#   2. Backups. Before any write that changes an existing file, the
#      current file is copied to data/backups/my_rounds_YYYYMMDD_HHMMSS.csv.
#      Only the newest 30 are kept so the folder doesn't grow forever.
#
#   3. Nothing is ever silently dropped. Loading checks every row; a row
#      that fails is reported (row number + what's wrong) and left out of
#      the REPORTS, but it stays in the FILE exactly as it was. Writes start
#      from the raw file text, not from the filtered rounds, so a bad row
#      can't be lost by saving a new round.
#
#   4. Schema version. Every row has a schema_version (see below). A file
#      written by a NEWER version of the tracker is refused, not guessed at.

import os
import re
import shutil
import tempfile
from datetime import datetime

import pandas as pd

from rounds import (
    LABELS,
    REQUIRED_COLUMNS,
    ROUND_COLUMNS,
    format_value,
    parse_round,
    validate_round,
)

DEFAULT_DATA_DIR = "data"
ROUNDS_FILE = "my_rounds.csv"

# ---------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------
# "Schema" = which columns the file has and what they mean. If a later
# phase changes that (renames a column, changes a unit...), the loader
# needs to know which version each row was written with.
#
# Why a COLUMN rather than a small sidecar file (e.g. my_rounds.version)?
#   - It travels with the data: Export/Import and backups carry it
#     automatically. A sidecar file could get separated or out of date.
#   - One file means one atomic write; two files can't be swapped together.
#   - The cost: the same number repeats on every row. That's harmless.
#
# Version history (add a line here, with a migration note, for every change):
#   1 - Phase 8/9 files. No schema_version column (a blank cell means 1).
#   2 - Testing phase (2026-10-12). Added `holes` (blank = 18) and
#       `schema_version`. Version 1 rows are read as 18-hole rounds and are
#       left as they are in the file - no rewrite needed.
SCHEMA_VERSION = 2
SCHEMA_COLUMN = "schema_version"
FILE_COLUMNS = ROUND_COLUMNS + [SCHEMA_COLUMN]

BACKUP_FOLDER = "backups"
BACKUPS_TO_KEEP = 30


class SchemaTooNewError(Exception):
    """The file was written by a newer version of the tracker than this one."""


def data_dir():
    """
    The folder your rounds (and their backups/) live in: data/, unless the
    environment variable ROUND_DATA_DIR names another folder.

    ROUND_DATA_DIR is "practice mode": point it at a scratch folder and the
    app reads and writes ONLY there, so you can try things out (or the
    self-check can run) without any chance of touching your real rounds:

        ROUND_DATA_DIR=/tmp/golf_scratch streamlit run 08_round_tracker.py

    It's read on every call, not once at import time, so changing it always
    takes effect. This is the only place it's read.
    """
    return os.environ.get("ROUND_DATA_DIR") or DEFAULT_DATA_DIR


def rounds_path():
    """Your rounds file: <data folder>/my_rounds.csv."""
    return os.path.join(data_dir(), ROUNDS_FILE)


def practice_folder():
    """
    The practice folder if ROUND_DATA_DIR points somewhere other than the
    real data/ folder, else None. The app shows a banner when it's set.
    """
    folder = data_dir()
    if os.path.abspath(folder) == os.path.abspath(DEFAULT_DATA_DIR):
        return None
    return os.path.abspath(folder)


# =====================================================================
# Reading
# =====================================================================
def _read_raw(path):
    """
    The file exactly as text: one row per line, every cell a string, blank
    cells as "". No type guessing, no rows dropped.

    dtype=str and keep_default_na=False tell pandas "don't convert
    anything": "72" stays "72", and an empty cell stays "" (instead of NaN).
    That's what lets us write rows back out exactly as they came in.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=FILE_COLUMNS, dtype=str)
    try:
        # utf-8-sig also accepts the invisible "BOM" mark Excel adds at the
        # start of CSVs, which would otherwise glue itself onto "date".
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        # A completely empty file (0 bytes) has no rounds in it.
        return pd.DataFrame(columns=FILE_COLUMNS, dtype=str)

    _check_schema(raw, where="Your rounds file")

    # Older files lack newer columns: add them, blank. Unknown extra
    # columns (e.g. one you added in a spreadsheet) are KEPT, at the end,
    # so a write never deletes anything.
    for col in FILE_COLUMNS:
        if col not in raw.columns:
            raw[col] = ""
    extra = [c for c in raw.columns if c not in FILE_COLUMNS]
    return raw[FILE_COLUMNS + extra]


def _check_schema(raw, where):
    """Raise SchemaTooNewError if any row was written by a newer version."""
    if SCHEMA_COLUMN not in raw.columns:
        return  # version 1 file
    versions = pd.to_numeric(raw[SCHEMA_COLUMN], errors="coerce")
    newest = versions.max()
    if pd.notna(newest) and newest > SCHEMA_VERSION:
        raise SchemaTooNewError(
            f"{where} was written by a newer version of the tracker (schema "
            f"{newest:g}); this version understands up to {SCHEMA_VERSION}. "
            f"Nothing was loaded or changed."
        )


def _row_problems(raw_row):
    """Parse + validate one raw row. Returns (round_dict, list of messages)."""
    version = raw_row.get(SCHEMA_COLUMN, "").strip()
    if version not in ("", "1", "2"):
        # Anything else that isn't "newer" (already refused) is a typo.
        return None, [f"schema_version '{version}' isn't a known version."]
    r, problems = parse_round(raw_row)
    if not problems:
        problems = validate_round(r)
    return r, [f"{LABELS[field]}: {message}" for field, message in problems]


def load_rounds(path=None):
    """
    Your rounds, checked, oldest first.

    Returns (rounds, skipped):
      - rounds:  a DataFrame of the VALID rounds (ROUND_COLUMNS).
      - skipped: one dict per row that failed, with its row number in the
                 file, what's wrong and its values - for the app to show.
                 Skipped rows stay in the file untouched.

    Raises SchemaTooNewError for a file from a newer version.
    """
    path = path or rounds_path()
    raw = _read_raw(path)

    good, skipped = [], []
    # .iterrows() gives (index, row) pairs, one per CSV row.
    for i, raw_row in raw.iterrows():
        r, problems = _row_problems(raw_row.to_dict())
        if problems:
            # i + 2: +1 because people count rows from 1, +1 for the header line.
            skipped.append({"row": i + 2, "problems": problems,
                            "values": raw_row.to_dict()})
        else:
            good.append(r)

    rounds = pd.DataFrame(good, columns=ROUND_COLUMNS)
    rounds["date"] = pd.to_datetime(rounds["date"])
    # Sort by date so "newest 20 rounds" really means the most recent 20.
    # kind="stable" keeps two rounds on the same date in the order entered.
    rounds = rounds.sort_values("date", kind="stable").reset_index(drop=True)
    return rounds, skipped


def find_duplicates(r, rounds):
    """
    Saved rounds with the same date, course and tees as round `r` - probably
    the same round entered twice. Course and tee are compared ignoring
    capitals and spaces at the ends, so "Kelab Golf " matches "kelab golf".
    """
    def same_text(column, value):
        return rounds[column].str.strip().str.casefold() == str(value).strip().casefold()

    same_date = rounds["date"].dt.date == pd.Timestamp(r["date"]).date()
    return rounds[same_date & same_text("course_name", r["course_name"])
                  & same_text("tee", r["tee"])]


# =====================================================================
# Writing (backup first, then an atomic replace)
# =====================================================================
def _backup(path):
    """
    Copy the current file into the backups folder (next to it). Returns the
    backup's path, or None if there was no file to back up yet.
    """
    if not os.path.exists(path):
        return None

    folder = os.path.join(os.path.dirname(path) or ".", BACKUP_FOLDER)
    os.makedirs(folder, exist_ok=True)

    stem = os.path.splitext(os.path.basename(path))[0]  # "my_rounds"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Two saves in the same second would get the same name, so the 2nd,
    # 3rd... get _02, _03. We count up from the HIGHEST number already used
    # this second (not the first free name): otherwise a name freed by
    # pruning could be reused, sort as the oldest, and be deleted at once.
    same_second = [n for s, n, _ in _list_backups(folder, stem) if s == stamp]
    n = max(same_second) + 1 if same_second else 1
    name = f"{stem}_{stamp}.csv" if n == 1 else f"{stem}_{stamp}_{n:02d}.csv"
    target = os.path.join(folder, name)

    # copy2 copies the contents AND the file's timestamps.
    shutil.copy2(path, target)
    _prune_backups(folder, stem)
    return target


def _list_backups(folder, stem):
    """
    Our backups in `folder`, oldest first, as (stamp, number, filename).

    Only files that look exactly like our backups are listed, so nothing
    else you put in that folder can ever be deleted by pruning.
    """
    pattern = re.compile(rf"^{re.escape(stem)}_(\d{{8}}_\d{{6}})(?:_(\d+))?\.csv$")
    found = []
    for name in os.listdir(folder):
        match = pattern.match(name)
        if match:
            # No _NN on the end = the first backup of that second (number 1).
            found.append((match.group(1), int(match.group(2) or 1), name))
    # Sort by time, then by number AS A NUMBER (so _10 comes after _09).
    return sorted(found)


def _prune_backups(folder, stem):
    """Delete all but the newest BACKUPS_TO_KEEP backups of this file."""
    backups = _list_backups(folder, stem)
    for _, _, name in backups[:-BACKUPS_TO_KEEP]:
        os.remove(os.path.join(folder, name))


def _write_and_sync(f, text):
    """
    Write text into an open file and force it onto the disk.

    flush() pushes Python's buffer to the operating system; os.fsync() asks
    the OS to put it physically on disk, so a power cut right after the
    swap can't leave an empty file. (A separate function so the self-check
    can replace it with one that "crashes" halfway through.)
    """
    f.write(text)
    f.flush()
    os.fsync(f.fileno())


def _atomic_write_text(path, text):
    """Write `text` to `path` via a temp file + os.replace (see rule 1 above)."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)

    # mkstemp creates a new, uniquely named file and gives us its handle.
    # It must be in the SAME folder: os.replace is only atomic within one disk.
    stem = os.path.splitext(os.path.basename(path))[0]
    handle, tmp_path = tempfile.mkstemp(dir=folder, prefix=f".{stem}_", suffix=".tmp")
    try:
        # newline="" writes the line endings exactly as pandas made them.
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as f:
            _write_and_sync(f, text)
        os.replace(tmp_path, path)  # the atomic swap
    except BaseException:
        # Something failed before the swap: the real file is untouched.
        # Tidy up the half-written temp file, then let the error continue
        # so the app can say "not saved".
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _write_raw(raw, path):
    """Back up the current file, then atomically write the raw table."""
    _backup(path)
    _atomic_write_text(path, raw.to_csv(index=False))


def _as_raw_row(r):
    """A validated round as one row of text, stamped with the current schema."""
    row = {col: format_value(r.get(col)) for col in ROUND_COLUMNS}
    row[SCHEMA_COLUMN] = str(SCHEMA_VERSION)
    return row


def _append_rows(raw, new_rows):
    """Add rows (dicts of text) to the raw table. Extra columns stay blank."""
    combined = pd.concat([raw, pd.DataFrame(new_rows)], ignore_index=True)
    return combined.fillna("")


def save_round(r, path=None):
    """
    Validate one round and add it to the file.
    Returns a list of (field, message) problems; empty means it was saved.

    The duplicate check (same date + course + tees) isn't done here: a
    duplicate CAN be legitimate (two rounds on one day), so the app asks
    you to confirm before calling this.

    May raise SchemaTooNewError, or OSError if the disk write fails - in
    both cases the file is unchanged.
    """
    path = path or rounds_path()
    problems = validate_round(r)
    if problems:
        return problems

    # Start from the RAW file (every row, even ones that fail validation
    # today), so saving can never drop a row.
    raw = _read_raw(path)
    _write_raw(_append_rows(raw, [_as_raw_row(r)]), path)
    return []


# =====================================================================
# Export / Import
# =====================================================================
def export_csv(path=None):
    """
    Your rounds file as CSV text, for the app's download button.

    It's an exact copy of the file - including any rows that currently fail
    validation - because an export is also a backup, and a backup must
    contain everything.
    """
    path = path or rounds_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=FILE_COLUMNS).to_csv(index=False)
    with open(path, encoding="utf-8-sig") as f:
        return f.read()


def _comparison_key(raw_row):
    """
    The row in a standard form, for spotting exact duplicates: "72" and
    "72.0" both become "72", and a blank holes cell becomes "18".
    """
    r, problems = parse_round(raw_row)
    if problems:
        # Unreadable row: compare its text as-is.
        return tuple(str(raw_row.get(c, "")).strip() for c in ROUND_COLUMNS)
    return tuple(format_value(r[c]) for c in ROUND_COLUMNS)


def import_rounds(file, path=None):
    """
    Merge rounds from an uploaded CSV into your saved rounds.

    All-or-nothing: if ANY row fails validation, nothing is imported and you
    get the list of problems by row number. Half-importing a file would be
    confusing to undo.

    Rows that exactly match a round you already have are skipped, so
    re-importing your own export doesn't create duplicates. Rows that only
    share date + course + tees with a saved round ARE imported, but listed
    as warnings so you can check them.

    Returns (number_added, problems, warnings).
    """
    path = path or rounds_path()
    try:
        incoming = pd.read_csv(file, dtype=str, keep_default_na=False,
                               encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        return 0, [f"Couldn't read that file as a CSV ({e})."], []

    missing = [c for c in REQUIRED_COLUMNS if c not in incoming.columns]
    if missing:
        return 0, [f"File is missing columns: {missing}"], []
    try:
        _check_schema(incoming, where="That file")
    except SchemaTooNewError as e:
        return 0, [str(e)], []

    # 1. Check every row before changing anything.
    problems, parsed = [], []
    for i, raw_row in incoming.iterrows():
        r, row_problems = _row_problems(raw_row.to_dict())
        problems += [f"Row {i + 2}: {p}" for p in row_problems]
        parsed.append(r)
    if problems:
        return 0, problems, []

    # 2. Skip exact duplicates (of saved rows, or of earlier rows in the file).
    raw = _read_raw(path)
    seen = {_comparison_key(row.to_dict()) for _, row in raw.iterrows()}
    saved, _ = load_rounds(path)
    new_rows, warnings = [], []
    for i, r in enumerate(parsed):
        key = tuple(format_value(r[c]) for c in ROUND_COLUMNS)
        if key in seen:
            continue
        seen.add(key)
        if len(find_duplicates(r, saved)) > 0:
            warnings.append(f"Row {i + 2}: you already have a round on {format_value(r['date'])} "
                            f"at {r['course_name']} ({r['tee']}) with different numbers. "
                            f"Both are kept - check it isn't the same round twice.")
        new_rows.append(_as_raw_row(r))

    # 3. Write only if something actually changes (no pointless backup).
    if new_rows:
        _write_raw(_append_rows(raw, new_rows), path)
    return len(new_rows), [], warnings
