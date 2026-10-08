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

import os

import pandas as pd

from rounds import OPTIONAL_DEFAULTS, ROUND_COLUMNS, add_missing_columns, validate_round

ROUNDS_PATH = "data/my_rounds.csv"


def load_rounds(path=ROUNDS_PATH):
    """All saved rounds, oldest first. An empty table if none saved yet."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=ROUND_COLUMNS)
    rounds = pd.read_csv(path, parse_dates=["date"])
    # Older files may be missing newer optional columns - add them.
    rounds = add_missing_columns(rounds)[ROUND_COLUMNS]
    # Sort by date so "newest 20 rounds" really means the most recent 20.
    # kind="stable" keeps two rounds on the same date in the order entered.
    return rounds.sort_values("date", kind="stable").reset_index(drop=True)


def _write_all(rounds, path):
    """Overwrite the file with this table (dates written as YYYY-MM-DD)."""
    rounds[ROUND_COLUMNS].to_csv(path, index=False, date_format="%Y-%m-%d")


def save_round(r, path=ROUNDS_PATH):
    """Validate one round and add it to the file. Returns the problem list."""
    problems = validate_round(r)
    if problems:
        return problems

    new = add_missing_columns(pd.DataFrame([r]))
    new["date"] = pd.to_datetime(new["date"])
    # Rewriting the whole file (instead of appending one line) is the
    # simplest way to make sure an older file gets upgraded to the current
    # columns. At a few hundred rounds this is still instant.
    _write_all(pd.concat([load_rounds(path), new], ignore_index=True), path)
    return []


def export_csv(path=ROUNDS_PATH):
    """Your rounds as CSV text, for the app's download button."""
    rounds = load_rounds(path)
    return rounds.to_csv(index=False, date_format="%Y-%m-%d")


def import_rounds(file, path=ROUNDS_PATH):
    """
    Merge rounds from an uploaded CSV into your saved rounds.

    All-or-nothing: if ANY row fails validation, nothing is imported and you
    get the list of problems by row number. Half-importing a file would be
    confusing to undo.

    Rows that exactly match a round you already have are skipped, so
    re-importing your own export doesn't create duplicates.

    Returns (number_added, problems).
    """
    incoming = pd.read_csv(file)

    # Optional columns may be absent (e.g. a file from before they existed).
    missing = [c for c in ROUND_COLUMNS
               if c not in incoming.columns and c not in OPTIONAL_DEFAULTS]
    if missing:
        return 0, [f"File is missing columns: {missing}"]

    incoming = add_missing_columns(incoming)[ROUND_COLUMNS]
    # errors="coerce" turns an unreadable date into NaT (blank), which
    # validate_round then reports as "date is required".
    incoming["date"] = pd.to_datetime(incoming["date"], errors="coerce")

    problems = []
    for i, row in incoming.iterrows():
        for p in validate_round(row):
            # i + 2: +1 because people count rows from 1, +1 for the header line.
            problems.append(f"Row {i + 2}: {p}")
    if problems:
        return 0, problems

    existing = load_rounds(path)
    combined = pd.concat([existing, incoming], ignore_index=True)
    # Compare as text so 72 and 72.0 (int vs float) count as the same.
    duplicate = combined.astype(str).duplicated()
    combined = combined[~duplicate]
    added = len(combined) - len(existing)

    combined = combined.sort_values("date", kind="stable")
    _write_all(combined, path)
    return added, []
