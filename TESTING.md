# Testing the round tracker with real rounds

A checklist for the testing weeks. Do **Part A (practice mode)** first: it uses a throwaway folder,
so you can try anything, including the things that should fail, without going near your real
rounds. Then do **Part B** with real data, and come back to the last sections whenever something
looks off.

The tracker is `08_round_tracker.py`. Your real rounds live in `data/my_rounds.csv`, with
automatic backups in `data/backups/`.

## Before you start

1. `source venv/bin/activate`, then `python selfcheck.py`.
   **Expect:** every line says `PASS`, ending `23 of 23 checks passed.` (Indented lines under a
   check are information, not failures.) If anything says `FAIL`, stop and note it down. Ignore
   the "missing ScriptRunContext" warnings at the top: they come from Streamlit's test tool.
2. The app only listens on this computer. `.streamlit/config.toml` sets
   `address = "localhost"`, so when Streamlit starts it shows only a `localhost` URL and no
   "Network URL". Other devices on your Wi-Fi can't open it.

## Part A: practice mode (nothing touches your real rounds)

Start the app pointed at a scratch folder:

```
ROUND_DATA_DIR=/tmp/golf_scratch streamlit run 08_round_tracker.py --server.address localhost
```

| # | Do this | Expect |
|---|---|---|
| A1 | Look at the top of the page. | A yellow banner: "Practice mode: data is saved in /tmp/golf_scratch, not your real rounds." If you don't see it, stop. You're on your real data. |
| A2 | First visit. | A "How to log your first round" box, what each stat means, a note that the app runs locally, and a **Known limitations** expander. Open it: a table of judgement calls (names and values) appears at the bottom. |
| A3 | **Log a made-up round** with every box filled. Press **Save round**. | "Round saved." The form empties and the round appears in the reports. |
| A4 | **Blank optional fields:** log another, leaving Length, Adjusted score and PCC empty. | Saved. In All rounds, adjusted_score and pcc are blank (not 0). |
| A5 | **No rating/slope:** log one more, leaving Course rating and Slope rating empty. | Saved. When it's selected in Round report you see "No course rating/slope for this round" instead of the expected scores. In All rounds, its `index` column says "no rating/slope, not used for index". |
| A6 | **Backup:** `ls /tmp/golf_scratch/backups/` | One backup per save after the first (the first save had nothing to back up). |
| A7 | **Duplicate:** enter the same date, course and tees as A3 and save. | A warning asks if it's a different round. "Same round - don't save" → "Not saved.", nothing added. "It's a different round - save it" → saved. |
| A8 | **Invalid input:** fill a full round, then change one box at a time and press Save round. Try: Score empty; Fairways hit more than Fairways possible; a date in the future; Total putts 80 with Score 90; 3-putts 4 with 10 putts; Up-and-down saves more than attempts. | Nothing saved. A red box at the top says how many things to fix, and each problem is in plain words **under the box it's about**. Your other entries stay. (10 putts allow at most 3 three-putts: 10 ÷ 3 rounded down.) |
| A9 | **Export, then import:** Export / Import → Download my rounds. Then stop the app, start it with `ROUND_DATA_DIR=/tmp/golf_scratch2` (same command as above), and import that file. | "Imported N new round(s)", and the same rounds and index as `/tmp/golf_scratch`. Import it again: "Imported 0 new round(s)". |
| A10 | **Index rule:** keep adding made-up 18-hole rounds (with rating and slope). | The index box says "need 3 rounds" until 3 eligible rounds, then a number labelled ESTIMATED, not official. Under it: "Uses N round(s)… Not used: …" lists 9-hole and no-rating rounds with the reason. |
| A11 | **Clean up:** stop the app, then `rm -rf /tmp/golf_scratch /tmp/golf_scratch2`. | `data/` hasn't changed: still no `my_rounds.csv` if you hadn't logged a real round yet. |

## Part B: real data

Start the app normally: `streamlit run 08_round_tracker.py`. **There must be no "Practice mode"
banner.**

| # | Do this | Expect |
|---|---|---|
| B1 | Log your first real round. | "Round saved." `data/my_rounds.csv` now exists. |
| B2 | Log your second real round. | `data/backups/` now has one file, `my_rounds_YYYYMMDD_HHMMSS.csv`, holding your rounds as they were before this save. Only the newest 30 are kept. |
| B3 | **Restart:** Ctrl+C, then start the app again. | All your rounds are still there. |
| B4 | **Duplicate check on real data:** type in the date, course and tees of a saved round and press Save. | The warning appears. Press **"Same round - don't save"**. The number of rounds logged doesn't change. |
| B5 | Watch the index as you log rounds. | "need 3 rounds" until 3 eligible rounds (18 holes, with rating and slope), then a number. Below 3 eighteen-hole rounds the strokes-lost sections say "Too few rounds to say much yet". |
| B6 | Export once a week (see the backup section below). | The file opens in a text editor with dates like `2026-10-12`. |

## Don't edit the CSV in Excel or Google Sheets

They can quietly change the date format (`2026-10-12` → `12/10/2026`) and number formats. Those
rows will then be flagged as problems (still in the file, not lost, but not used in the reports
until fixed). Dates must be **YYYY-MM-DD**. To move or fix rounds, use the app's Export and
Import. The same warning is on the Import screen.

## Keep in mind while testing (unresolved)

These come from the benchmark source (Shot Scope eBook, 2021) and are still open. They're also
listed in the app under **Known limitations**.

- **#5:** the source doesn't define what counts as an Up & Down attempt, or whether its fairway %
  leaves out par 3s. This app counts **par 4s and 5s only** and defines an attempt as a missed
  green where you tried to get down in two. If your short-game or driving gaps look odd, this
  may be why.
- **#6:** the source doesn't say whether its "handicap" is a WHS Handicap Index (its data predates
  WHS). The app treats it as roughly your index.
- The putting-confound flag never appears at index 25 and rarely at 20. That's by design (see the
  Known limitations table, `GIR_CONFOUND_GREENS`).

## When something looks wrong, write down

Keep a note (phone or paper) with one entry per problem:

1. **What I entered:** the date, course and tees, and the exact numbers you typed (or the file
   you imported).
2. **What I saw:** the exact message or number on screen. A screenshot is ideal.
3. **What I expected,** and why. For example: "index should be lower, my best round was 84".
4. **When:** the date and time, so the matching backup in `data/backups/` can be found.
5. **Practice or real?** Was the Practice mode banner showing?

Don't try to fix `data/my_rounds.csv` by hand. If a row shows up under "Rows with problems", note
it. It's still in the file and nothing has been lost.

## Backing up `data/my_rounds.csv` outside the repo

The file is **gitignored**, so git and GitHub never have a copy. `data/backups/` sits inside the
project folder, so deleting or losing that folder loses them too. Once a week (or after every
round, it's quick):

- In the app: **Export / Import → Download my rounds**, and keep the file somewhere outside the
  project, such as iCloud Drive or a USB stick, with the date in its name.
- Or in the terminal, from the project folder:
  `mkdir -p ~/Documents/golf-backups && cp data/my_rounds.csv ~/Documents/golf-backups/my_rounds_$(date +%Y%m%d).csv`

To restore: stop the app, then copy a backup over `data/my_rounds.csv` (from `data/backups/` or
your outside copy), and start the app again.
