# Testing the round tracker with real rounds

A checklist for the testing weeks. Work through it once at the start, then keep logging rounds
normally and come back to the last three sections whenever something looks off.

The tracker is `08_round_tracker.py`. Your rounds live in `data/my_rounds.csv`.

## Before you start

1. `source venv/bin/activate`, then `python selfcheck.py`.
   **Expect:** every line says `PASS`, ending `17 of 17 checks passed.` (Lines starting
   `flag:` are information, not failures.) If anything says `FAIL`, stop and note it down.
   Ignore the "missing ScriptRunContext" warnings at the top: they come from Streamlit's test
   tool and don't mean anything is wrong.
2. Start the app: `streamlit run 08_round_tracker.py --server.address localhost`
   **Expect:** a browser tab opens. `--server.address localhost` keeps the app on this computer.
   Without it, Streamlit also serves the page to your local network (the "Network URL" line), so
   anyone on the same Wi-Fi could open it.

## The checklist

| # | Do this | Expect |
|---|---|---|
| 1 | Open the app with no rounds saved. | A "How to log your first round" box, the list of what each stat means, a note that the app runs locally, and a **Known limitations** expander. No errors. |
| 2 | **Log a round:** fill in a real round with every box, including the optional ones if you have them. Press **Save round**. | "Round saved." The form empties. The round appears under Round report and All rounds. `data/my_rounds.csv` now exists. |
| 3 | **Blank optional fields:** log your next real round leaving Length, Adjusted score and PCC empty. | Saved. In All rounds, adjusted_score and pcc are blank (not 0). The differential uses your gross score. |
| 4 | **Backup:** after step 3, look in `data/backups/`. | One file, `my_rounds_YYYYMMDD_HHMMSS.csv`, holding your rounds as they were *before* the save in step 3. (The very first save makes no backup: there was nothing to back up yet.) Every later save adds one, and only the newest 30 are kept. |
| 5 | **Duplicate:** type in a round with the same date, course and tees as one you've saved. Press Save round. | A yellow warning asks whether it's a second round that day. Press **"Same round - don't save"**. You see "Not saved." and the number of rounds logged doesn't change. (Use "It's a different round - save it" only for a real second round.) |
| 6 | **Invalid input:** fill in a full round, then change one box at a time to something wrong and press Save round. Try: Score empty; Fairways hit more than Fairways possible; a date in the future; Total putts 80 with Score 90; 3-putts 5 with 12 putts; Up-and-down saves more than attempts. | Nothing is saved. A red box at the top of the form says how many things to fix, and each problem is in plain words **under the box it's about**. Your other entries stay in the form. Reload the page to clear it afterwards. |
| 7 | **Export:** open Export / Import, press **Download my rounds**. Open the file in a text editor (TextEdit, not Excel). | The same rounds as All rounds, one per line, dates written like `2026-10-12`. |
| 8 | **Import into a fresh scratch copy:** stop the app (Ctrl+C). Start a practice copy with `GOLF_ROUNDS_PATH=/tmp/golf_scratch/my_rounds.csv streamlit run 08_round_tracker.py --server.address localhost`. Import the file from step 7 and press **Import**. | The caption near the top shows the `/tmp/golf_scratch` path, and the first-run guide is showing. After importing: "Imported N new round(s)", where N is your number of rounds, and the same index and reports as your real data. Press Import again: "Imported 0 new round(s)". Afterwards stop the app and delete `/tmp/golf_scratch`. `data/my_rounds.csv` isn't touched. |
| 9 | **Restart:** start the app normally again (step 2 command). | All your rounds are still there. |
| 10 | **Index:** watch the "Handicap index (ESTIMATED, not official)" box as you log rounds. | It shows "need 3 rounds" until you have **3 rounds of 18 holes**, and then a number. 9-hole rounds don't count. Below 3 rounds, the strokes-lost sections show "Too few rounds to say much yet", and each section says how many rounds it's based on. |
| 11 | **A 9-hole round** (if you play one): pick 9 under Holes played. | Saved and listed in All rounds with an empty differential. It is not in the index or the strokes-lost report. |

## Keep in mind while testing (unresolved)

These come from the benchmark source (Shot Scope eBook, 2021) and are still open. They're also
listed in the app under **Known limitations**.

- **#5:** the source doesn't define what counts as an Up & Down attempt, or whether its fairway %
  leaves out par 3s. This app counts **par 4s and 5s only** and defines an attempt as a missed
  green where you tried to get down in two. If your short-game or driving gaps look odd, this
  may be why.
- **#6:** the source doesn't say whether its "handicap" is a WHS Handicap Index (its data predates
  WHS). The app treats it as roughly your index.

## When something looks wrong, write down

Keep a note (phone or paper) with one entry per problem:

1. **What I entered:** the date, course and tees, and the exact numbers you typed (or the file
   you imported).
2. **What I saw:** the exact message or number on screen. A screenshot is ideal.
3. **What I expected,** and why. For example: "index should be lower, my best round was 84".
4. **When:** the date and time, so the matching backup in `data/backups/` can be found.

Don't try to fix `data/my_rounds.csv` by hand. If a row shows up under "rows with problems",
note it. It's still in the file and nothing has been lost.

## Backing up `data/my_rounds.csv` outside the repo

The file is **gitignored**, so git and GitHub never have a copy. `data/backups/` sits inside the
project folder, so deleting or losing that folder loses them too. Once a week (or after every
round, it's quick):

- In the app: **Export / Import → Download my rounds**, and keep the file somewhere outside the
  project, such as iCloud Drive or a USB stick, with the date in its name.
- Or in the terminal, from the project folder:
  `mkdir -p ~/Documents/golf-backups && cp data/my_rounds.csv ~/Documents/golf-backups/my_rounds_$(date +%Y%m%d).csv`

To restore: stop the app, then copy a backup over `data/my_rounds.csv` (from `data/backups/` or
your outside copy), and start the app again. Don't open the CSV in Excel or Numbers and save it.
They can rewrite dates (`2026-10-12` → `12/10/2026`), and those rows will then be flagged as
problems.
