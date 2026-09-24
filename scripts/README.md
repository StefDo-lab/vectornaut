# Helper Scripts

One-off helper and maintenance scripts. They are not tests and are not part of any suite. Run them from the repository root, because they use paths relative to the current directory:

```powershell
& .\.venv\Scripts\python.exe scripts\check_open_encoding.py
```

| Script | Purpose |
| --- | --- |
| `check_encodings.py` | Checks whether `web_server.py`, `static/app.js` and `static/index.html` decode as UTF-8 and cp1252, and lists non-ASCII bytes. |
| `check_open_encoding.py` | Scans all `.py` files under the current directory for `open(` calls that do not pass `encoding`. |
| `cleanup_corrupted_history.py` | **Deletes** everything in `history/`, `reports/`, `generated_scripts/` and `generated_tests/` in the current directory (it ignores `VECTORNAUT_DATA_DIR` and does not touch `vectornaut.sqlite3`). |
| `read_transcript.py` | Obsolete debugging aid: prints user inputs and error steps from one hardcoded Windows agent transcript (`C:\Users\Stefan\...\transcript.jsonl`). |
| `search_log.py` | Obsolete debugging aid: extracts step 2519 from the same hardcoded transcript into `step_2519.txt`. |
