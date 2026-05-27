# Vectornaut Operations

## Local Development

```powershell
& .\.venv\Scripts\python.exe -m uvicorn web_server:app --host 127.0.0.1 --port 8080 --reload
```

Health check:

```text
GET /api/health
```

## Configuration

```text
GEMINI_API_KEY=...
VECTORNAUT_DATA_DIR=.
VECTORNAUT_MAX_EPOCHS=1000
VECTORNAUT_MAX_OPTIMIZATION_ROUNDS=10
VECTORNAUT_SCRIPT_TIMEOUT_SECONDS=60
```

If `GEMINI_API_KEY` is missing, `/api/run` falls back to mock mode unless the lower-level live model call is explicitly reached.

## Storage

The current storage model is hybrid:

- Full run documents: `history/*.json`
- Markdown reports: `reports/*.md`
- History index: `vectornaut.sqlite3`
- Evaluation records: `eval_runs/*.json`

The JSON files are still the source of truth. SQLite is currently an index for dashboard listing and a stepping stone toward a production database.

## Production Direction

For public/scalable operation:

- Keep the FastAPI backend as a separately deployable API service.
- Move long-running solver jobs behind a queue/worker.
- Replace local SQLite/file storage with Postgres plus object storage for reports, plots, generated scripts, and model artifacts.
- Add request authentication before exposing live model or dynamic-script execution.
- Run generated scripts only in a sandboxed worker with CPU/memory/time limits.
