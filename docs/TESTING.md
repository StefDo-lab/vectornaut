# Vectornaut Testing

## Fast Local Suite

Run the stable unit and smoke tests:

```powershell
& .\.venv\Scripts\python.exe -m unittest test_evaluation_unit.py test_pipeline_unit.py test_formulator_unit.py test_optimizer_unit.py test_synthesis_unit.py test_api_smoke.py
```

The API smoke test uses FastAPI `TestClient` and writes runtime data into a temporary `VECTORNAUT_DATA_DIR`.

## Manual Server Smoke

Start the server:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn web_server:app --host 127.0.0.1 --port 8080 --reload
```

Then check:

```powershell
curl.exe -i http://127.0.0.1:8080/api/health
curl.exe -i http://127.0.0.1:8080/api/history
```

## Live Tests

Files with names such as `test_*_live.py` exercise real model/API behavior. They are useful during exploration but are not part of the stable local suite because they can depend on API keys, network, model behavior, or longer runtimes.

## Runtime Data

Runtime outputs are ignored by Git:

```text
history/
reports/
eval_runs/
generated_scripts/
generated_tests/
saved_models/
static/plots/
vectornaut.sqlite3
```

Use `VECTORNAUT_DATA_DIR` to redirect all mutable run data:

```powershell
$env:VECTORNAUT_DATA_DIR="C:\tmp\vectornaut-data"
```
