# Vectornaut Testing

## Layout

```text
tests/                  Stable fast suite (offline, deterministic, unittest)
tests/__init__.py       Makes tests/ a package for unittest discovery
tests/live/             Live tests: need a running server, GEMINI_API_KEY, network, or long runtimes
tests/manual/           Offline exploratory scripts (solver integration checks), run by hand
scripts/                One-off helper and maintenance scripts (see scripts/README.md)
benchmarks/             Deterministic benchmark cases (eval_cases.json)
```

Every `test_*.py` directly inside `tests/` belongs to the stable suite. `tests/live/` and `tests/manual/` deliberately have no `__init__.py`, so unittest discovery does not descend into them. They also contain no `unittest.TestCase` classes, so even an accidental import would not run anything. Keep it that way: a new live or exploratory test goes into one of those folders, and a new fast offline `unittest` test goes directly into `tests/`.

All commands below are run from the repository root.

## Fast Local Suite

Run the stable unit and smoke tests:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

Run a single module:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_pipeline_unit
```

The API smoke test uses FastAPI `TestClient` and writes runtime data into a temporary `VECTORNAUT_DATA_DIR`. The generator tests (`test_script_generator.py`, `test_validation_generator.py`) use a mocked model client, but they execute the generated scripts as subprocesses. They write to `generated_scripts/` and `generated_tests/` under `VECTORNAUT_DATA_DIR` (default: the current directory) and write a temporary plot to `static/plots/`, which they remove afterwards. Set `VECTORNAUT_DATA_DIR` to a temporary folder to keep the working tree clean:

```powershell
$env:VECTORNAUT_DATA_DIR = Join-Path $env:TEMP "vectornaut-test-data"
& .\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

`test_formulator_unit.py` contains one opt-in live test. It is skipped unless `RUN_LIVE_TESTS=1` and `GEMINI_API_KEY` are set.

The full suite takes about a minute on a laptop CPU, most of it PINN training in the solver accuracy tests.

### What the suite covers

| Area | Modules |
|---|---|
| Pipeline, stages, validator, eval | `test_pipeline_unit`, `test_formulator_unit`, `test_optimizer_unit`, `test_synthesis_unit`, `test_validator_unit`, `test_evaluation_unit`, `test_benchmark_unit` |
| HTTP API in mock mode (offline ports of `tests/live/` scripts) | `test_api_smoke`, `test_api_mock_flows`, `test_chat_mock_flows`, `test_static_files` (shared setup in `offline_support.py`; it blocks network access and fails the test if a live model client is created) |
| Solver correctness against closed-form solutions | `test_solver_accuracy` (helpers in `solver_accuracy_helpers.py`), `test_solver_1d_fixes`, `test_solver_2d_fixes`, `test_metrics_validator_fixes`, `test_parameter_injection`, `test_model_cache` |
| Model configuration and comparison harness | `test_model_config`, `test_model_eval` |
| Generated scripts (mocked model client) | `test_script_generator`, `test_validation_generator` |

Set `VECTORNAUT_ACCURACY_TABLE=1` to print the solver accuracy measurement table (case x method: max error, relative L2 error, runtime).

### Known open issues

Tests marked `@unittest.expectedFailure` document known, not yet fixed behaviour. Each carries a comment explaining the cause. When a fix makes one pass, unittest reports an "unexpected success" and the run fails: remove the marker and keep the test as a regression test.

## Deterministic Benchmark

Run the quality benchmark cases:

```powershell
& .\.venv\Scripts\python.exe -m vectornaut.benchmark
```

The benchmark cases live in `benchmarks/eval_cases.json`. They cover known-good, warning, and failing result profiles so validator/eval behavior stays stable across changes.

## Answering Model Calls by Hand

`python -m vectornaut.llm_replay --session <dir> --query "<prompt>"` runs the pipeline in live mode but answers every model call from `<dir>/responses/NN.json` instead of Gemini. Each run stops at the first unanswered call and writes its full prompt and expected JSON schema to `<dir>/requests/NN_<Schema>.md`; write the answer and run the same command again. When all calls are answered the result lands in `<dir>/result.json` and `<dir>/report.md`. This is useful for testing the live code path without an API key, for reproducing a model answer that broke something, or for letting another model play Gemini's role.

## AI Model Comparison

`python -m vectornaut.model_eval --mock --config baseline=` runs the reference prompts in `benchmarks/ai_reference_cases.json` through the pipeline offline. Mock mode checks the harness, solvers and validator, not model quality. Live comparisons need `GEMINI_API_KEY` and cost API calls; see "Comparing AI models" in `docs/OPERATIONS.md`.

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

The scripts in `tests/live/` exercise real server, model, or API behavior. They are useful during exploration but are not part of the stable local suite because they depend on a running server, API keys, network, model behavior, or longer runtimes. Most are plain scripts rather than `unittest` modules; run them one at a time:

```powershell
& .\.venv\Scripts\python.exe tests\live\test_optimizer_live.py
```

- Need a running server on `http://127.0.0.1:8080` (start it as shown above): `test_2d_e2e.py`, `test_api.py`, `test_chat.py`, `test_live_chat.py`, `test_live_chat_opt.py`, `test_live_dynamic.py`, `test_live_plastron.py`, `test_myelin.py`, `test_optimizer_live.py`, `test_parameter_propagation.py`, `test_raw_response.py`, `test_real_remining.py`, `test_remining_live.py`, `test_server_encoding.py`, `test_static_headers.py`, `verify_live_umlauts.py`. Several of these send `is_mock: true` and only need the server; the others make the server call Gemini.
- Call Gemini directly (need `GEMINI_API_KEY` in `.env`): `test_gemini_direct.py`, `test_membrane_pipeline.py`.

## Manual Solver Checks

The scripts in `tests/manual/` run offline without an API key, but they are plain scripts with `assert` statements and print output rather than `unittest` tests. Some train a PINN, so they are slower than the stable suite:

```powershell
& .\.venv\Scripts\python.exe tests\manual\test_solver.py
```

Contents: `test_solver.py` (1D fluid, thermal and electrostatic solvers), `test_2d_solver.py` (2D PINN/FDM heat conduction), `test_plastron.py` (PlastronGlide analytical case), `test_caret.py` (SymPy `^` parsing).

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
