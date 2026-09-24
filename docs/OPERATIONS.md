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

### Model and thinking level per stage

Every stage that calls Gemini resolves its model at call time, so changing the environment takes effect without code changes:

```text
VECTORNAUT_MODEL_<STAGE>   -> model for one stage, e.g. VECTORNAUT_MODEL_AUDITOR=<model-id>
VECTORNAUT_MODEL           -> model for all stages without a stage-specific value
(default)                  -> gemini-3.5-flash
```

Stage names: `MINER`, `FORMULATOR`, `AUDITOR`, `OPTIMIZER`, `SYNTHESIZER`, `SCRIPT_GENERATOR`, `TEST_GENERATOR`, `CHAT`.

The thinking level can be set per stage with `VECTORNAUT_THINKING_<STAGE>` (`minimal`, `low`, `medium`, `high`). Invalid values are ignored with a warning and the stage default is used. Defaults: miner `medium`, formulator `medium`, auditor `high`; all other stages send no thinking config unless the variable is set. An explicit `thinking_level` passed to `ModelFormulator(...)` or `formulate_model(...)` wins over the environment.

Live `/api/run` responses list the model used per stage under `models`.

## Comparing AI models

`vectornaut.model_eval` answers "does a stronger model make the pipeline better?" with objective, deterministic scores. It runs the full pipeline for several model configurations on the fixed reference prompts in `benchmarks/ai_reference_cases.json` (riblets, plastron/slip films, 1D and 2D heat conduction, 2D Laplace/Poisson, electrostatics, a transient case that should route to `dynamic_script`, a structural case and a deliberately infeasible request that should be rejected).

```bash
# Plan and cost estimate only
python -m vectornaut.model_eval --config baseline= --config strong=VECTORNAUT_MODEL=<model-id> --repeats 3 --dry-run

# Offline check of the harness itself (no API calls)
python -m vectornaut.model_eval --mock --config baseline= --config strong=VECTORNAUT_MODEL=<model-id>

# Live comparison (needs GEMINI_API_KEY)
python -m vectornaut.model_eval \
  --config baseline= \
  --config strong_formulator=VECTORNAUT_MODEL_FORMULATOR=<model-id>,VECTORNAUT_MODEL_AUDITOR=<model-id> \
  --config high_thinking=VECTORNAUT_THINKING_MINER=high,VECTORNAUT_THINKING_FORMULATOR=high \
  --repeats 3
```

Options: `--config NAME=KEY=VALUE,...` (repeatable; `NAME=` runs the ambient environment, `KEY=` unsets a variable for that config; only `VECTORNAUT_*` keys, stage names and thinking levels are validated), `--cases id1,id2`, `--cases-file`, `--repeats N` (default 1), `--epochs` (default 80), `--max-rounds` (default 2), `--mock`, `--out DIR` (default `$VECTORNAUT_DATA_DIR/model_evals/<timestamp>/`), `--dry-run`, `--list-cases`, `--verbose`.

How it runs:

- Each config's overrides are applied only for its own runs and restored afterwards (also on errors). Configs are interleaved per case so changing API conditions affect all configs alike.
- Every run gets its own data directory (`runs/<config>/<case>/r<k>/`) with `result.json`, `pipeline.log`, cached PINN models and generated scripts, so one config cannot reuse another's cached PINN weights. `runs.jsonl` is appended after every run, and Ctrl+C still writes a partial report.
- Without `--mock` the harness refuses to start when `GEMINI_API_KEY` is missing (the pipeline would silently fall back to mock mode). A live run that still comes back as mock is scored as `mock_fallback`.

Costs: the plan line shows `configs x cases x repeats` pipeline runs before anything starts. A live run typically makes about 6 Gemini calls with the default 2 optimization rounds (miner, formulator, auditor per round, optimizer, synthesizer) and up to about 40 when all three concept attempts are re-mined and `dynamic_script` needs its self-correction retries. 10 cases x 2 configs x 3 repeats = 60 runs, i.e. roughly 360 calls, more with re-mining. Runs are sequential; expect minutes per live run with high thinking levels.

Reading the results (`model_eval.md`, full data in `model_eval.json`):

- Each run scores 0..1 as the weighted mean of the checks that apply to its case: pipeline success (3), audit outcome (2), validator status (2; pass 1, warn 0.5), validator score, solver method among the case's acceptable methods, 1D/2D dimensionality, domain keywords, relative error (full credit up to the case target, half credit up to 1.0), finite outputs, parameter ranges, re-mining within the case limit, `evaluate_run_output` passed, and generated-test validation for `dynamic_script`. For the infeasible case only "rejected" (3) and "audit rejected" (2) count.
- **Summary per config**: mean score and delta against the first config, "outcome met" rate (result produced, or request rejected for the infeasible case), pipeline success rate, mean concept attempts (1.0 = no re-mining), mean runtime and failure categories (`audit_rejected`, `solver_error`, `validator_fail`, `optimizer_rejected`, `api_error`, `accepted_infeasible`, `mock_fallback`, `exception:<Type>`).
- **Per case** and **check breakdown** show where configs differ, e.g. a stronger formulator should raise `dimensionality` on the 2D cases and `solver_method` on the transient case.
- Treat differences smaller than the repeat σ (shown with `--repeats` > 1) as noise. One repeat is enough to find crashes, not to rank models.
- `--mock` results do not measure any model: every stage returns the canned riblet or plastron concept (1D, PINN, audit always passes), so all configs score the same. Use mock mode only to check the harness and the deterministic part of the pipeline.
- The scores are only as good as the expectations: parameter ranges are deliberately wide (orders of magnitude), and a range that matches no parameter name is skipped.

## Storage

The current storage model is hybrid:

- Full run documents: `history/*.json`
- Markdown reports: `reports/*.md`
- History index: `vectornaut.sqlite3`
- Evaluation records: `eval_runs/*.json`
- Model comparisons: `model_evals/<timestamp>/` (see "Comparing AI models")

The JSON files are still the source of truth. SQLite is currently an index for dashboard listing and a stepping stone toward a production database.

## Production Direction

For public/scalable operation:

- Keep the FastAPI backend as a separately deployable API service.
- Move long-running solver jobs behind a queue/worker.
- Replace local SQLite/file storage with Postgres plus object storage for reports, plots, generated scripts, and model artifacts.
- Add request authentication before exposing live model or dynamic-script execution.
- Run generated scripts only in a sandboxed worker with CPU/memory/time limits.
