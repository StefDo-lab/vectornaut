# Vectornaut API Contract

Base URL in local development:

```text
http://127.0.0.1:8080
```

## System

### `GET /api/health`

Returns service and storage status.

Response:

```json
{
  "status": "ok",
  "service": "vectornaut",
  "data_dir": "...",
  "sqlite_index": "...",
  "sqlite_index_exists": true
}
```

## Run Pipeline

### `POST /api/run`

Runs the full Vectornaut pipeline and archives the result.

Request:

```json
{
  "query": "design drag reducing surface",
  "epochs": 80,
  "is_mock": true,
  "override_parameters": null,
  "previous_miner_output": null,
  "max_optimization_rounds": 2
}
```

Response shape:

```json
{
  "success": true,
  "query": "...",
  "epochs": 80,
  "is_mock": true,
  "miner": {},
  "auditor": {},
  "simulator": {},
  "validation": {
    "status": "pass",
    "reliability": "high",
    "score": 0.95,
    "checks": [],
    "warnings": [],
    "recommended_action": "accept"
  },
  "optimization_history": [],
  "synthesis": {},
  "failed_concepts": [],
  "models": {},
  "report_md": "..."
}
```

A successful run also carries `"status": "completed"`.

Rejected request (HTTP 200). When the miner or the auditor sets `request_feasible: false` (the request itself is physically impossible as stated, e.g. more energy out than in), or the deterministic validator check `physics_closed_system_efficiency` fails (efficiency/COP/amplification above 1 in a closed or adiabatic system without energy input; `recommended_action: "reject_request"`), the pipeline stops at once without re-mining and returns:

```json
{
  "success": false,
  "status": "rejected",
  "query": "...",
  "rejection": {"stage": "miner | auditor | validator", "reason": "...", "design_name": "...", "concept_attempt": 1},
  "miner": {},
  "auditor": null,
  "simulator": null,
  "validation": null,
  "optimization_history": [],
  "failed_concepts": [],
  "models": {},
  "report_md": "# Anfrage abgelehnt: ..."
}
```

`auditor`, `simulator` and `validation` are filled when the run got that far. The rejection is archived and listed in the history like any other run.

All concepts failed (HTTP 422). When every concept attempt fails (audit with `request_feasible` still true, solver error, validator fail or optimizer verdict), the response is:

```json
{
  "success": false,
  "status": "failed",
  "detail": "All bionic concepts failed validation (3 attempts).\n1. <design> (<source>): <reason>\n...",
  "error": {"type": "concepts_exhausted", "message": "...", "attempts": 3, "failed_concepts": [{"design_name": "...", "inspiration_source": "...", "reason": "...", "stage": "auditor | solver | validator | optimizer"}]},
  "query": "..."
}
```

Other unexpected errors still return HTTP 500 with `{"detail": "..."}`.

`models` maps each pipeline stage that made a live Gemini call in this run to the model name it used, e.g. `{"miner": "gemini-3.5-flash", "formulator": "gemini-3.5-flash", "auditor": "gemini-3.5-flash", "optimizer": "gemini-3.5-flash", "synthesizer": "gemini-3.5-flash"}`. `script_generator` and `test_generator` appear when a `dynamic_script` solve was attempted. In mock mode it is an empty object. Runs archived before this field existed do not contain it. See `docs/OPERATIONS.md` for the `VECTORNAUT_MODEL*` variables.

## History

### `GET /api/history?limit=30`

Lists archived runs. New runs are indexed in SQLite and still written as JSON files.

### `GET /api/history/{run_id}`

Loads a full archived run JSON document.

## Evaluation

Evaluation responses include the deterministic validator result under `validator`. A validator `fail` makes the eval fail; `warn` keeps the eval usable but marks it as requiring inspection or solver rerun.

During normal `/api/run` execution, validator `fail` is also used as control logic: the current concept is rejected and the pipeline attempts Re-Mining. Validator `warn` remains displayable and archived, but the response carries the recommended action.

The deterministic validator checks schema, finite numeric values, relative error, performance-gain sanity, simple Dirichlet boundaries such as `u(0)=0`, and simple derivative boundaries such as `u'(0)=1` or `du_dy(0)=1`.

If validator `warn` recommends `rerun_solver`, `/api/run` attempts deterministic solver fallbacks before returning the result. For 1D models it tries analytical/SciPy alternatives; for 2D models it tries FDM. Accepted fallback attempts are recorded in `optimization_history[].solver_fallbacks`.

### `POST /api/eval/judge`

Evaluates an existing run object against structural and numerical criteria.

### `POST /api/eval/run`

Runs the pipeline and persists an evaluation record.

### `POST /api/eval/batch`

Runs multiple evaluation cases.

### `GET /api/eval/runs`

Lists persisted evaluation records.

### `GET /api/eval/runs/{run_id}`

Loads a full evaluation record.

## Debug

### `POST /api/debug/replay`

Re-evaluates or optionally re-runs a previous history/eval record.

## Chat

### `POST /api/chat`

Provides a German explanation or parameter suggestion based on `current_run`.

Request:

```json
{
  "message": "Erklär mir die Testergebnisse",
  "history": [],
  "current_run": {},
  "is_mock": true
}
```

## Runtime Limits

The pipeline clamps potentially expensive run parameters:

```text
VECTORNAUT_MAX_EPOCHS=1000
VECTORNAUT_MAX_OPTIMIZATION_ROUNDS=10
VECTORNAUT_SCRIPT_TIMEOUT_SECONDS=60
```

The defaults are intentionally conservative for local test operation.
