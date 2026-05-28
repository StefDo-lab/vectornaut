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
  "report_md": "..."
}
```

## History

### `GET /api/history?limit=30`

Lists archived runs. New runs are indexed in SQLite and still written as JSON files.

### `GET /api/history/{run_id}`

Loads a full archived run JSON document.

## Evaluation

Evaluation responses include the deterministic validator result under `validator`. A validator `fail` makes the eval fail; `warn` keeps the eval usable but marks it as requiring inspection or solver rerun.

During normal `/api/run` execution, validator `fail` is also used as control logic: the current concept is rejected and the pipeline attempts Re-Mining. Validator `warn` remains displayable and archived, but the response carries the recommended action.

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
