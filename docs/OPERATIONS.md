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
VECTORNAUT_SYMBOLIC_TIMEOUT_S=20
VECTORNAUT_SOLVER_MEMO_DIR=
```

`VECTORNAUT_SYMBOLIC_TIMEOUT_S` is the time budget of one analytical (SymPy) solve of a 1D problem (default 20 s; `0` = no limit, solved in-process). The symbolic solve runs in a worker process that is terminated when it runs over; the SciPy BVP solution is then used and `solver_note` says so (`analytical: symbolic solve exceeded the time budget ...`). After a timeout for the design, the baseline of the same equation is solved with SciPy directly. When the auditor asks for a PINN, the SciPy solution is its reference and no symbolic solve is attempted; for `scipy` the analytical solution stays the independent reference (under the budget). The recorded case: an exponential source with two Robin boundary conditions in the facade-cooling explorer run, where `sp.solve` ran ~20 minutes.

`VECTORNAUT_SOLVER_MEMO_DIR` (off by default) memoises analytical solves on disk: results, deterministic failures and timeouts, keyed by equation, boundary conditions, symbols, parameter values, domain, SymPy version and the solver source. A memoised timeout is reused only while the budget is not larger. `python -m vectornaut.llm_replay` sets it to `<session>/solver_memo` (turn off with `--no-solver-memo`), so replays do not repeat expensive symbolic solves.

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

## Generated-script sandbox

The `dynamic_script` solver path runs model-written Python: the solver script, its parameter-sweep variants and the validation test module. All of them go through `vectornaut/sandbox.py` (`run_generated_script`):

- **Static pre-check** (`check_generated_code`): scripts importing network, process or FFI modules (`socket`, `ssl`, `requests`, `urllib`, `http`, `ftplib`, `smtplib`, `ctypes`, `subprocess`, `importlib`, ...) or calling `os.system`/`os.popen`/`os.exec*`/`os.spawn*`/`os.fork`/`os.kill`/`os.setsid`, `shutil.rmtree`, `exec`, `eval`, `compile`, `__import__` are rejected before execution. The rejection goes back to the model through the normal self-correction loop (3 attempts).
- **Test contract**: the validation test module no longer starts the solver itself. A trusted runner (`vectornaut/generated_test_runner.py`) imports the module, provides `from solver_harness import run_solver, nominal_params, is_finite_number`, runs the `unittest.TestCase`s and derives pass/fail from the unittest result; the module's own claims are ignored. An import error or a module without tests triggers self-correction; failing tests do not (they are the validation result).
- **Result contract**: a solver run with exit code 0 but `"success": false`, missing fields or non-finite metrics counts as a failure and triggers self-correction.
- **Environment**: only `PATH`, `LANG`/`LC_ALL`/`LC_CTYPE`, `TZ` (plus `SYSTEMROOT`/`WINDIR`/... on Windows) are passed through; `HOME`, `TMP*` and `MPLCONFIGDIR` point into the temporary directory; `MPLBACKEND=Agg`, `PYTHONNOUSERSITE=1`, single-threaded BLAS. Nothing whose name contains KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL, GOOGLE, GEMINI, AWS or AZURE, and no `VECTORNAUT_*` variable reaches the script.
- **Working directory**: a fresh temporary directory per run (`VECTORNAUT_SANDBOX_TMPDIR` to choose the parent), never the repository. Inputs are copied in, `results.json`/`plot.png` copied out to `generated_scripts/` and the plot directory, the directory is deleted afterwards.
- **Process control**: new session / process group (POSIX) or `CREATE_NEW_PROCESS_GROUP` (Windows); on timeout the whole group is killed (`os.killpg`, or `taskkill /T /F` with `proc.kill()` fallback on Windows). Stragglers are also killed when the script exits normally (POSIX).
- **Resource limits (POSIX only)**: CPU seconds = timeout + 5, address space `VECTORNAUT_SANDBOX_MEMORY_MB` (default 2048), file size `VECTORNAUT_SANDBOX_FILE_SIZE_MB` (default 200), no core dumps. Windows runs without these limits.
- **Network (best effort)**: `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` point to `http://127.0.0.1:9` and `NO_PROXY` is empty, so proxy-aware HTTP clients fail fast.

Timeouts: `VECTORNAUT_SCRIPT_TIMEOUT_SECONDS` (default 60) per solver run, `VECTORNAUT_TEST_SCRIPT_TIMEOUT_SECONDS` (default 5x the solver timeout) for the whole test module.

Plots: dynamic-script plots are written to the repository's `static/plots/` (served by the UI as `/plots/...`), independent of the working directory. `VECTORNAUT_PLOTS_DIR` overrides the location; the UI only shows plots that end up in `static/plots/`.

**This is not a security boundary.** The pre-check is a guard-rail against accidents, not against a determined script (it can be evaded, e.g. through native extensions or libraries that open sockets). Scripts can still read every file the server user can read, open raw network connections, and escape the process group from native code; Windows has no resource limits. Use generated scripts only locally with a trusted model. For untrusted input or public use, run the solver worker in a container or VM without network access, with a read-only filesystem apart from its work directory and with cgroup CPU/memory limits.

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

## Gemini access in Claude Code cloud environments

In a Claude Code cloud environment, store the Gemini key under **API credentials** (not as a plain environment variable, which every user of the environment can read): allowed site `generativelanguage.googleapis.com`, custom header `x-goog-api-key` with an empty prefix. The environment injects the header into requests to that host, so the code never sees the key. Vectornaut still needs `GEMINI_API_KEY` to be set to switch to live mode; any placeholder works, e.g. `GEMINI_API_KEY=injected-by-environment`.

Model calls are streamed by default (`VECTORNAUT_MODEL_STREAM=1`) with thought summaries included, so data keeps flowing while the model thinks: some egress proxies, including the Claude Code cloud environment's, cut requests that stay silent for ~30 s, which high-thinking calls exceed (seen as `502 Bad Gateway: upstream request failed`). Thought parts are discarded; the answer is parsed into the requested schema. Model calls also retry transient failures (HTTP 5xx and 429) with exponential backoff: `VECTORNAUT_MODEL_RETRIES` (default 4) and `VECTORNAUT_MODEL_RETRY_DELAY_S` (default 2 s). Set `VECTORNAUT_LOG_USAGE=1` to print summed token usage per model at process exit (for cost estimates: prompt, output and thinking tokens; thinking tokens are billed as output).

## Using Claude for individual stages

Any stage whose model name starts with `claude-` is sent to the Anthropic Messages API instead of Gemini, e.g. `VECTORNAUT_MODEL_CRITIC=claude-opus-5-5` (US$4 / 20 per M input / output tokens) or `claude-sonnet-5` (US$2 / 10). The adapter streams the request with adaptive thinking, maps the stage's thinking level to `output_config.effort` (minimal/low → low, medium → medium, high → high), passes the stage's pydantic response schema as structured output, and records usage like Gemini calls. Requests always go to `https://api.anthropic.com`; `ANTHROPIC_BASE_URL` is ignored on purpose because it may point at another service in hosted environments.

Credentials: `VECTORNAUT_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY`; in a cloud environment with injected API credentials (allowed site `api.anthropic.com`, header `x-api-key`, empty prefix) no variable is needed. Install the SDK with `pip install anthropic`.
