# Vectornaut Omni - Project Status & Handoff Protocol

This document is the single source of truth for the **Vectornaut Omni** project state. It is meant to bring any new contributor (human or AI agent) up to speed quickly at the start of a session.

Last reviewed: 2026-09-24

---

## 🎯 Project Overview & Core Goals

**Vectornaut Omni** grew out of a specialized "shark-skin drag simulator" into an open-ended, multi-domain physics solver platform.
* **Target Vision**: A web dashboard where users enter design prompts (e.g., riblet structures, thermal shields, wings), which are then turned into physics models, simulated, validated, and reported.
* **Core Domains**: Fluid dynamics (drag reduction, boundary layers), thermodynamics (heat transfer, transient shields), structural mechanics, and general 1D/2D partial differential equations (PDEs).

---

## 🔁 Pipeline

`vectornaut/pipeline.py` (`PipelineRunner`) runs each request through these stages:

1. **Miner** (`miner.py`): finds a bio-inspired concept for the query (Gemini).
2. **Formulator** (`formulator.py`): turns the concept into a governing equation, boundary conditions, and parameters (Gemini).
3. **Auditor** (`auditor.py`): checks physical plausibility and material limits and picks a solver method (Gemini). Queries containing `script`, `skript`, `dynamic solver`, or `custom solver` are routed to `dynamic_script`.
4. **Solver** (`solver_dispatcher.py` + `solvers/`): solves the model (see below).
5. **Validator** (`validator.py`): deterministic, non-AI checks of the solver output. It returns `accept`, `rerun_solver`, or `remine`. Warnings trigger automatic fallback to other solver methods.
6. **Optimizer** (`optimizer.py`): proposes parameter updates across optimization rounds (Gemini).
7. **Synthesizer / Reporting** (`synthesizer.py`, `reporting.py`): writes the practical/commercial summary and the Markdown run report, then archives the run.

Without `GEMINI_API_KEY`, `/api/run` falls back to mock mode, so the pipeline and tests work offline.

---

## 🏗️ Repository Layout

```text
web_server.py            FastAPI app (port 8080), mounts routes from vectornaut/api/
main.py                  Command-line entry point
vectornaut/
  api/                   HTTP routes: run, chat, history, eval, system (health)
  pipeline.py            Pipeline orchestration, solver fallback, solver comparison
  miner.py, formulator.py, auditor.py, optimizer.py, synthesizer.py   Gemini-backed stages
  validator.py           Deterministic result validation
  evaluation.py, benchmark.py   Eval records and the deterministic quality benchmark
  solver_dispatcher.py   Solver entry point: dispatch_and_solve() routes to 1D / 2D / dynamic script
  solvers/
    parsing.py           1D equation/BC parsing, missing-parameter detection, domain bounds
    pinn_model.py        GenericPINN network shared by 1D and 2D PINN solvers
    solvers_1d.py        SymPy analytical, SciPy BVP, 1D PINN
    solvers_2d.py        2D parsing, 2D FDM, 2D PINN
    model_cache.py       Save/reuse trained PINN models
    dynamic_script.py    Objective contract, parameter sweep, generated-script validation
  script_generator.py    Gemini writes a Python solver script, runs it, self-corrects (up to 3 attempts)
  test_generator.py      Gemini writes validation test scripts for generated solvers
  runtime_guards.py      Epoch/round limits and solver output sanity checks
  storage.py             Data directory paths and SQLite history index
  config.py              Pydantic schemas and Gemini client
static/                  Dashboard UI (index.html, app.js, style.css)
benchmarks/eval_cases.json   Benchmark cases (known-good, warning, failing profiles)
tests/                   Stable offline suite (unittest discovery)
tests/live/              Tests needing a running server, network, or GEMINI_API_KEY
tests/manual/            Offline scripts run by hand (solver checks)
scripts/                 One-off maintenance helpers (see scripts/README.md)
docs/                    API_CONTRACT.md, OPERATIONS.md, TESTING.md
```

Runtime data (history, reports, eval runs, generated scripts/tests, saved models, SQLite index) is written under `VECTORNAUT_DATA_DIR` (default: the current directory) and ignored by Git. Plots from dynamic scripts go to `static/plots/` so the web server can serve them.

---

## 🚀 Current Status

* **Working prototype.** The full pipeline runs end to end in mock mode and with a Gemini key.
* **Solvers**: SymPy analytical, SciPy BVP, and PyTorch PINN for 1D; FDM and PINN for 2D; generated Python scripts (`dynamic_script`) with a parameter sweep against an objective metric contract.
* **Validation**: deterministic validator with derivative boundary checks, reliability score, recommended action, and automatic solver fallback.
* **UI**: dashboard with prompt chips, assistant chat in the left rail, technical run reports rendered from Markdown, history, and developer JSON view/download.
* **Tests**: `python -m unittest discover -s tests -t .` runs 29 tests (1 skipped live test) offline in a few seconds. `python -m vectornaut.benchmark` matches 3/3 cases. See `docs/TESTING.md`.

### Recent maintenance (2026-09)
* Trained PINN models are stored under `VECTORNAUT_DATA_DIR/saved_models` (previously always relative to the working directory, which also let separate test runs reuse each other's models).
* `solver_dispatcher.py` was split into the `vectornaut/solvers/` package (pure refactor, behavior unchanged; old import paths still work).
* Root-level tests and helper scripts were moved into `tests/`, `tests/live/`, `tests/manual/`, and `scripts/`.

---

## ⚠️ Known Limitations & Open Issues

* **Security**: generated scripts run as plain subprocesses on the host, and the API has no authentication. Fine for local use; do not expose publicly before adding auth and a sandboxed worker (see `docs/OPERATIONS.md`).
* **Model fidelity**: the physics models are simplified 1D/2D idealizations. Reported gains (e.g., drag reduction percentages) are indicative, not engineering values.
* **AI model choice**: every Gemini stage uses `gemini-3.5-flash`, hardcoded in each module. There is no per-stage configuration yet, and no benchmark that measures the quality of the AI stages themselves (the current benchmark covers the validator/eval logic only).
* **Parameter write-back bug**: `AuditorOutput.audited_parameters_dict` builds a new dict on every access, so auto-injected parameters written back to it are lost, and the 1D path's re-merge drops them. Left as-is during the refactor to keep behavior unchanged.
* **Test runner**: the project uses `unittest`. Running `pytest tests` would also collect `tests/live/` and try to reach a server.
* **Deletion candidates**: `scripts/read_transcript.py` and `scripts/search_log.py` (hardcoded to one Windows transcript file); several overlapping encoding probes in `tests/live/`.

---

## 🔮 Next Steps & Roadmap

1. **Configurable AI models per stage** (e.g., via environment variables), plus a small set of reference prompts to compare models on the Formulator, Auditor, and script generation stages.
2. **Fix the parameter write-back bug** in `audited_parameters_dict`.
3. **Port mock-mode live tests** (`test_api`, `test_chat`, `test_optimizer_live`, `test_remining_live`, `test_parameter_propagation`) to FastAPI `TestClient` so they join the stable suite.
4. **More physical models**: structural beam bending, acoustics, multiphase flows in the auditor rules.
5. **Heatmap & canvas performance** for higher FDM/PINN grid resolutions (e.g., 100x100).
6. **Before any public deployment**: authentication, a job queue, and sandboxed execution of generated scripts.
