# Vectornaut Omni - Project Status & Handoff Protocol

This document is the single source of truth for the **Vectornaut Omni** project state. It is meant to bring any new contributor (human or AI agent) up to speed quickly at the start of a session.

Last reviewed: 2026-09-25

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
  model_eval.py          AI model comparison harness (reference prompts in benchmarks/ai_reference_cases.json)
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
benchmarks/              eval_cases.json (validator benchmark), ai_reference_cases.json (model comparison prompts)
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
* **Solvers**: SymPy analytical, SciPy BVP, and PyTorch PINN for 1D; sparse direct FDM and PINN for 2D on rectangular domains with constant or expression-valued edge conditions; generated Python scripts (`dynamic_script`) with a parameter sweep against an objective metric contract. Closed-form accuracy tests confirm analytical/SciPy to ~1e-10 and FDM to its discretisation order.
* **Validation**: deterministic validator with second-order derivative boundary checks, primary-vs-reference agreement, PINN divergence detection, reliability score, recommended action, and automatic solver fallback.
* **AI models**: every stage's model and thinking level is configurable (`VECTORNAUT_MODEL`, `VECTORNAUT_MODEL_<STAGE>`, `VECTORNAUT_THINKING_<STAGE>`; default `gemini-3.5-flash`). Live run results list the model used per stage in `models`. `python -m vectornaut.model_eval` compares configurations on 10 reference prompts.
* **UI**: dashboard with prompt chips, assistant chat in the left rail, technical run reports rendered from Markdown, history, and developer JSON view/download.
* **Tests**: `python -m unittest discover -s tests -t .` runs 439 tests offline in about a minute (1 skipped live test, 4 documented expected failures). `python -m vectornaut.benchmark` matches 3/3 cases. See `docs/TESTING.md`.

### Recent maintenance (2026-09)
* Structure: `solver_dispatcher.py` split into `vectornaut/solvers/`; root tests and helper scripts moved into `tests/`, `tests/live/`, `tests/manual/`, `scripts/`; trained models stored under `VECTORNAUT_DATA_DIR/saved_models`.
* Pipeline fixes: auto-injected default parameters are recorded on the auditor output and reach the 1D solvers (1D problems with such parameters previously always failed); the optimizer discards a concept only via an explicit `concept_failed` verdict instead of keyword matching; the mock chat no longer treats "Shark-Skin" as a ski design. Formulator-added parameters reach the auditor (`ModelFormulation.parameters`); the auditor prompt no longer imposes the riblet slip formula on every fluid concept; `MetricSpec.transform` for nonlinear figures of merit; implausible gains (> 500 %) are flagged as such in report and synthesis; the synthesis prompt says "auditiert" unless an optimization round changed the parameters.
* Solver fixes found by closed-form accuracy tests: thin-film wall derivative, BC value function calls and sub-micron domains in 1D domain detection, nondeterministic handling of parameters named like integration constants, bare dependent variables, `lambda` as a parameter name, numeric constants in PINNs, 1D PINN output scaling, audited slip values no longer overwritten, honest baselines and gains (1D and 2D), reported solver method matches the one actually used, 2D non-constant edge BCs and non-unit domains, independent 2D reference solution, PINN cache respects epochs and compares micro-scale inputs relatively (cache versioned).
* Offline `TestClient` ports of the mock-mode live scripts; `conftest.py` keeps pytest away from `tests/live/` and `tests/manual/`.

---

## 🧭 Idea-Space Explorer (`vectornaut/explorer/`, see `docs/EXPLORER.md`)

* **What**: a quality-diversity (MAP-Elites style) archive over fixed descriptor axes per profile. Each concept is placed in one cell; the best per cell is kept across runs, and every proposal (also failed/rejected ones) is counted, so the report shows where ideas are *proposed* vs. where they *work*.
* **Analysis first (archive version 6, materials)**: blind comparisons on two tasks showed a single careful direct answer beating the explorer's best picks (`benchmarks/explorer_trials/blind_comparison_2026-09-27/`). A new map therefore starts with one `analyst` call (`ProblemAnalysis`: quantitative load breakdown, levers ranked by magnitude, scope decision for levers outside the literal wording, conventional in-service baseline, objective/target/requirements/relevant values, and the analyst's best 3 concepts). The framing is fixed from it; the 3 concepts enter as `seed_direct` entries (pipeline + critic like all others), anchor refine/fill_gap/combine, and `map.md` answers "Did the map beat the direct answer?" (critic objective gain, novelty, requirements). Candidates may target a justified lever outside the wording (`scope_extension`, not penalised, the critic judges legitimacy). The generator runs `--depth deep` (3 candidates per round with a quantitative estimate and a self-critique); `--no-analyst` / `--depth broad` keep the old flow.
* **Search orders instead of "be creative"**: `refine` (mutate an elite), `fill_gap` (empty cells next to good ones, low proposal density first), `extrapolate` (follow the score trend one step beyond the explored edge of an ordinal axis), `combine` (mix two distant elites), `explore` (random unvisited cell), `seed` (cold start). Deterministic for a seed.
* **Profiles**: `materials` (candidates run through the normal pipeline via the new `PipelineRunRequest.concept` hook with `max_concept_attempts=1`; since version 6 the simulation is a feasibility/consistency gate — validator failure excludes, a critic-flagged contradiction lowers the evidence rank — and the score is the critic's objective gain net of the conventional equivalent + must-weighted requirement coverage + a small novelty term, simulated weight 0 by default (`--sim-weight`), × 0.3 for a failed hard check of the critic's checklist: processing/thermal stability, manufacturability, field record, geometry, simple bounds; `--scoring legacy` keeps the version-5 formula) and `business` (deterministic unit-economics model on the model's own estimates, sanity flags, optional critic call that refutes and corrects the inputs; a brainstorming aid, not validation).
* **Run**: `python -m vectornaut.explorer --profile materials|business --query "..." --rounds 3 --batch 6 [--mock]`; by hand: `python -m vectornaut.llm_replay --explorer --profile ... --session DIR ...`. Archive and reports under `VECTORNAUT_DATA_DIR/explorer/<profile>/`. Model stages `explorer`, `critic` and `analyst` (a stronger critic/analyst, e.g. `VECTORNAUT_MODEL_CRITIC=gemini-3.1-pro-preview`, `VECTORNAUT_MODEL_ANALYST=gemini-3.1-pro-preview`, is recommended).
* **Status**: role-play runs and one live Gemini run (facade) under version 5 (`benchmarks/explorer_trials/`); version 6 (analysis first, evaluator as a filter, stricter critic, depth) is tested offline only (`tests/test_explorer_*.py`, `tests/test_explorer_v6.py`). Re-scoring the live Gemini archive under version 6 changes little on its own (every entry was critic-reviewed); the processing-stability hard check its critic missed would move both calcite concepts from ranks 1 and 3 to 10–11 (`docs/EXPLORER.md`). A live version-6 validation run is pending.

---

## ⚠️ Known Limitations & Open Issues

* **Security**: generated scripts run as plain subprocesses on the host, and the API has no authentication. Fine for local use; do not expose publicly before adding auth and a sandboxed worker (see `docs/OPERATIONS.md`).
* **Model fidelity**: the physics models are simplified 1D/2D idealizations. Reported gains (e.g., drag reduction percentages) are indicative, not engineering values.
* **AI stage quality is unmeasured**: the comparison harness exists, but no live run has been done (no API key in the development environment). Mock mode only exercises canned concepts.
* **Open design questions** (tests marked `expectedFailure`): an explicit user override is changed again by the optimizer in later rounds (`test_api_mock_flows`), and overrides appear only in the audited parameters, not in `miner.parameters` (`test_chat_mock_flows`). Both need a product decision.
* **PINN limits** (tests marked `expectedFailure` in `test_solver_accuracy`): the 1D PINN is not nondimensionalised, so micrometre-scale domains train poorly (the validator flags this and falls back); the 2D PINN misses the 5 % target for spatially varying sources at the default training budget.
* **2D solver** assumes a plain Laplacian on the left-hand side (coefficients like `k*(...)` are dropped); FDM Neumann edges are first order.
* **Impossible requests**: miner and auditor can mark a request `request_feasible: false`, and a deterministic validator check flags efficiency/COP/amplification > 1 in closed or adiabatic systems; the pipeline then returns a structured `status: "rejected"` result (HTTP 200). Whether live models use the new fields is unmeasured; the mock miner/auditor never do, so the mock pipeline still accepts the `infeasible_perpetuum` case.
* **Plots** from dynamic scripts are written to `static/plots/` in the working directory, not under `VECTORNAUT_DATA_DIR`.
* **Deletion candidates**: `scripts/read_transcript.py` and `scripts/search_log.py` (hardcoded to one Windows transcript file); several overlapping encoding probes in `tests/live/`.

---

## 🔮 Next Steps & Roadmap

1. **Fix the remaining live-path findings (9–17)** in `docs/LIVE_PATH_FINDINGS.md` (1–8 are fixed; 9 is fixed except the transient/BVP template, see its status section) (from answering every model call by hand in six scenarios): metrics that measure the requested quantity, a real baseline/objective outside slip flow, a sanctioned "infeasible" outcome plus an energy-balance check, the 2D left-hand side, the dynamic-script sweep, auditor overwrites, and sandboxing of generated code.
2. **Run a live model comparison** once `GEMINI_API_KEY` is available and item 1 is done: one repeat first to find crashes, then at least 3 repeats per configuration before ranking.
3. **Decide the two open override questions** and adjust the pipeline accordingly.
4. **Impossible requests**: verify with the `infeasible_perpetuum` reference case in a live run that the models set `request_feasible: false` (miner/auditor prompts explain it).
5. **PINN improvements**: nondimensionalise the 1D domain (update `test_validator_flags_thin_film_pinn_garbage` to use a synthetic bad profile first), longer or adaptive training for 2D sources.
6. **2D solver**: honour left-hand-side coefficients, second-order Neumann edges.
7. **More physical models**: structural beam bending (4th order), acoustics, multiphase flows in the auditor rules.
8. **Heatmap & canvas performance** for higher FDM/PINN grid resolutions (e.g., 100x100).
9. **Before any public deployment**: authentication, a job queue, and sandboxed execution of generated scripts.
