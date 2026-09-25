# Live-Path Findings (model calls answered by hand)

Date: 2026-09-25. Method: `python -m vectornaut.llm_replay` (see `docs/TESTING.md`). The pipeline ran in live mode and every Gemini call was answered by Claude acting as a competent, honest model (and once as a careless one), then the results were checked against independent closed-form or numerical solutions.

## Scenarios

| Scenario | Path | Calls | Outcome | Solver numbers vs independent check |
|---|---|---|---|---|
| Liquid-infused hose liner (pressure-driven slip flow) | 1D analytical | 6 | complete, validator pass | exact |
| Bone-inspired beam, uniform load, simply supported | 1D analytical | 6 | complete, validator warn | field exact; reported metric wrong |
| Double-pane window, winter | 2D PINN + FDM reference | 8 (one concept re-mined) | complete, validator warn | PINN max error 0.035 K; metrics mislabelled |
| Re-entry heat shield, transient, nonlinear k(T) | dynamic_script (generated solver + tests) | 7 | complete | generated solver matches an independent implicit solver; headline metric from another parameter set |
| Impossible heater (>300 % efficiency, closed adiabatic system) | honest model | 9 | failed after 3 concept attempts (HTTP 500 for the user) | – |
| Same, careless model plays along | 1D analytical | 6 | complete, `success: true`, validator warn | "500 %" is an input parameter, not a result |

The solvers themselves were correct in every scenario. The problems are in what the pipeline measures, reports and passes between stages.

## Findings by severity

### High

1. **Reported metric is not the quantity asked for.** 1D metrics are always the derivative at `domain_min` (wall-shear logic, `solvers/solvers_1d.py`, `_solve_1d` in `solver_dispatcher.py`); 2D metrics are the field mean (`_solve_2d`). The auditor's `ui_metadata` label is attached unchecked (`reporting.py`). Beam: "max deflection 0.0008" is the support slope; true w_max is 0.000253 m. Window: "inner glass surface temperature 276.4 K" is the field mean; true value 288.2 K. Heater: "heat flux to the room 0.0000" is dT/dx at the adiabatic end. Fix: let the formulator/auditor choose the metric (value/derivative at a named boundary, max, mean, integral) and compute that.
2. **`performance_gain_pct` is structurally 0 outside slip-flow problems.** The no-effect baseline only neutralises `simulation_coefficient` and slip aliases, which non-fluid equations never contain. Window (true benefit ~52 % less heat loss), beam (~70 % mass saving) and hose (benefit is flow rate, not wall shear, which force balance fixes) all report 0.00 %, the objective contract fails, the optimizer has no signal, and the log still says "successfully optimized". Fix: let the auditor define the baseline design or objective; otherwise report "n/a", not 0 %.
3. **No way to reject an impossible request, and no physics check catches one.** Miner/Formulator schemas have no refusal field; the auditor may only fail "dangerous" designs; re-mining after an audit failure asks for a "completely new concept" (3 attempts, 6 wasted calls); the honest outcome reaches the user as HTTP 500 with a Python repr. A careless model's perpetual-motion design passes with `success: true` and a commercial report. Fix: `infeasible_request` + reason in Miner and Auditor schemas, a structured "rejected" result, and a deterministic energy-balance check for heat problems.
4. **2D solver drops the left-hand side of the equation.** Only the RHS is parsed; FDM and PINN always solve u_xx + u_yy = RHS. Coefficients such as `k*(...)` or anisotropy factors are silently lost, and the FDM reference shares the parser, so validation cannot notice.
5. **dynamic_script headline numbers come from a sweep variant, the rest from the baseline.** `_solve_dynamic_script` takes gain, relative error and metrics from `best_sweep` but profile, plot and parameters from the baseline run. The sweep varies the first four parameters in dict order (not the design variables, and also operating conditions), accepts candidates with relative error up to 100 %, ignores the contract's hard constraints and the script's own `success` flag, and runs before the validation tests.
6. **Generated code is not sandboxed.** Solver, sweep and test scripts run via `subprocess.run` with the full environment (including `GEMINI_API_KEY` / cloud credentials), in the repository working directory, with no network or filesystem isolation; the timeout does not kill grandchildren; the model-written test script's self-reported `success` is trusted. Local use only until sandboxed (see `docs/OPERATIONS.md`).

### Medium

7. **Auditor answers are overwritten.** Post-processing replaces the model's dimensionless numbers with one Reynolds number from a default velocity of 1.5 m/s (fluid) or a constant placeholder 100.0 (all other domains), and that value appears in the report (`auditor.py`). "Thermodynamics" does not match the "thermal" domain key, so UI defaults fall back to fluid labels.
8. **Round-2 auditor call is identical to round 1.** The auditor never sees optimizer adjustments; overrides are applied afterwards, so `simulation_coefficient` and audit notes describe the previous parameters, and the report mixes both. The call is also wasted.
9. **Formulator prompt vs schema.** The prompt demands new parameters be appended to a `parameters` field that `ModelFormulation` does not have; missing symbols then get auto-injected defaults of 1.0. The prompt forces transient problems into the BVP/[0,1]² template.
10. **2D validator coverage.** The BC check only understands 1D numeric Dirichlet conditions but is recorded as passed for 2D; relative error is normalised by absolute values (a Kelvin offset hides errors). A constant, mirrored or 2 K-shifted field scores almost the same as the correct one.
11. **A parser limitation discards the whole concept.** 2D Robin conditions (a standard way to model surface heat transfer) are unsupported; the exception goes to re-mining instead of asking the formulator to reformulate or trying another solver.
12. **Script correction gaps.** Exit code 0 counts as success even when the script's JSON says `"success": false`; failing validation tests never trigger correction; no guard against metrics at numerical-noise level (a 99.9996 % "gain" from two values below the ODE tolerance).
13. **The synthesizer and optimizer see too little.** No field values, validation status or relative error reach the synthesizer; the optimizer prompt asks about failed validation tests but gets no validation data.

### Low

14. Report formatting: 4 fixed decimals turn mm-scale results into `0.0000`; full 2D grids are dumped (hundreds of rows); "(genutzt in den Gleichungen)" is printed for coefficients the equations do not use; raw floats like `0.0022500000000000003`.
15. Slip aliases (`slip_length`, `lambda`, `slippage_coefficient`) are injected into every problem, including thermal and structural ones and generated-script parameter files.
16. Prompt languages are mixed (English for miner/formulator/auditor, German for the rest).
17. Dynamic-script plots are written to `static/plots/` in the working directory, not under `VECTORNAUT_DATA_DIR`.

## Implications for model choice

A more careful model changed the outcome only where the pipeline gives it a say: the honest model rejected the impossible request, the careless one produced a "successful" perpetual-motion report. Everything else in the list above is independent of the model and would affect a stronger model equally. Fixing findings 1–3 and 7–8 is a prerequisite for a live model comparison to measure anything meaningful.
