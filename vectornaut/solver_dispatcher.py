import os
import json
import re
import math
import sys
import subprocess
from datetime import datetime
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, convert_xor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import SimulatorOutput, MinerOutput, AuditorOutput

# The solver implementations live in the vectornaut.solvers subpackage. They are
# re-exported here so that existing imports from vectornaut.solver_dispatcher
# (and patches of names in this module) keep working.
from .solvers.dynamic_script import (
    _dynamic_objective_contract,
    _candidate_parameter_values,
    _run_dynamic_script_once,
    _dynamic_parameter_sweep,
    _run_generated_validation,
)
from .solvers.pinn_model import GenericPINN
from .solvers.parsing import (
    parse_equation_and_bcs,
    auto_detect_and_inject_missing_params,
    get_domain_bounds,
)
from .solvers.solvers_1d import (
    solve_analytical,
    solve_scipy_bvp,
    solve_pytorch_pinn,
)
from .solvers.solvers_2d import (
    parse_rhs_2d,
    parse_bc_2d_string,
    solve_fdm_2d,
    solve_pytorch_pinn_2d,
)
from .solvers.model_cache import load_cached_pinn, save_pinn_model
from .solvers.metrics import (
    GAIN_BASIS_BASELINE,
    GAIN_BASIS_BIONIC,
    GAIN_BASIS_NONE,
    MetricSpecError,
    apply_baseline_overrides,
    baseline_overrides,
    callables_from_bvp,
    callables_from_expr,
    callables_from_pinn,
    lower_is_better,
    metric_1d,
    metric_2d,
    references_any,
    relative_gain,
    resolve_metric_spec,
)


# Names under which equations/BCs may refer to the simulation coefficient. They are
# filled from simulation_coefficient only if the auditor did not provide them itself.
_COEFFICIENT_ALIASES = ('slippage_coefficient', 'lambda', 'slip_length')


def _merged_params(auditor_output: AuditorOutput) -> Dict[str, float]:
    # Merge audited parameters and simulation coefficient
    params = auditor_output.audited_parameters_dict.copy()
    params['simulation_coefficient'] = auditor_output.simulation_coefficient
    # An explicitly audited slip_length / lambda / slippage_coefficient wins over the
    # derived simulation coefficient; the aliases are only filled in when missing.
    for alias in _COEFFICIENT_ALIASES:
        if alias not in params:
            params[alias] = auditor_output.simulation_coefficient
    return params


def _is_bionic_effect_param(name: str) -> bool:
    """
    True for the parameters that carry the bionic effect (slip length / slippage
    coefficient aliases and the simulation coefficient). These are set to 0 for the
    no-effect baseline; geometric lengths (film/coating thickness, heights, ...) and
    material properties are never touched.
    """
    lowered = name.lower()
    return name == 'simulation_coefficient' or name in _COEFFICIENT_ALIASES or "slip" in lowered


def _join_notes(*notes: Optional[str]) -> Optional[str]:
    joined = "; ".join(note for note in notes if note)
    return joined or None


def _resolve_metric_spec(auditor_output: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The auditor's metric spec (None = default metric) and a note if it had to be ignored."""
    try:
        return resolve_metric_spec(auditor_output), None
    except MetricSpecError as err:
        print(f"[*] Metric spec ignored: {err}")
        return None, f"metric spec ignored: {err}"


def _baseline_setup(
    params: Dict[str, float],
    auditor_output: Any,
    problem_texts: List[str],
    is_effect_param: Callable[[str], bool],
) -> Tuple[Optional[Dict[str, float]], str, Optional[str]]:
    """
    Parameters of the baseline design the performance gain is measured against.
    Returns (baseline_params, gain_basis, note):
    - the auditor's baseline_parameters applied to the design parameters ('baseline_parameters');
    - otherwise the design with the bionic-effect parameters set to 0, if the equation or BCs
      use one of them ('bionic_effect'; equal to params if they are already 0);
    - otherwise (None, 'none', None): no baseline, the gain is not computable.
    """
    overrides = baseline_overrides(auditor_output)
    if overrides:
        audited = getattr(auditor_output, "audited_parameters_dict", {}) or {}
        baseline = apply_baseline_overrides(params, overrides, list(audited), _COEFFICIENT_ALIASES)
        note = None
        unknown = [name for name in overrides if name not in params]
        if unknown:
            note = f"baseline parameters not used by the model: {', '.join(unknown)}"
        return baseline, GAIN_BASIS_BASELINE, note

    effect_names = [name for name in params if is_effect_param(name)]
    used = references_any(effect_names, problem_texts)
    if not used:
        return None, GAIN_BASIS_NONE, None
    if not any(params[name] != 0.0 for name in used):
        # The effect parameter is used but already 0: the baseline is the design itself.
        return dict(params), GAIN_BASIS_BIONIC, None
    baseline = dict(params)
    for name in effect_names:
        baseline[name] = 0.0
    return baseline, GAIN_BASIS_BIONIC, None


def _performance_gain(
    design_metric: float,
    baseline_metric: Optional[float],
    gain_basis: str,
    note: Optional[str],
    legacy: bool,
    lower_better: bool,
) -> Tuple[float, str, Optional[str]]:
    """
    Returns (performance_gain_pct, gain_basis, note). Without a usable baseline the gain is
    0.0 with gain_basis 'none' (reported as n/a). The legacy formula (1 - design / baseline)
    * 100 is kept for the default metric against the no-bionic-effect baseline; otherwise the
    gain is the relative improvement of the metric, honouring lower_is_better.
    """
    if gain_basis == GAIN_BASIS_NONE:
        return 0.0, GAIN_BASIS_NONE, note
    if baseline_metric is None:
        return 0.0, GAIN_BASIS_NONE, _join_notes(note, "no baseline metric available")
    if legacy:
        if abs(baseline_metric) > 1e-8:
            return (1.0 - (design_metric / baseline_metric)) * 100.0, gain_basis, note
        return 0.0, GAIN_BASIS_NONE, _join_notes(note, "baseline metric is zero")
    gain = relative_gain(design_metric, baseline_metric, lower_better)
    if gain is None:
        return 0.0, GAIN_BASIS_NONE, _join_notes(note, "baseline metric is zero or not finite")
    return gain, gain_basis, note


def _metric_fields(
    metric_spec: Optional[Dict[str, Any]],
    baseline_metric: Optional[float],
    gain_basis: str,
    gain_note: Optional[str],
) -> Dict[str, Any]:
    """The metric/gain provenance fields of SimulatorOutput."""
    baseline_value = None
    if gain_basis != GAIN_BASIS_NONE and baseline_metric is not None and math.isfinite(baseline_metric):
        baseline_value = float(baseline_metric)
    return {
        "metric_spec": metric_spec,
        "metric_unit": (metric_spec or {}).get("unit"),
        "baseline_metric_value": baseline_value,
        "gain_basis": gain_basis,
        "gain_note": gain_note,
    }


def dispatch_and_solve(
    miner_output: MinerOutput,
    auditor_output: AuditorOutput,
    epochs: int = 200
) -> SimulatorOutput:
    """
    Main entry point. Dispatches the governing equation and boundary conditions
    to the correct solvers. Performs fallback if needed.
    """
    if auditor_output.solver_method.lower() == 'dynamic_script':
        return _solve_dynamic_script(miner_output, auditor_output, epochs)


    # 1. Extract inputs
    gov_eq = miner_output.governing_equation
    bcs = miner_output.boundary_conditions

    # Merge audited parameters and simulation coefficient
    merged = _merged_params(auditor_output)

    # Auto-detect and inject missing parameters used in equations or BCs
    params = auto_detect_and_inject_missing_params(
        gov_eq_str=gov_eq,
        bcs_list=bcs,
        independent_vars=miner_output.independent_variables,
        dependent_vars=miner_output.dependent_variables,
        params=merged
    )
    # Record the injected defaults on the auditor output so that they are saved in
    # history and synthesis. audited_parameters_dict is a read-only view, so the
    # AuditedParameter list itself has to be extended.
    injected = {k: v for k, v in params.items() if k not in merged}
    if injected:
        auditor_output.add_injected_parameters(injected)

    is_2d = len(miner_output.independent_variables) == 2
    if is_2d:
        return _solve_2d(miner_output, auditor_output, epochs, gov_eq, bcs, params)

    # 1D flow continues here
    return _solve_1d(miner_output, auditor_output, epochs, gov_eq, bcs, params)


def _solve_dynamic_script(
    miner_output: MinerOutput,
    auditor_output: AuditorOutput,
    epochs: int
) -> SimulatorOutput:
    from .script_generator import ScriptGenerator
    generator = ScriptGenerator()
    design_name = miner_output.design_name or "unknown_design"
    slug = re.sub(r'[^a-zA-Z0-9_]', '', design_name.lower().replace(" ", "_"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_relative_dir = os.path.join("static", "plots")
    os.makedirs(plot_relative_dir, exist_ok=True)
    plot_filename = f"plot_{slug}_{timestamp}.png"
    plot_local_path = os.path.join(plot_relative_dir, plot_filename)

    res = generator.generate_and_execute(
        miner_output=miner_output,
        auditor_output=auditor_output,
        epochs=epochs,
        plot_png_path=plot_local_path
    )
    base_params = _merged_params(auditor_output)
    parameter_sweep = _dynamic_parameter_sweep(
        miner_output=miner_output,
        auditor_output=auditor_output,
        script_path=res.get("script_path"),
        base_params=base_params,
        base_result=res,
        slug=slug,
        timestamp=timestamp,
    )
    best_sweep = parameter_sweep.get("best") or {}

    # Run test script validation on the fly
    (validation_passed, validation_report, validation_tests,
     test_script_path, test_output_path) = _run_generated_validation(miner_output, auditor_output, res)

    return SimulatorOutput(
        solver_method="dynamic_script",
        epochs_trained=0,
        final_loss=res.get("relative_error", 0.0),
        loss_history=[],
        performance_gain_pct=best_sweep.get("performance_gain_pct", res.get("performance_gain_pct", 0.0)),
        relative_error=best_sweep.get("relative_error", res.get("relative_error", 0.0)),
        sample_points=res.get("sample_points", []),
        solution_primary=res.get("solution_primary", []),
        solution_reference=res.get("solution_reference", []),
        primary_metric_value=best_sweep.get("primary_metric_value", res.get("primary_metric_value", 0.0)),
        reference_metric_value=best_sweep.get("reference_metric_value", res.get("reference_metric_value", 0.0)),
        custom_plot_url=f"/plots/{plot_filename}",
        validation_passed=validation_passed,
        validation_report=validation_report,
        validation_tests=validation_tests,
        script_path=res.get("script_path"),
        params_json_path=res.get("params_json_path"),
        test_script_path=test_script_path,
        test_output_path=test_output_path,
        execution_mode=res.get("execution_mode", "generated_python_subprocess"),
        objective_metric=parameter_sweep.get("objective"),
        parameter_sweep=parameter_sweep
    )


def _solve_2d(
    miner_output: MinerOutput,
    auditor_output: AuditorOutput,
    epochs: int,
    gov_eq: str,
    bcs: List[str],
    params: Dict[str, float]
) -> SimulatorOutput:
    # Imported here (not with the re-exports above) to keep this helper local to the 2D path.
    from .solvers.solvers_2d import parse_bcs_2d, parse_pde_2d

    x_name = miner_output.independent_variables[0]
    y_name = miner_output.independent_variables[1]
    dep_name = miner_output.dependent_variables[0]

    x_sym = sp.Symbol(x_name)
    y_sym = sp.Symbol(y_name)

    # The whole equation is parsed (all terms moved to one side) into the linear operator
    # a*u_xx + b*u_yy + c*u_xy + d*u_x + e*u_y + g*u = f that the FDM and PINN solve (see
    # parse_pde_2d). Unsupported forms (nonlinear, non-elliptic, unknown symbols) raise, so
    # the pipeline treats them as a solver failure. rhs_str is the whole equation, because
    # parse_rhs_2d (used for the baseline solve below) parses whole equations the same way.
    rhs_str = gov_eq
    rhs_expr = parse_pde_2d(gov_eq, dep_name, x_name, y_name, params)

    # Parses all edge BCs (constant or varying along the edge) and derives the rectangular
    # domain from them. Unparseable BCs raise, so the pipeline treats this as a solver
    # failure; missing edges default to zero flux (see parse_bcs_2d).
    bcs_parsed, x_bounds, y_bounds = parse_bcs_2d(bcs, dep_name, miner_output.independent_variables, params)

    epochs_2d = epochs
    loss_history = []
    final_loss = 0.0
    epochs_trained = 0
    grid_size = 20

    # Reference solver: 2D FDM
    x_vals, y_vals, u_fdm = solve_fdm_2d(
        rhs_expr, bcs_parsed, x_sym, y_sym, grid_size=grid_size, x_bounds=x_bounds, y_bounds=y_bounds
    )

    sample_points_2d = []
    solution_reference_2d = []
    for i in range(grid_size):
        for j in range(grid_size):
            sample_points_2d.append([float(x_vals[i]), float(y_vals[j])])
            solution_reference_2d.append(float(u_fdm[i, j]))

    method_requested = auditor_output.solver_method.lower()
    if method_requested not in ["pinn", "fdm"]:
        method_requested = "pinn"

    # The PINN cache is keyed on the domain as well (this also skips models cached by older
    # versions, which always trained on the unit square).
    cache_params = dict(params)
    cache_params.update({
        "_domain_x_min": float(x_bounds[0]), "_domain_x_max": float(x_bounds[1]),
        "_domain_y_min": float(y_bounds[0]), "_domain_y_max": float(y_bounds[1]),
    })
    if not rhs_expr.is_laplacian:
        # Older versions trained every 2D PINN on u_xx + u_yy = RHS; don't reuse those models.
        cache_params["_pde_general_operator"] = 1.0

    solution_primary_2d = []
    if method_requested == "pinn":
        try:
            def _use_cached_model(model, meta):
                nonlocal final_loss, loss_history, epochs_trained, solution_primary_2d
                final_loss = meta.get("final_loss", 0.0)
                loss_history = meta.get("loss_history", [])
                epochs_trained = len(loss_history) if loss_history else meta.get("epochs_trained", 400)

                with torch.no_grad():
                    xy_tensor = torch.tensor(sample_points_2d, dtype=torch.float32)
                    solution_primary_2d = model(xy_tensor).view(-1).numpy().tolist()

            pretrained_loaded = load_cached_pinn(
                prefix="pinn2d_",
                label="2D ",
                loaded_message="[*] Loaded pre-trained 2D model successfully.",
                gov_eq=gov_eq,
                bcs=bcs,
                params=cache_params,
                build_model=lambda: GenericPINN(input_dim=2, hidden_dim=32),
                on_loaded=_use_cached_model,
                epochs=epochs_2d,
            )

            if not pretrained_loaded:
                model, loss_history = solve_pytorch_pinn_2d(
                    rhs_expr, bcs_parsed, x_sym, y_sym, epochs=epochs_2d,
                    x_bounds=x_bounds, y_bounds=y_bounds
                )
                epochs_trained = epochs_2d
                final_loss = loss_history[-1]

                with torch.no_grad():
                    xy_tensor = torch.tensor(sample_points_2d, dtype=torch.float32)
                    solution_primary_2d = model(xy_tensor).view(-1).numpy().tolist()

                save_pinn_model(
                    model,
                    prefix="pinn2d_",
                    label="2D ",
                    design_name=miner_output.design_name,
                    gov_eq=gov_eq,
                    bcs=bcs,
                    params=cache_params,
                    final_loss=final_loss,
                    loss_history=loss_history,
                )
        except Exception as e:
            print(f"[*] 2D PINN solver failed: {e}. Falling back to 2D FDM.")
            method_requested = "fdm"

    if method_requested == "fdm" or not solution_primary_2d:
        method_requested = "fdm"
        solution_primary_2d = solution_reference_2d
        # The primary solution is the FDM field itself, so use an FDM solve on a grid refined
        # by a factor of 2 as the reference: its nodes contain the primary grid, and
        # relative_error then estimates the discretisation error of the primary solution.
        fine_size = 2 * (grid_size - 1) + 1
        _, _, u_fine = solve_fdm_2d(
            rhs_expr, bcs_parsed, x_sym, y_sym, grid_size=fine_size, x_bounds=x_bounds, y_bounds=y_bounds
        )
        u_fine_at_samples = u_fine[::2, ::2]
        solution_reference_2d = [
            float(u_fine_at_samples[i, j]) for i in range(grid_size) for j in range(grid_size)
        ]

    # Metric: the auditor's metric spec evaluated on the sampled fields, or (default) the
    # mean of the sampled field values.
    metric_spec, gain_note = _resolve_metric_spec(auditor_output)
    primary_metric = float(np.mean(solution_primary_2d))
    reference_metric = float(np.mean(solution_reference_2d))
    if metric_spec is not None:
        try:
            spec_primary = metric_2d(metric_spec, x_vals, y_vals,
                                     np.reshape(solution_primary_2d, (grid_size, grid_size)), params, x_name, y_name)
            spec_reference = metric_2d(metric_spec, x_vals, y_vals,
                                       np.reshape(solution_reference_2d, (grid_size, grid_size)), params, x_name, y_name)
            primary_metric, reference_metric = spec_primary, spec_reference
        except Exception as spec_err:
            print(f"[*] Metric spec {metric_spec} could not be evaluated: {spec_err}. Using the field mean.")
            gain_note = _join_notes(gain_note, f"metric spec not evaluable ({spec_err}); default metric field mean used")
            metric_spec = None

    # Performance gain: change of the primary metric relative to a baseline FDM solve of the
    # conventional design (auditor's baseline_parameters) or, without those, of the design
    # without the bionic effect (simulation coefficient and its aliases set to 0). If the
    # equation and BCs do not use the coefficient either, there is no baseline (gain n/a).
    coefficient_names = ["simulation_coefficient", "slippage_coefficient", "lambda", "slip_length"]
    baseline_params, gain_basis, baseline_note = _baseline_setup(
        params, auditor_output, [gov_eq] + list(bcs), lambda name: name in coefficient_names
    )
    gain_note = _join_notes(gain_note, baseline_note)
    baseline_metric = None
    if baseline_params is not None and baseline_params == params:
        # The coefficient is used but already 0 (or the baseline equals the design).
        baseline_metric = primary_metric
    elif baseline_params is not None:
        try:
            baseline_rhs = parse_rhs_2d(rhs_str, x_name, y_name, baseline_params)
            baseline_bcs, baseline_x_bounds, baseline_y_bounds = parse_bcs_2d(
                bcs, dep_name, miner_output.independent_variables, baseline_params
            )
            baseline_x_vals, baseline_y_vals, u_baseline = solve_fdm_2d(
                baseline_rhs, baseline_bcs, x_sym, y_sym, grid_size=grid_size,
                x_bounds=baseline_x_bounds, y_bounds=baseline_y_bounds
            )
            if metric_spec is None:
                baseline_metric = float(np.mean(u_baseline))
            else:
                baseline_metric = metric_2d(metric_spec, baseline_x_vals, baseline_y_vals, u_baseline,
                                            baseline_params, x_name, y_name)
        except Exception as e:
            print(f"[*] 2D baseline solve failed: {e}. Performance gain reported as n/a (0).")
            gain_note = _join_notes(gain_note, f"baseline solve failed: {e}")
            baseline_metric = None

    performance_gain, gain_basis, gain_note = _performance_gain(
        primary_metric, baseline_metric, gain_basis, gain_note,
        legacy=metric_spec is None and gain_basis == GAIN_BASIS_BIONIC,
        lower_better=lower_is_better(auditor_output),
    )

    abs_diff = np.abs(np.array(solution_primary_2d) - np.array(solution_reference_2d))
    ref_norm = np.abs(np.array(solution_reference_2d))
    relative_err = float(np.sum(abs_diff) / np.sum(ref_norm + 1e-8))

    return SimulatorOutput(
        solver_method=method_requested,
        epochs_trained=epochs_trained,
        final_loss=final_loss,
        loss_history=loss_history,
        performance_gain_pct=performance_gain,
        relative_error=relative_err,
        sample_points=sample_points_2d,
        solution_primary=solution_primary_2d,
        solution_reference=solution_reference_2d,
        primary_metric_value=primary_metric,
        reference_metric_value=reference_metric,
        **_metric_fields(metric_spec, baseline_metric, gain_basis, gain_note),
    )


def _solve_1d(
    miner_output: MinerOutput,
    auditor_output: AuditorOutput,
    epochs: int,
    gov_eq: str,
    bcs: List[str],
    params: Dict[str, float]
) -> SimulatorOutput:
    x_name = miner_output.independent_variables[0]
    y_name = miner_output.dependent_variables[0]

    # params: audited parameters merged with the simulation coefficient aliases and
    # auto-injected defaults (see dispatch_and_solve)
    params = params.copy()

    # Identify domain bounds
    domain_min, domain_max = get_domain_bounds(bcs, params, dependent_var=y_name)

    # 2. Parse symbols & expressions
    pde_rhs, _, x_sym, y_func, sym_dict = parse_equation_and_bcs(
        gov_eq, bcs, x_name, y_name, params
    )

    # 3. Solver execution routing
    method_requested = auditor_output.solver_method.lower()
    if method_requested not in ["pinn", "scipy", "analytical"]:
        print(f"[*] Solver method '{method_requested}' is not available for 1D problems. Using analytical.")
        method_requested = "analytical"
    # The solver that actually produced solution_primary (differs from the requested
    # one when a fallback is used); this is what solver_method reports.
    method_used = method_requested

    # We will try to solve the system analytically as the absolute reference
    analytical_sol_expr = None
    analytical_deriv = 0.0
    try:
        analytical_sol_expr, analytical_deriv = solve_analytical(
            pde_rhs, bcs, x_sym, y_func, sym_dict, params, domain_min, domain_max
        )
    except Exception as e:
        print(f"[*] SymPy Analytical solver failed: {e}. Falling back to SciPy BVP for reference.")

    # We also prepare a SciPy BVP numerical solver as secondary reference/primary
    scipy_sol_func = None
    scipy_deriv = 0.0
    try:
        scipy_sol_func, scipy_deriv = solve_scipy_bvp(
            pde_rhs, bcs, x_sym, y_func, sym_dict, params, domain_min, domain_max
        )
    except Exception as e:
        print(f"[*] SciPy BVP solver failed: {e}")

    # Metric spec (None = historical default: du/dx at domain_min) and baseline design.
    metric_spec, gain_note = _resolve_metric_spec(auditor_output)
    baseline_params, gain_basis, baseline_note = _baseline_setup(
        params, auditor_output, [gov_eq] + list(bcs), _is_bionic_effect_param
    )
    gain_note = _join_notes(gain_note, baseline_note)

    # Baseline solve. Without baseline_parameters the baseline is the design without the
    # bionic effect (slip length / slippage coefficient aliases and simulation coefficient
    # = 0). Geometric lengths such as film or coating thicknesses are kept: they define the
    # domain and the BC locations, not the bionic effect. None = no baseline available.
    # If the baseline equals the design, it is the reference solution itself (set below).
    baseline_deriv = None
    baseline_callables = None
    baseline_domain = (domain_min, domain_max)
    baseline_is_reference = baseline_params is not None and baseline_params == params
    if baseline_params is not None and not baseline_is_reference:
        if gain_basis == GAIN_BASIS_BASELINE:
            # The reference design may have other lengths, so the domain is derived again.
            baseline_domain = get_domain_bounds(bcs, baseline_params, dependent_var=y_name)
        b_min, b_max = baseline_domain
        try:
            baseline_expr, baseline_deriv = solve_analytical(
                pde_rhs, bcs, x_sym, y_func, sym_dict, baseline_params, b_min, b_max
            )
            baseline_callables = callables_from_expr(baseline_expr, x_sym)
        except Exception:
            try:
                baseline_interp, baseline_deriv = solve_scipy_bvp(
                    pde_rhs, bcs, x_sym, y_func, sym_dict, baseline_params, b_min, b_max
                )
                baseline_callables = callables_from_bvp(baseline_interp)
            except Exception as baseline_err:
                print(f"[*] Baseline solve failed: {baseline_err}. performance_gain_pct is reported as n/a (0).")
                gain_note = _join_notes(gain_note, f"baseline solve failed: {baseline_err}")
                baseline_deriv = None
        if baseline_deriv is not None and not math.isfinite(baseline_deriv):
            print(f"[*] Baseline wall derivative is not finite ({baseline_deriv}). performance_gain_pct is reported as n/a (0).")
            gain_note = _join_notes(gain_note, "baseline solution is not finite")
            baseline_deriv = None
            baseline_callables = None

    # Solve primary & reference based on selection
    sample_grid = np.linspace(domain_min, domain_max, 20)
    epochs_trained = 0
    final_loss = 0.0
    loss_history = []

    solution_primary = []
    solution_reference = []
    primary_metric = 0.0
    reference_metric = 0.0
    # (u, du/dx) callables of the primary and reference solutions, for the metric spec.
    primary_callables = None
    reference_callables = None

    if method_requested == "pinn":
        # 1. Primary: PyTorch PINN-lite
        try:
            def _use_cached_model(model, meta):
                nonlocal final_loss, loss_history, epochs_trained, solution_primary, primary_metric, primary_callables
                # Evaluate derivative at wall using autograd
                x_a_eval = torch.tensor([[domain_min]], requires_grad=True)
                u_a_eval = model(x_a_eval)
                pinn_deriv = torch.autograd.grad(u_a_eval, x_a_eval)[0].item()

                final_loss = meta.get("final_loss", 0.0)
                loss_history = meta.get("loss_history", [])
                epochs_trained = len(loss_history) if loss_history else meta.get("epochs_trained", epochs)

                with torch.no_grad():
                    torch_grid = torch.tensor(sample_grid, dtype=torch.float32).view(-1, 1)
                    solution_primary = model(torch_grid).view(-1).numpy().tolist()
                primary_metric = pinn_deriv
                primary_callables = callables_from_pinn(model)

            pretrained_loaded = load_cached_pinn(
                prefix="pinn_",
                label="",
                loaded_message="[*] Loaded pre-trained model successfully. Skipping training phase.",
                gov_eq=gov_eq,
                bcs=bcs,
                params=params,
                build_model=lambda: GenericPINN(),
                on_loaded=_use_cached_model,
                domain=(domain_min, domain_max),
                epochs=epochs,
            )

            if not pretrained_loaded:
                model, loss_history, pinn_deriv = solve_pytorch_pinn(
                    pde_rhs, bcs, x_sym, y_func, sym_dict, params, domain_min, domain_max, epochs
                )
                epochs_trained = epochs
                final_loss = loss_history[-1]

                with torch.no_grad():
                    torch_grid = torch.tensor(sample_grid, dtype=torch.float32).view(-1, 1)
                    solution_primary = model(torch_grid).view(-1).numpy().tolist()
                primary_metric = pinn_deriv
                primary_callables = callables_from_pinn(model)

                # Save the trained PINN model weights and metadata
                save_pinn_model(
                    model,
                    prefix="pinn_",
                    label="",
                    design_name=miner_output.design_name,
                    gov_eq=gov_eq,
                    bcs=bcs,
                    params=params,
                    final_loss=final_loss,
                    loss_history=loss_history,
                    domain=(domain_min, domain_max),
                    epochs=epochs,
                )

        except Exception as e:
            print(f"[*] PINN solver failed: {e}. Falling back to SciPy BVP.")
            method_requested = "scipy" # fall back
            method_used = "scipy"

    if method_requested == "scipy":
        # Primary is SciPy BVP
        if scipy_sol_func is not None:
            solution_primary = [float(scipy_sol_func(pt)) for pt in sample_grid]
            primary_metric = scipy_deriv
            primary_callables = callables_from_bvp(scipy_sol_func)
        elif analytical_sol_expr is not None:
            # Fallback to analytical
            print("[*] SciPy BVP unavailable. Using the analytical solution as primary.")
            f_lambdified = sp.lambdify(x_sym, analytical_sol_expr, "numpy")
            solution_primary = [float(f_lambdified(pt)) for pt in sample_grid]
            primary_metric = analytical_deriv
            primary_callables = callables_from_expr(analytical_sol_expr, x_sym)
            method_used = "analytical"
        else:
            raise RuntimeError("All primary numerical solvers failed.")

    elif method_requested == "analytical":
        # Primary is Analytical
        if analytical_sol_expr is not None:
            f_lambdified = sp.lambdify(x_sym, analytical_sol_expr, "numpy")
            solution_primary = [float(f_lambdified(pt)) for pt in sample_grid]
            primary_metric = analytical_deriv
            primary_callables = callables_from_expr(analytical_sol_expr, x_sym)
        elif scipy_sol_func is not None:
            # Fallback to SciPy
            print("[*] Analytical solution unavailable. Using SciPy BVP as primary.")
            solution_primary = [float(scipy_sol_func(pt)) for pt in sample_grid]
            primary_metric = scipy_deriv
            primary_callables = callables_from_bvp(scipy_sol_func)
            method_used = "scipy"
        else:
            raise RuntimeError("All primary analytical solvers failed.")

    # Reference Solution compilation (prefers Analytical, else SciPy)
    if analytical_sol_expr is not None:
        f_lambdified = sp.lambdify(x_sym, analytical_sol_expr, "numpy")
        solution_reference = [float(f_lambdified(pt)) for pt in sample_grid]
        reference_metric = analytical_deriv
        reference_callables = callables_from_expr(analytical_sol_expr, x_sym)
    elif scipy_sol_func is not None:
        solution_reference = [float(scipy_sol_func(pt)) for pt in sample_grid]
        reference_metric = scipy_deriv
        reference_callables = callables_from_bvp(scipy_sol_func)
    else:
        # Fallback reference = same as primary
        solution_reference = solution_primary
        reference_metric = primary_metric
        reference_callables = primary_callables

    # Metric spec: evaluate the requested quantity on the primary, reference and baseline
    # solutions. If it cannot be evaluated, the default wall-derivative metric is kept.
    if metric_spec is not None:
        try:
            spec_primary = metric_1d(metric_spec, *primary_callables, domain_min, domain_max, params)
            spec_reference = metric_1d(metric_spec, *reference_callables, domain_min, domain_max, params)
            primary_metric, reference_metric = spec_primary, spec_reference
        except Exception as spec_err:
            print(f"[*] Metric spec {metric_spec} could not be evaluated: {spec_err}. Using du/dx at the lower boundary.")
            gain_note = _join_notes(gain_note, f"metric spec not evaluable ({spec_err}); default metric du/dx at the lower boundary used")
            metric_spec = None

    baseline_metric = baseline_deriv
    if baseline_is_reference:
        baseline_metric = reference_metric
    elif metric_spec is not None and baseline_callables is not None:
        try:
            baseline_metric = metric_1d(metric_spec, *baseline_callables, baseline_domain[0], baseline_domain[1], baseline_params)
        except Exception as spec_err:
            gain_note = _join_notes(gain_note, f"metric spec not evaluable on the baseline ({spec_err})")
            baseline_metric = None
    elif metric_spec is not None:
        baseline_metric = None

    performance_gain, gain_basis, gain_note = _performance_gain(
        primary_metric, baseline_metric, gain_basis, gain_note,
        legacy=metric_spec is None and gain_basis == GAIN_BASIS_BIONIC,
        lower_better=lower_is_better(auditor_output),
    )

    # Calculate relative error
    abs_diff = np.abs(np.array(solution_primary) - np.array(solution_reference))
    ref_norm = np.abs(np.array(solution_reference))
    relative_err = float(np.sum(abs_diff) / np.sum(ref_norm + 1e-8))

    return SimulatorOutput(
        solver_method=method_used,
        epochs_trained=epochs_trained,
        final_loss=final_loss,
        loss_history=loss_history,
        performance_gain_pct=performance_gain,
        relative_error=relative_err,
        sample_points=sample_grid.tolist(),
        solution_primary=solution_primary,
        solution_reference=solution_reference,
        primary_metric_value=primary_metric,
        reference_metric_value=reference_metric,
        **_metric_fields(metric_spec, baseline_metric, gain_basis, gain_note),
    )
