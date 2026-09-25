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
from typing import Dict, List, Tuple, Any

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
    from .solvers.solvers_2d import parse_bcs_2d

    x_name = miner_output.independent_variables[0]
    y_name = miner_output.independent_variables[1]
    dep_name = miner_output.dependent_variables[0]

    if "=" not in gov_eq:
        raise ValueError(f"2D Governing equation must contain '=': {gov_eq}")
    lhs_str, rhs_str = gov_eq.split("=")

    x_sym = sp.Symbol(x_name)
    y_sym = sp.Symbol(y_name)

    rhs_expr = parse_rhs_2d(rhs_str, x_name, y_name, params)

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

    primary_metric = float(np.mean(solution_primary_2d))
    reference_metric = float(np.mean(solution_reference_2d))

    # Performance gain, as in 1D: change of the primary metric (mean field value) relative to a
    # baseline solve without the bionic effect, i.e. with the simulation coefficient and its
    # aliases set to 0: gain = (1 - primary / baseline) * 100. If the equation and BCs do not
    # use the coefficient, the baseline is the same problem and the gain is 0 by definition.
    coefficient_names = ["simulation_coefficient", "slippage_coefficient", "lambda", "slip_length"]
    problem_text = " ".join([gov_eq] + list(bcs))
    used_coefficients = [
        name for name in coefficient_names
        if params.get(name, 0.0) != 0.0 and re.search(rf"\b{re.escape(name)}\b", problem_text)
    ]
    performance_gain = 0.0
    if used_coefficients:
        try:
            baseline_params = params.copy()
            for name in coefficient_names:
                if name in baseline_params:
                    baseline_params[name] = 0.0
            baseline_rhs = parse_rhs_2d(rhs_str, x_name, y_name, baseline_params)
            baseline_bcs, baseline_x_bounds, baseline_y_bounds = parse_bcs_2d(
                bcs, dep_name, miner_output.independent_variables, baseline_params
            )
            _, _, u_baseline = solve_fdm_2d(
                baseline_rhs, baseline_bcs, x_sym, y_sym, grid_size=grid_size,
                x_bounds=baseline_x_bounds, y_bounds=baseline_y_bounds
            )
            baseline_metric = float(np.mean(u_baseline))
            if abs(baseline_metric) > 1e-8:
                performance_gain = (1.0 - (primary_metric / baseline_metric)) * 100.0
        except Exception as e:
            print(f"[*] 2D baseline solve (coefficient = 0) failed: {e}. Performance gain set to 0.")
            performance_gain = 0.0

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
        reference_metric_value=reference_metric
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
    domain_min, domain_max = get_domain_bounds(bcs, params)

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

    # We also solve the baseline reference solution without the bionic effect (slip
    # length / slippage coefficient aliases and simulation coefficient = 0) to evaluate
    # performance gain. Geometric lengths such as film or coating thicknesses are kept:
    # they define the domain and the BC locations, not the bionic effect.
    baseline_params = params.copy()
    for k in baseline_params:
        if _is_bionic_effect_param(k):
            baseline_params[k] = 0.0

    # None = no baseline available. If the effect parameters are already 0 (or absent),
    # the baseline is the reference solution itself (set below, gain 0 up to solver error).
    baseline_deriv = None
    baseline_is_reference = baseline_params == params
    if not baseline_is_reference:
        try:
            _, baseline_deriv = solve_analytical(
                pde_rhs, bcs, x_sym, y_func, sym_dict, baseline_params, domain_min, domain_max
            )
        except Exception:
            try:
                _, baseline_deriv = solve_scipy_bvp(
                    pde_rhs, bcs, x_sym, y_func, sym_dict, baseline_params, domain_min, domain_max
                )
            except Exception as baseline_err:
                print(f"[*] Baseline solve without the bionic effect failed: {baseline_err}. "
                      "performance_gain_pct is reported as 0.")
                baseline_deriv = None
        if baseline_deriv is not None and not math.isfinite(baseline_deriv):
            print(f"[*] Baseline wall derivative is not finite ({baseline_deriv}). performance_gain_pct is reported as 0.")
            baseline_deriv = None

    # Solve primary & reference based on selection
    sample_grid = np.linspace(domain_min, domain_max, 20)
    epochs_trained = 0
    final_loss = 0.0
    loss_history = []

    solution_primary = []
    solution_reference = []
    primary_metric = 0.0
    reference_metric = 0.0

    if method_requested == "pinn":
        # 1. Primary: PyTorch PINN-lite
        try:
            def _use_cached_model(model, meta):
                nonlocal final_loss, loss_history, epochs_trained, solution_primary, primary_metric
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
        elif analytical_sol_expr is not None:
            # Fallback to analytical
            print("[*] SciPy BVP unavailable. Using the analytical solution as primary.")
            f_lambdified = sp.lambdify(x_sym, analytical_sol_expr, "numpy")
            solution_primary = [float(f_lambdified(pt)) for pt in sample_grid]
            primary_metric = analytical_deriv
            method_used = "analytical"
        else:
            raise RuntimeError("All primary numerical solvers failed.")

    elif method_requested == "analytical":
        # Primary is Analytical
        if analytical_sol_expr is not None:
            f_lambdified = sp.lambdify(x_sym, analytical_sol_expr, "numpy")
            solution_primary = [float(f_lambdified(pt)) for pt in sample_grid]
            primary_metric = analytical_deriv
        elif scipy_sol_func is not None:
            # Fallback to SciPy
            print("[*] Analytical solution unavailable. Using SciPy BVP as primary.")
            solution_primary = [float(scipy_sol_func(pt)) for pt in sample_grid]
            primary_metric = scipy_deriv
            method_used = "scipy"
        else:
            raise RuntimeError("All primary analytical solvers failed.")

    # Reference Solution compilation (prefers Analytical, else SciPy)
    if analytical_sol_expr is not None:
        f_lambdified = sp.lambdify(x_sym, analytical_sol_expr, "numpy")
        solution_reference = [float(f_lambdified(pt)) for pt in sample_grid]
        reference_metric = analytical_deriv
    elif scipy_sol_func is not None:
        solution_reference = [float(scipy_sol_func(pt)) for pt in sample_grid]
        reference_metric = scipy_deriv
    else:
        # Fallback reference = same as primary
        solution_reference = solution_primary
        reference_metric = primary_metric

    # Calculate performance gain
    # E.g. drag reduction efficiency = (1.0 - (primary_deriv / baseline_deriv)) * 100
    if baseline_is_reference:
        baseline_deriv = reference_metric
    if baseline_deriv is not None and abs(baseline_deriv) > 1e-8:
        performance_gain = (1.0 - (primary_metric / baseline_deriv)) * 100.0
    else:
        performance_gain = 0.0

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
        reference_metric_value=reference_metric
    )
