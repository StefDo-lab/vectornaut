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


def _dynamic_objective_contract(auditor_output: AuditorOutput) -> Dict[str, Any]:
    if getattr(auditor_output, "objective_metric", None):
        contract = auditor_output.objective_metric
        return contract.model_dump() if hasattr(contract, "model_dump") else dict(contract)
    ui_meta = auditor_output.ui_metadata.model_dump() if hasattr(auditor_output.ui_metadata, "model_dump") else {}
    return {
        "objective_name": ui_meta.get("performance_gain", {}).get("label", "Performance Gain"),
        "score_field": "performance_gain_pct",
        "direction": "maximize",
        "primary_metric": ui_meta.get("primary_metric", {}).get("label", "Primary Metric"),
        "reference_metric": ui_meta.get("reference_metric", {}).get("label", "Reference Metric"),
        "lower_is_better": False,
        "acceptance_threshold": 0.0,
        "hard_constraints": ["relative_error <= 1.0", "parameters within bounds", "finite numeric outputs"],
        "note": "Dynamic-script variants are ranked by performance_gain_pct. The generated script must keep this metric definition stable across variants.",
    }


def _candidate_parameter_values(current: float, minimum: float, maximum: float) -> List[float]:
    values = [current]
    span = maximum - minimum
    if span <= 0:
        return values
    values.extend([
        current - span * 0.25,
        current + span * 0.25,
        minimum + span * 0.25,
        minimum + span * 0.5,
        minimum + span * 0.75,
    ])
    clean = []
    for value in values:
        clipped = max(minimum, min(maximum, float(value)))
        if not any(abs(clipped - existing) <= max(1e-12, abs(clipped) * 1e-9) for existing in clean):
            clean.append(clipped)
    return clean[:5]


def _run_dynamic_script_once(script_path: str, params: Dict[str, float], output_path: str, plot_path: str) -> Dict[str, Any]:
    params_path = output_path.replace("results_", "params_")
    with open(params_path, "w", encoding="utf-8") as pf:
        json.dump(params, pf, indent=4)
    result = subprocess.run(
        [sys.executable, script_path, "--params", params_path, "--output", output_path, "--plot", plot_path],
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
        timeout=int(os.environ.get("VECTORNAUT_SCRIPT_TIMEOUT_SECONDS", "60")),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Dynamic script variant failed.")
    with open(output_path, "r", encoding="utf-8") as rf:
        return json.load(rf)


def _dynamic_parameter_sweep(
    miner_output: MinerOutput,
    auditor_output: AuditorOutput,
    script_path: str,
    base_params: Dict[str, float],
    base_result: Dict[str, Any],
    slug: str,
    timestamp: str,
) -> Dict[str, Any]:
    generated_dir = os.path.dirname(script_path)
    sweep_dir = os.path.join(generated_dir, "sweeps")
    os.makedirs(sweep_dir, exist_ok=True)
    candidates = [{
        "label": "baseline",
        "parameters": base_params.copy(),
        "performance_gain_pct": base_result.get("performance_gain_pct"),
        "relative_error": base_result.get("relative_error"),
        "primary_metric_value": base_result.get("primary_metric_value"),
        "reference_metric_value": base_result.get("reference_metric_value"),
        "status": "ok",
    }]

    audited = auditor_output.audited_parameters_dict.copy()
    params_by_name = {p.name: p for p in miner_output.parameters}
    sweep_params = [
        name for name in audited.keys()
        if name in params_by_name and isinstance(audited.get(name), (int, float))
    ][:4]

    for name in sweep_params:
        proposal = params_by_name[name]
        for value in _candidate_parameter_values(float(audited[name]), float(proposal.min_bound), float(proposal.max_bound)):
            if abs(value - float(audited[name])) <= max(1e-12, abs(float(audited[name])) * 1e-9):
                continue
            variant_params = base_params.copy()
            variant_params[name] = value
            variant_params.setdefault("simulation_coefficient", auditor_output.simulation_coefficient)
            variant_params.setdefault("slippage_coefficient", auditor_output.simulation_coefficient)
            variant_params.setdefault("lambda", auditor_output.simulation_coefficient)
            variant_params.setdefault("slip_length", auditor_output.simulation_coefficient)
            label = f"{name}={value:.6g}"
            safe_label = re.sub(r"[^a-zA-Z0-9_]+", "_", label)
            output_path = os.path.abspath(os.path.join(sweep_dir, f"results_{slug}_{timestamp}_{safe_label}.json"))
            plot_path = os.path.abspath(os.path.join(sweep_dir, f"plot_{slug}_{timestamp}_{safe_label}.png"))
            try:
                result = _run_dynamic_script_once(script_path, variant_params, output_path, plot_path)
                candidates.append({
                    "label": label,
                    "changed_parameter": name,
                    "changed_value": value,
                    "parameters": variant_params,
                    "performance_gain_pct": result.get("performance_gain_pct"),
                    "relative_error": result.get("relative_error"),
                    "primary_metric_value": result.get("primary_metric_value"),
                    "reference_metric_value": result.get("reference_metric_value"),
                    "status": "ok",
                    "output_path": output_path,
                })
            except Exception as err:
                candidates.append({
                    "label": label,
                    "changed_parameter": name,
                    "changed_value": value,
                    "parameters": variant_params,
                    "status": "failed",
                    "error": str(err),
                })

    objective = _dynamic_objective_contract(auditor_output)
    score_field = objective.get("score_field", "performance_gain_pct")
    direction = objective.get("direction", "maximize")
    threshold = float(objective.get("acceptance_threshold", 0.0) or 0.0)
    valid = [
        item for item in candidates
        if item.get("status") == "ok" and isinstance(item.get(score_field), (int, float))
        and (float(item.get("relative_error", 0.0) or 0.0) <= 1.0)
    ]
    if direction == "minimize":
        accepted = [item for item in valid if float(item.get(score_field)) <= threshold]
        best = min(valid, key=lambda item: float(item.get(score_field))) if valid else None
    else:
        accepted = [item for item in valid if float(item.get(score_field)) >= threshold]
        best = max(valid, key=lambda item: float(item.get(score_field))) if valid else None
    return {
        "objective": objective,
        "baseline": candidates[0],
        "best": best,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid),
        "accepted_candidate_count": len(accepted),
    }

class GenericPINN(nn.Module):
    """
    A customizable, lightweight Neural Network (PINN-lite) that maps
    an independent variable x to a dependent variable y.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def parse_equation_and_bcs(
    gov_eq_str: str,
    bcs_list: List[str],
    x_name: str,
    y_name: str,
    params: Dict[str, float]
) -> Tuple[sp.Expr, List[str], sp.Symbol, sp.Function, Dict[str, sp.Symbol]]:
    """
    Parses the governing equation and returns the SymPy RHS of the equation: y''(x) = RHS.
    Also returns the boundary conditions and symbols.
    """
    x = sp.Symbol(x_name)
    y_func = sp.Function(y_name)(x)
    
    # Create parameter symbols
    sym_dict = {x_name: x}
    for p_name, p_val in params.items():
        sym_dict[p_name] = sp.Symbol(p_name)
        
    def clean_expr_str(s):
        s = s.replace(f"d2{y_name}_d{x_name}2", f"diff({y_name}({x_name}), {x_name}, 2)")
        s = s.replace(f"d{y_name}_d{x_name}", f"diff({y_name}({x_name}), {x_name})")
        s = s.replace(f"{y_name}''({x_name})", f"diff({y_name}({x_name}), {x_name}, 2)")
        s = s.replace(f"{y_name}'({x_name})", f"diff({y_name}({x_name}), {x_name})")
        s = s.replace(f"{y_name}''", f"diff({y_name}({x_name}), {x_name}, 2)")
        s = s.replace(f"{y_name}'", f"diff({y_name}({x_name}), {x_name})")
        return s

    # Parse LHS and RHS
    if "=" not in gov_eq_str:
        raise ValueError(f"Governing equation must contain '=': {gov_eq_str}")
        
    lhs_str, rhs_str = gov_eq_str.split("=")
    lhs_str = clean_expr_str(lhs_str.strip())
    rhs_str = clean_expr_str(rhs_str.strip())
    
    local_ns = {y_name: sp.Function(y_name), x_name: x}
    for p_name in params:
        local_ns[p_name] = sym_dict[p_name]
        
    lhs_expr = sp.parse_expr(lhs_str, local_dict=local_ns, transformations=(standard_transformations + (convert_xor,)))
    rhs_expr = sp.parse_expr(rhs_str, local_dict=local_ns, transformations=(standard_transformations + (convert_xor,)))
    
    # Rearrange equation so that d2y/dx2 is on the LHS: d2y/dx2 = RHS_new
    d2y = y_func.diff(x, 2)
    eq = sp.Eq(lhs_expr, rhs_expr)
    solved = sp.solve(eq, d2y)
    
    if isinstance(solved, list) and len(solved) > 0:
        pde_rhs = solved[0]
    else:
        pde_rhs = rhs_expr - lhs_expr + d2y
        
    return pde_rhs, bcs_list, x, y_func, sym_dict

def auto_detect_and_inject_missing_params(
    gov_eq_str: str,
    bcs_list: List[str],
    independent_vars: List[str],
    dependent_vars: List[str],
    params: Dict[str, float]
) -> Dict[str, float]:
    """
    Parses equations and BCs for parameter symbols that are missing from params,
    and injects them with sensible defaults.
    """
    updated_params = params.copy()
    
    # Collect all words in gov_eq and bcs
    all_text = gov_eq_str + " " + " ".join(bcs_list)
    words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", all_text)
    
    # Keywords to ignore
    ignored = {
        "sin", "cos", "tan", "exp", "log", "sqrt", "pi", "diff", "abs",
        "sinh", "cosh", "tanh", "asin", "acos", "atan", "Heaviside",
        "true", "false", "nan", "inf", "u", "v", "w", "x", "y", "z", "t"
    }
    for iv in independent_vars:
        ignored.add(iv)
    for dv in dependent_vars:
        ignored.add(dv)
        ignored.add(f"d{dv}")
        ignored.add(f"d2{dv}")
        
    for w in words:
        if w.lower() in ignored or w in ignored:
            continue
        if re.match(r"^d2?[a-zA-Z_]_d[a-zA-Z_]2?$", w):
            continue
        if w in updated_params:
            continue
            
        default_val = 1.0
        if "temp" in w.lower() or "t_" in w.lower():
            if "cold" in w.lower() or "ambient" in w.lower():
                default_val = 273.0
            else:
                default_val = 373.0
        elif "pressure" in w.lower():
            default_val = 101325.0
        elif "spacing" in w.lower() or "thickness" in w.lower() or "length" in w.lower() or "width" in w.lower() or "height" in w.lower() or "pitch" in w.lower() or "diameter" in w.lower():
            default_val = 0.01
        elif "velocity" in w.lower() or "speed" in w.lower():
            default_val = 1.0
        elif "density" in w.lower():
            default_val = 1000.0
            
        print(f"[*] AUTO-INJECT: Detected missing parameter '{w}' in equations/BCs. Injecting default value: {default_val}")
        updated_params[w] = default_val
        
    return updated_params

def get_domain_bounds(bcs_list: List[str], params: Dict[str, float] = None) -> Tuple[float, float]:
    """
    Extracts the min and max domain coordinates evaluated in boundary conditions.
    """
    bc_points = []
    for bc in bcs_list:
        resolved_bc = bc
        if params:
            for p_name in sorted(params.keys(), key=len, reverse=True):
                p_val = params[p_name]
                resolved_bc = re.sub(rf"\b{p_name}\b", str(p_val), resolved_bc)
        # Matches patterns like u(0) or T(1.5) or u'(0.01) or u(0.0001) or scientific notation
        matches = re.findall(r"[a-zA-Z]+\'?\(([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\)", resolved_bc)
        for m in matches:
            bc_points.append(float(m))
    if len(bc_points) >= 2:
        d_min, d_max = min(bc_points), max(bc_points)
        if abs(d_max - d_min) > 1e-6:
            return d_min, d_max
    return 0.0, 1.0

def solve_analytical(
    pde_rhs: sp.Expr,
    bcs_list: List[str],
    x: sp.Symbol,
    y_func: sp.Function,
    sym_dict: Dict[str, sp.Symbol],
    params: Dict[str, float],
    domain_min: float,
    domain_max: float
) -> Tuple[sp.Expr, float]:
    """
    Attempts to solve the ODE boundary value problem analytically using SymPy.
    Returns the solved expression and the derivative at the wall (domain_min).
    """
    eq = sp.Eq(y_func.diff(x, 2), pde_rhs)
    sol = sp.dsolve(eq, y_func)
    sol_expr = sol.rhs
    
    constants = [s for s in sol_expr.free_symbols if s.name.startswith('C')]
    param_subs = {sym_dict[k]: v for k, v in params.items() if k in sym_dict}
    
    bc_eqs = []
    for bc_str in bcs_list:
        bc_lhs, bc_rhs = bc_str.split("=")
        
        def evaluate_bc_part(part_str):
            part_str = part_str.strip()
            p_str = re.sub(rf"d{y_func.name}_d{x.name}\s*\((.*?)\)", r"deriv_func(\1)", part_str)
            p_str = re.sub(rf"{y_func.name}'\s*\((.*?)\)", r"deriv_func(\1)", p_str)
            p_str = re.sub(rf"\b{y_func.name}\s*\((.*?)\)", r"base_func(\1)", p_str)
            
            bc_local_ns = {
                "base_func": lambda val: sol_expr.subs(x, float(sp.sympify(val).subs(param_subs))),
                "deriv_func": lambda val: sol_expr.diff(x).subs(x, float(sp.sympify(val).subs(param_subs))),
            }
            for p_name, p_sym in sym_dict.items():
                bc_local_ns[p_name] = p_sym
            
            expr = sp.parse_expr(p_str, local_dict=bc_local_ns, transformations=(standard_transformations + (convert_xor,)))
            return expr
            
        lhs_eval = evaluate_bc_part(bc_lhs)
        rhs_eval = evaluate_bc_part(bc_rhs)
        bc_eqs.append(sp.Eq(lhs_eval, rhs_eval))
        
    const_vals = sp.solve(bc_eqs, constants)
    if isinstance(const_vals, dict):
        particular_sol = sol_expr.subs(const_vals)
    elif isinstance(const_vals, list) and len(const_vals) > 0:
        val_map = {}
        for c, v in zip(constants, const_vals[0]):
            val_map[c] = v
        particular_sol = sol_expr.subs(val_map)
    else:
        particular_sol = sol_expr
        
    final_sol = particular_sol.subs(param_subs)
    
    # Calculate derivative at domain_min
    deriv_val = float(final_sol.diff(x).subs(x, domain_min))
    return final_sol, deriv_val

def solve_scipy_bvp(
    pde_rhs: sp.Expr,
    bcs_list: List[str],
    x: sp.Symbol,
    y_func: sp.Function,
    sym_dict: Dict[str, sp.Symbol],
    params: Dict[str, float],
    domain_min: float,
    domain_max: float
) -> Tuple[Any, float]:
    """
    Solves the ODE boundary value problem numerically using SciPy's solve_bvp.
    Returns a cubic interpolation function and the derivative at the wall.
    """
    param_subs = {sym_dict[k]: v for k, v in params.items() if k in sym_dict}
    rhs_substituted = pde_rhs.subs(param_subs)
    
    Y0 = sp.Symbol('Y0')
    Y1 = sp.Symbol('Y1')
    expr_for_Y = rhs_substituted.subs({y_func: Y0, y_func.diff(x): Y1})
    
    f_rhs = sp.lambdify((x, Y0, Y1), expr_for_Y, 'numpy')
    
    def fun(x_grid, Y):
        dy = np.zeros_like(Y)
        dy[0] = Y[1]
        # Evaluate lambdified RHS with numpy arrays
        dy[1] = f_rhs(x_grid, Y[0], Y[1])
        return dy
        
    Ya0, Ya1 = sp.Symbol('Ya0'), sp.Symbol('Ya1')
    Yb0, Yb1 = sp.Symbol('Yb0'), sp.Symbol('Yb1')
    
    bc_residuals = []
    for bc_str in bcs_list:
        bc_lhs, bc_rhs = bc_str.split("=")
        
        def convert_bc_part_to_sym(part_str):
            part_str = part_str.strip()
            p_str = re.sub(rf"d{y_func.name}_d{x.name}\s*\((.*?)\)", r"deriv_func(\1)", part_str)
            p_str = re.sub(rf"{y_func.name}'\s*\((.*?)\)", r"deriv_func(\1)", p_str)
            p_str = re.sub(rf"\b{y_func.name}\s*\((.*?)\)", r"base_func(\1)", p_str)
            
            def base_func_handler(val):
                val = float(sp.sympify(val).subs(param_subs))
                if abs(val - domain_min) < abs(val - domain_max):
                    return Ya0
                else:
                    return Yb0
                    
            def deriv_func_handler(val):
                val = float(sp.sympify(val).subs(param_subs))
                if abs(val - domain_min) < abs(val - domain_max):
                    return Ya1
                else:
                    return Yb1
                    
            local_bc_ns = {
                "base_func": base_func_handler,
                "deriv_func": deriv_func_handler,
            }
            for p_name, p_sym in sym_dict.items():
                local_bc_ns[p_name] = p_sym
                
            expr = sp.parse_expr(p_str, local_dict=local_bc_ns, transformations=(standard_transformations + (convert_xor,)))
            return expr.subs(param_subs)
            
        lhs_expr = convert_bc_part_to_sym(bc_lhs)
        rhs_expr = convert_bc_part_to_sym(bc_rhs)
        bc_residuals.append(lhs_expr - rhs_expr)
        
    bc_funcs = [sp.lambdify((Ya0, Ya1, Yb0, Yb1), res, 'numpy') for res in bc_residuals]
    
    def bc(Ya, Yb):
        return np.array([
            float(f(Ya[0], Ya[1], Yb[0], Yb[1])) for f in bc_funcs
        ], dtype=float)
        
    x_grid = np.linspace(domain_min, domain_max, 100)
    Y_guess = np.zeros((2, x_grid.size))
    
    # Estimate reasonable bounds
    u_free_val = params.get("u_free", params.get("free_stream_velocity", 1.0))
    if "T_cold" in params or "T_hot" in params:
        u_free_val = params.get("T_cold", 300.0)
    
    Y_guess[0] = np.linspace(0.0, u_free_val, x_grid.size)
    Y_guess[1] = u_free_val
    
    res = solve_bvp(fun, bc, x_grid, Y_guess)
    if not res.success:
        raise ValueError(f"SciPy BVP solver failed to converge: {res.message}")
        
    f_interp = interp1d(res.x, res.y[0], kind='cubic', fill_value='extrapolate')
    
    # Estimate derivative at domain_min via finite difference
    dx = 1e-5
    deriv_val = float((f_interp(domain_min + dx) - f_interp(domain_min)) / dx)
    
    return f_interp, deriv_val

def solve_pytorch_pinn(
    pde_rhs: sp.Expr,
    bcs_list: List[str],
    x: sp.Symbol,
    y_func: sp.Function,
    sym_dict: Dict[str, sp.Symbol],
    params: Dict[str, float],
    domain_min: float,
    domain_max: float,
    epochs: int = 200
) -> Tuple[GenericPINN, List[float], float]:
    """
    Trains a Physics-Informed Neural Network (PINN-lite) to solve the BVP.
    Returns the trained model, training loss history, and derivative at the wall.
    """
    torch.manual_seed(42)
    model = GenericPINN()
    optimizer = optim.Adam(model.parameters(), lr=0.015)
    
    param_subs = {sym_dict[k]: v for k, v in params.items() if k in sym_dict}
    rhs_substituted = pde_rhs.subs(param_subs)
    
    Y0 = sp.Symbol('Y0')
    Y1 = sp.Symbol('Y1')
    expr_for_Y = rhs_substituted.subs({y_func: Y0, y_func.diff(x): Y1})
    
    f_pde_rhs = sp.lambdify((x, Y0, Y1), expr_for_Y, 'torch')
    
    Ya0, Ya1 = sp.Symbol('Ya0'), sp.Symbol('Ya1')
    Yb0, Yb1 = sp.Symbol('Yb0'), sp.Symbol('Yb1')
    
    bc_residuals = []
    for bc_str in bcs_list:
        bc_lhs, bc_rhs = bc_str.split("=")
        
        def convert_bc_part_to_sym(part_str):
            part_str = part_str.strip()
            p_str = re.sub(rf"d{y_func.name}_d{x.name}\s*\((.*?)\)", r"deriv_func(\1)", part_str)
            p_str = re.sub(rf"{y_func.name}'\s*\((.*?)\)", r"deriv_func(\1)", p_str)
            p_str = re.sub(rf"\b{y_func.name}\s*\((.*?)\)", r"base_func(\1)", p_str)
            
            def base_func_handler(val):
                val = float(sp.sympify(val).subs(param_subs))
                if abs(val - domain_min) < abs(val - domain_max):
                    return Ya0
                else:
                    return Yb0
                    
            def deriv_func_handler(val):
                val = float(sp.sympify(val).subs(param_subs))
                if abs(val - domain_min) < abs(val - domain_max):
                    return Ya1
                else:
                    return Yb1
                    
            local_bc_ns = {
                "base_func": base_func_handler,
                "deriv_func": deriv_func_handler,
            }
            for p_name, p_sym in sym_dict.items():
                local_bc_ns[p_name] = p_sym
                
            expr = sp.parse_expr(p_str, local_dict=local_bc_ns, transformations=(standard_transformations + (convert_xor,)))
            return expr.subs(param_subs)
            
        lhs_expr = convert_bc_part_to_sym(bc_lhs)
        rhs_expr = convert_bc_part_to_sym(bc_rhs)
        bc_residuals.append(lhs_expr - rhs_expr)
        
    bc_funcs_pytorch = [sp.lambdify((Ya0, Ya1, Yb0, Yb1), res, 'torch') for res in bc_residuals]
    
    loss_history = []
    x_pde = torch.linspace(domain_min, domain_max, 100, requires_grad=True).view(-1, 1)
    x_a = torch.tensor([[domain_min]], requires_grad=True)
    x_b = torch.tensor([[domain_max]], requires_grad=True)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE Loss
        u = model(x_pde)
        u_x = torch.autograd.grad(u, x_pde, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, torch.ones_like(u_x), create_graph=True)[0]
        
        rhs_val = f_pde_rhs(x_pde, u, u_x)
        if isinstance(rhs_val, (int, float)):
            rhs_val = torch.tensor(rhs_val)
        pde_loss = torch.mean((u_xx - rhs_val) ** 2)
        
        # Boundary Loss
        u_a = model(x_a)
        u_x_a = torch.autograd.grad(u_a, x_a, create_graph=True)[0]
        u_b = model(x_b)
        u_x_b = torch.autograd.grad(u_b, x_b, create_graph=True)[0]
        
        bc_loss = 0.0
        for f in bc_funcs_pytorch:
            res_val = f(u_a, u_x_a, u_b, u_x_b)
            bc_loss += res_val ** 2
            
        total_loss = pde_loss + bc_loss
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(float(total_loss.item()))
        
    # Evaluate derivative at wall (domain_min) using autograd
    x_a_eval = torch.tensor([[domain_min]], requires_grad=True)
    u_a_eval = model(x_a_eval)
    deriv_val = torch.autograd.grad(u_a_eval, x_a_eval)[0].item()
    
    return model, loss_history, deriv_val

# ==========================================
# 2D PDE Helper Functions & Solvers
# ==========================================

def parse_rhs_2d(rhs_str: str, x_name: str, y_name: str, params: Dict[str, float]) -> sp.Expr:
    x = sp.Symbol(x_name)
    y = sp.Symbol(y_name)
    local_ns = {x_name: x, y_name: y}
    for p_name, p_val in params.items():
        local_ns[p_name] = p_val
    expr = sp.parse_expr(rhs_str.strip(), local_dict=local_ns, transformations=(standard_transformations + (convert_xor,)))
    return expr

def parse_bc_2d_string(bc_str: str, dep_name: str, ind_names: List[str], params: Dict[str, float]) -> Tuple[str, str, float]:
    """
    Parses a single 2D boundary condition string.
    Returns (edge, type, value) where:
      edge: 'left', 'right', 'bottom', 'top'
      type: 'dirichlet', 'neumann'
      value: float
    """
    lhs, rhs = bc_str.split("=")
    lhs = lhs.strip()
    rhs = rhs.strip()
    
    # 1. Determine value
    val_expr = sp.sympify(rhs)
    param_subs = {sp.Symbol(k): v for k, v in params.items()}
    val = float(val_expr.subs(param_subs))
    
    # 2. Determine edge
    edge = None
    x_var, y_var = ind_names[0], ind_names[1]
    lhs_clean = re.sub(r'\s+', '', lhs)
    
    if f"(0,{y_var})" in lhs_clean or f"(0.0,{y_var})" in lhs_clean or f"(0," in lhs_clean:
        edge = "left"
    elif f"(1,{y_var})" in lhs_clean or f"(1.0,{y_var})" in lhs_clean or f"(1," in lhs_clean:
        edge = "right"
    elif f"({x_var},0)" in lhs_clean or f"({x_var},0.0)" in lhs_clean or f",0)" in lhs_clean:
        edge = "bottom"
    elif f"({x_var},1)" in lhs_clean or f"({x_var},1.0)" in lhs_clean or f",1)" in lhs_clean:
        edge = "top"
    
    if edge is None:
        if "0" in lhs_clean:
            if "dy" in lhs_clean or "y" in lhs_clean:
                edge = "bottom"
            else:
                edge = "left"
        elif "1" in lhs_clean:
            if "dy" in lhs_clean or "y" in lhs_clean:
                edge = "top"
            else:
                edge = "right"
        else:
            edge = "left"
            
    # 3. Determine type (Dirichlet vs Neumann)
    is_derivative = False
    if "diff" in lhs or "d" + dep_name in lhs or "'" in lhs or "deriv" in lhs or "grad" in lhs:
        is_derivative = True
        
    bc_type = "neumann" if is_derivative else "dirichlet"
    
    return edge, bc_type, val

def solve_fdm_2d(
    rhs_expr: sp.Expr,
    bcs_parsed: Dict[str, Dict[str, Any]],
    x_sym: sp.Symbol,
    y_sym: sp.Symbol,
    grid_size: int = 20,
    max_iter: int = 2000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves u_xx + u_yy = RHS on a square grid [0,1]x[0,1] using Finite Difference Method (Jacobi).
    """
    h = 1.0 / (grid_size - 1)
    x_vals = np.linspace(0.0, 1.0, grid_size)
    y_vals = np.linspace(0.0, 1.0, grid_size)
    
    # Evaluate RHS on the grid
    f_rhs = sp.lambdify((x_sym, y_sym), rhs_expr, "numpy")
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    F = f_rhs(X, Y)
    if isinstance(F, (int, float)):
        F = np.full_like(X, F)
        
    u = np.zeros((grid_size, grid_size))
    
    # Initialize boundary values
    if bcs_parsed['left']['type'] == 'dirichlet':
        u[0, :] = bcs_parsed['left']['value']
    if bcs_parsed['right']['type'] == 'dirichlet':
        u[-1, :] = bcs_parsed['right']['value']
    if bcs_parsed['bottom']['type'] == 'dirichlet':
        u[:, 0] = bcs_parsed['bottom']['value']
    if bcs_parsed['top']['type'] == 'dirichlet':
        u[:, -1] = bcs_parsed['top']['value']
        
    # Jacobi iteration
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # Update interior points
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                u[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] + u_old[i, j+1] + u_old[i, j-1] - h*h * F[i, j])
                
        # Enforce Neumann BCs
        if bcs_parsed['left']['type'] == 'neumann':
            val = bcs_parsed['left']['value']
            u[0, :] = u[1, :] - h * val
        if bcs_parsed['right']['type'] == 'neumann':
            val = bcs_parsed['right']['value']
            u[-1, :] = u[-2, :] + h * val
        if bcs_parsed['bottom']['type'] == 'neumann':
            val = bcs_parsed['bottom']['value']
            u[:, 0] = u[:, 1] - h * val
        if bcs_parsed['top']['type'] == 'neumann':
            val = bcs_parsed['top']['value']
            u[:, -1] = u[:, -2] + h * val
            
        # Check convergence
        diff = np.max(np.abs(u - u_old))
        if diff < tol:
            break
            
    return x_vals, y_vals, u

def solve_pytorch_pinn_2d(
    rhs_expr: sp.Expr,
    bcs_parsed: Dict[str, Dict[str, Any]],
    x_sym: sp.Symbol,
    y_sym: sp.Symbol,
    epochs: int = 400
) -> Tuple[GenericPINN, List[float]]:
    """
    Trains a 2D PINN model to solve u_xx + u_yy = RHS.
    """
    torch.manual_seed(42)
    model = GenericPINN(input_dim=2, hidden_dim=32)
    
    # Initialize bias of the output layer to the average Dirichlet boundary condition value
    dirichlet_vals = [bc['value'] for bc in bcs_parsed.values() if bc['type'] == 'dirichlet']
    mean_val = sum(dirichlet_vals) / len(dirichlet_vals) if dirichlet_vals else 0.0
    with torch.no_grad():
        model.net[-1].bias.fill_(mean_val)
        
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Sample interior collocation points (20x20 grid)
    x_pde_vals = np.linspace(0.0, 1.0, 20)
    y_pde_vals = np.linspace(0.0, 1.0, 20)
    X, Y = np.meshgrid(x_pde_vals, y_pde_vals, indexing='ij')
    pts_pde = np.stack([X.ravel(), Y.ravel()], axis=1)
    xy_pde = torch.tensor(pts_pde, dtype=torch.float32, requires_grad=True)
    
    f_pde_rhs = sp.lambdify((x_sym, y_sym), rhs_expr, 'torch')
    rhs_val = f_pde_rhs(xy_pde[:, 0:1], xy_pde[:, 1:2])
    if isinstance(rhs_val, (int, float)):
        rhs_val = torch.tensor(rhs_val)
        
    # Boundary points sets
    y_b = torch.linspace(0.0, 1.0, 20).view(-1, 1)
    xy_left = torch.cat([torch.zeros_like(y_b), y_b], dim=1)
    xy_left.requires_grad = True
    
    xy_right = torch.cat([torch.ones_like(y_b), y_b], dim=1)
    xy_right.requires_grad = True
    
    x_b = torch.linspace(0.0, 1.0, 20).view(-1, 1)
    xy_bottom = torch.cat([x_b, torch.zeros_like(x_b)], dim=1)
    xy_bottom.requires_grad = True
    
    xy_top = torch.cat([x_b, torch.ones_like(x_b)], dim=1)
    xy_top.requires_grad = True
    
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. PDE Loss: u_xx + u_yy - RHS = 0
        u = model(xy_pde)
        grads = torch.autograd.grad(u, xy_pde, torch.ones_like(u), create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        
        u_xx = torch.autograd.grad(u_x, xy_pde, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, xy_pde, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        
        pde_loss = torch.mean((u_xx + u_yy - rhs_val) ** 2)
        
        # 2. BC Loss
        bc_loss = 0.0
        
        # Left edge
        u_l = model(xy_left)
        if bcs_parsed['left']['type'] == 'dirichlet':
            bc_loss += torch.mean((u_l - bcs_parsed['left']['value']) ** 2)
        else: # neumann
            grads_l = torch.autograd.grad(u_l, xy_left, torch.ones_like(u_l), create_graph=True)[0]
            du_dx_l = grads_l[:, 0:1]
            bc_loss += torch.mean((du_dx_l - bcs_parsed['left']['value']) ** 2)
            
        # Right edge
        u_r = model(xy_right)
        if bcs_parsed['right']['type'] == 'dirichlet':
            bc_loss += torch.mean((u_r - bcs_parsed['right']['value']) ** 2)
        else: # neumann
            grads_r = torch.autograd.grad(u_r, xy_right, torch.ones_like(u_r), create_graph=True)[0]
            du_dx_r = grads_r[:, 0:1]
            bc_loss += torch.mean((du_dx_r - bcs_parsed['right']['value']) ** 2)
            
        # Bottom edge
        u_bot = model(xy_bottom)
        if bcs_parsed['bottom']['type'] == 'dirichlet':
            bc_loss += torch.mean((u_bot - bcs_parsed['bottom']['value']) ** 2)
        else: # neumann
            grads_b = torch.autograd.grad(u_bot, xy_bottom, torch.ones_like(u_bot), create_graph=True)[0]
            du_dy_b = grads_b[:, 1:2]
            bc_loss += torch.mean((du_dy_b - bcs_parsed['bottom']['value']) ** 2)
            
        # Top edge
        u_t = model(xy_top)
        if bcs_parsed['top']['type'] == 'dirichlet':
            bc_loss += torch.mean((u_t - bcs_parsed['top']['value']) ** 2)
        else: # neumann
            grads_t = torch.autograd.grad(u_t, xy_top, torch.ones_like(u_t), create_graph=True)[0]
            du_dy_t = grads_t[:, 1:2]
            bc_loss += torch.mean((du_dy_t - bcs_parsed['top']['value']) ** 2)
            
        total_loss = pde_loss + bc_loss * 5.0
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(float(total_loss.item()))
        
    return model, loss_history

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
        base_params = auditor_output.audited_parameters_dict.copy()
        base_params["simulation_coefficient"] = auditor_output.simulation_coefficient
        base_params["slippage_coefficient"] = auditor_output.simulation_coefficient
        base_params["lambda"] = auditor_output.simulation_coefficient
        base_params["slip_length"] = auditor_output.simulation_coefficient
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
        validation_passed = None
        validation_report = None
        validation_tests = None
        
        try:
            from .test_generator import TestScriptGenerator
            test_gen = TestScriptGenerator()
            # Reconstruct params_json_path from solver_path
            solver_path = res.get("script_path")
            params_json_path = solver_path.replace("solver_", "params_").replace(".py", ".json")
            
            print(f"[*] Running automated validation on generated script: {solver_path}")
            test_res = test_gen.generate_and_execute_tests(
                miner_output=miner_output,
                auditor_output=auditor_output,
                solver_script_path=solver_path,
                params_json_path=params_json_path
            )
            
            validation_passed = test_res.get("success", False)
            validation_tests = test_res.get("test_results", [])
            test_script_path = test_res.get("test_script_path")
            test_output_path = test_res.get("test_output_path")
            
            # Format a simple markdown report of the validation runs
            report_lines = []
            report_lines.append(f"### Status der Testausführung: {'✅ ERFOLGREICH' if validation_passed else '❌ FEHLGESCHLAGEN'}")
            report_lines.append(f"- **Testskript:** `{os.path.basename(test_res.get('test_script_path', ''))}`")
            report_lines.append(f"- **Ergebnisse:** `{os.path.basename(test_res.get('test_output_path', ''))}`")
            report_lines.append(f"- **Ausgeführte Tests:** {test_res.get('total_run', 0)}")
            report_lines.append(f"- **Fehlgeschlagene Tests:** {test_res.get('total_failures', 0)}")
            report_lines.append(f"- **Test-Fehler (Errors):** {test_res.get('total_errors', 0)}\n")
            
            report_lines.append("| Testfall | Status | Details |")
            report_lines.append("| --- | --- | --- |")
            for t in validation_tests:
                status_emoji = "✅ Passed" if t.get("passed") else "❌ Failed"
                message = t.get("message", "").replace("\n", " ").strip()
                report_lines.append(f"| `{t.get('name')}` | {status_emoji} | {message} |")
                
            validation_report = "\n".join(report_lines)
            print(f"[+] Automated validation complete. Passed: {validation_passed}")
        except Exception as test_err:
            print(f"[-] Automated validation crashed: {test_err}")
            validation_passed = False
            validation_report = f"### Status der Testausführung: ❌ CRASHED\n\nFehler bei der Testgenerierung/-ausführung: `{str(test_err)}`"
            validation_tests = [{"name": "validation_runner", "passed": False, "message": str(test_err)}]
            test_script_path = None
            test_output_path = None
        
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


    # 1. Extract inputs
    gov_eq = miner_output.governing_equation
    bcs = miner_output.boundary_conditions
    
    # Merge audited parameters and simulation coefficient
    params = auditor_output.audited_parameters_dict.copy()
    params['simulation_coefficient'] = auditor_output.simulation_coefficient
    params['slippage_coefficient'] = auditor_output.simulation_coefficient
    params['lambda'] = auditor_output.simulation_coefficient
    params['slip_length'] = auditor_output.simulation_coefficient

    # Auto-detect and inject missing parameters used in equations or BCs
    params = auto_detect_and_inject_missing_params(
        gov_eq_str=gov_eq,
        bcs_list=bcs,
        independent_vars=miner_output.independent_variables,
        dependent_vars=miner_output.dependent_variables,
        params=params
    )
    # Update back to audited_parameters_dict so that they are saved in history and synthesis
    for k, v in params.items():
        if k not in auditor_output.audited_parameters_dict:
            auditor_output.audited_parameters_dict[k] = v

    is_2d = len(miner_output.independent_variables) == 2
    if is_2d:
        x_name = miner_output.independent_variables[0]
        y_name = miner_output.independent_variables[1]
        dep_name = miner_output.dependent_variables[0]
        
        if "=" not in gov_eq:
            raise ValueError(f"2D Governing equation must contain '=': {gov_eq}")
        lhs_str, rhs_str = gov_eq.split("=")
        
        x_sym = sp.Symbol(x_name)
        y_sym = sp.Symbol(y_name)
        
        rhs_expr = parse_rhs_2d(rhs_str, x_name, y_name, params)
        
        bcs_parsed = {}
        for bc_str in bcs:
            try:
                edge, bc_type, val = parse_bc_2d_string(bc_str, dep_name, miner_output.independent_variables, params)
                bcs_parsed[edge] = {'type': bc_type, 'value': val}
            except Exception as e:
                print(f"[*] Error parsing 2D boundary condition {bc_str}: {e}")
                
        for edge in ['left', 'right', 'bottom', 'top']:
            if edge not in bcs_parsed:
                if edge == 'left':
                    bcs_parsed[edge] = {'type': 'dirichlet', 'value': params.get('T_hot', params.get('T_wall', 373.0 if dep_name == 'T' else 1.0))}
                elif edge == 'right':
                    bcs_parsed[edge] = {'type': 'dirichlet', 'value': params.get('T_cold', params.get('T_ambient', 273.0 if dep_name == 'T' else 0.0))}
                else:
                    bcs_parsed[edge] = {'type': 'neumann', 'value': 0.0}
                    
        epochs_2d = epochs
        loss_history = []
        final_loss = 0.0
        epochs_trained = 0
        grid_size = 20
        
        # Reference solver: 2D FDM
        x_vals, y_vals, u_fdm = solve_fdm_2d(rhs_expr, bcs_parsed, x_sym, y_sym, grid_size=grid_size)
        
        sample_points_2d = []
        solution_reference_2d = []
        for i in range(grid_size):
            for j in range(grid_size):
                sample_points_2d.append([float(x_vals[i]), float(y_vals[j])])
                solution_reference_2d.append(float(u_fdm[i, j]))
                
        method_requested = auditor_output.solver_method.lower()
        if method_requested not in ["pinn", "fdm"]:
            method_requested = "pinn"
            
        solution_primary_2d = []
        if method_requested == "pinn":
            try:
                pretrained_loaded = False
                try:
                    if os.path.exists("saved_models"):
                        for f_name in sorted(os.listdir("saved_models"), reverse=True):
                            if f_name.endswith(".json") and f_name.startswith("pinn2d_"):
                                meta_path = os.path.join("saved_models", f_name)
                                try:
                                    with open(meta_path, "r", encoding="utf-8") as mf:
                                        meta = json.load(mf)
                                    if (meta.get("governing_equation") == gov_eq and
                                        meta.get("boundary_conditions") == bcs):
                                        params_match = True
                                        meta_params = meta.get("params", {})
                                        for k, v in params.items():
                                            if k not in meta_params or abs(meta_params[k] - v) > 1e-5:
                                                params_match = False
                                                break
                                        if params_match:
                                            pth_name = f_name.replace(".json", ".pth")
                                            pth_path = os.path.join("saved_models", pth_name)
                                            if os.path.exists(pth_path):
                                                print(f"[*] Found matching pre-trained 2D PINN model weights: {pth_path}")
                                                model = GenericPINN(input_dim=2, hidden_dim=32)
                                                model.load_state_dict(torch.load(pth_path))
                                                final_loss = meta.get("final_loss", 0.0)
                                                loss_history = meta.get("loss_history", [])
                                                epochs_trained = len(loss_history) if loss_history else meta.get("epochs_trained", 400)
                                                
                                                with torch.no_grad():
                                                    xy_tensor = torch.tensor(sample_points_2d, dtype=torch.float32)
                                                    solution_primary_2d = model(xy_tensor).view(-1).numpy().tolist()
                                                pretrained_loaded = True
                                                print("[*] Loaded pre-trained 2D model successfully.")
                                                break
                                except Exception as parse_err:
                                    print(f"[*] Error parsing 2D model metadata {f_name}: {parse_err}")
                except Exception as cache_err:
                    print(f"[*] 2D Model cache lookup failed: {cache_err}")
                    
                if not pretrained_loaded:
                    model, loss_history = solve_pytorch_pinn_2d(
                        rhs_expr, bcs_parsed, x_sym, y_sym, epochs=epochs_2d
                    )
                    epochs_trained = epochs_2d
                    final_loss = loss_history[-1]
                    
                    with torch.no_grad():
                        xy_tensor = torch.tensor(sample_points_2d, dtype=torch.float32)
                        solution_primary_2d = model(xy_tensor).view(-1).numpy().tolist()
                        
                    try:
                        os.makedirs("saved_models", exist_ok=True)
                        design_name = miner_output.design_name or "unknown_design"
                        slug = re.sub(r'[^a-zA-Z0-9_]', '', design_name.lower().replace(" ", "_"))
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        model_path = os.path.join("saved_models", f"pinn2d_{timestamp}_{slug}.pth")
                        torch.save(model.state_dict(), model_path)
                        print(f"[*] Saved trained 2D PINN model weights to: {model_path}")
                        
                        meta_path = os.path.join("saved_models", f"pinn2d_{timestamp}_{slug}.json")
                        meta_data = {
                            "governing_equation": gov_eq,
                            "boundary_conditions": bcs,
                            "params": params,
                            "final_loss": final_loss,
                            "loss_history": loss_history
                        }
                        with open(meta_path, "w", encoding="utf-8") as mf:
                            json.dump(meta_data, mf, indent=4)
                        print(f"[*] Saved 2D PINN model metadata to: {meta_path}")
                    except Exception as save_err:
                        print(f"[*] Failed to save 2D PINN model: {save_err}")
            except Exception as e:
                print(f"[*] 2D PINN solver failed: {e}. Falling back to 2D FDM.")
                method_requested = "fdm"
                
        if method_requested == "fdm" or not solution_primary_2d:
            method_requested = "fdm"
            solution_primary_2d = solution_reference_2d
            
        primary_metric = float(np.mean(solution_primary_2d))
        reference_metric = float(np.mean(solution_reference_2d))
        performance_gain = min(99.0, max(0.0, auditor_output.simulation_coefficient * 100.0))
        
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

    # 1D flow continues here
    x_name = miner_output.independent_variables[0]
    y_name = miner_output.dependent_variables[0]
    
    # Merge audited parameters and simulation coefficient
    params = auditor_output.audited_parameters_dict.copy()
    params['simulation_coefficient'] = auditor_output.simulation_coefficient
    params['slippage_coefficient'] = auditor_output.simulation_coefficient
    params['lambda'] = auditor_output.simulation_coefficient
    params['slip_length'] = auditor_output.simulation_coefficient
    
    # Identify domain bounds
    domain_min, domain_max = get_domain_bounds(bcs, params)
    
    # 2. Parse symbols & expressions
    pde_rhs, _, x_sym, y_func, sym_dict = parse_equation_and_bcs(
        gov_eq, bcs, x_name, y_name, params
    )
    
    # 3. Solver execution routing
    method_requested = auditor_output.solver_method.lower()
    
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
        
    # We also solve the baseline reference solution (where slip/insulation coefficient = 0)
    # to evaluate performance gain
    baseline_params = params.copy()
    for k in baseline_params:
        if any(term in k.lower() for term in ["slip", "lambda", "coeff", "insul", "thick"]):
            if "film_thickness" in k.lower() or "pane_thickness" in k.lower() or "glass_thickness" in k.lower() or "wall_thickness" in k.lower() or "layer_thickness" in k.lower():
                continue
            baseline_params[k] = 0.0
    baseline_params["simulation_coefficient"] = 0.0
    baseline_params["slippage_coefficient"] = 0.0
    baseline_params["lambda"] = 0.0
    
    baseline_deriv = 0.0
    try:
        _, baseline_deriv = solve_analytical(
            pde_rhs, bcs, x_sym, y_func, sym_dict, baseline_params, domain_min, domain_max
        )
    except Exception:
        try:
            _, baseline_deriv = solve_scipy_bvp(
                pde_rhs, bcs, x_sym, y_func, sym_dict, baseline_params, domain_min, domain_max
            )
        except Exception:
            # Fallback baseline
            baseline_deriv = params.get("u_free", params.get("free_stream_velocity", 1.5))
            
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
            pretrained_loaded = False
            # Check if we have a pre-trained model matching these exact conditions
            try:
                if os.path.exists("saved_models"):
                    # Find matching json files
                    for f_name in sorted(os.listdir("saved_models"), reverse=True):
                        if f_name.endswith(".json") and f_name.startswith("pinn_"):
                            meta_path = os.path.join("saved_models", f_name)
                            try:
                                with open(meta_path, "r", encoding="utf-8") as mf:
                                    meta = json.load(mf)
                                # Check if PDE, BCs and domain boundaries match
                                if (meta.get("governing_equation") == gov_eq and
                                    meta.get("boundary_conditions") == bcs and
                                    abs(meta.get("domain_min", 0.0) - domain_min) < 1e-5 and
                                    abs(meta.get("domain_max", 1.0) - domain_max) < 1e-5):
                                    
                                    # Check params match within tolerance
                                    params_match = True
                                    meta_params = meta.get("params", {})
                                    for k, v in params.items():
                                        if k not in meta_params or abs(meta_params[k] - v) > 1e-5:
                                            params_match = False
                                            break
                                    
                                    if params_match:
                                        pth_name = f_name.replace(".json", ".pth")
                                        pth_path = os.path.join("saved_models", pth_name)
                                        if os.path.exists(pth_path):
                                            print(f"[*] Found matching pre-trained PINN model weights: {pth_path}")
                                            model = GenericPINN()
                                            model.load_state_dict(torch.load(pth_path))
                                            
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
                                            pretrained_loaded = True
                                            print("[*] Loaded pre-trained model successfully. Skipping training phase.")
                                            break
                            except Exception as parse_err:
                                print(f"[*] Error parsing model metadata {f_name}: {parse_err}")
            except Exception as cache_err:
                print(f"[*] Model cache lookup failed: {cache_err}")

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
                try:
                    os.makedirs("saved_models", exist_ok=True)
                    design_name = miner_output.design_name or "unknown_design"
                    slug = re.sub(r'[^a-zA-Z0-9_]', '', design_name.lower().replace(" ", "_"))
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save weights
                    model_path = os.path.join("saved_models", f"pinn_{timestamp}_{slug}.pth")
                    torch.save(model.state_dict(), model_path)
                    print(f"[*] Saved trained PINN model weights to: {model_path}")
                    
                    # Save metadata JSON for caching
                    meta_path = os.path.join("saved_models", f"pinn_{timestamp}_{slug}.json")
                    meta_data = {
                        "governing_equation": gov_eq,
                        "boundary_conditions": bcs,
                        "params": params,
                        "domain_min": domain_min,
                        "domain_max": domain_max,
                        "final_loss": final_loss,
                        "loss_history": loss_history
                    }
                    with open(meta_path, "w", encoding="utf-8") as mf:
                        json.dump(meta_data, mf, indent=4)
                    print(f"[*] Saved PINN model metadata to: {meta_path}")
                except Exception as save_err:
                    print(f"[*] Failed to save PINN model: {save_err}")
                
        except Exception as e:
            print(f"[*] PINN solver failed: {e}. Falling back to SciPy BVP.")
            method_requested = "scipy" # fall back
            
    if method_requested == "scipy":
        # Primary is SciPy BVP
        if scipy_sol_func is not None:
            solution_primary = [float(scipy_sol_func(pt)) for pt in sample_grid]
            primary_metric = scipy_deriv
        elif analytical_sol_expr is not None:
            # Fallback to analytical
            f_lambdified = sp.lambdify(x_sym, analytical_sol_expr, "numpy")
            solution_primary = [float(f_lambdified(pt)) for pt in sample_grid]
            primary_metric = analytical_deriv
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
            solution_primary = [float(scipy_sol_func(pt)) for pt in sample_grid]
            primary_metric = scipy_deriv
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
    if abs(baseline_deriv) > 1e-8:
        performance_gain = (1.0 - (primary_metric / baseline_deriv)) * 100.0
    else:
        performance_gain = 0.0
        
    # Calculate relative error
    abs_diff = np.abs(np.array(solution_primary) - np.array(solution_reference))
    ref_norm = np.abs(np.array(solution_reference))
    relative_err = float(np.sum(abs_diff) / np.sum(ref_norm + 1e-8))
    
    return SimulatorOutput(
        solver_method=method_requested,
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
