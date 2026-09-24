import re
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, convert_xor
import numpy as np
import torch
import torch.optim as optim
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Any

from .pinn_model import GenericPINN


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
