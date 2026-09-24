import re
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, convert_xor
from typing import Dict, List, Tuple


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
