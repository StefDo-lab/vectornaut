import keyword
import math
import re
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, convert_xor
from typing import Dict, List, Optional, Tuple


# Python keywords that can show up as parameter names (e.g. the slip length `lambda` in
# the schema example "u(0) = lambda * du_dy(0)"). sp.parse_expr evaluates Python code,
# so these names are renamed token-wise before parsing. Operator-like keywords and
# True/False/None keep their Python meaning.
_OPERATOR_KEYWORDS = {"and", "or", "not", "is", "in", "if", "else", "True", "False", "None"}
_RENAMED_KEYWORDS = frozenset(k for k in keyword.kwlist if k not in _OPERATOR_KEYWORDS)


def safe_symbol_name(name: str) -> str:
    """
    Returns the identifier under which the parameter `name` is visible to sp.parse_expr
    (see escape_keywords). Non-keyword names are returned unchanged.
    """
    # The prefix keeps the escaped name distinct from user parameters such as 'lambda_'.
    return f"_kw_{name}" if name in _RENAMED_KEYWORDS else name


def escape_keywords(expr_str: str) -> str:
    """
    Renames Python-keyword identifiers (e.g. 'lambda') in an expression string so that it
    can be passed to sp.parse_expr. Use together with safe_symbol_name for the local_dict keys.
    """
    return re.sub(r"\b[A-Za-z_]\w*\b", lambda m: safe_symbol_name(m.group(0)), expr_str)


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
        # A bare dependent variable ("-k**2 * u") means u(x): the name itself is bound to
        # the undefined function class below, which cannot be used in arithmetic.
        s = re.sub(rf"\b{re.escape(y_name)}\b(?!\s*\()", f"{y_name}({x_name})", s)
        return s

    # Parse LHS and RHS
    if "=" not in gov_eq_str:
        raise ValueError(f"Governing equation must contain '=': {gov_eq_str}")
        
    lhs_str, rhs_str = gov_eq_str.split("=")
    lhs_str = clean_expr_str(lhs_str.strip())
    rhs_str = clean_expr_str(rhs_str.strip())
    
    local_ns = {y_name: sp.Function(y_name), x_name: x}
    for p_name in params:
        local_ns[safe_symbol_name(p_name)] = sym_dict[p_name]
        
    lhs_expr = sp.parse_expr(escape_keywords(lhs_str), local_dict=local_ns, transformations=(standard_transformations + (convert_xor,)))
    rhs_expr = sp.parse_expr(escape_keywords(rhs_str), local_dict=local_ns, transformations=(standard_transformations + (convert_xor,)))
    
    # Rearrange equation so that d2y/dx2 is on the LHS: d2y/dx2 = RHS_new
    d2y = y_func.diff(x, 2)
    eq = sp.Eq(lhs_expr, rhs_expr)
    solved = sp.solve(eq, d2y)
    
    if isinstance(solved, list) and len(solved) > 0:
        pde_rhs = solved[0]
    else:
        pde_rhs = rhs_expr - lhs_expr + d2y
        
    return pde_rhs, bcs_list, x, y_func, sym_dict

# Derivative-like names with one-letter variables (du_dx, d2u_dy2, dp_dx): never injected,
# even if the variable is not listed (the historical rule).
_SHORT_DERIVATIVE_PATTERN = re.compile(r"^d2?[a-zA-Z_]_d[a-zA-Z_]2?$")
# Fallback when the dependent/independent variables are unknown: derivative-like names whose
# variable names have up to four characters (dc_dxi, d2c_dxi2, d2u_dxdy). Longer parts are
# left alone so that parameters such as 'density_dimensionless' can still be injected.
_DERIVATIVE_NAME_PATTERN = re.compile(
    r"^d2?[A-Za-z_][A-Za-z0-9]{0,3}_d[A-Za-z_][A-Za-z0-9]{0,3}?(?:2|d[A-Za-z_][A-Za-z0-9]{0,3})?$"
)


def derivative_names(independent_vars: List[str], dependent_vars: List[str]) -> set:
    """
    The derivative notations the solvers understand, spelled out for the actual variable
    names: dU_dX, d2U_dX2, d2U_dXdY (both orders) and the subscripts U_X, U_XX, U_XY.
    """
    names = set()
    ivs = [iv for iv in (independent_vars or []) if iv]
    for dv in dependent_vars or []:
        if not dv:
            continue
        for a in ivs:
            names.update({f"d{dv}_d{a}", f"d2{dv}_d{a}2", f"{dv}_{a}", f"{dv}_{a}{a}"})
            for b in ivs:
                if a != b:
                    names.update({f"d2{dv}_d{a}d{b}", f"{dv}_{a}{b}"})
    return names


def auto_detect_and_inject_missing_params(
    gov_eq_str: str,
    bcs_list: List[str],
    independent_vars: List[str],
    dependent_vars: List[str],
    params: Dict[str, float]
) -> Dict[str, float]:
    """
    Parses equations and BCs for parameter symbols that are missing from params,
    and injects them with sensible defaults. Derivative notations of the actual variables
    (e.g. d2u_dxi2, dc_dxi for coordinate 'xi') are never injected, nor are other
    one-letter derivative-like names such as dp_dx. Without variable names, derivative-like
    names with short multi-character parts count as derivatives (_DERIVATIVE_NAME_PATTERN).
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
    derivatives = derivative_names(independent_vars, dependent_vars)
    variables_known = bool(derivatives)

    for w in words:
        if w.lower() in ignored or w in ignored:
            continue
        if w in derivatives or _SHORT_DERIVATIVE_PATTERN.match(w):
            continue
        if not variables_known and _DERIVATIVE_NAME_PATTERN.match(w):
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

# Function names that can appear in boundary-condition values (e.g. "c(1) = exp(-2)")
# and must not be mistaken for the dependent variable evaluated at a boundary point.
# Case-sensitive, so that dependent variables such as 'Gamma' or 'Max' still count.
_MATH_FUNCTION_NAMES = frozenset({
    "exp", "log", "ln", "log10", "sqrt", "abs", "Abs", "sin", "cos", "tan", "cot", "sec", "csc",
    "sinh", "cosh", "tanh", "coth", "asin", "acos", "atan", "atan2", "arcsin", "arccos", "arctan",
    "asinh", "acosh", "atanh", "arcsinh", "arccosh", "arctanh", "erf", "erfc", "gamma",
    "Heaviside", "heaviside", "sign", "max", "min", "Max", "Min", "floor", "ceiling", "ceil",
    "pow", "diff", "besselj", "bessely", "besseli", "besselk", "float", "int",
})

# name, optional primes, and a parenthesised argument without nested parentheses
_BC_CALL_PATTERN = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*'*\s*\(([^()]*)\)")
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _evaluate_bc_location(arg_str: str) -> Optional[float]:
    """Returns the numeric value of a BC location such as '0', '1e-05' or '0.5*2', else None."""
    arg_str = arg_str.strip()
    if not arg_str:
        return None
    if _NUMBER_PATTERN.fullmatch(arg_str):
        return float(arg_str)
    try:
        val = sp.parse_expr(escape_keywords(arg_str), transformations=(standard_transformations + (convert_xor,)))
        if not (isinstance(val, sp.Expr) and val.is_number):
            return None
        val = float(val)
    except Exception:
        return None
    return val if math.isfinite(val) else None


def get_domain_bounds(
    bcs_list: List[str],
    params: Dict[str, float] = None,
    dependent_var: Optional[str] = None
) -> Tuple[float, float]:
    """
    Extracts the min and max domain coordinates evaluated in boundary conditions.
    Only evaluations of the dependent variable or its derivatives count as BC locations:
    with dependent_var (e.g. 'u') these are u(..), u'(..), u''(..), du_d<x>(..) and
    d2u_d<x>2(..); without it, any call except known math functions (exp(-2), sin(1), ...).
    """
    resolved_bcs = []
    for bc in bcs_list:
        resolved_bc = bc
        if params:
            for p_name in sorted(params.keys(), key=len, reverse=True):
                p_val = params[p_name]
                resolved_bc = re.sub(rf"\b{re.escape(p_name)}\b", str(p_val), resolved_bc)
        resolved_bcs.append(resolved_bc)

    def collect_points(is_bc_function):
        points = []
        for resolved_bc in resolved_bcs:
            # Matches patterns like u(0) or T(1.5) or u'(0.01) or du_dy(0.0001) or scientific notation
            for name, arg in _BC_CALL_PATTERN.findall(resolved_bc):
                if not is_bc_function(name):
                    continue
                val = _evaluate_bc_location(arg)
                if val is not None:
                    points.append(val)
        return points

    bc_points = []
    if dependent_var:
        dv = re.escape(dependent_var)
        dependent_pattern = re.compile(rf"(?:{dv}|d2?{dv}_d\w+)")
        bc_points = collect_points(lambda name: dependent_pattern.fullmatch(name) is not None)
    if not bc_points:
        # No dependent variable given (or its name is not used in the BCs)
        bc_points = collect_points(lambda name: name not in _MATH_FUNCTION_NAMES)

    if len(bc_points) >= 2:
        d_min, d_max = min(bc_points), max(bc_points)
        # Any finite domain of positive length is valid, including sub-micron films.
        if math.isfinite(d_min) and math.isfinite(d_max) and d_max > d_min:
            return d_min, d_max
    return 0.0, 1.0
