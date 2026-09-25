import re
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import standard_transformations, convert_xor
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import torch
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Union

from .pinn_model import GenericPINN


# ==========================================
# 2D PDE Helper Functions & Solvers
# ==========================================

EDGES_2D = ("left", "right", "bottom", "top")
UNIT_BOUNDS = (0.0, 1.0)


_TRANSFORMATIONS = standard_transformations + (convert_xor,)

# Terms of the general linear second-order operator, in this order:
#   c_xx*u_xx + c_yy*u_yy + c_xy*u_xy + c_x*u_x + c_y*u_y + c_u*u = f(x, y)
PDE_TERMS_2D = ("u_xx", "u_yy", "u_xy", "u_x", "u_y", "u")
_DERIVATIVE_ORDERS = {(2, 0): "u_xx", (0, 2): "u_yy", (1, 1): "u_xy", (1, 0): "u_x", (0, 1): "u_y"}


def _term_label(term: str, dep: str, x: str, y: str) -> str:
    return {
        "u_xx": f"d2{dep}_d{x}2", "u_yy": f"d2{dep}_d{y}2", "u_xy": f"d2{dep}_d{x}d{y}",
        "u_x": f"d{dep}_d{x}", "u_y": f"d{dep}_d{y}", "u": dep,
    }[term]


class LinearPDE2D:
    """
    A linear second-order 2D PDE

        c_xx*u_xx + c_yy*u_yy + c_xy*u_xy + c_x*u_x + c_y*u_y + c_u*u = rhs

    with coefficients and right-hand side given as SymPy expressions in x and y (parameters
    already substituted). Built by parse_pde_2d; solve_fdm_2d and solve_pytorch_pinn_2d accept
    it in place of a bare source term (a bare expression f means u_xx + u_yy = f).
    """

    def __init__(self, coeffs: Dict[str, sp.Expr], rhs: sp.Expr, x_sym: sp.Symbol, y_sym: sp.Symbol,
                 equation: str = ""):
        self.coeffs = {term: sp.sympify(coeffs.get(term, 0)) for term in PDE_TERMS_2D}
        self.rhs = sp.sympify(rhs)
        self.x_sym = x_sym
        self.y_sym = y_sym
        self.equation = equation
        if self.coeffs["u_xx"] == 0 or self.coeffs["u_yy"] == 0:
            missing = [t for t in ("u_xx", "u_yy") if self.coeffs[t] == 0]
            raise ValueError(
                f"2D governing equation {equation!r} has no {' / '.join(missing)} term; the 2D solvers "
                "need both second derivatives (elliptic equation a*u_xx + b*u_yy + ... = f with a, b of the same sign)."
            )
        # Constant coefficients: check ellipticity right away (varying ones are checked on the grid).
        principal = [self.coeffs[t] for t in ("u_xx", "u_yy", "u_xy")]
        if not any(c.free_symbols for c in principal):
            a, b, c = (float(v) for v in principal)
            _check_elliptic(np.array([a]), np.array([b]), np.array([c]), equation)

    @classmethod
    def laplacian(cls, rhs: sp.Expr, x_sym: sp.Symbol, y_sym: sp.Symbol) -> "LinearPDE2D":
        return cls({"u_xx": sp.Integer(1), "u_yy": sp.Integer(1)}, rhs, x_sym, y_sym, equation="u_xx + u_yy = f")

    @property
    def is_laplacian(self) -> bool:
        """True for u_xx + u_yy = f (the form the 2D solvers handled before the general operator)."""
        return (self.coeffs["u_xx"] == 1 and self.coeffs["u_yy"] == 1 and
                all(self.coeffs[t] == 0 for t in ("u_xy", "u_x", "u_y", "u")))

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Evaluates the coefficients and the right-hand side on the points (X, Y) and normalises
        them pointwise by s = sign(c_xx) * max(|c_xx|, |c_yy|), so that the returned equation has
        positive second-derivative coefficients of order one (the equation is unchanged; this
        only scales the PINN residual and the FDM rows). Returns (coefficients, rhs); terms whose
        coefficient is identically zero are omitted. For u_xx + u_yy = f this returns exactly
        ({'u_xx': 1, 'u_yy': 1}, f).
        Raises ValueError if a value is not finite or the equation is not elliptic somewhere.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        values = {
            term: _eval_field_2d(coeff, self.x_sym, self.y_sym, X, Y, f"coefficient of {term}")
            for term, coeff in self.coeffs.items() if coeff != 0
        }
        rhs = _eval_field_2d(self.rhs, self.x_sym, self.y_sym, X, Y, "source term")
        a = values["u_xx"]
        b = values["u_yy"]
        c = values.get("u_xy", np.zeros_like(a))
        _check_elliptic(a, b, c, self.equation)
        scale = np.sign(a) * np.maximum(np.abs(a), np.abs(b))
        if np.all(scale == 1.0):
            return values, rhs
        return {term: v / scale for term, v in values.items()}, rhs / scale

    def __repr__(self) -> str:
        terms = " + ".join(f"({c})*{t}" for t, c in self.coeffs.items() if c != 0)
        return f"LinearPDE2D({terms} = {self.rhs})"


def _check_elliptic(a: np.ndarray, b: np.ndarray, c: np.ndarray, equation: str) -> None:
    # a*u_xx + b*u_yy + c*u_xy is elliptic iff a*b - c^2/4 > 0 (then a and b have the same sign).
    disc = a * b - 0.25 * c * c
    if np.any(disc <= 0.0):
        k = int(np.argmin(disc))
        raise ValueError(
            f"2D governing equation {equation!r} is not elliptic (coefficients of u_xx, u_yy, u_xy = "
            f"{float(a.flat[k]):g}, {float(b.flat[k]):g}, {float(c.flat[k]):g}); the 2D solvers need "
            "a*u_xx + b*u_yy + ... with a, b of the same sign (and a*b > c^2/4 for a mixed term c*u_xy)."
        )


def _eval_field_2d(expr: sp.Expr, x_sym: sp.Symbol, y_sym: sp.Symbol, X: np.ndarray, Y: np.ndarray,
                   what: str) -> np.ndarray:
    """Evaluates a SymPy expression in x, y on the given points (numpy)."""
    expr = sp.sympify(expr)
    unknown = expr.free_symbols - {x_sym, y_sym}
    if unknown or expr.atoms(AppliedUndef):
        raise ValueError(f"2D {what} may only depend on {x_sym} and {y_sym}: {expr}")
    X = np.asarray(X, dtype=float)
    f = sp.lambdify((x_sym, y_sym), expr, "numpy")
    try:
        with np.errstate(all="ignore"):
            F = np.asarray(f(X, Y), dtype=float) * np.ones_like(X, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"2D {what} cannot be evaluated as a real number: {expr} ({e})") from e
    if not np.all(np.isfinite(F)):
        raise ValueError(f"2D {what} is not finite on the domain: {expr}")
    return F


def _as_pde_2d(pde: Union[LinearPDE2D, sp.Expr, float], x_sym: sp.Symbol, y_sym: sp.Symbol) -> LinearPDE2D:
    if isinstance(pde, LinearPDE2D):
        return pde
    return LinearPDE2D.laplacian(sp.sympify(pde), x_sym, y_sym)


def _rewrite_derivative_notation(s: str, dep: str, x: str, y: str, params: Dict[str, float]) -> str:
    """
    Rewrites the derivative notations of the governing equation into SymPy calls on dep(x, y):
    d2u_dx2, d2u_dy2, d2u_dxdy / d2u_dydx, du_dx, du_dy (optionally followed by '(x, y)'),
    subscripts u_xx, u_yy, u_xy / u_yx, u_x, u_y (unless such a name is a parameter),
    and a bare u (-> u(x, y)). diff(...), Derivative(...) and laplacian(u) are handled by SymPy.
    """
    d, X, Y = re.escape(dep), re.escape(x), re.escape(y)
    args = rf"(?:\s*\(\s*{X}\s*,\s*{Y}\s*\))?"
    u = f"{dep}({x}, {y})"

    def deriv(*variables: str) -> str:
        return f"Derivative({u}, {', '.join(variables)})"

    patterns = [
        (rf"\bd2{d}_d{X}2\b{args}", deriv(x, x)),
        (rf"\bd2{d}_d{Y}2\b{args}", deriv(y, y)),
        (rf"\bd2{d}_d{X}d{Y}\b{args}", deriv(x, y)),
        (rf"\bd2{d}_d{Y}d{X}\b{args}", deriv(x, y)),
        (rf"\bd{d}_d{X}\b{args}", deriv(x)),
        (rf"\bd{d}_d{Y}\b{args}", deriv(y)),
    ]
    for pattern, replacement in patterns:
        s = re.sub(pattern, lambda m, r=replacement: r, s)

    subscripts = {x + x: (x, x), y + y: (y, y), x + y: (x, y), y + x: (x, y), x: (x,), y: (y,)}

    def subscript(m: "re.Match") -> str:
        if m.group(0).split("(")[0].strip() in params:
            return m.group(0)
        return deriv(*subscripts[m.group(1)])

    alternatives = "|".join(re.escape(k) for k in sorted(subscripts, key=len, reverse=True))
    s = re.sub(rf"\b{d}_({alternatives})\b{args}", subscript, s)
    # A bare dependent variable ("- k**2 * u") means u(x, y).
    s = re.sub(rf"\b{d}\b(?!\s*\()", lambda m: u, s)
    return s


def _infer_dependent_2d(equation: str, x: str, y: str, params: Dict[str, float]) -> str:
    """Finds the dependent variable of a 2D equation from its second-derivative terms."""
    X, Y = re.escape(x), re.escape(y)
    candidates = set(re.findall(rf"\bd2([A-Za-z_]\w*?)_d(?:{X}2|{Y}2|{X}d{Y}|{Y}d{X})\b", equation))
    candidates |= set(re.findall(r"\b(?:laplacian|Laplacian|diff|Derivative)\(\s*([A-Za-z_]\w*)", equation))
    candidates |= (set(re.findall(rf"\b([A-Za-z_]\w*?)_{X}{X}\b", equation)) &
                   set(re.findall(rf"\b([A-Za-z_]\w*?)_{Y}{Y}\b", equation)))
    candidates -= set(params) | {x, y}
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot identify the dependent variable of the 2D governing equation {equation!r} "
            f"(candidates: {sorted(candidates) or 'none'})."
        )
    return candidates.pop()


def parse_pde_2d(
    equation: str,
    dep_name: Optional[str],
    x_name: str,
    y_name: str,
    params: Dict[str, float]
) -> LinearPDE2D:
    """
    Parses a whole 2D governing equation ('lhs = rhs'), moves all terms to one side and
    extracts the linear operator

        c_xx*u_xx + c_yy*u_yy + c_xy*u_xy + c_x*u_x + c_y*u_y + c_u*u = f(x, y)

    Coefficients may be numbers, parameter expressions or functions of x and y. Recognised
    notations: d2u_dx2, d2u_dy2, d2u_dxdy, du_dx, du_dy, u_xx/u_yy/u_xy/u_x/u_y, diff(u, x, 2),
    Derivative(u, x), laplacian(u) and u itself (reaction terms such as -k**2*u).
    Raises ValueError for anything the 2D solvers cannot represent: no '=' or several, nonlinear
    terms (u*u_x, u**2, sin(u), ...), third or higher derivatives, unknown symbols or functions,
    or a non-elliptic principal part.
    """
    from .parsing import escape_keywords, safe_symbol_name

    if equation.count("=") != 1:
        raise ValueError(f"2D governing equation must contain exactly one '=': {equation!r}")
    if dep_name is None:
        dep_name = _infer_dependent_2d(equation, x_name, y_name, params)

    x = sp.Symbol(x_name)
    y = sp.Symbol(y_name)
    u_func = sp.Function(dep_name)
    u = u_func(x, y)

    def laplacian(f):
        return sp.diff(f, x, 2) + sp.diff(f, y, 2)

    local_ns: Dict[str, Any] = {safe_symbol_name(p): v for p, v in params.items()}
    local_ns.update({"laplacian": laplacian, "Laplacian": laplacian,
                     safe_symbol_name(x_name): x, safe_symbol_name(y_name): y,
                     safe_symbol_name(dep_name): u_func})

    sides = []
    for side in equation.split("="):
        text = _rewrite_derivative_notation(side.strip(), dep_name, x_name, y_name, params)
        try:
            sides.append(sp.sympify(sp.parse_expr(escape_keywords(text), local_dict=local_ns,
                                                  transformations=_TRANSFORMATIONS)))
        except Exception as e:
            raise ValueError(f"Could not parse 2D governing equation {equation!r}: {e}") from e
    expr = sides[0] - sides[1]

    # Replace the derivatives of u (and u itself) by placeholder symbols.
    placeholders = {term: sp.Dummy(term) for term in PDE_TERMS_2D}
    replacements = {}
    for der in expr.atoms(sp.Derivative):
        counts = dict(der.variable_count)
        order = (int(counts.pop(x, 0)), int(counts.pop(y, 0)))
        term = _DERIVATIVE_ORDERS.get(order)
        if der.expr != u or counts or term is None:
            raise ValueError(
                f"2D governing equation {equation!r} contains the unsupported derivative {der}; "
                f"only first and second derivatives of {dep_name}({x_name}, {y_name}) are supported."
            )
        replacements[der] = placeholders[term]
    expr = expr.xreplace(replacements).xreplace({u: placeholders["u"]})
    undefined = expr.atoms(AppliedUndef)
    if undefined:
        raise ValueError(
            f"2D governing equation {equation!r} uses unsupported function(s) {sorted(map(str, undefined))}; "
            f"{dep_name} may only appear as {dep_name}({x_name}, {y_name}) or through its derivatives."
        )

    unknown_symbols = set(placeholders.values())
    coeffs = {}
    for term, sym in placeholders.items():
        coeff = sp.diff(expr, sym)
        if coeff.free_symbols & unknown_symbols:
            raise ValueError(
                f"2D governing equation {equation!r} is nonlinear in {_term_label(term, dep_name, x_name, y_name)}; "
                "the 2D solvers support only linear equations "
                "a*u_xx + b*u_yy + c*u_xy + d*u_x + e*u_y + g*u = f(x, y)."
            )
        coeffs[term] = sp.sympify(coeff)
    rhs = sp.sympify(-expr.xreplace({sym: 0 for sym in placeholders.values()}))

    for what, value in list(coeffs.items()) + [("rhs", rhs)]:
        unknown = value.free_symbols - {x, y}
        if unknown:
            raise ValueError(
                f"2D governing equation {equation!r}: the {'right-hand side' if what == 'rhs' else 'coefficient of ' + _term_label(what, dep_name, x_name, y_name)} "
                f"depends on unknown symbol(s) {sorted(map(str, unknown))}."
            )
    if not any(coeffs[t] != 0 for t in _DERIVATIVE_ORDERS.values()):
        raise ValueError(f"2D governing equation {equation!r} contains no derivative of {dep_name}.")
    return LinearPDE2D(coeffs, rhs, x, y, equation=equation)


def parse_rhs_2d(rhs_str: str, x_name: str, y_name: str, params: Dict[str, float]) -> Union[sp.Expr, LinearPDE2D]:
    """
    Parses a source term f(x, y) (the right-hand side of u_xx + u_yy = f). A whole equation
    ('lhs = rhs') is parsed with parse_pde_2d instead (dependent variable inferred) and returned
    as a LinearPDE2D; both are accepted by solve_fdm_2d and solve_pytorch_pinn_2d.
    """
    if "=" in rhs_str:
        return parse_pde_2d(rhs_str, None, x_name, y_name, params)
    x = sp.Symbol(x_name)
    y = sp.Symbol(y_name)
    local_ns = {x_name: x, y_name: y}
    for p_name, p_val in params.items():
        local_ns[p_name] = p_val
    expr = sp.parse_expr(rhs_str.strip(), local_dict=local_ns, transformations=_TRANSFORMATIONS)
    return expr


def _parse_numeric_2d(expr_str: str, params: Dict[str, float]) -> float:
    """Evaluates a coordinate expression such as '0', '2.5' or 'L' (a parameter) to a float."""
    local_ns = {p_name: p_val for p_name, p_val in params.items()}
    expr = sp.parse_expr(expr_str, local_dict=local_ns, transformations=(standard_transformations + (convert_xor,)))
    return float(expr)


def _locate_bc_2d(lhs: str, x_var: str, y_var: str, params: Dict[str, float]) -> Tuple[str, float]:
    """
    Determines on which line a 2D boundary condition lives.
    Returns (axis, coordinate): axis 'x' means the BC fixes x (a vertical edge, left/right),
    axis 'y' means it fixes y (a horizontal edge, bottom/top).
    """
    lhs_clean = re.sub(r'\s+', '', lhs)

    # 1. Argument list of the form (x_value, y) or (x, y_value), e.g. T(2, y), dT_dy(x, H)
    arg_pairs = re.findall(r"\(([^(),]+),([^(),]+)\)", lhs_clean)
    for a_str, b_str in reversed(arg_pairs):
        if a_str == x_var and b_str != y_var:
            try:
                return "y", _parse_numeric_2d(b_str, params)
            except Exception:
                pass
        elif b_str == y_var and a_str != x_var:
            try:
                return "x", _parse_numeric_2d(a_str, params)
            except Exception:
                pass

    # 2. Legacy heuristics for other notations (unit square coordinates 0 and 1)
    edge = None
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

    return {
        "left": ("x", 0.0),
        "right": ("x", 1.0),
        "bottom": ("y", 0.0),
        "top": ("y", 1.0),
    }[edge]


def _split_bc_2d(bc_str: str, dep_name: str) -> Tuple[str, str, str]:
    """Splits a BC string into (lhs, rhs, bc_type)."""
    if bc_str.count("=") != 1:
        raise ValueError(f"expected exactly one '=' in boundary condition: {bc_str!r}")
    lhs, rhs = bc_str.split("=")
    lhs = lhs.strip()
    rhs = rhs.strip()

    # Determine type (Dirichlet vs Neumann)
    is_derivative = False
    if "diff" in lhs or "d" + dep_name in lhs or "'" in lhs or "deriv" in lhs or "grad" in lhs:
        is_derivative = True

    bc_type = "neumann" if is_derivative else "dirichlet"
    return lhs, rhs, bc_type


def _parse_bc_value_2d(
    rhs: str,
    x_var: str,
    y_var: str,
    axis: str,
    coord: float,
    params: Dict[str, float]
) -> Union[float, sp.Expr]:
    """
    Parses the value of an edge BC. Constant values are returned as float, values that vary
    along the edge as a SymPy expression in the along-edge coordinate (y on left/right edges,
    x on bottom/top edges). Raises ValueError for values that depend on anything else (e.g.
    on the unknown field itself, as in Robin conditions, which the 2D solvers do not support).
    """
    x = sp.Symbol(x_var)
    y = sp.Symbol(y_var)
    local_ns = {p_name: p_val for p_name, p_val in params.items()}
    local_ns[x_var] = x
    local_ns[y_var] = y
    expr = sp.parse_expr(rhs, local_dict=local_ns, transformations=(standard_transformations + (convert_xor,)))
    expr = sp.sympify(expr)

    fixed_sym, along_sym = (x, y) if axis == "x" else (y, x)
    expr = expr.subs(fixed_sym, coord)

    undefined = expr.atoms(AppliedUndef)
    if undefined:
        raise ValueError(f"value depends on unsupported function(s) {sorted(map(str, undefined))}")
    unknown = expr.free_symbols - {along_sym}
    if unknown:
        raise ValueError(f"value depends on unknown symbol(s) {sorted(map(str, unknown))}")

    if not expr.free_symbols:
        val = complex(expr.evalf())
        if abs(val.imag) > 1e-12 or not np.isfinite(val.real):
            raise ValueError(f"value is not a finite real number: {expr}")
        return float(val.real)
    return expr


def _edge_bounds(coords: List[float]) -> Tuple[float, float]:
    """Rectangle bounds along one axis from the coordinates at which BCs are given."""
    if not coords:
        return UNIT_BOUNDS
    lo, hi = min(coords), max(coords)
    if abs(hi - lo) > 1e-12 * max(1.0, abs(hi), abs(lo)):
        return lo, hi
    # Only one edge of this axis is specified: assume the other one at 0 (or at 1 for a BC at 0).
    c = lo
    if c > 0.0:
        return 0.0, c
    if c < 0.0:
        return c, 0.0
    return UNIT_BOUNDS


def _edge_name(axis: str, coord: float, bounds: Tuple[float, float]) -> str:
    lo, hi = bounds
    tol = 1e-9 * max(1.0, abs(hi - lo))
    if abs(coord - lo) <= tol:
        return "left" if axis == "x" else "bottom"
    if abs(coord - hi) <= tol:
        return "right" if axis == "x" else "top"
    raise ValueError(f"{'x' if axis == 'x' else 'y'} = {coord} is not on the domain boundary {bounds}")


def parse_bc_2d_string(
    bc_str: str,
    dep_name: str,
    ind_names: List[str],
    params: Dict[str, float],
    x_bounds: Tuple[float, float] = UNIT_BOUNDS,
    y_bounds: Tuple[float, float] = UNIT_BOUNDS
) -> Tuple[str, str, Union[float, sp.Expr]]:
    """
    Parses a single 2D boundary condition string.
    Returns (edge, type, value) where:
      edge: 'left', 'right', 'bottom', 'top' (edges of the rectangle x_bounds x y_bounds)
      type: 'dirichlet', 'neumann'
      value: float, or a SymPy expression in the along-edge coordinate for non-constant values
    Raises ValueError if the BC cannot be parsed.
    """
    x_var, y_var = ind_names[0], ind_names[1]
    lhs, rhs, bc_type = _split_bc_2d(bc_str, dep_name)
    axis, coord = _locate_bc_2d(lhs, x_var, y_var, params)
    edge = _edge_name(axis, coord, x_bounds if axis == "x" else y_bounds)
    val = _parse_bc_value_2d(rhs, x_var, y_var, axis, coord, params)
    return edge, bc_type, val


def parse_bcs_2d(
    bcs: List[str],
    dep_name: str,
    ind_names: List[str],
    params: Dict[str, float]
) -> Tuple[Dict[str, Dict[str, Any]], Tuple[float, float], Tuple[float, float]]:
    """
    Parses all 2D boundary conditions and derives the rectangular domain
    [x_min, x_max] x [y_min, y_max] from the coordinates at which they are given
    (numbers or parameter names, e.g. 'T(L, y) = 0').
    Returns (bcs_parsed, x_bounds, y_bounds); bcs_parsed maps every edge to
    {'type': 'dirichlet'|'neumann', 'value': float or SymPy expression, 'var': along-edge symbol}.
    Raises ValueError if a BC cannot be parsed or no edge carries a Dirichlet condition.
    """
    x_var, y_var = ind_names[0], ind_names[1]
    x_sym, y_sym = sp.Symbol(x_var), sp.Symbol(y_var)

    located = []
    for bc_str in bcs:
        try:
            lhs, rhs, bc_type = _split_bc_2d(bc_str, dep_name)
            axis, coord = _locate_bc_2d(lhs, x_var, y_var, params)
        except Exception as e:
            print(f"[*] Error parsing 2D boundary condition {bc_str}: {e}")
            raise ValueError(f"Could not parse 2D boundary condition '{bc_str}': {e}") from e
        located.append((bc_str, rhs, bc_type, axis, coord))

    x_bounds = _edge_bounds([coord for _, _, _, axis, coord in located if axis == "x"])
    y_bounds = _edge_bounds([coord for _, _, _, axis, coord in located if axis == "y"])

    bcs_parsed: Dict[str, Dict[str, Any]] = {}
    for bc_str, rhs, bc_type, axis, coord in located:
        try:
            edge = _edge_name(axis, coord, x_bounds if axis == "x" else y_bounds)
            val = _parse_bc_value_2d(rhs, x_var, y_var, axis, coord, params)
        except Exception as e:
            print(f"[*] Error parsing 2D boundary condition {bc_str}: {e}")
            raise ValueError(f"Could not parse 2D boundary condition '{bc_str}': {e}") from e
        bcs_parsed[edge] = {'type': bc_type, 'value': val, 'var': y_sym if axis == "x" else x_sym}

    # Edges without a boundary condition are treated as insulated / zero-flux walls
    # (homogeneous Neumann, du/dn = 0).
    for edge in EDGES_2D:
        if edge not in bcs_parsed:
            print(f"[*] No 2D boundary condition for the {edge} edge; assuming zero flux (Neumann 0).")
            bcs_parsed[edge] = {'type': 'neumann', 'value': 0.0, 'var': y_sym if edge in ("left", "right") else x_sym}

    if not any(bc['type'] == 'dirichlet' for bc in bcs_parsed.values()):
        raise ValueError("2D problem has no Dirichlet boundary condition; the solution is not unique.")

    return bcs_parsed, x_bounds, y_bounds


def _bc_values(bc: Dict[str, Any], coords: np.ndarray) -> np.ndarray:
    """Evaluates an edge BC value at the given along-edge coordinates."""
    coords = np.asarray(coords, dtype=float)
    val = bc['value']
    if isinstance(val, sp.Basic) and val.free_symbols:
        f_val = sp.lambdify(bc['var'], val, "numpy")
        values = np.asarray(f_val(coords), dtype=float) * np.ones_like(coords)
    else:
        values = np.full_like(coords, float(val))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"2D boundary value {val} is not finite on the edge")
    return values


def solve_fdm_2d(
    rhs_expr: Union[LinearPDE2D, sp.Expr],
    bcs_parsed: Dict[str, Dict[str, Any]],
    x_sym: sp.Symbol,
    y_sym: sp.Symbol,
    grid_size: int = 20,
    max_iter: int = 2000,
    tol: float = 1e-6,
    x_bounds: Tuple[float, float] = UNIT_BOUNDS,
    y_bounds: Tuple[float, float] = UNIT_BOUNDS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves a linear elliptic PDE on the rectangle x_bounds x y_bounds using the Finite
    Difference Method with a direct sparse solve. rhs_expr is either a LinearPDE2D
    (a*u_xx + b*u_yy + c*u_xy + d*u_x + e*u_y + g*u = f, see parse_pde_2d) or a bare source
    term f for u_xx + u_yy = f.
    Interior: central differences (5-point stencil, plus the 4 diagonal neighbours for u_xy).
    A first-derivative term switches to a one-sided upwind difference at nodes where the
    central one would violate the discrete maximum principle (cell Peclet number |d|*h/a > 2).
    Edges: Dirichlet or first-order one-sided Neumann.
    max_iter and tol are kept for backwards compatibility (the former Jacobi iteration) and
    are no longer used.
    """
    pde = _as_pde_2d(rhs_expr, x_sym, y_sym)
    nx = ny = grid_size
    x_vals = np.linspace(x_bounds[0], x_bounds[1], nx)
    y_vals = np.linspace(y_bounds[0], y_bounds[1], ny)
    hx = (x_bounds[1] - x_bounds[0]) / (nx - 1)
    hy = (y_bounds[1] - y_bounds[0]) / (ny - 1)

    # Evaluate the (normalised) coefficients and the RHS on the grid
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    coeffs, F = pde.evaluate(X, Y)
    zeros = np.zeros_like(X)
    A = coeffs['u_xx']
    B = coeffs['u_yy']
    C = coeffs.get('u_xy', zeros)
    Dx = coeffs.get('u_x', zeros)
    Dy = coeffs.get('u_y', zeros)
    E = coeffs.get('u', zeros)
    upwind_x = np.abs(Dx) * hx > 2.0 * A
    upwind_y = np.abs(Dy) * hy > 2.0 * B
    interior = np.zeros_like(X, dtype=bool)
    interior[1:-1, 1:-1] = True
    if np.any((upwind_x | upwind_y) & interior):
        print("[*] 2D FDM: advection dominates on this grid (cell Peclet number > 2); "
              "using upwind differences for the first-derivative terms there.")

    edge_values = {
        'left': _bc_values(bcs_parsed['left'], y_vals),
        'right': _bc_values(bcs_parsed['right'], y_vals),
        'bottom': _bc_values(bcs_parsed['bottom'], x_vals),
        'top': _bc_values(bcs_parsed['top'], x_vals),
    }
    edge_nodes = {
        'left': [(0, j) for j in range(ny)],
        'right': [(nx - 1, j) for j in range(ny)],
        'bottom': [(i, 0) for i in range(nx)],
        'top': [(i, ny - 1) for i in range(nx)],
    }

    # Corner nodes belong to two edges. Assign them in the order of the former Jacobi solver:
    # Dirichlet edges (left, right, bottom, top), then Neumann edges in the same order; the
    # last edge wins.
    owner = {}
    for kind in ('dirichlet', 'neumann'):
        for edge in EDGES_2D:
            if bcs_parsed[edge]['type'] == kind:
                for node in edge_nodes[edge]:
                    owner[node] = edge

    def idx(i, j):
        return i * ny + j

    rows, cols, data = [], [], []
    b = np.zeros(nx * ny)

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        data.append(v)

    cx = 1.0 / (hx * hx)
    cy = 1.0 / (hy * hy)
    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            edge = owner.get((i, j))
            if edge is None:
                # Interior: a (u[i+1,j] - 2u + u[i-1,j]) / hx^2 + b (u[i,j+1] - 2u + u[i,j-1]) / hy^2
                #   + c u_xy + d u_x + e u_y + g u = F
                a_ij, b_ij, d_ij, e_ij = A[i, j], B[i, j], Dx[i, j], Dy[i, j]
                diag = -2.0 * a_ij * cx - 2.0 * b_ij * cy + E[i, j]
                east, west = a_ij * cx, a_ij * cx
                north, south = b_ij * cy, b_ij * cy
                if not upwind_x[i, j]:
                    east += 0.5 * d_ij / hx
                    west -= 0.5 * d_ij / hx
                elif d_ij > 0.0:        # forward difference (u[i+1,j] - u) / hx
                    east += d_ij / hx
                    diag -= d_ij / hx
                else:                   # backward difference (u - u[i-1,j]) / hx
                    west -= d_ij / hx
                    diag += d_ij / hx
                if not upwind_y[i, j]:
                    north += 0.5 * e_ij / hy
                    south -= 0.5 * e_ij / hy
                elif e_ij > 0.0:
                    north += e_ij / hy
                    diag -= e_ij / hy
                else:
                    south -= e_ij / hy
                    diag += e_ij / hy
                add(k, k, diag)
                add(k, idx(i + 1, j), east)
                add(k, idx(i - 1, j), west)
                add(k, idx(i, j + 1), north)
                add(k, idx(i, j - 1), south)
                if C[i, j] != 0.0:
                    # u_xy ~ (u[i+1,j+1] - u[i+1,j-1] - u[i-1,j+1] + u[i-1,j-1]) / (4 hx hy)
                    cxy = C[i, j] / (4.0 * hx * hy)
                    add(k, idx(i + 1, j + 1), cxy)
                    add(k, idx(i - 1, j - 1), cxy)
                    add(k, idx(i + 1, j - 1), -cxy)
                    add(k, idx(i - 1, j + 1), -cxy)
                b[k] = F[i, j]
                continue

            g = edge_values[edge][j if edge in ('left', 'right') else i]
            if bcs_parsed[edge]['type'] == 'dirichlet':
                add(k, k, 1.0)
                b[k] = g
            elif edge == 'left':    # (u[1,j] - u[0,j]) / hx = g
                add(k, k, 1.0)
                add(k, idx(1, j), -1.0)
                b[k] = -hx * g
            elif edge == 'right':   # (u[-1,j] - u[-2,j]) / hx = g
                add(k, k, 1.0)
                add(k, idx(nx - 2, j), -1.0)
                b[k] = hx * g
            elif edge == 'bottom':  # (u[i,1] - u[i,0]) / hy = g
                add(k, k, 1.0)
                add(k, idx(i, 1), -1.0)
                b[k] = -hy * g
            else:                   # top: (u[i,-1] - u[i,-2]) / hy = g
                add(k, k, 1.0)
                add(k, idx(i, ny - 2), -1.0)
                b[k] = hy * g

    A = sparse.csr_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny))
    u = np.asarray(spsolve(A, b), dtype=float).reshape(nx, ny)
    if not np.all(np.isfinite(u)):
        raise ValueError("2D FDM system could not be solved (singular or ill-posed problem).")

    return x_vals, y_vals, u

def solve_pytorch_pinn_2d(
    rhs_expr: Union[LinearPDE2D, sp.Expr],
    bcs_parsed: Dict[str, Dict[str, Any]],
    x_sym: sp.Symbol,
    y_sym: sp.Symbol,
    epochs: int = 400,
    x_bounds: Tuple[float, float] = UNIT_BOUNDS,
    y_bounds: Tuple[float, float] = UNIT_BOUNDS
) -> Tuple[GenericPINN, List[float]]:
    """
    Trains a 2D PINN model on the rectangle x_bounds x y_bounds for a linear elliptic PDE:
    rhs_expr is a LinearPDE2D (a*u_xx + b*u_yy + c*u_xy + d*u_x + e*u_y + g*u = f, see
    parse_pde_2d) or a bare source term f for u_xx + u_yy = f. The PDE residual uses the
    coefficients normalised by LinearPDE2D.evaluate.
    """
    pde = _as_pde_2d(rhs_expr, x_sym, y_sym)
    torch.manual_seed(42)
    model = GenericPINN(input_dim=2, hidden_dim=32)

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])

    # Boundary points sets
    y_b = torch.linspace(y_min, y_max, 20).view(-1, 1)
    xy_left = torch.cat([torch.full_like(y_b, x_min), y_b], dim=1)
    xy_left.requires_grad = True

    xy_right = torch.cat([torch.full_like(y_b, x_max), y_b], dim=1)
    xy_right.requires_grad = True

    x_b = torch.linspace(x_min, x_max, 20).view(-1, 1)
    xy_bottom = torch.cat([x_b, torch.full_like(x_b, y_min)], dim=1)
    xy_bottom.requires_grad = True

    xy_top = torch.cat([x_b, torch.full_like(x_b, y_max)], dim=1)
    xy_top.requires_grad = True

    # Target values of the BCs on the boundary points (constant tensors; values may vary along the edge)
    y_b_np = y_b.view(-1).numpy().astype(float)
    x_b_np = x_b.view(-1).numpy().astype(float)
    bc_values_np = {
        edge: _bc_values(bcs_parsed[edge], y_b_np if edge in ('left', 'right') else x_b_np)
        for edge in EDGES_2D
    }
    bc_targets = {
        edge: torch.tensor(values, dtype=torch.float32).view(-1, 1)
        for edge, values in bc_values_np.items()
    }

    # Initialize bias of the output layer to the average Dirichlet boundary condition value
    dirichlet_vals = [float(np.mean(bc_values_np[edge])) for edge, bc in bcs_parsed.items() if bc['type'] == 'dirichlet']
    mean_val = sum(dirichlet_vals) / len(dirichlet_vals) if dirichlet_vals else 0.0
    with torch.no_grad():
        model.net[-1].bias.fill_(mean_val)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Sample interior collocation points (20x20 grid)
    x_pde_vals = np.linspace(x_min, x_max, 20)
    y_pde_vals = np.linspace(y_min, y_max, 20)
    X, Y = np.meshgrid(x_pde_vals, y_pde_vals, indexing='ij')
    pts_pde = np.stack([X.ravel(), Y.ravel()], axis=1)
    xy_pde = torch.tensor(pts_pde, dtype=torch.float32, requires_grad=True)

    # The source term does not depend on the network, so it is evaluated once with numpy on the
    # collocation points and kept as a constant tensor (no autograd graph, no torch printer).
    # The same holds for the coefficients of the operator; a coefficient that is 1 everywhere is
    # not multiplied (u_xx + u_yy = f is trained exactly as before the general operator).
    coeff_np, rhs_np = pde.evaluate(X, Y)
    rhs_val = torch.tensor(rhs_np.reshape(-1, 1), dtype=torch.float32)
    coeff_vals = {
        term: (None if np.all(values == 1.0) else torch.tensor(values.reshape(-1, 1), dtype=torch.float32))
        for term, values in coeff_np.items() if np.any(values != 0.0)
    }

    def _term(term, value):
        coeff = coeff_vals[term]
        return value if coeff is None else coeff * value

    boundary_sets = [
        ('left', xy_left, 0),
        ('right', xy_right, 0),
        ('bottom', xy_bottom, 1),
        ('top', xy_top, 1),
    ]

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1. PDE Loss: a*u_xx + b*u_yy [+ c*u_xy + d*u_x + e*u_y + g*u] - RHS = 0
        u = model(xy_pde)
        grads = torch.autograd.grad(u, xy_pde, torch.ones_like(u), create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]

        grads_x = torch.autograd.grad(u_x, xy_pde, torch.ones_like(u_x), create_graph=True)[0]
        u_xx = grads_x[:, 0:1]
        u_yy = torch.autograd.grad(u_y, xy_pde, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

        residual = _term('u_xx', u_xx) + _term('u_yy', u_yy)
        if 'u_xy' in coeff_vals:
            residual = residual + _term('u_xy', grads_x[:, 1:2])
        if 'u_x' in coeff_vals:
            residual = residual + _term('u_x', u_x)
        if 'u_y' in coeff_vals:
            residual = residual + _term('u_y', u_y)
        if 'u' in coeff_vals:
            residual = residual + _term('u', u)
        pde_loss = torch.mean((residual - rhs_val) ** 2)

        # 2. BC Loss (left, right, bottom, top edges)
        bc_loss = 0.0
        for edge, xy_edge, normal_dim in boundary_sets:
            u_e = model(xy_edge)
            if bcs_parsed[edge]['type'] == 'dirichlet':
                bc_loss += torch.mean((u_e - bc_targets[edge]) ** 2)
            else: # neumann: du/dx on left/right, du/dy on bottom/top
                grads_e = torch.autograd.grad(u_e, xy_edge, torch.ones_like(u_e), create_graph=True)[0]
                du_dn = grads_e[:, normal_dim:normal_dim + 1]
                bc_loss += torch.mean((du_dn - bc_targets[edge]) ** 2)

        total_loss = pde_loss + bc_loss * 5.0
        total_loss.backward()
        optimizer.step()

        loss_history.append(float(total_loss.item()))

    return model, loss_history
