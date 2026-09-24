import re
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import standard_transformations, convert_xor
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Union

from .pinn_model import GenericPINN


# ==========================================
# 2D PDE Helper Functions & Solvers
# ==========================================

EDGES_2D = ("left", "right", "bottom", "top")
UNIT_BOUNDS = (0.0, 1.0)


def parse_rhs_2d(rhs_str: str, x_name: str, y_name: str, params: Dict[str, float]) -> sp.Expr:
    x = sp.Symbol(x_name)
    y = sp.Symbol(y_name)
    local_ns = {x_name: x, y_name: y}
    for p_name, p_val in params.items():
        local_ns[p_name] = p_val
    expr = sp.parse_expr(rhs_str.strip(), local_dict=local_ns, transformations=(standard_transformations + (convert_xor,)))
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


def _eval_source_2d(rhs_expr: sp.Expr, x_sym: sp.Symbol, y_sym: sp.Symbol, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Evaluates the source term f(x, y) on the given points (numpy, no autograd needed)."""
    rhs_expr = sp.sympify(rhs_expr)
    unknown = rhs_expr.free_symbols - {x_sym, y_sym}
    if unknown or rhs_expr.atoms(AppliedUndef):
        raise ValueError(f"2D source term may only depend on {x_sym} and {y_sym}: {rhs_expr}")
    f_rhs = sp.lambdify((x_sym, y_sym), rhs_expr, "numpy")
    F = np.asarray(f_rhs(X, Y), dtype=float) * np.ones_like(X, dtype=float)
    if not np.all(np.isfinite(F)):
        raise ValueError(f"2D source term is not finite on the domain: {rhs_expr}")
    return F


def solve_fdm_2d(
    rhs_expr: sp.Expr,
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
    Solves u_xx + u_yy = RHS on the rectangle x_bounds x y_bounds using the Finite Difference
    Method (5-point stencil, first-order one-sided Neumann edges) with a direct sparse solve.
    max_iter and tol are kept for backwards compatibility (the former Jacobi iteration) and
    are no longer used.
    """
    nx = ny = grid_size
    x_vals = np.linspace(x_bounds[0], x_bounds[1], nx)
    y_vals = np.linspace(y_bounds[0], y_bounds[1], ny)
    hx = (x_bounds[1] - x_bounds[0]) / (nx - 1)
    hy = (y_bounds[1] - y_bounds[0]) / (ny - 1)

    # Evaluate RHS on the grid
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    F = _eval_source_2d(rhs_expr, x_sym, y_sym, X, Y)

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
                # Interior: (u[i+1,j] - 2u + u[i-1,j]) / hx^2 + (u[i,j+1] - 2u + u[i,j-1]) / hy^2 = F
                add(k, k, -2.0 * cx - 2.0 * cy)
                add(k, idx(i + 1, j), cx)
                add(k, idx(i - 1, j), cx)
                add(k, idx(i, j + 1), cy)
                add(k, idx(i, j - 1), cy)
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
    rhs_expr: sp.Expr,
    bcs_parsed: Dict[str, Dict[str, Any]],
    x_sym: sp.Symbol,
    y_sym: sp.Symbol,
    epochs: int = 400,
    x_bounds: Tuple[float, float] = UNIT_BOUNDS,
    y_bounds: Tuple[float, float] = UNIT_BOUNDS
) -> Tuple[GenericPINN, List[float]]:
    """
    Trains a 2D PINN model to solve u_xx + u_yy = RHS on the rectangle x_bounds x y_bounds.
    """
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
    rhs_val = torch.tensor(_eval_source_2d(rhs_expr, x_sym, y_sym, X, Y).reshape(-1, 1), dtype=torch.float32)

    boundary_sets = [
        ('left', xy_left, 0),
        ('right', xy_right, 0),
        ('bottom', xy_bottom, 1),
        ('top', xy_top, 1),
    ]

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
