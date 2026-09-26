import hashlib
import json
import math
import multiprocessing
import os
import pickle
import re
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, convert_xor
import numpy as np
import torch
import torch.optim as optim
from scipy.integrate import solve_bvp
from typing import Callable, Dict, List, Optional, Tuple, Any

from .parsing import escape_keywords, safe_symbol_name
from .pinn_model import GenericPINN


class SolverResultError(ValueError):
    """A 1D solver produced no usable solution (non-finite, complex, BCs not met, ...).
    Raised so that the dispatcher's fallback to the other solver runs."""


# Boundary values used in BC residual expressions: u(a), u'(a), u(b), u'(b).
_BC_SYMS = sp.symbols("Ya0 Ya1 Yb0 Yb1")

# Relative tolerance of the result checks (BC and ODE residuals relative to the solution scale).
_CHECK_RTOL = 1e-6


def _bc_residual_exprs(
    bcs_list: List[str],
    x: sp.Symbol,
    y_func: sp.Function,
    sym_dict: Dict[str, sp.Symbol],
    param_subs: Dict[sp.Symbol, float],
    domain_min: float,
    domain_max: float,
) -> List[sp.Expr]:
    """
    BC residuals lhs - rhs in the boundary values Ya0 = u(a), Ya1 = u'(a), Yb0 = u(b),
    Yb1 = u'(b) (each BC location is mapped to the nearer end of the domain), with the
    parameter values substituted.
    """
    Ya0, Ya1, Yb0, Yb1 = _BC_SYMS
    residuals = []
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
                local_bc_ns[safe_symbol_name(p_name)] = p_sym

            expr = sp.parse_expr(escape_keywords(p_str), local_dict=local_bc_ns, transformations=(standard_transformations + (convert_xor,)))
            return expr.subs(param_subs)

        residuals.append(convert_bc_part_to_sym(bc_lhs) - convert_bc_part_to_sym(bc_rhs))
    return residuals


# ==========================================
# Analytical (SymPy)
# ==========================================

class _NoRetry(SolverResultError):
    """Analytical failure that a second attempt with numeric coefficients cannot fix."""


def _numpy_values(expr: sp.Expr, x: sp.Symbol, grid: np.ndarray, what: str) -> np.ndarray:
    """expr evaluated on grid (numpy/scipy printer as in metrics.callables_from_expr); must be real and finite."""
    try:
        fn = sp.lambdify(x, expr, ["scipy", "numpy"])
        with np.errstate(all="ignore"):
            values = np.asarray(fn(grid)) * np.ones_like(grid)
    except Exception as exc:
        raise SolverResultError(f"{what} cannot be evaluated numerically ({type(exc).__name__}: {exc})")
    if np.iscomplexobj(values):
        raise SolverResultError(f"{what} is complex-valued")
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise SolverResultError(f"{what} is not finite on the domain")
    return values


def _verify_analytical(
    final_sol: sp.Expr,
    rhs_numeric: sp.Expr,
    bc_residuals: List[sp.Expr],
    x: sp.Symbol,
    y_func: sp.Function,
    domain_min: float,
    domain_max: float,
) -> None:
    """
    Checks a closed-form BVP solution before it is used: no unevaluated integrals, no
    free symbols besides x, real and finite on the domain (with the numpy printer the
    metrics use), ODE residual and BC residuals small relative to the solution scale.
    Raises SolverResultError otherwise.
    """
    if final_sol.has(sp.Integral):
        raise _NoRetry("closed form contains unevaluated integrals")
    free = final_sol.free_symbols - {x}
    if free:
        raise SolverResultError(f"closed form has unresolved symbols {sorted(str(s) for s in free)}")
    if final_sol.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise SolverResultError("closed form is not finite (zoo/nan: degenerate coefficient)")

    grid = np.linspace(domain_min, domain_max, 201)
    length = abs(domain_max - domain_min) or 1.0
    u = _numpy_values(final_sol, x, grid, "solution")
    du = _numpy_values(sp.diff(final_sol, x), x, grid, "solution derivative")
    d2u = _numpy_values(sp.diff(final_sol, x, 2), x, grid, "second derivative")

    # ODE residual u'' - rhs(x, u, u') relative to the size of the terms.
    Y0, Y1 = sp.symbols("Y0 Y1")
    rhs_Y = rhs_numeric.subs({y_func.diff(x): Y1}).subs({y_func: Y0})
    if rhs_Y.free_symbols - {x, Y0, Y1}:
        raise SolverResultError(f"equation has symbols without values {sorted(str(s) for s in rhs_Y.free_symbols - {x, Y0, Y1})}")
    try:
        f_rhs = sp.lambdify((x, Y0, Y1), rhs_Y, ["scipy", "numpy"])
        with np.errstate(all="ignore"):
            rhs_vals = np.asarray(f_rhs(grid, u, du), dtype=complex) * np.ones_like(grid)
    except Exception as exc:
        raise SolverResultError(f"equation cannot be evaluated on the solution ({type(exc).__name__}: {exc})")
    if not np.all(np.isfinite(rhs_vals)):
        raise SolverResultError("equation right-hand side is not finite on the solution")
    ode_res = float(np.max(np.abs(d2u - rhs_vals)))
    ode_scale = max(float(np.max(np.abs(d2u))), float(np.max(np.abs(rhs_vals))))
    if not ode_res <= _CHECK_RTOL * ode_scale + 1e-300:
        raise SolverResultError(f"ODE residual {ode_res:.3g} exceeds {_CHECK_RTOL:g} x {ode_scale:.3g}")

    # BC residuals, relative to the boundary terms and the solution scale.
    value_scale = max(float(np.max(np.abs(u))), float(np.max(np.abs(du))) * length)
    deriv_scale = max(float(np.max(np.abs(du))), float(np.max(np.abs(u))) / length)
    for res in bc_residuals:
        try:
            r = complex(sp.N(res))
        except Exception as exc:
            raise SolverResultError(f"boundary condition residual not evaluable ({exc})")
        terms = sp.Add.make_args(sp.expand(res))
        try:
            term_scale = sum(abs(complex(sp.N(t))) for t in terms)
        except Exception:
            term_scale = abs(r)
        scale = term_scale + value_scale + deriv_scale
        if not (np.isfinite(r.real) and np.isfinite(r.imag)) or abs(r) > _CHECK_RTOL * scale + 1e-300:
            raise SolverResultError(f"boundary condition not met (residual {abs(r):.3g}, scale {scale:.3g})")


def _floats_as_symbols(expr: sp.Expr) -> Tuple[sp.Expr, Dict[sp.Symbol, float]]:
    """
    expr with each float replaced by a positive placeholder symbol (times its sign); integer
    values up to 1000 become Integers, zeros vanish. Returns (expr, {placeholder: value}).
    Used after substituting the parameter values: degenerate combinations have already
    cancelled (log(G/G) -> 0), and dsolve still sees symbolic coefficients - it recurses
    endlessly on float coefficients (SymPy 1.14) and can hang on the exact rationals of
    arbitrary floats. The sign lets it pick sin/cos instead of complex exponentials.
    """
    replacements, values = {}, {}
    for f in expr.atoms(sp.Float):
        v = float(f)
        if v == 0.0:
            replacements[f] = sp.Integer(0)
        elif v == int(v) and abs(v) <= 1000:
            replacements[f] = sp.Integer(int(v))
        else:
            placeholder = sp.Dummy("c", positive=True)
            replacements[f] = placeholder if v > 0 else -placeholder
            values[placeholder] = sp.Float(abs(v))
    return expr.xreplace(replacements), values


def _solve_analytical_once(
    rhs: sp.Expr,
    bcs_list: List[str],
    x: sp.Symbol,
    y_func: sp.Function,
    sym_dict: Dict[str, sp.Symbol],
    param_subs: Dict[sp.Symbol, float],
    float_values: Optional[Dict[sp.Symbol, float]] = None,
) -> Tuple[sp.Expr, List[sp.Expr]]:
    """
    dsolve + integration constants from the BCs. Returns the solution with parameter values
    and the BC residuals with constants and parameter values substituted (for the checks).
    float_values maps placeholder symbols in rhs back to their numbers (after dsolve).
    """
    eq = sp.Eq(y_func.diff(x, 2), rhs)
    try:
        sol = sp.dsolve(eq, y_func)
    except NotImplementedError as exc:
        raise _NoRetry(f"dsolve: {exc}")
    if isinstance(sol, list):
        raise SolverResultError("dsolve returned several solution branches")
    sol_expr = sol.rhs

    # Integration constants are the symbols dsolve introduced (C1, C2, ...), i.e. those not
    # already in the equation. Parameters such as 'Cf' or 'C_p' are not constants. Sorted so
    # that the order passed to sp.solve does not depend on set iteration (PYTHONHASHSEED).
    constants = sorted(sol_expr.free_symbols - eq.free_symbols, key=lambda s: (len(s.name), s.name))
    known_symbols = set(sym_dict.values())
    for i, c in enumerate(constants):
        if c in known_symbols:
            # dsolve reused the name of a parameter that only appears in the BCs (e.g. 'C1')
            renamed = sp.Dummy(c.name)
            sol_expr = sol_expr.subs(c, renamed)
            constants[i] = renamed
    if float_values:
        sol_expr = sol_expr.xreplace(float_values)

    bc_residuals = []
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
                bc_local_ns[safe_symbol_name(p_name)] = p_sym

            return sp.parse_expr(escape_keywords(p_str), local_dict=bc_local_ns, transformations=(standard_transformations + (convert_xor,)))

        bc_residuals.append(evaluate_bc_part(bc_lhs) - evaluate_bc_part(bc_rhs))

    const_vals = sp.solve(bc_residuals, constants, dict=True) if constants else [{}]
    if not const_vals:
        raise SolverResultError("the boundary conditions give no solution for the integration constants")
    particular_sol = sol_expr.subs(const_vals[0])
    final_sol = particular_sol.subs(param_subs)
    checked_residuals = [sp.sympify(r).subs(const_vals[0]).subs(param_subs) for r in bc_residuals]
    return final_sol, checked_residuals


def _solve_analytical_impl(
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

    First the general solution with symbolic parameters (values substituted afterwards).
    If that result is not usable - degenerate parameter values such as log(G/G) = 0 give
    1/0 terms (zoo/nan), complex exponentials, BCs without a solution - the ODE is solved
    again with the parameter values substituted first, so that degenerate terms cancel
    before dsolve (see _floats_as_symbols). Every result is checked (finite and real on the domain,
    ODE and BC residuals); if no attempt passes, SolverResultError is raised so that the
    caller falls back to the SciPy BVP solver.
    """
    param_subs = {sym_dict[k]: v for k, v in params.items() if k in sym_dict}
    rhs_numeric = pde_rhs.subs(param_subs)

    attempts = [("symbolic parameters", pde_rhs, {})]
    if pde_rhs.free_symbols - {x}:
        attempts.append(("numeric parameters",) + _floats_as_symbols(rhs_numeric))

    errors = []
    for label, rhs, float_values in attempts:
        try:
            final_sol, bc_residuals = _solve_analytical_once(rhs, bcs_list, x, y_func, sym_dict, param_subs, float_values)
            _verify_analytical(final_sol, rhs_numeric, bc_residuals, x, y_func, domain_min, domain_max)
            deriv_val = complex(sp.N(final_sol.diff(x).subs(x, domain_min)))
            if deriv_val.imag != 0.0 or not np.isfinite(deriv_val.real):
                raise SolverResultError(f"wall derivative is not a finite real number ({deriv_val})")
            return final_sol, float(deriv_val.real)
        except _NoRetry as exc:
            errors.append(f"{label}: {exc}")
            break
        except SolverResultError as exc:
            errors.append(f"{label}: {exc}")
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise SolverResultError("analytical solution not usable (" + "; ".join(errors) + ")")

# ------------------------------------------
# Time budget and memo for the symbolic solve
# ------------------------------------------
# sp.dsolve / sp.solve can run for many minutes on innocent-looking problems (the recorded case: an
# exponential source with two Robin boundary conditions, ~20 min). SymPy cannot be interrupted from
# another thread, so the symbolic solve runs in a worker process that is terminated when the budget
# is exceeded; SymbolicTimeoutError (a SolverResultError) then lets the caller fall back to SciPy.

SYMBOLIC_TIMEOUT_ENV = "VECTORNAUT_SYMBOLIC_TIMEOUT_S"
DEFAULT_SYMBOLIC_TIMEOUT_S = 20.0
# Opt-in on-disk memo of analytical results (set by llm_replay to <session>/solver_memo).
SOLVER_MEMO_ENV = "VECTORNAUT_SOLVER_MEMO_DIR"


class SymbolicTimeoutError(SolverResultError):
    """The symbolic (SymPy) solve exceeded its time budget and was terminated."""


def symbolic_timeout_s() -> float:
    """Time budget of one symbolic solve in seconds (env VECTORNAUT_SYMBOLIC_TIMEOUT_S, default 20; <= 0: no limit)."""
    raw = os.environ.get(SYMBOLIC_TIMEOUT_ENV)
    if raw is None or not str(raw).strip():
        return DEFAULT_SYMBOLIC_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_SYMBOLIC_TIMEOUT_S
    return value if math.isfinite(value) and value > 0 else 0.0


def _analytical_worker(conn: Any, args: Tuple[Any, ...]) -> None:
    """Runs in the worker process: solves and sends ('ok', result) or ('err', exception)."""
    try:
        payload = ("ok", _solve_analytical_impl(*args))
    except BaseException as exc:  # sent back and re-raised in the parent
        payload = ("err", exc)
    try:
        conn.send(payload)
    except Exception as send_err:
        conn.send(("err", SolverResultError(f"analytical result could not be returned from the worker "
                                            f"({type(send_err).__name__}: {send_err})")))
    finally:
        conn.close()


def _mp_context():
    methods = multiprocessing.get_all_start_methods()
    # fork: the worker inherits the parsed expressions (nothing to pickle on the way in, no re-import).
    return multiprocessing.get_context("fork" if "fork" in methods else "spawn")


def _run_with_budget(args: Tuple[Any, ...], budget: float) -> Tuple[sp.Expr, float]:
    """_solve_analytical_impl(*args) in a worker process, terminated after ``budget`` seconds."""
    ctx = _mp_context()
    receiver, sender = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_analytical_worker, args=(sender, args), daemon=True)
    try:
        proc.start()
    except (AssertionError, OSError, ValueError) as exc:
        # e.g. inside a daemonic process, which may not have children: solve in-process, unbounded.
        receiver.close()
        sender.close()
        print(f"[*] Symbolic time budget unavailable ({exc}); solving in-process.")
        return _solve_analytical_impl(*args)
    sender.close()
    kind, value = None, None
    try:
        if receiver.poll(budget):
            kind, value = receiver.recv()
    except (EOFError, OSError):
        kind = "died"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(2.0)
            if proc.is_alive():
                proc.kill()
        proc.join()
        receiver.close()
    if kind is None:
        raise SymbolicTimeoutError(f"symbolic solve exceeded the time budget of {budget:g} s ({SYMBOLIC_TIMEOUT_ENV})")
    if kind == "died":
        raise SolverResultError(f"symbolic solve worker exited without a result (exit code {proc.exitcode})")
    if kind == "err":
        raise value
    return value


_CODE_VERSION: Optional[str] = None


def _code_version() -> str:
    """Hash of this module's source and the SymPy version: memo entries of other solver code are not reused."""
    global _CODE_VERSION
    if _CODE_VERSION is None:
        digest = hashlib.sha256(sp.__version__.encode("utf-8"))
        try:
            with open(__file__, "rb") as f:
                digest.update(f.read())
        except OSError:
            pass
        _CODE_VERSION = digest.hexdigest()[:16]
    return _CODE_VERSION


def solver_memo_dir() -> Optional[str]:
    """The memo folder (env VECTORNAUT_SOLVER_MEMO_DIR) or None when the memo is off (default)."""
    path = (os.environ.get(SOLVER_MEMO_ENV) or "").strip()
    return path or None


def _memo_key(kind: str, pde_rhs: sp.Expr, bcs_list: List[str], x: sp.Symbol, y_func: Any,
              sym_dict: Dict[str, sp.Symbol], params: Dict[str, float], domain_min: float, domain_max: float) -> str:
    def number(value: Any) -> str:
        try:
            return repr(float(value))
        except (TypeError, ValueError):
            return repr(value)

    parts = [kind, _code_version(), sp.srepr(pde_rhs), [str(b) for b in bcs_list], str(x), str(y_func),
             sorted((str(k), sp.srepr(v)) for k, v in sym_dict.items()),
             sorted((str(k), number(v)) for k, v in params.items()), number(domain_min), number(domain_max)]
    return hashlib.sha256(json.dumps(parts, ensure_ascii=True).encode("utf-8")).hexdigest()


def _memo_read(path: str) -> Optional[Tuple[str, Any]]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _memo_write(path: str, record: Tuple[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(record, f)
        os.replace(tmp, path)
    except Exception:
        pass


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
    Analytical solution of the 1D BVP (see ``_solve_analytical_impl``) under a time budget
    (``symbolic_timeout_s``, env VECTORNAUT_SYMBOLIC_TIMEOUT_S, default 20 s): the symbolic solve runs
    in a worker process that is terminated when the budget is exceeded, raising SymbolicTimeoutError
    (a SolverResultError, so the caller's SciPy fallback runs). With VECTORNAUT_SOLVER_MEMO_DIR set,
    results, deterministic failures and timeouts are memoised on disk, keyed by the equation, BCs,
    symbols, parameter values, domain, SymPy version and solver source; a memoised timeout is reused
    only while the budget is not larger than the one that timed out.
    """
    args = (pde_rhs, bcs_list, x, y_func, sym_dict, params, domain_min, domain_max)
    budget = symbolic_timeout_s()
    memo = solver_memo_dir()
    path = None
    if memo:
        path = os.path.join(memo, "analytical",
                            _memo_key("analytical", pde_rhs, bcs_list, x, y_func, sym_dict, params, domain_min,
                                      domain_max) + ".pkl")
        record = _memo_read(path)
        if record is not None:
            kind, value = record
            if kind == "ok":
                return value
            if kind == "err" and isinstance(value, BaseException):
                raise value
            if kind == "timeout" and budget > 0 and budget <= float(value):
                raise SymbolicTimeoutError(f"symbolic solve exceeded the time budget of {float(value):g} s "
                                           f"({SYMBOLIC_TIMEOUT_ENV}; memoised)")
    try:
        result = _run_with_budget(args, budget) if budget > 0 else _solve_analytical_impl(*args)
    except SymbolicTimeoutError:
        if path:
            _memo_write(path, ("timeout", budget))
        raise
    except SolverResultError as exc:
        if path:
            _memo_write(path, ("err", exc))
        raise
    if path:
        _memo_write(path, ("ok", result))
    return result


# ==========================================
# SciPy BVP
# ==========================================

def _linear_bc_parts(res: sp.Expr) -> Optional[Tuple[float, Dict[sp.Symbol, float]]]:
    """(constant, {boundary symbol: coefficient}) of a BC residual linear in the boundary values."""
    syms = [s for s in _BC_SYMS if s in res.free_symbols]
    if res.free_symbols - set(_BC_SYMS):
        return None
    coeffs = {}
    try:
        for s in syms:
            c = sp.diff(res, s)
            if c.free_symbols:
                return None
            coeffs[s] = float(c)
        const = float(res.subs({s: 0 for s in syms}))
    except Exception:
        return None
    if not np.isfinite(const) or not all(np.isfinite(v) for v in coeffs.values()):
        return None
    return const, coeffs


def _bvp_scaling(
    bc_residuals: List[sp.Expr],
    length: float,
    rhs_func: Callable,
    x_grid: np.ndarray,
) -> Tuple[float, float]:
    """
    Shift and scale of the dependent variable for the nondimensional BVP, u = shift + scale * v:
    shift = mean Dirichlet value; scale = the largest of the deviations from the shift each
    linear BC implies (|residual at u = shift, u' = 0| / (sum |value coefficients| + sum
    |derivative coefficients| / length); i.e. half the Dirichlet range, |Neumann value| * length,
    Robin terms) and the forcing scale max|u''| * length**2 / 8 at u = shift, u' = 0. No floor:
    a 1e-8 m displacement is solved in units of ~1e-8 m. Falls back to max(|shift|, 1).
    """
    Ya0, Ya1, Yb0, Yb1 = _BC_SYMS
    value_syms, deriv_syms = (Ya0, Yb0), (Ya1, Yb1)
    linear = [p for p in (_linear_bc_parts(r) for r in bc_residuals) if p is not None]
    dirichlet = []
    for const, coeffs in linear:
        if len(coeffs) == 1:
            (sym, coeff), = coeffs.items()
            if sym in value_syms and coeff != 0.0:
                dirichlet.append(-const / coeff)
    shift = float(np.mean(dirichlet)) if dirichlet else 0.0

    candidates = []
    for const, coeffs in linear:
        denom = sum(abs(c) for s, c in coeffs.items() if s in value_syms) + \
            sum(abs(c) for s, c in coeffs.items() if s in deriv_syms) / length
        if denom > 0.0:
            at_shift = const + sum(c * shift for s, c in coeffs.items() if s in value_syms)
            candidates.append(abs(at_shift) / denom)
    try:
        with np.errstate(all="ignore"):
            forcing = np.abs(np.asarray(rhs_func(x_grid, shift, 0.0), dtype=float)) * np.ones_like(x_grid)
        if forcing.size and np.all(np.isfinite(forcing)):
            candidates.append(float(np.max(forcing)) * length ** 2 / 8.0)
    except Exception:
        pass
    candidates = [c for c in candidates if np.isfinite(c) and c > 0.0]
    scale = max(candidates) if candidates else max(abs(shift), 1.0)
    return shift, scale


class BVPSolution:
    """
    u(x) of a solve_bvp result in the original variables (the solve itself runs in
    nondimensional t = (x - x0) / length, u = shift + scale * v). Callable like the former
    interp1d result; bvp_sol(x) returns [u, du/dx] from the BVP's own C1 spline.
    """

    def __init__(self, sol: Any, x0: float, length: float, shift: float, scale: float):
        self._sol, self._x0, self._length, self._shift, self._scale = sol, x0, length, shift, scale

    def bvp_sol(self, x: Any) -> np.ndarray:
        t = (np.asarray(x, dtype=float) - self._x0) / self._length
        v = self._sol(t)
        return np.array([self._shift + self._scale * v[0], self._scale / self._length * v[1]])

    def __call__(self, x: Any) -> Any:
        return self.bvp_sol(x)[0]


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
    Returns a callable u(x) (BVPSolution, with bvp_sol(x) -> [u, u']) and the derivative at the wall.

    The problem is solved in nondimensional form: t = (x - domain_min) / L and
    u = shift + scale * v with shift/scale estimated from the BCs and the forcing
    (_bvp_scaling), so that e.g. displacements of 1e-8 m are not lost below the absolute
    BC tolerance. BC residuals are normalised by their sensitivity to v. The solve uses
    tol = 1e-8 (1e-6 if the mesh limit is reached), is repeated once with the scale of the
    first solution if the estimate was off by more than 10x, and the BC residuals are
    checked afterwards; any failure raises SolverResultError.
    """
    param_subs = {sym_dict[k]: v for k, v in params.items() if k in sym_dict}
    rhs_substituted = pde_rhs.subs(param_subs)

    Y0 = sp.Symbol('Y0')
    Y1 = sp.Symbol('Y1')
    expr_for_Y = rhs_substituted.subs({y_func: Y0, y_func.diff(x): Y1})
    f_rhs = sp.lambdify((x, Y0, Y1), expr_for_Y, 'numpy')

    bc_residuals = _bc_residual_exprs(bcs_list, x, y_func, sym_dict, param_subs, domain_min, domain_max)
    bc_funcs = [sp.lambdify(_BC_SYMS, res, 'numpy') for res in bc_residuals]
    bc_grads = [[sp.lambdify(_BC_SYMS, sp.diff(res, s), 'numpy') for s in _BC_SYMS] for res in bc_residuals]

    x0 = float(domain_min)
    length = float(domain_max - domain_min)
    if not length > 0.0:
        raise SolverResultError(f"empty domain [{domain_min}, {domain_max}]")
    x_grid = np.linspace(domain_min, domain_max, 100)
    shift, scale = _bvp_scaling(bc_residuals, length, f_rhs, x_grid)

    def solve(shift: float, scale: float, v_guess: np.ndarray, t_grid: np.ndarray):
        # Normalisation of each BC residual: its change per unit change of v (value) or
        # v_t (derivative), evaluated at the guess; residuals are then in units of v.
        ua, ub = shift + scale * v_guess[0, 0], shift + scale * v_guess[0, -1]
        dua, dub = scale / length * v_guess[1, 0], scale / length * v_guess[1, -1]
        norms = []
        for grads in bc_grads:
            try:
                g = [abs(float(fn(ua, dua, ub, dub))) for fn in grads]
            except Exception:
                g = [float("nan")] * 4
            norm = (g[0] + g[2]) * scale + (g[1] + g[3]) * scale / length
            norms.append(norm if np.isfinite(norm) and norm > 0.0 else scale)

        def fun(t, V):
            xs = x0 + length * t
            u, du = shift + scale * V[0], scale / length * V[1]
            with np.errstate(all="ignore"):
                d2u = f_rhs(xs, u, du)
            return np.vstack([V[1], np.ones_like(t) * d2u * length ** 2 / scale])

        def bc(Va, Vb):
            ya = (shift + scale * Va[0], scale / length * Va[1])
            yb = (shift + scale * Vb[0], scale / length * Vb[1])
            return np.array([float(f(ya[0], ya[1], yb[0], yb[1])) / n for f, n in zip(bc_funcs, norms)], dtype=float)

        res = None
        for tol in (1e-8, 1e-6):
            res = solve_bvp(fun, bc, t_grid, v_guess, tol=tol, bc_tol=tol, max_nodes=100000)
            if res.success or res.status != 1:  # status 1: mesh limit reached, retry with a looser tol
                break
        return res, norms

    # Initial guess: linear between the Dirichlet values if both ends have one, else u = shift.
    t_grid = np.linspace(0.0, 1.0, x_grid.size)
    ends = {}
    for res_expr in bc_residuals:
        parts = _linear_bc_parts(res_expr)
        if parts and len(parts[1]) == 1:
            (sym, coeff), = parts[1].items()
            if sym in (_BC_SYMS[0], _BC_SYMS[2]) and coeff != 0.0:
                ends[sym] = -parts[0] / coeff
    u_guess = np.full(t_grid.size, shift)
    if len(ends) == 2:
        u_guess = ends[_BC_SYMS[0]] + (ends[_BC_SYMS[2]] - ends[_BC_SYMS[0]]) * t_grid
    V_guess = np.vstack([(u_guess - shift) / scale, np.gradient(u_guess, t_grid) / scale])

    res, norms = solve(shift, scale, V_guess, t_grid)
    if res.success:
        # Re-solve once in the scale of the solution if the estimate was off by > 10x.
        actual = float(np.max(np.abs(scale * res.y[0])))
        if actual > 0.0 and np.isfinite(actual) and not (0.1 <= actual / scale <= 10.0):
            u_first = shift + scale * res.y[0]
            du_first = scale * res.y[1]
            new_scale = actual
            res2, norms2 = solve(shift, new_scale, np.vstack([(u_first - shift) / new_scale, du_first / new_scale]), res.x)
            if res2.success:
                res, norms, scale = res2, norms2, new_scale
    if not res.success:
        raise SolverResultError(f"SciPy BVP solver failed to converge: {res.message}")

    solution = BVPSolution(res.sol, x0, length, shift, scale)

    # Checks in the original variables: finite, BCs met to _CHECK_RTOL of the solution scale.
    u_nodes, du_nodes = solution.bvp_sol(x0 + length * res.x)
    if not (np.all(np.isfinite(u_nodes)) and np.all(np.isfinite(du_nodes))):
        raise SolverResultError("SciPy BVP solution is not finite")
    ua, dua = solution.bvp_sol(domain_min)
    ub, dub = solution.bvp_sol(domain_max)
    for f, norm, bc_str in zip(bc_funcs, norms, bcs_list):
        residual = abs(float(f(ua, dua, ub, dub))) / norm * scale
        if not residual <= _CHECK_RTOL * max(scale, float(np.max(np.abs(u_nodes - shift)))):
            raise SolverResultError(f"SciPy BVP solution does not meet '{bc_str}' (residual {residual:.3g} in units of u)")

    # Derivative at domain_min from the BVP's own C1 spline (the second component of the
    # first-order system). A fixed finite-difference step would span the whole domain for thin films.
    deriv_val = float(solution.bvp_sol(domain_min)[1])
    return solution, deriv_val

def _pinn_output_scaling(
    bc_residuals: List[sp.Expr],
    value_syms: Tuple[sp.Symbol, sp.Symbol],
    deriv_syms: Tuple[sp.Symbol, sp.Symbol],
    length: float,
    rhs_func: Any = None,
    x_grid: Any = None
) -> Tuple[float, float]:
    """
    Output shift and scale for the 1D PINN from the linear single-point BCs:
    shift = mean Dirichlet value, scale = max(1, half the Dirichlet range, |Neumann value| * length,
    max|u''| * length**2 / 8). The last term is the Poiseuille-type forcing scale, with u'' taken
    from rhs_func(x_grid, u=shift, u'=0). The floor of 1 keeps the unscaled network for fields
    of order one or smaller.
    """
    dirichlet_vals, neumann_vals = [], []
    for res in bc_residuals:
        syms = res.free_symbols
        if len(syms) != 1:
            continue
        sym = next(iter(syms))
        try:
            coeff = res.diff(sym)
            if coeff.free_symbols or coeff == 0:
                continue  # nonlinear in the boundary value
            val = float(-res.subs(sym, 0) / coeff)
        except Exception:
            continue
        if not np.isfinite(val):
            continue
        if sym in value_syms:
            dirichlet_vals.append(val)
        elif sym in deriv_syms:
            neumann_vals.append(val)
    shift = float(np.mean(dirichlet_vals)) if dirichlet_vals else 0.0
    candidates = [1.0]
    if dirichlet_vals:
        candidates.append(0.5 * (max(dirichlet_vals) - min(dirichlet_vals)))
    candidates += [abs(g) * abs(length) for g in neumann_vals]
    if rhs_func is not None and x_grid is not None:
        try:
            with np.errstate(all="ignore"):
                forcing = np.abs(np.asarray(rhs_func(x_grid, shift, 0.0), dtype=float))
            if forcing.size and np.all(np.isfinite(forcing)):
                candidates.append(float(np.max(forcing)) * length ** 2 / 8.0)
        except Exception:
            pass
    scale = max(c for c in candidates if np.isfinite(c))
    return shift, scale

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
    # evalf folds numeric sub-expressions (pi**2, exp(-2)) into floats; otherwise the torch
    # printer emits pow(float, int) / exp(int), which torch.pow / torch.exp reject.
    expr_for_Y = rhs_substituted.subs({y_func: Y0, y_func.diff(x): Y1}).evalf()
    
    f_pde_rhs = sp.lambdify((x, Y0, Y1), expr_for_Y, 'torch')
    
    Ya0, Ya1, Yb0, Yb1 = _BC_SYMS
    bc_residuals = [
        sp.sympify(res).evalf()
        for res in _bc_residual_exprs(bcs_list, x, y_func, sym_dict, param_subs, domain_min, domain_max)
    ]

    bc_funcs_pytorch = [sp.lambdify((Ya0, Ya1, Yb0, Yb1), res, 'torch') for res in bc_residuals]
    
    # Scale the network output to the boundary values and the forcing (the 1D analogue of the
    # 2D PINN's output-bias initialisation), so that e.g. a 260-310 K field is learned in O(1) units.
    output_shift, output_scale = _pinn_output_scaling(
        bc_residuals, (Ya0, Yb0), (Ya1, Yb1), domain_max - domain_min,
        rhs_func=sp.lambdify((x, Y0, Y1), expr_for_Y, 'numpy'),
        x_grid=np.linspace(domain_min, domain_max, 100),
    )
    if output_shift != 0.0 or output_scale != 1.0:
        print(f"[*] 1D PINN output scaling: u = {output_shift:.6g} + {output_scale:.6g} * net({x.name})")
        model.set_output_scaling(output_shift, output_scale)
    
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
