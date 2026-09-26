"""
Declarative solver metrics and the baseline performance gain.

The auditor may describe the quantity a design is about with a MetricSpec
(config.py): kind in METRIC_KINDS, an optional location, an optional scale expression
in the parameters and a unit. The functions here evaluate that spec on a 1D solution
(given as callables for u and du/dx) or on a 2D field sampled on a rectangular grid,
and compute the relative improvement of the metric over a baseline design.

Without a spec the dispatcher keeps its historical metrics (du/dx at the lower
boundary in 1D, the mean of the sampled field in 2D).

An optional spec.transform (a SymPy expression in m, the scaled metric, and parameter
names) turns the metric into a nonlinear figure of merit before the gain is computed,
e.g. 'sqrt(2*adhesion_energy/m)' for a detachment stress from a compliance. It is parsed
with a whitelist of names and functions (parse_metric_transform), never with eval on
arbitrary text.
"""
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, convert_xor

from .parsing import escape_keywords, safe_symbol_name


METRIC_KINDS = ("value_at", "derivative_at", "max", "min", "max_abs", "mean", "integral")
_LOCATED_KINDS = ("value_at", "derivative_at")
_KIND_ALIASES = {
    "value": "value_at",
    "derivative": "derivative_at",
    "gradient_at": "derivative_at",
    "flux_at": "derivative_at",
    "maximum": "max",
    "minimum": "min",
    "abs_max": "max_abs",
    "maxabs": "max_abs",
    "average": "mean",
    "avg": "mean",
    "integral_over_domain": "integral",
}

# gain_basis values reported in SimulatorOutput.gain_basis
GAIN_BASIS_BASELINE = "baseline_parameters"
GAIN_BASIS_BIONIC = "bionic_effect"
GAIN_BASIS_NONE = "none"

# Samples used for max/min searches and line integrals.
_FINE_POINTS_1D = 2001
_FINE_POINTS_LINE = 401
_REFINE_2D = 4


class MetricSpecError(ValueError):
    """The metric spec cannot be evaluated for this problem."""


def _as_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(getattr(value, "__dict__", {}))


def resolve_metric_spec(auditor_output: Any) -> Optional[Dict[str, Any]]:
    """
    Returns the auditor's metric spec as a normalised dict
    {kind, location, scale, unit, label}, or None if no (usable) spec was given.
    Raises MetricSpecError for an unknown kind.
    """
    spec = _as_plain_dict(getattr(auditor_output, "metric", None))
    kind = str(spec.get("kind") or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not kind:
        return None
    kind = _KIND_ALIASES.get(kind, kind)
    if kind not in METRIC_KINDS:
        raise MetricSpecError(f"unknown metric kind {spec.get('kind')!r} (expected one of {', '.join(METRIC_KINDS)})")

    def _text(name: str) -> Optional[str]:
        raw = spec.get(name)
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    return {
        "kind": kind,
        "location": _text("location"),
        "scale": _text("scale"),
        "unit": _text("unit"),
        "label": _text("label"),
        "transform": _text("transform"),
    }


def eval_param_expr(expr_str: str, params: Dict[str, float], extra: Optional[Dict[str, float]] = None) -> float:
    """Evaluates a numeric expression in the parameter names (e.g. 'k/L', '2*h', '0.5')."""
    local_ns: Dict[str, Any] = {safe_symbol_name(name): value for name, value in params.items()}
    if extra:
        local_ns.update({safe_symbol_name(name): value for name, value in extra.items()})
    try:
        expr = sp.parse_expr(
            escape_keywords(str(expr_str).strip()),
            local_dict=local_ns,
            transformations=(standard_transformations + (convert_xor,)),
        )
        value = complex(sp.sympify(expr).evalf())
    except Exception as err:
        raise MetricSpecError(f"cannot evaluate {expr_str!r}: {err}") from err
    if abs(value.imag) > 1e-12 * max(1.0, abs(value.real)) or not math.isfinite(value.real):
        raise MetricSpecError(f"{expr_str!r} is not a finite real number")
    return float(value.real)


def metric_scale(spec: Dict[str, Any], params: Dict[str, float]) -> float:
    if not spec.get("scale"):
        return 1.0
    return eval_param_expr(spec["scale"], params)


# ==========================================
# Metric transform (nonlinear figure of merit)
# ==========================================

# Name of the raw (scaled) metric in a transform expression. It wins over a parameter of the same name.
TRANSFORM_METRIC_SYMBOL = "m"
_TRANSFORM_MAX_LENGTH = 300
_TRANSFORM_FUNCTIONS = {
    "sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "abs": sp.Abs, "Abs": sp.Abs, "min": sp.Min, "Min": sp.Min, "max": sp.Max, "Max": sp.Max,
}
_TRANSFORM_CONSTANTS = {"pi": sp.pi}
# Numbers, identifiers, whitespace, arithmetic operators, parentheses and commas only.
_TRANSFORM_ALLOWED = re.compile(r"^[A-Za-z0-9_\s+\-*/^().,]*$")
_NUMBER_LITERAL = re.compile(r"(?<![A-Za-z0-9_.])(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?![A-Za-z0-9_.])")
# The only globals sympy's parser needs for the code it generates from numbers and symbols.
_TRANSFORM_GLOBALS = {
    "__builtins__": {},
    "Symbol": sp.Symbol, "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
}


def parse_metric_transform(transform: str, param_names: List[str]) -> sp.Expr:
    """
    Parses a metric transform such as 'sqrt(2*adhesion_energy/m)' into a SymPy expression.
    Only numbers, m, the given parameter names, + - * / ** ^, parentheses, commas and the
    functions in _TRANSFORM_FUNCTIONS are allowed; attribute access, dunder names, strings,
    indexing and unknown names are rejected before anything is parsed. Raises MetricSpecError.
    """
    text = str(transform or "").strip()
    if not text:
        raise MetricSpecError("metric transform is empty")
    if len(text) > _TRANSFORM_MAX_LENGTH:
        raise MetricSpecError(f"metric transform is longer than {_TRANSFORM_MAX_LENGTH} characters")
    if not _TRANSFORM_ALLOWED.match(text):
        bad = sorted({ch for ch in text if not _TRANSFORM_ALLOWED.match(ch)})
        raise MetricSpecError(f"metric transform {text!r} contains characters that are not allowed: {''.join(bad)!r}")
    if "__" in text:
        raise MetricSpecError(f"metric transform {text!r} contains '__'")
    without_numbers = _NUMBER_LITERAL.sub(" 0 ", text)
    if "." in without_numbers:
        raise MetricSpecError(f"metric transform {text!r}: attribute access ('.') is not allowed")
    params = {str(name) for name in param_names}
    allowed = {TRANSFORM_METRIC_SYMBOL} | params | set(_TRANSFORM_FUNCTIONS) | set(_TRANSFORM_CONSTANTS)
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", without_numbers)
    unknown = sorted({name for name in identifiers if name not in allowed})
    if unknown:
        raise MetricSpecError(
            f"metric transform {text!r} uses unknown names {', '.join(unknown)} "
            f"(allowed: m, parameter names, {', '.join(sorted(_TRANSFORM_FUNCTIONS))}, pi)"
        )
    if TRANSFORM_METRIC_SYMBOL not in identifiers:
        raise MetricSpecError(f"metric transform {text!r} does not depend on the metric m")

    local_ns: Dict[str, Any] = {safe_symbol_name(name): sp.Symbol(name) for name in params}
    local_ns.update(_TRANSFORM_FUNCTIONS)
    local_ns.update(_TRANSFORM_CONSTANTS)
    local_ns[TRANSFORM_METRIC_SYMBOL] = sp.Symbol(TRANSFORM_METRIC_SYMBOL)
    try:
        expr = sp.parse_expr(
            escape_keywords(text),
            local_dict=local_ns,
            global_dict=dict(_TRANSFORM_GLOBALS),
            transformations=(standard_transformations + (convert_xor,)),
        )
    except Exception as err:
        raise MetricSpecError(f"cannot parse metric transform {text!r}: {err}") from err
    if not isinstance(expr, sp.Expr):
        raise MetricSpecError(f"metric transform {text!r} is not a numeric expression")
    allowed_symbols = {TRANSFORM_METRIC_SYMBOL} | params
    stray = sorted(str(sym) for sym in expr.free_symbols if str(sym) not in allowed_symbols)
    if stray:
        raise MetricSpecError(f"metric transform {text!r} uses unknown symbols {', '.join(stray)}")
    return expr


def apply_metric_transform(transform: Optional[str], value: float, params: Dict[str, float]) -> float:
    """
    The figure of merit transform(m=value, parameters); without a transform the value itself.
    Raises MetricSpecError if the transform is invalid or its result is not a finite real
    number (e.g. sqrt of a negative metric, division by a zero metric).
    """
    if not transform:
        return float(value)
    expr = parse_metric_transform(transform, list(params))
    subs = {sp.Symbol(TRANSFORM_METRIC_SYMBOL): float(value)}
    for sym in expr.free_symbols:
        name = str(sym)
        if name != TRANSFORM_METRIC_SYMBOL:
            subs[sym] = float(params[name])
    try:
        result = complex(expr.subs(subs).evalf())
    except Exception as err:
        raise MetricSpecError(f"metric transform {transform!r} cannot be evaluated at m={value:g}: {err}") from err
    if (not (math.isfinite(result.real) and math.isfinite(result.imag))
            or abs(result.imag) > 1e-12 * max(1.0, abs(result.real))):
        raise MetricSpecError(f"metric transform {transform!r} is not a finite real number at m={value:g}")
    return float(result.real)


# ==========================================
# 1D
# ==========================================

def callables_from_expr(expr: sp.Expr, x_sym: sp.Symbol) -> Tuple[Callable, Callable]:
    """(u, du/dx) as vectorised numpy callables from a SymPy solution."""
    f = sp.lambdify(x_sym, expr, ["scipy", "numpy"])
    df = sp.lambdify(x_sym, sp.diff(expr, x_sym), ["scipy", "numpy"])

    def _vec(fn):
        def wrapped(x):
            x_arr = np.asarray(x, dtype=float)
            return np.asarray(fn(x_arr), dtype=float) * np.ones_like(x_arr)
        return wrapped
    return _vec(f), _vec(df)


def callables_from_bvp(f_interp: Any) -> Tuple[Callable, Optional[Callable]]:
    """(u, du/dx) from solve_scipy_bvp's result (du/dx from the BVP's own C1 spline if attached)."""
    sol = getattr(f_interp, "bvp_sol", None)
    if sol is None:
        return (lambda x: np.asarray(f_interp(np.asarray(x, dtype=float)), dtype=float)), None
    return (lambda x: np.asarray(sol(np.asarray(x, dtype=float))[0], dtype=float),
            lambda x: np.asarray(sol(np.asarray(x, dtype=float))[1], dtype=float))


def callables_from_pinn(model: Any) -> Tuple[Callable, Callable]:
    """(u, du/dx) of a 1D PINN; du/dx via autograd."""
    import torch

    def f(x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        with torch.no_grad():
            out = model(torch.tensor(x_arr, dtype=torch.float32).view(-1, 1)).view(-1).numpy().astype(float)
        return out.reshape(np.shape(x)) if np.ndim(x) else out[0]

    def df(x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        xt = torch.tensor(x_arr, dtype=torch.float32).view(-1, 1).requires_grad_(True)
        u = model(xt)
        grad = torch.autograd.grad(u, xt, torch.ones_like(u))[0].view(-1).detach().numpy().astype(float)
        return grad.reshape(np.shape(x)) if np.ndim(x) else grad[0]
    return f, df


def _fd_derivative(f: Callable, x0: float, lo: float, hi: float) -> float:
    """Second-order finite difference of f at x0 (one-sided 3-point stencil at the domain ends)."""
    h = 1e-4 * (hi - lo)
    if x0 - h < lo:
        return float((-3.0 * f(x0) + 4.0 * f(x0 + h) - f(x0 + 2.0 * h)) / (2.0 * h))
    if x0 + h > hi:
        return float((3.0 * f(x0) - 4.0 * f(x0 - h) + f(x0 - 2.0 * h)) / (2.0 * h))
    return float((f(x0 + h) - f(x0 - h)) / (2.0 * h))


def _location_1d(location: Optional[str], params: Dict[str, float], lo: float, hi: float) -> float:
    if location is None:
        raise MetricSpecError("value_at/derivative_at needs a location")
    text = str(location).strip()
    # Accept "x=1", "at x = 1" and similar: keep the part after the last '='.
    if "=" in text:
        text = text.rsplit("=", 1)[1].strip()
    lowered = text.lower()
    if lowered in ("min", "domain_min", "start", "lower", "left", "wall"):
        return lo
    if lowered in ("max", "domain_max", "end", "upper", "right"):
        return hi
    x0 = eval_param_expr(text, params, extra={"domain_min": lo, "domain_max": hi})
    tol = 1e-9 * max(1.0, abs(hi - lo))
    if x0 < lo - tol or x0 > hi + tol:
        raise MetricSpecError(f"location {location!r} = {x0:g} is outside the domain [{lo:g}, {hi:g}]")
    return min(max(x0, lo), hi)


def _refined_extremum(g: Callable, lo: float, hi: float, find_max: bool) -> float:
    """max (or min) of g on [lo, hi]: dense sampling, then a bounded refinement around the best sample."""
    xs = np.linspace(lo, hi, _FINE_POINTS_1D)
    values = np.asarray(g(xs), dtype=float) * np.ones_like(xs)
    if not np.all(np.isfinite(values)):
        raise MetricSpecError("solution is not finite on the domain")
    idx = int(np.argmax(values) if find_max else np.argmin(values))
    best = float(values[idx])
    a, b = xs[max(idx - 1, 0)], xs[min(idx + 1, len(xs) - 1)]
    if b > a:
        try:
            from scipy.optimize import minimize_scalar
            sign = -1.0 if find_max else 1.0
            res = minimize_scalar(lambda t: sign * float(g(t)), bounds=(a, b), method="bounded",
                                  options={"xatol": 1e-12 * max(1.0, hi - lo)})
            if res.success and math.isfinite(res.fun):
                candidate = sign * float(res.fun)
                best = max(best, candidate) if find_max else min(best, candidate)
        except Exception:
            pass
    return best


def metric_1d(
    spec: Dict[str, Any],
    f: Callable,
    df: Optional[Callable],
    domain_min: float,
    domain_max: float,
    params: Dict[str, float],
) -> float:
    """Evaluates a metric spec on a 1D solution u = f(x) (df = du/dx or None for finite differences)."""
    kind = spec["kind"]
    lo, hi = float(domain_min), float(domain_max)
    scale = metric_scale(spec, params)

    if kind == "value_at":
        value = float(f(_location_1d(spec.get("location"), params, lo, hi)))
    elif kind == "derivative_at":
        x0 = _location_1d(spec.get("location"), params, lo, hi)
        value = float(df(x0)) if df is not None else _fd_derivative(f, x0, lo, hi)
    elif kind in ("max", "min", "max_abs"):
        # Extremum of the scaled field (a negative scale swaps max and min).
        g = lambda x: scale * np.asarray(f(x), dtype=float)
        if kind == "max":
            return _refined_extremum(g, lo, hi, find_max=True)
        if kind == "min":
            return _refined_extremum(g, lo, hi, find_max=False)
        return max(abs(_refined_extremum(g, lo, hi, True)), abs(_refined_extremum(g, lo, hi, False)))
    elif kind in ("mean", "integral"):
        from scipy.integrate import quad
        integral, _ = quad(lambda t: float(f(t)), lo, hi, limit=200)
        value = integral / (hi - lo) if kind == "mean" else integral
    else:  # pragma: no cover - resolve_metric_spec rejects unknown kinds
        raise MetricSpecError(f"unknown metric kind {kind!r}")

    result = scale * value
    if not math.isfinite(result):
        raise MetricSpecError(f"metric {kind} is not finite")
    return float(result)


# ==========================================
# 2D
# ==========================================

_NUMBER_PAIR = re.compile(r"^\(?\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)?$")


def _location_2d(
    location: Optional[str], x_name: str, y_name: str, params: Dict[str, float],
    x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]
) -> Optional[Tuple[str, Any]]:
    """
    Parses a 2D location: None (whole domain), ('line_x', x0) for 'x=x0', ('line_y', y0)
    for 'y=y0', or ('point', (x0, y0)) for '(x0, y0)'.
    """
    if location is None:
        return None
    text = str(location).strip()
    if not text or text.lower() in ("domain", "all", "whole", "whole_domain"):
        return None
    extra = {"x_min": x_bounds[0], "x_max": x_bounds[1], "y_min": y_bounds[0], "y_max": y_bounds[1]}
    edge_names = {
        "left": ("line_x", x_bounds[0]), "right": ("line_x", x_bounds[1]),
        "bottom": ("line_y", y_bounds[0]), "top": ("line_y", y_bounds[1]),
    }
    if text.lower() in edge_names:
        return edge_names[text.lower()]
    if "=" in text:
        var, value = [part.strip() for part in text.split("=", 1)]
        coord = eval_param_expr(value, params, extra=extra)
        if var in (x_name, "x"):
            return "line_x", coord
        if var in (y_name, "y"):
            return "line_y", coord
        raise MetricSpecError(f"location {location!r}: expected '{x_name}=...' or '{y_name}=...'")
    match = _NUMBER_PAIR.match(text)
    if match:
        return "point", (eval_param_expr(match.group(1), params, extra=extra),
                         eval_param_expr(match.group(2), params, extra=extra))
    raise MetricSpecError(f"cannot parse 2D location {location!r} (use '{x_name}=1', '{y_name}=0' or '(0.5, 0.5)')")


def _check_inside(value: float, bounds: Tuple[float, float], what: str) -> float:
    lo, hi = bounds
    tol = 1e-9 * max(1.0, abs(hi - lo))
    if value < lo - tol or value > hi + tol:
        raise MetricSpecError(f"{what} = {value:g} is outside the domain [{lo:g}, {hi:g}]")
    return min(max(value, lo), hi)


def metric_2d(
    spec: Dict[str, Any],
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    field: np.ndarray,
    params: Dict[str, float],
    x_name: str = "x",
    y_name: str = "y",
) -> float:
    """
    Evaluates a metric spec on a 2D field sampled on the grid x_vals x y_vals
    (field[i, j] = u(x_vals[i], y_vals[j])). Values and derivatives between nodes come
    from the interpolating bicubic spline of the grid values; on a line ('x=1') values and
    normal derivatives are averaged along the line.
    """
    from scipy.integrate import trapezoid
    from scipy.interpolate import RectBivariateSpline

    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    U = np.asarray(field, dtype=float).reshape(len(x_vals), len(y_vals))
    if not np.all(np.isfinite(U)):
        raise MetricSpecError("2D field is not finite")
    kx = min(3, len(x_vals) - 1)
    ky = min(3, len(y_vals) - 1)
    spline = RectBivariateSpline(x_vals, y_vals, U, kx=kx, ky=ky, s=0)
    x_bounds = (float(x_vals[0]), float(x_vals[-1]))
    y_bounds = (float(y_vals[0]), float(y_vals[-1]))

    kind = spec["kind"]
    scale = metric_scale(spec, params)
    loc = _location_2d(spec.get("location"), x_name, y_name, params, x_bounds, y_bounds)

    def line_samples(dx: int = 0, dy: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        axis, coord = loc
        if axis == "line_x":
            coord = _check_inside(coord, x_bounds, x_name)
            along = np.linspace(y_bounds[0], y_bounds[1], _FINE_POINTS_LINE)
            return along, spline.ev(np.full_like(along, coord), along, dx=dx, dy=dy)
        coord = _check_inside(coord, y_bounds, y_name)
        along = np.linspace(x_bounds[0], x_bounds[1], _FINE_POINTS_LINE)
        return along, spline.ev(along, np.full_like(along, coord), dx=dx, dy=dy)

    def line_mean(values: np.ndarray, along: np.ndarray) -> float:
        return float(trapezoid(values, along) / (along[-1] - along[0]))

    if kind in _LOCATED_KINDS:
        if loc is None:
            raise MetricSpecError(f"{kind} needs a location such as '{x_name}=1' or '(0.5, 0.5)'")
        if loc[0] == "point":
            px = _check_inside(loc[1][0], x_bounds, x_name)
            py = _check_inside(loc[1][1], y_bounds, y_name)
            if kind == "derivative_at":
                raise MetricSpecError("derivative_at needs a line location ('x=...' gives d/dx, 'y=...' gives d/dy)")
            value = float(spline.ev(px, py))
        else:
            if kind == "value_at":
                along, values = line_samples()
            elif loc[0] == "line_x":
                along, values = line_samples(dx=1)
            else:
                along, values = line_samples(dy=1)
            value = line_mean(values, along)
        result = scale * value
    elif kind in ("max", "min", "max_abs"):
        if loc is None:
            xf = np.linspace(x_bounds[0], x_bounds[1], _REFINE_2D * (len(x_vals) - 1) + 1)
            yf = np.linspace(y_bounds[0], y_bounds[1], _REFINE_2D * (len(y_vals) - 1) + 1)
            values = np.concatenate([spline(xf, yf).ravel(), U.ravel()])
        elif loc[0] == "point":
            values = np.array([spline.ev(loc[1][0], loc[1][1])])
        else:
            _, values = line_samples()
        values = scale * np.asarray(values, dtype=float)
        if kind == "max":
            result = float(np.max(values))
        elif kind == "min":
            result = float(np.min(values))
        else:
            result = float(np.max(np.abs(values)))
    elif kind in ("mean", "integral"):
        if loc is None:
            integral = float(spline.integral(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]))
            area = (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0])
            value = integral / area if kind == "mean" else integral
        elif loc[0] == "point":
            raise MetricSpecError(f"{kind} over a point is not defined; use a line or the whole domain")
        else:
            along, values = line_samples()
            integral = float(trapezoid(values, along))
            value = integral / (along[-1] - along[0]) if kind == "mean" else integral
        result = scale * value
    else:  # pragma: no cover
        raise MetricSpecError(f"unknown metric kind {kind!r}")

    if not math.isfinite(result):
        raise MetricSpecError(f"metric {kind} is not finite")
    return float(result)


# ==========================================
# Baseline and gain
# ==========================================

def baseline_overrides(auditor_output: Any) -> Dict[str, float]:
    """The auditor's baseline_parameters as a {name: value} dict (empty if none were given)."""
    raw = getattr(auditor_output, "baseline_parameters", None)
    overrides: Dict[str, float] = {}
    for item in raw or []:
        data = _as_plain_dict(item)
        name = str(data.get("name") or "").strip()
        value = data.get("value")
        if not name or value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            overrides[name] = value
    return overrides


def apply_baseline_overrides(
    params: Dict[str, float],
    overrides: Dict[str, float],
    audited_names: List[str],
    coefficient_aliases: Tuple[str, ...],
) -> Dict[str, float]:
    """
    params with the baseline overrides applied. An override of simulation_coefficient also
    moves the slip aliases that were derived from it (not the ones audited explicitly).
    """
    baseline = dict(params)
    baseline.update(overrides)
    if "simulation_coefficient" in overrides:
        for alias in coefficient_aliases:
            if alias not in audited_names and alias not in overrides:
                baseline[alias] = overrides["simulation_coefficient"]
    return baseline


def lower_is_better(auditor_output: Any) -> bool:
    contract = _as_plain_dict(getattr(auditor_output, "objective_metric", None))
    return bool(contract.get("lower_is_better", False))


def relative_gain(design: float, baseline: float, lower_better: bool) -> Optional[float]:
    """
    Relative improvement of the design metric over the baseline in percent:
    (baseline - design) / |baseline| * 100 if lower is better, else (design - baseline) / |baseline| * 100.
    None if the baseline metric is zero or not finite.
    """
    if not (math.isfinite(design) and math.isfinite(baseline)) or baseline == 0.0:
        return None
    diff = (baseline - design) if lower_better else (design - baseline)
    return diff / abs(baseline) * 100.0


def references_any(names: List[str], texts: List[str]) -> List[str]:
    """The names that occur as identifiers in any of the texts (equation, BCs)."""
    joined = " ".join(str(t) for t in texts)
    return [name for name in names if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", joined)]
