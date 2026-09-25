# -*- coding: utf-8 -*-
"""Tests for the general linear 2D operator (finding 4 in docs/LIVE_PATH_FINDINGS.md).

The 2D solvers used to parse only the right-hand side of the governing equation and always
solve u_xx + u_yy = RHS, silently dropping coefficients, anisotropy factors and terms on the
left-hand side. The whole equation is now parsed into

    a*u_xx + b*u_yy + c*u_xy + d*u_x + e*u_y + g*u = f(x, y)

(parse_pde_2d) and solved by the sparse FDM and the 2D PINN. Every case below has a closed-form
or manufactured solution; anything that cannot be represented must raise a clear ValueError.
"""
import math
import os
import shutil
import tempfile
import unittest

import numpy as np
import sympy as sp

from tests.solver_accuracy_helpers import Case, run_case
from vectornaut.solvers.solvers_2d import LinearPDE2D, parse_bcs_2d, parse_pde_2d, parse_rhs_2d, solve_fdm_2d

PINN_EPOCHS = 300
_PREVIOUS_TORCH_THREADS = None


def setUpModule():
    # Same setup as tests/test_solver_accuracy.py: one intra-op thread for the tiny PINNs.
    global _PREVIOUS_TORCH_THREADS
    import torch
    _PREVIOUS_TORCH_THREADS = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    import torch
    if _PREVIOUS_TORCH_THREADS:
        torch.set_num_threads(_PREVIOUS_TORCH_THREADS)


def fdm(equation, bcs, dep="u", params=None, grid_size=21):
    """Parses and solves a 2D problem with the FDM; returns (X, Y, u) on the grid."""
    params = params or {}
    pde = parse_pde_2d(equation, dep, "x", "y", params)
    parsed, xb, yb = parse_bcs_2d(bcs, dep, ["x", "y"], params)
    xs, ys, u = solve_fdm_2d(pde, parsed, sp.Symbol("x"), sp.Symbol("y"), grid_size=grid_size,
                             x_bounds=xb, y_bounds=yb)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return X, Y, u


def dirichlet_bcs(exact_expr, dep="u", x_bounds=(0, 1), y_bounds=(0, 1)):
    """Dirichlet BCs on all four edges from a SymPy expression of the exact solution."""
    x, y = sp.symbols("x y")
    (x0, x1), (y0, y1) = x_bounds, y_bounds
    return [
        f"{dep}({x0}, y) = {sp.sstr(exact_expr.subs(x, x0))}",
        f"{dep}({x1}, y) = {sp.sstr(exact_expr.subs(x, x1))}",
        f"{dep}(x, {y0}) = {sp.sstr(exact_expr.subs(y, y0))}",
        f"{dep}(x, {y1}) = {sp.sstr(exact_expr.subs(y, y1))}",
    ]


def manufactured_source(exact_expr, a=1, b=1, c=0, d=0, e=0, g=0):
    x, y = sp.symbols("x y")
    u = exact_expr
    return sp.simplify(a * sp.diff(u, x, 2) + b * sp.diff(u, y, 2) + c * sp.diff(u, x, y)
                       + d * sp.diff(u, x) + e * sp.diff(u, y) + g * u)


# Anisotropic conduction with a cold bottom edge (the window case of finding 4): the left and
# right faces are held at the surface temperatures, the bottom edge (spacer) is cold and the top
# edge is insulated. With w = T - L(x), L the linear through-thickness profile,
#   w_xx + a w_yy = 0, w(0, y) = w(1, y) = 0, w(x, 0) = T_B - L(x), w_y(x, 1) = 0
#   => w = sum_n b_n sin(n pi x) cosh(n pi (1 - y) / sqrt(a)) / cosh(n pi / sqrt(a)).
WINDOW_T0, WINDOW_T1, WINDOW_TB = 264.67, 288.22, 270.0
WINDOW_BCS = [f"T(0, y) = {WINDOW_T0}", f"T(1, y) = {WINDOW_T1}", f"T(x, 0) = {WINDOW_TB}", "dT_dy(x, 1) = 0"]


def window_series(a, X, Y, terms=400):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T = WINDOW_T0 + (WINDOW_T1 - WINDOW_T0) * X
    for n in range(1, terms + 1):
        k = n * math.pi
        b_n = 2.0 * ((WINDOW_TB - WINDOW_T0) * (1.0 - math.cos(k)) / k + (WINDOW_T1 - WINDOW_T0) * math.cos(k) / k)
        lam = k / math.sqrt(a)
        # cosh(lam (1 - y)) / cosh(lam), written overflow-free
        decay = (np.exp(-lam * Y) + np.exp(-lam * (2.0 - Y))) / (1.0 + math.exp(-2.0 * lam))
        T = T + b_n * np.sin(k * X) * decay
    return T


class ParsePde2DTest(unittest.TestCase):

    def assertCoeffs(self, pde, expected, rhs):
        for term in ("u_xx", "u_yy", "u_xy", "u_x", "u_y", "u"):
            self.assertEqual(sp.simplify(pde.coeffs[term] - expected.get(term, 0)), 0, term)
        self.assertEqual(sp.simplify(pde.rhs - rhs), 0)

    def test_plain_laplace_and_poisson_forms(self):
        pde = parse_pde_2d("d2T_dx2 + d2T_dy2 = -volumetric_source", "T", "x", "y", {"volumetric_source": 2.0})
        self.assertTrue(pde.is_laplacian)
        self.assertCoeffs(pde, {"u_xx": 1, "u_yy": 1}, -2.0)
        x, y = sp.symbols("x y")
        pde = parse_pde_2d("d2u_dx2 + d2u_dy2 = -2*pi**2 * sin(pi * x) * sin(pi * y)", "u", "x", "y", {})
        self.assertTrue(pde.is_laplacian)
        self.assertCoeffs(pde, {"u_xx": 1, "u_yy": 1}, -2 * sp.pi ** 2 * sp.sin(sp.pi * x) * sp.sin(sp.pi * y))

    def test_coefficient_on_the_left_hand_side(self):
        pde = parse_pde_2d("k*(d2T_dx2 + d2T_dy2) = -q", "T", "x", "y", {"k": 2.0, "q": 3.0})
        self.assertFalse(pde.is_laplacian)
        self.assertCoeffs(pde, {"u_xx": 2, "u_yy": 2}, -3)

    def test_anisotropy_factor_from_normalisation(self):
        eq = "d2T_dx2 + ((2*glass_thickness + gap_width)/window_height)**2 * d2T_dy2 = 0"
        params = {"glass_thickness": 0.004, "gap_width": 0.016, "window_height": 1.2}
        pde = parse_pde_2d(eq, "T", "x", "y", params)
        self.assertCoeffs(pde, {"u_xx": 1, "u_yy": (0.024 / 1.2) ** 2}, 0)

    def test_terms_on_both_sides_and_all_term_types(self):
        pde = parse_pde_2d("d2u_dx2 + 3*du_dx = -d2u_dy2 + 0.5*d2u_dxdy - du_dy + k**2*u + x*y",
                           "u", "x", "y", {"k": 2.0})
        x, y = sp.symbols("x y")
        self.assertCoeffs(pde, {"u_xx": 1, "u_yy": 1, "u_xy": -0.5, "u_x": 3, "u_y": 1, "u": -4}, x * y)

    def test_other_notations(self):
        x, y = sp.symbols("x y")
        for eq in ("T_xx + T_yy - 2*T_x = sin(x)", "diff(T, x, 2) + diff(T, y, 2) - 2*diff(T, x) = sin(x)",
                   "laplacian(T) - 2*dT_dx(x, y) = sin(x)", "d2T_dx2(x, y) + d2T_dy2(x, y) = sin(x) + 2*dT_dx"):
            with self.subTest(eq=eq):
                self.assertCoeffs(parse_pde_2d(eq, "T", "x", "y", {}), {"u_xx": 1, "u_yy": 1, "u_x": -2}, sp.sin(x))

    def test_spatially_varying_coefficient(self):
        x, y = sp.symbols("x y")
        pde = parse_pde_2d("(1 + x)*d2u_dx2 + d2u_dy2 = y", "u", "x", "y", {})
        self.assertCoeffs(pde, {"u_xx": 1 + x, "u_yy": 1}, y)

    def test_python_keyword_parameter(self):
        pde = parse_pde_2d("lambda*(d2u_dx2 + d2u_dy2) = 1", "u", "x", "y", {"lambda": 0.5})
        self.assertCoeffs(pde, {"u_xx": 0.5, "u_yy": 0.5}, 1)

    def test_parse_rhs_2d_accepts_whole_equations(self):
        # The dispatcher's baseline solve re-parses the whole equation with parse_rhs_2d.
        pde = parse_rhs_2d("k*(d2T_dx2 + d2T_dy2) = -q", "x", "y", {"k": 2.0, "q": 3.0})
        self.assertIsInstance(pde, LinearPDE2D)
        self.assertCoeffs(pde, {"u_xx": 2, "u_yy": 2}, -3)
        # A bare source term is still parsed as before.
        self.assertEqual(parse_rhs_2d("-q", "x", "y", {"q": 3.0}), -3.0)

    def test_rejections(self):
        cases = {
            "d2u_dx2 + d2u_dy2 = u**2": "nonlinear",
            "d2u_dx2 + u*d2u_dy2 = 0": "nonlinear",
            "d2u_dx2 + d2u_dy2 + du_dx*du_dy = 0": "nonlinear",
            "d2u_dx2 + d2u_dy2 = sin(u)": "nonlinear",
            "d2u_dx2 - d2u_dy2 = 0": "not elliptic",
            "d2u_dx2 + d2u_dy2 + 3*d2u_dxdy = 0": "not elliptic",
            "d2u_dx2 = 1": "no u_yy term",
            "d2u_dx2 + d2u_dy2 = unknown_source": "unknown symbol",
            "d2u_dx2 + d2u_dy2 = f(x)": "unsupported function",
            "diff(u, x, 3) + d2u_dy2 = 0": "unsupported derivative",
            "d2u_dx2 + d2u_dy2 == 0": "exactly one '='",
            "d2u_dx2 + d2u_dy2": "exactly one '='",
        }
        for eq, message in cases.items():
            with self.subTest(eq=eq):
                with self.assertRaises(ValueError) as ctx:
                    parse_pde_2d(eq, "u", "x", "y", {})
                self.assertIn(message, str(ctx.exception))


class Fdm2DGeneralTest(unittest.TestCase):

    def test_lhs_coefficient_same_as_divided_source(self):
        # k (u_xx + u_yy) = -q  ==  u_xx + u_yy = -q/k; exact u = q/(2k) x (1 - x).
        params = {"k": 2.0, "q": 3.0}
        bcs = ["u(0, y) = 0", "u(1, y) = 0", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"]
        X, Y, u1 = fdm("k*(d2u_dx2 + d2u_dy2) = -q", bcs, params=params)
        _, _, u2 = fdm("d2u_dx2 + d2u_dy2 = -q/k", bcs, params=params)
        self.assertLess(np.max(np.abs(u1 - u2)), 1e-12)
        self.assertLess(np.max(np.abs(u1 - 0.75 * X * (1 - X))), 1e-10)
        # The old behaviour (coefficient dropped) would give twice the amplitude.
        self.assertAlmostEqual(float(u1.max()), 0.75 * 0.25, delta=1e-10)

    def test_negative_lhs_coefficient(self):
        # -k Laplace(u) = q (sign flipped on both sides) gives the same field.
        params = {"k": 2.0, "q": 3.0}
        bcs = ["u(0, y) = 0", "u(1, y) = 0", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"]
        X, Y, u = fdm("-k*d2u_dx2 - k*d2u_dy2 = q", bcs, params=params)
        self.assertLess(np.max(np.abs(u - 0.75 * X * (1 - X))), 1e-10)

    def test_anisotropic_window_with_cold_edge(self):
        # Finding 4: with (L/H)^2 = 4e-4 on d2T_dy2 the pipeline gave 272.91 K at (0.5, 0.25)
        # for both forms; the anisotropic answer is 276.45 K.
        x_mid = 20  # grid_size 41: node 20 is x = 0.5, node 10 is y = 0.25
        _, _, u_iso = fdm("d2T_dx2 + d2T_dy2 = 0", WINDOW_BCS, dep="T", grid_size=41)
        _, _, u_aniso = fdm("d2T_dx2 + a*d2T_dy2 = 0", WINDOW_BCS, dep="T", params={"a": 4e-4}, grid_size=41)
        self.assertAlmostEqual(float(u_iso[x_mid, 10]), float(window_series(1.0, 0.5, 0.25)), delta=0.01)
        self.assertAlmostEqual(float(window_series(1.0, 0.5, 0.25)), 272.91, delta=0.01)
        self.assertAlmostEqual(float(u_aniso[x_mid, 10]), float(window_series(4e-4, 0.5, 0.25)), delta=0.01)
        self.assertAlmostEqual(float(u_aniso[x_mid, 10]), 276.445, delta=0.01)
        # The same equation written with the anisotropy on the other term (divided by a).
        _, _, u_scaled = fdm("2500*d2T_dx2 + d2T_dy2 = 0", WINDOW_BCS, dep="T", grid_size=41)
        self.assertLess(np.max(np.abs(u_scaled - u_aniso)), 1e-9)

    def test_moderate_anisotropy_matches_series(self):
        # a = 0.25: the cold edge reaches further into the domain than for a = 4e-4 but less
        # than for a = 1. Compared away from the bottom edge, whose corners carry discontinuous
        # boundary data (a local O(1) error at the node next to the corner on every grid).
        errors = []
        for n in (21, 41):
            X, Y, u = fdm("d2T_dx2 + 0.25*d2T_dy2 = 0", WINDOW_BCS, dep="T", grid_size=n)
            exact = window_series(0.25, X, Y)
            errors.append(float(np.max(np.abs(u - exact)[Y >= 0.2])))
        self.assertLess(errors[1], 0.02)
        self.assertLess(errors[1], 0.35 * errors[0])  # second order
        _, _, u_iso = fdm("d2T_dx2 + d2T_dy2 = 0", WINDOW_BCS, dep="T", grid_size=41)
        self.assertGreater(np.max(np.abs(u_iso - exact)), 1.0)

    def test_advection_diffusion_1d_profile(self):
        # u_xx + u_yy = Pe u_x, u(0) = 0, u(1) = 1, insulated y edges:
        # u = (exp(Pe x) - 1) / (exp(Pe) - 1). Central differences, O(h^2).
        pe = 5.0
        X, Y, u = fdm("d2u_dx2 + d2u_dy2 = Pe*du_dx", ["u(0, y) = 0", "u(1, y) = 1", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"],
                      params={"Pe": pe}, grid_size=41)
        exact = (np.exp(pe * X) - 1.0) / (math.exp(pe) - 1.0)
        self.assertLess(np.max(np.abs(u - exact)), 5e-3)
        # Halving h reduces the error ~4x (second order).
        X2, _, u2 = fdm("d2u_dx2 + d2u_dy2 = Pe*du_dx", ["u(0, y) = 0", "u(1, y) = 1", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"],
                        params={"Pe": pe}, grid_size=81)
        exact2 = (np.exp(pe * X2) - 1.0) / (math.exp(pe) - 1.0)
        self.assertLess(np.max(np.abs(u2 - exact2)), 0.35 * np.max(np.abs(u - exact)))

    def test_advection_dominated_stays_bounded(self):
        # Pe = 200 on a 21-node grid: the cell Peclet number is 10, central differences would
        # oscillate; the upwind switch keeps the solution within the boundary values.
        pe = 200.0
        X, Y, u = fdm("d2u_dx2 + d2u_dy2 - Pe*du_dx = 0", ["u(0, y) = 0", "u(1, y) = 1", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"],
                      params={"Pe": pe})
        self.assertGreaterEqual(float(u.min()), -1e-12)
        self.assertLessEqual(float(u.max()), 1.0 + 1e-12)
        self.assertTrue(np.all(np.diff(u, axis=0) >= -1e-12))
        # Away from the outflow boundary layer the exact solution is ~0.
        self.assertLess(float(np.max(u[X < 0.7])), 1e-6)

    def test_advection_diffusion_manufactured(self):
        # u_xx + 2 u_yy + 3 u_x - u_y = f with u = sin(x) exp(y) + x y^2, Dirichlet edges.
        x, y = sp.symbols("x y")
        exact = sp.sin(x) * sp.exp(y) + x * y ** 2
        f = manufactured_source(exact, a=1, b=2, d=3, e=-1)
        eq = f"d2u_dx2 + 2*d2u_dy2 + 3*du_dx - du_dy = {sp.sstr(f)}"
        X, Y, u = fdm(eq, dirichlet_bcs(exact), grid_size=41)
        ex = sp.lambdify((x, y), exact, "numpy")(X, Y)
        self.assertLess(np.max(np.abs(u - ex)), 2e-4)

    def test_reaction_term_exponential_solution(self):
        # u_xx + u_yy - k^2 u = 0 with u = exp(0.6 k x + 0.8 k y), k = 2.
        x, y = sp.symbols("x y")
        exact = sp.exp(sp.Rational(6, 5) * x + sp.Rational(8, 5) * y)
        X, Y, u = fdm("d2u_dx2 + d2u_dy2 - k**2 * u = 0", dirichlet_bcs(exact), params={"k": 2.0}, grid_size=41)
        ex = np.exp(1.2 * X + 1.6 * Y)
        self.assertLess(np.max(np.abs(u - ex)) / ex.max(), 2e-4)

    def test_reaction_term_sinh_solution(self):
        # u_xx + u_yy = k^2 u, u(0) = 0, u(1) = 1, insulated y edges: u = sinh(k x) / sinh(k).
        k = 3.0
        X, Y, u = fdm("d2u_dx2 + d2u_dy2 = k**2 * u", ["u(0, y) = 0", "u(1, y) = 1", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"],
                      params={"k": k}, grid_size=41)
        self.assertLess(np.max(np.abs(u - np.sinh(k * X) / math.sinh(k))), 2e-3)

    def test_mixed_derivative_manufactured(self):
        x, y = sp.symbols("x y")
        exact = sp.sin(x) * sp.cos(2 * y) + x ** 2 * y
        f = manufactured_source(exact, a=1, b=1, c=sp.Rational(1, 2))
        X, Y, u = fdm(f"d2u_dx2 + d2u_dy2 + 0.5*d2u_dxdy = {sp.sstr(f)}", dirichlet_bcs(exact), grid_size=41)
        ex = sp.lambdify((x, y), exact, "numpy")(X, Y)
        self.assertLess(np.max(np.abs(u - ex)), 2e-4)

    def test_spatially_varying_coefficient_on_rectangle(self):
        # (1 + x) u_xx + u_yy = f on [0, 2] x [0, 1] with u = x^3 + x y^2.
        x, y = sp.symbols("x y")
        exact = x ** 3 + x * y ** 2
        f = sp.expand((1 + x) * sp.diff(exact, x, 2) + sp.diff(exact, y, 2))
        X, Y, u = fdm(f"(1 + x)*d2u_dx2 + d2u_dy2 = {sp.sstr(f)}", dirichlet_bcs(exact, x_bounds=(0, 2)), grid_size=41)
        self.assertAlmostEqual(float(X.max()), 2.0)
        ex = X ** 3 + X * Y ** 2
        self.assertLess(np.max(np.abs(u - ex)) / ex.max(), 1e-3)

    def test_coefficient_changing_sign_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            fdm("(x - 0.5)*d2u_dx2 + d2u_dy2 = 0", ["u(0, y) = 0", "u(1, y) = 1"])
        self.assertIn("not elliptic", str(ctx.exception))

    def test_bare_source_still_means_laplacian(self):
        x, y = sp.symbols("x y")
        parsed, xb, yb = parse_bcs_2d(["u(0, y) = 0", "u(1, y) = 0.5", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"],
                                      "u", ["x", "y"], {})
        _, _, u_old = solve_fdm_2d(sp.Integer(1), parsed, x, y, grid_size=21, x_bounds=xb, y_bounds=yb)
        _, _, u_new = solve_fdm_2d(parse_pde_2d("d2u_dx2 + d2u_dy2 = 1", "u", "x", "y", {}), parsed, x, y,
                                   grid_size=21, x_bounds=xb, y_bounds=yb)
        np.testing.assert_array_equal(u_old, u_new)


class Dispatch2DGeneralTest(unittest.TestCase):
    """End-to-end through dispatch_and_solve (FDM reference + PINN)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="vectornaut_2d_general_")
        cls._previous = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = cls._tmp

    @classmethod
    def tearDownClass(cls):
        if cls._previous is None:
            os.environ.pop("VECTORNAUT_DATA_DIR", None)
        else:
            os.environ["VECTORNAUT_DATA_DIR"] = cls._previous
        shutil.rmtree(cls._tmp, ignore_errors=True)

    LHS_BCS = ["u(0, y) = 0", "u(1, y) = 0", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"]

    def lhs_case(self, name, equation):
        return Case(name, equation, self.LHS_BCS, ["x", "y"], "u",
                    exact=lambda x, y: 0.75 * x * (1 - x), params={"k": 2.0, "q": 3.0})

    def test_fdm_lhs_coefficient(self):
        run = run_case(self.lhs_case("lhs_coefficient_fdm", "k*(d2u_dx2 + d2u_dy2) = -q"), "fdm", record=False)
        self.assertEqual(run.sim.solver_method, "fdm")
        self.assertLess(run.errors()["rel_max"], 1e-9)
        self.assertLess(run.sim.relative_error, 1e-9)

    def test_fdm_anisotropic_window(self):
        case = Case("window_anisotropic_fdm", "d2T_dx2 + ((2*glass_thickness + gap_width)/window_height)**2 * d2T_dy2 = 0",
                    WINDOW_BCS, ["x", "y"], "T", exact=lambda x, y: window_series(4e-4, x, y),
                    params={"glass_thickness": 0.004, "gap_width": 0.016, "window_height": 1.2})
        run = run_case(case, "fdm", record=False)
        # The cold-edge boundary layer (thickness ~ sqrt(a)/pi = 0.006) is not resolved by the
        # 20 x 20 grid, so compare from y = 0.2 on; there T is the through-thickness profile.
        away = run.points[:, 1] >= 0.2
        self.assertLess(run.errors(away)["max_abs"], 1e-3)
        # The dropped-anisotropy (isotropic) field is ~3.5 K colder at (0.5, 0.25).
        self.assertGreater(float(window_series(4e-4, 0.5, 0.25) - window_series(1.0, 0.5, 0.25)), 3.0)

    def test_pinn_lhs_coefficient_trains_like_divided_source(self):
        # After normalisation k (u_xx + u_yy) = -q is exactly the problem u_xx + u_yy = -q/k.
        run1 = run_case(self.lhs_case("lhs_coefficient_pinn", "k*(d2u_dx2 + d2u_dy2) = -q"), "pinn",
                        epochs=PINN_EPOCHS, record=False)
        run2 = run_case(self.lhs_case("divided_source_pinn", "d2u_dx2 + d2u_dy2 = -q/k"), "pinn",
                        epochs=PINN_EPOCHS, record=False)
        self.assertEqual(run1.sim.solver_method, "pinn", run1.log)
        np.testing.assert_allclose(run1.sim.solution_primary, run2.sim.solution_primary, rtol=0, atol=1e-6)
        self.assertLess(run1.errors()["max_abs"], 0.03)

    def test_pinn_advection_diffusion(self):
        pe = 2.0
        case = Case("advection_pinn", "d2u_dx2 + d2u_dy2 - Pe*du_dx = 0",
                    ["u(0, y) = 0", "u(1, y) = 1", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"], ["x", "y"], "u",
                    exact=lambda x, y: (np.exp(pe * x) - 1.0) / (math.exp(pe) - 1.0), params={"Pe": pe})
        run = run_case(case, "pinn", epochs=PINN_EPOCHS, record=False)
        self.assertEqual(run.sim.solver_method, "pinn", run.log)
        self.assertLess(run.errors()["max_abs"], 0.05)
        # The Laplace solution (advection dropped) is u = x, up to 0.24 away from the exact one.
        self.assertGreater(float(np.max(np.abs(run.points[:, 0] - run.exact_values()))), 0.2)

    def test_pinn_reaction(self):
        k = 3.0
        case = Case("reaction_pinn", "d2u_dx2 + d2u_dy2 = k**2 * u",
                    ["u(0, y) = 0", "u(1, y) = 1", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"], ["x", "y"], "u",
                    exact=lambda x, y: np.sinh(k * x) / math.sinh(k), params={"k": k})
        run = run_case(case, "pinn", epochs=PINN_EPOCHS, record=False)
        self.assertEqual(run.sim.solver_method, "pinn", run.log)
        self.assertLess(run.errors()["max_abs"], 0.05)

    def test_baseline_solve_uses_the_whole_equation(self):
        # (1 + c) Laplace(u) = -2 with u = 0 on x = 0, 1: u = x (1 - x) / (1 + c). The baseline
        # (c = 0) has twice the mean, so the gain is 50 % (0 % if the LHS were dropped).
        case = Case("lhs_baseline", "(1 + simulation_coefficient)*(d2u_dx2 + d2u_dy2) = -2", self.LHS_BCS,
                    ["x", "y"], "u", exact=lambda x, y: 0.5 * x * (1 - x), coefficient=1.0)
        run = run_case(case, "fdm", record=False)
        self.assertLess(run.errors()["rel_max"], 1e-9)
        self.assertAlmostEqual(run.sim.performance_gain_pct, 50.0, delta=1e-6)

    def test_nonlinear_equation_fails_loudly(self):
        case = Case("nonlinear_2d", "d2T_dx2 + d2T_dy2 = T**2",
                    ["T(0, y) = 0", "T(1, y) = 1", "dT_dy(x, 0) = 0", "dT_dy(x, 1) = 0"], ["x", "y"], "T",
                    exact=lambda x, y: x)
        with self.assertRaises(ValueError) as ctx:
            run_case(case, "pinn", record=False)
        self.assertIn("nonlinear", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
