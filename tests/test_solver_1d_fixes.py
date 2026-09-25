# -*- coding: utf-8 -*-
"""Offline regression tests for the 1D parsing / solver fixes.

Complements tests/test_solver_accuracy.py (which exercises dispatch_and_solve with
the analytical solver for most of these bugs) with direct checks of the parsing
helpers, the SciPy and PINN code paths, and the GenericPINN output scaling.
"""
import contextlib
import io
import math
import os
import shutil
import tempfile
import unittest

import numpy as np
import sympy as sp
import torch

from tests.solver_accuracy_helpers import Case, run_case
from vectornaut.solvers.parsing import (
    escape_keywords,
    get_domain_bounds,
    parse_equation_and_bcs,
    safe_symbol_name,
)
from vectornaut.solvers.pinn_model import GenericPINN
from vectornaut.solvers.solvers_1d import (
    _pinn_output_scaling,
    solve_analytical,
    solve_pytorch_pinn,
    solve_scipy_bvp,
)

PINN_EPOCHS = 300


def _parse(equation, bcs, x_name, y_name, params):
    return parse_equation_and_bcs(equation, bcs, x_name, y_name, params)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class DomainBoundsTest(unittest.TestCase):

    def test_math_function_values_are_not_bc_locations(self):
        bcs = ["c(0) = 1", "c(1) = exp(-2) + sin(3) * sqrt(4)"]
        self.assertEqual(get_domain_bounds(bcs), (0.0, 1.0))
        self.assertEqual(get_domain_bounds(bcs, {}, dependent_var="c"), (0.0, 1.0))

    def test_capitalised_dependent_variable_is_not_a_math_function(self):
        self.assertEqual(get_domain_bounds(["Gamma(0) = 0", "Gamma(2) = gamma(1)"]), (0.0, 2.0))

    def test_dependent_variable_mode_ignores_other_calls(self):
        bcs = ["u(0.5) = g(-3)", "du_dy(2) = 0"]
        self.assertEqual(get_domain_bounds(bcs, None, dependent_var="u"), (0.5, 2.0))
        # Without the dependent variable name, unknown calls still count (old behaviour).
        self.assertEqual(get_domain_bounds(bcs), (-3.0, 2.0))

    def test_dependent_variable_mode_falls_back_when_name_unused(self):
        bcs = ["u(0) = 0", "u(2) = exp(1)"]
        self.assertEqual(get_domain_bounds(bcs, None, dependent_var="velocity"), (0.0, 2.0))

    def test_derivative_notations(self):
        self.assertEqual(get_domain_bounds(["u(0) = 0", "u'(3) = 0"], None, "u"), (0.0, 3.0))
        self.assertEqual(get_domain_bounds(["T(1) = 0", "d2T_dx2(4) = 0"], None, "T"), (1.0, 4.0))

    def test_submicron_and_parameter_locations(self):
        self.assertEqual(get_domain_bounds(["u(0) = 0", "u(h) = 0"], {"h": 5e-7}), (0.0, 5e-7))
        self.assertEqual(get_domain_bounds(["u(0) = 0", "u(h) = 0"], {"h": 1e-9}, "u"), (0.0, 1e-9))
        self.assertEqual(get_domain_bounds(["u(0) = 0", "du_dy(h/2) = 0"], {"h": 1e-6}), (0.0, 5e-7))

    def test_degenerate_domains_fall_back_to_unit_interval(self):
        self.assertEqual(get_domain_bounds(["u(0) = 0", "du_dy(0) = 1"]), (0.0, 1.0))
        self.assertEqual(get_domain_bounds(["u(0) = 0", "u(oo) = 1"]), (0.0, 1.0))
        self.assertEqual(get_domain_bounds(["u(2) = 1"]), (0.0, 1.0))

    def test_keyword_parameter_in_bc(self):
        bcs = ["u(0) = lambda * du_dy(0)", "u(L) = 1"]
        self.assertEqual(get_domain_bounds(bcs, {"lambda": 0.1, "L": 2.0}, "u"), (0.0, 2.0))


class KeywordEscapingTest(unittest.TestCase):

    def test_escape_is_token_wise(self):
        self.assertEqual(safe_symbol_name("lambda"), "_kw_lambda")
        self.assertEqual(safe_symbol_name("lambda_"), "lambda_")
        self.assertEqual(escape_keywords("lambda * lambda_s + lambda_ + for"), "_kw_lambda * lambda_s + lambda_ + _kw_for")

    def test_lambda_and_lambda_underscore_are_distinct(self):
        params = {"lambda_": 3.0, "lambda": 1.0}
        bcs = ["u(0) = 0", "u(1) = 0"]
        rhs, _, y, u, syms = _parse("d2u_dy2 = -lambda_ - lambda", bcs, "y", "u", params)
        expr, deriv = solve_analytical(rhs, bcs, y, u, syms, params, 0.0, 1.0)
        # u'' = -4 -> u = 2 y (1 - y), u'(0) = 2
        self.assertAlmostEqual(deriv, 2.0, places=12)

    def test_lambda_slip_bc_scipy_and_pinn(self):
        params = {"lambda": 0.25, "u_free": 2.0}
        bcs = ["u(0) = lambda * du_dy(0)", "u(1) = u_free"]
        rhs, _, y, u, syms = _parse("d2u_dy2 = 0", bcs, "y", "u", params)
        _, scipy_deriv = solve_scipy_bvp(rhs, bcs, y, u, syms, params, 0.0, 1.0)
        self.assertAlmostEqual(scipy_deriv, 1.6, places=9)
        _, _, pinn_deriv = _quiet(solve_pytorch_pinn, rhs, bcs, y, u, syms, params, 0.0, 1.0, PINN_EPOCHS)
        self.assertAlmostEqual(pinn_deriv, 1.6, delta=1e-2)


class BareDependentVariableTest(unittest.TestCase):

    def test_bare_variable_means_function_of_x(self):
        params = {"k": 2.0}
        bcs = ["u(0) = 0", "u(1) = 1"]
        rhs, _, y, u, _ = _parse("d2u_dy2 = -k**2 * u", bcs, "y", "u", params)
        self.assertEqual(sp.expand(rhs + u * sp.Symbol("k") ** 2), 0)

    def test_names_containing_the_variable_are_untouched(self):
        params = {"mu": 1.0, "u_free": 1.0, "T_inf": 2.0}
        rhs, _, x, T, syms = _parse("d2T_dx2 = mu * (T - T_inf) + u_free", ["T(0) = 1", "T(1) = 1"], "x", "T", params)
        self.assertEqual(sp.expand(rhs - (syms["mu"] * (T - syms["T_inf"]) + syms["u_free"])), 0)

    def test_bare_variable_scipy_and_pinn(self):
        params = {"k": 2.0}
        bcs = ["u(0) = 0", "u(1) = 1"]
        rhs, _, y, u, syms = _parse("d2u_dy2 = -k**2 * u", bcs, "y", "u", params)
        exact = 2.0 / math.sin(2.0)
        f, scipy_deriv = solve_scipy_bvp(rhs, bcs, y, u, syms, params, 0.0, 1.0)
        self.assertAlmostEqual(scipy_deriv, exact, delta=1e-6 * exact)
        self.assertAlmostEqual(float(f(0.5)), math.sin(1.0) / math.sin(2.0), delta=1e-6)
        _, _, pinn_deriv = _quiet(solve_pytorch_pinn, rhs, bcs, y, u, syms, params, 0.0, 1.0, PINN_EPOCHS)
        self.assertAlmostEqual(pinn_deriv, exact, delta=2e-2 * exact)


class IntegrationConstantTest(unittest.TestCase):

    def test_parameter_named_c1_only_in_bcs(self):
        # dsolve names its constants C1, C2 even though a BC parameter is called C1.
        params = {"C1": 3.0}
        bcs = ["u(0) = C1", "u(1) = 0"]
        rhs, _, y, u, syms = _parse("d2u_dy2 = 0", bcs, "y", "u", params)
        expr, deriv = solve_analytical(rhs, bcs, y, u, syms, params, 0.0, 1.0)
        self.assertAlmostEqual(float(expr.subs(y, 0.0)), 3.0, places=12)
        self.assertAlmostEqual(deriv, -3.0, places=12)

    def test_parameters_starting_with_c(self):
        params = {"Cf": 2.0, "C_p": 1.0, "C1": 1.0}
        bcs = ["u(0) = 0", "u(1) = 0"]
        rhs, _, y, u, syms = _parse("d2u_dy2 = -Cf * C_p - C1", bcs, "y", "u", params)
        expr, deriv = solve_analytical(rhs, bcs, y, u, syms, params, 0.0, 1.0)
        self.assertAlmostEqual(float(expr.subs(y, 0.5)), 3.0 / 8.0, places=12)
        self.assertAlmostEqual(deriv, 1.5, places=12)


class ScipyWallDerivativeTest(unittest.TestCase):

    def test_derivative_uses_bvp_solution(self):
        for h in (1e-7, 1e-5, 1.0, 100.0):
            params = {"G": 1.0e4, "mu": 1.0e-3, "h": h}
            bcs = ["u(0) = 0", "u(h) = 0"]
            lo, hi = get_domain_bounds(bcs, params, "u")
            self.assertEqual((lo, hi), (0.0, h))
            rhs, _, y, u, syms = _parse("d2u_dy2 = -G / mu", bcs, "y", "u", params)
            _, deriv = solve_scipy_bvp(rhs, bcs, y, u, syms, params, lo, hi)
            exact = 1.0e4 / 1.0e-3 * h / 2.0
            self.assertAlmostEqual(deriv, exact, delta=1e-6 * exact, msg=f"h={h}")


class PinnNumericConstantsTest(unittest.TestCase):

    def test_numeric_constants_in_equation_and_bc(self):
        params = {"Lc": 0.5}
        bcs = ["c(0) = 1", "c(1) = exp(-2)"]
        lo, hi = get_domain_bounds(bcs, params)
        rhs, _, x, c, syms = _parse("d2c_dx2 = c(x) / Lc**2 + 0 * pi**2", bcs, "x", "c", params)
        model, losses, deriv = _quiet(solve_pytorch_pinn, rhs, bcs, x, c, syms, params, lo, hi, PINN_EPOCHS)
        self.assertTrue(all(np.isfinite(losses)))
        self.assertAlmostEqual(deriv, -2.0, delta=0.05)
        with torch.no_grad():
            u1 = float(model(torch.tensor([[1.0]])).item())
        # measured: 1.0e-2 off at 300 epochs (BC fit at the far end)
        self.assertAlmostEqual(u1, math.exp(-2.0), delta=3e-2)


class PinnOutputScalingTest(unittest.TestCase):

    def test_scaling_from_boundary_values(self):
        Ya0, Ya1, Yb0, Yb1 = sp.symbols("Ya0 Ya1 Yb0 Yb1")
        vals, ders = (Ya0, Yb0), (Ya1, Yb1)
        self.assertEqual(_pinn_output_scaling([Ya0 - 310.0, Yb0 - 260.0], vals, ders, 1.0), (285.0, 25.0))
        # O(1) fields keep the unscaled network (scale floor 1).
        self.assertEqual(_pinn_output_scaling([Ya0, Yb0], vals, ders, 1.0), (0.0, 1.0))
        self.assertEqual(_pinn_output_scaling([Ya0 - 1.0, Yb0 - 3.0], vals, ders, 1.0), (2.0, 1.0))
        # Neumann value times the domain length sets the scale; Robin BCs are ignored.
        self.assertEqual(_pinn_output_scaling([Ya0 - 300.0, Yb1 + 50.0], vals, ders, 2.0), (300.0, 100.0))
        self.assertEqual(_pinn_output_scaling([Ya0 - 0.25 * Ya1, Yb0 - 2.0], vals, ders, 1.0), (2.0, 1.0))
        self.assertEqual(_pinn_output_scaling([Ya0 ** 2 - 4.0, Yb0 - 2.0], vals, ders, 1.0), (2.0, 1.0))
        # Forcing scale max|u''| L^2 / 8 (u'' = -2000 on [0, 1] -> 250); a non-finite forcing is ignored.
        grid = np.linspace(0.0, 1.0, 5)
        self.assertEqual(_pinn_output_scaling([Ya0, Yb0], vals, ders, 1.0, lambda x, u, du: -2000.0 + 0 * x, grid),
                         (0.0, 250.0))
        self.assertEqual(_pinn_output_scaling([Ya0, Yb0], vals, ders, 1.0, lambda x, u, du: 1.0 / x, grid), (0.0, 1.0))

    def test_default_model_is_unscaled(self):
        torch.manual_seed(0)
        model = GenericPINN(input_dim=2, hidden_dim=8)
        x = torch.rand(5, 2)
        self.assertTrue(torch.equal(model(x), model.net(x)))

    def test_scaling_is_saved_and_legacy_weights_load(self):
        model = GenericPINN()
        model.set_output_scaling(285.0, 25.0)
        restored = GenericPINN()
        restored.load_state_dict(model.state_dict())
        x = torch.linspace(0.0, 1.0, 7).view(-1, 1)
        self.assertTrue(torch.equal(restored(x), model(x)))

        # Weights cached before the scaling buffers existed load with the identity scaling.
        legacy = {k: v for k, v in model.state_dict().items() if k.startswith("net.")}
        old = GenericPINN()
        old.load_state_dict(legacy)
        self.assertTrue(torch.equal(old(x), model.net(x)))


class PinnForcingScaleTest(unittest.TestCase):

    def test_strongly_forced_channel_flow(self):
        # The shark-skin case from tests/manual/test_solver.py: u'' = -2000, max u ~ 250.
        params = {"G": 2.0, "mu": 0.001, "U": 1.5}
        bcs = ["u(0) = 0", "u(1) = U"]
        rhs, _, y, u, syms = _parse("d2u_dy2 = -G / mu", bcs, "y", "u", params)
        model, _, deriv = _quiet(solve_pytorch_pinn, rhs, bcs, y, u, syms, params, 0.0, 1.0, 200)
        grid = np.linspace(0.0, 1.0, 20)
        exact = 1000.0 * grid * (1.0 - grid) + 1.5 * grid
        with torch.no_grad():
            pred = model(torch.tensor(grid, dtype=torch.float32).view(-1, 1)).view(-1).numpy()
        # measured 6.9e-4 (the unscaled network was off by ~70 %)
        self.assertLess(np.max(np.abs(pred - exact)) / np.max(np.abs(exact)), 1e-2)
        self.assertAlmostEqual(deriv, 1001.5, delta=0.02 * 1001.5)


class _TempDataDir(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="vectornaut_1d_fixes_")
        cls._previous = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = cls._tmp

    @classmethod
    def tearDownClass(cls):
        if cls._previous is None:
            os.environ.pop("VECTORNAUT_DATA_DIR", None)
        else:
            os.environ["VECTORNAUT_DATA_DIR"] = cls._previous
        shutil.rmtree(cls._tmp, ignore_errors=True)


class DispatchPinnTest(_TempDataDir):

    def test_scaled_pinn_is_reused_from_cache(self):
        case = Case("offset_temperature_cache", "d2T_dx2 = 0", ["T(0) = T_hot", "T(1) = T_cold"], ["x"], "T",
                    exact=lambda x: 310.0 - 50.0 * x, params={"T_hot": 310.0, "T_cold": 260.0})
        first = run_case(case, "pinn", epochs=PINN_EPOCHS, record=False)
        second = run_case(case, "pinn", epochs=PINN_EPOCHS, record=False)
        self.assertEqual(first.sim.solver_method, "pinn", first.log)
        self.assertIn("Loaded pre-trained model", second.log)
        self.assertLess(first.errors()["max_abs"], 0.5)
        np.testing.assert_allclose(second.sim.solution_primary, first.sim.solution_primary, rtol=0, atol=1e-9)
        self.assertAlmostEqual(second.sim.primary_metric_value, first.sim.primary_metric_value, places=9)

    def test_pinn_with_exp_bc_value(self):
        case = Case("decay_exp_bc_pinn", "d2c_dx2 = c(x) / Lc**2", ["c(0) = 1", "c(1) = exp(-2)"], ["x"], "c",
                    exact=lambda x: np.exp(-2.0 * x), params={"Lc": 0.5})
        run = run_case(case, "pinn", epochs=PINN_EPOCHS, record=False)
        self.assertEqual(run.sim.solver_method, "pinn", run.log)
        self.assertEqual((run.sim.sample_points[0], run.sim.sample_points[-1]), (0.0, 1.0))
        self.assertLess(run.errors()["rel_max"], 3e-2)  # measured 1.0e-2
        self.assertAlmostEqual(run.sim.primary_metric_value, -2.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
