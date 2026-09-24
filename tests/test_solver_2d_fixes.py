# -*- coding: utf-8 -*-
"""Regression tests for the 2D solver fixes (expression-valued edge BCs, rectangular
domains, zero-flux default edges, loud BC parse failures, the refined FDM reference and
the baseline-derived performance gain)."""
import math
import os
import shutil
import tempfile
import unittest

import numpy as np
import sympy as sp

from tests.solver_accuracy_helpers import Case, run_case
from vectornaut.solvers.solvers_2d import parse_bc_2d_string, parse_bcs_2d, solve_fdm_2d


class ParseBcs2DTest(unittest.TestCase):

    def test_unit_square_edges_and_constant_values(self):
        self.assertEqual(parse_bc_2d_string("T(0, y) = T_hot", "T", ["x", "y"], {"T_hot": 300.0}),
                         ("left", "dirichlet", 300.0))
        self.assertEqual(parse_bc_2d_string("dT_dy(x, 1) = 0", "T", ["x", "y"], {}), ("top", "neumann", 0.0))
        self.assertEqual(parse_bc_2d_string("T(x, 0) = pi**2 / 8", "T", ["x", "y"], {})[2], math.pi ** 2 / 8)

    def test_expression_value_along_edge(self):
        edge, bc_type, val = parse_bc_2d_string("u(x, 1) = sin(pi * x) + y", "u", ["x", "y"], {})
        self.assertEqual((edge, bc_type), ("top", "dirichlet"))
        # y is fixed to 1 on the top edge; the value varies with x only.
        self.assertEqual(sp.simplify(val - (sp.sin(sp.pi * sp.Symbol("x")) + 1)), 0)

    def test_domain_from_numbers_and_named_lengths(self):
        bcs = ["T(0, y) = 0", "T(L, y) = 1", "dT_dy(x, -1) = 0", "dT_dy(x, H) = 0"]
        parsed, x_bounds, y_bounds = parse_bcs_2d(bcs, "T", ["x", "y"], {"L": 3.0, "H": 2.0})
        self.assertEqual(x_bounds, (0.0, 3.0))
        self.assertEqual(y_bounds, (-1.0, 2.0))
        self.assertEqual(parsed["right"]["type"], "dirichlet")
        self.assertEqual(parsed["bottom"]["type"], "neumann")

    def test_missing_edges_default_to_zero_flux(self):
        parsed, x_bounds, y_bounds = parse_bcs_2d(["T(0, y) = 1", "T(2, y) = 0"], "T", ["x", "y"], {})
        self.assertEqual(x_bounds, (0.0, 2.0))
        self.assertEqual(y_bounds, (0.0, 1.0))
        for edge in ("bottom", "top"):
            self.assertEqual((parsed[edge]["type"], parsed[edge]["value"]), ("neumann", 0.0))

    def test_unparseable_bc_raises(self):
        # Robin condition: depends on the unknown field, not supported by the 2D solvers.
        with self.assertRaises(ValueError):
            parse_bcs_2d(["T(0, y) = 1", "dT_dx(1, y) = -Bi * T(1, y)"], "T", ["x", "y"], {"Bi": 1.0})
        with self.assertRaises(ValueError):
            parse_bcs_2d(["T(0, y) = 1", "T(1, y) = unknown_symbol"], "T", ["x", "y"], {})

    def test_pure_neumann_problem_raises(self):
        with self.assertRaises(ValueError):
            parse_bcs_2d(["dT_dx(0, y) = 0", "dT_dx(1, y) = 0"], "T", ["x", "y"], {})

    def test_fdm_on_rectangle_with_varying_neumann_edge(self):
        # u = x^2 - y^2 is harmonic; du/dy = 2y on y=1 gives 2 (constant), du/dx = 2 on x=1.
        x, y = sp.symbols("x y")
        bcs = ["u(0, y) = -y**2", "u(2, y) = 4 - y**2", "u(x, 0) = x**2", "u(x, 1) = x**2 - 1"]
        parsed, xb, yb = parse_bcs_2d(bcs, "u", ["x", "y"], {})
        xs, ys, u = solve_fdm_2d(sp.Integer(0), parsed, x, y, grid_size=21, x_bounds=xb, y_bounds=yb)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.assertLess(np.max(np.abs(u - (X ** 2 - Y ** 2))), 1e-10)


class Dispatch2DTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="vectornaut_2d_fixes_")
        cls._previous = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = cls._tmp

    @classmethod
    def tearDownClass(cls):
        if cls._previous is None:
            os.environ.pop("VECTORNAUT_DATA_DIR", None)
        else:
            os.environ["VECTORNAUT_DATA_DIR"] = cls._previous
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_unparseable_bc_fails_loudly(self):
        case = Case("robin_2d", "d2T_dx2 + d2T_dy2 = 0",
                    ["T(0, y) = 1", "dT_dx(1, y) = -Bi * T(1, y)"], ["x", "y"], "T",
                    exact=lambda x, y: x, params={"Bi": 1.0})
        with self.assertRaises(ValueError):
            run_case(case, "fdm", record=False)

    def test_pinn_with_expression_bc(self):
        case = Case("laplace_sin_top_pinn", "d2u_dx2 + d2u_dy2 = 0",
                    ["u(0, y) = 0", "u(1, y) = 0", "u(x, 0) = 0", "u(x, 1) = sin(pi * x)"], ["x", "y"], "u",
                    exact=lambda x, y: np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi))
        run = run_case(case, "pinn", epochs=300, record=False)
        self.assertEqual(run.sim.solver_method, "pinn", run.log)
        self.assertLess(run.errors()["rel_max"], 0.1)

    def test_pinn_on_non_unit_domain(self):
        case = Case("laplace_wide_slab_pinn", "d2T_dx2 + d2T_dy2 = 0",
                    ["T(0, y) = 0", "T(2, y) = 2", "dT_dy(x, 0) = 0", "dT_dy(x, 1) = 0"], ["x", "y"], "T",
                    exact=lambda x, y: x)
        run = run_case(case, "pinn", epochs=300, record=False)
        self.assertEqual(run.sim.solver_method, "pinn", run.log)
        self.assertAlmostEqual(float(run.points[:, 0].max()), 2.0)
        self.assertLess(run.errors()["rel_max"], 2e-2)

    def test_fdm_reference_is_refined_solution(self):
        case = Case("poisson_neumann_flux", "d2u_dx2 + d2u_dy2 = 1",
                    ["u(0, y) = 0", "du_dx(1, y) = 1", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"], ["x", "y"], "u",
                    exact=lambda x, y: 0.5 * x ** 2)
        run = run_case(case, "fdm", record=False)
        exact = run.exact_values()
        primary = np.asarray(run.sim.solution_primary)
        reference = np.asarray(run.sim.solution_reference)
        # The refined reference is closer to the exact solution than the primary field.
        self.assertLess(np.max(np.abs(reference - exact)), 0.75 * np.max(np.abs(primary - exact)))
        self.assertGreater(run.sim.relative_error, 0.01)

    def test_performance_gain_from_baseline_solve(self):
        # T = 1 - c x: mean 1 - c/2 = 0.75 for c = 0.5, baseline (c = 0) mean 1 -> 25 % reduction.
        case = Case("coefficient_bc", "d2T_dx2 + d2T_dy2 = 0",
                    ["T(0, y) = 1", "T(1, y) = 1 - simulation_coefficient", "dT_dy(x, 0) = 0", "dT_dy(x, 1) = 0"],
                    ["x", "y"], "T", exact=lambda x, y: 1.0 - 0.5 * x, coefficient=0.5)
        run = run_case(case, "fdm", record=False)
        self.assertLess(run.errors()["rel_max"], 1e-9)
        self.assertAlmostEqual(run.sim.performance_gain_pct, 25.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
