# -*- coding: utf-8 -*-
"""Solver/validator robustness fixes found in the explorer role-play runs (2026-09-26).

- Analytical solver: degenerate parameter values (log(G_s/G_b) = 0 for equal moduli, a
  decay rate of 0) made the symbolic general solution 1/0 -> zoo/nan, and the non-finite
  baseline metric turned the gain into n/a. The result is now checked and, if unusable,
  solved again with the numbers substituted first; otherwise SolverResultError triggers
  the SciPy fallback.
- SciPy BVP: 1e-8 m displacements were solved in unscaled units with the default absolute
  BC tolerance (BC met to ~23 %, gain 66.7 % instead of 62.1 %). It now solves a
  nondimensional problem and checks the BCs.
- Validator: the derivative BC check used a 3-point difference on 20 samples, which a sharp
  Gaussian load (width ~1 % of the domain) cannot resolve; it now uses the solver's own
  boundary derivative, and otherwise allows for the truncation error of the stencil.
- Auditor: the riblet slip estimate is kept dimensional (no division by 10 * spacing) and
  only recomputed when the equations use the coefficient.
- Fallbacks between solvers are reported in SimulatorOutput.solver_note.
"""
import contextlib
import io
import math
import os
import shutil
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.solver_accuracy_helpers import Case, build_inputs
from vectornaut.auditor import Auditor
from vectornaut.config import (
    AuditedParameter,
    AuditorOutput,
    BaselineParameter,
    DimensionlessNumber,
    MetricSpec,
    MinerOutput,
    ObjectiveMetricContract,
    ParameterProposal,
)
from vectornaut.solvers.parsing import get_domain_bounds, parse_equation_and_bcs
from vectornaut.solvers.solvers_1d import SolverResultError, solve_analytical, solve_scipy_bvp
from vectornaut.validator import validate_run_data, validate_run_output


def _parse(eq, bcs, x_name, y_name, params):
    lo, hi = get_domain_bounds(bcs, params, y_name)
    rhs, _, x, y, syms = parse_equation_and_bcs(eq, bcs, x_name, y_name, params)
    return rhs, x, y, syms, lo, hi


class _TempDataDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="vectornaut_robustness_")
        self._previous = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = self._tmp

    def tearDown(self):
        if self._previous is None:
            os.environ.pop("VECTORNAUT_DATA_DIR", None)
        else:
            os.environ["VECTORNAUT_DATA_DIR"] = self._previous
        shutil.rmtree(self._tmp, ignore_errors=True)

    def dispatch(self, case, method, metric=None, baseline=None, lower_is_better=None):
        from vectornaut.solver_dispatcher import dispatch_and_solve

        miner, auditor = build_inputs(case, method)
        if metric is not None:
            auditor.metric = MetricSpec(**metric)
        if baseline is not None:
            auditor.baseline_parameters = [BaselineParameter(name=k, value=v) for k, v in baseline.items()]
        if lower_is_better is not None:
            auditor.objective_metric = ObjectiveMetricContract(
                objective_name="gain", primary_metric="primary", reference_metric="reference",
                lower_is_better=lower_is_better, acceptance_threshold=0.0,
            )
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = dispatch_and_solve(miner, auditor, epochs=5)
        return miner, auditor, sim


# ---------------------------------------------------------------------------
# Graded-modulus foul-release coating (explorer run e00001, guided).
# G(z) = G_b exp(k z), k = ln(G_s / G_b), d/dz(G u') = 0 -> u'' = -k u', u(0) = 0,
# u'(1) = tau t / G_s (z normalised by the thickness t). Closed form:
#   u(1) = tau t / G_s * (1 - exp(-k)) / k = tau t (1/G_s - 1/G_b) / ln(G_b / G_s);  k = 0: tau t / G.
# Metric: compliance m = u(1) / tau, detachment stress sqrt(2 w / m) (lower is better).
# ---------------------------------------------------------------------------
COATING_EQ = "d2u_dz2 = -log(surface_shear_modulus / base_shear_modulus) * du_dz"
COATING_BCS = ["u(0) = 0", "du_dz(1) = wall_shear_stress * coating_thickness / surface_shear_modulus"]
COATING_DESIGN = {"coating_thickness": 0.002, "surface_shear_modulus": 1.0e5, "base_shear_modulus": 2.0e6,
                  "wall_shear_stress": 18.0, "adhesion_work_barnacle": 0.05}
COATING_BASELINE = {"coating_thickness": 0.0003, "surface_shear_modulus": 330000.0, "base_shear_modulus": 330000.0}
COATING_METRIC = {"kind": "value_at", "location": "1", "scale": "1/wall_shear_stress",
                  "transform": "sqrt(2*adhesion_work_barnacle/m)", "unit": "Pa", "label": "detachment stress"}


def coating_u1(t, g_s, g_b, tau):
    if g_s == g_b:
        return tau * t / g_s
    return tau * t * (1.0 / g_s - 1.0 / g_b) / math.log(g_b / g_s)


def coating_stress(params):
    p = dict(COATING_DESIGN, **params)
    m = coating_u1(p["coating_thickness"], p["surface_shear_modulus"], p["base_shear_modulus"], p["wall_shear_stress"]) / p["wall_shear_stress"]
    return math.sqrt(2.0 * p["adhesion_work_barnacle"] / m)


class DegenerateAnalyticalTest(unittest.TestCase):

    def test_equal_moduli_solved_exactly(self):
        params = dict(COATING_DESIGN, **COATING_BASELINE)
        rhs, z, u, syms, lo, hi = _parse(COATING_EQ, COATING_BCS, "z", "u", params)
        expr, deriv = solve_analytical(rhs, COATING_BCS, z, u, syms, params, lo, hi)
        exact_u1 = 18.0 * 0.0003 / 330000.0
        self.assertAlmostEqual(float(expr.subs(z, 1.0)), exact_u1, delta=1e-12 * exact_u1)
        self.assertAlmostEqual(deriv, exact_u1, delta=1e-12 * exact_u1)

    def test_graded_design_unchanged(self):
        params = dict(COATING_DESIGN)
        rhs, z, u, syms, lo, hi = _parse(COATING_EQ, COATING_BCS, "z", "u", params)
        expr, _ = solve_analytical(rhs, COATING_BCS, z, u, syms, params, lo, hi)
        exact_u1 = coating_u1(0.002, 1.0e5, 2.0e6, 18.0)
        self.assertAlmostEqual(float(expr.subs(z, 1.0)), exact_u1, delta=1e-10 * exact_u1)

    def test_zero_decay_rate_solved_exactly(self):
        # u'' = -q exp(-a y) with a = 0: the symbolic solution has 1/a**2 terms.
        params = {"q": 3.0, "a": 0.0}
        bcs = ["u(0) = 0", "u(1) = 0"]
        rhs, y, u, syms, lo, hi = _parse("d2u_dy2 = -q * exp(-a * y)", bcs, "y", "u", params)
        expr, deriv = solve_analytical(rhs, bcs, y, u, syms, params, lo, hi)
        self.assertAlmostEqual(float(expr.subs(y, 0.5)), 3.0 / 8.0, places=12)
        self.assertAlmostEqual(deriv, 1.5, places=12)

    def test_unusable_closed_form_raises(self):
        # The equation itself is singular for a = 0; no solver result must be passed on.
        params = {"q": 1.0, "a": 0.0}
        bcs = ["u(0) = 0", "u(1) = 0"]
        rhs, y, u, syms, lo, hi = _parse("d2u_dy2 = -q * exp(-a * y) / a", bcs, "y", "u", params)
        with self.assertRaises(SolverResultError):
            solve_analytical(rhs, bcs, y, u, syms, params, lo, hi)

    def test_oscillatory_solution_is_real(self):
        # Symbolic dsolve gives exp(+-y sqrt(-k)) (complex for k > 0); the numeric retry gives sin/cos.
        params = {"k": 4.0}
        bcs = ["u(0) = 0", "u(1) = 1"]
        rhs, y, u, syms, lo, hi = _parse("d2u_dy2 = -k * u", bcs, "y", "u", params)
        expr, deriv = solve_analytical(rhs, bcs, y, u, syms, params, lo, hi)
        self.assertAlmostEqual(float(expr.subs(y, 0.5)), math.sin(1.0) / math.sin(2.0), places=12)
        self.assertAlmostEqual(deriv, 2.0 / math.sin(2.0), places=12)


class TinyScaleBvpTest(unittest.TestCase):

    def _check(self, params):
        rhs, z, u, syms, lo, hi = _parse(COATING_EQ, COATING_BCS, "z", "u", params)
        f, deriv = solve_scipy_bvp(rhs, COATING_BCS, z, u, syms, params, lo, hi)
        p = dict(COATING_DESIGN, **params)
        exact_u1 = coating_u1(p["coating_thickness"], p["surface_shear_modulus"], p["base_shear_modulus"], p["wall_shear_stress"])
        neumann = p["wall_shear_stress"] * p["coating_thickness"] / p["surface_shear_modulus"]
        self.assertAlmostEqual(float(f(1.0)), exact_u1, delta=1e-6 * exact_u1)
        self.assertAlmostEqual(float(f.bvp_sol(1.0)[1]), neumann, delta=1e-6 * neumann)  # BC u'(1)
        self.assertLess(abs(float(f(0.0))), 1e-6 * exact_u1)                               # BC u(0)
        self.assertTrue(np.isfinite(deriv))

    def test_baseline_displacement_1e8_m(self):
        # Before: u(1) = 1.264e-8 instead of 1.636e-8 (BC met to ~23 %).
        self._check(dict(COATING_DESIGN, **COATING_BASELINE))

    def test_design_displacement(self):
        self._check(dict(COATING_DESIGN))

    def test_scales_from_1e_minus_12_to_1e6(self):
        for g in (1e-12, 1.0, 1e6):
            params = {"g": g}
            bcs = ["u(0) = 0", "du_dy(1) = g"]
            rhs, y, u, syms, lo, hi = _parse("d2u_dy2 = 2 * du_dy", bcs, "y", "u", params)
            f, _ = solve_scipy_bvp(rhs, bcs, y, u, syms, params, lo, hi)
            exact = g * (1.0 - math.exp(-2.0)) / 2.0  # u = g e^{-2} (e^{2y} - 1) / 2
            self.assertAlmostEqual(float(f(1.0)), exact, delta=1e-6 * exact, msg=f"g={g}")


class CoatingGainTest(_TempDataDir):
    """e00001: correct gain 1 - sigma_design / sigma_baseline = 62.1 % (reported: 66.7 %)."""

    def test_gain_against_equal_moduli_baseline(self):
        sigma_design = coating_stress({})
        sigma_base = coating_stress(COATING_BASELINE)
        exact_gain = (1.0 - sigma_design / sigma_base) * 100.0
        self.assertAlmostEqual(sigma_base, 10488.1, delta=0.1)
        self.assertAlmostEqual(exact_gain, 62.14, delta=0.01)
        case = Case("coating", COATING_EQ, COATING_BCS, ["z"], "u", exact=lambda z: 0 * z, params=COATING_DESIGN)
        for method in ("analytical", "scipy"):
            _, _, sim = self.dispatch(case, method, metric=COATING_METRIC, baseline=COATING_BASELINE, lower_is_better=True)
            self.assertEqual(sim.gain_basis, "baseline_parameters", sim.gain_note)
            self.assertIsNone(sim.gain_note)
            self.assertAlmostEqual(sim.baseline_metric_value, sigma_base, delta=1e-6 * sigma_base, msg=method)
            self.assertAlmostEqual(sim.primary_metric_value, sigma_design, delta=1e-6 * sigma_design, msg=method)
            self.assertAlmostEqual(sim.performance_gain_pct, exact_gain, delta=1e-4, msg=method)
            self.assertIsNone(sim.solver_note)

    def test_fallback_reason_is_reported(self):
        case = Case("coating", COATING_EQ, COATING_BCS, ["z"], "u", exact=lambda z: 0 * z, params=COATING_DESIGN)
        with mock.patch("vectornaut.solver_dispatcher.solve_analytical", side_effect=SolverResultError("closed form is not finite")):
            _, _, sim = self.dispatch(case, "analytical", metric=COATING_METRIC, baseline=COATING_BASELINE, lower_is_better=True)
        self.assertEqual(sim.solver_method, "scipy")
        self.assertIn("analytical solution failed (closed form is not finite); SciPy BVP used as primary", sim.solver_note)
        self.assertIn("baseline: analytical solution failed", sim.solver_note)
        self.assertAlmostEqual(sim.performance_gain_pct, (1.0 - coating_stress({}) / coating_stress(COATING_BASELINE)) * 100.0, delta=1e-4)


# ---------------------------------------------------------------------------
# Sharp Gaussian load (explorer run e00013, guided): w'' = a w - b exp(-pi (x L / l)**2),
# w'(0) = 0, w(1) = 0 with a load width l / L = 0.02, far below the 20-sample spacing.
# ---------------------------------------------------------------------------
GAUSS_EQ = ("d2w_dx2 = (gel_youngs_modulus / gel_thickness) * half_span**2 / (crust_shear_modulus * crust_thickness) * w"
            " - half_span**2 / (crust_shear_modulus * crust_thickness) * load_pressure * exp(-pi * (x * half_span / load_width)**2)")
GAUSS_BCS = ["dw_dx(0) = 0", "w(1) = 0"]
GAUSS_PARAMS = {"crust_shear_modulus": 5.0e6, "crust_thickness": 0.0003, "gel_thickness": 0.0025,
                "gel_youngs_modulus": 180000.0, "half_span": 0.015, "load_pressure": 500000.0, "load_width": 0.0003}


class SharpSourceValidatorTest(_TempDataDir):

    @classmethod
    def setUpClass(cls):
        # One solve for the class: the symbolic attempt takes a few seconds before it
        # reports unevaluated integrals and the SciPy BVP solution is used.
        tmp = tempfile.mkdtemp(prefix="vectornaut_robustness_gauss_")
        previous = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = tmp
        try:
            from vectornaut.solver_dispatcher import dispatch_and_solve
            cls.miner, cls.auditor = build_inputs(
                Case("gauss", GAUSS_EQ, GAUSS_BCS, ["x"], "w", exact=lambda x: 0 * x, params=GAUSS_PARAMS), "scipy")
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls.sim = dispatch_and_solve(cls.miner, cls.auditor, epochs=5)
        finally:
            if previous is None:
                os.environ.pop("VECTORNAUT_DATA_DIR", None)
            else:
                os.environ["VECTORNAUT_DATA_DIR"] = previous
            shutil.rmtree(tmp, ignore_errors=True)

    def _validate(self, sim):
        return validate_run_output(self.miner, self.auditor, sim)

    def _check(self, result):
        return next(c for c in result.checks if c.name == "physics_derivative_boundary_conditions")

    def test_solution_meets_bc_and_reports_reference_fallback(self):
        self.assertEqual(self.sim.solver_method, "scipy")
        self.assertIn("reference: analytical solution failed", self.sim.solver_note or "")
        derivs = {round(d["location"], 12): d["value"] for d in self.sim.boundary_derivatives}
        self.assertLess(abs(derivs[0.0]), 1e-9)

    def test_correct_sharp_source_is_not_flagged(self):
        # Before: residual 6.1e-4 from the samples vs tolerance 1.1e-5 -> warn.
        result = self._validate(self.sim)
        check = self._check(result)
        self.assertTrue(check.passed, check.detail)
        self.assertIn("solver derivative", check.detail)
        self.assertEqual(result.status, "pass", [c.detail for c in result.checks if not c.passed])

    def test_wrong_solver_derivative_is_flagged(self):
        wrong = [dict(d) for d in self.sim.boundary_derivatives]
        wrong[0]["value"] = -6.0e-4  # w'(0) != 0
        check = self._check(self._validate(self.sim.model_copy(update={"boundary_derivatives": wrong})))
        self.assertFalse(check.passed, check.detail)


class DerivativeAllowanceTest(unittest.TestCase):
    """Sample-based check without solver derivatives: truncation-error allowance of the stencil."""
    y = np.linspace(0.0, 1.0, 20)

    def _run(self, u, bcs):
        return {
            "miner": {"governing_equation": "d2u_dy2 = u / d**2", "boundary_conditions": bcs,
                      "independent_variables": ["y"], "dependent_variables": ["u"]},
            "auditor": {"audit_passed": True},
            "simulator": {"solver_method": "analytical", "final_loss": 0.0, "relative_error": 0.0,
                          "performance_gain_pct": 0.0, "sample_points": list(self.y),
                          "solution_primary": list(u), "solution_reference": list(u),
                          "primary_metric_value": 1.0, "reference_metric_value": 1.0},
        }

    def _check(self, result):
        return next(c for c in result.checks if c.name == "physics_derivative_boundary_conditions")

    def test_boundary_layer_profile_passes(self):
        # u = exp(-y / 0.1): the 3-point estimate of u'(0) = -10 is off by ~0.9 (> 5 % of 10).
        u = np.exp(-self.y / 0.1)
        check = self._check(validate_run_data(self._run(u, ["du_dy(0) = -10", "u(1) = 4.5399929762484854e-05"])))
        self.assertTrue(check.passed, check.detail)

    def test_wrong_flux_on_boundary_layer_profile_is_flagged(self):
        u = np.exp(-self.y / 0.1)
        check = self._check(validate_run_data(self._run(u, ["du_dy(0) = 0", "u(1) = 4.5399929762484854e-05"])))
        self.assertFalse(check.passed, check.detail)


# ---------------------------------------------------------------------------
# Auditor: riblet coefficient
# ---------------------------------------------------------------------------

def _audit(params, coefficient, bcs, overrides=None):
    ui = {"domain_name": "Fluid Dynamics", "independent_var": {"label": "y", "unit": "-"},
          "dependent_var": {"label": "u", "unit": "m/s"}, "primary_metric": {"label": "Wall shear stress"},
          "reference_metric": {"label": "Wall shear stress (analytical)"}, "performance_gain": {"label": "Drag reduction"}}
    audit = AuditorOutput(
        audit_passed=True, audit_notes="model notes",
        audited_parameters=[AuditedParameter(name=k, value=v) for k, v in params.items()],
        dimensionless_numbers=[DimensionlessNumber(name="ReynoldsNumber", value=5000.0)],
        simulation_coefficient=coefficient, solver_method="analytical", ui_metadata=ui,
    )
    miner = MinerOutput(
        design_name="Riblet foil", inspiration_source="shark skin", domain="Fluid Dynamics", physical_mechanism="-",
        parameters=[ParameterProposal(name=k, value=v, min_bound=v / 10.0, max_bound=v * 10.0, justification="-")
                    for k, v in params.items()],
        governing_equation="d2u_dy2 = -G / mu", boundary_conditions=bcs,
        independent_variables=["y"], dependent_variables=["u"], svg_schematic="<svg></svg>",
    )
    client = mock.MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(parsed=audit)
    with contextlib.redirect_stdout(io.StringIO()):
        return Auditor(client=client).audit_design(miner, override_parameters=overrides, user_query="drag")


class RibletCoefficientTest(unittest.TestCase):
    H, S = 5.0e-5, 1.0e-4  # 50 um high, 100 um spacing (< 5 mm: formerly divided by 10 * s)
    PARAMS = {"riblet_height": H, "riblet_spacing": S, "viscosity": 1e-3, "G": 1.0, "mu": 1e-3}

    def test_recompute_stays_dimensional(self):
        result = _audit(self.PARAMS, 4.5e-6, ["u(0) = slip_length * du_dy(0)", "u(1) = 0"])
        expected = 0.2 * self.S * (1.0 - math.exp(-2.0 * self.H / self.S))  # 1.26e-5 m, not 0.0126
        self.assertAlmostEqual(result.simulation_coefficient, expected, delta=1e-15)
        self.assertIn("grobe Schätzung", result.audit_notes)
        self.assertIn("Luchini", result.audit_notes)

    def test_no_recompute_when_equations_do_not_use_the_coefficient(self):
        result = _audit(self.PARAMS, 4.5e-6, ["u(0) = 0", "u(1) = 0"])
        self.assertAlmostEqual(result.simulation_coefficient, 4.5e-6, delta=1e-18)
        self.assertIn("nicht aus der Riblet-Geometrie neu berechnet", result.audit_notes)
        self.assertNotIn("simulation_coefficient: ", result.audit_notes)

    def test_override_slip_length_is_used_as_given(self):
        result = _audit(self.PARAMS, 4.5e-6, ["u(0) = slip_length * du_dy(0)", "u(1) = 0"], overrides={"slip_length": 2.0e-5})
        self.assertAlmostEqual(result.simulation_coefficient, 2.0e-5, delta=1e-18)


if __name__ == "__main__":
    unittest.main()
