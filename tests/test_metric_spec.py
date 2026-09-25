# -*- coding: utf-8 -*-
"""Offline tests for the declarative metric spec and the baseline performance gain.

The auditor may describe the reported metric (AuditorOutput.metric: value_at,
derivative_at, max, min, max_abs, mean, integral with a location, a scale expression
and a unit) and the conventional reference design (AuditorOutput.baseline_parameters).
These tests check every metric kind in 1D and 2D against closed forms, the gain
against a baseline for both objective directions, and the "no baseline" path
(gain_basis 'none': reported as n/a, no objective-contract warning).
"""
import contextlib
import io
import os
import re
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import torch

from tests.solver_accuracy_helpers import Case, build_inputs
from vectornaut.config import AuditorOutput, BaselineParameter, MetricSpec, ObjectiveMetricContract
from vectornaut.reporting import generate_markdown_report_content
from vectornaut.validator import validate_run_output


class _TempDataDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="vectornaut_metric_spec_")
        self._previous = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = self._tmp

    def tearDown(self):
        if self._previous is None:
            os.environ.pop("VECTORNAUT_DATA_DIR", None)
        else:
            os.environ["VECTORNAUT_DATA_DIR"] = self._previous
        shutil.rmtree(self._tmp, ignore_errors=True)

    def solve(self, case, method, metric=None, baseline=None, lower_is_better=None, epochs=200,
              threshold=0.0):
        """Runs dispatch_and_solve for a case with an optional metric spec, baseline and objective."""
        from vectornaut.solver_dispatcher import dispatch_and_solve

        miner, auditor = build_inputs(case, method)
        if metric is not None:
            auditor.metric = MetricSpec(**metric)
        if baseline is not None:
            auditor.baseline_parameters = [BaselineParameter(name=k, value=v) for k, v in baseline.items()]
        if lower_is_better is not None:
            auditor.objective_metric = ObjectiveMetricContract(
                objective_name="gain", primary_metric="primary", reference_metric="reference",
                lower_is_better=lower_is_better, acceptance_threshold=threshold,
            )
        torch.manual_seed(0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = dispatch_and_solve(miner, auditor, epochs=epochs)
        return miner, auditor, sim


# Simply supported beam under uniform load in normalized form (q L^4 / EI = 1, x = X/L):
# w'' = -x (1 - x) / 2, w(0) = w(1) = 0  ->  w = (x - 2 x^3 + x^4) / 24, w_max = 5/384 at x = 1/2.
BEAM = Case("beam_normalized", "d2w_dx2 = -q_norm * x * (1 - x) / 2", ["w(0) = 0", "w(1) = 0"], ["x"], "w",
            exact=lambda x: q_norm_exact(x), params={"q_norm": 1.0})


def q_norm_exact(x):
    return (x - 2.0 * x ** 3 + x ** 4) / 24.0


# Poiseuille flow in a channel normalized by its height h: u'' = -G/mu, u(0) = u(1) = 0,
# du/dy(0) = G / (2 mu) = 2; wall shear stress tau = mu/h * du/dy(0) = G / (2 h) = 100 Pa.
POISEUILLE = Case("poiseuille_scaled", "d2u_dy2 = -G / mu", ["u(0) = 0", "u(1) = 0"], ["y"], "u",
                  exact=lambda y: 2.0 * y * (1.0 - y), params={"G": 2.0, "mu": 0.5, "h": 0.01})

# Wall with internal heat source S, adiabatic at x = 0, room temperature at x = 1, normalized by
# the thickness L: T = T_room + S/2 (1 - x^2), dT/dx(1) = -S. Heat flux into the room
# q = -k/L dT/dx(1) = k S / L = 2 * 3 / 0.5 = 12 W/m^2.
HEATER = Case("heater_wall", "d2T_dx2 = -S", ["dT_dx(0) = 0", "T(1) = T_room"], ["x"], "T",
              exact=lambda x: 293.0 + 1.5 * (1.0 - x ** 2), params={"S": 3.0, "k": 2.0, "L": 0.5, "T_room": 293.0})


class MetricKinds1DTest(_TempDataDir):
    BEAM_EXPECTED = {
        ("max_abs", None): 5.0 / 384.0,
        ("max", None): 5.0 / 384.0,
        ("min", None): 0.0,
        ("value_at", "0.5"): 5.0 / 384.0,
        ("derivative_at", "0"): 1.0 / 24.0,
        ("derivative_at", "1"): -1.0 / 24.0,
        ("mean", None): 1.0 / 120.0,
        ("integral", None): 1.0 / 120.0,
    }

    def test_beam_metric_kinds_match_closed_form(self):
        for method in ("analytical", "scipy"):
            for (kind, location), expected in self.BEAM_EXPECTED.items():
                with self.subTest(method=method, kind=kind, location=location):
                    _, _, sim = self.solve(BEAM, method, metric={"kind": kind, "location": location, "unit": "-"})
                    self.assertEqual(sim.solver_method, method)
                    self.assertEqual(sim.metric_spec["kind"], kind)
                    tol = 1e-10 if method == "analytical" else 1e-6
                    self.assertAlmostEqual(sim.primary_metric_value, expected, delta=tol * 5.0 / 384.0 * 100)
                    self.assertAlmostEqual(sim.reference_metric_value, expected, delta=1e-10)

    def test_beam_max_deflection_physical_units(self):
        # 5 q L^4 / (384 EI) for q = 2000 N/m, L = 1 m, EI = 1.0852e5 N m^2 -> 0.240 mm
        case = Case("beam_physical", "d2w_dx2 = -uniform_load * span**4 * x * (1 - x) / (2 * EI)",
                    ["w(0) = 0", "w(1) = 0"], ["x"], "w", exact=lambda x: x,
                    params={"uniform_load": 2000.0, "span": 1.0, "EI": 1.0852e5})
        _, _, sim = self.solve(case, "analytical", metric={"kind": "max_abs", "unit": "m"})
        expected = 5.0 * 2000.0 / (384.0 * 1.0852e5)
        self.assertAlmostEqual(sim.primary_metric_value, expected, delta=1e-12)
        self.assertEqual(sim.metric_unit, "m")

    def test_wall_shear_with_scale(self):
        spec = {"kind": "derivative_at", "location": "0", "scale": "mu/h", "unit": "Pa"}
        for method, tol in (("analytical", 1e-9), ("scipy", 1e-6), ("pinn", 2e-2)):
            with self.subTest(method=method):
                _, _, sim = self.solve(POISEUILLE, method, metric=spec, epochs=300)
                self.assertEqual(sim.solver_method, method)
                self.assertAlmostEqual(sim.primary_metric_value, 100.0, delta=tol * 100.0)
                self.assertAlmostEqual(sim.reference_metric_value, 100.0, delta=1e-6)

    def test_heat_flux_at_far_end(self):
        for location in ("1", "x=1", "domain_max"):
            with self.subTest(location=location):
                _, _, sim = self.solve(HEATER, "analytical", metric={
                    "kind": "derivative_at", "location": location, "scale": "-k/L", "unit": "W/m^2"})
                self.assertAlmostEqual(sim.primary_metric_value, 12.0, delta=1e-9)
        # The default metric is dT/dx at the adiabatic end: 0 (the reported "heat flux 0.0000").
        _, _, default = self.solve(HEATER, "analytical")
        self.assertAlmostEqual(default.primary_metric_value, 0.0, delta=1e-9)
        self.assertIsNone(default.metric_spec)

    def test_value_at_parameter_location(self):
        case = Case("couette_gap", "d2u_dy2 = 0", ["u(0) = 0", "u(h) = U"], ["y"], "u",
                    exact=lambda y: y, params={"U": 3.0, "h": 0.5})
        _, _, sim = self.solve(case, "analytical", metric={"kind": "value_at", "location": "h/2"})
        self.assertAlmostEqual(sim.primary_metric_value, 1.5, delta=1e-12)

    def test_default_metric_unchanged(self):
        _, _, sim = self.solve(POISEUILLE, "analytical")
        self.assertAlmostEqual(sim.primary_metric_value, 2.0, delta=1e-9)
        self.assertIsNone(sim.metric_spec)
        self.assertIsNone(sim.metric_unit)

    def test_invalid_spec_falls_back_to_default(self):
        for bad in ({"kind": "median"}, {"kind": "value_at", "location": "7"}, {"kind": "value_at", "location": "nope"}):
            with self.subTest(spec=bad):
                _, _, sim = self.solve(POISEUILLE, "analytical", metric=bad)
                self.assertAlmostEqual(sim.primary_metric_value, 2.0, delta=1e-9)
                self.assertIsNone(sim.metric_spec)
                self.assertTrue(sim.gain_note)


class BaselineGain1DTest(_TempDataDir):
    FLUX = {"kind": "derivative_at", "location": "1", "scale": "-k/L", "unit": "W/m^2"}

    def test_gain_lower_is_better(self):
        # Baseline with twice the source: flux 24 -> design 12 is 50 % lower.
        _, _, sim = self.solve(HEATER, "analytical", metric=self.FLUX, baseline={"S": 6.0}, lower_is_better=True)
        self.assertEqual(sim.gain_basis, "baseline_parameters")
        self.assertAlmostEqual(sim.baseline_metric_value, 24.0, delta=1e-9)
        self.assertAlmostEqual(sim.performance_gain_pct, 50.0, delta=1e-9)

    def test_gain_higher_is_better(self):
        _, _, sim = self.solve(HEATER, "scipy", metric=self.FLUX, baseline={"S": 6.0}, lower_is_better=False)
        self.assertAlmostEqual(sim.performance_gain_pct, -50.0, delta=1e-4)
        _, _, sim = self.solve(HEATER, "analytical", metric=self.FLUX, baseline={"S": 1.5}, lower_is_better=False)
        self.assertAlmostEqual(sim.performance_gain_pct, 100.0, delta=1e-9)

    def test_baseline_uses_its_own_scale_and_domain(self):
        # Scale uses the baseline's k; a baseline with another gap length gets its own domain.
        _, _, sim = self.solve(HEATER, "analytical", metric=self.FLUX, baseline={"k": 4.0}, lower_is_better=True)
        self.assertAlmostEqual(sim.baseline_metric_value, 24.0, delta=1e-9)
        case = Case("couette_gap", "d2u_dy2 = 0", ["u(0) = 0", "u(h) = U"], ["y"], "u",
                    exact=lambda y: 2.0 * y, params={"U": 1.0, "h": 0.5})
        _, _, sim = self.solve(case, "analytical", metric={"kind": "derivative_at", "location": "0"},
                               baseline={"h": 1.0}, lower_is_better=True)
        self.assertAlmostEqual(sim.primary_metric_value, 2.0, delta=1e-12)
        self.assertAlmostEqual(sim.baseline_metric_value, 1.0, delta=1e-12)
        self.assertAlmostEqual(sim.performance_gain_pct, -100.0, delta=1e-9)

    def test_beam_deflection_vs_stiffer_baseline(self):
        # Design EI = 1e5, baseline (solid) EI = 3e5: w_max is 3x the baseline -> -200 % (lower is better).
        case = Case("beam_ei", "d2w_dx2 = -uniform_load * x * (1 - x) / (2 * EI)", ["w(0) = 0", "w(1) = 0"],
                    ["x"], "w", exact=lambda x: x, params={"uniform_load": 2000.0, "EI": 1.0e5})
        _, _, sim = self.solve(case, "analytical", metric={"kind": "max_abs", "unit": "m"},
                               baseline={"EI": 3.0e5}, lower_is_better=True)
        self.assertAlmostEqual(sim.primary_metric_value, 5.0 * 2000.0 / (384.0 * 1.0e5), delta=1e-15)
        self.assertAlmostEqual(sim.performance_gain_pct, -200.0, delta=1e-6)

    def test_baseline_parameters_accept_mapping(self):
        auditor = AuditorOutput.model_validate({
            "audit_passed": True, "audit_notes": "", "audited_parameters": [], "dimensionless_numbers": [],
            "simulation_coefficient": 0.0, "solver_method": "analytical",
            "ui_metadata": build_inputs(HEATER, "analytical")[1].ui_metadata.model_dump(),
            "baseline_parameters": {"S": 6.0},
        })
        self.assertEqual(auditor.baseline_parameters_dict, {"S": 6.0})

    def test_slip_baseline_still_used_without_baseline_parameters(self):
        case = Case("couette_slip", "d2u_dy2 = 0", ["u(0) = slip_length * du_dy(0)", "u(1) = U"], ["y"], "u",
                    exact=lambda y: 2.0 * (y + 0.25) / 1.25, params={"U": 2.0}, coefficient=0.25)
        _, _, sim = self.solve(case, "analytical")
        self.assertEqual(sim.gain_basis, "bionic_effect")
        self.assertAlmostEqual(sim.performance_gain_pct, 20.0, delta=1e-6)
        self.assertAlmostEqual(sim.baseline_metric_value, 2.0, delta=1e-9)

    def test_no_baseline_gain_not_available(self):
        _, _, sim = self.solve(BEAM, "analytical", metric={"kind": "max_abs"})
        self.assertEqual(sim.gain_basis, "none")
        self.assertEqual(sim.performance_gain_pct, 0.0)
        self.assertIsNone(sim.baseline_metric_value)
        # An unused simulation coefficient does not create a baseline either.
        case = Case("coating_gap", "d2u_dy2 = 0", ["u(0) = 0", "u(coating_thickness) = U"], ["y"], "u",
                    exact=lambda y: 2.0 * y, params={"U": 1.0, "coating_thickness": 0.5}, coefficient=0.3)
        _, _, sim = self.solve(case, "pinn", epochs=50)
        self.assertEqual(sim.gain_basis, "none")
        self.assertEqual(sim.performance_gain_pct, 0.0)


# 2D fields on the unit square with insulated top/bottom edges: T = T0 + (T1 - T0) x.
LINEAR_2D = Case("linear_slab_2d", "d2T_dx2 + d2T_dy2 = 0",
                 ["T(0, y) = T_out", "T(1, y) = T_in", "dT_dy(x, 0) = 0", "dT_dy(x, 1) = 0"], ["x", "y"], "T",
                 exact=lambda x, y: -2.0 + 3.0 * x, params={"T_out": -2.0, "T_in": 1.0, "k": 0.8})
# u = x^2 (the 5-point stencil is exact for quadratics): mean 1/3, du/dx(1) = 2.
QUADRATIC_2D = Case("quadratic_2d", "d2u_dx2 + d2u_dy2 = 2",
                    ["u(0, y) = 0", "u(1, y) = 1", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"], ["x", "y"], "u",
                    exact=lambda x, y: x ** 2)


class MetricKinds2DTest(_TempDataDir):
    def test_linear_field_metric_kinds(self):
        expected = [
            ({"kind": "value_at", "location": "x=1"}, 1.0),
            ({"kind": "value_at", "location": "(0.5, 0.3)"}, -0.5),
            ({"kind": "derivative_at", "location": "x=0"}, 3.0),
            ({"kind": "derivative_at", "location": "x=1", "scale": "k"}, 2.4),
            ({"kind": "derivative_at", "location": "y=0"}, 0.0),
            ({"kind": "max"}, 1.0),
            ({"kind": "min"}, -2.0),
            ({"kind": "max_abs"}, 2.0),
            ({"kind": "mean"}, -0.5),
            ({"kind": "integral"}, -0.5),
            ({"kind": "mean", "location": "x=1"}, 1.0),
            ({"kind": "max", "location": "right"}, 1.0),
        ]
        for spec, value in expected:
            with self.subTest(spec=spec):
                _, _, sim = self.solve(LINEAR_2D, "fdm", metric=spec)
                self.assertEqual(sim.solver_method, "fdm")
                self.assertEqual(sim.metric_spec["kind"], spec["kind"])
                self.assertAlmostEqual(sim.primary_metric_value, value, delta=1e-8)
                self.assertAlmostEqual(sim.reference_metric_value, value, delta=1e-8)

    def test_quadratic_field_metrics(self):
        for spec, value in (({"kind": "mean"}, 1.0 / 3.0), ({"kind": "integral"}, 1.0 / 3.0),
                            ({"kind": "derivative_at", "location": "x=1"}, 2.0),
                            ({"kind": "value_at", "location": "(0.5, 0.5)"}, 0.25)):
            with self.subTest(spec=spec):
                _, _, sim = self.solve(QUADRATIC_2D, "fdm", metric=spec)
                self.assertAlmostEqual(sim.primary_metric_value, value, delta=1e-6)

    def test_default_metric_is_sample_mean(self):
        _, _, sim = self.solve(QUADRATIC_2D, "fdm")
        self.assertAlmostEqual(sim.primary_metric_value, float(np.mean(sim.solution_primary)), delta=1e-12)
        self.assertIsNone(sim.metric_spec)
        self.assertEqual(sim.gain_basis, "none")

    def test_derivative_at_point_is_rejected(self):
        _, _, sim = self.solve(LINEAR_2D, "fdm", metric={"kind": "derivative_at", "location": "(0.5, 0.5)"})
        self.assertIsNone(sim.metric_spec)
        self.assertIn("derivative_at", sim.gain_note)


class BaselineGain2DTest(_TempDataDir):
    FLUX = {"kind": "derivative_at", "location": "x=1", "scale": "k", "unit": "W/m^2"}

    def test_heat_loss_reduction_vs_baseline(self):
        # Heat flux k * (T_in - T_out) = 0.8 * 3 = 2.4; baseline k = 1.6 -> 4.8: 50 % less heat loss.
        _, _, sim = self.solve(LINEAR_2D, "fdm", metric=self.FLUX, baseline={"k": 1.6}, lower_is_better=True)
        self.assertEqual(sim.gain_basis, "baseline_parameters")
        self.assertAlmostEqual(sim.primary_metric_value, 2.4, delta=1e-8)
        self.assertAlmostEqual(sim.baseline_metric_value, 4.8, delta=1e-8)
        self.assertAlmostEqual(sim.performance_gain_pct, 50.0, delta=1e-6)
        self.assertEqual(sim.metric_unit, "W/m^2")

    def test_gain_higher_is_better_2d(self):
        # Baseline with a smaller temperature difference (T_in = 0.5): flux 2.0 -> design 2.4 is 20 % higher.
        _, _, sim = self.solve(LINEAR_2D, "fdm", metric=self.FLUX, baseline={"T_in": 0.5}, lower_is_better=False)
        self.assertAlmostEqual(sim.baseline_metric_value, 2.0, delta=1e-8)
        self.assertAlmostEqual(sim.performance_gain_pct, 20.0, delta=1e-6)

    def test_no_baseline_2d(self):
        _, _, sim = self.solve(LINEAR_2D, "fdm", metric=self.FLUX, lower_is_better=True)
        self.assertEqual(sim.gain_basis, "none")
        self.assertEqual(sim.performance_gain_pct, 0.0)


class GainNotAvailableReportingTest(_TempDataDir):
    def _run_data(self, case, method, **kwargs):
        miner, auditor, sim = self.solve(case, method, **kwargs)
        auditor_data = {**auditor.model_dump(), "audited_parameters_dict": auditor.audited_parameters_dict}
        validation = validate_run_output(miner, auditor_data, sim, optimization_history=[])
        data = {
            "miner": miner.model_dump(),
            "auditor": auditor_data,
            "simulator": sim.model_dump(),
            "validation": validation.model_dump(),
            "optimization_history": [{"round": 1, "parameters": auditor.audited_parameters_dict,
                                      "simulation_coefficient": auditor.simulation_coefficient,
                                      "simulator": sim.model_dump(), "validation": validation.model_dump()}],
        }
        return data, validation, generate_markdown_report_content(data, "20260925_120000")

    def test_no_baseline_reports_na_and_validator_does_not_warn(self):
        data, validation, report = self._run_data(
            BEAM, "analytical", metric={"kind": "max_abs", "unit": "m", "label": "Maximum deflection"},
            lower_is_better=True, threshold=30.0)
        check = next(c for c in validation.checks if c.name == "objective_metric_contract")
        self.assertTrue(check.passed)
        self.assertEqual(check.severity, "info")
        self.assertEqual(validation.status, "pass")
        self.assertFalse(any("performance_gain_pct" in w for w in validation.warnings))
        self.assertIn("**gain:** **n/a**", report)
        self.assertNotIn("0.00%", report)
        self.assertIn("**n/a**", report.split("## 7.")[1])  # optimization history table
        self.assertIn("`0.0130208 m`", report)              # w_max = 5/384 with its unit, 6 significant digits
        self.assertIn("(in den Gleichungen nicht verwendet)", report)

    def test_available_gain_still_checked(self):
        # With a baseline the objective contract is evaluated as before (gain 50 % < threshold 60 -> warning).
        _, validation, report = self._run_data(
            HEATER, "analytical", metric=BaselineGain1DTest.FLUX, baseline={"S": 6.0},
            lower_is_better=True, threshold=60.0)
        check = next(c for c in validation.checks if c.name == "objective_metric_contract")
        self.assertFalse(check.passed)
        self.assertEqual(check.severity, "warning")
        self.assertIn("**gain:** **50.00%**", report)
        self.assertIn("`12 W/m^2`", report)
        self.assertIn("`24 W/m^2`", report)  # baseline metric

    def test_legacy_result_without_gain_basis_is_unchanged(self):
        # Results recorded before gain_basis existed keep the old contract check and gain text.
        simulator = {"solver_method": "analytical", "relative_error": 0.0, "performance_gain_pct": 0.0,
                     "sample_points": [0.0, 1.0], "solution_primary": [0.0, 1.0], "solution_reference": [0.0, 1.0],
                     "primary_metric_value": 1.0, "reference_metric_value": 1.0}
        auditor = {"objective_metric": {"score_field": "performance_gain_pct", "direction": "maximize",
                                        "acceptance_threshold": 10.0}}
        validation = validate_run_output({}, auditor, simulator)
        check = next(c for c in validation.checks if c.name == "objective_metric_contract")
        self.assertFalse(check.passed)
        report = generate_markdown_report_content({"simulator": simulator, "auditor": auditor}, "x")
        self.assertIn("**0.00%**", report)

    def test_2d_solution_table_is_capped(self):
        _, _, report = self._run_data(LINEAR_2D, "fdm")
        section = report.split("### Berechnete Stützpunkte und Feldwerte")[1].split("##")[0]
        rows = [line for line in section.splitlines() if re.match(r"^\| \(", line)]
        self.assertLessEqual(len(rows), 25)
        self.assertGreaterEqual(len(rows), 16)
        self.assertIn("von 400 Stützpunkten gezeigt", section)
        # Corners of the domain are part of the subsampled grid.
        self.assertTrue(any(line.startswith("| (0, 0) |") for line in rows))
        self.assertTrue(any(line.startswith("| (1, 1) |") for line in rows))

    def test_coefficient_usage_text(self):
        case = Case("couette_slip", "d2u_dy2 = 0", ["u(0) = slip_length * du_dy(0)", "u(1) = U"], ["y"], "u",
                    exact=lambda y: 2.0 * (y + 0.25) / 1.25, params={"U": 2.0}, coefficient=0.25)
        _, _, report = self._run_data(case, "analytical")
        self.assertIn("`0.25` (genutzt in den Gleichungen)", report)
        self.assertIn("**gain:** **20.00%**", report)


class OptimizerPromptTest(unittest.TestCase):
    def test_metric_and_na_gain_in_history_summary(self):
        from vectornaut.optimizer import _metric_summary
        text = _metric_summary({"primary_metric_value": 2.5333e-4, "metric_unit": "m",
                                "metric_spec": {"kind": "max_abs", "label": "Maximum deflection"},
                                "gain_basis": "none", "performance_gain_pct": 0.0})
        self.assertIn("Metrik (Maximum deflection): 0.00025333 m", text)
        self.assertIn("Performance Gain: n/a", text)
        text = _metric_summary({"primary_metric_value": 37.95, "metric_unit": "W/m^2", "baseline_metric_value": 79.11,
                                "gain_basis": "baseline_parameters", "performance_gain_pct": 52.03})
        self.assertIn("Baseline): 79.11 W/m^2", text)
        self.assertIn("Performance Gain: 52.03%", text)


class AuditorPassThroughTest(unittest.TestCase):
    def test_metric_and_baseline_survive_post_processing(self):
        from types import SimpleNamespace
        from vectornaut.auditor import Auditor

        miner, raw = build_inputs(HEATER, "analytical")
        raw.metric = MetricSpec(kind="derivative_at", location="1", scale="-k/L", unit="W/m^2")
        raw.baseline_parameters = [BaselineParameter(name="S", value=6.0)]
        raw.baseline_description = "no insulation"

        class _Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(parsed=raw)

        out = Auditor(client=SimpleNamespace(models=_Models())).audit_design(miner, user_query="heater")
        self.assertEqual(out.metric.kind, "derivative_at")
        self.assertEqual(out.baseline_parameters_dict, {"S": 6.0})
        self.assertEqual(out.baseline_description, "no insulation")


if __name__ == "__main__":
    unittest.main()
