# -*- coding: utf-8 -*-
"""dynamic_script result and sweep correctness (LIVE_PATH_FINDINGS 5 and 12)."""
import json
import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock

from vectornaut.config import (
    AuditedParameter, AuditorOutput, AxisMetadata, DimensionlessNumber, MetricMetadata,
    MinerOutput, ObjectiveMetricContract, ParameterProposal, UIMetadata,
)
from vectornaut.script_generator import GeneratedScriptResponse
from vectornaut.solvers.dynamic_script import (
    MAX_SWEEP_RELATIVE_ERROR,
    _dynamic_parameter_sweep,
    noise_level_warning,
    parse_hard_constraint,
)
from vectornaut.test_generator import GeneratedTestScriptResponse
from vectornaut.validator import validate_run_output

# Toy solver: metric = 10 * pore (lower is better) vs reference 10, gain = (1 - pore) * 100.
# success=false for pore < 0.3 (exit code still 0), relative_error 0.2 for pore > 0.7,
# peak_temperature_K above 2000 for pore < 0.33; "noise" scales both metrics to 1e-12.
SOLVER = r'''
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--params"); ap.add_argument("--output"); ap.add_argument("--plot")
a = ap.parse_args()
with open(a.params, encoding="utf-8") as f:
    p = json.load(f)
pore = float(p["pore"])
scale = 1e-12 if p.get("noise") else 1.0
primary, reference = 10.0 * pore * scale, 10.0 * scale
out = {
    "success": pore >= 0.3 or not p.get("flag_small_pores"),
    "performance_gain_pct": (1.0 - primary / reference) * 100.0,
    "relative_error": 0.2 if pore > 0.7 else 1e-6,
    "sample_points": [0.0, 0.5, 1.0],
    "solution_primary": [300.0, 350.0, 400.0],
    "solution_reference": [300.0, 360.0, 420.0],
    "primary_metric_value": primary,
    "reference_metric_value": reference,
    "peak_temperature_K": 2000.0 + (0.33 - pore) * 10000.0,
}
with open(a.output, "w", encoding="utf-8") as f:
    json.dump(out, f)
with open(a.plot, "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n")
'''

PASSING_TESTS = '''
import unittest
from solver_harness import run_solver, nominal_params


class SolverPhysicalValidation(unittest.TestCase):
    def test_nominal_run(self):
        run = run_solver(nominal_params())
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(run.output["success"])
'''

FAILING_TESTS = PASSING_TESTS + '''
    def test_physical_invariants(self):
        self.assertLess(run_solver(nominal_params()).output["performance_gain_pct"], 0.0)
'''


def _miner():
    return MinerOutput(
        design_name="Sweep Toy",
        inspiration_source="test",
        domain="Thermodynamics",
        physical_mechanism="test",
        parameters=[
            ParameterProposal(name="heat_flux", value=1e5, min_bound=1e4, max_bound=1e6, justification="operating condition"),
            ParameterProposal(name="pulse_duration", value=100.0, min_bound=10.0, max_bound=500.0, justification="operating condition"),
            ParameterProposal(name="pore", value=0.5, min_bound=0.1, max_bound=1.0, justification="design"),
            ParameterProposal(name="thickness", value=0.02, min_bound=0.01, max_bound=0.05, justification="design"),
        ],
        governing_equation="dT_dt = d2T_dx2",
        boundary_conditions=["T(0) = 1"],
        independent_variables=["x", "t"],
        dependent_variables=["T"],
        svg_schematic="<svg></svg>",
    )


def _auditor(design_variables=None, hard_constraints=None, extra_params=None):
    params = {"heat_flux": 1e5, "pulse_duration": 100.0, "pore": 0.5, "thickness": 0.02}
    params.update(extra_params or {})
    return AuditorOutput(
        audit_passed=True,
        audit_notes="ok",
        audited_parameters=[AuditedParameter(name=k, value=v) for k, v in params.items()],
        dimensionless_numbers=[DimensionlessNumber(name="Fo", value=0.1)],
        simulation_coefficient=0.1,
        solver_method="dynamic_script",
        ui_metadata=UIMetadata(
            domain_name="Thermal",
            independent_var=AxisMetadata(label="x", unit="m"),
            dependent_var=AxisMetadata(label="T", unit="K"),
            primary_metric=MetricMetadata(label="Bondline rise"),
            reference_metric=MetricMetadata(label="Reference rise"),
            performance_gain=MetricMetadata(label="Reduction"),
        ),
        objective_metric=ObjectiveMetricContract(
            objective_name="Reduction",
            primary_metric="Bondline rise",
            reference_metric="Reference rise",
            lower_is_better=True,
            hard_constraints=hard_constraints if hard_constraints is not None else [f"relative_error <= {MAX_SWEEP_RELATIVE_ERROR}"],
            design_variables=design_variables,
        ),
    )


class SweepTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.script_path = os.path.join(self.tmp, "solver_toy.py")
        with open(self.script_path, "w", encoding="utf-8") as f:
            f.write(SOLVER)

    def tearDown(self):
        self._tmp.cleanup()

    def sweep(self, auditor, base_extra=None, skip_reason=None):
        base_params = dict(auditor.audited_parameters_dict)
        base_params.update(base_extra or {})
        base_result = {
            "success": True, "performance_gain_pct": 50.0, "relative_error": 1e-6,
            "sample_points": [0.0, 0.5, 1.0], "solution_primary": [300.0, 350.0, 400.0],
            "solution_reference": [300.0, 360.0, 420.0],
            "primary_metric_value": 5.0, "reference_metric_value": 10.0, "peak_temperature_K": 1830.0,
        }
        return _dynamic_parameter_sweep(
            miner_output=_miner(), auditor_output=auditor, script_path=self.script_path,
            base_params=base_params, base_result=base_result, slug="toy", timestamp="20260101_000000",
            skip_reason=skip_reason,
        )


class DesignVariableSelectionTest(SweepTestBase):
    def test_only_design_variables_are_swept(self):
        sweep = self.sweep(_auditor(design_variables=["pore"]))

        self.assertEqual(sweep["swept_parameters"], ["pore"])
        self.assertEqual(sweep["sweep_parameter_source"], "auditor_design_variables")
        changed = {c.get("changed_parameter") for c in sweep["candidates"][1:]}
        self.assertEqual(changed, {"pore"})
        for candidate in sweep["candidates"]:
            self.assertEqual(candidate["parameters"]["heat_flux"], 1e5)
            self.assertEqual(candidate["parameters"]["pulse_duration"], 100.0)

    def test_fallback_without_design_variables_is_noted(self):
        sweep = self.sweep(_auditor(design_variables=None))

        self.assertEqual(sweep["sweep_parameter_source"], "fallback_first_parameters")
        self.assertEqual(sweep["swept_parameters"], ["heat_flux", "pulse_duration", "pore", "thickness"])
        self.assertIn("no design_variables", sweep["note"])

    def test_skipped_sweep_runs_no_variants(self):
        sweep = self.sweep(_auditor(design_variables=["pore"]), skip_reason="generated validation tests did not pass")

        self.assertEqual(sweep["candidate_count"], 1)
        self.assertEqual(sweep["swept_parameters"], [])
        self.assertIn("validation", sweep["skipped"])


class CandidateAcceptanceTest(SweepTestBase):
    def test_success_flag_error_and_hard_constraints_filter_candidates(self):
        auditor = _auditor(
            design_variables=["pore"],
            hard_constraints=[
                "relative_error (grid convergence) < 5%",
                "peak_temperature_K <= 2000 K (material limit)",
                "all temperatures physically plausible",
            ],
        )
        sweep = self.sweep(auditor, base_extra={"flag_small_pores": 1})
        by_label = {c["label"]: c for c in sweep["candidates"]}

        # pore=0.275: exit code 0 but success=false
        self.assertEqual(by_label["pore=0.275"]["status"], "rejected")
        self.assertTrue(any("success" in r for r in by_label["pore=0.275"]["rejection_reasons"]))
        # pore=0.325: peak_temperature_K = 2050 > 2000
        self.assertEqual(by_label["pore=0.325"]["status"], "rejected")
        self.assertTrue(by_label["pore=0.325"]["constraint_violations"])
        # pore=0.725: relative_error 0.2 > 0.05
        self.assertEqual(by_label["pore=0.725"]["status"], "rejected")
        self.assertTrue(any("relative_error" in r for r in by_label["pore=0.725"]["rejection_reasons"]))
        self.assertEqual(by_label["pore=0.55"]["status"], "ok")

        # Best remaining candidate is the baseline itself -> no recommendation.
        self.assertEqual(sweep["best"]["label"], "baseline")
        self.assertIsNone(sweep["recommendation"])
        self.assertEqual(
            [item["constraint"] for item in sweep["hard_constraints_not_evaluated"]],
            ["all temperatures physically plausible"],
        )

    def test_better_variant_is_only_a_recommendation(self):
        sweep = self.sweep(_auditor(design_variables=["pore"], hard_constraints=[]))

        self.assertEqual(sweep["headline_source"], "baseline")
        self.assertEqual(sweep["baseline"]["performance_gain_pct"], 50.0)
        self.assertEqual(sweep["best"]["label"], "pore=0.275")
        self.assertEqual(sweep["recommendation"]["parameters"]["pore"], 0.275)
        self.assertAlmostEqual(sweep["recommendation"]["performance_gain_pct"], 72.5)
        self.assertEqual(sweep["recommendation"]["baseline_performance_gain_pct"], 50.0)

    def test_parse_hard_constraint(self):
        self.assertEqual(parse_hard_constraint("relative_error (energy balance) < 1%"), ("relative_error", "<", 0.01))
        self.assertEqual(parse_hard_constraint("performance_gain_pct >= 5 %"), ("performance_gain_pct", ">=", 5.0))
        self.assertEqual(parse_hard_constraint("T_max <= 2000 K"), ("T_max", "<=", 2000.0))
        self.assertEqual(parse_hard_constraint("thickness ≥ 1e-3 m"), ("thickness", ">=", 0.001))
        self.assertIsNone(parse_hard_constraint("peak surface temperature <= 2000 K"))
        self.assertIsNone(parse_hard_constraint("x <= 5 * y"))
        self.assertIsNone(parse_hard_constraint("finite numeric outputs"))


class NoiseGuardTest(SweepTestBase):
    def test_noise_level_metrics_are_flagged(self):
        tiny = {"primary_metric_value": 1e-9, "reference_metric_value": 3e-9, "performance_gain_pct": 66.7,
                "solution_primary": [300.0, 1800.0], "solution_reference": [300.0, 1800.0]}
        self.assertIsNotNone(noise_level_warning(tiny))
        normal = dict(tiny, primary_metric_value=0.002, reference_metric_value=0.2)
        self.assertIsNone(noise_level_warning(normal))

    def test_noise_level_variants_are_not_counted_as_gain(self):
        sweep = self.sweep(_auditor(design_variables=["pore"], hard_constraints=[]), base_extra={"noise": 1})

        variants = sweep["candidates"][1:]
        self.assertTrue(variants)
        for candidate in variants:
            self.assertTrue(candidate["noise_level"])
            self.assertEqual(candidate["status"], "rejected")
        self.assertIsNone(sweep["recommendation"])


class SolveDynamicScriptTest(unittest.TestCase):
    """Full _solve_dynamic_script path with mocked model answers and real sandboxed runs."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {
            "VECTORNAUT_DATA_DIR": os.path.join(self._tmp.name, "data"),
            "VECTORNAUT_PLOTS_DIR": os.path.join(self._tmp.name, "plots"),
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self._tmp.cleanup()

    def solve(self, auditor, tests_code=PASSING_TESTS):
        from vectornaut.solver_dispatcher import _solve_dynamic_script

        solver_response = MagicMock()
        solver_response.parsed = GeneratedScriptResponse(explanation="toy", code=SOLVER)
        tests_response = MagicMock()
        tests_response.parsed = GeneratedTestScriptResponse(explanation="toy tests", code=tests_code)
        client = MagicMock()
        client.models.generate_content.side_effect = [solver_response, tests_response]
        with mock.patch("vectornaut.script_generator.get_client", return_value=client), \
                mock.patch("vectornaut.test_generator.get_client", return_value=client):
            return _solve_dynamic_script(_miner(), auditor, epochs=1)

    def test_headline_numbers_match_reported_parameters(self):
        auditor = _auditor(design_variables=["pore"], hard_constraints=[])
        sim = self.solve(auditor)

        self.assertTrue(sim.validation_passed)
        with open(sim.params_json_path, encoding="utf-8") as f:
            params_used = json.load(f)
        self.assertEqual(params_used["pore"], auditor.audited_parameters_dict["pore"])
        # Headline = baseline run with the audited parameters (pore 0.5 -> 50 %).
        self.assertAlmostEqual(sim.performance_gain_pct, 50.0)
        self.assertAlmostEqual(sim.primary_metric_value, 5.0)
        self.assertAlmostEqual(sim.reference_metric_value, 10.0)
        sweep = sim.parameter_sweep
        self.assertEqual(sweep["baseline"]["parameters"], params_used)
        self.assertEqual(sweep["baseline"]["performance_gain_pct"], sim.performance_gain_pct)
        # The better sweep variant is reported separately and not adopted.
        self.assertEqual(sweep["recommendation"]["parameters"]["pore"], 0.275)
        self.assertNotEqual(sweep["best"]["performance_gain_pct"], sim.performance_gain_pct)
        self.assertTrue(os.path.exists(os.path.join(self._tmp.name, "plots", os.path.basename(sim.custom_plot_url))))

    def test_failed_validation_skips_the_sweep(self):
        sim = self.solve(_auditor(design_variables=["pore"], hard_constraints=[]), tests_code=FAILING_TESTS)

        self.assertFalse(sim.validation_passed)
        self.assertEqual(sim.parameter_sweep["candidate_count"], 1)
        self.assertIsNotNone(sim.parameter_sweep["skipped"])
        self.assertIsNone(sim.parameter_sweep["recommendation"])

    def test_noise_level_baseline_is_not_counted_and_warned(self):
        auditor = _auditor(design_variables=["pore"], hard_constraints=[], extra_params={"noise": 1.0})
        sim = self.solve(auditor)

        self.assertEqual(sim.performance_gain_pct, 0.0)
        self.assertAlmostEqual(sim.parameter_sweep["raw_performance_gain_pct"], 50.0)
        codes = [w["code"] for w in sim.parameter_sweep["warnings"]]
        self.assertIn("noise_level_metric", codes)
        self.assertIn("numerical-noise level", sim.validation_report)

        validation = validate_run_output(_miner(), auditor, sim)
        check = next(c for c in validation.checks if c.name == "dynamic_script_noise_level_metric")
        self.assertFalse(check.passed)
        self.assertEqual(check.severity, "warning")


if __name__ == "__main__":
    unittest.main()
