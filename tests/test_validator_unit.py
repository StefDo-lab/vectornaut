# -*- coding: utf-8 -*-
import unittest

from vectornaut.validator import validate_run_data


def valid_run():
    return {
        "miner": {
            "design_name": "Validated Concept",
            "governing_equation": "d2u_dy2 = 0",
            "boundary_conditions": ["u(0) = 0", "u(1) = 1"],
            "dependent_variables": ["u"],
        },
        "auditor": {"audit_passed": True},
        "simulator": {
            "solver_method": "analytical",
            "relative_error": 0.01,
            "performance_gain_pct": 12.0,
            "sample_points": [0.0, 0.5, 1.0],
            "solution_primary": [0.0, 0.5, 1.0],
            "solution_reference": [0.0, 0.49, 0.98],
            "primary_metric_value": 0.8,
            "reference_metric_value": 1.0,
        },
        "optimization_history": [],
    }


class ValidatorUnitTest(unittest.TestCase):
    def test_validator_passes_stable_run(self):
        result = validate_run_data(valid_run(), {"max_relative_error": 0.1})

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.recommended_action, "accept")
        self.assertGreaterEqual(result.score, 0.8)

    def test_validator_warns_on_high_relative_error(self):
        run = valid_run()
        run["simulator"]["relative_error"] = 1.5

        result = validate_run_data(run, {"max_relative_error": 0.1})

        self.assertEqual(result.status, "warn")
        self.assertEqual(result.recommended_action, "rerun_solver")
        failed = {check.name for check in result.checks if not check.passed}
        self.assertIn("numeric_relative_error", failed)

    def test_validator_fails_on_broken_solution_arrays(self):
        run = valid_run()
        run["simulator"]["solution_reference"] = []

        result = validate_run_data(run)

        self.assertEqual(result.status, "fail")
        failed = {check.name for check in result.checks if not check.passed}
        self.assertIn("schema_aligned_solution_arrays", failed)

    def test_validator_checks_simple_derivative_boundary_conditions(self):
        run = valid_run()
        run["miner"]["boundary_conditions"] = ["u(0) = 0", "u'(0) = 1"]
        run["simulator"]["sample_points"] = [0.0, 0.5, 1.0]
        run["simulator"]["solution_primary"] = [0.0, 0.5, 1.0]
        run["simulator"]["solution_reference"] = [0.0, 0.5, 1.0]

        result = validate_run_data(run, {"derivative_boundary_tolerance": 0.05})

        self.assertEqual(result.status, "pass")
        passed = {check.name for check in result.checks if check.passed}
        self.assertIn("physics_derivative_boundary_conditions", passed)

    def test_validator_warns_on_bad_derivative_boundary_conditions(self):
        run = valid_run()
        run["miner"]["boundary_conditions"] = ["u(0) = 0", "du_dy(0) = 2"]
        run["miner"]["independent_variables"] = ["y"]
        run["simulator"]["sample_points"] = [0.0, 0.5, 1.0]
        run["simulator"]["solution_primary"] = [0.0, 0.5, 1.0]
        run["simulator"]["solution_reference"] = [0.0, 0.5, 1.0]

        result = validate_run_data(run, {"derivative_boundary_tolerance": 0.05})

        self.assertEqual(result.status, "warn")
        failed = {check.name for check in result.checks if not check.passed}
        self.assertIn("physics_derivative_boundary_conditions", failed)

    def test_validator_enforces_objective_metric_contract(self):
        run = valid_run()
        run["auditor"]["objective_metric"] = {
            "objective_name": "Damage Reduction",
            "score_field": "performance_gain_pct",
            "direction": "maximize",
            "primary_metric": "Damage Risk",
            "reference_metric": "Reference Damage Risk",
            "lower_is_better": True,
            "acceptance_threshold": 0.0,
            "hard_constraints": ["relative_error <= 1.0"],
        }
        run["simulator"]["performance_gain_pct"] = -12.0

        result = validate_run_data(run)

        self.assertEqual(result.status, "warn")
        failed = {check.name for check in result.checks if not check.passed}
        self.assertIn("objective_metric_contract", failed)


if __name__ == "__main__":
    unittest.main()
