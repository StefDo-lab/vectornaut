# -*- coding: utf-8 -*-
import unittest

from vectornaut.evaluation import evaluate_run_output


class EvaluationUnitTest(unittest.TestCase):
    def test_evaluate_valid_run_output(self):
        run = {
            "miner": {
                "design_name": "Test Design",
                "governing_equation": "d2u_dy2 = 0",
            },
            "auditor": {
                "audit_passed": True,
            },
            "simulator": {
                "solver_method": "analytical",
                "relative_error": 0.01,
                "sample_points": [0.0, 0.5, 1.0],
                "solution_primary": [0.0, 0.5, 1.0],
                "solution_reference": [0.0, 0.49, 0.98],
                "performance_gain_pct": 12.0,
                "primary_metric_value": 0.8,
                "reference_metric_value": 1.0,
            },
            "synthesis": {"executive_summary": "ok"},
            "report_md": "# Report",
        }

        result = evaluate_run_output(run, {"max_relative_error": 0.1})

        self.assertTrue(result["passed"])
        self.assertTrue(result["scores"]["schema_valid"])
        self.assertTrue(result["scores"]["solver_stable"])
        self.assertEqual(result["validator"]["status"], "pass")

    def test_evaluate_rejects_bad_solver_output(self):
        run = {
            "miner": {
                "design_name": "Bad Design",
                "governing_equation": "d2u_dy2 = 0",
            },
            "auditor": {
                "audit_passed": True,
            },
            "simulator": {
                "solver_method": "pinn",
                "relative_error": 5.0,
                "sample_points": [0.0],
                "solution_primary": [1.0],
                "solution_reference": [],
                "primary_metric_value": 1.0,
                "reference_metric_value": 1.0,
            },
        }

        result = evaluate_run_output(run, {"max_relative_error": 0.1, "min_sample_count": 2})

        self.assertFalse(result["passed"])
        failed_names = {check["name"] for check in result["summary"]["failed_errors"]}
        self.assertIn("relative_error_threshold", failed_names)
        self.assertIn("solution_arrays", failed_names)


if __name__ == "__main__":
    unittest.main()
