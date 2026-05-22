# -*- coding: utf-8 -*-
import os
import sys
import unittest
import json
from unittest.mock import MagicMock

# Make sure we can import vectornaut
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vectornaut.config import (
    MinerOutput, AuditorOutput, ParameterProposal, AuditedParameter,
    DimensionlessNumber, UIMetadata, AxisMetadata, MetricMetadata
)
from vectornaut.test_generator import TestScriptGenerator, GeneratedTestScriptResponse

class TestValidationGenerator(unittest.TestCase):
    def setUp(self):
        # Create minimal mock inputs
        self.miner_output = MinerOutput(
            design_name="Mock Wing Test",
            inspiration_source="Mock Organism",
            domain="Fluid Dynamics",
            physical_mechanism="Mock Mechanism",
            parameters=[
                ParameterProposal(
                    name="viscosity",
                    value=0.001,
                    min_bound=0.0001,
                    max_bound=0.01,
                    justification="Test param"
                )
            ],
            governing_equation="d2u_dy2 = -1.0",
            boundary_conditions=["u(0) = 0", "u(1) = 1"],
            independent_variables=["y"],
            dependent_variables=["u"],
            svg_schematic="<svg></svg>"
        )

        self.auditor_output = AuditorOutput(
            audit_passed=True,
            audit_notes="Verification clean",
            audited_parameters=[
                AuditedParameter(name="viscosity", value=0.001)
            ],
            dimensionless_numbers=[
                DimensionlessNumber(name="ReynoldsNumber", value=100.0)
            ],
            simulation_coefficient=0.001,
            solver_method="dynamic_script",
            ui_metadata=UIMetadata(
                domain_name="Fluid Dynamics",
                independent_var=AxisMetadata(label="y", unit="m"),
                dependent_var=AxisMetadata(label="u", unit="m/s"),
                primary_metric=MetricMetadata(label="Primary Flow"),
                reference_metric=MetricMetadata(label="Reference Flow"),
                performance_gain=MetricMetadata(label="Efficiency")
            )
        )

        # Create temporary mock solver and parameter files
        self.solver_path = os.path.abspath("dummy_solver.py")
        self.params_path = os.path.abspath("dummy_params.json")

        with open(self.solver_path, "w", encoding="utf-8") as f:
            f.write("# Dummy solver\n")

        with open(self.params_path, "w", encoding="utf-8") as f:
            json.dump({"viscosity": 0.001}, f)

    def tearDown(self):
        # Clean up files created during setUp
        if os.path.exists(self.solver_path):
            os.remove(self.solver_path)
        if os.path.exists(self.params_path):
            os.remove(self.params_path)

        # Clean up generated tests directory if files exist
        if os.path.exists("generated_tests"):
            for file in os.listdir("generated_tests"):
                if "mock_wing_test" in file.lower():
                    try:
                        os.remove(os.path.join("generated_tests", file))
                    except Exception:
                        pass

    def test_validation_self_correction_loop(self):
        print("\n[*] Starting Unit Test for TestScriptGenerator Self-Correction Loop...")
        
        # First mock response: test script that crashes
        bad_code = """
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--solver')
parser.add_argument('--params')
parser.add_argument('--output')
args = parser.parse_args()

raise ValueError("Intentional crash for test self-correction loop validation")
"""

        # Second mock response: working test script that outputs the expected JSON structure
        good_code = """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--solver')
parser.add_argument('--params')
parser.add_argument('--output')
args = parser.parse_args()

results = {
    "success": True,
    "total_run": 3,
    "total_failures": 0,
    "total_errors": 0,
    "test_results": [
        {"name": "test_nominal_run", "passed": True, "message": ""},
        {"name": "test_parameter_limits_and_safety", "passed": True, "message": ""},
        {"name": "test_physical_invariants", "passed": True, "message": ""}
    ]
}

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(results, f)
"""

        mock_response_1 = MagicMock()
        mock_response_1.parsed = GeneratedTestScriptResponse(
            explanation="Faulty test script to trigger correction.",
            code=bad_code
        )

        mock_response_2 = MagicMock()
        mock_response_2.parsed = GeneratedTestScriptResponse(
            explanation="Corrected test script.",
            code=good_code
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [mock_response_1, mock_response_2]

        generator = TestScriptGenerator(client=mock_client)
        
        # Run generator
        results = generator.generate_and_execute_tests(
            miner_output=self.miner_output,
            auditor_output=self.auditor_output,
            solver_script_path=self.solver_path,
            params_json_path=self.params_path
        )
        
        print("[+] Test Validation Generator Results:")
        print(results)

        # Assertions
        self.assertTrue(results["success"])
        self.assertEqual(results["total_run"], 3)
        self.assertEqual(results["total_failures"], 0)
        self.assertEqual(results["total_errors"], 0)
        self.assertEqual(len(results["test_results"]), 3)
        self.assertTrue(os.path.exists(results["test_script_path"]))
        self.assertTrue(os.path.exists(results["test_output_path"]))
        
        # Verify that generate_content was called exactly twice (correction loop succeeded)
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        print("[+] Self-correction validation unit test passed successfully!")

        # Clean up files created during test
        if os.path.exists(results["test_script_path"]):
            os.remove(results["test_script_path"])
        if os.path.exists(results["test_output_path"]):
            os.remove(results["test_output_path"])

if __name__ == "__main__":
    unittest.main()
