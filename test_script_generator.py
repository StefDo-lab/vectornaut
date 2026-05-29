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
from vectornaut.script_generator import ScriptGenerator, GeneratedScriptResponse

class TestScriptGenerator(unittest.TestCase):
    def setUp(self):
        # Create minimal mock inputs
        self.miner_output = MinerOutput(
            design_name="Mock Wing",
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

    def test_self_correction_loop(self):
        print("\n[*] Starting Unit Test for ScriptGenerator Self-Correction Loop...")
        
        # First mock response (will fail/crash due to deliberate syntax error/value error):
        bad_code = """
import argparse
import json
import sys

# Deliberately crash with ValueError to trigger self-correction
raise ValueError("Intentional crash for testing self-correction loop")
"""

        # Second mock response (will succeed):
        good_code = """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--params', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--plot', required=True)
args = parser.parse_args()

# Read parameters
with open(args.params, 'r') as f:
    params = json.load(f)

# Write output results
results = {
    "success": True,
    "performance_gain_pct": 45.2,
    "relative_error": 0.01,
    "sample_points": [0.0, 0.5, 1.0],
    "solution_primary": [1.0, 1.5, 2.0],
    "solution_reference": [1.0, 1.25, 1.5],
    "primary_metric_value": 2.0,
    "reference_metric_value": 1.5
}
with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(results, f)

# Write a tiny PNG signature payload; the generator only requires that the plot file exists.
with open(args.plot, 'wb') as f:
    f.write(b'\\x89PNG\\r\\n\\x1a\\n')
"""

        mock_response_1 = MagicMock()
        mock_response_1.parsed = GeneratedScriptResponse(
            explanation="This code is faulty on purpose.",
            code=bad_code
        )

        mock_response_2 = MagicMock()
        mock_response_2.parsed = GeneratedScriptResponse(
            explanation="This code is corrected and will succeed.",
            code=good_code
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [mock_response_1, mock_response_2]

        generator = ScriptGenerator(client=mock_client)
        
        # Paths for test run
        test_plot_path = os.path.abspath("static/plots/test_plot.png")
        
        # Run generator
        results = generator.generate_and_execute(
            miner_output=self.miner_output,
            auditor_output=self.auditor_output,
            epochs=10,
            plot_png_path=test_plot_path
        )
        
        print("[+] Unit Test Results:")
        print(results)

        # Assertions
        self.assertTrue(results["success"])
        self.assertEqual(results["performance_gain_pct"], 45.2)
        self.assertEqual(results["relative_error"], 0.01)
        self.assertTrue(os.path.exists(test_plot_path))
        self.assertTrue(os.path.exists(results["script_path"]))
        
        # Verify that generate_content was called exactly twice (meaning correction loop succeeded)
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        print("[+] Self-correction unit test passed successfully!")

        # Clean up files created
        if os.path.exists(test_plot_path):
            os.remove(test_plot_path)
        if os.path.exists(results["script_path"]):
            os.remove(results["script_path"])
        # Clean up any params and results files in generated_scripts/
        for root, dirs, files in os.walk("generated_scripts"):
            for file in files:
                if "mock_wing" in file.lower() or "solver_mock_wing" in file.lower():
                    try:
                        os.remove(os.path.join(root, file))
                    except Exception:
                        pass

if __name__ == "__main__":
    unittest.main()
