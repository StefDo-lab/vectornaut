# -*- coding: utf-8 -*-
import unittest
import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from vectornaut.config import MinerConceptOutput, ParameterProposal
from vectornaut.formulator import ModelFormulator

class TestFormulatorUnit(unittest.TestCase):
    def setUp(self):
        self.formulator = ModelFormulator()
        self.concept_plastron = MinerConceptOutput(
            design_name="PlastronGlide Hydrophobic Ski Base",
            inspiration_source="Collembola cuticle (springtail)",
            domain="Fluid Dynamics",
            physical_mechanism="The hierarchical micro- and nanostructures of the Collembola cuticle trap a persistent layer of air...",
            parameters=[
                ParameterProposal(
                    name="film_thickness",
                    value=0.00001,
                    min_bound=0.000001,
                    max_bound=0.0001,
                    justification="Thickness of the liquid meltwater film"
                ),
                ParameterProposal(
                    name="slip_length",
                    value=0.00002,
                    min_bound=0.000001,
                    max_bound=0.0001,
                    justification="Effective Navier slip length"
                ),
                ParameterProposal(
                    name="viscosity",
                    value=0.00179,
                    min_bound=0.001,
                    max_bound=0.002,
                    justification="Dynamic viscosity of water"
                ),
                ParameterProposal(
                    name="pressure_gradient",
                    value=-10000.0,
                    min_bound=-1000000.0,
                    max_bound=0.0,
                    justification="Pressure gradient driving fluid flow"
                ),
                ParameterProposal(
                    name="ski_velocity",
                    value=10.0,
                    min_bound=0.0,
                    max_bound=50.0,
                    justification="Relative velocity of the ski base"
                )
            ],
            svg_schematic="<svg></svg>"
        )

        self.concept_riblet = MinerConceptOutput(
            design_name="Shark-Skin Inspired Riblet Foil",
            inspiration_source="Galeocerdo cuvier (Tiger Shark)",
            domain="Fluid Dynamics",
            physical_mechanism="Micro-grooves aligned with flow direction...",
            parameters=[
                ParameterProposal(
                    name="riblet_height",
                    value=0.015,
                    min_bound=0.001,
                    max_bound=0.1,
                    justification="Optimal height"
                ),
                ParameterProposal(
                    name="riblet_spacing",
                    value=0.03,
                    min_bound=0.005,
                    max_bound=0.2,
                    justification="Optimizes vortex spacing control"
                ),
                ParameterProposal(
                    name="viscosity",
                    value=0.001,
                    min_bound=0.0001,
                    max_bound=0.01,
                    justification="Water viscosity"
                ),
                ParameterProposal(
                    name="free_stream_velocity",
                    value=1.5,
                    min_bound=0.1,
                    max_bound=5.0,
                    justification="Operating velocity"
                ),
                ParameterProposal(
                    name="pressure_gradient",
                    value=2.0,
                    min_bound=0.0,
                    max_bound=10.0,
                    justification="Simulates external pressure"
                )
            ],
            svg_schematic="<svg></svg>"
        )

    def test_mock_formulate_plastron(self):
        output = self.formulator.mock_formulate_model("reduce drag on ski", self.concept_plastron)
        self.assertEqual(output.design_name, self.concept_plastron.design_name)
        self.assertIn("pressure_gradient", output.governing_equation)
        self.assertTrue(any("u(0)" in bc for bc in output.boundary_conditions))
        self.assertEqual(output.independent_variables, ["y"])
        self.assertEqual(output.dependent_variables, ["u"])

    def test_mock_formulate_riblet(self):
        output = self.formulator.mock_formulate_model("drag reduction foil", self.concept_riblet)
        self.assertEqual(output.design_name, self.concept_riblet.design_name)
        self.assertIn("pressure_gradient", output.governing_equation)
        self.assertTrue(any("slippage_coefficient" in bc for bc in output.boundary_conditions))
        self.assertEqual(output.independent_variables, ["y"])
        self.assertEqual(output.dependent_variables, ["u"])

    def test_live_formulate_if_api_key_available(self):
        if os.environ.get("RUN_LIVE_TESTS") != "1":
            self.skipTest("Skipping live model formulation test: set RUN_LIVE_TESTS=1 to enable.")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("Skipping live model formulation test: GEMINI_API_KEY not found.")
            
        # Run live model formulation with low thinking to speed up testing
        output = self.formulator.formulate_model(
            "reduce drag on ski", 
            self.concept_plastron, 
            thinking_level="low"
        )
        self.assertIsNotNone(output.governing_equation)
        self.assertTrue(len(output.boundary_conditions) >= 2)
        self.assertTrue(len(output.independent_variables) >= 1)
        self.assertTrue(len(output.dependent_variables) >= 1)

if __name__ == "__main__":
    unittest.main()
