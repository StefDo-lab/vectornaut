# -*- coding: utf-8 -*-
import unittest
from vectornaut.config import MinerOutput, ParameterProposal
from vectornaut.optimizer import Optimizer

class TestOptimizerUnit(unittest.TestCase):
    def setUp(self):
        self.optimizer = Optimizer()
        self.miner_output = MinerOutput(
            design_name="Test Bionic Design",
            inspiration_source="Nature",
            domain="Fluid Dynamics",
            physical_mechanism="A test mechanism",
            parameters=[
                ParameterProposal(
                    name="param1",
                    value=0.5,
                    min_bound=0.1,
                    max_bound=1.0,
                    justification="Justification 1"
                ),
                ParameterProposal(
                    name="param2",
                    value=20.0,
                    min_bound=5.0,
                    max_bound=50.0,
                    justification="Justification 2"
                )
            ],
            governing_equation="d2u_dy2 = -param1",
            boundary_conditions=["u(0) = 0", "u(1) = param2"],
            independent_variables=["y"],
            dependent_variables=["u"],
            svg_schematic="<svg></svg>"
        )

    def test_mock_optimize_flow(self):
        # Round 1
        history_r1 = [{
            "round": 1,
            "parameters": {"param1": 0.5, "param2": 20.0},
            "simulation_coefficient": 0.5,
            "simulator": {"performance_gain_pct": 12.5, "relative_error": 1e-4, "validation_passed": True}
        }]
        
        decision_r1 = self.optimizer.mock_optimize(self.miner_output, history_r1)
        self.assertTrue(decision_r1.continue_optimization)
        self.assertTrue(len(decision_r1.adjustments) > 0)
        
        # Verify that adjustments are within bounds
        for adj in decision_r1.adjustments:
            prop = next(p for p in self.miner_output.parameters if p.name == adj.name)
            self.assertTrue(prop.min_bound <= adj.value <= prop.max_bound)
            
        # Round 2 (should converge/stop in mock mode)
        history_r2 = history_r1.copy()
        history_r2.append({
            "round": 2,
            "parameters": {adj.name: adj.value for adj in decision_r1.adjustments},
            "simulation_coefficient": 0.6,
            "simulator": {"performance_gain_pct": 15.2, "relative_error": 8e-5, "validation_passed": True}
        })
        
        decision_r2 = self.optimizer.mock_optimize(self.miner_output, history_r2)
        self.assertFalse(decision_r2.continue_optimization)
        self.assertEqual(len(decision_r2.adjustments), 0)

if __name__ == "__main__":
    unittest.main()
