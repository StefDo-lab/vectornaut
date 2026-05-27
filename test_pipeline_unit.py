# -*- coding: utf-8 -*-
import unittest
from types import SimpleNamespace

from vectornaut.pipeline import PipelineRunRequest, PipelineRunner


class Dumpable(SimpleNamespace):
    def model_dump(self):
        return dict(self.__dict__)


class AuditorOutput(Dumpable):
    @property
    def audited_parameters_dict(self):
        return {"slip_length": 0.001}

    @property
    def dimensionless_numbers_dict(self):
        return {"ReynoldsNumber": 1000.0}


class FakeMiner:
    def mock_mine_design(self, query):
        return Dumpable(
            design_name="Mock Concept",
            inspiration_source="Mock Source",
            physical_mechanism="Mock mechanism",
        )


class FakeFormulator:
    def mock_formulate_model(self, query, concept):
        return Dumpable(
            design_name=concept.design_name,
            inspiration_source=concept.inspiration_source,
            domain="Fluid Dynamics",
            physical_mechanism=concept.physical_mechanism,
            parameters=[],
            governing_equation="d2u_dy2 = 0",
            boundary_conditions=["u(0) = 0", "u(1) = 1"],
            independent_variables=["y"],
            dependent_variables=["u"],
            svg_schematic="<svg></svg>",
        )


class FakeAuditor:
    def mock_audit_design(self, miner_output, override_parameters=None, user_query=None):
        return AuditorOutput(
            audit_passed=True,
            audit_notes="ok",
            audited_parameters=[],
            dimensionless_numbers=[],
            simulation_coefficient=0.001,
            solver_method="analytical",
            ui_metadata={},
        )


class FakeSynthesizer:
    def mock_generate_synthesis(self):
        return Dumpable(executive_summary="ok")


def fake_solver(miner_output, auditor_output, epochs):
    return Dumpable(
        solver_method="analytical",
        epochs_trained=0,
        final_loss=0.0,
        loss_history=[],
        performance_gain_pct=10.0,
        relative_error=0.0,
        sample_points=[0.0, 1.0],
        solution_primary=[0.0, 1.0],
        solution_reference=[0.0, 1.0],
        primary_metric_value=1.0,
        reference_metric_value=1.0,
    )


class PipelineUnitTest(unittest.TestCase):
    def test_pipeline_runner_orchestrates_with_injected_dependencies(self):
        runner = PipelineRunner(
            miner=FakeMiner(),
            formulator=FakeFormulator(),
            auditor=FakeAuditor(),
            optimizer=object(),
            synthesizer=FakeSynthesizer(),
            solver=fake_solver,
        )

        output = runner.run(PipelineRunRequest(
            query="test",
            is_mock=True,
            epochs=5,
            max_optimization_rounds=1,
        ))

        self.assertTrue(output["success"])
        self.assertEqual(output["query"], "test")
        self.assertEqual(output["miner"]["design_name"], "Mock Concept")
        self.assertEqual(output["simulator"]["solver_method"], "analytical")
        self.assertEqual(len(output["optimization_history"]), 1)


if __name__ == "__main__":
    unittest.main()
