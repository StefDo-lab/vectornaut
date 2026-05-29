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
    def __init__(self, solver_method="analytical"):
        self.solver_method = solver_method

    def mock_audit_design(self, miner_output, override_parameters=None, user_query=None):
        return AuditorOutput(
            audit_passed=True,
            audit_notes="ok",
            audited_parameters=[],
            dimensionless_numbers=[],
            simulation_coefficient=0.001,
            solver_method=self.solver_method,
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


def invalid_but_runtime_complete_solver(miner_output, auditor_output, epochs):
    return Dumpable(
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


def fallback_sensitive_solver(miner_output, auditor_output, epochs):
    method = auditor_output.solver_method
    relative_error = 1.5 if method == "pinn" else 0.0
    return Dumpable(
        solver_method=method,
        epochs_trained=10 if method == "pinn" else 0,
        final_loss=0.2 if method == "pinn" else 0.0,
        loss_history=[1.0, 0.2] if method == "pinn" else [],
        performance_gain_pct=10.0,
        relative_error=relative_error,
        sample_points=[0.0, 1.0],
        solution_primary=[0.0, 1.0],
        solution_reference=[0.0, 1.0],
        primary_metric_value=1.0,
        reference_metric_value=1.0,
    )


def worse_fallback_solver(miner_output, auditor_output, epochs):
    if auditor_output.solver_method == "pinn":
        return Dumpable(
            solver_method="pinn",
            epochs_trained=10,
            final_loss=0.2,
            loss_history=[1.0, 0.2],
            performance_gain_pct=10.0,
            relative_error=1.5,
            sample_points=[0.0, 1.0],
            solution_primary=[0.0, 1.0],
            solution_reference=[0.0, 1.0],
            primary_metric_value=1.0,
            reference_metric_value=1.0,
        )
    return Dumpable(
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


def dynamic_metadata_solver(miner_output, auditor_output, epochs):
    method = auditor_output.solver_method
    data = {
        "solver_method": method,
        "epochs_trained": 0,
        "final_loss": 0.0,
        "loss_history": [],
        "performance_gain_pct": 12.0,
        "relative_error": 0.0,
        "sample_points": [0.0, 1.0],
        "solution_primary": [0.0, 1.0],
        "solution_reference": [0.0, 1.0],
        "primary_metric_value": 1.0,
        "reference_metric_value": 1.0,
    }
    if method == "dynamic_script":
        data.update({
            "validation_passed": True,
            "validation_tests": [{"name": "test_nominal_run", "passed": True, "message": ""}],
            "script_path": "generated_scripts/solver_mock.py",
            "params_json_path": "generated_scripts/params_mock.json",
            "test_script_path": "generated_tests/test_solver_mock.py",
            "test_output_path": "generated_tests/results_test_mock.json",
            "execution_mode": "generated_python_subprocess",
        })
    return Dumpable(**data)


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
        self.assertEqual(output["validation"]["status"], "pass")
        self.assertEqual(output["validation"]["recommended_action"], "accept")
        self.assertEqual(len(output["optimization_history"]), 1)

    def test_pipeline_rejects_validator_failures(self):
        runner = PipelineRunner(
            miner=FakeMiner(),
            formulator=FakeFormulator(),
            auditor=FakeAuditor(),
            optimizer=object(),
            synthesizer=FakeSynthesizer(),
            solver=invalid_but_runtime_complete_solver,
        )

        with self.assertRaises(ValueError) as ctx:
            runner.run(PipelineRunRequest(
                query="test",
                is_mock=True,
                epochs=5,
                max_optimization_rounds=1,
            ))

        self.assertIn("All bionic concepts failed validation", str(ctx.exception))

    def test_pipeline_falls_back_when_validator_requests_solver_rerun(self):
        runner = PipelineRunner(
            miner=FakeMiner(),
            formulator=FakeFormulator(),
            auditor=FakeAuditor(solver_method="pinn"),
            optimizer=object(),
            synthesizer=FakeSynthesizer(),
            solver=fallback_sensitive_solver,
        )

        output = runner.run(PipelineRunRequest(
            query="test",
            is_mock=True,
            epochs=5,
            max_optimization_rounds=1,
        ))

        self.assertEqual(output["simulator"]["solver_method"], "analytical")
        self.assertEqual(output["auditor"]["solver_method"], "analytical")
        self.assertEqual(output["validation"]["status"], "pass")
        self.assertEqual(output["validation"]["recommended_action"], "accept")
        self.assertEqual(output["optimization_history"][0]["solver_fallbacks"][0]["solver_method"], "analytical")

    def test_pipeline_keeps_original_warning_when_fallback_is_worse(self):
        runner = PipelineRunner(
            miner=FakeMiner(),
            formulator=FakeFormulator(),
            auditor=FakeAuditor(solver_method="pinn"),
            optimizer=object(),
            synthesizer=FakeSynthesizer(),
            solver=worse_fallback_solver,
        )

        output = runner.run(PipelineRunRequest(
            query="test",
            is_mock=True,
            epochs=5,
            max_optimization_rounds=1,
        ))

        self.assertEqual(output["simulator"]["solver_method"], "pinn")
        self.assertEqual(output["auditor"]["solver_method"], "pinn")
        self.assertEqual(output["validation"]["status"], "warn")
        self.assertEqual(output["validation"]["recommended_action"], "rerun_solver")
        self.assertEqual(output["optimization_history"][0]["solver_fallbacks"][0]["status"], "fail")

    def test_solver_compare_can_run_dynamic_script_path(self):
        runner = PipelineRunner(
            miner=FakeMiner(),
            formulator=FakeFormulator(),
            auditor=FakeAuditor(),
            optimizer=object(),
            synthesizer=FakeSynthesizer(),
            solver=dynamic_metadata_solver,
        )
        current_run = runner.run(PipelineRunRequest(
            query="test",
            is_mock=True,
            epochs=5,
            max_optimization_rounds=1,
        ))

        comparison = runner.compare_solvers(
            current_run=current_run,
            methods=["dynamic_script"],
            epochs=5,
            is_mock=True,
        )

        self.assertTrue(comparison["success"])
        self.assertEqual(comparison["best_solver"], "dynamic_script")
        self.assertEqual(comparison["results"][0]["status"], "pass")
        self.assertEqual(comparison["results"][0]["recommended_action"], "accept")


if __name__ == "__main__":
    unittest.main()
