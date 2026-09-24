# -*- coding: utf-8 -*-
"""Offline tests for the per-stage model / thinking level configuration."""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from google.genai import types

from vectornaut.config import (
    DEFAULT_MODEL_NAME,
    MODEL_STAGES,
    AuditedParameter,
    AuditorOutput,
    DimensionlessNumber,
    MinerConceptOutput,
    MinerOutput,
    ParameterProposal,
    get_model_name,
    get_thinking_config,
    get_thinking_level,
)


def _clean_env():
    """Patches os.environ without any VECTORNAUT_MODEL* / VECTORNAUT_THINKING* variables."""
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    for key in list(os.environ):
        if key.startswith("VECTORNAUT_MODEL") or key.startswith("VECTORNAUT_THINKING"):
            del os.environ[key]
    return patcher


def _mock_client(parsed):
    client = mock.MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(parsed=parsed)
    return client


def _call_kwargs(client):
    client.models.generate_content.assert_called_once()
    return client.models.generate_content.call_args.kwargs


def _concept():
    return MinerConceptOutput(
        design_name="Testkonzept Haifischhaut",
        inspiration_source="Galeocerdo cuvier",
        domain="Fluid Dynamics",
        physical_mechanism="Riblets reduzieren den Reibungswiderstand.",
        parameters=[
            ParameterProposal(name="viscosity", value=0.001, min_bound=0.0001, max_bound=0.01, justification="Wasser"),
        ],
        svg_schematic="<svg></svg>",
    )


def _miner_output():
    return MinerOutput(
        design_name="Testkonzept Haifischhaut",
        inspiration_source="Galeocerdo cuvier",
        domain="Fluid Dynamics",
        physical_mechanism="Riblets reduzieren den Reibungswiderstand.",
        parameters=[
            ParameterProposal(name="viscosity", value=0.001, min_bound=0.0001, max_bound=0.01, justification="Wasser"),
        ],
        governing_equation="d2u_dy2 = -1 / viscosity",
        boundary_conditions=["u(0) = 0", "u(1) = 1"],
        independent_variables=["y"],
        dependent_variables=["u"],
        svg_schematic="<svg></svg>",
    )


def _auditor_output():
    return AuditorOutput(
        audit_passed=True,
        audit_notes="ok",
        audited_parameters=[AuditedParameter(name="viscosity", value=0.001)],
        dimensionless_numbers=[DimensionlessNumber(name="ReynoldsNumber", value=100.0)],
        simulation_coefficient=0.001,
        solver_method="analytical",
        ui_metadata={
            "domain_name": "Fluid Dynamics",
            "independent_var": {"label": "Channel Height", "unit": "m"},
            "dependent_var": {"label": "Flow Velocity", "unit": "m/s"},
            "primary_metric": {"label": "Wall Shear Stress"},
            "reference_metric": {"label": "Analytical Wall Shear Stress"},
            "performance_gain": {"label": "Drag Reduction Efficiency"},
        },
    )


class TestModelResolution(unittest.TestCase):
    def setUp(self):
        self.addCleanup(_clean_env().stop)

    def test_default_model_for_every_stage(self):
        for stage in MODEL_STAGES:
            self.assertEqual(get_model_name(stage), DEFAULT_MODEL_NAME)
        self.assertEqual(DEFAULT_MODEL_NAME, "gemini-3.5-flash")

    def test_global_model_overrides_default(self):
        os.environ["VECTORNAUT_MODEL"] = "global-model"
        self.assertEqual(get_model_name("miner"), "global-model")
        self.assertEqual(get_model_name("chat"), "global-model")

    def test_stage_model_overrides_global(self):
        os.environ["VECTORNAUT_MODEL"] = "global-model"
        os.environ["VECTORNAUT_MODEL_AUDITOR"] = "auditor-model"
        os.environ["VECTORNAUT_MODEL_SCRIPT_GENERATOR"] = "script-model"
        self.assertEqual(get_model_name("auditor"), "auditor-model")
        self.assertEqual(get_model_name("script_generator"), "script-model")
        self.assertEqual(get_model_name("miner"), "global-model")

    def test_blank_values_are_ignored(self):
        os.environ["VECTORNAUT_MODEL"] = "   "
        os.environ["VECTORNAUT_MODEL_MINER"] = ""
        self.assertEqual(get_model_name("miner"), DEFAULT_MODEL_NAME)

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            get_model_name("minr")
        with self.assertRaises(ValueError):
            get_thinking_level("minr", "medium")


class TestThinkingResolution(unittest.TestCase):
    def setUp(self):
        self.addCleanup(_clean_env().stop)

    def test_default_is_used_without_env(self):
        self.assertEqual(get_thinking_level("auditor", "high"), "high")
        self.assertIsNone(get_thinking_level("optimizer", None))
        self.assertIsNone(get_thinking_config("optimizer"))

    def test_env_overrides_default_case_insensitive(self):
        os.environ["VECTORNAUT_THINKING_AUDITOR"] = " LOW "
        self.assertEqual(get_thinking_level("auditor", "high"), "low")
        self.assertEqual(get_thinking_config("auditor", "high").thinking_level, types.ThinkingLevel.LOW)
        # Other stages are unaffected
        self.assertEqual(get_thinking_level("miner", "medium"), "medium")

    def test_invalid_env_falls_back_to_default(self):
        os.environ["VECTORNAUT_THINKING_AUDITOR"] = "extreme"
        os.environ["VECTORNAUT_THINKING_OPTIMIZER"] = "0"
        with mock.patch("builtins.print"):
            self.assertEqual(get_thinking_level("auditor", "high"), "high")
            self.assertIsNone(get_thinking_level("optimizer", None))
            self.assertIsNone(get_thinking_config("optimizer"))

    def test_env_can_enable_thinking_for_stage_without_default(self):
        os.environ["VECTORNAUT_THINKING_SYNTHESIZER"] = "medium"
        self.assertEqual(get_thinking_config("synthesizer").thinking_level, types.ThinkingLevel.MEDIUM)

    def test_explicit_override_wins_over_env(self):
        os.environ["VECTORNAUT_THINKING_FORMULATOR"] = "high"
        config = get_thinking_config("formulator", default="medium", override="low")
        self.assertEqual(config.thinking_level, types.ThinkingLevel.LOW)


class TestStagesUseConfiguredModel(unittest.TestCase):
    def setUp(self):
        self.addCleanup(_clean_env().stop)

    def test_miner_defaults_unchanged(self):
        from vectornaut.miner import Miner

        client = _mock_client(_concept())
        Miner(client=client).mine_design("reibungsarme Oberfläche")

        kwargs = _call_kwargs(client)
        self.assertEqual(kwargs["model"], "gemini-3.5-flash")
        self.assertEqual(kwargs["config"].thinking_config.thinking_level, types.ThinkingLevel.MEDIUM)

    def test_miner_uses_env_model_and_thinking(self):
        from vectornaut.miner import Miner

        os.environ["VECTORNAUT_MODEL"] = "global-model"
        os.environ["VECTORNAUT_MODEL_MINER"] = "miner-model"
        os.environ["VECTORNAUT_THINKING_MINER"] = "low"
        client = _mock_client(_concept())
        Miner(client=client).mine_design("reibungsarme Oberfläche")

        kwargs = _call_kwargs(client)
        self.assertEqual(kwargs["model"], "miner-model")
        self.assertEqual(kwargs["config"].thinking_config.thinking_level, types.ThinkingLevel.LOW)

    def test_auditor_uses_env_model_and_default_thinking(self):
        from vectornaut.auditor import Auditor

        os.environ["VECTORNAUT_MODEL_AUDITOR"] = "auditor-model"
        client = _mock_client(_auditor_output())
        result = Auditor(client=client).audit_design(_miner_output(), user_query="Test")

        kwargs = _call_kwargs(client)
        self.assertEqual(kwargs["model"], "auditor-model")
        self.assertEqual(kwargs["config"].thinking_config.thinking_level, types.ThinkingLevel.HIGH)
        self.assertTrue(result.audit_passed)

    def test_formulator_explicit_thinking_level_wins(self):
        from vectornaut.config import ModelFormulation
        from vectornaut.formulator import ModelFormulator

        os.environ["VECTORNAUT_MODEL"] = "global-model"
        os.environ["VECTORNAUT_THINKING_FORMULATOR"] = "high"
        formulation = ModelFormulation(
            governing_equation="d2u_dy2 = -1 / viscosity",
            boundary_conditions=["u(0) = 0", "u(1) = 1"],
            independent_variables=["y"],
            dependent_variables=["u"],
        )

        with mock.patch("builtins.print"):
            env_client = _mock_client(formulation)
            ModelFormulator(client=env_client).formulate_model("q", _concept())
            ctor_client = _mock_client(formulation)
            ModelFormulator(client=ctor_client, thinking_level="low").formulate_model("q", _concept())
            call_client = _mock_client(formulation)
            ModelFormulator(client=call_client, thinking_level="low").formulate_model("q", _concept(), thinking_level="medium")

        self.assertEqual(_call_kwargs(env_client)["model"], "global-model")
        self.assertEqual(_call_kwargs(env_client)["config"].thinking_config.thinking_level, types.ThinkingLevel.HIGH)
        self.assertEqual(_call_kwargs(ctor_client)["config"].thinking_config.thinking_level, types.ThinkingLevel.LOW)
        self.assertEqual(_call_kwargs(call_client)["config"].thinking_config.thinking_level, types.ThinkingLevel.MEDIUM)

    def test_optimizer_sends_no_thinking_config_by_default(self):
        from vectornaut.optimizer import Optimizer, OptimizerDecision

        os.environ["VECTORNAUT_MODEL_OPTIMIZER"] = "optimizer-model"
        client = _mock_client(OptimizerDecision(continue_optimization=False, reasoning="Konvergiert.", adjustments=[]))
        history = [{
            "round": 1,
            "parameters": {"viscosity": 0.001},
            "simulator": {"performance_gain_pct": 5.0, "relative_error": 0.0},
        }]
        Optimizer(client=client).optimize(_miner_output(), history)

        kwargs = _call_kwargs(client)
        self.assertEqual(kwargs["model"], "optimizer-model")
        self.assertIsNone(kwargs["config"].thinking_config)

    def test_model_is_resolved_at_call_time(self):
        from vectornaut.synthesizer import SynthesisReport, Synthesizer

        report = SynthesisReport(
            executive_summary="a", pros_and_cons="b", mechanical_limits="c", manufacturing_methods="d",
            cost_estimation="e", validation_experiments="f", industry_partners="g",
        )
        client = _mock_client(report)
        synthesizer = Synthesizer(client=client)
        sim = SimpleNamespace(solver_method="analytical", performance_gain_pct=1.0, relative_error=0.0)

        os.environ["VECTORNAUT_MODEL_SYNTHESIZER"] = "first-model"
        synthesizer.generate_synthesis(_miner_output(), _auditor_output(), sim, "q")
        os.environ["VECTORNAUT_MODEL_SYNTHESIZER"] = "second-model"
        synthesizer.generate_synthesis(_miner_output(), _auditor_output(), sim, "q")

        models = [call.kwargs["model"] for call in client.models.generate_content.call_args_list]
        self.assertEqual(models, ["first-model", "second-model"])


class _Dumpable(SimpleNamespace):
    def model_dump(self):
        return dict(self.__dict__)


class _LiveMiner:
    def mine_design(self, query, failed_concepts=None):
        return _concept()


class _LiveFormulator:
    def formulate_model(self, query, concept):
        return _miner_output()


class _LiveAuditor:
    def audit_design(self, miner_output, override_parameters=None, user_query=None):
        return _auditor_output()

    def mock_audit_design(self, miner_output, override_parameters=None, user_query=None):
        return _auditor_output()


class _LiveOptimizer:
    def optimize(self, miner_output, history):
        return _Dumpable(continue_optimization=False, reasoning="Konvergiert.", adjustments=[])


class _LiveSynthesizer:
    def generate_synthesis(self, **kwargs):
        return _Dumpable(executive_summary="ok")

    def mock_generate_synthesis(self):
        return _Dumpable(executive_summary="mock")


class _MockMiner:
    def mock_mine_design(self, query):
        return _concept()


class _MockFormulator:
    def mock_formulate_model(self, query, concept):
        return _miner_output()


class _MockOptimizer:
    def mock_optimize(self, miner_output, history):
        return _Dumpable(continue_optimization=False, reasoning="Konvergiert.", adjustments=[])


def _solver(miner_output, auditor_output, epochs):
    return _Dumpable(
        solver_method="analytical",
        epochs_trained=0,
        final_loss=0.0,
        loss_history=[],
        performance_gain_pct=10.0,
        relative_error=0.0,
        sample_points=[0.0, 1.0],
        solution_primary=[0.0, 1.0],
        solution_reference=[0.0, 1.0],
        primary_metric_value=0.9,
        reference_metric_value=1.0,
        custom_plot_url=None,
        validation_passed=None,
        validation_report=None,
        validation_tests=None,
    )


class TestPipelineRecordsModels(unittest.TestCase):
    def setUp(self):
        self.addCleanup(_clean_env().stop)

    def test_live_run_records_model_per_stage(self):
        from vectornaut.pipeline import PipelineRunRequest, PipelineRunner

        os.environ["GEMINI_API_KEY"] = "test-key-not-used"
        os.environ["VECTORNAUT_MODEL"] = "global-model"
        os.environ["VECTORNAUT_MODEL_AUDITOR"] = "auditor-model"
        runner = PipelineRunner(
            miner=_LiveMiner(),
            formulator=_LiveFormulator(),
            auditor=_LiveAuditor(),
            optimizer=_LiveOptimizer(),
            synthesizer=_LiveSynthesizer(),
            solver=_solver,
        )
        with mock.patch("builtins.print"):
            output = runner.run(PipelineRunRequest(query="Test", is_mock=False, max_optimization_rounds=2))

        self.assertFalse(output["is_mock"])
        self.assertEqual(output["models"], {
            "miner": "global-model",
            "formulator": "global-model",
            "auditor": "auditor-model",
            "optimizer": "global-model",
            "synthesizer": "global-model",
        })

    def test_mock_run_records_no_models(self):
        from vectornaut.pipeline import PipelineRunRequest, PipelineRunner

        runner = PipelineRunner(
            miner=_MockMiner(),
            formulator=_MockFormulator(),
            auditor=_LiveAuditor(),
            optimizer=_MockOptimizer(),
            synthesizer=_LiveSynthesizer(),
            solver=_solver,
        )
        with mock.patch("builtins.print"):
            output = runner.run(PipelineRunRequest(query="Test", is_mock=True, max_optimization_rounds=2))

        self.assertTrue(output["is_mock"])
        self.assertEqual(output["models"], {})


if __name__ == "__main__":
    unittest.main()
