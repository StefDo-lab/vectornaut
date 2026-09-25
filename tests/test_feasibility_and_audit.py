# -*- coding: utf-8 -*-
"""
Physically impossible requests (finding 3), auditor post-processing (finding 7) and
the round-2 auditor prompt (finding 8) from docs/LIVE_PATH_FINDINGS.md.
"""
import contextlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from tests.offline_support import OfflineApiTestCase
from tests.test_pipeline_unit import (
    AuditorOutput as FakeAuditorOutput,
    Dumpable,
    FakeFormulator,
    FakeSynthesizer,
    fake_solver,
)
from vectornaut.auditor import Auditor, compute_reynolds_number, match_ui_domain
from vectornaut.config import (
    AuditedParameter,
    AuditorOutput,
    DimensionlessNumber,
    MinerConceptOutput,
    MinerOutput,
    ParameterProposal,
)
from vectornaut.miner import Miner
from vectornaut.model_eval import ModelConfig, extract_run_facts, run_case, score_run
from vectornaut.pipeline import ConceptsExhaustedError, PipelineRunRequest, PipelineRunner
from vectornaut.validator import check_closed_system_efficiency, validate_run_output

_ORIGINAL_MOCK_MINE = Miner.mock_mine_design

HEATER_QUERY = (
    "Entwickle eine bionische Heizoberfläche, die in einem vollständig geschlossenen, adiabaten System "
    "ohne jede Energiezufuhr dauerhaft mehr Wärme abgibt, als ihr zugeführt wird (Wirkungsgrad über 300 %)."
)
FIRST_LAW = "A closed adiabatic system without energy input cannot release net heat (first law of thermodynamics)."


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Pipeline fakes that count their calls
# ---------------------------------------------------------------------------

class CountingMiner:
    def __init__(self, feasible=True, reason=None):
        self.feasible = feasible
        self.reason = reason
        self.calls = 0

    def mock_mine_design(self, query):
        self.calls += 1
        return Dumpable(
            design_name=f"Concept {self.calls}",
            inspiration_source="Source",
            physical_mechanism="Mechanism",
            request_feasible=self.feasible,
            infeasibility_reason=self.reason,
        )


class CountingFormulator(FakeFormulator):
    def __init__(self):
        self.calls = 0

    def mock_formulate_model(self, query, concept):
        self.calls += 1
        return super().mock_formulate_model(query, concept)


class CountingAuditor:
    def __init__(self, audit_passed=True, request_feasible=True, reason=None):
        self.audit_passed = audit_passed
        self.request_feasible = request_feasible
        self.reason = reason
        self.calls = 0

    def mock_audit_design(self, miner_output, override_parameters=None, user_query=None):
        self.calls += 1
        return FakeAuditorOutput(
            audit_passed=self.audit_passed,
            audit_notes="audit notes",
            request_feasible=self.request_feasible,
            infeasibility_reason=self.reason,
            audited_parameters=[],
            dimensionless_numbers=[],
            simulation_coefficient=0.001,
            solver_method="analytical",
            ui_metadata={},
        )


class CountingSolver:
    def __init__(self):
        self.calls = 0

    def __call__(self, miner_output, auditor_output, epochs):
        self.calls += 1
        return fake_solver(miner_output, auditor_output, epochs)


class FailingOptimizer:
    def mock_optimize(self, miner_output, history):
        raise AssertionError("the optimizer must not run for a rejected request")


def _runner(miner, auditor, formulator=None, solver=None):
    return PipelineRunner(
        miner=miner,
        formulator=formulator or CountingFormulator(),
        auditor=auditor,
        optimizer=FailingOptimizer(),
        synthesizer=FakeSynthesizer(),
        solver=solver or CountingSolver(),
    )


def _request(query="impossible heater", rounds=2):
    return PipelineRunRequest(query=query, is_mock=True, epochs=5, max_optimization_rounds=rounds)


class PipelineRejectionTest(unittest.TestCase):
    def test_miner_infeasible_stops_before_formulation(self):
        miner = CountingMiner(feasible=False, reason=FIRST_LAW)
        formulator = CountingFormulator()
        auditor = CountingAuditor()
        solver = CountingSolver()

        result = _quiet(_runner(miner, auditor, formulator, solver).run, _request())

        self.assertIs(result["success"], False)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["rejection"]["stage"], "miner")
        self.assertEqual(result["rejection"]["reason"], FIRST_LAW)
        self.assertEqual(result["rejection"]["concept_attempt"], 1)
        self.assertEqual(result["miner"]["design_name"], "Concept 1")
        self.assertIsNone(result["auditor"])
        self.assertEqual((miner.calls, formulator.calls, auditor.calls, solver.calls), (1, 0, 0, 0))

    def test_miner_infeasible_without_reason_gets_a_readable_default(self):
        result = _quiet(_runner(CountingMiner(feasible=False), CountingAuditor()).run, _request())
        self.assertEqual(result["status"], "rejected")
        self.assertIn("physically infeasible", result["rejection"]["reason"])

    def test_auditor_infeasible_stops_without_remining(self):
        miner = CountingMiner()
        auditor = CountingAuditor(audit_passed=False, request_feasible=False, reason=FIRST_LAW)
        solver = CountingSolver()

        result = _quiet(_runner(miner, auditor, solver=solver).run, _request())

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["rejection"]["stage"], "auditor")
        self.assertEqual(result["rejection"]["reason"], FIRST_LAW)
        self.assertEqual(result["auditor"]["audit_notes"], "audit notes")
        self.assertEqual(result["failed_concepts"], [])
        self.assertEqual((miner.calls, auditor.calls, solver.calls), (1, 1, 0))

    def test_auditor_infeasible_without_reason_falls_back_to_notes(self):
        auditor = CountingAuditor(audit_passed=False, request_feasible=False)
        result = _quiet(_runner(CountingMiner(), auditor).run, _request())
        self.assertEqual(result["rejection"]["reason"], "audit notes")

    def test_concept_level_audit_failure_still_remines(self):
        miner = CountingMiner()
        auditor = CountingAuditor(audit_passed=False, request_feasible=True)

        with self.assertRaises(ConceptsExhaustedError) as ctx:
            _quiet(_runner(miner, auditor).run, _request())

        self.assertEqual(miner.calls, 3)
        err = ctx.exception
        self.assertIsInstance(err, ValueError)  # existing callers catch ValueError
        self.assertEqual(err.attempts, 3)
        self.assertEqual([item["stage"] for item in err.failed_concepts], ["auditor"] * 3)
        message = str(err)
        self.assertTrue(message.startswith("All bionic concepts failed validation (3 attempts)."))
        self.assertIn("1. Concept 1 (Source): Audit failed: audit notes", message)
        self.assertNotIn("[{", message)  # no Python repr

    def test_validator_rejects_efficiency_above_one_in_closed_system(self):
        # The careless-model scenario: the concept claims a heat amplification of 3.5
        # inside a closed adiabatic housing, and the auditor lets it pass.
        class CarelessFormulator(CountingFormulator):
            def mock_formulate_model(self, query, concept):
                output = super().mock_formulate_model(query, concept)
                output.domain = "Thermodynamics"
                output.physical_mechanism = "Gibt in einem geschlossenen adiabaten Gehäuse mehr Wärme ab als zugeführt."
                return output

        class CarelessAuditorOutput(FakeAuditorOutput):
            @property
            def audited_parameters_dict(self):
                return {"amplification_factor": 3.5, "heat_input": 1e5}

        class CarelessAuditor(CountingAuditor):
            def mock_audit_design(self, miner_output, override_parameters=None, user_query=None):
                self.calls += 1
                return CarelessAuditorOutput(
                    audit_passed=True, audit_notes="ok", audited_parameters=[], dimensionless_numbers=[],
                    simulation_coefficient=3.5, solver_method="analytical", ui_metadata={},
                )

        miner = CountingMiner()
        result = _quiet(_runner(miner, CarelessAuditor(), CarelessFormulator()).run, _request(HEATER_QUERY))

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["rejection"]["stage"], "validator")
        self.assertIn("amplification_factor=3.5 > 1", result["rejection"]["reason"])
        self.assertEqual(result["validation"]["recommended_action"], "reject_request")
        self.assertEqual(miner.calls, 1)

    def test_live_mode_records_only_the_stages_that_ran(self):
        class LiveMiner:
            def mine_design(self, query, failed_concepts=None):
                return MinerConceptOutput(
                    design_name="Perpetuum", inspiration_source="none", domain="Thermodynamics",
                    physical_mechanism="impossible", parameters=[], svg_schematic="<svg></svg>",
                    request_feasible=False, infeasibility_reason=FIRST_LAW,
                )

        class LiveFormulator:
            def formulate_model(self, query, concept):
                raise AssertionError("formulator must not run")

        runner = PipelineRunner(
            miner=LiveMiner(), formulator=LiveFormulator(), auditor=object(), optimizer=object(),
            synthesizer=object(), solver=CountingSolver(),
        )
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            result = _quiet(runner.run, PipelineRunRequest(query=HEATER_QUERY, is_mock=False, epochs=5))

        self.assertEqual(result["status"], "rejected")
        self.assertIs(result["is_mock"], False)
        self.assertEqual(list(result["models"]), ["miner"])
        self.assertIs(result["miner"]["request_feasible"], False)

    def test_successful_run_reports_completed_status(self):
        result = _quiet(_runner(CountingMiner(), CountingAuditor()).run, _request(rounds=1))
        self.assertIs(result["success"], True)
        self.assertEqual(result["status"], "completed")


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

def _infeasible_concept(*args, **kwargs):
    return MinerConceptOutput(
        design_name="Perpetuum Heater",
        inspiration_source="none",
        domain="Thermodynamics",
        physical_mechanism="Would need more heat out than in.",
        parameters=[ParameterProposal(name="thermal_conductivity", value=1.0, min_bound=0.1, max_bound=10.0, justification="-")],
        svg_schematic="<svg></svg>",
        request_feasible=False,
        infeasibility_reason=FIRST_LAW,
    )


def _infeasible_audit(miner_output, override_parameters=None, user_query=None):
    return AuditorOutput(
        audit_passed=False,
        audit_notes="Request violates the first law.",
        request_feasible=False,
        infeasibility_reason=FIRST_LAW,
        audited_parameters=[],
        dimensionless_numbers=[],
        simulation_coefficient=0.0,
        solver_method="analytical",
        ui_metadata={
            "domain_name": "Thermodynamics",
            "independent_var": {"label": "x", "unit": "m"},
            "dependent_var": {"label": "T", "unit": "K"},
            "primary_metric": {"label": "q"},
            "reference_metric": {"label": "q"},
            "performance_gain": {"label": "gain"},
        },
    )


class RejectionApiTest(OfflineApiTestCase):
    def _assert_rejected_and_archived(self, body, stage):
        self.assertIs(body["success"], False)
        self.assertEqual(body["status"], "rejected")
        self.assertEqual(body["rejection"]["stage"], stage)
        self.assertEqual(body["rejection"]["reason"], FIRST_LAW)
        self.assertIn("# Anfrage abgelehnt: physikalisch nicht umsetzbar", body["report_md"])
        self.assertIn(FIRST_LAW, body["report_md"])

        history = self.client.get("/api/history").json()["runs"]
        self.assertEqual(len(history), 1)
        stored = self.client.get(f"/api/history/{history[0]['id']}").json()
        self.assertEqual(stored["status"], "rejected")
        self.assertEqual(stored["rejection"]["stage"], stage)

    def test_miner_rejection_returns_200_and_is_archived(self):
        with mock.patch("vectornaut.miner.Miner.mock_mine_design", side_effect=_infeasible_concept) as mined:
            body = self.run_pipeline(query=HEATER_QUERY)
        self.assertEqual(mined.call_count, 1)
        self._assert_rejected_and_archived(body, "miner")
        self.assertEqual(body["miner"]["design_name"], "Perpetuum Heater")

    def test_auditor_rejection_returns_200_and_is_archived(self):
        with mock.patch("vectornaut.auditor.Auditor.mock_audit_design", side_effect=_infeasible_audit) as audited, \
                mock.patch("vectornaut.miner.Miner.mock_mine_design", wraps=_quiet_default_concept) as mined:
            body = self.run_pipeline(query=HEATER_QUERY)
        self.assertEqual((mined.call_count, audited.call_count), (1, 1))
        self._assert_rejected_and_archived(body, "auditor")
        self.assertIn("Request violates the first law.", body["report_md"])

    def test_exhausted_concepts_return_structured_422(self):
        failed = [
            {"design_name": f"Concept {i}", "inspiration_source": "Source", "reason": "Audit failed: unsafe", "stage": "auditor"}
            for i in (1, 2, 3)
        ]
        with mock.patch("vectornaut.api.run_routes.run_pipeline", side_effect=ConceptsExhaustedError(failed, 3)):
            response = self.post_json("/api/run", {"query": "x", "is_mock": True}, expected_status=422)
        body = response.json()
        self.assertIs(body["success"], False)
        self.assertEqual(body["status"], "failed")
        self.assertIsInstance(body["detail"], str)
        self.assertIn("3. Concept 3 (Source): Audit failed: unsafe", body["detail"])
        self.assertEqual(body["error"]["type"], "concepts_exhausted")
        self.assertEqual(body["error"]["failed_concepts"], failed)


def _quiet_default_concept(query):
    # The unpatched mock miner (captured at import time, before any patch).
    return _ORIGINAL_MOCK_MINE(Miner(), query)


# ---------------------------------------------------------------------------
# Auditor post-processing and round-2 prompt
# ---------------------------------------------------------------------------

def _thermal_miner_output():
    return MinerOutput(
        design_name="Fin",
        inspiration_source="Rete mirabile",
        domain="Thermodynamics / Heat Transfer",
        physical_mechanism="Fin conduction",
        parameters=[
            ParameterProposal(name="fin_length", value=0.02, min_bound=0.005, max_bound=0.05, justification="-"),
            ParameterProposal(name="amplification_factor", value=3.5, min_bound=3.0, max_bound=5.0, justification="-"),
        ],
        governing_equation="d2T_dx2 = 0",
        boundary_conditions=["T(0) = 1", "dT_dx(1) = 0"],
        independent_variables=["x"],
        dependent_variables=["T"],
        svg_schematic="<svg></svg>",
    )


def _ui_metadata(domain_name="Heat Transfer"):
    return {
        "domain_name": domain_name,
        "independent_var": {"label": "x", "unit": "-"},
        "dependent_var": {"label": "T", "unit": "K"},
        "primary_metric": {"label": "Base heat flux"},
        "reference_metric": {"label": "Base heat flux (analytical)"},
        "performance_gain": {"label": "Heat uptake change"},
    }


def _model_audit(**overrides):
    data = dict(
        audit_passed=True,
        audit_notes="Model notes.",
        audited_parameters=[
            AuditedParameter(name="fin_length", value=0.02),
            AuditedParameter(name="amplification_factor", value=3.5),
        ],
        dimensionless_numbers=[
            DimensionlessNumber(name="FinParameter_mL", value=0.9877),
            DimensionlessNumber(name="BiotNumber", value=2.44e-5),
        ],
        simulation_coefficient=3.5,
        solver_method="analytical",
        ui_metadata=_ui_metadata(),
    )
    data.update(overrides)
    return AuditorOutput(**data)


def _audit(parsed, miner_output=None, **kwargs):
    client = mock.MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(parsed=parsed)
    result = Auditor(client=client).audit_design(miner_output or _thermal_miner_output(), **kwargs)
    return result, client.models.generate_content.call_args.kwargs["contents"]


class AuditorPostProcessingTest(unittest.TestCase):
    def test_model_dimensionless_numbers_are_kept(self):
        result, _ = _audit(_model_audit())
        self.assertEqual(result.dimensionless_numbers_dict, {"FinParameter_mL": 0.9877, "BiotNumber": 2.44e-5})

    def test_no_placeholder_or_default_velocity_reynolds_number(self):
        result, _ = _audit(_model_audit(dimensionless_numbers=[]))
        self.assertEqual(result.dimensionless_numbers, [])

        fluid = _thermal_miner_output().model_copy(update={"domain": "Fluid Dynamics"})
        result, _ = _audit(_model_audit(dimensionless_numbers=[], audited_parameters=[
            AuditedParameter(name="viscosity", value=0.001),  # no velocity, length or density
        ]), miner_output=fluid)
        self.assertEqual(result.dimensionless_numbers, [])

    def test_reynolds_number_added_only_from_actual_parameters(self):
        fluid = _thermal_miner_output().model_copy(update={"domain": "Fluid Dynamics"})
        params = [
            AuditedParameter(name="velocity", value=2.0),
            AuditedParameter(name="hydraulic_diameter", value=0.02),
            AuditedParameter(name="kinematic_viscosity", value=1e-6),
        ]
        result, _ = _audit(_model_audit(dimensionless_numbers=[], audited_parameters=params), miner_output=fluid)
        self.assertAlmostEqual(result.dimensionless_numbers_dict["ReynoldsNumber"], 40000.0)
        self.assertIn("ReynoldsNumber = 40000 ergänzt", result.audit_notes)

        given = [DimensionlessNumber(name="Re_D", value=39000.0)]
        result, _ = _audit(_model_audit(dimensionless_numbers=given, audited_parameters=params), miner_output=fluid)
        self.assertEqual(result.dimensionless_numbers_dict, {"Re_D": 39000.0})

        self.assertAlmostEqual(
            compute_reynolds_number({"velocity": 1.0, "length": 0.1, "viscosity": 1e-3, "density": 1000.0})[0], 1e5
        )
        self.assertIsNone(compute_reynolds_number({"velocity": 1.0, "length": 0.1, "viscosity": 1e-3}))

    def test_domain_names_map_to_ui_default_domains(self):
        cases = {
            "Thermodynamics": "thermal",
            "Thermodynamics / Heat Transfer": "thermal",
            "Heat Transfer": "thermal",
            "Structural Mechanics": "structural",
            "Solid Mechanics / Biomechanics": "structural",
            "Electrostatics": "electro",
            "Electromagnetics": "electro",
            "Fluid Dynamics": "fluid",
            "Fluid Mechanics": "fluid",
            "Acoustics": None,
        }
        for domain, expected in cases.items():
            self.assertEqual(match_ui_domain(domain), expected, domain)

    def test_rejected_audit_notes_are_not_modified(self):
        result, _ = _audit(_model_audit(audit_passed=False, audit_notes="Structural collapse."))
        self.assertIs(result.audit_passed, False)
        self.assertEqual(result.audit_notes, "Structural collapse.")

    def test_notes_list_programmatic_changes_only_when_something_changed(self):
        result, _ = _audit(_model_audit())
        self.assertEqual(result.audit_notes, "Model notes.")
        self.assertNotIn("Programmatic safety checks applied", result.audit_notes)

        result, _ = _audit(_model_audit(audited_parameters=[AuditedParameter(name="fin_length", value=0.5)]))
        self.assertEqual(result.audited_parameters_dict["fin_length"], 0.05)  # clamped to the Miner's bounds
        self.assertIn("Programmatische Korrekturen: fin_length: 0.5 -> 0.05", result.audit_notes)

    def test_request_infeasible_passes_through_and_fails_the_audit(self):
        result, _ = _audit(_model_audit(request_feasible=False, infeasibility_reason=FIRST_LAW))
        self.assertIs(result.request_feasible, False)
        self.assertEqual(result.infeasibility_reason, FIRST_LAW)
        self.assertIs(result.audit_passed, False)

        feasible, _ = _audit(_model_audit(infeasibility_reason="stray text"))
        self.assertIs(feasible.request_feasible, True)
        self.assertIsNone(feasible.infeasibility_reason)

    def test_recorded_answers_without_new_fields_still_parse(self):
        parsed = AuditorOutput.model_validate(_model_audit().model_dump(exclude={"request_feasible", "infeasibility_reason"}))
        self.assertIs(parsed.request_feasible, True)
        concept = MinerConceptOutput.model_validate({
            "design_name": "a", "inspiration_source": "b", "domain": "c", "physical_mechanism": "d",
            "parameters": [], "svg_schematic": "",
        })
        self.assertIs(concept.request_feasible, True)
        self.assertIsNone(concept.infeasibility_reason)

    def test_prompt_explains_feasibility_fields(self):
        _, prompt = _audit(_model_audit())
        self.assertIn("request_feasible=false", prompt)
        self.assertIn("audit_passed=false and keep request_feasible=true", prompt)

    def test_round_two_prompt_contains_override_values(self):
        _, round_one = _audit(_model_audit())
        self.assertNotIn("PARAMETER VALUES TO AUDIT THIS ROUND", round_one)
        self.assertIn("'amplification_factor': 3.5", round_one)

        _, round_two = _audit(_model_audit(), override_parameters={"amplification_factor": 5.0})
        self.assertIn("PARAMETER VALUES TO AUDIT THIS ROUND", round_two)
        self.assertIn("{'amplification_factor': 5.0}", round_two)
        self.assertIn("Proposed Parameters: {'fin_length': 0.02, 'amplification_factor': 5.0}", round_two)
        self.assertNotIn("'amplification_factor': 3.5", round_two)

    def test_override_values_are_used_and_coefficient_follows_its_parameter(self):
        # The model answered with the stale value (as a replayed round-1 answer does).
        result, _ = _audit(_model_audit(), override_parameters={"amplification_factor": 5.0})
        self.assertEqual(result.audited_parameters_dict["amplification_factor"], 5.0)
        self.assertEqual(result.simulation_coefficient, 5.0)
        self.assertIn("amplification_factor: 3.5 -> 5 (Vorgabe aus Override übernommen)", result.audit_notes)
        self.assertIn("simulation_coefficient: 3.5 -> 5", result.audit_notes)

        # Overrides are still clamped to the Miner's bounds.
        result, _ = _audit(_model_audit(), override_parameters={"amplification_factor": 9.0})
        self.assertEqual(result.audited_parameters_dict["amplification_factor"], 5.0)

    def test_model_answer_that_already_uses_the_override_is_not_annotated(self):
        answer = _model_audit(
            audited_parameters=[
                AuditedParameter(name="fin_length", value=0.02),
                AuditedParameter(name="amplification_factor", value=5.0),
            ],
            simulation_coefficient=5.0,
        )
        result, _ = _audit(answer, override_parameters={"amplification_factor": 5.0})
        self.assertEqual(result.simulation_coefficient, 5.0)
        self.assertEqual(result.audit_notes, "Model notes.")


class MinerPromptTest(unittest.TestCase):
    def test_miner_prompt_explains_feasibility_fields(self):
        from vectornaut.miner import Miner

        client = mock.MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(parsed=_infeasible_concept())
        concept = Miner(client=client).mine_design(HEATER_QUERY, failed_concepts=[{"reason": "Audit failed: x"}])
        prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("request_feasible=false", prompt)
        self.assertIn("conservation law", prompt)
        self.assertIn("setze request_feasible=false", prompt)  # re-mining hint
        self.assertIs(concept.request_feasible, False)


# ---------------------------------------------------------------------------
# Deterministic closed-system efficiency check
# ---------------------------------------------------------------------------

def _miner(mechanism, domain="Thermodynamics"):
    return {"design_name": "d", "domain": domain, "physical_mechanism": mechanism}


class ClosedSystemEfficiencyCheckTest(unittest.TestCase):
    def test_flags_amplification_in_closed_adiabatic_housing(self):
        check = check_closed_system_efficiency(
            _miner("Wärme wird in einem geschlossenen adiabaten Gehäuse kaskadenartig verstärkt."),
            {"audited_parameters_dict": {"amplification_factor": 3.5}},
        )
        self.assertIsNotNone(check)
        self.assertEqual(check.severity, "error")
        self.assertIn("amplification_factor=3.5 > 1", check.detail)

    def test_flags_efficiency_from_query_context(self):
        check = check_closed_system_efficiency(
            _miner("Heating surface."), {"audited_parameters_dict": {"heating_efficiency_ratio": 3.0}}, user_query=HEATER_QUERY,
        )
        self.assertIsNotNone(check)

    def test_percent_efficiency_needs_to_exceed_100(self):
        miner = _miner("isolated system without energy input")
        self.assertIsNone(check_closed_system_efficiency(miner, {"audited_parameters_dict": {"efficiency": 85.0}}))
        self.assertIsNotNone(check_closed_system_efficiency(miner, {"audited_parameters_dict": {"efficiency_pct": 350.0}}))

    def test_open_systems_and_normal_values_are_not_flagged(self):
        cases = [
            # Heat pump: COP > 1 is fine in an open system.
            (_miner("Heat pump drawing ambient heat, compressor with adiabatic compression."), {"cop": 3.5}),
            # Fin with an adiabatic tip.
            (_miner("Fin with adiabatic tip conducting heat to the air."), {"fin_efficiency": 0.77}),
            # Closed system but an efficiency below 1.
            (_miner("closed adiabatic system"), {"efficiency": 0.9}),
            # Amplification outside an energy context.
            (_miner("isolated system of optical fibres", domain="Optics"), {"amplification_factor": 20.0}),
        ]
        for miner, params in cases:
            self.assertIsNone(check_closed_system_efficiency(miner, {"audited_parameters_dict": params}), miner)

    def test_validator_maps_failure_to_reject_request(self):
        sim = {
            "solver_method": "analytical", "relative_error": 0.0, "performance_gain_pct": 0.0,
            "sample_points": [0.0, 1.0], "solution_primary": [1.0, 1.0], "solution_reference": [1.0, 1.0],
            "primary_metric_value": 0.0, "reference_metric_value": 0.0,
        }
        miner = {**_miner("geschlossenen adiabaten Gehäuse"), "boundary_conditions": [], "dependent_variables": ["T"]}
        result = validate_run_output(miner, {"audited_parameters_dict": {"amplification_factor": 3.5}}, sim)
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.recommended_action, "reject_request")

        plain = validate_run_output(miner, {"audited_parameters_dict": {"thermal_conductivity": 3.5}}, sim)
        self.assertNotIn("physics_closed_system_efficiency", [check.name for check in plain.checks])


# ---------------------------------------------------------------------------
# Model comparison harness
# ---------------------------------------------------------------------------

REJECT_EXPECT = {"outcome": "rejected", "audit_passed": False, "max_remines": 2}


class ModelEvalRejectionTest(unittest.TestCase):
    def test_structured_rejection_counts_as_explicit_rejection(self):
        result = {
            "success": False, "status": "rejected", "rejection": {"stage": "miner", "reason": FIRST_LAW},
            "miner": {"design_name": "d"}, "auditor": None, "simulator": None, "validation": None,
            "failed_concepts": [], "optimization_history": [], "models": {},
        }
        facts = extract_run_facts(result, log_text="[*] Starting Concept Attempt 1...")
        self.assertEqual(facts["category"], "request_rejected")
        self.assertEqual(facts["rejection_stage"], "miner")
        self.assertEqual(facts["remines"], 0)
        scored = score_run(REJECT_EXPECT, facts)
        self.assertTrue(scored["outcome_met"])
        self.assertEqual(scored["score"], 1.0, scored["failed_checks"])

    def test_readable_exhausted_error_is_still_classified(self):
        failed = [{"design_name": "c", "reason": "Audit failed: impossible", "stage": "auditor"}] * 3

        def run_fn(request):
            raise ConceptsExhaustedError(failed, 3)

        with tempfile.TemporaryDirectory() as tmp:
            record = run_case(
                {"id": "c1", "query": "q", "expect": dict(REJECT_EXPECT)}, ModelConfig("baseline"), 1,
                mock=True, epochs=1, max_rounds=1, run_dir=tmp, run_fn=run_fn,
            )
        self.assertEqual(record["facts"]["category"], "audit_rejected")
        self.assertEqual(record["facts"]["remine_reasons"], ["audit_rejected"] * 3)
        self.assertTrue(record["outcome_met"])


if __name__ == "__main__":
    unittest.main()
