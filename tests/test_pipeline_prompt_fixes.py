# -*- coding: utf-8 -*-
"""
Prompt/schema fixes found in role-play runs (docs/LIVE_PATH_FINDINGS.md, finding 9 and
follow-ups):

- the formulator can add and change parameters through ModelFormulation.parameters, and
  they reach MinerOutput and the auditor prompt (recorded answers without the field parse);
- the auditor prompt offers the riblet slip formula only for riblet geometries, and the
  post-processing recomputes the coefficient only when both riblet parameters exist;
- MetricSpec.transform turns the metric into a nonlinear figure of merit before the gain
  is computed (1D and 2D closed forms), and an unsafe or invalid transform is rejected;
- an implausible gain (validator gain-sanity check failed) is flagged in the report and the
  synthesizer prompt; gain_basis 'none' is not sanity-checked as if 0.0 were a result;
- the synthesizer labels the parameters 'optimiert' only after an optimization round.
"""
import contextlib
import io
import json
import math
import os
import shutil
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import torch

from tests.solver_accuracy_helpers import Case, build_inputs
from vectornaut.auditor import Auditor
from vectornaut.config import (
    AuditedParameter,
    AuditorOutput,
    BaselineParameter,
    DimensionlessNumber,
    MetricSpec,
    MinerConceptOutput,
    MinerOutput,
    ModelFormulation,
    ObjectiveMetricContract,
    ParameterProposal,
)
from vectornaut.formulator import ModelFormulator, merge_formulated_parameters
from vectornaut.reporting import generate_markdown_report_content, gain_implausible
from vectornaut.solvers.metrics import MetricSpecError, apply_metric_transform, parse_metric_transform
from vectornaut.synthesizer import Synthesizer
from vectornaut.validator import validate_run_output

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _client_returning(parsed):
    client = mock.MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(parsed=parsed)
    return client


def _prompt_of(client) -> str:
    return client.models.generate_content.call_args.kwargs["contents"]


def _concept(**overrides):
    data = dict(
        design_name="PeristomeFlow liner",
        inspiration_source="Nepenthes peristome",
        domain="Fluid Dynamics",
        physical_mechanism="Liquid-infused grooves give an effective wall slip. Baseline: smooth elastomer hose wall.",
        parameters=[
            ParameterProposal(name="slip_length", value=1e-5, min_bound=1e-7, max_bound=5e-5,
                              justification="Assumes silicone oil / water viscosity ratio of 20."),
            ParameterProposal(name="viscosity", value=1e-3, min_bound=8e-4, max_bound=1.8e-3,
                              justification="Water at 0-30 C."),
        ],
        svg_schematic="<svg></svg>",
    )
    data.update(overrides)
    return MinerConceptOutput(**data)


def _formulation(parameters=None):
    return ModelFormulation(
        governing_equation="d2u_dy2 = channel_half_height**2 * pressure_gradient / viscosity",
        boundary_conditions=["u(0) = (slip_length / channel_half_height) * du_dy(0)", "du_dy(1) = 0"],
        independent_variables=["y"],
        dependent_variables=["u"],
        parameters=parameters,
    )


def _formulate(formulation, concept=None):
    client = _client_returning(formulation)
    formulator = ModelFormulator(client=client)
    with contextlib.redirect_stdout(io.StringIO()):
        miner_output = formulator.formulate_model("hose query", concept or _concept())
    return formulator, miner_output, _prompt_of(client)


# ---------------------------------------------------------------------------
# 1. Formulator parameters
# ---------------------------------------------------------------------------

class FormulatorParametersTest(unittest.TestCase):
    NEW = [
        ParameterProposal(name="channel_half_height", value=0.005, min_bound=0.002, max_bound=0.02,
                          justification="Hose inner radius."),
        ParameterProposal(name="pressure_gradient", value=-2000.0, min_bound=-20000.0, max_bound=-100.0,
                          justification="Pump-driven pressure drop."),
    ]

    def test_added_parameters_reach_miner_output(self):
        formulator, out, _ = _formulate(_formulation(self.NEW))
        self.assertEqual([p.name for p in out.parameters],
                         ["slip_length", "viscosity", "channel_half_height", "pressure_gradient"])
        self.assertEqual(out.proposed_parameters["pressure_gradient"], -2000.0)
        self.assertEqual(out.suggested_bounds["channel_half_height"], [0.002, 0.02])
        self.assertEqual(len(formulator.last_parameter_changes), 2)
        self.assertTrue(all("added" in change for change in formulator.last_parameter_changes))

    def test_existing_parameter_bounds_are_updated_and_others_kept(self):
        widened = ParameterProposal(name="slip_length", value=1e-5, min_bound=1e-8, max_bound=1e-4,
                                    justification="Widened: measured slip lengths up to 100 um.")
        formulator, out, _ = _formulate(_formulation([widened] + self.NEW))
        params = {p.name: p for p in out.parameters}
        self.assertEqual(params["slip_length"].min_bound, 1e-8)
        self.assertEqual(params["slip_length"].max_bound, 1e-4)
        self.assertEqual(params["slip_length"].value, 1e-5)
        self.assertIn("Widened", params["slip_length"].justification)
        # Not listed by the formulator: unchanged.
        self.assertEqual(params["viscosity"].max_bound, 1.8e-3)
        self.assertEqual(params["viscosity"].justification, "Water at 0-30 C.")
        change = next(c for c in formulator.last_parameter_changes if c.startswith("slip_length"))
        self.assertIn("bounds [1e-07, 5e-05] -> [1e-08, 0.0001]", change)
        self.assertNotIn("value", change)

    def test_unchanged_relisting_logs_nothing(self):
        merged, changes = merge_formulated_parameters(_concept().parameters, [p.model_copy() for p in _concept().parameters])
        self.assertEqual([p.model_dump() for p in merged], [p.model_dump() for p in _concept().parameters])
        self.assertEqual(changes, [])

    def test_added_parameters_reach_the_auditor_prompt(self):
        widened = ParameterProposal(name="slip_length", value=1e-5, min_bound=1e-8, max_bound=1e-4, justification="w")
        _, out, _ = _formulate(_formulation([widened] + self.NEW))
        audit = AuditorOutput(
            audit_passed=True, audit_notes="ok",
            audited_parameters=[AuditedParameter(name=k, value=v) for k, v in out.proposed_parameters.items()],
            dimensionless_numbers=[], simulation_coefficient=1e-5, solver_method="analytical",
            ui_metadata=_fluid_ui(),
        )
        client = _client_returning(audit)
        with contextlib.redirect_stdout(io.StringIO()):
            result = Auditor(client=client).audit_design(out, user_query="hose")
        prompt = _prompt_of(client)
        self.assertIn("'pressure_gradient': -2000.0", prompt)
        self.assertIn("'channel_half_height': [0.002, 0.02]", prompt)
        self.assertIn("'slip_length': [1e-08, 0.0001]", prompt)
        # Not clamped to the old upper bound and no auto-injected default needed.
        self.assertEqual(result.audited_parameters_dict["pressure_gradient"], -2000.0)

    def test_recorded_answers_without_parameters_still_parse(self):
        path = os.path.join(REPO_ROOT, "benchmarks", "llm_sessions", "riblet_hose", "responses", "02.json")
        with open(path, encoding="utf-8") as f:
            formulation = ModelFormulation.model_validate(json.load(f))
        self.assertIsNone(formulation.parameters)
        formulator, out, _ = _formulate(formulation)
        self.assertEqual([p.model_dump() for p in out.parameters], [p.model_dump() for p in _concept().parameters])
        self.assertEqual(formulator.last_parameter_changes, [])

    def test_prompt_matches_schema_and_solver_support(self):
        _, _, prompt = _formulate(_formulation())
        self.assertIn("'parameters' field of the returned JSON", prompt)
        self.assertIn("Proposed parameters you do not list are kept unchanged", prompt)
        self.assertNotIn("linear combination of parameters", prompt)
        self.assertIn("LINEAR in the dependent variable and its derivatives", prompt)
        self.assertIn("may depend on the independent variable and the parameters", prompt)
        self.assertIn("parameters", ModelFormulation.model_fields)

    def test_concept_text_and_assumptions_are_not_dropped(self):
        # Explorer candidates arrive as PipelineRunRequest.concept: their mechanism text (which
        # carries the stated baseline) and each parameter's justification must reach the prompt.
        _, _, prompt = _formulate(_formulation())
        self.assertIn("Baseline: smooth elastomer hose wall.", prompt)
        self.assertIn("Assumes silicone oil / water viscosity ratio of 20.", prompt)
        self.assertIn("Nepenthes peristome", prompt)
        self.assertIn("slip_length: value=1e-05, bounds=[1e-07, 5e-05]", prompt)


# ---------------------------------------------------------------------------
# 2. Auditor prompt and riblet post-processing
# ---------------------------------------------------------------------------

def _fluid_ui():
    return {
        "domain_name": "Fluid Dynamics",
        "independent_var": {"label": "y", "unit": "-"},
        "dependent_var": {"label": "u", "unit": "m/s"},
        "primary_metric": {"label": "Wall shear stress"},
        "reference_metric": {"label": "Wall shear stress (analytical)"},
        "performance_gain": {"label": "Drag reduction"},
    }


def _fluid_miner(params, bcs=None):
    return MinerOutput(
        design_name="Fluid concept", inspiration_source="-", domain="Fluid Dynamics", physical_mechanism="-",
        parameters=[ParameterProposal(name=k, value=v, min_bound=v / 10.0, max_bound=v * 10.0, justification="-")
                    for k, v in params.items()],
        governing_equation="d2u_dy2 = -1", boundary_conditions=bcs or ["u(0) = 0", "u(1) = 1"],
        independent_variables=["y"], dependent_variables=["u"], svg_schematic="<svg></svg>",
    )


def _audit_fluid(params, coefficient, bcs=None):
    audit = AuditorOutput(
        audit_passed=True, audit_notes="model notes",
        audited_parameters=[AuditedParameter(name=k, value=v) for k, v in params.items()],
        dimensionless_numbers=[DimensionlessNumber(name="ReynoldsNumber", value=5000.0)],
        simulation_coefficient=coefficient, solver_method="analytical", ui_metadata=_fluid_ui(),
    )
    client = _client_returning(audit)
    with contextlib.redirect_stdout(io.StringIO()):
        result = Auditor(client=client).audit_design(_fluid_miner(params, bcs), user_query="drag")
    return result, _prompt_of(client)


class AuditorRibletTest(unittest.TestCase):
    def test_prompt_offers_riblet_formula_only_as_example(self):
        _, prompt = _audit_fluid({"groove_depth": 2e-5, "viscosity": 1e-3}, 3e-6)
        self.assertNotIn("For Fluid Dynamics (drag reduction), calculate the slippage length", prompt)
        self.assertIn("Only for riblet geometries", prompt)
        self.assertIn("For every other mechanism derive the coefficient yourself", prompt)
        self.assertNotIn("density = 1000", prompt)
        self.assertNotIn("L = 1.0 m", prompt)
        self.assertIn("actual characteristic length", prompt)

    def test_non_riblet_fluid_concept_keeps_model_coefficient(self):
        result, _ = _audit_fluid({"groove_depth": 2e-5, "viscosity": 1e-3}, 3e-6)
        self.assertAlmostEqual(result.simulation_coefficient, 3e-6, delta=1e-18)
        self.assertNotIn("simulation_coefficient:", result.audit_notes)

    def test_only_one_riblet_parameter_does_not_trigger_formula(self):
        result, _ = _audit_fluid({"riblet_height": 0.01, "viscosity": 1e-3}, 3e-6)
        self.assertAlmostEqual(result.simulation_coefficient, 3e-6, delta=1e-18)

    # The recompute now also requires that the equations use the coefficient: the BCs here
    # reference slip_length (before, the default "u(0) = 0" BCs were enough).
    def test_riblet_geometry_recomputes_coefficient(self):
        h, s = 0.01, 0.02
        result, _ = _audit_fluid({"riblet_height": h, "riblet_spacing": s, "viscosity": 1e-3}, 3e-6,
                                 bcs=["u(0) = slip_length * du_dy(0)", "u(1) = 1"])
        expected = 0.2 * s * (1.0 - math.exp(-2.0 * h / s))
        self.assertAlmostEqual(result.simulation_coefficient, expected, delta=1e-12)
        self.assertIn("Riblet-Geometrie", result.audit_notes)


# ---------------------------------------------------------------------------
# 3. Metric transform
# ---------------------------------------------------------------------------

class _SolveCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="vectornaut_prompt_fixes_")
        self._previous = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = self._tmp

    def tearDown(self):
        if self._previous is None:
            os.environ.pop("VECTORNAUT_DATA_DIR", None)
        else:
            os.environ["VECTORNAUT_DATA_DIR"] = self._previous
        shutil.rmtree(self._tmp, ignore_errors=True)

    def solve(self, case, method, metric=None, baseline=None, lower_is_better=None):
        from vectornaut.solver_dispatcher import dispatch_and_solve

        miner, auditor = build_inputs(case, method)
        if metric is not None:
            auditor.metric = MetricSpec(**metric)
        if baseline is not None:
            auditor.baseline_parameters = [BaselineParameter(name=k, value=v) for k, v in baseline.items()]
        if lower_is_better is not None:
            auditor.objective_metric = ObjectiveMetricContract(
                objective_name="gain", primary_metric="primary", reference_metric="reference",
                lower_is_better=lower_is_better, acceptance_threshold=0.0,
            )
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = dispatch_and_solve(miner, auditor, epochs=50)
        return miner, auditor, sim


# Beam under uniform load in normalized form: w'' = -q x (1 - x) / 2, w(0) = w(1) = 0
# -> w_max = 5 q / 384. Read the peak deflection as the compliance C of a foul-release coating:
# detachment stress sigma_c = sqrt(2 w / C) with adhesion energy w.
COMPLIANCE_1D = Case("compliance_beam", "d2w_dx2 = -q_norm * x * (1 - x) / 2", ["w(0) = 0", "w(1) = 0"], ["x"], "w",
                     exact=lambda x: (x - 2.0 * x ** 3 + x ** 4) / 24.0,
                     params={"q_norm": 1.0, "adhesion_energy": 0.5})
SIGMA = {"kind": "max_abs", "unit": "Pa", "label": "Critical detachment stress",
         "transform": "sqrt(2*adhesion_energy/m)"}

# 2D slab T = T_out + (T_in - T_out) x, flux k dT/dx at x = 1 = k * 3.
SLAB_2D = Case("slab_2d", "d2T_dx2 + d2T_dy2 = 0",
               ["T(0, y) = T_out", "T(1, y) = T_in", "dT_dy(x, 0) = 0", "dT_dy(x, 1) = 0"], ["x", "y"], "T",
               exact=lambda x, y: -2.0 + 3.0 * x,
               params={"T_out": -2.0, "T_in": 1.0, "k": 0.8, "adhesion_energy": 0.5})


class MetricTransformTest(_SolveCase):
    def test_1d_detachment_stress_from_compliance(self):
        # Design q = 1: C = 5/384, sigma = sqrt(384/5). Stiffer baseline q = 0.5: C = 2.5/384.
        # Lower sigma is better: gain = 1 - sqrt(C_b / C_d) = 1 - 1/sqrt(2) (the raw compliance
        # "gain" with lower_is_better would be -100 %).
        _, _, sim = self.solve(COMPLIANCE_1D, "analytical", metric=SIGMA, baseline={"q_norm": 0.5},
                               lower_is_better=True)
        self.assertEqual(sim.gain_basis, "baseline_parameters")
        self.assertAlmostEqual(sim.primary_metric_value, math.sqrt(384.0 / 5.0), delta=1e-6)
        self.assertAlmostEqual(sim.reference_metric_value, math.sqrt(384.0 / 5.0), delta=1e-6)
        self.assertAlmostEqual(sim.baseline_metric_value, math.sqrt(384.0 / 2.5), delta=1e-6)
        self.assertAlmostEqual(sim.performance_gain_pct, (1.0 - 1.0 / math.sqrt(2.0)) * 100.0, delta=1e-5)
        self.assertEqual(sim.metric_spec["transform"], "sqrt(2*adhesion_energy/m)")
        self.assertEqual(sim.metric_unit, "Pa")
        self.assertIsNone(sim.gain_note)

    def test_1d_transform_uses_baseline_parameters(self):
        # The baseline coating has another adhesion energy: sigma_b = sqrt(2 * 2 / C).
        _, _, sim = self.solve(COMPLIANCE_1D, "analytical", metric=SIGMA, baseline={"adhesion_energy": 2.0},
                               lower_is_better=True)
        self.assertAlmostEqual(sim.baseline_metric_value, math.sqrt(4.0 * 384.0 / 5.0), delta=1e-6)
        self.assertAlmostEqual(sim.performance_gain_pct, 50.0, delta=1e-5)

    def test_2d_transform(self):
        # Flux 2.4 (k = 0.8) vs baseline 4.8 (k = 1.6); figure of merit sqrt(2 w / flux), higher is better:
        # gain = sqrt(4.8 / 2.4) - 1 = sqrt(2) - 1.
        metric = {"kind": "derivative_at", "location": "x=1", "scale": "k", "transform": "sqrt(2*adhesion_energy/m)"}
        _, _, sim = self.solve(SLAB_2D, "fdm", metric=metric, baseline={"k": 1.6}, lower_is_better=False)
        self.assertEqual(sim.gain_basis, "baseline_parameters")
        self.assertAlmostEqual(sim.primary_metric_value, math.sqrt(1.0 / 2.4), delta=1e-7)
        self.assertAlmostEqual(sim.baseline_metric_value, math.sqrt(1.0 / 4.8), delta=1e-7)
        self.assertAlmostEqual(sim.performance_gain_pct, (math.sqrt(2.0) - 1.0) * 100.0, delta=1e-5)

    def test_invalid_transform_is_rejected_with_note(self):
        for transform in ("m.__class__", "__import__('os').system('echo pwned')", "m + undefined_name",
                          "sqrt(-m)", "adhesion_energy * 2"):
            with self.subTest(transform=transform):
                metric = dict(SIGMA, transform=transform)
                _, _, sim = self.solve(COMPLIANCE_1D, "analytical", metric=metric, baseline={"q_norm": 0.5},
                                       lower_is_better=True)
                self.assertEqual(sim.gain_basis, "none")
                self.assertEqual(sim.performance_gain_pct, 0.0)
                self.assertIsNone(sim.baseline_metric_value)
                self.assertIn("metric transform rejected", sim.gain_note)
                self.assertIn("not computed", sim.gain_note)
                # Untransformed metric (w_max = 5/384), no longer labelled with the transformed unit.
                self.assertAlmostEqual(sim.primary_metric_value, 5.0 / 384.0, delta=1e-9)
                self.assertIsNone(sim.metric_spec["transform"])
                self.assertIsNone(sim.metric_unit)

    def test_parser_whitelist(self):
        self.assertAlmostEqual(apply_metric_transform("sqrt(2*w/m)", 0.5, {"w": 1.0}), 2.0)
        self.assertAlmostEqual(apply_metric_transform("m^2 + lambda", 3.0, {"lambda": 1.0}), 10.0)
        self.assertAlmostEqual(apply_metric_transform("max(m, 1e-3) * 2.5E2", 3.0, {}), 750.0)
        self.assertEqual(apply_metric_transform(None, 3.0, {}), 3.0)
        for bad in ("m.real", "1.real + m", "m[0]", "lambda: m", "m if m else 1", "exec(m)",
                    "Symbol('a') + m", "m == 1", "(m, m)", "", "x" * 400 + "m"):
            with self.subTest(bad=bad), self.assertRaises(MetricSpecError):
                parse_metric_transform(bad, ["w"])


# ---------------------------------------------------------------------------
# 3b. Gain sanity: implausible gain and gain_basis 'none'
# ---------------------------------------------------------------------------

BEAM = Case("beam_normalized", "d2w_dx2 = -q_norm * x * (1 - x) / 2", ["w(0) = 0", "w(1) = 0"], ["x"], "w",
            exact=lambda x: (x - 2.0 * x ** 3 + x ** 4) / 24.0, params={"q_norm": 1.0})
DEFLECTION = {"kind": "max_abs", "unit": "m", "label": "Maximum deflection"}


def _run_data(miner, auditor, sim):
    auditor_data = {**auditor.model_dump(), "audited_parameters_dict": auditor.audited_parameters_dict}
    validation = validate_run_output(miner, auditor_data, sim, optimization_history=[])
    data = {
        "miner": miner.model_dump(),
        "auditor": auditor_data,
        "simulator": sim.model_dump(),
        "validation": validation.model_dump(),
        "optimization_history": [{"round": 1, "parameters": auditor.audited_parameters_dict,
                                  "simulation_coefficient": auditor.simulation_coefficient,
                                  "simulator": sim.model_dump(), "validation": validation.model_dump()}],
    }
    return data, validation, generate_markdown_report_content(data, "20260925_120000")


def _synth_prompt(miner, auditor, sim, history=None, validation=None):
    client = _client_returning(Synthesizer().mock_generate_synthesis())
    Synthesizer(client=client).generate_synthesis(miner, auditor, sim, "q",
                                                  optimization_history=history, validation_result=validation)
    return _prompt_of(client)


class GainSanityTest(_SolveCase):
    def test_implausible_gain_is_flagged_in_report_and_synthesizer(self):
        # Baseline load 0.1: w_max 10x smaller; higher-is-better deflection "gain" = 900 %.
        miner, auditor, sim = self.solve(BEAM, "analytical", metric=DEFLECTION, baseline={"q_norm": 0.1},
                                         lower_is_better=False)
        self.assertAlmostEqual(sim.performance_gain_pct, 900.0, delta=1e-4)
        data, validation, report = _run_data(miner, auditor, sim)
        check = next(c for c in validation.checks if c.name == "physics_performance_gain_sanity")
        self.assertFalse(check.passed)
        self.assertTrue(gain_implausible(data["simulator"], data["validation"]))
        self.assertIn("**gain:** **implausibel**", report)
        self.assertNotIn("**900.00%**", report)
        self.assertIn("implausibel (900.00%)", report.split("## 7.")[1])  # optimization history table
        prompt = _synth_prompt(miner, auditor, sim, validation=validation)
        self.assertIn("IMPLAUSIBEL", prompt)
        self.assertNotIn("900.00%", prompt)
        # Without a validation result the default 500 % limit applies.
        self.assertIn("IMPLAUSIBEL", _synth_prompt(miner, auditor, sim))

    def test_plausible_gain_is_presented(self):
        miner, auditor, sim = self.solve(BEAM, "analytical", metric=DEFLECTION, baseline={"q_norm": 2.0},
                                         lower_is_better=True)
        data, validation, report = _run_data(miner, auditor, sim)
        self.assertFalse(gain_implausible(data["simulator"], data["validation"]))
        self.assertIn("**gain:** **50.00%**", report)
        self.assertIn("Effizienz (Performance Gain): 50.00%", _synth_prompt(miner, auditor, sim, validation=validation))

    def test_gain_basis_none_is_not_sanity_checked(self):
        miner, auditor, sim = self.solve(BEAM, "analytical", metric=DEFLECTION, lower_is_better=True)
        self.assertEqual(sim.gain_basis, "none")
        _, validation, report = _run_data(miner, auditor, sim)
        check = next(c for c in validation.checks if c.name == "physics_performance_gain_sanity")
        self.assertTrue(check.passed)
        self.assertEqual(check.severity, "info")
        self.assertIn("not evaluated", check.detail)
        self.assertNotIn("performance_gain_pct=0.0", check.detail)
        self.assertIn("**gain:** **n/a**", report)
        self.assertNotIn("implausibel", report)
        self.assertIn("n/a", _synth_prompt(miner, auditor, sim, validation=validation))

    def test_legacy_result_without_gain_basis_is_still_checked(self):
        simulator = {"solver_method": "analytical", "relative_error": 0.0, "performance_gain_pct": 1200.0,
                     "sample_points": [0.0, 1.0], "solution_primary": [0.0, 1.0], "solution_reference": [0.0, 1.0],
                     "primary_metric_value": 1.0, "reference_metric_value": 1.0}
        validation = validate_run_output({}, {}, simulator)
        check = next(c for c in validation.checks if c.name == "physics_performance_gain_sanity")
        self.assertFalse(check.passed)
        self.assertEqual(check.severity, "warning")


# ---------------------------------------------------------------------------
# 4. Synthesizer parameter label
# ---------------------------------------------------------------------------

class SynthesizerLabelTest(_SolveCase):
    def test_label_depends_on_optimization(self):
        miner, auditor, sim = self.solve(BEAM, "analytical")
        one_round = [{"round": 1, "parameters": {"q_norm": 1.0}}]
        self.assertIn("Parameter (auditiert):", _synth_prompt(miner, auditor, sim, history=one_round))
        self.assertIn("Parameter (auditiert):", _synth_prompt(miner, auditor, sim))
        # A second round with identical parameters is not an optimization either.
        same = one_round + [{"round": 2, "parameters": {"q_norm": 1.0}}]
        self.assertIn("Parameter (auditiert):", _synth_prompt(miner, auditor, sim, history=same))
        changed = one_round + [{"round": 2, "parameters": {"q_norm": 0.8}}]
        prompt = _synth_prompt(miner, auditor, sim, history=changed)
        self.assertIn("Parameter (optimiert, 2 Runden):", prompt)
        self.assertNotIn("Parameter (auditiert)", prompt)


if __name__ == "__main__":
    unittest.main()
