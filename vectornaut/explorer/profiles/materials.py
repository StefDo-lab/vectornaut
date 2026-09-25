# -*- coding: utf-8 -*-
"""
Materials profile: bio- and physically inspired materials, surfaces and microstructures.

Evaluation runs every candidate through the existing pipeline (formulator -> auditor ->
solver -> validator -> optimizer -> synthesizer) via ``PipelineRunRequest.concept``, with
re-mining disabled (``max_concept_attempts=1``) so that exactly this concept is judged.
"""
import contextlib
import io
import math
import random
import zlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from vectornaut.config import MinerConceptOutput, ParameterProposal
from vectornaut.explorer.archive import EVALUATED, FAILED, INFEASIBLE
from vectornaut.explorer.descriptors import NOMINAL, ORDINAL, Axis, DescriptorSpace, normalize_token
from vectornaut.explorer.profiles.base import (
    EvaluationContext,
    EvaluationResult,
    ExplorerProfile,
    PreparedCandidate,
    finite,
)
from vectornaut.explorer.schemas import (
    BackOfEnvelope,
    DescriptorAssignment,
    MaterialsCandidate,
    MaterialsCandidateBatch,
)

MATERIALS_SPACE = DescriptorSpace([
    Axis("mechanism_class", NOMINAL, (
        "interfacial_slip", "flow_redirection", "trapped_gas_or_liquid", "porous_transport",
        "graded_stiffness", "architected_lattice", "radiative_control", "phase_change",
        "electrostatic_field", "other",
    ), description="The main physical mechanism the concept relies on.", value_help={
        "interfacial_slip": "changes the boundary condition at a wall (slip, lubrication, shear-free zones)",
        "flow_redirection": "geometry that steers flow or vortices (riblets, grooves, vanes)",
        "trapped_gas_or_liquid": "a retained layer of gas or liquid (plastron, infused lubricant)",
        "porous_transport": "transport through pores, capillaries or membranes",
        "graded_stiffness": "stiffness or density varying through the material",
        "architected_lattice": "a designed cellular/lattice geometry carrying load or heat",
        "radiative_control": "emission/absorption/reflection of radiation",
        "phase_change": "melting, evaporation, condensation or other latent heat",
        "electrostatic_field": "charges or electric fields do the work",
    }),
    Axis("length_scale", ORDINAL, ("nm", "sub_um", "um", "10_um", "100_um", "mm", "cm_plus"),
         description="Characteristic feature size of the structure (ordered from small to large).", value_help={
             "nm": "below 100 nm", "sub_um": "100 nm - 1 um", "um": "1 - 10 um", "10_um": "10 - 100 um",
             "100_um": "100 um - 1 mm", "mm": "1 - 10 mm", "cm_plus": "10 mm and above",
         }),
    Axis("inspiration_origin", NOMINAL, (
        "plant", "animal", "microbe", "geology", "atmosphere_ocean", "technology", "other",
    ), description="Where the idea is borrowed from."),
    Axis("governing_quantity", NOMINAL, (
        "wall_shear", "flow_rate", "heat_flux", "temperature", "deflection", "stress", "field_strength", "other",
    ), description="The quantity the evaluator must compute to judge the benefit."),
])

# Saturation scale of the gain score: a gain of GAIN_SCALE_PCT percent scores 1 - 1/e.
GAIN_SCALE_PCT = 20.0
# Estimated (not simulated) gains count half.
ESTIMATE_DISCOUNT = 0.5
W_VALIDITY = 0.4
W_GAIN = 0.6
VALIDATOR_STATUS_FACTOR = {"pass": 1.0, "warn": 0.8}
UNAVAILABLE_GAIN_BASES = {"n_a", "na", "none", "unavailable", "not_available", "not_applicable", "not_computed"}
PLACEHOLDER_SVG = (
    '<svg viewBox="0 0 400 200" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="400" height="200" rx="8" ry="8" fill="#0f172a"/>'
    '<text x="200" y="105" fill="#00f5d4" font-size="14" text-anchor="middle">explorer candidate</text></svg>'
)


def saturating(gain_pct: Optional[float]) -> float:
    if gain_pct is None or gain_pct <= 0:
        return 0.0
    return 1.0 - math.exp(-gain_pct / GAIN_SCALE_PCT)


def simulated_gain(simulator: Mapping[str, Any]) -> tuple:
    """
    (gain_pct or None, basis label). The gain counts as unavailable if it is missing or
    not finite, or if the simulator declares it so via a ``gain_basis``-like field
    ("n/a", "unavailable", ...), which newer solver versions report when no baseline exists.
    """
    basis = None
    for key in ("gain_basis", "performance_gain_basis", "baseline_basis"):
        if key in simulator and simulator.get(key) is not None:
            basis = simulator.get(key)
            break
    if isinstance(basis, str) and normalize_token(basis) in UNAVAILABLE_GAIN_BASES:
        return None, basis
    if isinstance(basis, Mapping) and normalize_token(basis.get("status", "")) in UNAVAILABLE_GAIN_BASES:
        return None, basis
    return finite(simulator.get("performance_gain_pct")), basis


def score_pipeline_result(result: Mapping[str, Any], candidate: Any) -> EvaluationResult:
    """Score 0..100 = 100 * (0.4 * validity + 0.6 * gain_score); breakdown stored explicitly."""
    simulator = result.get("simulator") or {}
    validation = result.get("validation") or {}
    v_status = str(validation.get("status") or "")
    v_score = finite(validation.get("score")) or 0.0
    validity = max(0.0, min(1.0, v_score)) * VALIDATOR_STATUS_FACTOR.get(v_status, 0.0)

    gain, gain_basis = simulated_gain(simulator)
    boe = getattr(candidate, "back_of_envelope", None)
    estimate = finite(getattr(boe, "value", None)) if boe is not None else None
    unit = str(getattr(boe, "unit", "") or "").strip() if boe is not None else ""
    estimate_is_pct = unit == "%" or normalize_token(unit) in ("pct", "percent", "percentage")

    if gain is not None:
        basis = "simulated"
        gain_score = saturating(gain)
    else:
        basis = "estimated, not simulated"
        gain_score = saturating(estimate if estimate_is_pct else None) * ESTIMATE_DISCOUNT

    score = 100.0 * (W_VALIDITY * validity + W_GAIN * gain_score)
    breakdown = {
        "basis": basis,
        "formula": "100 * (0.4 * validity + 0.6 * gain_score); gain_score = 1 - exp(-gain_pct / 20)"
                   + ("" if basis == "simulated" else " * 0.5 (estimate discount)"),
        "components": {"validity": round(validity, 6), "gain_score": round(gain_score, 6)},
        "pipeline_status": result.get("status"),
        "validator_status": v_status or None,
        "validator_score": v_score,
        "performance_gain_pct": gain,
        "gain_basis": gain_basis,
        "estimated_gain_pct": estimate if estimate_is_pct else None,
        "estimate_unit": getattr(boe, "unit", None) if boe is not None else None,
        "solver_method": simulator.get("solver_method"),
        "relative_error": finite(simulator.get("relative_error")),
        "primary_metric_value": finite(simulator.get("primary_metric_value")),
        "is_mock": bool(result.get("is_mock")),
    }
    return EvaluationResult(status=EVALUATED, score=round(score, 4), breakdown=breakdown)


def to_miner_concept(candidate: MaterialsCandidate) -> MinerConceptOutput:
    return MinerConceptOutput(
        design_name=candidate.title,
        inspiration_source=candidate.inspiration_source,
        domain=candidate.domain,
        physical_mechanism=candidate.physical_mechanism or candidate.summary,
        parameters=[ParameterProposal.model_validate(p.model_dump()) for p in candidate.parameters],
        svg_schematic=PLACEHOLDER_SVG,
    )


# ---------------------------------------------------------------------------
# mock generator: a fixed synthetic landscape, so the loop can be tested offline
# ---------------------------------------------------------------------------

_MOCK_MECHANISM = {
    "interfacial_slip": 1.0, "trapped_gas_or_liquid": 1.2, "flow_redirection": 0.7, "porous_transport": 0.5,
    "graded_stiffness": 0.4, "architected_lattice": 0.45, "radiative_control": 0.3, "phase_change": 0.35,
    "electrostatic_field": 0.3, "other": 0.25,
}
# Rises with feature size up to mm, then drops: a trend to extrapolate along, with an optimum.
_MOCK_SCALE = {"nm": 0.2, "sub_um": 0.35, "um": 0.55, "10_um": 0.8, "100_um": 1.1, "mm": 1.4, "cm_plus": 0.6}
_MOCK_ORIGIN = {"plant": 1.0, "animal": 0.9, "microbe": 1.15, "geology": 0.8, "atmosphere_ocean": 1.05,
                "technology": 0.85, "other": 0.7}
_MOCK_QUANTITY = {"wall_shear": 1.0, "flow_rate": 0.95}
# Where a model's prior would put free proposals (textbook cluster).
_MOCK_PRIOR = {
    "mechanism_class": ("interfacial_slip", "flow_redirection", "trapped_gas_or_liquid"),
    "length_scale": ("um", "10_um", "100_um"),
    "inspiration_origin": ("animal", "plant"),
    "governing_quantity": ("wall_shear",),
}


def mock_landscape(descriptors: Mapping[str, str]) -> float:
    return (
        _MOCK_MECHANISM[descriptors["mechanism_class"]]
        * _MOCK_SCALE[descriptors["length_scale"]]
        * _MOCK_ORIGIN[descriptors["inspiration_origin"]]
        * _MOCK_QUANTITY.get(descriptors["governing_quantity"], 0.5)
    )


def mock_infeasible_reason(descriptors: Mapping[str, str]) -> Optional[str]:
    if descriptors.get("mechanism_class") in ("radiative_control", "electrostatic_field") and \
            descriptors.get("governing_quantity") in ("wall_shear", "flow_rate"):
        return ("(mock) Radiation or static fields do not act on the momentum balance of a neutral, "
                "non-conducting liquid film at these scales.")
    return None


def _order_rng(order_id: str) -> random.Random:
    return random.Random(zlib.crc32(order_id.encode("utf-8")))


class MaterialsProfile(ExplorerProfile):
    name = "materials"
    title = "Bio- and physically inspired materials and surfaces"
    space = MATERIALS_SPACE
    report_axes = ("mechanism_class", "length_scale")
    batch_schema = MaterialsCandidateBatch
    textbook_solutions = (
        "Shark-skin riblets for drag reduction",
        "Lotus-leaf superhydrophobic surfaces",
        "Nepenthes (pitcher plant) slippery liquid-infused surfaces (SLIPS)",
        "Gecko-foot dry adhesion",
        "Namib desert beetle fog harvesting",
        "Honeycomb sandwich cores",
        "Moth-eye anti-reflection nanostructures",
    )

    def __init__(self, runner_factory: Optional[Callable[[], Any]] = None):
        # Tests inject a PipelineRunner with fake stages; default is the real pipeline.
        self.runner_factory = runner_factory

    def domain_brief(self) -> str:
        return ("materials, surfaces and microstructures inspired by nature or by distant technologies, "
                "judged by a physics simulation of the quantity they are meant to improve.")

    def evaluator_brief(self) -> str:
        return (
            "- The formulator turns each candidate into a steady 1D boundary value problem (second-order ODE on a\n"
            "  normalised interval) or a steady linear 2D PDE on a rectangle; the auditor checks physical\n"
            "  plausibility; analytical/SciPy/PINN/FDM solvers solve it; a deterministic validator checks the result.\n"
            "- Works well: thin-film and channel shear flow with slip or modified walls, heat conduction with sources\n"
            "  and fixed or flux boundaries, diffusion through layers, electrostatic potentials, beam deflection.\n"
            "- Cannot model: resolved turbulence, strongly transient behaviour, moving interfaces, chemistry,\n"
            "  detailed optics. Express such effects as an effective coefficient (slip length, conductivity,\n"
            "  heat-transfer coefficient) if at all."
        )

    def candidate_instructions(self) -> str:
        return (
            "- Materials fields: inspiration_source, domain, physical_mechanism, and 3-8 parameters (snake_case\n"
            "  names, SI values, min_bound < value < max_bound, a justification each) that are enough to write\n"
            "  the steady model.\n"
            "- back_of_envelope.value: the estimated improvement of the governing quantity over a plain, untreated\n"
            "  baseline in percent (unit '%'); it is used only if the simulation cannot compute a gain."
        )

    def sanity_issues(self, candidate: Any, descriptors: Mapping[str, str]) -> List[str]:
        issues = []
        if not (candidate.physical_mechanism or "").strip():
            issues.append("physical_mechanism is empty")
        if not (candidate.domain or "").strip():
            issues.append("domain is empty")
        if not candidate.parameters:
            issues.append("no parameters: the evaluator cannot build a model")
        names = set()
        for param in candidate.parameters:
            values = [finite(param.value), finite(param.min_bound), finite(param.max_bound)]
            if any(v is None for v in values):
                issues.append(f"parameter {param.name}: value or bounds not finite")
                continue
            value, low, high = values
            if low >= high:
                issues.append(f"parameter {param.name}: min_bound {low:g} >= max_bound {high:g}")
            elif not low <= value <= high:
                issues.append(f"parameter {param.name}: value {value:g} outside [{low:g}, {high:g}]")
            if param.name in names:
                issues.append(f"parameter {param.name} listed twice")
            names.add(param.name)
        return issues

    # ---- mock ---------------------------------------------------------
    def mock_candidate(self, order: Mapping[str, Any]) -> MaterialsCandidate:
        order_id = order["order_id"]
        rng = _order_rng(order_id)
        target = dict(order.get("target") or {})
        descriptors = {}
        for axis in self.space.axes:
            if axis.name in target:
                descriptors[axis.name] = target[axis.name]
            else:
                descriptors[axis.name] = rng.choice(_MOCK_PRIOR[axis.name])
        if self.space.is_full(target):
            reason = mock_infeasible_reason(target)
            if reason:
                return MaterialsCandidate(order_id=order_id, target_feasible=False, infeasibility_reason=reason)
        if order.get("strategy") == "explore" and zlib.crc32(order_id.encode()) % 5 == 0:
            # Honest off-target labelling: the mock "model" drifts back towards its prior.
            descriptors["inspiration_origin"] = "animal"
        if mock_infeasible_reason(descriptors):
            descriptors["governing_quantity"] = "heat_flux"

        spacing = mock_landscape(descriptors)
        if order.get("strategy") == "refine":
            spacing *= rng.uniform(0.95, 1.15)
        spacing = round(max(spacing, 0.005), 6)
        height = round(spacing / 2.0, 6)
        label = " / ".join(descriptors[name] for name in self.space.names)
        return MaterialsCandidate(
            order_id=order_id,
            title=f"Mock {label} concept ({order_id})",
            summary=f"Mock concept for {label}.",
            descriptors=[DescriptorAssignment(axis=k, value=v) for k, v in descriptors.items()],
            back_of_envelope=BackOfEnvelope(quantity="wall shear reduction", formula="gain ~ 10 %/unit * spacing",
                                            value=round(10.0 * spacing, 3), unit="%"),
            main_risk="mock risk: structure wears off",
            novelty_vs_known="mock: differs from riblets by construction",
            inspiration_source=f"mock {descriptors['inspiration_origin']} source",
            domain="Fluid Dynamics",
            physical_mechanism=f"Mock mechanism {descriptors['mechanism_class']} at {descriptors['length_scale']}.",
            parameters=[
                ParameterProposal(name="riblet_height", value=height, min_bound=height / 10, max_bound=height * 10,
                                  justification="mock"),
                ParameterProposal(name="riblet_spacing", value=spacing, min_bound=spacing / 10, max_bound=spacing * 10,
                                  justification="mock"),
                ParameterProposal(name="viscosity", value=0.001, min_bound=0.0001, max_bound=0.01, justification="mock"),
                ParameterProposal(name="free_stream_velocity", value=1.5, min_bound=0.1, max_bound=5.0,
                                  justification="mock"),
                ParameterProposal(name="pressure_gradient", value=2.0, min_bound=0.0, max_bound=10.0,
                                  justification="mock"),
            ],
        )

    def mock_batch(self, query: str, orders: Sequence[Mapping[str, Any]]) -> MaterialsCandidateBatch:
        return MaterialsCandidateBatch(
            function_analysis="(mock) reduce the momentum transfer at the wall",
            mechanism_classes_considered=["interfacial_slip", "flow_redirection"],
            analogues_considered=["(mock) fish mucus", "(mock) glacier basal sliding"],
            candidates=[self.mock_candidate(order) for order in orders],
        )

    # ---- evaluation ---------------------------------------------------
    def _runner(self):
        if self.runner_factory is not None:
            return self.runner_factory()
        from vectornaut.pipeline import PipelineRunner
        return PipelineRunner()

    def evaluate_one(self, item: PreparedCandidate, ctx: EvaluationContext) -> EvaluationResult:
        from vectornaut.pipeline import ConceptsExhaustedError, PipelineRunRequest

        request = PipelineRunRequest(
            query=ctx.query,
            concept=to_miner_concept(item.candidate),
            is_mock=ctx.mock,
            epochs=ctx.epochs,
            max_optimization_rounds=ctx.opt_rounds,
            max_concept_attempts=1,
        )
        log = io.StringIO()
        try:
            if ctx.verbose:
                result = self._runner().run(request)
            else:
                with contextlib.redirect_stdout(log):
                    result = self._runner().run(request)
        except ConceptsExhaustedError as err:
            reasons = [c.get("reason") for c in err.failed_concepts if c.get("reason")]
            stages = [c.get("stage") for c in err.failed_concepts if c.get("stage")]
            return EvaluationResult(
                status=FAILED, reason="; ".join(reasons) or str(err),
                breakdown={"basis": "pipeline failed", "failed_stage": stages[0] if stages else None},
                raw={"error": str(err), "failed_concepts": err.failed_concepts, "log_tail": log.getvalue()[-4000:]},
            )
        except Exception as err:  # NeedResponse (replay pause) is a BaseException and passes through.
            return EvaluationResult(
                status=FAILED, reason=f"{type(err).__name__}: {err}", breakdown={"basis": "pipeline error"},
                raw={"error": f"{type(err).__name__}: {err}", "log_tail": log.getvalue()[-4000:]},
            )

        raw = {"pipeline_result": result, "log_tail": log.getvalue()[-4000:]}
        if result.get("status") == "rejected":
            rejection = result.get("rejection") or {}
            return EvaluationResult(
                status=INFEASIBLE, reason=rejection.get("reason") or "pipeline rejected the concept as infeasible",
                breakdown={"basis": "rejected", "rejected_by": rejection.get("stage")}, raw=raw,
            )
        scored = score_pipeline_result(result, item.candidate)
        scored.raw = raw
        return scored

    def evaluate(self, items: Sequence[PreparedCandidate], ctx: EvaluationContext) -> List[EvaluationResult]:
        return [self.evaluate_one(item, ctx) for item in items]
