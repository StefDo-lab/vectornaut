# -*- coding: utf-8 -*-
"""
Materials profile: bio- and physically inspired materials, surfaces and microstructures.

Evaluation runs every candidate through the existing pipeline (formulator -> auditor ->
solver -> validator -> optimizer -> synthesizer) via ``PipelineRunRequest.concept``, with
re-mining disabled (``max_concept_attempts=1``) so that exactly this concept is judged.
A critic call per batch (``get_model_name("critic")``) then sees each candidate's concept,
formulation, audited parameters, metric, baseline, simulated gain and own estimate, and
returns a plausible real-world gain, assumption issues, killer risks and a 0..1 rating per
requirement of the request. The score combines both (see "scoring" below).
"""
import contextlib
import io
import math
import random
import zlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from vectornaut.config import MinerConceptOutput, ParameterProposal, get_client, get_model_name, get_thinking_config
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
    MaterialsCriticBatch,
    MaterialsCriticReview,
    Requirement,
    RequirementRating,
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

# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
#
#   score = 100 * gate * (W_GAIN * gain_score + W_REQ * requirement_coverage)
#   gate  = validator_score * (1.0 pass | 0.8 warn | 0 fail or missing)
#   gain_score = 1 - exp(-used_gain_pct / GAIN_SCALE_PCT)        (0 for gains <= 0)
#                * ESTIMATE_DISCOUNT                              (estimated tier only)
#   estimated tier: score <= ESTIMATED_SCORE_CAP
#
# Evidence tiers: "simulated" (a usable simulated gain) ranks above "estimated" (no usable
# simulated gain; the critic's and the candidate's estimates are used). An estimate that
# replaced an implausible simulated gain ranks lowest (see archive.rank_key).

# Saturation scale of the gain score: a gain of GAIN_SCALE_PCT percent scores 1 - 1/e.
GAIN_SCALE_PCT = 20.0
W_GAIN = 0.5
W_REQ = 0.5
# Estimated (not simulated) gains count half, and an estimated entry scores at most this.
ESTIMATE_DISCOUNT = 0.5
ESTIMATED_SCORE_CAP = 50.0
# Requirement coverage used when no critic rated the candidate.
REQ_UNRATED_DEFAULT = 0.5
# A simulated gain beyond this (in percent, absolute) is treated as an artefact (the
# validator's physics_performance_gain_sanity check uses the same bound).
MAX_PLAUSIBLE_GAIN_PCT = 500.0
# Simulated and critic gain differing by more than this factor -> use the lower one.
SENSITIVITY_RATIO = 2.0
GAIN_SANITY_CHECK = "physics_performance_gain_sanity"
VALIDATOR_STATUS_FACTOR = {"pass": 1.0, "warn": 0.8}
UNAVAILABLE_GAIN_BASES = {"n_a", "na", "none", "unavailable", "not_available", "not_applicable", "not_computed"}

TIER_SIMULATED = "simulated"
TIER_ESTIMATED = "estimated"
EVIDENCE_RANK = {TIER_SIMULATED: 2, TIER_ESTIMATED: 1}
IMPLAUSIBLE_RANK = 0

# Flags stored in score_breakdown["flags"].
FLAG_IMPLAUSIBLE = "implausible_gain"
FLAG_SENSITIVE = "model_assumption_sensitive"
FLAG_GAIN_UNAVAILABLE = "gain_unavailable"
FLAG_REQ_UNRATED = "requirements_unrated"
FLAG_REQ_PARTIAL = "requirements_partially_rated"
FLAG_CRITIC_MISSING = "critic_missing"
FLAG_CRITIC_FAILED = "critic_failed"
FLAG_BASELINE_UNSTATED = "baseline_unstated"

PLACEHOLDER_SVG = (
    '<svg viewBox="0 0 400 200" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="400" height="200" rx="8" ry="8" fill="#0f172a"/>'
    '<text x="200" y="105" fill="#00f5d4" font-size="14" text-anchor="middle">explorer candidate</text></svg>'
)


def saturating(gain_pct: Optional[float]) -> float:
    if gain_pct is None or gain_pct <= 0:
        return 0.0
    return 1.0 - math.exp(-gain_pct / GAIN_SCALE_PCT)


def score_formula(estimated: bool) -> str:
    """The formula as applied, generated from the constants (so it matches the code)."""
    gain = f"gain_score = 1 - exp(-used_gain_pct / {GAIN_SCALE_PCT:g})"
    if estimated:
        gain = f"gain_score = (1 - exp(-used_gain_pct / {GAIN_SCALE_PCT:g})) * {ESTIMATE_DISCOUNT:g} (estimate discount)"
    text = (f"score = 100 * gate * ({W_GAIN:g} * gain_score + {W_REQ:g} * requirement_coverage); "
            f"gate = validator_score * (1 pass / 0.8 warn / 0 fail); {gain}")
    if estimated:
        text += f"; estimated tier capped at {ESTIMATED_SCORE_CAP:g}"
    return text


def _raw_simulated_gain(simulator: Mapping[str, Any]) -> tuple:
    """(raw value as reported, basis label, declared unavailable?)."""
    basis = None
    for key in ("gain_basis", "performance_gain_basis", "baseline_basis"):
        if key in simulator and simulator.get(key) is not None:
            basis = simulator.get(key)
            break
    unavailable = False
    if isinstance(basis, str) and normalize_token(basis) in UNAVAILABLE_GAIN_BASES:
        unavailable = True
    if isinstance(basis, Mapping) and normalize_token(basis.get("status", "")) in UNAVAILABLE_GAIN_BASES:
        unavailable = True
    return simulator.get("performance_gain_pct"), basis, unavailable


def simulated_gain(simulator: Mapping[str, Any]) -> tuple:
    """
    (gain_pct or None, basis label). The gain counts as unavailable if it is missing or
    not finite, or if the simulator declares it so via a ``gain_basis``-like field
    ("n/a", "none", ...), which the solver reports when no baseline exists.
    """
    raw, basis, unavailable = _raw_simulated_gain(simulator)
    return (None if unavailable else finite(raw)), basis


def gain_sanity_check(validation: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The validator's gain-sanity check as {passed, detail}, or None if it did not run."""
    for check in validation.get("checks") or []:
        if not isinstance(check, Mapping) or check.get("name") != GAIN_SANITY_CHECK:
            continue
        passed = check.get("passed")
        if passed is None and check.get("status") is not None:
            passed = str(check.get("status")).lower() == "pass"
        return {"passed": bool(passed) if passed is not None else None, "detail": check.get("detail")}
    return None


def assess_simulated_gain(result: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Decides whether the simulated gain may count. Returns {usable_gain, raw_gain, basis,
    flags, sanity}. A gain counts only if it is reported, finite, not declared unavailable,
    within +-MAX_PLAUSIBLE_GAIN_PCT and the validator's gain-sanity check did not fail.
    """
    simulator = result.get("simulator") or {}
    validation = result.get("validation") or {}
    raw, basis, unavailable = _raw_simulated_gain(simulator)
    sanity = gain_sanity_check(validation)
    flags: List[str] = []
    usable = None
    value = finite(raw)
    if unavailable or raw is None:
        flags.append(FLAG_GAIN_UNAVAILABLE)
    elif value is None:
        flags.append(FLAG_IMPLAUSIBLE)            # reported, but NaN/inf/not a number
    elif abs(value) > MAX_PLAUSIBLE_GAIN_PCT or (sanity is not None and sanity["passed"] is False):
        flags.append(FLAG_IMPLAUSIBLE)
    else:
        usable = value
    raw_number = value if value is not None else (None if raw is None else str(raw))
    return {"usable_gain": usable, "raw_gain": raw_number, "basis": basis, "flags": flags, "sanity": sanity}


def differs_strongly(a: float, b: float, ratio: float = SENSITIVITY_RATIO) -> bool:
    """True if two gains differ by more than ``ratio`` (or have different signs)."""
    if a <= 0 or b <= 0:
        return (a > 0) != (b > 0)
    return max(a, b) / min(a, b) > ratio


def _estimate_pct(candidate: Any) -> tuple:
    boe = getattr(candidate, "back_of_envelope", None)
    if boe is None:
        return None, None
    unit = str(getattr(boe, "unit", "") or "").strip()
    is_pct = unit == "%" or normalize_token(unit) in ("pct", "percent", "percentage")
    return (finite(getattr(boe, "value", None)) if is_pct else None), getattr(boe, "unit", None)


def requirement_coverage(requirements: Sequence[Mapping[str, Any]], review: Any) -> Dict[str, Any]:
    """
    Mean critic coverage (0..1) over the stored requirements. Requirements the critic did
    not rate count REQ_UNRATED_DEFAULT; without any rating the coverage is
    REQ_UNRATED_DEFAULT and the entry is flagged. Without a stored list, the critic's own
    ratings are averaged.
    """
    ratings: Dict[str, Dict[str, Any]] = {}
    for rating in (getattr(review, "requirement_coverage", None) or []) if review is not None else []:
        value = finite(getattr(rating, "coverage", None))
        name = str(getattr(rating, "name", "") or "").strip()
        if not name or value is None:
            continue
        ratings.setdefault(normalize_token(name), {"name": name, "coverage": max(0.0, min(1.0, value)),
                                                   "reason": getattr(rating, "reason", "") or ""})
    flags: List[str] = []
    rows: List[Dict[str, Any]] = []
    if requirements:
        for req in requirements:
            found = ratings.get(normalize_token(req.get("name", "")))
            rows.append({"name": req.get("name"), "coverage": found["coverage"] if found else None,
                         "reason": found["reason"] if found else ""})
    else:
        rows = [dict(r) for r in ratings.values()]
    rated = [r["coverage"] for r in rows if r["coverage"] is not None]
    if not rated:
        flags.append(FLAG_REQ_UNRATED)
        coverage = REQ_UNRATED_DEFAULT
    else:
        if len(rated) < len(rows):
            flags.append(FLAG_REQ_PARTIAL)
        coverage = sum(r["coverage"] if r["coverage"] is not None else REQ_UNRATED_DEFAULT for r in rows) / len(rows)
    return {"coverage": coverage, "rows": rows, "flags": flags}


def score_pipeline_result(result: Mapping[str, Any], candidate: Any, review: Any = None,
                          requirements: Sequence[Mapping[str, Any]] = (), critic_status: Optional[str] = None
                          ) -> EvaluationResult:
    """
    Scores one pipeline result (0..100, formula in ``score_formula``). ``review`` is the
    materials critic's review (MaterialsCriticReview) or None; ``critic_status`` is
    'missing' / 'failed' when a critic was asked but gave nothing for this candidate.
    """
    simulator = result.get("simulator") or {}
    validation = result.get("validation") or {}
    v_status = str(validation.get("status") or "")
    v_score = finite(validation.get("score")) or 0.0
    gate = max(0.0, min(1.0, v_score)) * VALIDATOR_STATUS_FACTOR.get(v_status, 0.0)

    sim = assess_simulated_gain(result)
    flags = list(sim["flags"])
    estimate, estimate_unit = _estimate_pct(candidate)
    critic_gain = finite(getattr(review, "plausible_gain_pct", None)) if review is not None else None

    if sim["usable_gain"] is not None:
        tier = TIER_SIMULATED
        simulated = sim["usable_gain"]
        used, source = simulated, "simulation"
        if critic_gain is not None and differs_strongly(simulated, critic_gain):
            flags.append(FLAG_SENSITIVE)
            if critic_gain < simulated:
                used, source = critic_gain, "critic (simulation differs by more than %gx)" % SENSITIVITY_RATIO
            else:
                source = "simulation (critic differs by more than %gx)" % SENSITIVITY_RATIO
        gain_score = saturating(used)
        basis = "simulated" if source == "simulation" else "simulated, critic-checked"
    else:
        tier = TIER_ESTIMATED
        options = [(v, name) for v, name in ((critic_gain, "critic"), (estimate, "candidate estimate")) if v is not None]
        if len(options) == 2:
            used = min(critic_gain, estimate)
            source = "min(critic, candidate estimate) = " + ("critic" if critic_gain <= estimate else "candidate estimate")
        elif options:
            used, source = options[0]
        else:
            used, source = None, "none"
        gain_score = saturating(used) * ESTIMATE_DISCOUNT
        basis = "estimated, not simulated"

    req = requirement_coverage(requirements, review)
    flags.extend(req["flags"])
    if critic_status == "failed":
        flags.append(FLAG_CRITIC_FAILED)
    elif critic_status == "missing":
        flags.append(FLAG_CRITIC_MISSING)
    baseline = str(getattr(candidate, "baseline", "") or "").strip()
    if not baseline:
        flags.append(FLAG_BASELINE_UNSTATED)

    score = 100.0 * gate * (W_GAIN * gain_score + W_REQ * req["coverage"])
    capped = False
    if tier == TIER_ESTIMATED and score > ESTIMATED_SCORE_CAP:
        score, capped = ESTIMATED_SCORE_CAP, True
    rank = EVIDENCE_RANK[tier]
    if tier == TIER_ESTIMATED and FLAG_IMPLAUSIBLE in flags:
        rank = IMPLAUSIBLE_RANK

    critic_block = None
    if review is not None:
        critic_block = {
            "plausible_gain_pct": critic_gain,
            "plausible_gain_reasoning": getattr(review, "plausible_gain_reasoning", "") or "",
            "key_assumption_issues": list(getattr(review, "key_assumption_issues", None) or []),
            "killer_risks": list(getattr(review, "killer_risks", None) or []),
        }
    breakdown = {
        "basis": basis,
        "evidence_tier": tier,
        "evidence_rank": rank,
        "flags": flags,
        "formula": score_formula(tier == TIER_ESTIMATED),
        "weights": {"gain": W_GAIN, "requirements": W_REQ},
        "components": {"gate": round(gate, 6), "gain_score": round(gain_score, 6),
                       "requirement_coverage": round(req["coverage"], 6)},
        "capped": capped,
        "requirement_coverage": round(req["coverage"], 4),
        "requirements": req["rows"],
        "used_gain_pct": used,
        "gain_source": source,
        "simulated_gain_pct": sim["raw_gain"],
        "performance_gain_pct": sim["usable_gain"],
        "critic_plausible_gain_pct": critic_gain,
        "estimated_gain_pct": estimate,
        "estimate_unit": estimate_unit,
        "estimate_baseline": baseline or None,
        "gain_basis": sim["basis"],
        "gain_sanity": sim["sanity"],
        "critic": critic_block,
        "pipeline_status": result.get("status"),
        "validator_status": v_status or None,
        "validator_score": v_score,
        "solver_method": simulator.get("solver_method"),
        "relative_error": finite(simulator.get("relative_error")),
        "primary_metric_value": finite(simulator.get("primary_metric_value")),
        "is_mock": bool(result.get("is_mock")),
    }
    return EvaluationResult(status=EVALUATED, score=round(score, 4), breakdown=breakdown)


def pipeline_mechanism_text(candidate: MaterialsCandidate) -> str:
    """
    ``physical_mechanism`` handed to the pipeline: the mechanism plus, compactly and labelled,
    the summary, the comparison baseline, the candidate's estimate (value, formula), the main
    risk and the novelty statement, so the formulator and auditor see the candidate's
    baseline and assumptions (MinerConceptOutput has no fields for them).
    """
    def clean(text: Any) -> str:
        return " ".join(str(text or "").split())

    mechanism = clean(candidate.physical_mechanism)
    summary = clean(candidate.summary)
    parts = [mechanism or summary]
    if mechanism and summary and summary != mechanism:
        parts.append(f"Summary: {summary}")
    baseline = clean(getattr(candidate, "baseline", ""))
    if baseline:
        parts.append(f"Comparison baseline: {baseline}")
    boe = candidate.back_of_envelope
    if boe is not None:
        value = finite(boe.value)
        value_text = f"{value:g}" if value is not None else str(boe.value)
        estimate = f"Candidate estimate: {clean(boe.quantity) or 'benefit'} = {value_text} {clean(boe.unit)}".rstrip()
        if clean(boe.formula):
            estimate += f" ({clean(boe.formula)})"
        parts.append(estimate)
    if clean(candidate.main_risk):
        parts.append(f"Main risk: {clean(candidate.main_risk)}")
    if clean(candidate.novelty_vs_known):
        parts.append(f"Novelty vs known: {clean(candidate.novelty_vs_known)}")
    return " ".join(part.rstrip(".") + "." for part in parts if part)


def to_miner_concept(candidate: MaterialsCandidate) -> MinerConceptOutput:
    mechanism = pipeline_mechanism_text(candidate)
    return MinerConceptOutput(
        design_name=candidate.title,
        inspiration_source=candidate.inspiration_source,
        domain=candidate.domain,
        physical_mechanism=mechanism,
        parameters=[ParameterProposal.model_validate(p.model_dump()) for p in candidate.parameters],
        svg_schematic=PLACEHOLDER_SVG,
    )


# ---------------------------------------------------------------------------
# critic: one call per batch, after the pipeline ran
# ---------------------------------------------------------------------------

def _short(text: Any, limit: int = 400) -> str:
    text = " ".join(str(text if text is not None else "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def critic_candidate_block(space: DescriptorSpace, item: PreparedCandidate, result: Mapping[str, Any]) -> str:
    """What the critic sees about one candidate: concept, formulation, parameters, metric, gains."""
    c = item.candidate
    model = result.get("miner") or {}
    auditor = result.get("auditor") or {}
    simulator = result.get("simulator") or {}
    sim = assess_simulated_gain(result)
    params = ", ".join(f"{p.get('name')}={p.get('value'):g}" if isinstance(p.get("value"), (int, float))
                       else f"{p.get('name')}={p.get('value')}"
                       for p in (auditor.get("audited_parameters") or []) if isinstance(p, Mapping))
    baseline_params = ", ".join(f"{p.get('name')}={p.get('value')}" for p in (auditor.get("baseline_parameters") or [])
                                if isinstance(p, Mapping))
    metric = simulator.get("metric_spec") or auditor.get("metric") or {}
    metric_text = ", ".join(f"{k}={v}" for k, v in metric.items() if v is not None) if isinstance(metric, Mapping) else str(metric)
    boe = c.back_of_envelope
    raw = sim["raw_gain"]
    if sim["usable_gain"] is not None:
        gain_text = f"{sim['usable_gain']:.4g} %"
    elif FLAG_IMPLAUSIBLE in sim["flags"]:
        gain_text = f"{raw} % (REJECTED as implausible: {(sim['sanity'] or {}).get('detail') or 'outside +-500 %'})"
    else:
        gain_text = "not available (no baseline could be simulated)"
    lines = [
        f"[{item.order['order_id']}] {c.title}",
        f"  cell: {space.describe(item.descriptors)}",
        f"  concept: {_short(c.physical_mechanism or c.summary, 600)}",
        f"  candidate's baseline: {_short(getattr(c, 'baseline', '') or '(not stated)', 200)}",
        f"  candidate's estimate: {boe.value:g} {boe.unit} ({_short(boe.quantity, 120)}; {_short(boe.formula, 240)})"
        if boe is not None else "  candidate's estimate: (none)",
        f"  formulation: {_short(model.get('governing_equation'), 200)}; BCs: "
        f"{_short('; '.join(str(b) for b in (model.get('boundary_conditions') or [])), 300)}",
        f"  audited parameters: {_short(params, 500) or '(none)'}",
        f"  metric: {_short(metric_text, 240) or '(default)'}; simulated baseline: "
        f"{_short(auditor.get('baseline_description') or simulator.get('gain_basis') or '(none)', 160)}"
        + (f" with {_short(baseline_params, 200)}" if baseline_params else ""),
        f"  simulated gain: {gain_text}",
        f"  stated main risk: {_short(c.main_risk, 240)}",
    ]
    return "\n".join(lines)


def critic_prompt(space: DescriptorSpace, query: str, items: Sequence[PreparedCandidate],
                  results: Sequence[Mapping[str, Any]], requirements: Sequence[Mapping[str, Any]]) -> str:
    reqs = "\n".join(f"- {r['name']}: {r.get('criterion') or ''}" for r in requirements) or \
        "- (none extracted: rate the requirements you read from the request, with short snake_case names)"
    blocks = "\n".join(critic_candidate_block(space, item, result) for item, result in zip(items, results))
    return f"""You are the Critic of Vectornaut's materials explorer. Every candidate below was turned into a
simplified steady model (1D/2D) and simulated. Simulated gains depend on modelling choices (an
equivalent gap or slip length picked to match a target, a laminar model standing in for turbulent
flow, a baseline that is not the real alternative). Judge how much of each benefit is real and how
well each concept meets the request's requirements.

REQUEST: "{query}"

REQUIREMENTS:
{reqs}

For every candidate:
- plausible_gain_pct: your best estimate of the real-world improvement of the request's main benefit,
  in percent, against the conventional state-of-the-art solution (e.g. a clean, standard hull coating;
  not a fouled or untreated surface). Use known values of comparable concepts. null if you cannot
  estimate it. Do not copy the simulated number.
- plausible_gain_reasoning: one or two sentences.
- key_assumption_issues: the modelling choices that drive the simulated gain, most important first.
- killer_risks: what could make the concept unworkable in practice, most severe first.
- requirement_coverage: one rating per requirement (name exactly as listed, coverage 0..1, a one-line
  reason). Count the whole system: energy and material consumption, maintenance, what is released
  into the sea, service life.

CANDIDATES
{blocks}
"""


# ---------------------------------------------------------------------------
# mock generator: a fixed synthetic landscape, so the loop can be tested offline
# ---------------------------------------------------------------------------


MOCK_REQUIREMENTS = (
    {"name": "low_friction_drag", "criterion": "lower frictional drag than a clean standard hull coating"},
    {"name": "non_toxic_antifouling", "criterion": "keeps fouling off without releasing biocides"},
    {"name": "multi_year_durability", "criterion": "works for several years in seawater"},
)
# Mock critic: requirement coverage per mechanism class (durability/consumption proxies).
_MOCK_COVERAGE = {
    "interfacial_slip": 0.6, "trapped_gas_or_liquid": 0.4, "flow_redirection": 0.8, "porous_transport": 0.5,
    "graded_stiffness": 0.7, "architected_lattice": 0.6, "radiative_control": 0.3, "phase_change": 0.3,
    "electrostatic_field": 0.4, "other": 0.5,
}
# Mock critic: plausible gain = simulated gain * factor (factor chosen by title hash).
_MOCK_CRITIC_FACTORS = (1.0, 0.8, 0.3, 1.2)

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
    relevance_axes = ("governing_quantity",)
    textbook_solutions = (
        "Shark-skin riblets for drag reduction",
        "Lotus-leaf superhydrophobic surfaces",
        "Nepenthes (pitcher plant) slippery liquid-infused surfaces (SLIPS)",
        "Gecko-foot dry adhesion",
        "Namib desert beetle fog harvesting",
        "Honeycomb sandwich cores",
        "Moth-eye anti-reflection nanostructures",
    )

    def __init__(self, runner_factory: Optional[Callable[[], Any]] = None, critic_client: Any = None):
        # Tests inject a PipelineRunner with fake stages; default is the real pipeline.
        self.runner_factory = runner_factory
        self.critic_client = critic_client

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
            "- baseline: name the baseline your estimate compares against. Default (use it unless the request\n"
            "  says otherwise): the conventional state-of-the-art solution for the request, e.g. for a hull\n"
            "  coating a clean, standard foul-release or antifouling hull coating - not a fouled or untreated\n"
            "  hull. Use the same kind of baseline for every candidate so the estimates stay comparable.\n"
            "- back_of_envelope.value: the estimated improvement of the request's main benefit over that\n"
            "  baseline in percent (unit '%'). A critic compares it with the simulated gain; if the simulation\n"
            "  cannot compute a gain, the lower of your and the critic's estimate is used at half weight.\n"
            "- Every candidate is also rated against each requirement of the request (e.g. no toxic release,\n"
            "  service life, energy or material consumption): a concept that buys its benefit by violating a\n"
            "  requirement scores low."
        )

    def analysis_instructions(self, archive: Any) -> str:
        stored = list((archive.request_analysis.get("relevant") or {}).get("governing_quantity") or [])
        if stored:
            return ("RELEVANT GOVERNING QUANTITIES (from the function analysis; explore and diversify orders only use\n"
                    f"these and the quantities of elites): {', '.join(stored)}. Repeat or extend them in\n"
                    "relevant_governing_quantities.\n")
        return ("In step 1 also list relevant_governing_quantities: the governing_quantity tokens that measure a\n"
                "benefit the request actually asks for (a drag request: wall_shear, perhaps flow_rate; a durability\n"
                "or adhesion requirement: stress). Explore orders only target cells with these quantities.\n")

    def relevant_values_from_batch(self, batch: Any) -> Tuple[Dict[str, List[str]], List[str]]:
        axis = self.space.axis("governing_quantity")
        values, dropped = [], []
        for raw in getattr(batch, "relevant_governing_quantities", None) or []:
            token = normalize_token(raw)
            if token in axis.values:
                if token not in values:
                    values.append(token)
            else:
                dropped.append(str(raw))
        return ({"governing_quantity": values} if values else {}), dropped

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
            baseline="(mock) clean standard foul-release hull coating",
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
            requirements=[Requirement(**r) for r in MOCK_REQUIREMENTS],
            relevant_governing_quantities=["wall_shear", "flow_rate"],
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

    def run_pipeline(self, item: PreparedCandidate, ctx: EvaluationContext) -> EvaluationResult:
        """
        Runs the pipeline for one candidate. Returns a FAILED or INFEASIBLE result, or an
        EVALUATED placeholder (no score yet) whose ``raw["pipeline_result"]`` holds the result.
        """
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
        return EvaluationResult(status=EVALUATED, score=None, raw=raw)

    # ---- critic -------------------------------------------------------
    @staticmethod
    def mock_review(item: PreparedCandidate, result: Mapping[str, Any],
                    requirements: Sequence[Mapping[str, Any]]) -> MaterialsCriticReview:
        """Deterministic stand-in: scales the simulated gain (sometimes by 0.3 -> 'sensitive')."""
        key = zlib.crc32(item.candidate.title.encode("utf-8"))
        factor = _MOCK_CRITIC_FACTORS[key % len(_MOCK_CRITIC_FACTORS)]
        usable = assess_simulated_gain(result)["usable_gain"]
        estimate, _ = _estimate_pct(item.candidate)
        base = usable if usable is not None else (estimate if estimate is not None else 0.0)
        coverage = _MOCK_COVERAGE.get(item.descriptors.get("mechanism_class"), 0.5)
        names = [r["name"] for r in requirements] or [r["name"] for r in MOCK_REQUIREMENTS]
        ratings = [RequirementRating(name=name, coverage=round(max(0.0, min(1.0, coverage + 0.1 * ((key + i) % 3 - 1))), 3),
                                     reason="(mock) rating from the mechanism class")
                   for i, name in enumerate(names)]
        return MaterialsCriticReview(
            order_id=item.order["order_id"], plausible_gain_pct=round(base * factor, 4),
            plausible_gain_reasoning=f"(mock) simulated or estimated gain x {factor}",
            key_assumption_issues=["(mock) effective slip length chosen, not derived"] if factor < 0.5 else [],
            killer_risks=["(mock) coating wears off within a season"],
            requirement_coverage=ratings,
        )

    def run_critic(self, items: Sequence[PreparedCandidate], results: Sequence[Mapping[str, Any]],
                   ctx: EvaluationContext) -> Dict[str, MaterialsCriticReview]:
        """One critic call for the batch; returns reviews by order_id."""
        requirements = list(ctx.requirements or [])
        if ctx.mock:
            reviews = [self.mock_review(item, result, requirements) for item, result in zip(items, results)]
        else:
            from google.genai import types

            client = ctx.critic_client or self.critic_client or get_client()
            response = client.models.generate_content(
                model=get_model_name("critic"),
                contents=critic_prompt(self.space, ctx.query, items, results, requirements),
                config=types.GenerateContentConfig(
                    thinking_config=get_thinking_config("critic", default="medium"),
                    response_mime_type="application/json",
                    response_schema=MaterialsCriticBatch,
                ),
            )
            parsed = getattr(response, "parsed", None)
            if parsed is None:
                parsed = MaterialsCriticBatch.model_validate_json(response.text)
            elif not isinstance(parsed, MaterialsCriticBatch):
                parsed = MaterialsCriticBatch.model_validate(parsed if isinstance(parsed, dict) else parsed.model_dump())
            reviews = list(parsed.reviews)
        wanted = {item.order["order_id"] for item in items}
        found: Dict[str, MaterialsCriticReview] = {}
        for review in reviews:
            if review.order_id in wanted and review.order_id not in found:
                found[review.order_id] = review
        return found

    # ---- evaluation ---------------------------------------------------
    def evaluate(self, items: Sequence[PreparedCandidate], ctx: EvaluationContext) -> List[EvaluationResult]:
        results = [self.run_pipeline(item, ctx) for item in items]
        done = [(item, res) for item, res in zip(items, results) if res.status == EVALUATED]
        reviews: Dict[str, MaterialsCriticReview] = {}
        critic_error = None
        if ctx.use_critic and done:
            try:
                reviews = self.run_critic([i for i, _ in done], [r.raw["pipeline_result"] for _, r in done], ctx)
            except Exception as err:  # a failed critic must not lose the simulations; NeedResponse passes.
                critic_error = f"{type(err).__name__}: {err}"
        for item, res in done:
            review = reviews.get(item.order["order_id"])
            status = None
            if ctx.use_critic and review is None:
                status = "failed" if critic_error else "missing"
            scored = score_pipeline_result(res.raw["pipeline_result"], item.candidate, review=review,
                                           requirements=ctx.requirements, critic_status=status)
            scored.breakdown["critic_used"] = review is not None
            if critic_error:
                scored.breakdown["critic_error"] = critic_error[:300]
            res.score, res.breakdown = scored.score, scored.breakdown
            res.raw["critic_review"] = review.model_dump() if review is not None else None
            res.raw["score_breakdown"] = scored.breakdown
        return results
