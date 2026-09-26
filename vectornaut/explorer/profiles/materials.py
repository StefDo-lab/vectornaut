# -*- coding: utf-8 -*-
"""
Materials profile: bio- and physically inspired materials, surfaces and microstructures.

Evaluation runs every candidate through the existing pipeline (formulator -> auditor ->
solver -> validator -> optimizer -> synthesizer) via ``PipelineRunRequest.concept``, with
re-mining disabled (``max_concept_attempts=1``) so that exactly this concept is judged.
A critic call per batch (``get_model_name("critic")``) then sees each candidate's concept,
formulation, audited parameters, metric, baseline, simulated gain and own estimate, together
with the map's objective and conventional baseline, and returns three numbers (the plausible
benefit on the candidate's own simulated quantity, the plausible contribution to the objective,
and the contribution a conventional measure with the same physical effect would make), flags
(relabelled analogue, non-conventional baseline, irrelevant quantity, proxy by construction),
assumption issues, killer risks and a 0..1 rating per requirement of the request (must or nice).
The score combines them with an objective scale set per archive (see "scoring" below).

Archive version 6: the analyst (``analyst_prompt`` / ``mock_analysis``, schema ProblemAnalysis) frames
the map and seeds it with its best 3 concepts; the critic additionally answers a hard-check checklist
(``HARD_CHECK_NAMES``), rates novelty, says whether the simulation contradicts the claim and whether a
scope extension is legitimate; the default scoring mode "filter" uses the simulation only as a gate
(``ScoringConfig``, ``apply_scoring``; "legacy" keeps the version-5 formula).
"""
import contextlib
import io
import math
import random
import re
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from vectornaut.config import MinerConceptOutput, ParameterProposal, get_client, get_model_name, get_thinking_config
from vectornaut.explorer.archive import EVALUATED, FAILED, INFEASIBLE, MUST, TIE_EPSILON, normalize_priority
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
    DirectConcept,
    HardCheck,
    LoadComponent,
    Lever,
    ProblemAnalysis,
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
        "wall_shear", "flow_rate", "heat_flux", "temperature", "deflection", "stress", "fouling_adhesion",
        "degradation_rate", "field_strength", "other",
    ), description=("The quantity the evaluator must compute to judge the concept's own benefit. It need not be "
                    "the request's objective itself (a fouling-release coating is simulated on fouling_adhesion, "
                    "and the critic judges how that translates into the objective)."), value_help={
        "wall_shear": "friction at a wetted wall (drag of a clean surface)",
        "stress": "mechanical stress in the material (strength, delamination, load spreading)",
        "fouling_adhesion": "adhesion or release stress of fouling organisms on the surface (lower = easier release)",
        "degradation_rate": "loss of function per time: wear, erosion, hydrolysis, delamination, reservoir depletion",
    }),
])

# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
#
#   score = 100 * gate * (W_OBJ * objective_score + W_SIM * simulated_score + W_REQ * requirement_score)
#           * relabel_factor * baseline_factor * must_factor
#   gate  = validator_score * (1.0 pass | 0.8 warn | 0 fail or missing)
#   objective_score = (1 - exp(-objective_gain_pct / S)) * d                       (0 for gains <= 0)
#       objective_gain_pct: the critic's plausible contribution to the stated OBJECTIVE against the
#       stated BASELINE minus the gain of a conventional measure with the same physical effect
#       (critic's conventional_equivalent_gain_pct, e.g. equal-R insulation), floored at 0
#       (without a critic: the candidate's own estimate);
#       d = 1 if a critic judged it next to a relevant simulation, else ESTIMATE_DISCOUNT
#       S = the archive's objective scale (see objective_scale_for): TARGET_SCALE_FACTOR * the
#       stated target gain, else ADAPTIVE_SCALE_FACTOR * the median positive critic objective gain
#       of the archive (bounded), else OBJECTIVE_SCALE_PCT
#   simulated_score = 1 - exp(-simulated_benefit_pct / SIM_SCALE_PCT)             (0 for gains <= 0)
#       only if the simulated gain is usable AND the simulated quantity is relevant to the
#       request; simulated_benefit_pct = min(simulated gain, critic's plausible simulated benefit)
#       whenever the critic gave one; 0 if the critic calls the gain a proxy by construction
#   requirement_score = (1 - MUST_MIN_WEIGHT) * mean(all ratings) + MUST_MIN_WEIGHT * min(must ratings)
#   must_factor = 1 - MUST_GATE_MAX_PENALTY * max(0, (MUST_GATE_THRESHOLD - min must) / MUST_GATE_THRESHOLD)
#   baseline_factor = BASELINE_FACTOR if the critic says the baseline is not the conventional one
#   relabel_factor = RELABEL_FACTOR if the critic calls the concept a relabelled analogue, else 1
#   estimated tier: score <= ESTIMATED_SCORE_CAP
#
# Evidence tiers: "simulated" only if the simulated number enters the score (usable and on a
# relevant quantity); otherwise "estimated". An estimate that replaced an implausible simulated
# gain ranks lowest (see archive.rank_key).
#
# Objective scale (archive version 5). A gain of S percent scores 1 - 1/e = 0.63. Version 4 used a
# fixed S = 2 %, chosen on the hull-coating role-play (critic objective gains 0.3-1.5 % of
# time-averaged drag). In the facade-cooling role-play the gains were 3-35 % of the heat load and
# all scored ~45 of 45 objective points, so the ranking fell to requirement-rating noise and an
# insulation concept in disguise ranked first. S is therefore set per archive and recomputed after
# every round (all entries are then re-scored, so scores stay comparable within a map):
#   - a stated target gain T (generator's target_gain_pct): S = T / 2, so reaching the target scores
#     0.86 (bounded to TARGET_SCALE_MIN_PCT..OBJECTIVE_SCALE_MAX_PCT);
#   - else, with at least ADAPTIVE_MIN_POINTS positive critic objective gains (net of the conventional
#     equivalent) in the archive: S = 2 * median, bounded to OBJECTIVE_SCALE_PCT..OBJECTIVE_SCALE_MAX_PCT,
#     so the archive's typical gain scores 0.39 (as 1 % did at S = 2 % for the hull), rounded to two
#     significant digits; the hull archive (median 0.8 %) stays at the floor of 2 %;
#   - else OBJECTIVE_SCALE_PCT.
# With S = 2 % a gain of 0.5 / 1.0 / 1.5 % scores 0.22 / 0.39 / 0.53, i.e. 10 / 18 / 24 points at
# W_OBJ = 0.45 (the version-4 hull reasoning: the objective separates the concepts, the simulated term
# at most 10 points and never above the critic, requirement coverage keeps the hard constraints).
#
# Requirements (version 5): the facade baseline ranked a cork cladding (bionic rating 0.15) first
# for a "bionisch" request, because a plain mean let the objective and the other ratings average the
# violated requirement away. Requirements are 'must' (default: explicitly stated in the request) or
# 'nice'; the weakest must-requirement enters the requirement score and, below MUST_GATE_THRESHOLD,
# scales the whole score down (0.15 -> x0.75, 0 -> x0.5).
#
# Archive version 6: the evaluator is a FILTER, not a ranking signal (scoring mode "filter", the
# default; "legacy" keeps the version-5 formula above). In blind comparisons on two tasks (hull
# coating, facade cooling) a careful direct answer beat the explorer's best picks; the explorer's
# picks were biased by the simple 1D/2D evaluator (simulated gains, validator scores) and by proxy
# metrics. Now:
#   score = 100 * gate * (FILTER_W_OBJ * objective_score + FILTER_W_REQ * requirement_score
#                         + FILTER_W_NOV * novelty + FILTER_W_SIM * simulated_score)
#           * relabel_factor * baseline_factor * must_factor * hard_check_factor
#   gate = 1 if the validator passed or warned (a warning is flagged), 0 if it failed or is missing
#       (the entry is excluded: score 0, flag validator_failed); the validator SCORE no longer scales
#   objective_score as above, d = 1 whenever the critic gave an objective gain (whatever the
#       simulation), else ESTIMATE_DISCOUNT (the candidate's own estimate, capped at ESTIMATED_SCORE_CAP)
#   novelty = the critic's novelty_rating (0..1 vs known products and literature; NOVELTY_UNRATED_DEFAULT
#       if unrated)
#   simulated_score: weight FILTER_W_SIM = 0 by default (configurable, --sim-weight; the weights are
#       then renormalised to sum 1)
#   hard_check_factor = HARD_CHECK_FACTOR if the critic's checklist has a failed hard check
#       (processing/thermal stability, manufacturability, field record, geometry, simple bounds)
# The critic sees the simulation only as a consistency check: simulation_contradicts_claim -> flag,
# the entry drops to evidence rank 1 (a non-contradicted critic-reviewed entry in the same cell wins).
# Evidence ranks (filter): 2 = critic-reviewed (labelled "critic-reviewed + simulation-consistent" or
# "critic-reviewed only"), 1 = contradicted by the simulation or not reviewed by a critic, 0 = not
# reviewed and the simulated gain was implausible. The hard-check factor also applies in legacy mode.
OBJECTIVE_SCALE_PCT = 2.0
OBJECTIVE_SCALE_MAX_PCT = 50.0
TARGET_SCALE_FACTOR = 0.5
TARGET_SCALE_MIN_PCT = 0.5
ADAPTIVE_SCALE_FACTOR = 2.0
ADAPTIVE_MIN_POINTS = 3
SIM_SCALE_PCT = 20.0
# Legacy (version 5) weights.
W_OBJ = 0.45
W_SIM = 0.10
W_REQ = 0.45
# Filter (version 6, default) weights.
FILTER_W_OBJ = 0.50
FILTER_W_REQ = 0.40
FILTER_W_NOV = 0.10
FILTER_W_SIM = 0.0
# A failed hard check of the critic's checklist (e.g. calcite in a glaze fired above its
# decomposition temperature; sub-ambient facade under full sun) multiplies the score by this.
HARD_CHECK_FACTOR = 0.3
HARD_CHECK_NAMES = ("processing_stability", "manufacturability", "field_record", "geometry_applicability",
                    "physical_bounds")
# Novelty used when the critic gave no rating (neutral, like an unrated requirement).
NOVELTY_UNRATED_DEFAULT = 0.5
# Filter mode: the validator is a gate (pass/warn -> 1, fail/missing -> 0), not a scaling factor.
FILTER_GATE = {"pass": 1.0, "warn": 1.0}
SCORING_FILTER = "filter"
SCORING_LEGACY = "legacy"
SCORING_MODES = (SCORING_FILTER, SCORING_LEGACY)
# Objective gains not judged next to a relevant simulation count half, and an estimated entry
# scores at most ESTIMATED_SCORE_CAP.
ESTIMATE_DISCOUNT = 0.5
ESTIMATED_SCORE_CAP = 50.0
# A concept the critic calls a relabelled analogue (same physics as its parent, new origin label).
RELABEL_FACTOR = 0.5
# The gain is not measured against the conventional solution (critic: baseline_conventional false).
BASELINE_FACTOR = 0.8
# Must-requirements: weight of the weakest one in the requirement score, and the soft gate below
# MUST_GATE_THRESHOLD (linear down to 1 - MUST_GATE_MAX_PENALTY at a rating of 0).
MUST_MIN_WEIGHT = 0.5
MUST_GATE_THRESHOLD = 0.3
MUST_GATE_MAX_PENALTY = 0.5
# The conventional equivalent covers at least this share of the plausible objective gain.
MOSTLY_CONVENTIONAL_SHARE = 0.5
# Requirement coverage used when no critic rated the candidate.
REQ_UNRATED_DEFAULT = 0.5
# A simulated gain beyond this (in percent, absolute) is treated as an artefact (the
# validator's physics_performance_gain_sanity check uses the same bound).
MAX_PLAUSIBLE_GAIN_PCT = 500.0
# Simulation and critic "differ strongly" when |a - b| > max(SENSITIVITY_ABS_PP, SENSITIVITY_REL * max(|a|, |b|)).
# This only sets the flag: the lower of the two numbers is always used when the critic gave one.
SENSITIVITY_REL = 0.5
SENSITIVITY_ABS_PP = 1.0
GAIN_SANITY_CHECK = "physics_performance_gain_sanity"
VALIDATOR_STATUS_FACTOR = {"pass": 1.0, "warn": 0.8}
UNAVAILABLE_GAIN_BASES = {"n_a", "na", "none", "unavailable", "not_available", "not_applicable", "not_computed"}

TIER_SIMULATED = "simulated"
TIER_ESTIMATED = "estimated"
EVIDENCE_RANK = {TIER_SIMULATED: 2, TIER_ESTIMATED: 1}
IMPLAUSIBLE_RANK = 0

# Sources of the objective scale.
SCALE_DEFAULT = "default"
SCALE_TARGET = "target"
SCALE_ARCHIVE = "archive"

# Flags stored in score_breakdown["flags"].
FLAG_IMPLAUSIBLE = "implausible_gain"
FLAG_SENSITIVE = "model_assumption_sensitive"
FLAG_GAIN_UNAVAILABLE = "gain_unavailable"
FLAG_QUANTITY_IRRELEVANT = "simulated_quantity_irrelevant"
FLAG_OBJECTIVE_UNCHECKED = "objective_gain_unchecked"
FLAG_OBJECTIVE_MISSING = "objective_gain_missing"
FLAG_RELABELLED = "relabelled_analogue"
FLAG_BASELINE_NOT_CONVENTIONAL = "baseline_not_conventional"
FLAG_PROXY = "proxy_by_construction"
FLAG_MOSTLY_CONVENTIONAL = "mostly_conventional_effect"
FLAG_MUST_UNMET = "must_requirement_unmet"
FLAG_REQ_UNRATED = "requirements_unrated"
FLAG_REQ_PARTIAL = "requirements_partially_rated"
FLAG_CRITIC_MISSING = "critic_missing"
FLAG_CRITIC_FAILED = "critic_failed"
FLAG_BASELINE_UNSTATED = "baseline_unstated"
# Archive version 6.
FLAG_HARD_CHECK = "hard_check_failed"
FLAG_SIM_CONTRADICTS = "simulation_contradicts_claim"
FLAG_NOVELTY_UNRATED = "novelty_unrated"
FLAG_SCOPE_EXTENSION = "scope_extension"
FLAG_SCOPE_NOT_LEGITIMATE = "scope_extension_not_legitimate"
FLAG_VALIDATOR_FAILED = "validator_failed"
FLAG_VALIDATOR_WARNING = "validator_warning"

# Evidence labels (score_breakdown["evidence_label"]).
LABEL_CONSISTENT = "critic-reviewed + simulation-consistent"
LABEL_REVIEWED = "critic-reviewed only"
LABEL_CONTRADICTED = "critic-reviewed, contradicted by the simulation"
LABEL_SIM_ONLY = "simulation only (no critic)"
LABEL_UNREVIEWED = "unreviewed estimate"


@dataclass(frozen=True)
class ScoringConfig:
    """
    How entries are scored. ``mode`` "filter" (archive version 6, default): the simulation is a
    feasibility/consistency gate, ranking = critic objective gain + requirements + novelty;
    ``sim_weight`` (default None = FILTER_W_SIM = 0) adds the simulated benefit back with that
    weight (all weights renormalised to sum 1). "legacy": the version-5 formula (W_OBJ/W_SIM/W_REQ,
    validator score as a factor, simulation-driven evidence tiers).
    """
    mode: str = SCORING_FILTER
    sim_weight: Optional[float] = None

    @classmethod
    def legacy(cls) -> "ScoringConfig":
        return cls(mode=SCORING_LEGACY)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ScoringConfig":
        data = data or {}
        mode = str(data.get("mode") or SCORING_FILTER).strip().lower()
        if mode not in SCORING_MODES:
            mode = SCORING_FILTER
        weight = finite(data.get("sim_weight"))
        return cls(mode=mode, sim_weight=weight if weight is not None and weight > 0 else None)

    @property
    def is_filter(self) -> bool:
        return self.mode == SCORING_FILTER

    def weights(self) -> Dict[str, float]:
        if not self.is_filter:
            return {"objective": W_OBJ, "simulated": W_SIM, "requirements": W_REQ}
        weights = {"objective": FILTER_W_OBJ, "simulated": FILTER_W_SIM, "requirements": FILTER_W_REQ,
                   "novelty": FILTER_W_NOV}
        if self.sim_weight:
            weights["simulated"] = float(self.sim_weight)
            total = sum(weights.values())
            weights = {k: round(v / total, 6) for k, v in weights.items()}
        return weights

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "sim_weight": self.sim_weight, "weights": self.weights()}


DEFAULT_SCORING = ScoringConfig()


def scoring_of(archive: Any) -> ScoringConfig:
    """The archive's stored scoring configuration (default: filter)."""
    stored = getattr(archive, "scoring", None) if archive is not None else None
    return ScoringConfig.from_dict(stored if isinstance(stored, Mapping) else None)

# Words in the request or its requirements that ask for a biological model (bionic, bio-inspired,
# biomimetic; German: bionisch, Bionik). Then only biological inspiration origins are preferred
# (fill_gap, diversify, combine, extrapolate); other origins only via explore at a low weight.
_BIO_REQUEST_PATTERN = re.compile(
    r"\bbioni(?:c|cs|k|sch\w*)\b|\bbio[-\s]?inspir\w*|\bbiomim\w*|\bbiologically[-\s]inspired\b"
    r"|\bderived from (?:a |an )?biological\b|\bbiological (?:model|role model|analogue|analog|system)s?\b",
    re.IGNORECASE,
)
BIOLOGICAL_ORIGINS = ("plant", "animal", "microbe")

PLACEHOLDER_SVG = (
    '<svg viewBox="0 0 400 200" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="400" height="200" rx="8" ry="8" fill="#0f172a"/>'
    '<text x="200" y="105" fill="#00f5d4" font-size="14" text-anchor="middle">explorer candidate</text></svg>'
)


def saturating(gain_pct: Optional[float], scale: float = SIM_SCALE_PCT) -> float:
    if gain_pct is None or gain_pct <= 0:
        return 0.0
    return 1.0 - math.exp(-gain_pct / scale)


def _two_digits(value: float) -> float:
    return float(f"{value:.2g}")


def objective_scale_for(target_gain_pct: Optional[float], gains: Sequence[float]) -> Tuple[float, str, str]:
    """
    (scale in percent, source, detail) of the objective term: from a stated target gain T
    (S = TARGET_SCALE_FACTOR * T), else from the positive critic objective gains of the archive
    (S = ADAPTIVE_SCALE_FACTOR * median, at least OBJECTIVE_SCALE_PCT), else OBJECTIVE_SCALE_PCT.
    Bounded above by OBJECTIVE_SCALE_MAX_PCT and rounded to two significant digits.
    """
    target = finite(target_gain_pct)
    if target is not None and target > 0:
        raw = TARGET_SCALE_FACTOR * target
        scale = _two_digits(min(max(raw, TARGET_SCALE_MIN_PCT), OBJECTIVE_SCALE_MAX_PCT))
        return scale, SCALE_TARGET, (f"stated target gain {target:g} % x {TARGET_SCALE_FACTOR:g} "
                                     f"(reaching the target scores {saturating(target, scale):.2f})")
    positive = sorted(g for g in (finite(v) for v in gains or ()) if g is not None and g > 0)
    if len(positive) >= ADAPTIVE_MIN_POINTS:
        n = len(positive)
        median = positive[n // 2] if n % 2 else 0.5 * (positive[n // 2 - 1] + positive[n // 2])
        raw = ADAPTIVE_SCALE_FACTOR * median
        scale = _two_digits(min(max(raw, OBJECTIVE_SCALE_PCT), OBJECTIVE_SCALE_MAX_PCT))
        bound = "" if scale == _two_digits(raw) else f", bounded to {OBJECTIVE_SCALE_PCT:g}..{OBJECTIVE_SCALE_MAX_PCT:g} %"
        return scale, SCALE_ARCHIVE, (f"{ADAPTIVE_SCALE_FACTOR:g} x median {median:.3g} % of {n} positive critic "
                                      f"objective gains (net of the conventional equivalent){bound}")
    return OBJECTIVE_SCALE_PCT, SCALE_DEFAULT, (f"default: fewer than {ADAPTIVE_MIN_POINTS} positive critic objective "
                                                "gains and no stated target")


def score_formula(scale: float = OBJECTIVE_SCALE_PCT, source: str = SCALE_DEFAULT,
                  scoring: Optional[ScoringConfig] = None) -> str:
    """The formula as applied, generated from the constants (so it matches the code)."""
    scoring = scoring or DEFAULT_SCORING
    if scoring.is_filter:
        w = scoring.weights()
        return (
            f"[filter scoring] score = 100 * gate * ({w['objective']:g} * objective_score + {w['requirements']:g} * "
            f"requirement_score + {w['novelty']:g} * novelty + {w['simulated']:g} * simulated_score) * relabel_factor "
            "* baseline_factor * must_factor * hard_check_factor; "
            "gate = 1 if the validator passed or warned, 0 if it failed (excluded); "
            f"objective_score = (1 - exp(-objective_gain_pct / {scale:g})) * d with the objective scale {scale:g} % "
            f"({source}), objective_gain_pct = critic's plausible objective gain minus its conventional-equivalent gain "
            f"(floored at 0), d = 1 if a critic gave the objective gain, else {ESTIMATE_DISCOUNT:g} (own estimate); "
            f"novelty = critic's novelty rating 0..1 (unrated {NOVELTY_UNRATED_DEFAULT:g}); "
            f"simulated_score = 1 - exp(-min(simulated, critic) / {SIM_SCALE_PCT:g}) (weight {w['simulated']:g}: the "
            "simulation is a consistency check, not a ranking signal); "
            f"requirement_score = {1 - MUST_MIN_WEIGHT:g} * mean + {MUST_MIN_WEIGHT:g} * min over must requirements; "
            f"must_factor = 1 - {MUST_GATE_MAX_PENALTY:g} * max(0, ({MUST_GATE_THRESHOLD:g} - min must) / "
            f"{MUST_GATE_THRESHOLD:g}); baseline_factor = {BASELINE_FACTOR:g} if the baseline is not the conventional one; "
            f"relabel_factor = {RELABEL_FACTOR:g} for a relabelled analogue; hard_check_factor = {HARD_CHECK_FACTOR:g} "
            f"if a hard check failed; entries without a critic review capped at {ESTIMATED_SCORE_CAP:g}"
        )
    return (
        f"score = 100 * gate * ({W_OBJ:g} * objective_score + {W_SIM:g} * simulated_score + "
        f"{W_REQ:g} * requirement_score) * relabel_factor * baseline_factor * must_factor; "
        "gate = validator_score * (1 pass / 0.8 warn / 0 fail); "
        f"objective_score = (1 - exp(-objective_gain_pct / {scale:g})) * d with the objective scale {scale:g} % "
        f"({source}), objective_gain_pct = critic's plausible objective gain minus its conventional-equivalent gain "
        f"(floored at 0), d = 1 if a critic judged the objective next to a relevant simulation, else "
        f"{ESTIMATE_DISCOUNT:g}; "
        f"simulated_score = 1 - exp(-simulated_benefit_pct / {SIM_SCALE_PCT:g}) if the simulated quantity is relevant, "
        "else 0, simulated_benefit_pct = min(simulated, critic) when the critic gave a number, 0 for a proxy by "
        f"construction; requirement_score = {1 - MUST_MIN_WEIGHT:g} * mean + {MUST_MIN_WEIGHT:g} * min over must "
        f"requirements; must_factor = 1 - {MUST_GATE_MAX_PENALTY:g} * max(0, ({MUST_GATE_THRESHOLD:g} - min must) / "
        f"{MUST_GATE_THRESHOLD:g}); baseline_factor = {BASELINE_FACTOR:g} if the baseline is not the conventional one; "
        f"relabel_factor = {RELABEL_FACTOR:g} for a relabelled analogue, else 1; "
        f"hard_check_factor = {HARD_CHECK_FACTOR:g} if a hard check failed; "
        f"estimated tier capped at {ESTIMATED_SCORE_CAP:g}"
    )


def simulated_benefit_used(simulated: float, critic_sim: Optional[float]) -> Tuple[float, str, bool]:
    """
    (benefit used, source, differ strongly?) for a usable, relevant simulated gain: the lower of
    the simulation and the critic's plausible simulated benefit whenever the critic gave one
    (a model result is never scored above the critic's real-world estimate); the simulation
    alone without a critic value. 'Differ strongly' only sets the model_assumption_sensitive flag.
    """
    if critic_sim is None:
        return simulated, "simulation", False
    sensitive = differs_strongly(simulated, critic_sim)
    note = "; simulation and critic differ strongly" if sensitive else ""
    if critic_sim < simulated:
        return critic_sim, f"critic (lower than the simulation{note})", sensitive
    return simulated, f"simulation (not above the critic{note})", sensitive


def net_objective_gain(plausible: Optional[float], conventional: Optional[float]) -> Tuple[Optional[float], bool]:
    """
    (objective gain beyond the conventional equivalent, mostly conventional?). The gain a
    conventional measure with the same physical effect would give (e.g. equal-R insulation) is
    subtracted from a positive plausible gain, floored at 0; 'mostly conventional' when it covers
    at least MOSTLY_CONVENTIONAL_SHARE of the plausible gain. Without a conventional value the
    plausible gain counts as it is.
    """
    if plausible is None or conventional is None or plausible <= 0:
        return plausible, False
    conventional = max(conventional, 0.0)
    return max(0.0, plausible - conventional), conventional >= MOSTLY_CONVENTIONAL_SHARE * plausible


def requirement_terms(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Mean coverage (unrated rows count REQ_UNRATED_DEFAULT), the weakest must-requirement, the
    requirement score (soft minimum) and the must factor (soft gate) for rows {name, coverage,
    priority}. Rows without a priority are must-requirements.
    """
    if not rows:
        return {"coverage": REQ_UNRATED_DEFAULT, "must_min": None, "score": REQ_UNRATED_DEFAULT, "must_factor": 1.0,
                "must_unmet": []}
    values = [(r.get("coverage") if finite(r.get("coverage")) is not None else REQ_UNRATED_DEFAULT, r) for r in rows]
    mean = sum(v for v, _ in values) / len(values)
    must = [(v, r) for v, r in values if normalize_priority(r.get("priority")) == MUST]
    if not must:
        return {"coverage": mean, "must_min": None, "score": mean, "must_factor": 1.0, "must_unmet": []}
    must_min = min(v for v, _ in must)
    score = (1.0 - MUST_MIN_WEIGHT) * mean + MUST_MIN_WEIGHT * must_min
    shortfall = max(0.0, (MUST_GATE_THRESHOLD - must_min) / MUST_GATE_THRESHOLD)
    unmet = [r.get("name") for v, r in must if v < MUST_GATE_THRESHOLD]
    return {"coverage": mean, "must_min": must_min, "score": score,
            "must_factor": 1.0 - MUST_GATE_MAX_PENALTY * shortfall, "must_unmet": unmet}


def combine_score(tier: str, gate: float, factor: float, simulated_benefit: Optional[float],
                  objective_gain: Optional[float], discount: float, requirement_score: float,
                  objective_scale: float = OBJECTIVE_SCALE_PCT, novelty: Optional[float] = None,
                  scoring: Optional[ScoringConfig] = None, cap: Optional[bool] = None) -> Dict[str, Any]:
    """
    The numeric part of the score (shared by scoring and re-scoring). ``factor`` is the product
    relabel_factor * baseline_factor * must_factor * hard_check_factor. ``cap`` (default: the
    estimated tier) applies ESTIMATED_SCORE_CAP.
    """
    scoring = scoring or DEFAULT_SCORING
    weights = scoring.weights()
    simulated_score = saturating(simulated_benefit, SIM_SCALE_PCT)
    objective_score = saturating(objective_gain, objective_scale) * discount
    novelty_score = NOVELTY_UNRATED_DEFAULT if novelty is None else max(0.0, min(1.0, float(novelty)))
    scale = 100.0 * gate * factor
    contributions = {"objective": scale * weights["objective"] * objective_score,
                     "simulated": scale * weights["simulated"] * simulated_score,
                     "requirements": scale * weights["requirements"] * requirement_score}
    components = {"gate": round(gate, 6), "objective_score": round(objective_score, 6),
                  "simulated_score": round(simulated_score, 6), "requirement_score": round(requirement_score, 6)}
    if "novelty" in weights:
        contributions["novelty"] = scale * weights["novelty"] * novelty_score
        components["novelty"] = round(novelty_score, 6)
    score = sum(contributions.values())
    capped = False
    apply_cap = (tier == TIER_ESTIMATED) if cap is None else cap
    if apply_cap and score > ESTIMATED_SCORE_CAP:
        score, capped = ESTIMATED_SCORE_CAP, True
    return {
        "score": round(score, 4), "capped": capped, "components": components,
        "contributions": {k: round(v, 4) for k, v in contributions.items()},
    }


def tiebreak_values(critic_obj: Optional[float], killer_risks: Sequence[Any],
                    simulated_benefit: Optional[float], critic_present: bool,
                    scoring: Optional[ScoringConfig] = None, consistent: bool = False,
                    novelty: Optional[float] = None) -> Optional[List[Optional[float]]]:
    """
    Tie-break key stored as score_breakdown['tiebreak'] (archive.tiebreak_key): higher critic
    objective gain (net of the conventional equivalent), then fewer killer risks, then (legacy) the
    higher simulated benefit used, or (filter) simulation-consistent before not, then higher
    novelty. None without a critic review (no tie-break then).
    """
    if not critic_present:
        return None
    if scoring is not None and scoring.is_filter:
        return [critic_obj, -float(len(killer_risks or [])), 1.0 if consistent else 0.0, novelty]
    return [critic_obj, -float(len(killer_risks or [])), simulated_benefit]

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


def differs_strongly(a: float, b: float, rel: float = SENSITIVITY_REL, abs_pp: float = SENSITIVITY_ABS_PP) -> bool:
    """
    True if two gains (in percent) disagree: |a - b| > max(abs_pp, rel * max(|a|, |b|)), or their
    signs differ while |a - b| > abs_pp. Noise below ``abs_pp`` percentage points never counts
    (0.02 vs 0 is agreement); -28.7 vs -5 is a disagreement although both are negative.
    """
    diff = abs(a - b)
    if diff <= abs_pp:
        return False
    if (a > 0) != (b > 0):
        return True
    return diff > rel * max(abs(a), abs(b))


def _estimate_pct(candidate: Any) -> tuple:
    boe = getattr(candidate, "back_of_envelope", None)
    if boe is None:
        return None, None
    unit = str(getattr(boe, "unit", "") or "").strip()
    is_pct = unit == "%" or normalize_token(unit) in ("pct", "percent", "percentage")
    return (finite(getattr(boe, "value", None)) if is_pct else None), getattr(boe, "unit", None)


def _descriptor(candidate: Any, axis: str) -> Optional[str]:
    for pair in getattr(candidate, "descriptors", None) or []:
        if normalize_token(getattr(pair, "axis", "")) == axis:
            return normalize_token(getattr(pair, "value", "")) or None
    return None


def _priorities(requirements: Optional[Sequence[Mapping[str, Any]]]) -> Dict[str, str]:
    return {normalize_token(r.get("name", "")): normalize_priority(r.get("priority")) for r in requirements or ()}


def requirement_coverage(requirements: Sequence[Mapping[str, Any]], review: Any) -> Dict[str, Any]:
    """
    Critic coverage (0..1) per stored requirement, with its priority ('must' or 'nice'), and the
    requirement terms (``requirement_terms``): mean, weakest must-requirement, requirement score
    and must factor. Requirements the critic did not rate count REQ_UNRATED_DEFAULT; without any
    rating the coverage is REQ_UNRATED_DEFAULT and the entry is flagged. Without a stored list,
    the critic's own ratings are used (all 'must').
    """
    ratings: Dict[str, Dict[str, Any]] = {}
    for rating in (getattr(review, "requirement_coverage", None) or []) if review is not None else []:
        value = finite(getattr(rating, "coverage", None))
        name = str(getattr(rating, "name", "") or "").strip()
        if not name or value is None:
            continue
        ratings.setdefault(normalize_token(name), {"name": name, "coverage": max(0.0, min(1.0, value)),
                                                   "reason": getattr(rating, "reason", "") or "", "priority": MUST})
    flags: List[str] = []
    rows: List[Dict[str, Any]] = []
    if requirements:
        for req in requirements:
            found = ratings.get(normalize_token(req.get("name", "")))
            rows.append({"name": req.get("name"), "coverage": found["coverage"] if found else None,
                         "reason": found["reason"] if found else "", "priority": normalize_priority(req.get("priority"))})
    else:
        rows = [dict(r) for r in ratings.values()]
    rated = [r["coverage"] for r in rows if r["coverage"] is not None]
    if not rated:
        flags.append(FLAG_REQ_UNRATED)
        terms = requirement_terms([])
    else:
        if len(rated) < len(rows):
            flags.append(FLAG_REQ_PARTIAL)
        terms = requirement_terms(rows)
        if terms["must_unmet"]:
            flags.append(FLAG_MUST_UNMET)
    return {"coverage": terms["coverage"], "rows": rows, "flags": flags, "terms": terms}


def quantity_relevance(quantity: Optional[str], relevant_quantities: Optional[Sequence[str]],
                       review: Any = None) -> Tuple[bool, str]:
    """
    Whether the simulated quantity may count towards the score. Irrelevant if the critic says so,
    or if the function analysis named relevant quantities and this one is not among them.
    """
    critic_says = getattr(review, "simulated_quantity_relevant", None) if review is not None else None
    if critic_says is False:
        return False, "the critic judged the simulated quantity irrelevant to the objective"
    stated = [normalize_token(v) for v in (relevant_quantities or [])]
    if stated and quantity and quantity not in stated:
        return False, f"governing_quantity={quantity} is not among the relevant quantities ({', '.join(stated)})"
    if critic_says is True:
        return True, "the critic judged the simulated quantity relevant"
    if stated and quantity:
        return True, "named by the function analysis"
    return True, "no relevance information (counted as relevant)"


def _objective_terms(critic_obj: Optional[float], conventional: Optional[float],
                     estimate: Optional[float]) -> Tuple[Optional[float], str, List[str]]:
    """(objective gain used, source, flags): the critic's gain net of the conventional equivalent, else the estimate."""
    if critic_obj is not None:
        net, mostly = net_objective_gain(critic_obj, conventional)
        if conventional is None or critic_obj <= 0:
            return net, "critic", []
        source = f"critic, net of the conventional equivalent ({critic_obj:g} - {max(conventional, 0.0):g} %)"
        return net, source, [FLAG_MOSTLY_CONVENTIONAL] if mostly else []
    if estimate is not None:
        return estimate, "candidate estimate (not checked by a critic)", [FLAG_OBJECTIVE_UNCHECKED]
    return None, "none", [FLAG_OBJECTIVE_MISSING]


def _hard_failures(checks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The failed hard checks ({name, reason}) of a critic checklist."""
    return [{"name": str(c.get("name") or ""), "reason": str(c.get("reason") or "")}
            for c in checks or () if isinstance(c, Mapping) and c.get("passed") is False]


def evidence_label(critic_present: bool, sim_ok: bool, proxy: bool, contradicts: bool) -> str:
    if critic_present:
        if contradicts:
            return LABEL_CONTRADICTED
        return LABEL_CONSISTENT if sim_ok and not proxy else LABEL_REVIEWED
    return LABEL_SIM_ONLY if sim_ok and not proxy else LABEL_UNREVIEWED


def apply_scoring(facts: Mapping[str, Any], scoring: Optional[ScoringConfig] = None,
                  objective_scale_pct: Optional[float] = None,
                  objective_scale_source: str = SCALE_DEFAULT) -> Tuple[float, Dict[str, Any]]:
    """
    The whole scoring rule, from the facts of one evaluation (shared by scoring and re-scoring, so
    both give the same result). Returns (score, derived breakdown fields). ``facts``:

    validator_status, validator_score, usable_gain (usable simulated gain or None), sim_relevant,
    implausible, gain_unavailable, critic_present, critic_sim, critic_obj, conventional, estimate,
    proxy, relabelled, baseline_conventional, contradicts (True/False/None), hard_checks (list of
    {name, passed, reason}), novelty (0..1 or None), scope_extension, scope_legitimate, killer_risks,
    requirement_rows ({name, coverage, reason, priority}), coverage_fallback (old breakdowns without
    rows), critic_status ('failed' / 'missing' / None), baseline_unstated.
    """
    scoring = scoring or DEFAULT_SCORING
    scale = finite(objective_scale_pct) or OBJECTIVE_SCALE_PCT
    filt = scoring.is_filter
    flags: List[str] = []
    if facts.get("gain_unavailable"):
        flags.append(FLAG_GAIN_UNAVAILABLE)
    if facts.get("implausible"):
        flags.append(FLAG_IMPLAUSIBLE)
    new_flags: List[str] = []

    status = str(facts.get("validator_status") or "")
    v_score = finite(facts.get("validator_score")) or 0.0
    if filt:
        gate = FILTER_GATE.get(status, 0.0)
        if gate == 0.0:
            new_flags.append(FLAG_VALIDATOR_FAILED)
        elif status == "warn":
            new_flags.append(FLAG_VALIDATOR_WARNING)
    else:
        gate = max(0.0, min(1.0, v_score)) * VALIDATOR_STATUS_FACTOR.get(status, 0.0)

    critic_present = bool(facts.get("critic_present"))
    usable = finite(facts.get("usable_gain"))
    relevant = bool(facts.get("sim_relevant"))
    proxy = bool(facts.get("proxy"))
    contradicts = facts.get("contradicts") is True
    critic_sim = finite(facts.get("critic_sim"))
    sim_ok = usable is not None and relevant

    # Simulated component: only a usable simulated gain on a relevant quantity, never above the
    # critic's plausible simulated benefit, not at all for a proxy (weight 0 in filter mode by default).
    sim_used, sim_source = None, "none"
    if sim_ok:
        sim_used, sim_source, sensitive = simulated_benefit_used(usable, critic_sim)
        if sensitive:
            flags.append(FLAG_SENSITIVE)
        basis = "simulated" if critic_sim is None else "simulated, critic-checked"
        if proxy:
            flags.append(FLAG_PROXY)
            sim_used, sim_source = None, "none (the critic calls the simulated gain a proxy by construction)"
            basis = "simulated, proxy by construction (simulated part not scored)"
    else:
        if usable is not None:
            flags.append(FLAG_QUANTITY_IRRELEVANT)
            basis = "estimated (simulated quantity not relevant)"
        else:
            basis = "estimated, not simulated"
        if proxy:
            flags.append(FLAG_PROXY)
    if contradicts:
        new_flags.append(FLAG_SIM_CONTRADICTS)

    if filt:
        tier = TIER_SIMULATED if sim_ok and not contradicts else TIER_ESTIMATED
        if critic_present:
            rank = EVIDENCE_RANK[TIER_SIMULATED] if not contradicts else EVIDENCE_RANK[TIER_ESTIMATED]
        else:
            rank = IMPLAUSIBLE_RANK if facts.get("implausible") else EVIDENCE_RANK[TIER_ESTIMATED]
    else:
        tier = TIER_SIMULATED if sim_ok else TIER_ESTIMATED
        rank = EVIDENCE_RANK[tier]
        if tier == TIER_ESTIMATED and facts.get("implausible"):
            rank = IMPLAUSIBLE_RANK

    # Objective component: the critic's contribution beyond the conventional equivalent (else the estimate).
    critic_obj = finite(facts.get("critic_obj"))
    conventional = finite(facts.get("conventional"))
    obj_used, obj_source, obj_flags = _objective_terms(critic_obj, conventional, finite(facts.get("estimate")))
    flags.extend(obj_flags)
    if filt:
        discount = 1.0 if critic_obj is not None else ESTIMATE_DISCOUNT
    else:
        discount = 1.0 if (tier == TIER_SIMULATED and critic_obj is not None) else ESTIMATE_DISCOUNT

    # Requirements.
    rows = [dict(r) for r in facts.get("requirement_rows") or []]
    rated = [r for r in rows if finite(r.get("coverage")) is not None]
    if rated:
        if len(rated) < len(rows):
            flags.append(FLAG_REQ_PARTIAL)
        terms = requirement_terms(rows)
        if terms["must_unmet"]:
            flags.append(FLAG_MUST_UNMET)
    else:
        fallback = finite(facts.get("coverage_fallback"))
        if fallback is not None and not rows:
            terms = {"coverage": fallback, "must_min": None, "score": fallback, "must_factor": 1.0, "must_unmet": []}
        else:
            flags.append(FLAG_REQ_UNRATED)
            terms = requirement_terms([])

    if facts.get("critic_status") == "failed":
        flags.append(FLAG_CRITIC_FAILED)
    elif facts.get("critic_status") == "missing":
        flags.append(FLAG_CRITIC_MISSING)
    if facts.get("baseline_unstated"):
        flags.append(FLAG_BASELINE_UNSTATED)
    relabelled = bool(facts.get("relabelled"))
    relabel_factor = RELABEL_FACTOR if relabelled else 1.0
    if relabelled:
        flags.append(FLAG_RELABELLED)
    baseline_conventional = facts.get("baseline_conventional") is not False
    if not baseline_conventional:
        flags.append(FLAG_BASELINE_NOT_CONVENTIONAL)
    baseline_factor = BASELINE_FACTOR if not baseline_conventional else 1.0
    must_factor = float(terms.get("must_factor") if terms.get("must_factor") is not None else 1.0)

    # Hard checks of the critic's checklist (both modes; absent in older reviews).
    failures = _hard_failures(facts.get("hard_checks") or [])
    hard_factor = HARD_CHECK_FACTOR if failures else 1.0
    if failures:
        new_flags.append(FLAG_HARD_CHECK)
    novelty = finite(facts.get("novelty"))
    novelty = None if novelty is None else max(0.0, min(1.0, novelty))
    if filt and novelty is None and critic_present:
        new_flags.append(FLAG_NOVELTY_UNRATED)
    if facts.get("scope_extension"):
        new_flags.append(FLAG_SCOPE_EXTENSION)
        if facts.get("scope_legitimate") is False:
            new_flags.append(FLAG_SCOPE_NOT_LEGITIMATE)

    cap = (not critic_present) if filt else None
    combined = combine_score(tier, gate, relabel_factor * baseline_factor * must_factor * hard_factor, sim_used,
                             obj_used, discount, float(terms["score"]), scale, novelty=novelty, scoring=scoring,
                             cap=cap)
    label = evidence_label(critic_present, sim_ok, proxy, contradicts)
    critic_net = obj_used if critic_obj is not None else None
    derived = {
        "basis": basis,
        "evidence_tier": tier,
        "evidence_rank": rank,
        "evidence_label": label,
        "flags": flags + new_flags,
        "requirements": rows,
        "scoring_mode": scoring.mode,
        "formula": score_formula(scale, objective_scale_source, scoring),
        "weights": scoring.weights(),
        "scales_pct": {"objective": scale, "simulated": SIM_SCALE_PCT},
        "objective_scale_source": objective_scale_source,
        "components": combined["components"],
        "objective_discount": discount,
        "relabel_factor": relabel_factor,
        "baseline_factor": baseline_factor,
        "must_factor": round(must_factor, 6),
        "hard_check_factor": hard_factor,
        "hard_check_failures": failures,
        "novelty_rating": novelty,
        "contributions": combined["contributions"],
        "capped": combined["capped"],
        "requirement_coverage": round(float(terms["coverage"]), 4),
        "requirement_score": round(float(terms["score"]), 4),
        "must_min_coverage": None if terms.get("must_min") is None else round(float(terms["must_min"]), 4),
        "objective_gain_pct": obj_used,
        "objective_gain_source": obj_source,
        "simulated_benefit_used_pct": sim_used,
        "simulated_benefit_source": sim_source,
        "conventional_equivalent_gain_pct": conventional,
        "tiebreak": tiebreak_values(critic_net, facts.get("killer_risks") or [], sim_used, critic_present,
                                    scoring=scoring, consistent=label == LABEL_CONSISTENT, novelty=novelty),
    }
    return combined["score"], derived


def _review_hard_checks(review: Any) -> List[Dict[str, Any]]:
    out = []
    for check in getattr(review, "hard_checks", None) or [] if review is not None else []:
        name = str(getattr(check, "name", "") or "").strip()
        if not name:
            continue
        out.append({"name": name, "passed": getattr(check, "passed", None), "reason": getattr(check, "reason", "") or ""})
    return out


def score_pipeline_result(result: Mapping[str, Any], candidate: Any, review: Any = None,
                          requirements: Sequence[Mapping[str, Any]] = (), critic_status: Optional[str] = None,
                          relevant_quantities: Optional[Sequence[str]] = None,
                          governing_quantity: Optional[str] = None,
                          objective_scale_pct: Optional[float] = None,
                          objective_scale_source: str = SCALE_DEFAULT,
                          scoring: Optional[ScoringConfig] = None) -> EvaluationResult:
    """
    Scores one pipeline result (0..100, formula in ``score_formula``). ``review`` is the
    materials critic's review (MaterialsCriticReview) or None; ``critic_status`` is
    'missing' / 'failed' when a critic was asked but gave nothing for this candidate.
    ``relevant_quantities`` are the governing quantities the function analysis named;
    ``governing_quantity`` defaults to the candidate's own descriptor. ``objective_scale_pct``
    is the archive's objective scale (default OBJECTIVE_SCALE_PCT); ``scoring`` the scoring mode
    (default: filter, archive version 6).
    """
    simulator = result.get("simulator") or {}
    validation = result.get("validation") or {}
    v_status = str(validation.get("status") or "")
    v_score = finite(validation.get("score")) or 0.0
    sim = assess_simulated_gain(result)
    estimate, estimate_unit = _estimate_pct(candidate)
    critic_sim = finite(getattr(review, "plausible_simulated_benefit_pct", None)) if review is not None else None
    critic_obj = finite(getattr(review, "plausible_objective_gain_pct", None)) if review is not None else None
    conventional = finite(getattr(review, "conventional_equivalent_gain_pct", None)) if review is not None else None
    quantity = governing_quantity or _descriptor(candidate, "governing_quantity")
    relevant, relevance_reason = quantity_relevance(quantity, relevant_quantities, review)
    proxy = bool(getattr(review, "proxy_by_construction", False)) if review is not None else False
    req = requirement_coverage(requirements, review)
    baseline = str(getattr(candidate, "baseline", "") or "").strip()
    relabelled = bool(getattr(review, "relabelled_analogue", False)) if review is not None else False
    hard_checks = _review_hard_checks(review)
    novelty = finite(getattr(review, "novelty_rating", None)) if review is not None else None
    contradicts = getattr(review, "simulation_contradicts_claim", None) if review is not None else None
    scope_extension = bool(getattr(candidate, "scope_extension", False))

    critic_block = None
    if review is not None:
        critic_block = {
            "plausible_simulated_benefit_pct": critic_sim,
            "plausible_objective_gain_pct": critic_obj,
            "conventional_equivalent_gain_pct": conventional,
            "plausible_gain_reasoning": getattr(review, "plausible_gain_reasoning", "") or "",
            "simulated_quantity_relevant": getattr(review, "simulated_quantity_relevant", None),
            "relabelled_analogue": relabelled,
            "relabel_reason": getattr(review, "relabel_reason", "") or "",
            "baseline_conventional": getattr(review, "baseline_conventional", None),
            "baseline_issue": getattr(review, "baseline_issue", "") or "",
            "proxy_by_construction": proxy,
            "proxy_reason": getattr(review, "proxy_reason", "") or "",
            "key_assumption_issues": list(getattr(review, "key_assumption_issues", None) or []),
            "killer_risks": list(getattr(review, "killer_risks", None) or []),
            "hard_checks": hard_checks,
            "simulation_contradicts_claim": contradicts,
            "simulation_consistency_note": getattr(review, "simulation_consistency_note", "") or "",
            "novelty_rating": novelty,
            "novelty_reason": getattr(review, "novelty_reason", "") or "",
            "scope_extension_legitimate": getattr(review, "scope_extension_legitimate", None),
            "scope_extension_note": getattr(review, "scope_extension_note", "") or "",
        }
    facts = {
        "validator_status": v_status, "validator_score": v_score,
        "usable_gain": sim["usable_gain"], "sim_relevant": relevant,
        "implausible": FLAG_IMPLAUSIBLE in sim["flags"], "gain_unavailable": FLAG_GAIN_UNAVAILABLE in sim["flags"],
        "critic_present": review is not None, "critic_sim": critic_sim, "critic_obj": critic_obj,
        "conventional": conventional, "estimate": estimate, "proxy": proxy, "relabelled": relabelled,
        "baseline_conventional": not (review is not None and getattr(review, "baseline_conventional", None) is False),
        "contradicts": contradicts, "hard_checks": hard_checks, "novelty": novelty,
        "scope_extension": scope_extension,
        "scope_legitimate": getattr(review, "scope_extension_legitimate", None) if review is not None else None,
        "killer_risks": (critic_block or {}).get("killer_risks") or [],
        "requirement_rows": req["rows"], "critic_status": critic_status, "baseline_unstated": not baseline,
    }
    score, derived = apply_scoring(facts, scoring, objective_scale_pct, objective_scale_source)
    breakdown: Dict[str, Any] = {
        "critic_objective_gain_pct": critic_obj,
        "estimated_gain_pct": estimate,
        "estimate_unit": estimate_unit,
        "estimate_baseline": baseline or None,
        "simulated_gain_pct": sim["raw_gain"],
        "performance_gain_pct": sim["usable_gain"],
        "critic_simulated_benefit_pct": critic_sim,
        "simulated_quantity": quantity,
        "simulated_quantity_relevant": relevant,
        "simulated_quantity_relevance": relevance_reason,
        "gain_basis": sim["basis"],
        "gain_sanity": sim["sanity"],
        "critic": critic_block,
        "simulation_contradicts_claim": contradicts,
        "scope_extension": scope_extension,
        "scope_extension_reason": str(getattr(candidate, "scope_extension_reason", "") or "") if scope_extension else "",
        "pipeline_status": result.get("status"),
        "validator_status": v_status or None,
        "validator_score": v_score,
        "solver_method": simulator.get("solver_method"),
        "relative_error": finite(simulator.get("relative_error")),
        "primary_metric_value": finite(simulator.get("primary_metric_value")),
        "is_mock": bool(result.get("is_mock")),
    }
    breakdown.update(derived)
    return EvaluationResult(status=EVALUATED, score=score, breakdown=breakdown)


def facts_from_breakdown(breakdown: Mapping[str, Any],
                         requirements: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """
    The facts of a stored breakdown (archive version 3 to 6) for ``apply_scoring``. Fields that did
    not exist yet count as absent: the proxy flag (version 3), the conventional-equivalent gain
    (before 5), hard checks, novelty, simulation contradiction and scope extension (before 6).
    Requirement priorities come from ``requirements`` (the archive's list; missing = must).
    """
    flags = list(breakdown.get("flags") or [])
    critic = dict(breakdown.get("critic") or {})
    components = breakdown.get("components") or {}
    status = breakdown.get("validator_status")
    v_score = finite(breakdown.get("validator_score"))
    if not status:
        # Very old breakdowns: only the gate is known (a pass at that score).
        gate = finite(components.get("gate")) or 0.0
        status, v_score = ("pass" if gate > 0 else "fail"), gate
    relevant = breakdown.get("simulated_quantity_relevant")
    if relevant is None:
        relevant = breakdown.get("evidence_tier") == TIER_SIMULATED
    conventional = finite(critic.get("conventional_equivalent_gain_pct"))
    if conventional is None:
        conventional = finite(breakdown.get("conventional_equivalent_gain_pct"))
    estimate = breakdown.get("estimated_gain_pct") if "estimated_gain_pct" in breakdown else None
    if "estimated_gain_pct" not in breakdown and finite(breakdown.get("critic_objective_gain_pct")) is None:
        estimate = breakdown.get("objective_gain_pct")
    priorities = _priorities(requirements)
    rows = []
    for row in breakdown.get("requirements") or []:
        row = dict(row)
        row["priority"] = priorities.get(normalize_token(row.get("name") or ""), normalize_priority(row.get("priority")))
        rows.append(row)
    status_flag = "failed" if FLAG_CRITIC_FAILED in flags else ("missing" if FLAG_CRITIC_MISSING in flags else None)
    hard_checks = critic.get("hard_checks") or [
        {"name": f.get("name"), "passed": False, "reason": f.get("reason")} for f in breakdown.get("hard_check_failures") or []]
    contradicts = critic.get("simulation_contradicts_claim")
    if contradicts is None:
        contradicts = breakdown.get("simulation_contradicts_claim")
    return {
        "validator_status": status, "validator_score": v_score,
        "usable_gain": finite(breakdown.get("performance_gain_pct")), "sim_relevant": bool(relevant),
        "implausible": FLAG_IMPLAUSIBLE in flags, "gain_unavailable": FLAG_GAIN_UNAVAILABLE in flags,
        "critic_present": bool(breakdown.get("critic")),
        "critic_sim": finite(breakdown.get("critic_simulated_benefit_pct")),
        "critic_obj": finite(breakdown.get("critic_objective_gain_pct")),
        "conventional": conventional, "estimate": finite(estimate),
        "proxy": bool(critic.get("proxy_by_construction")),
        "relabelled": bool(critic.get("relabelled_analogue")) or FLAG_RELABELLED in flags
        or (finite(breakdown.get("relabel_factor")) or 1.0) < 1.0,
        "baseline_conventional": not (FLAG_BASELINE_NOT_CONVENTIONAL in flags or critic.get("baseline_conventional") is False),
        "contradicts": contradicts, "hard_checks": hard_checks,
        "novelty": critic.get("novelty_rating") if critic.get("novelty_rating") is not None else breakdown.get("novelty_rating"),
        "scope_extension": bool(breakdown.get("scope_extension")),
        "scope_legitimate": critic.get("scope_extension_legitimate"),
        "killer_risks": critic.get("killer_risks") or [],
        "requirement_rows": rows, "coverage_fallback": breakdown.get("requirement_coverage"),
        "critic_status": status_flag, "baseline_unstated": FLAG_BASELINE_UNSTATED in flags,
    }


def rescore_breakdown(breakdown: Mapping[str, Any], requirements: Optional[Sequence[Mapping[str, Any]]] = None,
                      objective_scale_pct: Optional[float] = None,
                      objective_scale_source: str = SCALE_DEFAULT,
                      scoring: Optional[ScoringConfig] = None) -> Optional[Tuple[float, Dict[str, Any]]]:
    """
    Re-scores a stored breakdown (archive version 3 to 6) with the current rule (``apply_scoring``),
    the given objective scale and scoring mode (default: filter). Everything the rule needs is in the
    breakdown (``facts_from_breakdown``). Returns (score, new breakdown) or None if the breakdown has
    no tier (not a materials score).
    """
    tier = breakdown.get("evidence_tier")
    components = breakdown.get("components") or {}
    if tier not in EVIDENCE_RANK or "gate" not in components:
        return None
    score, derived = apply_scoring(facts_from_breakdown(breakdown, requirements), scoring, objective_scale_pct,
                                   objective_scale_source)
    new = dict(breakdown)
    new.update(derived)
    return score, new


def archive_objective_gains(archive: Any) -> List[float]:
    """The critic's objective gains (net of the conventional equivalent) of every evaluated materials entry."""
    gains = []
    for entry_id in sorted(archive.entries):
        entry = archive.entries[entry_id]
        breakdown = entry.get("score_breakdown") or {}
        if entry.get("status") != EVALUATED or breakdown.get("evidence_tier") not in EVIDENCE_RANK:
            continue
        critic_obj = finite(breakdown.get("critic_objective_gain_pct"))
        if critic_obj is None:
            continue
        conventional = finite((breakdown.get("critic") or {}).get("conventional_equivalent_gain_pct"))
        if conventional is None:
            conventional = finite(breakdown.get("conventional_equivalent_gain_pct"))
        net, _ = net_objective_gain(critic_obj, conventional)
        gains.append(net)
    return gains


def rescore_archive(archive: Any, round_no: Optional[int] = None, force: bool = False,
                    from_version: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Recomputes the archive's objective scale (``objective_scale_for``: stated target, else the
    archive's critic objective gains, else the default) and stores it with its source. If it
    changed (or ``force``), re-scores every evaluated materials entry from its breakdown and
    rebuilds the elites. Returns a summary, or None if nothing changed.
    """
    previous = archive.objective_scale
    scoring = scoring_of(archive)
    scale, source, detail = objective_scale_for(archive.target_gain_pct, archive_objective_gains(archive))
    changed = archive.set_objective_scale(scale, source, detail, round_no)
    if not changed and not force:
        return None
    rescored = 0
    for entry_id in sorted(archive.entries):
        entry = archive.entries[entry_id]
        if entry.get("status") != EVALUATED or entry.get("score") is None:
            continue
        result = rescore_breakdown(entry.get("score_breakdown") or {}, archive.requirements, scale, source,
                                   scoring=scoring)
        if result is None:
            continue
        score, breakdown = result
        if from_version is not None:
            breakdown["rescored_from"] = {"version": from_version, "score": entry.get("score"),
                                          "formula": (entry.get("score_breakdown") or {}).get("formula")}
        entry["score"], entry["score_breakdown"] = score, breakdown
        rescored += 1
    return {"objective_scale_pct": scale, "objective_scale_source": source, "objective_scale_detail": detail,
            "previous_objective_scale_pct": previous.get("pct"), "rescored_entries": rescored,
            "scoring_mode": scoring.mode,
            "elite_changes": archive.recompute_elites()}

# Words that show the request asks for a service life, and words that show a baseline in service.
_SERVICE_LIFE_WORDS = ("year", "jahr", "durab", "lasting", "lifetime", "service life", "lebensdauer", "haltbar",
                       "hält", "halt ", "fouling", "bewuchs", "long-term", "langzeit", "months", "monat")
_IN_SERVICE_WORDS = ("after", "month", "year", "in service", "in-service", "aged", "fouled", "time-averaged",
                     "time averaged", "over the", "nach ", "monat", "jahr", "betrieb", "gealtert")


def framing_warnings(query: str, objective: str, baseline: str) -> List[str]:
    """
    Deterministic plausibility check of the stored framing: a request that asks for years of
    service (or mentions fouling) needs an objective and a baseline in service condition, not a
    freshly applied clean surface. Returns warnings (empty if fine or if nothing is stored yet).
    """
    text = (query or "").lower()
    if not any(word in text for word in _SERVICE_LIFE_WORDS):
        return []
    warnings = []
    if objective and not any(word in objective.lower() for word in _IN_SERVICE_WORDS):
        warnings.append("the request asks for years of service, but the objective statement names no time horizon "
                        "(state it as a time average over the service interval)")
    if baseline and not any(word in baseline.lower() for word in _IN_SERVICE_WORDS):
        warnings.append("the request asks for years of service, but the baseline statement looks like a freshly "
                        "applied surface (state its condition in service, e.g. 'after 12 months')")
    return warnings


def pipeline_mechanism_text(candidate: MaterialsCandidate, objective_statement: str = "",
                            baseline_statement: str = "") -> str:
    """
    ``physical_mechanism`` handed to the pipeline: the mechanism plus, compactly and labelled,
    the summary, the comparison baseline, the map's objective and conventional baseline, the
    candidate's estimate (value, formula), the main risk and the novelty statement, so the
    formulator and auditor see the candidate's baseline and assumptions (MinerConceptOutput has
    no fields for them) and simulate against the conventional solution, not a parent concept.
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
    if clean(objective_statement):
        parts.append(f"Objective of the request: {clean(objective_statement)}")
    if clean(baseline_statement):
        parts.append(f"Conventional baseline for the simulation (compare against this, never against a parent or "
                     f"sibling concept): {clean(baseline_statement)}")
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


def to_miner_concept(candidate: MaterialsCandidate, objective_statement: str = "",
                     baseline_statement: str = "") -> MinerConceptOutput:
    mechanism = pipeline_mechanism_text(candidate, objective_statement, baseline_statement)
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


# Order-context keys that hold the parent concepts of a search order (see strategies.py).
_PARENT_CONTEXT_KEYS = ("elite", "parent_a", "parent_b", "neighbours", "edge_elites")


def order_parents(order: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The parent concepts named in an order's context (title, cell, summary), in context order."""
    context = order.get("context") or {}
    found: List[Dict[str, Any]] = []
    for key in _PARENT_CONTEXT_KEYS:
        value = context.get(key)
        for item in (value if isinstance(value, list) else [value]):
            if isinstance(item, Mapping) and item.get("title"):
                found.append(item)
    return found


def critic_candidate_block(space: DescriptorSpace, item: PreparedCandidate, result: Mapping[str, Any],
                           relevant_quantities: Sequence[str] = ()) -> str:
    """What the critic sees about one candidate: concept, order and parents, formulation, parameters, metric, gains."""
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
    quantity = item.descriptors.get("governing_quantity")
    if relevant_quantities:
        named = "named" if quantity in relevant_quantities else "NOT named"
        quantity_text = f"{quantity} ({named} by the function analysis: {', '.join(relevant_quantities)})"
    else:
        quantity_text = str(quantity)
    strategy = item.order.get("strategy") or "?"
    parents = order_parents(item.order)
    lines = [
        f"[{item.order['order_id']}] {c.title}",
        f"  cell: {space.describe(item.descriptors)}",
        f"  search order: {strategy}" + (f"; parent(s): " + " | ".join(
            f"'{_short(p.get('title'), 100)}' ({_short(p.get('cell'), 160)}): {_short(p.get('summary'), 200)}"
            for p in parents[:3]) if parents else ""),
        f"  inspiration: {_short(c.inspiration_source, 160) or '(not stated)'}",
        f"  concept: {_short(c.physical_mechanism or c.summary, 600)}",
        f"  candidate's baseline: {_short(getattr(c, 'baseline', '') or '(not stated)', 200)}",
        f"  candidate's estimate: {boe.value:g} {boe.unit} ({_short(boe.quantity, 120)}; {_short(boe.formula, 240)})"
        if boe is not None else "  candidate's estimate: (none)",
        f"  formulation: {_short(model.get('governing_equation'), 200)}; BCs: "
        f"{_short('; '.join(str(b) for b in (model.get('boundary_conditions') or [])), 300)}",
        f"  audited parameters: {_short(params, 500) or '(none)'}",
        f"  simulated quantity: {quantity_text}",
        f"  metric: {_short(metric_text, 240) or '(default)'}; simulated baseline: "
        f"{_short(auditor.get('baseline_description') or simulator.get('gain_basis') or '(none)', 160)}"
        + (f" with {_short(baseline_params, 200)}" if baseline_params else ""),
        f"  simulated gain: {gain_text}",
        f"  stated main risk: {_short(c.main_risk, 240)}",
    ]
    if strategy == "seed_direct":
        lines.insert(3, "  origin: the analyst's careful direct answer (a seed of the map); judge it exactly like the others")
    if getattr(c, "self_critique", ""):
        lines.append(f"  candidate's self-critique: {_short(c.self_critique, 300)}")
    if getattr(c, "scope_extension", False):
        lines.append(f"  SCOPE EXTENSION (acts outside the request's literal wording): "
                     f"{_short(getattr(c, 'scope_extension_reason', '') or '(no reason given)', 300)}")
    return "\n".join(lines)


HARD_CHECK_TEXT = """HARD CHECKS (answer every check for every candidate in hard_checks, name exactly as listed; passed=false
only for a real, specific failure you can state with numbers; null if you cannot judge). A failed hard check
multiplies the candidate's score by {factor:g}, so do not fail a check on a vague doubt, and do not pass one
because the concept is attractive:
- processing_stability: are the named materials stable at the named process and service conditions?
  e.g. calcite/aragonite decompose or transform far below glaze or sintering temperatures (CaCO3 releases
  CO2 from ~600-850 C; glazes fire at ~1000-1250 C), polymers above their decomposition temperature,
  hydrated phases dehydrate, aerogels collapse under firing or wet-dry cycling.
- manufacturability: can it be made at the intended scale (hundreds of m2 of facade, a whole ship hull)
  with known processes, tolerances and a plausible cost?
- field_record: is the material class known to fail in this service condition within the required
  service life (chalking and dirt pick-up of white coatings, UV embrittlement of polymers, hydrolysis,
  delamination of soft coatings, freeze-thaw cracking) without the concept addressing it?
- geometry_applicability: do the headline numbers apply to the stated geometry, orientation and condition
  (a vertical facade is not a roof; a fouled hull is not a clean lab plate; a coupon is not a building)?
- physical_bounds: does a claimed effect violate a simple bound? e.g. a sub-ambient surface temperature
  under peak sun on a vertical facade (it sees half the sky and hot surroundings; even 95 % solar
  reflectance absorbs ~30-40 W/m2, more than its net radiative loss to the sky), a reduction larger than
  the load component the concept acts on, a drag reduction beyond the friction share."""

CALIBRATION_TEXT = """CALIBRATION (typical real-world orders of magnitude; anchor your numbers on them):
- A cool/white roof coating on a low-slope roof in a hot climate lowers the roof heat gain by ~20-40 % but
  the whole-building cooling energy typically by ~5-15 %; on vertical walls the same coating gives a few
  percent, because walls get less peak irradiance and windows usually dominate the solar gain.
- External shading of windows (louvres or screens, shading coefficient ~0.2-0.3) cuts the solar gain
  through the glazing by ~60-80 %, i.e. ~20-40 % of the cooling load where glazing dominates.
- A fouling-release coating vs a biocidal antifouling changes time-averaged hull drag or fuel by roughly
  +-2-5 %; riblets or compliant coatings on a clean ship hull ~0-3 %; air lubrication of a flat-bottomed
  ship ~5-10 % net fuel.
Most new coatings deliver a fraction of their laboratory headline numbers in service. Be strict."""


def critic_prompt(space: DescriptorSpace, query: str, items: Sequence[PreparedCandidate],
                  results: Sequence[Mapping[str, Any]], requirements: Sequence[Mapping[str, Any]],
                  objective_statement: str = "", baseline_statement: str = "",
                  relevant_quantities: Sequence[str] = (), analysis_text: str = "") -> str:
    reqs = "\n".join(f"- {r['name']} [{normalize_priority(r.get('priority'))}]: {r.get('criterion') or ''}"
                     for r in requirements) or \
        "- (none extracted: rate the requirements you read from the request, with short snake_case names)"
    objective = objective_statement or ("(not stated: use the request's main benefit as it applies over the "
                                        "service life and conditions the request names)")
    baseline = baseline_statement or ("(not stated: the conventional state-of-the-art solution in the same "
                                      "service condition)")
    blocks = "\n".join(critic_candidate_block(space, item, result, relevant_quantities)
                       for item, result in zip(items, results))
    analysis = (f"\nPROBLEM ANALYSIS (by the analyst, fixed for this map; use it to judge magnitudes and scope):\n"
                f"{analysis_text}\n") if analysis_text else ""
    return f"""You are the Critic of Vectornaut's materials explorer. Every candidate below was turned into a
simplified steady model (1D/2D) and simulated. The simulation is only a feasibility and consistency check:
it cannot measure the real benefit (its gains depend on modelling choices: an equivalent gap or slip length
picked to match a target, a laminar model standing in for turbulent flow, a baseline that is not the real
alternative). Judge each concept at system level: how much it really contributes to the request's
objective, whether the simulation contradicts its claim, whether it passes the hard checks, how new it is,
and how well it meets the requirements.

REQUEST: "{query}"

OBJECTIVE (fixed for this map): {objective}
CONVENTIONAL BASELINE (fixed for this map): {baseline}
{analysis}
REQUIREMENTS ([must] = stated by the request: a concept that fails it does not answer the request and
loses much of its score; [nice] = desirable):
{reqs}

{HARD_CHECK_TEXT.format(factor=HARD_CHECK_FACTOR)}

{CALIBRATION_TEXT}

For every candidate:
- plausible_simulated_benefit_pct: your best estimate of the real-world improvement of the candidate's
  OWN simulated quantity (e.g. release stress, peak stress, wall shear, heat flux), in percent against
  the conventional baseline, positive = better. Use known values of comparable concepts; do not copy the
  simulated number. null if you cannot estimate it.
- plausible_objective_gain_pct: your best estimate of the candidate's contribution to the OBJECTIVE
  against the CONVENTIONAL BASELINE, in percent, positive = better, over the service life and conditions
  the objective names, for the stated geometry and orientation. A concept whose own quantity is not the
  objective can still improve it (e.g. an easier release of deposits that keeps the surface working); a
  concept that improves the fresh state but degrades faster may not. null if you cannot estimate it.
- conventional_equivalent_gain_pct: the objective gain, in percent against the same baseline, that a
  CONVENTIONAL measure achieving the same physical effect would give: e.g. an equal-R layer of standard
  insulation for a concept whose benefit is added thermal resistance (cork, aerogel, an air cavity), a
  thicker standard coating for a benefit that comes from thickness, a standard shading device of the
  same shading factor. 0 if the effect has no conventional equivalent (the mechanism itself is new);
  null if you cannot estimate it. Only the gain beyond it counts as the concept's own contribution.
- plausible_gain_reasoning: one or two sentences covering the numbers.
- simulation_contradicts_claim: true if the simulation result contradicts the claimed mechanism or
  benefit (no effect, or the opposite sign, for the physics the claim relies on); false if consistent;
  null if the simulation says nothing about the claim. Say what it shows in simulation_consistency_note.
  A modest simulated number is not a contradiction; a simulation that cannot represent the mechanism is
  null, not true.
- simulated_quantity_relevant: true if the simulated quantity drives the objective or a requirement,
  false if it is beside the point, null if unsure.
- hard_checks: the five checks above, one entry each.
- novelty_rating: 0..1 against known products and the literature (0 = a commercial product or textbook
  solution, 0.5 = a known idea in a new combination or application, 1 = no known precedent); name the
  closest known product or publication in novelty_reason.
- scope_extension_legitimate: only for candidates marked SCOPE EXTENSION: true if acting on that lever
  still answers the request (the problem analysis justifies it), false if it sidesteps the request; one
  line in scope_extension_note. Do not lower the other numbers because of the extension itself.
- relabelled_analogue: true if the concept is the same physics as its parent (or another listed concept)
  with only the inspiration label changed, i.e. the claimed origin does not actually supply the
  mechanism; name the copied concept in relabel_reason.
- baseline_conventional: false if the candidate's or the simulation's baseline is not the conventional
  baseline above (e.g. a parent concept, an untreated or fouled surface, an idealised case); say what it
  is in baseline_issue.
- proxy_by_construction: true if the simulated gain follows directly from an input or baseline choice,
  not from modelled physics: e.g. a leaching or loss flux compared against a baseline that contains the
  leaching species while the design contains none (the gain is ~100 % by definition), or a gain that
  equals an assumed input ratio (friction coefficients, settlement factors) with nothing solved in
  between. Name the choice in proxy_reason.
- key_assumption_issues: the modelling choices that drive the simulated gain, most important first.
- killer_risks: what could make the concept unworkable in practice, most severe first.
- requirement_coverage: one rating per requirement (name exactly as listed, coverage 0..1, a one-line
  reason). Count the whole system: energy and material consumption, maintenance, what is released
  into the environment, service life. Rate a [must] requirement below 0.3 only if the concept really
  fails it.

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
    {"name": "low_application_cost", "criterion": "applied with standard yard equipment", "priority": "nice"},
)
# Mock critic: requirement coverage per mechanism class (durability/consumption proxies).
_MOCK_COVERAGE = {
    "interfacial_slip": 0.6, "trapped_gas_or_liquid": 0.4, "flow_redirection": 0.8, "porous_transport": 0.5,
    "graded_stiffness": 0.7, "architected_lattice": 0.6, "radiative_control": 0.3, "phase_change": 0.3,
    "electrostatic_field": 0.4, "other": 0.5,
}
# Mock critic: plausible simulated benefit = simulated gain * factor (factor chosen by title hash).
_MOCK_CRITIC_FACTORS = (1.0, 0.8, 0.3, 1.2)
# Mock critic: share of the simulated benefit that reaches the objective (time-averaged drag).
_MOCK_OBJECTIVE_SHARE = {"wall_shear": 0.3, "flow_rate": 0.3, "fouling_adhesion": 0.5}
# Mock function analysis: mechanism classes that can plausibly lower wall friction.
MOCK_RELEVANT_MECHANISMS = ("interfacial_slip", "flow_redirection", "trapped_gas_or_liquid", "graded_stiffness")
# Mock critic: share of the objective gain a conventional measure with the same effect would give.
_MOCK_CONVENTIONAL_SHARE = {"trapped_gas_or_liquid": 0.6, "graded_stiffness": 0.2}
MOCK_OBJECTIVE = "(mock) time-averaged hull friction drag over a 5-year docking interval, including fouling"
MOCK_BASELINE = "(mock) conventional biocide-free silicone foul-release coating after 12 months in service"

_MOCK_MECHANISM = {
    "interfacial_slip": 1.0, "trapped_gas_or_liquid": 1.2, "flow_redirection": 0.7, "porous_transport": 0.5,
    "graded_stiffness": 0.4, "architected_lattice": 0.45, "radiative_control": 0.3, "phase_change": 0.35,
    "electrostatic_field": 0.3, "other": 0.25,
}
# Rises with feature size up to mm, then drops: a trend to extrapolate along, with an optimum.
_MOCK_SCALE = {"nm": 0.2, "sub_um": 0.35, "um": 0.55, "10_um": 0.8, "100_um": 1.1, "mm": 1.4, "cm_plus": 0.6}
_MOCK_ORIGIN = {"plant": 1.0, "animal": 0.9, "microbe": 1.15, "geology": 0.8, "atmosphere_ocean": 1.05,
                "technology": 0.85, "other": 0.7}
_MOCK_QUANTITY = {"wall_shear": 1.0, "flow_rate": 0.95, "fouling_adhesion": 0.9}
# Where a model's prior would put free proposals (textbook cluster).
_MOCK_PRIOR = {
    "mechanism_class": ("interfacial_slip", "flow_redirection", "trapped_gas_or_liquid"),
    "length_scale": ("um", "10_um", "100_um"),
    "inspiration_origin": ("animal", "plant"),
    "governing_quantity": ("wall_shear", "fouling_adhesion"),
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
    soft_relevance_axes = ("mechanism_class",)
    compatibility_axes = ("mechanism_class", "length_scale")
    origin_axes = ("inspiration_origin",)
    combine_anchor_axes = ("governing_quantity", "mechanism_class")
    trend_group_axes = ("mechanism_class",)
    textbook_solutions = (
        "Shark-skin riblets for drag reduction",
        "Lotus-leaf superhydrophobic surfaces",
        "Nepenthes (pitcher plant) slippery liquid-infused surfaces (SLIPS)",
        "Gecko-foot dry adhesion",
        "Namib desert beetle fog harvesting",
        "Honeycomb sandwich cores",
        "Moth-eye anti-reflection nanostructures",
    )

    analysis_schema = ProblemAnalysis

    def __init__(self, runner_factory: Optional[Callable[[], Any]] = None, critic_client: Any = None,
                 scoring: Any = None):
        # Tests inject a PipelineRunner with fake stages; default is the real pipeline.
        self.runner_factory = runner_factory
        self.critic_client = critic_client
        # Scoring for new and migrated archives: "filter" (default), "legacy", a dict or a ScoringConfig.
        if isinstance(scoring, ScoringConfig):
            self.scoring = scoring
        elif isinstance(scoring, Mapping):
            self.scoring = ScoringConfig.from_dict(scoring)
        else:
            self.scoring = ScoringConfig(mode=str(scoring or SCORING_FILTER))
            if self.scoring.mode not in SCORING_MODES:
                raise ValueError(f"Unknown scoring mode '{scoring}' (expected {', '.join(SCORING_MODES)}).")

    def default_scoring(self) -> Dict[str, Any]:
        return self.scoring.to_dict()

    def normalize_scoring(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        mode = str((config or {}).get("mode") or SCORING_FILTER).strip().lower()
        if mode not in SCORING_MODES:
            raise ValueError(f"Unknown scoring mode '{mode}' (expected {', '.join(SCORING_MODES)}).")
        return ScoringConfig.from_dict(config).to_dict()

    def domain_brief(self) -> str:
        return ("materials, surfaces and microstructures inspired by nature or by distant technologies, "
                "checked for feasibility by a physics simulation and judged by a critic at system level.")

    def evaluator_brief(self) -> str:
        return (
            "- The simulation is a feasibility and consistency check: a concept whose model fails validation is\n"
            "  excluded, and the critic flags a simulation that contradicts the claim. The ranking comes from the\n"
            "  critic's system-level judgement (objective gain, requirements, novelty), not from simulated gains.\n"
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
            "- baseline: name the baseline your estimate compares against: the map's BASELINE statement (the\n"
            "  conventional solution in the same service condition), never a parent, neighbour or sibling concept,\n"
            "  and never a fouled or untreated surface. The formulator and auditor are told to simulate against it.\n"
            "- governing_quantity: the quantity the simulation computes for THIS concept (e.g. fouling_adhesion\n"
            "  for a foul-release coating, wall_shear for riblets). It need not be the objective itself; the\n"
            "  critic judges how the simulated benefit translates into the objective.\n"
            "- back_of_envelope.value: the estimated improvement of the OBJECTIVE over that baseline in percent\n"
            "  (unit '%'). The score combines the critic's plausible objective gain beyond what a conventional\n"
            "  measure with the same physical effect would give (e.g. equal-R standard insulation for a concept\n"
            "  whose benefit is added thermal resistance), requirement coverage and the critic's novelty rating;\n"
            "  the simulation is a feasibility and consistency check (a failed validation excludes the concept),\n"
            "  and a failed hard check (materials unstable at the named process temperature, not manufacturable\n"
            "  at scale, a material class with a known field-failure record, numbers from another geometry, a\n"
            f"  violated simple bound) multiplies the score by {HARD_CHECK_FACTOR:g}.\n"
            "- A diversify or fill_gap order that changes inspiration_origin needs a mechanism actually taken\n"
            "  from a system of that origin: name the system in inspiration_source and what it does there. The\n"
            "  parent's physics with a new origin label is a relabelled analogue: the critic flags it and its\n"
            f"  score is multiplied by {RELABEL_FACTOR:g}. If no system of that origin offers a mechanism, say so\n"
            "  (target_feasible=false only if none can exist).\n"
            "- Every candidate is also rated against each requirement of the request (e.g. no toxic release,\n"
            "  service life, energy or material consumption): a concept that buys its benefit by violating a\n"
            "  requirement scores low, and one that fails a 'must' requirement loses up to half of its score."
        )

    def analysis_instructions(self, archive: Any) -> str:
        parts = [self._framing_instructions(archive)]
        stored = list((archive.request_analysis.get("relevant") or {}).get("governing_quantity") or [])
        if stored:
            parts.append("RELEVANT GOVERNING QUANTITIES (from the function analysis; explore and diversify orders only use\n"
                         f"these and the quantities of elites): {', '.join(stored)}. Repeat or extend them in\n"
                         "relevant_governing_quantities.\n")
        else:
            parts.append("In step 1 also list relevant_governing_quantities: the governing_quantity tokens that measure a\n"
                         "benefit the request actually asks for, directly or over the service life (a drag request:\n"
                         "wall_shear; if fouling or years of service matter: fouling_adhesion, degradation_rate; a\n"
                         "strength or delamination requirement: stress). Explore orders only target cells with these\n"
                         "quantities, and a simulated benefit on another quantity does not count towards the score.\n")
        mechanisms = list((archive.request_analysis.get("relevant") or {}).get("mechanism_class") or [])
        if mechanisms:
            parts.append("RELEVANT MECHANISM CLASSES (from the function analysis; fill_gap and diversify orders only use\n"
                         f"these and the mechanisms of the top elites, explore rarely goes elsewhere): {', '.join(mechanisms)}.\n"
                         "Repeat or extend them in relevant_mechanism_classes.\n")
        else:
            parts.append("In step 2 also list relevant_mechanism_classes: the mechanism_class tokens that can plausibly\n"
                         "deliver the request's main benefit (a heat-load request: e.g. radiative_control,\n"
                         "trapped_gas_or_liquid, architected_lattice, phase_change; not interfacial_slip). fill_gap and\n"
                         "diversify orders then stay within them; explore reaches the others at a low weight.\n")
        origins = archive.preferred_values("inspiration_origin") if hasattr(archive, "preferred_values") else None
        if origins:
            reason = (archive.request_analysis.get("preferred_reason") or {}).get("inspiration_origin") or ""
            parts.append(f"PREFERRED INSPIRATION ORIGINS: {', '.join(origins)} ({reason}). fill_gap, diversify,\n"
                         "combine and extrapolate orders only target these origins; the mechanism must really come\n"
                         "from such a biological system. Other origins appear only in explore orders.\n")
        return "\n".join(parts)

    def _framing_instructions(self, archive: Any) -> str:
        objective, baseline = archive.objective_statement, archive.baseline_statement
        target = archive.target_gain_pct if hasattr(archive, "target_gain_pct") else None
        target_text = (f"- target gain: {target:g} % (stored; repeat it in target_gain_pct)\n" if target is not None else
                       "- target_gain_pct: if the request, your objective statement or a requirement names a numeric\n"
                       "  improvement of the objective (e.g. '>20 %' -> 20), give that number; null if none is stated.\n"
                       "  It sets the objective scale of the score (reaching the target scores 0.86 of the objective part).\n")
        if objective and baseline:
            text = ("OBJECTIVE AND BASELINE (fixed for this map; the critic judges every candidate's contribution to\n"
                    "this objective against this baseline; repeat them in objective_statement and baseline_statement):\n"
                    f"- objective: {objective}\n- baseline: {baseline}\n" + target_text)
            warnings = self.framing_warnings(archive)
            if warnings:
                text += "  Note: " + "; ".join(warnings) + ".\n"
            return text
        known = ""
        if objective or baseline:
            known = (f"  Already stored: objective = {objective or '(none)'}; baseline = {baseline or '(none)'}. "
                     "Fill in the missing one.\n")
        return (
            "OBJECTIVE AND BASELINE (step 1, stored once for this map):\n"
            "- objective_statement: the request's main benefit stated as it applies over the stated service life\n"
            "  and operating conditions. If the request asks for years of service, durability or fouling control,\n"
            "  state it as a time average over the service interval including degradation and fouling (e.g.\n"
            "  'time-averaged hull friction drag over a 5-year docking interval, including the effect of\n"
            "  fouling'), not as the benefit of a freshly applied clean surface.\n"
            "- baseline_statement: the conventional state-of-the-art solution a concept must beat on that objective,\n"
            "  in the same condition (e.g. 'conventional biocide-free silicone foul-release coating after 12-24\n"
            "  months in service'). Every candidate and simulation compares against it.\n"
            + target_text + known
        )

    # ---- analysis-first stage (archive version 6) ------------------------
    def analyst_prompt(self, query: str, archive: Any) -> str:
        """The analyst's prompt: system-level analysis, framing, and the analyst's own best 3 concepts."""
        bio = ""
        preferred = self.preferred_values(query, ())
        if preferred:
            bio = ("\nThe request asks for a biological model: the requirement list must say so, and the mechanism of each\n"
                   "direct concept must really come from a biological system (name it in inspiration_source). A lever that\n"
                   "no biological system can supply may still be named in the analysis.\n")
        return f"""You are the Analyst of Vectornaut's materials explorer. Before any search starts, analyse the request at
system level, as a senior engineer would before proposing anything, and then give your own best direct
answer. Your analysis fixes the framing of the whole map (objective, baseline, requirements); your three
concepts become its seeds, are evaluated like every other candidate, and the search then looks for better
or more novel concepts around them. A search that starts from a wrong framing wastes all its effort, so
take the time to get the physics and the numbers right.

REQUEST: "{query}"
{bio}
THE MAP (closed vocabulary for the relevant values and the descriptors of your direct concepts):
{self.space.vocabulary_text()}

WHAT THE EVALUATOR CAN CHECK (it is a feasibility and consistency check of each concept, not a measure
of its real benefit):
{self.evaluator_brief()}

WORK IN THIS ORDER
1. system_analysis and load_breakdown: where does the load, loss or failure the request is about actually
   come from? Give a quantitative breakdown with rough numbers and shares (e.g. a building's summer heat
   load: solar gain through the glazing, conduction through opaque walls and roof, ventilation and
   infiltration, internal gains; W/m2 and %; a hull: friction vs form and wave resistance, the added
   friction from fouling over the docking interval). Name the service condition (ageing, soiling,
   fouling) and the typical in-service performance of the conventional solution.
2. levers: the main levers, ranked by expected magnitude (percent improvement of the objective against the
   conventional in-service baseline). Mark every lever that the request's literal wording excludes
   (within_literal_scope=false; e.g. the windows of a request that names a coating or facade structure)
   and say whether extending the scope to it is justified (extension_justified): it is where the load
   comes from and it still serves the intent of the request. Set scope_extension_justified and
   scope_extension_reason for the map.
3. Framing (fixed for the whole map): conventional_baseline (in service, with numbers), objective_statement
   (the main benefit over the stated service life and conditions; a time average including ageing, soiling
   or fouling when the request asks for years of service), baseline_statement (the conventional solution
   in the same in-service condition), target_gain_pct (the number the request asks for, else a realistic
   target from your lever ranking, else null).
4. requirements (3-6; priority 'must' for what the request states explicitly, e.g. 'without', 'at least',
   'bionic', 'ohne', 'mindestens', 'bionisch'; 'nice' for implied or desirable properties),
   relevant_mechanism_classes, relevant_governing_quantities and relevant_inspiration_origins (tokens
   from the vocabulary).
5. direct_concepts: your best 3 concepts as a careful direct answer. Spend the effort on depth, not
   breadth: target the largest levers (a justified scope extension is allowed: set scope_extension and
   scope_extension_reason). For each: what exactly (materials, dimensions, process, how it is fixed to
   the building/hull); key_physics with numbers; realistic_benefit_pct against the in-service baseline
   and benefit_reasoning from the load breakdown (share of the load it acts on x effect size, minus
   losses); risks, most severe first (processing and thermal stability of the named materials at the
   named process conditions, manufacturability at scale, known field failures of the material class,
   whether the numbers hold for the stated geometry and orientation); self_critique (the strongest
   objection and what it does to your number). Fill the candidate fields as well: title, summary,
   descriptors (one per axis, from the vocabulary), inspiration_source, domain, physical_mechanism,
   baseline (the baseline_statement), back_of_envelope (the realistic benefit of the objective in %, unit
   '%', with the formula), main_risk, novelty_vs_known, and 3-8 parameters (snake_case, SI values,
   min_bound < value < max_bound, a justification each) that are enough to write a steady 1D/2D model.
"""

    def mock_analysis(self, query: str) -> ProblemAnalysis:
        """Deterministic stand-in for the analyst (hull-drag landscape of the mock generator)."""
        cells = (
            ("graded_stiffness", "100_um", "animal", "fouling_adhesion", False,
             "Graded-modulus soft foul-release skin"),
            ("trapped_gas_or_liquid", "10_um", "plant", "wall_shear", False,
             "Plastron-retaining micro-pillar skin"),
            ("trapped_gas_or_liquid", "cm_plus", "technology", "wall_shear", True,
             "Bottom air-lubrication cavity with a gas-retaining liner"),
        )
        concepts = []
        for i, (mechanism, scale, origin, quantity, scope, title) in enumerate(cells, 1):
            target = {"mechanism_class": mechanism, "length_scale": scale, "inspiration_origin": origin,
                      "governing_quantity": quantity}
            base = self.mock_candidate({"order_id": f"mock-direct-{i}", "strategy": "seed_direct", "target": target})
            data = base.model_dump()
            data.update(order_id="", title=f"(mock direct) {title}", key_physics="(mock) friction share x effect",
                        realistic_benefit_pct=round(float(data["back_of_envelope"]["value"]), 3),
                        benefit_reasoning="(mock) 70 % friction share x effect size", risks=["(mock) wear"],
                        self_critique="(mock) lab numbers rarely hold in service",
                        scope_extension=scope,
                        scope_extension_reason="(mock) the hull bottom is where most friction arises; air "
                                               "lubrication is not a coating" if scope else "")
            concepts.append(DirectConcept.model_validate(data))
        return ProblemAnalysis(
            system_analysis="(mock) Friction is ~70 % of the resistance of a slow ship; fouling adds 10-40 % over "
                            "a 5-year interval, so fouling control is the dominant lever.",
            load_breakdown=[LoadComponent(name="skin friction (clean)", share_pct=60.0, rough_value="(mock)"),
                            LoadComponent(name="added friction from fouling", share_pct=25.0, rough_value="(mock)"),
                            LoadComponent(name="form and wave resistance", share_pct=15.0, rough_value="(mock)")],
            levers=[Lever(name="fouling control", acts_on="added friction from fouling", expected_magnitude_pct=10.0),
                    Lever(name="air lubrication of the flat bottom", acts_on="skin friction (clean)",
                          expected_magnitude_pct=7.0, within_literal_scope=False, extension_justified=True,
                          note="(mock) not a coating, but acts on the largest share"),
                    Lever(name="clean-hull friction reduction", acts_on="skin friction (clean)",
                          expected_magnitude_pct=2.0)],
            scope_extension_justified=True,
            scope_extension_reason="(mock) air lubrication acts on the largest load share although it is not a coating",
            conventional_baseline=MOCK_BASELINE,
            objective_statement=MOCK_OBJECTIVE,
            baseline_statement=MOCK_BASELINE,
            target_gain_pct=None,
            requirements=[Requirement(**r) for r in MOCK_REQUIREMENTS],
            relevant_mechanism_classes=list(MOCK_RELEVANT_MECHANISMS),
            relevant_governing_quantities=["wall_shear", "flow_rate", "fouling_adhesion"],
            relevant_inspiration_origins=["animal", "plant"],
            direct_concepts=concepts,
        )

    def depth_instructions(self, depth: str) -> str:
        """Extra generator rules for --depth deep (fewer, deeper candidates)."""
        if depth != "deep":
            return ""
        return (
            "- DEPTH MODE (few, deep candidates): reason about each candidate as carefully as a direct answer.\n"
            "  back_of_envelope: a quantitative estimate derived from the problem analysis' load breakdown (share of\n"
            "  the load the concept acts on x effect size, minus losses), with the numbers in the formula;\n"
            "  self_critique: the strongest objection against your own concept and number (processing and\n"
            "  thermal stability of the named materials, manufacturability, field record of the material class,\n"
            "  geometry/orientation, simple physical bounds) and what it does to the number; the value in\n"
            "  back_of_envelope is the estimate AFTER that critique. Prefer one convincing concept over a clever label."
        )

    def scope_instructions(self, archive: Any) -> str:
        """Generator rule for scope extensions (only when the analysis justified one)."""
        analysis = archive.problem_analysis if hasattr(archive, "problem_analysis") else None
        if not analysis:
            return ""
        levers = [lv for lv in analysis.get("levers") or [] if not lv.get("within_literal_scope", True)
                  and lv.get("extension_justified")]
        if not (analysis.get("scope_extension_justified") and levers):
            return ("- Scope: the problem analysis found no justified scope extension; stay within the request's "
                    "wording (scope_extension=false).")
        names = "; ".join(f"{lv.get('name')} ({lv.get('note') or 'justified'})" for lv in levers[:4])
        return ("- Scope extensions are allowed: the problem analysis justified levers outside the request's literal\n"
                f"  wording: {names}. A candidate that targets one sets scope_extension=true and names the lever and\n"
                "  the reason in scope_extension_reason. It is not penalised; the critic states whether the extension\n"
                "  is legitimate. Everything else stays within the wording.")

    def score_note(self, archive: Any = None) -> str:
        """Explanation of the score for the map report (generated from the constants and the archive's scale)."""
        scale = (archive.objective_scale if archive is not None and hasattr(archive, "objective_scale") else {}) or {}
        pct, source = scale.get("pct") or OBJECTIVE_SCALE_PCT, scale.get("source") or SCALE_DEFAULT
        detail = f" Objective scale {pct:g} % ({source}: {scale.get('detail')})." if scale.get("detail") else ""
        scoring = scoring_of(archive) if archive is not None else DEFAULT_SCORING
        if scoring.is_filter:
            return (f"{score_formula(pct, source, scoring)}.{detail} The simulation is a feasibility and consistency "
                    "gate, not a ranking signal: a failed validator excludes the entry (score 0), and the critic flags a "
                    "simulation that contradicts the claim (evidence rank 1). Evidence labels: 'critic-reviewed + "
                    "simulation-consistent' (tier 'simulated') or 'critic-reviewed only' (tier 'estimated'); both rank "
                    "equal. 'points obj / sim / req / nov' splits the score into its parts (before the cap). The scale "
                    "is recomputed after every round and all entries are then re-scored, so scores are comparable "
                    f"within this map. Within a cell, scores within {TIE_EPSILON:g} point are a tie decided by the "
                    "critic's objective gain, then fewer killer risks, then simulation consistency and novelty.")
        return (f"{score_formula(pct, source, scoring)}.{detail} Tier 'simulated' only when a usable simulated gain on a "
                "relevant governing quantity enters the score (never above the critic's plausible simulated benefit; not "
                "at all for a proxy by construction); otherwise tier 'estimated' (the simulated number is shown but not "
                "scored). The objective gain is the critic's plausible contribution to the stated objective against the "
                "stated baseline, minus what a conventional measure with the same physical effect would give. "
                "'points obj / sim / req' splits the score into its three parts (before the cap). The scale is recomputed "
                "after every round and all entries are then re-scored, so scores are comparable within this map. Elites "
                "are ranked by tier first; within a cell, scores within "
                f"{TIE_EPSILON:g} point are a tie decided by the critic's objective gain, then fewer killer risks.")

    def preferred_values(self, query: str, requirements: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple[List[str], str]]:
        """
        Biological inspiration origins only, if the request or one of its requirements asks for a
        bionic / bio-inspired / biomimetic concept (keywords, deterministic). Otherwise no preference.
        """
        sources = [("the request", query or "")]
        for req in requirements or ():
            text = f"{req.get('name') or ''} {req.get('criterion') or ''}".replace("_", " ")
            sources.append((f"requirement '{req.get('name')}'", text))
        for label, text in sources:
            match = _BIO_REQUEST_PATTERN.search(text)
            if match:
                return {"inspiration_origin": (list(BIOLOGICAL_ORIGINS),
                                               f"{label} asks for a biological model ('{match.group(0)}')")}
        return {}

    def migrate_archive(self, archive: Any, from_version: int) -> Optional[Dict[str, Any]]:
        """
        Version 3 or 4 -> 5: sets the archive's objective scale (no target was stored before version 5,
        so it comes from the archive's critic objective gains; the hull-coating archive, median 0.8 %,
        stays at the 2 % floor), re-scores every evaluated entry from its stored breakdown with the
        current formula (``rescore_breakdown``: must-requirements, baseline factor; the
        conventional-equivalent gain did not exist and counts as absent) and rebuilds the elites with
        the current comparison (tiebreak). The old score is kept in ``score_breakdown.rescored_from``;
        round logs stay as recorded.
        """
        if from_version >= 6:
            return None
        if not archive.scoring:
            # Version 6: the profile's scoring (default: filter, the evaluator as a gate) for the old entries.
            archive.set_scoring(self.default_scoring())
        return rescore_archive(archive, round_no=None, force=True, from_version=from_version)

    def update_scoring(self, archive: Any, round_no: Optional[int] = None, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Recomputes the objective scale from the archive (after a round, or when the runner starts) and
        re-scores every entry if it changed or ``force`` (e.g. a new scoring mode) (``rescore_archive``).
        Returns the summary or None.
        """
        return rescore_archive(archive, round_no=round_no, force=force)

    def framing_warnings(self, archive: Any) -> List[str]:
        queries = archive.data.get("queries") or []
        return framing_warnings(queries[0] if queries else "", archive.objective_statement, archive.baseline_statement)

    def relevant_values_from_batch(self, batch: Any) -> Tuple[Dict[str, List[str]], List[str]]:
        found: Dict[str, List[str]] = {}
        dropped: List[str] = []
        for axis_name, field in (("governing_quantity", "relevant_governing_quantities"),
                                 ("mechanism_class", "relevant_mechanism_classes")):
            axis = self.space.axis(axis_name)
            values = []
            for raw in getattr(batch, field, None) or []:
                token = normalize_token(raw)
                if token in axis.values:
                    if token not in values:
                        values.append(token)
                else:
                    dropped.append(str(raw))
            if values:
                found[axis_name] = values
        return found, dropped

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
            baseline=MOCK_BASELINE,
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
            objective_statement=MOCK_OBJECTIVE,
            baseline_statement=MOCK_BASELINE,
            relevant_governing_quantities=["wall_shear", "flow_rate", "fouling_adhesion"],
            mechanism_classes_considered=["interfacial_slip", "flow_redirection"],
            relevant_mechanism_classes=list(MOCK_RELEVANT_MECHANISMS),
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
            concept=to_miner_concept(item.candidate, ctx.objective_statement, ctx.baseline_statement),
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
        """
        Deterministic stand-in: scales the simulated gain (sometimes by 0.3 -> 'sensitive'); a share
        of it reaches the objective; about half of the fill_gap/diversify candidates that changed the
        origin of their parent are called relabelled analogues.
        """
        key = zlib.crc32(item.candidate.title.encode("utf-8"))
        factor = _MOCK_CRITIC_FACTORS[key % len(_MOCK_CRITIC_FACTORS)]
        usable = assess_simulated_gain(result)["usable_gain"]
        estimate, _ = _estimate_pct(item.candidate)
        base = usable if usable is not None else (estimate if estimate is not None else 0.0)
        quantity = item.descriptors.get("governing_quantity")
        coverage = _MOCK_COVERAGE.get(item.descriptors.get("mechanism_class"), 0.5)
        names = [r["name"] for r in requirements] or [r["name"] for r in MOCK_REQUIREMENTS]
        ratings = [RequirementRating(name=name, coverage=round(max(0.0, min(1.0, coverage + 0.1 * ((key + i) % 3 - 1))), 3),
                                     reason="(mock) rating from the mechanism class")
                   for i, name in enumerate(names)]
        mechanism = item.descriptors.get("mechanism_class")
        objective = round(base * factor * _MOCK_OBJECTIVE_SHARE.get(quantity, 0.1), 4)
        relabelled = False
        parents = order_parents(item.order)
        origin = item.descriptors.get("inspiration_origin")
        if item.order.get("strategy") in ("fill_gap", "diversify") and parents and key % 2 == 0:
            relabelled = f"inspiration_origin={origin}" not in str(parents[0].get("cell") or "")
        # Version-6 fields: novelty (direct seeds are closer to known solutions), a failed hard check for
        # about one in nine concepts, a contradiction when the simulation shows no gain for a claimed one.
        seed = item.order.get("strategy") == "seed_direct"
        novelty = 0.3 if seed else round(0.4 + 0.1 * (key % 5), 2)
        hard_failed = key % 9 == 0
        hard_checks = [HardCheck(name=name, passed=not (hard_failed and name == "physical_bounds"),
                                 reason="(mock) claimed effect exceeds a simple bound" if hard_failed and
                                 name == "physical_bounds" else "(mock) ok")
                       for name in HARD_CHECK_NAMES]
        contradicts = bool(usable is not None and usable <= 0 and (estimate or 0) > 0)
        scope = bool(getattr(item.candidate, "scope_extension", False))
        return MaterialsCriticReview(
            order_id=item.order["order_id"],
            hard_checks=hard_checks,
            simulation_contradicts_claim=contradicts,
            simulation_consistency_note="(mock) simulated gain vs claim",
            novelty_rating=novelty,
            novelty_reason="(mock) closest known: silicone foul-release coatings",
            scope_extension_legitimate=True if scope else None,
            scope_extension_note="(mock) the analysis justified this lever" if scope else "",
            plausible_simulated_benefit_pct=round(base * factor, 4),
            plausible_objective_gain_pct=objective,
            conventional_equivalent_gain_pct=round(objective * _MOCK_CONVENTIONAL_SHARE.get(mechanism, 0.0), 4),
            plausible_gain_reasoning=f"(mock) simulated or estimated gain x {factor}",
            relabelled_analogue=relabelled,
            relabel_reason=f"(mock) same physics as '{parents[0].get('title')}'" if relabelled else "",
            baseline_conventional=True,
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
                contents=critic_prompt(self.space, ctx.query, items, results, requirements,
                                       objective_statement=ctx.objective_statement,
                                       baseline_statement=ctx.baseline_statement,
                                       relevant_quantities=(ctx.relevant or {}).get("governing_quantity") or (),
                                       analysis_text=ctx.analysis_text or ""),
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
                                           requirements=ctx.requirements, critic_status=status,
                                           relevant_quantities=(ctx.relevant or {}).get("governing_quantity"),
                                           governing_quantity=item.descriptors.get("governing_quantity"),
                                           objective_scale_pct=ctx.objective_scale_pct,
                                           objective_scale_source=ctx.objective_scale_source,
                                           scoring=ScoringConfig.from_dict(ctx.scoring or self.default_scoring()))
            scored.breakdown["critic_used"] = review is not None
            if critic_error:
                scored.breakdown["critic_error"] = critic_error[:300]
            res.score, res.breakdown = scored.score, scored.breakdown
            res.raw["critic_review"] = review.model_dump() if review is not None else None
            res.raw["score_breakdown"] = scored.breakdown
        return results
