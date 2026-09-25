# -*- coding: utf-8 -*-
"""
Business profile: business ideas scored by a transparent, deterministic unit-economics
model.

Important: every input of that model is an estimate made by the language model itself
(optionally corrected by a second "critic" call that tries to refute the idea). The score
therefore ranks the model's own estimates in a consistent way. It is a structured
brainstorming aid, not a market validation.

Model (all amounts in EUR, time in months)::

    gross profit / customer / month   gp        = revenue_per_customer_month * gross_margin
    customer lifetime                  L         = min(1 / churn, 60)
    lifetime value                     LTV       = gp * L
    LTV / CAC, payback months          LTV / CAC, CAC / gp
    active customers after 3 years     N         = addressable_customers * reachable_share_3y
    customer-months in 3 years         18 * N    (linear ramp from 0 to N)
    revenue 3y                         R3        = revenue_per_customer_month * 18 N
    acquisitions in 3 years            A         = N * (1 + 18 * churn)   (net new + replaced)
    profit 3y                          P3        = R3 * margin - CAC * A - 3 * fixed_costs - capex
    return on spend                    ROI       = P3 / (CAC * A + 3 * fixed_costs + capex)
    break-even active customers        fixed_costs / (12 * gp - 12 * churn * CAC)

Score (0..100)::

    base  = 0.3 * clamp((LTV/CAC - 1) / 4) + 0.2 * clamp(1 - payback / 36)
          + 0.3 * clamp((ROI + 1) / 2)    + 0.2 * clamp((log10 R3 - 5) / 4)
    score = 100 * base * (1 - sanity_penalty) * (1 - critic_penalty)
"""
import math
import random
import zlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

from vectornaut.config import get_client, get_model_name, get_thinking_config
from vectornaut.explorer.archive import EVALUATED
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
    BusinessCandidate,
    BusinessCandidateBatch,
    CriticBatch,
    CriticReview,
    DescriptorAssignment,
    UnitEconomicsInputs,
)

BUSINESS_SPACE = DescriptorSpace([
    Axis("customer_segment", NOMINAL, (
        "consumers", "smb", "enterprise", "public_sector", "developers", "healthcare", "industry", "education",
    ), description="Who pays."),
    Axis("revenue_model", NOMINAL, (
        "subscription", "transaction_fee", "marketplace", "licensing", "hardware_plus_service", "advertising",
        "usage_based",
    ), description="How the business earns money."),
    Axis("market_scale", ORDINAL, ("niche", "regional", "national", "continental", "global"),
         description="Size of the target market (ordered)."),
    Axis("advantage_type", NOMINAL, ("cost", "time", "quality", "access", "compliance", "sustainability"),
         description="The main reason customers switch."),
    Axis("capital_intensity", ORDINAL, ("bootstrap", "seed", "series_a", "heavy"),
         description="Capital needed before the business can sustain itself (ordered)."),
])

LIFETIME_CAP_MONTHS = 60.0
RAMP_CUSTOMER_MONTHS_FACTOR = 18.0   # linear ramp 0 -> N over 36 months
MAX_REACHABLE_SHARE = 0.30
SANITY_PENALTY_CAP = 0.8
CRITIC_VERDICT_PENALTY = {"survives": 0.0, "weakened": 0.15, "refuted": 0.4}
CRITIC_RISK_PENALTY = 0.05
CRITIC_RISK_PENALTY_CAP = 0.15
CRITIC_PENALTY_CAP = 0.6
WEIGHTS = {"ltv_cac": 0.3, "payback": 0.2, "roi": 0.3, "scale": 0.2}

INPUT_FIELDS = tuple(UnitEconomicsInputs.model_fields)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _inputs_dict(inputs: Any) -> Dict[str, Optional[float]]:
    if inputs is None:
        return {}
    data = inputs.model_dump() if hasattr(inputs, "model_dump") else dict(inputs)
    return {name: finite(data.get(name)) for name in INPUT_FIELDS}


def sanity_flags(inputs: Mapping[str, Optional[float]], descriptors: Mapping[str, str]) -> List[Dict[str, Any]]:
    """Implausible inputs: each flag carries a code, a readable detail and a score penalty."""
    flags: List[Dict[str, Any]] = []

    def flag(code: str, detail: str, penalty: float):
        flags.append({"code": code, "detail": detail, "penalty": penalty})

    missing = [name for name in INPUT_FIELDS if inputs.get(name) is None]
    if missing:
        flag("inputs_missing", f"missing or non-finite inputs: {', '.join(missing)}", 1.0)
        return flags
    rev, margin, cac, churn = (inputs[k] for k in ("monthly_revenue_per_customer", "gross_margin", "cac", "monthly_churn"))
    addressable, share = inputs["addressable_customers"], inputs["reachable_share_3y"]
    fixed, capex = inputs["fixed_costs_per_year"], inputs["upfront_capex"]

    if rev <= 0 or addressable <= 0:
        flag("invalid_inputs", "revenue per customer and addressable customers must be > 0", 1.0)
    if churn >= 1.0:
        flag("churn_ge_100pct", f"monthly churn {churn:.0%} means no customer stays a month", 0.3)
    elif churn <= 0:
        flag("churn_non_positive", "monthly churn <= 0 means customers never leave", 0.2)
    elif churn < 0.005:
        flag("churn_implausibly_low", f"monthly churn {churn:.2%} (< 6 % a year)", 0.1)
    if margin < 0 or margin > 1:
        flag("margin_out_of_range", f"gross margin {margin} outside 0..1", 0.3)
    elif margin > 0.95 and descriptors.get("revenue_model") == "hardware_plus_service":
        flag("hardware_margin_gt_95pct", f"gross margin {margin:.0%} with hardware in the offer", 0.25)
    elif margin > 0.98:
        flag("margin_gt_98pct", f"gross margin {margin:.0%}", 0.1)
    if cac <= 0:
        flag("cac_non_positive", "customer acquisition cost <= 0", 0.3)
    if share > MAX_REACHABLE_SHARE:
        flag("reachable_share_gt_30pct", f"{share:.0%} of the addressable market after 3 years", 0.25)
    elif share < 0:
        flag("reachable_share_negative", "negative reachable share", 0.3)
    if fixed < 0 or capex < 0:
        flag("negative_costs", "fixed costs or capex below zero", 0.2)
    scale = descriptors.get("market_scale")
    if scale == "niche" and addressable > 5e6:
        flag("scale_mismatch", f"'niche' market with {addressable:,.0f} addressable customers", 0.1)
    if scale == "global" and addressable < 1e4:
        flag("scale_mismatch", f"'global' market with only {addressable:,.0f} addressable customers", 0.1)
    intensity = descriptors.get("capital_intensity")
    if intensity == "bootstrap" and capex > 5e5:
        flag("capital_mismatch", f"'bootstrap' with {capex:,.0f} EUR upfront capex", 0.1)
    if intensity == "heavy" and capex < 1e6:
        flag("capital_mismatch", f"'heavy' capital intensity with only {capex:,.0f} EUR capex", 0.1)
    if churn > 0 and cac > 0 and margin > 0 and rev > 0:
        ltv_cac = rev * _clamp(margin) * min(1.0 / churn, LIFETIME_CAP_MONTHS) / cac
        if ltv_cac > 20:
            flag("ltv_cac_implausible", f"LTV/CAC {ltv_cac:.1f} (> 20)", 0.15)
    return flags


def unit_economics(inputs: Mapping[str, Optional[float]]) -> Dict[str, Any]:
    """Deterministic formulas (see module docstring). Inputs are clamped to computable values first."""
    rev = max(inputs["monthly_revenue_per_customer"], 0.0)
    margin = _clamp(inputs["gross_margin"])
    cac = max(inputs["cac"], 1.0)
    raw_churn = inputs["monthly_churn"]
    # churn <= 0 would mean eternal customers: use the lifetime cap instead.
    churn = 1.0 / LIFETIME_CAP_MONTHS if raw_churn <= 0 else min(raw_churn, 1.0)
    addressable = max(inputs["addressable_customers"], 0.0)
    share = _clamp(inputs["reachable_share_3y"], 0.0, MAX_REACHABLE_SHARE)
    fixed = max(inputs["fixed_costs_per_year"], 0.0)
    capex = max(inputs["upfront_capex"], 0.0)

    gp = rev * margin
    lifetime = min(1.0 / churn, LIFETIME_CAP_MONTHS)
    ltv = gp * lifetime
    ltv_cac = ltv / cac
    payback = cac / gp if gp > 0 else math.inf
    customers_3y = addressable * share
    customer_months = RAMP_CUSTOMER_MONTHS_FACTOR * customers_3y
    revenue_3y = rev * customer_months
    gross_profit_3y = revenue_3y * margin
    acquisitions = customers_3y * (1.0 + RAMP_CUSTOMER_MONTHS_FACTOR * churn)
    acquisition_spend = cac * acquisitions
    spend = acquisition_spend + 3.0 * fixed + capex
    profit_3y = gross_profit_3y - spend
    roi = profit_3y / spend if spend > 0 else (math.inf if profit_3y > 0 else 0.0)
    contribution = 12.0 * gp - 12.0 * churn * cac
    break_even = fixed / contribution if contribution > 0 else math.inf
    return {
        "effective_inputs": {
            "monthly_revenue_per_customer": rev, "gross_margin": margin, "cac": cac, "monthly_churn": churn,
            "addressable_customers": addressable, "reachable_share_3y": share,
            "fixed_costs_per_year": fixed, "upfront_capex": capex,
        },
        "gross_profit_per_customer_month": gp,
        "lifetime_months": lifetime,
        "ltv": ltv,
        "ltv_cac": ltv_cac,
        "payback_months": payback,
        "customers_3y": customers_3y,
        "revenue_3y": revenue_3y,
        "gross_profit_3y": gross_profit_3y,
        "acquisitions_3y": acquisitions,
        "acquisition_spend_3y": acquisition_spend,
        "profit_3y": profit_3y,
        "roi_3y": roi,
        "annual_contribution_per_customer": contribution,
        "break_even_customers": break_even,
    }


def base_components(econ: Mapping[str, Any]) -> Dict[str, float]:
    revenue = econ["revenue_3y"]
    return {
        "ltv_cac": _clamp((econ["ltv_cac"] - 1.0) / 4.0),
        "payback": _clamp(1.0 - econ["payback_months"] / 36.0) if math.isfinite(econ["payback_months"]) else 0.0,
        "roi": _clamp((econ["roi_3y"] + 1.0) / 2.0) if math.isfinite(econ["roi_3y"]) else 1.0,
        "scale": _clamp((math.log10(revenue) - 5.0) / 4.0) if revenue > 0 else 0.0,
    }


def normalize_verdict(raw: Any) -> str:
    token = normalize_token(raw)
    return token if token in CRITIC_VERDICT_PENALTY else "weakened"


def critic_penalty(review: Optional[Any]) -> float:
    if review is None:
        return 0.0
    verdict = normalize_verdict(getattr(review, "verdict", None))
    risks = [r for r in (getattr(review, "killer_risks", None) or []) if str(r).strip()]
    penalty = CRITIC_VERDICT_PENALTY[verdict] + min(CRITIC_RISK_PENALTY * len(risks), CRITIC_RISK_PENALTY_CAP)
    return min(penalty, CRITIC_PENALTY_CAP)


def _json_number(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def score_business(inputs: Any, descriptors: Mapping[str, str], review: Optional[Any] = None) -> EvaluationResult:
    original = _inputs_dict(inputs)
    adjusted = _inputs_dict(getattr(review, "adjusted_inputs", None)) if review is not None else {}
    use_adjusted = bool(adjusted) and all(adjusted.get(k) is not None for k in INPUT_FIELDS)
    final = adjusted if use_adjusted else original

    original_flags = sanity_flags(original, descriptors)
    flags = sanity_flags(final, descriptors)
    breakdown: Dict[str, Any] = {
        "basis": "self-estimated unit economics" + (" (critic-adjusted)" if use_adjusted else ""),
        "formula": "100 * (0.3*ltv_cac + 0.2*payback + 0.3*roi + 0.2*scale) * (1 - sanity_penalty) * (1 - critic_penalty)",
        "inputs_original": original,
        "inputs_used": final,
        "inputs_source": "critic" if use_adjusted else "generator",
        "sanity_flags": flags,
        "sanity_flags_original_inputs": original_flags,
    }
    if any(f["penalty"] >= 1.0 for f in flags):
        breakdown.update({"components": {k: 0.0 for k in WEIGHTS}, "sanity_penalty": 1.0, "critic_penalty": None})
        return EvaluationResult(status=EVALUATED, score=0.0, breakdown=breakdown,
                                reason="; ".join(f["detail"] for f in flags if f["penalty"] >= 1.0))

    econ = unit_economics(final)
    components = base_components(econ)
    base = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    sanity = min(sum(f["penalty"] for f in flags), SANITY_PENALTY_CAP)
    critic = critic_penalty(review)
    score = 100.0 * base * (1.0 - sanity) * (1.0 - critic)
    breakdown.update({
        "components": {k: round(v, 6) for k, v in components.items()},
        "base": round(base, 6),
        "sanity_penalty": round(sanity, 6),
        "critic_penalty": round(critic, 6),
        "critic_verdict": normalize_verdict(review.verdict) if review is not None else None,
        "critic_killer_risks": list(review.killer_risks) if review is not None else [],
        "economics": {k: (_json_number(v) if not isinstance(v, dict) else v) for k, v in econ.items()},
    })
    return EvaluationResult(status=EVALUATED, score=round(score, 4), breakdown=breakdown)


# ---------------------------------------------------------------------------
# mock generator and critic (synthetic landscape for offline tests)
# ---------------------------------------------------------------------------

_SEG = {  # arpu, cac, churn, national addressable
    "consumers": (12.0, 30.0, 0.06, 5e6), "smb": (150.0, 900.0, 0.03, 3e5),
    "enterprise": (2500.0, 25000.0, 0.012, 3e3), "public_sector": (1500.0, 20000.0, 0.01, 2e3),
    "developers": (40.0, 150.0, 0.05, 3e5), "healthcare": (900.0, 8000.0, 0.015, 2e4),
    "industry": (3000.0, 30000.0, 0.012, 1e4), "education": (300.0, 2500.0, 0.02, 3e4),
}
_REV = {  # margin, revenue multiplier
    "subscription": (0.8, 1.0), "transaction_fee": (0.7, 1.0), "marketplace": (0.75, 0.9),
    "licensing": (0.9, 1.1), "hardware_plus_service": (0.45, 1.6), "advertising": (0.6, 0.3),
    "usage_based": (0.7, 1.1),
}
_SCALE = {"niche": (0.05, 0.08), "regional": (0.3, 0.04), "national": (1.0, 0.02),
          "continental": (8.0, 0.01), "global": (40.0, 0.006)}
_ADV = {"cost": 1.2, "time": 1.0, "quality": 0.9, "access": 1.3, "compliance": 1.1, "sustainability": 0.8}
_CAPITAL = {"bootstrap": (1.2e5, 2e4), "seed": (6e5, 3e5), "series_a": (2.5e6, 2e6), "heavy": (8e6, 2e7)}
_PRIOR = {
    "customer_segment": ("consumers", "smb"),
    "revenue_model": ("subscription", "marketplace"),
    "market_scale": ("national",),
    "advantage_type": ("cost", "time"),
    "capital_intensity": ("seed",),
}


def mock_inputs(descriptors: Mapping[str, str], jitter: float = 1.0) -> UnitEconomicsInputs:
    arpu, cac, churn, addressable = _SEG[descriptors["customer_segment"]]
    margin, rev_mult = _REV[descriptors["revenue_model"]]
    scale_mult, share = _SCALE[descriptors["market_scale"]]
    fixed, capex = _CAPITAL[descriptors["capital_intensity"]]
    if descriptors["revenue_model"] == "hardware_plus_service" and descriptors["advantage_type"] == "sustainability":
        margin = 0.97  # deliberately implausible: exercises the sanity flags
    return UnitEconomicsInputs(
        monthly_revenue_per_customer=round(arpu * rev_mult * jitter, 4),
        gross_margin=margin,
        cac=cac,
        monthly_churn=churn,
        addressable_customers=addressable * scale_mult,
        reachable_share_3y=round(min(share * _ADV[descriptors["advantage_type"]], 0.3), 6),
        fixed_costs_per_year=fixed,
        upfront_capex=capex,
    )


def mock_infeasible_reason(descriptors: Mapping[str, str]) -> Optional[str]:
    if descriptors.get("customer_segment") == "public_sector" and descriptors.get("revenue_model") == "advertising":
        return "(mock) Public authorities do not buy advertising reach and may not show ads in official services."
    return None


class BusinessProfile(ExplorerProfile):
    name = "business"
    title = "Business ideas"
    space = BUSINESS_SPACE
    report_axes = ("customer_segment", "revenue_model")
    batch_schema = BusinessCandidateBatch
    textbook_solutions = (
        "Generic 'Uber for X' marketplaces",
        "Horizontal note-taking / to-do SaaS",
        "Meal-kit subscriptions",
        "Generic AI chatbot wrapper without proprietary data",
        "Food delivery apps",
    )

    def __init__(self, critic_client: Any = None):
        self.critic_client = critic_client

    def domain_brief(self) -> str:
        return ("business ideas (products, services, platforms) answering the request, judged by a transparent "
                "unit-economics model built from your own estimates.")

    def evaluator_brief(self) -> str:
        return (
            "- A deterministic unit-economics model computes LTV, LTV/CAC, payback, 3-year revenue and profit and\n"
            "  break-even customers from the eight inputs you estimate (fields of 'inputs').\n"
            "- Implausible inputs are penalised (churn >= 100 %/month, gross margin > 95 % with hardware, CAC <= 0,\n"
            "  more than 30 % of the addressable market after 3 years, ...). A critic may correct your inputs.\n"
            "- Ideas whose value cannot be expressed in these inputs (pure research, non-profits) cannot be scored."
        )

    def candidate_instructions(self) -> str:
        return (
            "- Business fields: value_proposition, target_customer, advantage_mechanism, all eight 'inputs' (EUR,\n"
            "  fractions as 0..1), and input_assumptions naming the basis of each estimate. Estimate honestly: an\n"
            "  optimistic input is corrected by the critic and costs score.\n"
            "- back_of_envelope: e.g. year-3 revenue = customers x revenue per customer x 12."
        )

    def sanity_issues(self, candidate: Any, descriptors: Mapping[str, str]) -> List[str]:
        issues = []
        if candidate.inputs is None:
            issues.append("inputs are missing: the unit-economics model cannot run")
        else:
            data = _inputs_dict(candidate.inputs)
            bad = [k for k, v in data.items() if v is None]
            if bad:
                issues.append(f"non-finite inputs: {', '.join(bad)}")
        if not (candidate.value_proposition or "").strip():
            issues.append("value_proposition is empty")
        return issues

    # ---- mock ---------------------------------------------------------
    def mock_candidate(self, order: Mapping[str, Any]) -> BusinessCandidate:
        order_id = order["order_id"]
        rng = random.Random(zlib.crc32(order_id.encode("utf-8")))
        target = dict(order.get("target") or {})
        descriptors = {axis.name: target.get(axis.name) or rng.choice(_PRIOR[axis.name]) for axis in self.space.axes}
        if self.space.is_full(target) and mock_infeasible_reason(target):
            return BusinessCandidate(order_id=order_id, target_feasible=False,
                                     infeasibility_reason=mock_infeasible_reason(target))
        if mock_infeasible_reason(descriptors):
            descriptors["revenue_model"] = "subscription"
        jitter = rng.uniform(0.95, 1.2) if order.get("strategy") == "refine" else 1.0
        inputs = mock_inputs(descriptors, jitter)
        label = " / ".join(descriptors[name] for name in self.space.names)
        revenue = inputs.monthly_revenue_per_customer * 12 * inputs.addressable_customers * inputs.reachable_share_3y
        return BusinessCandidate(
            order_id=order_id,
            title=f"Mock {label} business ({order_id})",
            summary=f"Mock business for {label}.",
            descriptors=[DescriptorAssignment(axis=k, value=v) for k, v in descriptors.items()],
            back_of_envelope=BackOfEnvelope(quantity="year-3 revenue", formula="customers x ARPU x 12",
                                            value=round(revenue, 2), unit="EUR"),
            main_risk="mock risk: customers do not switch",
            novelty_vs_known="mock: none",
            value_proposition=f"Mock value for {descriptors['customer_segment']}",
            target_customer=descriptors["customer_segment"],
            advantage_mechanism=f"mock {descriptors['advantage_type']} advantage",
            inputs=inputs,
            input_assumptions="mock segment averages",
        )

    def mock_batch(self, query: str, orders: Sequence[Mapping[str, Any]]) -> BusinessCandidateBatch:
        return BusinessCandidateBatch(
            function_analysis="(mock) customers need the job done cheaper",
            mechanism_classes_considered=["automation", "pooling"],
            analogues_considered=["(mock) cooperative buying"],
            candidates=[self.mock_candidate(order) for order in orders],
        )

    @staticmethod
    def mock_review(item: PreparedCandidate) -> CriticReview:
        inputs = item.candidate.inputs
        adjusted = UnitEconomicsInputs(**{
            **inputs.model_dump(),
            "cac": inputs.cac * 1.3,
            "reachable_share_3y": inputs.reachable_share_3y * 0.7,
            "monthly_churn": min(inputs.monthly_churn * 1.2, 0.99),
        })
        econ = unit_economics(_inputs_dict(adjusted))
        verdict = "refuted" if econ["ltv_cac"] < 1 else "weakened" if econ["ltv_cac"] < 3 else "survives"
        return CriticReview(
            order_id=item.order["order_id"], verdict=verdict,
            killer_risks=[f"(mock) incumbents in {item.descriptors['customer_segment']} react on price"],
            adjusted_inputs=adjusted, notes="(mock) CAC +30 %, share -30 %, churn +20 %",
        )

    # ---- critic -------------------------------------------------------
    def critic_prompt(self, query: str, items: Sequence[PreparedCandidate]) -> str:
        lines = []
        for item in items:
            c = item.candidate
            lines.append(f"[{item.order['order_id']}] {c.title}")
            lines.append(f"  cell: {self.space.describe(item.descriptors)}")
            lines.append(f"  value proposition: {c.value_proposition}")
            lines.append(f"  target customer: {c.target_customer}; advantage: {c.advantage_mechanism}")
            lines.append(f"  inputs: {c.inputs.model_dump_json()}")
            lines.append(f"  input assumptions: {c.input_assumptions}")
            lines.append(f"  stated main risk: {c.main_risk}")
        candidates = "\n".join(lines)
        return f"""You are the Critic of Vectornaut's business explorer. Try to refute each business idea below.
The ideas answer the request: "{query}"

For every candidate:
- killer_risks: the risks that could kill it (regulation, incumbents, distribution, unit costs,
  willingness to pay), most severe first. Be concrete; do not list generic startup risks.
- adjusted_inputs: all eight unit-economics inputs as you would estimate them (EUR, fractions 0..1),
  or null if the original estimates are plausible. Typical corrections: CAC too low, churn too
  low, reachable share too high, margins that ignore delivery costs.
- verdict: 'survives' (plausible as estimated), 'weakened' (works only with your corrections) or
  'refuted' (a killer risk makes it unworkable).
- notes: why you adjusted the inputs.

CANDIDATES
{candidates}
"""

    def run_critic(self, items: Sequence[PreparedCandidate], ctx: EvaluationContext) -> Dict[str, CriticReview]:
        if ctx.mock:
            reviews = [self.mock_review(item) for item in items]
        else:
            from google.genai import types

            client = ctx.critic_client or self.critic_client or get_client()
            response = client.models.generate_content(
                model=get_model_name("critic"),
                contents=self.critic_prompt(ctx.query, items),
                config=types.GenerateContentConfig(
                    thinking_config=get_thinking_config("critic", default="medium"),
                    response_mime_type="application/json",
                    response_schema=CriticBatch,
                ),
            )
            parsed = getattr(response, "parsed", None)
            if parsed is None:
                parsed = CriticBatch.model_validate_json(response.text)
            reviews = list(parsed.reviews)
        wanted = {item.order["order_id"] for item in items}
        result: Dict[str, CriticReview] = {}
        for review in reviews:
            if review.order_id in wanted and review.order_id not in result:
                result[review.order_id] = review
        return result

    # ---- evaluation ---------------------------------------------------
    def evaluate(self, items: Sequence[PreparedCandidate], ctx: EvaluationContext) -> List[EvaluationResult]:
        reviews = self.run_critic(items, ctx) if (ctx.use_critic and items) else {}
        results = []
        for item in items:
            review = reviews.get(item.order["order_id"])
            result = score_business(item.candidate.inputs, item.descriptors, review)
            result.breakdown["critic_used"] = review is not None
            result.raw = {
                "candidate": item.candidate.model_dump(),
                "critic_review": review.model_dump() if review is not None else None,
                "score_breakdown": result.breakdown,
            }
            results.append(result)
        return results
