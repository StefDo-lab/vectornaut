# -*- coding: utf-8 -*-
"""
Response schemas of the explorer's model calls (generator, business critic, materials critic).

Like the pipeline schemas in ``vectornaut.config`` they avoid free-form dicts (Gemini
response schemas handle lists of objects better), so descriptors are a list of
``{axis, value}`` pairs. Most candidate fields have defaults so that a candidate that
only reports ``target_feasible=false`` still parses.
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from vectornaut.config import ParameterProposal


class DescriptorAssignment(BaseModel):
    axis: str = Field(description="Axis name, exactly as listed in the vocabulary")
    value: str = Field(description="One allowed value of that axis, exactly as listed")


class BackOfEnvelope(BaseModel):
    quantity: str = Field(description="What is estimated, e.g. 'wall shear reduction vs smooth wall' or 'year-3 revenue'")
    formula: str = Field(description="The formula or chain of reasoning with the numbers used, e.g. 'tau_slip/tau_0 = 1/(1+b/h) = 1/(1+2e-6/1e-5)'")
    value: float = Field(description="The resulting number")
    unit: str = Field(description="Unit of the value, e.g. '%', 'EUR', 'K'")


class CandidateBase(BaseModel):
    order_id: str = Field(description="The order_id of the search order this candidate answers")
    title: str = Field(default="", description="Short, specific name of the concept")
    summary: str = Field(default="", description="Two or three sentences: what it is and why it should work")
    descriptors: List[DescriptorAssignment] = Field(default_factory=list, description="One entry per axis, describing the concept honestly (even if it misses the target cell)")
    back_of_envelope: Optional[BackOfEnvelope] = Field(default=None, description="A rough quantitative estimate of the main benefit")
    main_risk: str = Field(default="", description="The single most likely reason this concept fails")
    novelty_vs_known: str = Field(default="", description="The closest known solution and what is new compared to it")
    target_feasible: bool = Field(default=True, description="False if no concept can exist in the target cell (physically or commercially impossible); then explain in infeasibility_reason and leave the concept fields empty")
    infeasibility_reason: Optional[str] = Field(default=None, description="If target_feasible is false: why the target cell cannot contain a working concept")
    infeasibility_scope: List[str] = Field(default_factory=list, description="If target_feasible is false (optional): the axis names the impossibility depends on, e.g. ['mechanism_class', 'length_scale'] when no value of the other axes (no origin, no quantity) could make it work. At least two axes of the target; the whole pattern is then closed. Empty = only this exact target cell")


class MaterialsCandidate(CandidateBase):
    baseline: str = Field(default="", description="The baseline your back_of_envelope estimate compares against, stated explicitly (default: the map's baseline_statement, i.e. the conventional solution in the same service condition; never a parent or sibling concept)")
    inspiration_source: str = Field(default="", description="The natural or technical system the idea is borrowed from")
    domain: str = Field(default="", description="Scientific domain, e.g. Fluid Dynamics, Thermodynamics, Structural Mechanics")
    physical_mechanism: str = Field(default="", description="The physical mechanism and how it maps onto the request")
    parameters: List[ParameterProposal] = Field(default_factory=list, description="Physical parameters with SI values and plausible bounds, enough to build a 1D/2D steady model")


class UnitEconomicsInputs(BaseModel):
    monthly_revenue_per_customer: float = Field(description="Average revenue per active customer per month, EUR (for one-off sales: price times purchases per month)")
    gross_margin: float = Field(description="Gross margin as a fraction 0..1 (revenue minus cost of goods/service delivery)")
    cac: float = Field(description="Customer acquisition cost per new customer, EUR")
    monthly_churn: float = Field(description="Fraction of active customers lost per month, 0..1 (for repeat purchases: 1 - monthly repeat probability)")
    addressable_customers: float = Field(description="Number of customers that could buy this in the target market")
    reachable_share_3y: float = Field(description="Fraction of addressable customers that are active customers after 3 years, 0..1")
    fixed_costs_per_year: float = Field(description="Fixed operating costs per year (team, rent, tooling), EUR")
    upfront_capex: float = Field(description="One-off investment before launch (development, hardware, certification), EUR")


class BusinessCandidate(CandidateBase):
    value_proposition: str = Field(default="", description="What the customer gets and pays for")
    target_customer: str = Field(default="", description="Who exactly buys it")
    advantage_mechanism: str = Field(default="", description="Why this is cheaper/faster/better than the status quo, in mechanism terms")
    inputs: Optional[UnitEconomicsInputs] = Field(default=None, description="Your estimates of the unit-economics inputs")
    input_assumptions: str = Field(default="", description="One sentence per input on where the estimate comes from")


class Requirement(BaseModel):
    name: str = Field(description="Short snake_case name of the requirement, e.g. 'low_friction_drag'")
    criterion: str = Field(default="", description="One line: when a concept meets this requirement")
    priority: str = Field(default="must", description="'must' for a requirement the request states explicitly (e.g. 'without', 'at least', 'bionic', 'ohne', 'mindestens', 'bionisch'): a concept that fails it is not an answer to the request; 'nice' for an implied or desirable property (cost, practicality)")


class ExplorerCandidateBatch(BaseModel):
    function_analysis: str = Field(default="", description="Step 1: the functions the request needs, independent of any solution")
    requirements: List[Requirement] = Field(default_factory=list, description="Step 1: every explicit or clearly implied requirement of the request (3-6), each with a short name and a one-line criterion")
    mechanism_classes_considered: List[str] = Field(default_factory=list, description="Step 2: mechanism classes that could deliver those functions")
    analogues_considered: List[str] = Field(default_factory=list, description="Step 3: analogues from distant fields that were considered")


class MaterialsCandidateBatch(ExplorerCandidateBatch):
    objective_statement: str = Field(default="", description="Step 1: the request's main benefit stated as it applies over the stated service life and conditions, e.g. 'time-averaged hull friction drag over a 5-year docking interval, including the effect of fouling' (not the drag of a freshly applied clean surface if the request asks for years of service)")
    baseline_statement: str = Field(default="", description="Step 1: the conventional state-of-the-art solution a concept must beat on that objective, in the same condition, e.g. 'conventional biocide-free silicone foul-release coating after 12-24 months in service'")
    relevant_governing_quantities: List[str] = Field(default_factory=list, description="Step 1: the governing_quantity tokens (from the vocabulary) that measure a benefit the request actually asks for")
    relevant_mechanism_classes: List[str] = Field(default_factory=list, description="Step 2 (optional): the mechanism_class tokens (from the vocabulary) that can plausibly deliver the request's main benefit; targeted search orders stay within them (explore reaches the others at a low weight)")
    target_gain_pct: Optional[float] = Field(default=None, description="Step 1 (optional): the numeric improvement of the objective against the baseline, in percent, that the request (or your objective statement / requirements) asks for, e.g. 20 for '>20 %'; null if no number is stated")
    candidates: List[MaterialsCandidate] = Field(default_factory=list, description="Step 4: exactly one candidate per search order")


class BusinessCandidateBatch(ExplorerCandidateBatch):
    candidates: List[BusinessCandidate] = Field(default_factory=list, description="Step 4: exactly one candidate per search order")


class CriticReview(BaseModel):
    order_id: str = Field(description="order_id of the reviewed candidate")
    verdict: str = Field(description="'survives', 'weakened' or 'refuted'")
    killer_risks: List[str] = Field(default_factory=list, description="Risks that could kill the business, most severe first")
    adjusted_inputs: Optional[UnitEconomicsInputs] = Field(default=None, description="Your corrected unit-economics inputs (all fields), or null if the original estimates are plausible")
    notes: str = Field(default="", description="Why the inputs were adjusted")


class CriticBatch(BaseModel):
    reviews: List[CriticReview] = Field(default_factory=list, description="Exactly one review per candidate")


class RequirementRating(BaseModel):
    name: str = Field(description="Requirement name, exactly as listed")
    coverage: float = Field(description="How well the concept meets the requirement: 0 = not at all or contradicts it, 1 = fully")
    reason: str = Field(default="", description="One line: why")


class MaterialsCriticReview(BaseModel):
    order_id: str = Field(description="order_id of the reviewed candidate")
    plausible_simulated_benefit_pct: Optional[float] = Field(default=None, description="Your best estimate of the real-world improvement of the candidate's OWN simulated quantity (its governing_quantity, e.g. fouling release stress), in percent, against the conventional baseline (positive = better); null if it cannot be estimated. Do not copy the simulated number")
    plausible_objective_gain_pct: Optional[float] = Field(default=None, description="Your best estimate of the candidate's contribution to the stated OBJECTIVE against the stated BASELINE (e.g. time-averaged drag including fouling over the service life), in percent (positive = better); null if it cannot be estimated")
    conventional_equivalent_gain_pct: Optional[float] = Field(default=None, description="The objective gain in percent that a CONVENTIONAL measure achieving the same physical effect would give against the same baseline (e.g. an equal-R conventional insulation layer for a concept whose benefit is added thermal resistance; a thicker conventional coating for a benefit that comes from thickness); 0 if the effect has no conventional equivalent; null if you cannot estimate it. Only the gain beyond it counts as the concept's own")
    plausible_gain_reasoning: str = Field(default="", description="One or two sentences: how you arrived at the numbers")
    simulated_quantity_relevant: Optional[bool] = Field(default=None, description="True if the simulated quantity drives the objective or a requirement (e.g. fouling release for time-averaged drag), false if it is beside the point; null if unsure")
    relabelled_analogue: bool = Field(default=False, description="True if the concept is the same physics as its parent (or another listed concept) with only the inspiration label changed, i.e. the claimed origin does not actually supply the mechanism")
    relabel_reason: str = Field(default="", description="If relabelled_analogue: one line naming the concept it copies")
    baseline_conventional: Optional[bool] = Field(default=None, description="False if the candidate's or the simulation's baseline is not the conventional solution (e.g. a parent concept, an untreated or fouled surface, an idealised case); null if unsure")
    baseline_issue: str = Field(default="", description="If baseline_conventional is false: one line saying what the baseline is instead")
    proxy_by_construction: bool = Field(default=False, description="True if the simulated gain follows directly from an input or baseline choice rather than from modelled physics (e.g. a leaching flux compared against a baseline that contains the leaching species while the design contains none, or a gain equal to an assumed input ratio); the simulated benefit then does not count")
    proxy_reason: str = Field(default="", description="If proxy_by_construction: one line naming the input or baseline choice that produces the gain")
    key_assumption_issues: List[str] = Field(default_factory=list, description="Modelling choices that drive the simulated gain (free parameters, gap sizes, laminar models of turbulent flow, missing losses), most important first")
    killer_risks: List[str] = Field(default_factory=list, description="Risks that could make the concept unworkable in practice, most severe first")
    requirement_coverage: List[RequirementRating] = Field(default_factory=list, description="One rating per listed requirement")


class MaterialsCriticBatch(BaseModel):
    reviews: List[MaterialsCriticReview] = Field(default_factory=list, description="Exactly one review per candidate")
