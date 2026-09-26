# -*- coding: utf-8 -*-
"""Profile interface: axes, prompt text, candidate checks, mock generator and evaluator."""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

from pydantic import BaseModel

from vectornaut.explorer.descriptors import DescriptorSpace


@dataclass
class EvaluationResult:
    status: str                     # archive.EVALUATED / FAILED / INFEASIBLE
    score: Optional[float] = None   # 0..100
    breakdown: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    raw: Any = None


@dataclass
class PreparedCandidate:
    """A generated candidate that passed descriptor and sanity checks."""
    order: Dict[str, Any]
    candidate: Any
    descriptors: Dict[str, str]
    concept: Dict[str, Any]


@dataclass
class EvaluationContext:
    query: str
    mock: bool
    epochs: int = 40
    opt_rounds: int = 1
    use_critic: bool = True
    verbose: bool = False
    critic_client: Any = None
    # The request's requirements ({name, criterion}), fixed per archive (see Archive.set_requirements).
    requirements: List[Dict[str, str]] = field(default_factory=list)
    # The request's objective over the stated service life and the conventional baseline in the
    # same condition (materials; fixed per archive, see Archive.set_framing).
    objective_statement: str = ""
    baseline_statement: str = ""
    # Relevant values per relevance axis as named by the function analysis (not by elites).
    relevant: Dict[str, List[str]] = field(default_factory=dict)
    # The archive's objective scale (materials: percent, with its source; None = profile default).
    objective_scale_pct: Optional[float] = None
    objective_scale_source: str = "default"
    extra: Dict[str, Any] = field(default_factory=dict)


def finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class ExplorerProfile:
    name: str = ""
    title: str = ""
    space: DescriptorSpace
    # Two axes for the Markdown map tables (rows, columns).
    report_axes: Tuple[str, str]
    batch_schema: Type[BaseModel]
    textbook_solutions: Sequence[str] = ()
    # Axes whose values must be relevant to the request; explore/fill_gap/diversify only use
    # relevant values (from the generator's function analysis plus the values of elites).
    relevance_axes: Tuple[str, ...] = ()
    # Axes whose value pair decides whether a mixture can work at all (materials: mechanism and
    # length scale). combine and explore prefer pairs that were feasible elsewhere; empty = no check.
    compatibility_axes: Tuple[str, ...] = ()
    # Axes that name where an idea comes from (materials: inspiration_origin). A fill_gap or
    # diversify order that changes one must take the mechanism from the new origin.
    origin_axes: Tuple[str, ...] = ()
    # combine: axes whose value the child takes from the stronger parent (evidence tier, then
    # score). The first is always anchored; each further one only if the weaker parent's value
    # was never evaluated together with the anchored value (materials: governing_quantity, then
    # mechanism_class). Empty = free mixing.
    combine_anchor_axes: Tuple[str, ...] = ()
    # extrapolate: marginal trends (best score per ordinal value) only within one value of these
    # axes (materials: mechanism_class). Empty = marginal over everything.
    trend_group_axes: Tuple[str, ...] = ()
    # Soft relevance axes (materials: mechanism_class): fill_gap and diversify only use values named
    # by the function analysis (or held by the top elites); explore reaches the others at a low weight.
    soft_relevance_axes: Tuple[str, ...] = ()

    # ---- request analysis -----------------------------------------------
    def preferred_values(self, query: str, requirements: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple[List[str], str]]:
        """Preferred values per axis derived from the request and its requirements: {axis: (values, reason)}."""
        return {}

    def migrate_archive(self, archive: Any, from_version: int) -> Optional[Dict[str, Any]]:
        """Adapts an older, vocabulary-compatible archive after loading (e.g. re-scoring); returns a summary."""
        return None

    def update_scoring(self, archive: Any, round_no: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Recomputes archive-level scoring parameters (materials: the objective scale) and re-scores; summary or None."""
        return None

    # ---- prompt text --------------------------------------------------
    def domain_brief(self) -> str:
        raise NotImplementedError

    def evaluator_brief(self) -> str:
        raise NotImplementedError

    def candidate_instructions(self) -> str:
        raise NotImplementedError

    def analysis_instructions(self, archive: Any) -> str:
        """Extra step-1 instructions (e.g. which values of a relevance axis matter)."""
        return ""

    def framing_warnings(self, archive: Any) -> List[str]:
        """Deterministic doubts about the stored objective/baseline statements (shown in prompts and reports)."""
        return []

    def relevant_values_from_batch(self, batch: Any) -> Tuple[Dict[str, List[str]], List[str]]:
        """Relevant values per relevance axis named in a generator batch, and rejected tokens."""
        return {}, []

    # ---- candidates ---------------------------------------------------
    def concept_payload(self, candidate: Any) -> Dict[str, Any]:
        data = candidate.model_dump()
        for key in ("order_id", "descriptors", "target_feasible", "infeasibility_reason"):
            data.pop(key, None)
        return data

    def sanity_issues(self, candidate: Any, descriptors: Mapping[str, str]) -> List[str]:
        """Profile-specific reasons to reject a candidate before evaluation."""
        return []

    def mock_batch(self, query: str, orders: Sequence[Mapping[str, Any]]) -> BaseModel:
        raise NotImplementedError

    # ---- evaluation ---------------------------------------------------
    def evaluate(self, items: Sequence[PreparedCandidate], ctx: EvaluationContext) -> List[EvaluationResult]:
        raise NotImplementedError
