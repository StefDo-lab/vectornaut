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
