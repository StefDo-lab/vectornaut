# -*- coding: utf-8 -*-
"""
Analysis-first stage (archive version 6): one deep, system-level analysis of the request before any
candidate batch (``get_model_name("analyst")``, schema = the profile's ``analysis_schema``, materials:
``ProblemAnalysis``).

Why: in blind comparisons on two tasks (hull coating, facade cooling) a single careful direct answer
from the same model beat the explorer's best picks. The direct answers reasoned at system level
(facade: the windows dominate the heat load -> external shading), spent their effort on a few deep
candidates, and were not biased by the evaluator's simple models. The analyst therefore

- decides the framing once, deliberately: where the load/loss/failure comes from (a quantitative
  breakdown), the levers ranked by magnitude, which levers the literal wording excludes and whether
  extending the scope is justified, the conventional in-service baseline, objective, target,
  requirements and relevant values (stored on the archive; they replace the generator's step-1 framing);
- gives its own best 3 concepts (a careful direct answer). They enter the archive as ``seed_direct``
  entries, are evaluated like every other candidate, anchor the strategies, and the map report
  compares the best map finds with them ("Did the map beat the direct answer?").

``analysis_brief`` renders the stored analysis compactly for the generator and critic prompts.
"""
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from vectornaut.config import get_client, get_model_name, get_thinking_config
from vectornaut.explorer.schemas import BackOfEnvelope

SEED_DIRECT = "seed_direct"
# The analyst's careful direct answer: its best this-many concepts become seeds (extra ones are ignored).
MAX_DIRECT_CONCEPTS = 3
# Fields of a direct concept kept in the stored analysis (the full candidate is the seed entry).
DIRECT_SUMMARY_FIELDS = ("title", "summary", "key_physics", "realistic_benefit_pct", "benefit_reasoning", "risks",
                         "self_critique", "scope_extension", "scope_extension_reason")
MAX_BRIEF_CHARS = 2500


def _clean(text: Any) -> str:
    return " ".join(str(text if text is not None else "").split())


def _cut(text: Any, limit: int) -> str:
    text = _clean(text)
    return text if len(text) <= limit else text[: max(limit - 1, 1)].rstrip() + "…"


def analysis_to_dict(analysis: Any) -> Dict[str, Any]:
    """The analysis as stored on the archive (direct concepts compactly; the full seeds are entries)."""
    data = analysis.model_dump() if hasattr(analysis, "model_dump") else dict(analysis or {})
    concepts = []
    for concept in (data.pop("direct_concepts", None) or [])[:MAX_DIRECT_CONCEPTS]:
        item = {key: concept.get(key) for key in DIRECT_SUMMARY_FIELDS if key in concept}
        item["descriptors"] = {d.get("axis"): d.get("value") for d in concept.get("descriptors") or []
                               if isinstance(d, Mapping)}
        concepts.append(item)
    data["direct_concepts"] = concepts
    return data


def analysis_notes(analysis: Any, relevant: Mapping[str, List[str]], dropped: Sequence[str]) -> Dict[str, Any]:
    """The framing part of an analysis in the shape of generator batch notes (for the archive update)."""
    notes: Dict[str, Any] = {
        "function_analysis": _clean(getattr(analysis, "system_analysis", "")),
        "objective_statement": _clean(getattr(analysis, "objective_statement", "")),
        "baseline_statement": _clean(getattr(analysis, "baseline_statement", "")),
        "requirements": [
            {"name": _clean(getattr(r, "name", "")), "criterion": _clean(getattr(r, "criterion", "")),
             "priority": getattr(r, "priority", None)}
            for r in getattr(analysis, "requirements", None) or [] if _clean(getattr(r, "name", ""))
        ],
        "source": "analyst",
    }
    target = getattr(analysis, "target_gain_pct", None)
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        notes["target_gain_pct"] = float(target)
    if relevant or dropped:
        notes["relevant_values"] = dict(relevant)
    if dropped:
        notes["relevant_values_rejected"] = list(dropped)
    return notes


def analysis_brief(analysis: Optional[Mapping[str, Any]], limit: int = MAX_BRIEF_CHARS,
                   seed_scores: Optional[Mapping[str, Any]] = None) -> str:
    """
    Compact text of the stored analysis for later prompts: the load breakdown, the levers ranked by
    magnitude (with scope marks), the scope decision, the conventional baseline and the direct-answer
    seeds. Shortened until it fits ``limit`` characters.
    """
    if not analysis:
        return ""

    def render(text_limit: int, max_items: int) -> str:
        lines: List[str] = []
        if analysis.get("system_analysis"):
            lines.append(f"System analysis: {_cut(analysis['system_analysis'], text_limit * 2)}")
        loads = analysis.get("load_breakdown") or []
        if loads:
            parts = []
            for item in loads[:max_items]:
                share = item.get("share_pct")
                share_text = f" {share:g} %" if isinstance(share, (int, float)) else ""
                rough = f" ({_cut(item.get('rough_value'), text_limit // 2)})" if item.get("rough_value") else ""
                parts.append(f"{_cut(item.get('name'), 80)}{share_text}{rough}")
            lines.append("Load breakdown: " + "; ".join(parts))
        levers = analysis.get("levers") or []
        if levers:
            parts = []
            for i, lever in enumerate(levers[:max_items], 1):
                size = lever.get("expected_magnitude_pct")
                size_text = f" ~{size:g} %" if isinstance(size, (int, float)) else ""
                scope = ""
                if lever.get("within_literal_scope") is False:
                    scope = " [outside the literal wording; extension " + (
                        "justified]" if lever.get("extension_justified") else "not justified]")
                parts.append(f"{i}. {_cut(lever.get('name'), 80)}{size_text}{scope}")
            lines.append("Levers (largest first): " + "; ".join(parts))
        if analysis.get("scope_extension_justified") or analysis.get("scope_extension_reason"):
            verdict = "justified" if analysis.get("scope_extension_justified") else "not justified"
            lines.append(f"Scope extension: {verdict} ({_cut(analysis.get('scope_extension_reason'), text_limit)})")
        if analysis.get("conventional_baseline"):
            lines.append(f"Conventional in-service baseline: {_cut(analysis['conventional_baseline'], text_limit)}")
        seeds = analysis.get("direct_concepts") or []
        if seeds:
            parts = []
            for seed in seeds[:max_items]:
                benefit = seed.get("realistic_benefit_pct")
                benefit_text = f", realistic benefit ~{benefit:.3g} %" if isinstance(benefit, (int, float)) else ""
                ext = " [scope extension]" if seed.get("scope_extension") else ""
                score = (seed_scores or {}).get(seed.get("title"))
                score_text = f", map score {score:.1f}" if isinstance(score, (int, float)) else ""
                parts.append(f"'{_cut(seed.get('title'), 90)}'{benefit_text}{score_text}{ext}")
            lines.append("Direct-answer seeds (the map must beat these): " + "; ".join(parts))
        return "\n".join(lines)

    for text_limit, max_items in ((400, 6), (240, 5), (160, 4), (100, 3), (60, 3)):
        text = render(text_limit, max_items)
        if len(text) <= limit:
            return text
    return text[: limit - 1] + "…"


class ProblemAnalyst:
    """One analyst call per archive (mock: the profile's deterministic ``mock_analysis``)."""

    def __init__(self, profile: Any, client: Any = None, mock: bool = False):
        self.profile = profile
        self.client = client
        self.mock = mock

    @property
    def supported(self) -> bool:
        return getattr(self.profile, "analysis_schema", None) is not None

    def prompt(self, query: str, archive: Any) -> str:
        return self.profile.analyst_prompt(query, archive)

    def _call_model(self, prompt: str) -> Any:
        from google.genai import types

        schema = self.profile.analysis_schema
        client = self.client or get_client()
        response = client.models.generate_content(
            model=get_model_name("analyst"),
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=get_thinking_config("analyst", default="high"),
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            text = getattr(response, "text", None)
            if not text:
                raise ValueError("The analyst model returned an empty response.")
            parsed = schema.model_validate_json(text)
        elif not isinstance(parsed, schema):
            parsed = schema.model_validate(parsed if isinstance(parsed, dict) else parsed.model_dump())
        return parsed

    def analyse(self, query: str, archive: Any) -> Tuple[Any, str]:
        """(analysis, prompt)."""
        prompt = self.prompt(query, archive)
        analysis = self.profile.mock_analysis(query) if self.mock else self._call_model(prompt)
        return analysis, prompt


def complete_direct_concept(concept: Any) -> Any:
    """
    Fills candidate fields the analyst left empty from its deep fields, so a good direct answer is not
    rejected by the generic candidate checks: back_of_envelope from realistic_benefit_pct (+ reasoning),
    main_risk from the first risk, summary and physical_mechanism from key_physics. Returns the concept.
    """
    benefit = getattr(concept, "realistic_benefit_pct", None)
    if getattr(concept, "back_of_envelope", None) is None and isinstance(benefit, (int, float)) \
            and not isinstance(benefit, bool):
        concept.back_of_envelope = BackOfEnvelope(
            quantity="realistic improvement of the objective vs the in-service baseline",
            formula=_clean(getattr(concept, "benefit_reasoning", "")) or "analyst's estimate",
            value=float(benefit), unit="%")
    risks = list(getattr(concept, "risks", None) or [])
    if not _clean(getattr(concept, "main_risk", "")) and risks:
        concept.main_risk = _clean(risks[0])
    physics = _clean(getattr(concept, "key_physics", ""))
    if not _clean(getattr(concept, "summary", "")) and physics:
        concept.summary = _cut(physics, 400)
    if not _clean(getattr(concept, "physical_mechanism", "")):
        concept.physical_mechanism = physics or _clean(getattr(concept, "summary", ""))
    return concept


def seed_orders(space: Any, count: int, round_no: int) -> List[Dict[str, Any]]:
    """One search order per direct concept (strategy seed_direct, no target: the analyst chose the cell)."""
    orders = []
    for idx in range(1, count + 1):
        orders.append({
            "order_id": f"r{round_no:03d}-{idx:02d}",
            "strategy": SEED_DIRECT,
            "target": {},
            "target_key": space.cell_key({}),
            "context": {"direct_concept": idx},
            "rationale": "The analyst's careful direct answer (seed of the map); evaluated like every other candidate.",
            "parent_ids": [],
        })
    return orders
