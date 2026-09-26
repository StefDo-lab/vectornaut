# -*- coding: utf-8 -*-
"""
Candidate generator: one model call per batch, one candidate per search order.

The prompt follows a fixed structure (function analysis -> mechanism classes ->
analogue search -> candidates), states every search order explicitly (target cell,
why, context), lists nearby archived concepts and textbook solutions to avoid, and lets
the model report that a target cell cannot contain a working concept.

``check_batch`` matches candidates to orders and applies the deterministic checks:
descriptor vocabulary, required fields, finite numbers, duplicate titles, and the
profile's own sanity checks. It also reads the request analysis of step 1 (requirements,
relevant values of relevance axes) and validates it against the vocabulary.

Order contexts are shortened field by field (``compact_context``), so the JSON in the
prompt always stays valid and the key fields (ids, titles, scores, main risks) survive.
"""
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from vectornaut.config import get_client, get_model_name, get_thinking_config
from vectornaut.explorer.archive import Archive
from vectornaut.explorer.descriptors import DescriptorError, normalize_token
from vectornaut.explorer.profiles.base import ExplorerProfile

# Item statuses after checking.
OK = "ok"
MISSING = "missing"
INVALID = "invalid"
REJECTED = "rejected"
TARGET_INFEASIBLE = "target_infeasible"

MAX_EXCLUDED_TITLES = 30
MAX_CONTEXT_CHARS = 2000
# Fields kept longer when the context is shortened (never dropped).
KEY_FIELDS = ("id", "title", "score", "basis", "evidence_tier", "flags", "requirement_coverage", "main_risk",
              "differs_in", "neighbour_value", "gap_value")
KEY_FIELD_MIN_CHARS = 160
# Tried in this order until the JSON fits; below the last step long text is dropped.
STRING_LIMITS = (400, 240, 160, 100, 60)
LIST_LIMIT = 8
DROPPABLE_FIELDS = ("summary", "points", "cell")
ELLIPSIS = "…"


@dataclass
class GeneratedItem:
    order: Dict[str, Any]
    candidate: Any = None
    status: str = MISSING
    descriptors: Optional[Dict[str, str]] = None
    issues: List[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    items: List[GeneratedItem]
    prompt: str
    batch_notes: Dict[str, Any]
    unmatched: List[str]


def _shorten(value: Any, limit: int, key: Optional[str] = None, drop: Sequence[str] = (),
             key_min: int = KEY_FIELD_MIN_CHARS) -> Any:
    """Recursively shortens strings (key fields keep at least ``key_min`` characters) and long lists."""
    if isinstance(value, str):
        cap = max(limit, key_min) if key in KEY_FIELDS else limit
        return value if len(value) <= cap else value[: max(cap - 1, 1)].rstrip() + ELLIPSIS
    if isinstance(value, Mapping):
        return {k: _shorten(v, limit, key=str(k), drop=drop, key_min=key_min) for k, v in value.items() if k not in drop}
    if isinstance(value, (list, tuple)):
        items = [_shorten(v, limit, key=key, drop=drop, key_min=key_min) for v in list(value)[:LIST_LIMIT]]
        if len(value) > LIST_LIMIT:
            items.append(f"{ELLIPSIS} {len(value) - LIST_LIMIT} more")
        return items
    if isinstance(value, float) and math.isfinite(value):
        return round(value, 4)
    return value


def compact_context(value: Any, limit: int = MAX_CONTEXT_CHARS) -> str:
    """
    JSON text of an order context of at most ``limit`` characters where possible. Long text
    fields are cut per field (with an ellipsis), then droppable fields (summary, trend points,
    cell descriptions) are removed; the result is always valid JSON and key fields are kept.
    """
    def dump(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, default=str)

    text = dump(value)
    if len(text) <= limit:
        return text
    for string_limit in STRING_LIMITS:
        text = dump(_shorten(value, string_limit))
        if len(text) <= limit:
            return text
    for count in range(1, len(DROPPABLE_FIELDS) + 1):
        text = dump(_shorten(value, STRING_LIMITS[-1], drop=DROPPABLE_FIELDS[:count]))
        if len(text) <= limit:
            return text
    # Last resort: shorter key fields too (they are never dropped).
    for key_min in (100, 60, 40):
        text = dump(_shorten(value, 40, drop=DROPPABLE_FIELDS, key_min=key_min))
        if len(text) <= limit:
            return text
    return text


def _compact(value: Any, limit: int = MAX_CONTEXT_CHARS) -> str:
    return compact_context(value, limit)


def neighbourhood_titles(archive: Archive, orders: Sequence[Mapping[str, Any]], limit: int = MAX_EXCLUDED_TITLES) -> List[str]:
    """Titles of archived concepts in or next to the targeted cells (plus all elites), sorted."""
    space = archive.space
    titles = set()
    for entry in archive.elite_entries():
        if entry.get("title"):
            titles.add(entry["title"])
    for entry in archive.entries.values():
        descriptors = entry.get("descriptors")
        title = entry.get("title")
        if not descriptors or not title:
            continue
        for order in orders:
            target = order.get("target") or {}
            if not target:
                continue
            if space.is_full(target):
                if space.distance(descriptors, target) <= 1:
                    titles.add(title)
                    break
            elif space.matches(descriptors, target):
                titles.add(title)
                break
    return sorted(titles)[:limit]


def build_prompt(profile: ExplorerProfile, query: str, orders: Sequence[Mapping[str, Any]], archive: Archive) -> str:
    space = profile.space
    lines: List[str] = []
    for order in orders:
        target = order.get("target") or {}
        if not target:
            target_text = "no target: choose freely"
        elif space.is_full(target):
            target_text = f"{space.describe(target)} (all axes fixed)"
        else:
            target_text = f"{space.describe(target)} (only these axes fixed; choose the others)"
        lines.append(f"[{order['order_id']}] strategy={order['strategy']}")
        lines.append(f"  target cell: {target_text}")
        lines.append(f"  why: {order.get('rationale', '')}")
        if order.get("context"):
            lines.append(f"  context: {_compact(order['context'])}")
    orders_text = "\n".join(lines)
    excluded = neighbourhood_titles(archive, orders)
    excluded_text = "\n".join(f"- {title}" for title in excluded) or "- (none yet)"
    textbook_text = "\n".join(f"- {item}" for item in profile.textbook_solutions) or "- (none)"
    requirements = archive.requirements
    if requirements:
        requirements_text = (
            "REQUIREMENTS OF THE REQUEST (fixed for this map; a critic rates every candidate against each of them,\n"
            "and requirement coverage counts as much as the simulated benefit; repeat them in `requirements`):\n"
            + "\n".join(f"- {r['name']}: {r['criterion']}" for r in requirements)
        )
    else:
        requirements_text = (
            "REQUIREMENTS OF THE REQUEST: not extracted yet. In step 1, list every explicit or clearly implied\n"
            "requirement in `requirements` (3-6 items: short snake_case name + one-line criterion). A critic\n"
            "will rate every candidate against them, and coverage counts as much as the simulated benefit."
        )
    analysis_text = profile.analysis_instructions(archive)

    return f"""You are the Explorer of Vectornaut. Vectornaut keeps a map of the idea space for a request:
every concept is placed in one cell of a fixed grid of descriptor axes, and the best concept
per cell is kept. You receive search orders that point at specific cells (gaps next to good
concepts, one step beyond a trend, mixtures of distant good concepts). Answer every order
with one concrete candidate, or say that the target cell cannot contain a working concept.

REQUEST: "{query}"

DOMAIN: {profile.domain_brief()}

WHAT THE EVALUATOR CAN CHECK (propose only concepts it can evaluate):
{profile.evaluator_brief()}

THE MAP (closed vocabulary: use exactly these axis names and value tokens):
{space.vocabulary_text()}

WORK IN FOUR STEPS
1. Function analysis (function_analysis, requirements): which functions must a solution to the
   request deliver, independent of any particular solution, and which requirements must it meet?
2. Mechanism classes (mechanism_classes_considered): which classes of mechanism can deliver
   those functions?
3. Analogue search (analogues_considered): where in distant fields (biology, geology,
   atmosphere/ocean, other technologies, other industries) is each mechanism already at work?
4. Candidates (candidates): for each search order, one concept that fits its target cell.

{requirements_text}
{analysis_text}
SEARCH ORDERS (answer each with exactly one candidate carrying the same order_id):
{orders_text}

DO NOT PROPOSE these again (already in the archive near the targets):
{excluded_text}

AVOID textbook solutions unless heavily adapted (say how in novelty_vs_known):
{textbook_text}

RULES
- descriptors: one entry per axis, values from the vocabulary above, describing the concept
  you actually propose. If you cannot hit the target cell, propose the closest honest concept
  and label its real cell. Never label a concept with the target cell only to satisfy the order.
- target_feasible=false: only if NO working concept can exist in the target cell (it would
  violate a physical law, or be commercially impossible by construction). Then give a
  one-sentence infeasibility_reason naming the law or constraint and leave the concept
  fields empty. Do not use it for concepts that are merely difficult or unusual.
- back_of_envelope: a rough estimate of the main benefit with the formula and the numbers
  you used, the resulting value and its unit.
- main_risk: the single most likely reason the concept fails.
- novelty_vs_known: the closest known solution and what differs.
{profile.candidate_instructions()}
"""


class CandidateGenerator:
    def __init__(self, profile: ExplorerProfile, client: Any = None, mock: bool = False):
        self.profile = profile
        self.client = client
        self.mock = mock

    def _call_model(self, prompt: str) -> Any:
        from google.genai import types

        schema = self.profile.batch_schema
        client = self.client or get_client()
        response = client.models.generate_content(
            model=get_model_name("explorer"),
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=get_thinking_config("explorer", default="medium"),
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            text = getattr(response, "text", None)
            if not text:
                raise ValueError("The explorer model returned an empty response.")
            parsed = schema.model_validate_json(text)
        elif not isinstance(parsed, schema):
            parsed = schema.model_validate(parsed if isinstance(parsed, dict) else parsed.model_dump())
        return parsed

    def generate(self, query: str, orders: Sequence[Mapping[str, Any]], archive: Archive) -> GenerationResult:
        prompt = build_prompt(self.profile, query, orders, archive)
        if not orders:
            return GenerationResult(items=[], prompt=prompt, batch_notes={}, unmatched=[])
        batch = self.profile.mock_batch(query, orders) if self.mock else self._call_model(prompt)
        return check_batch(self.profile, orders, batch, archive, prompt=prompt)


def _generic_issues(candidate: Any) -> List[str]:
    issues = []
    if not (candidate.title or "").strip():
        issues.append("title is empty")
    boe = candidate.back_of_envelope
    if boe is None:
        issues.append("back_of_envelope is missing")
    else:
        value = boe.value
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            issues.append("back_of_envelope.value is not a finite number")
        if not (boe.formula or "").strip():
            issues.append("back_of_envelope.formula is empty")
    if not (candidate.main_risk or "").strip():
        issues.append("main_risk is empty")
    return issues


def check_batch(profile: ExplorerProfile, orders: Sequence[Mapping[str, Any]], batch: Any,
                archive: Archive, prompt: str = "") -> GenerationResult:
    space = profile.space
    by_order = {order["order_id"]: GeneratedItem(order=dict(order)) for order in orders}
    unmatched: List[str] = []
    seen_titles = {normalize_token(title) for title in archive.titles() if title}

    for candidate in list(getattr(batch, "candidates", None) or []):
        item = by_order.get(candidate.order_id)
        if item is None:
            unmatched.append(f"{candidate.order_id}: unknown order_id ({candidate.title})")
            continue
        if item.candidate is not None:
            unmatched.append(f"{candidate.order_id}: second candidate for the same order ignored ({candidate.title})")
            continue
        item.candidate = candidate
        target = item.order.get("target") or {}

        if not candidate.target_feasible:
            if not target:
                item.status = REJECTED
                item.issues.append("target_feasible=false on an order without a target")
                continue
            item.status = TARGET_INFEASIBLE
            item.issues.append((candidate.infeasibility_reason or "").strip() or "no reason given")
            continue

        try:
            item.descriptors = space.validate_pairs(candidate.descriptors)
        except DescriptorError as err:
            item.status = INVALID
            item.issues.extend(err.problems)
            continue

        issues = _generic_issues(candidate)
        title_token = normalize_token(candidate.title)
        if title_token and title_token in seen_titles:
            issues.append(f"duplicate of an archived or batch concept title: '{candidate.title}'")
        issues.extend(profile.sanity_issues(candidate, item.descriptors))
        if title_token:
            seen_titles.add(title_token)
        if issues:
            item.status = REJECTED
            item.issues.extend(issues)
        else:
            item.status = OK

    for item in by_order.values():
        if item.candidate is None:
            item.issues.append("the model returned no candidate for this order")

    notes = {
        "function_analysis": getattr(batch, "function_analysis", "") or "",
        "requirements": [
            {"name": str(getattr(r, "name", "") or "").strip(), "criterion": str(getattr(r, "criterion", "") or "").strip()}
            for r in (getattr(batch, "requirements", None) or []) if str(getattr(r, "name", "") or "").strip()
        ],
        "mechanism_classes_considered": list(getattr(batch, "mechanism_classes_considered", []) or []),
        "analogues_considered": list(getattr(batch, "analogues_considered", []) or []),
    }
    relevant, dropped = profile.relevant_values_from_batch(batch)
    if relevant or dropped:
        notes["relevant_values"] = relevant
    if dropped:
        notes["relevant_values_rejected"] = dropped
    return GenerationResult(items=[by_order[o["order_id"]] for o in orders], prompt=prompt,
                            batch_notes=notes, unmatched=unmatched)
