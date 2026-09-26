# -*- coding: utf-8 -*-
"""
Search strategies: turn the archive into a batch of targeted search orders.

Every function here is pure (it reads the archive, never writes it) and deterministic
for a given ``random.Random``. An order is a JSON-friendly dict::

    {order_id, strategy, target, target_key, context, rationale, parent_ids}

``target`` is a full cell or a partial assignment (``{}`` for a free "seed" proposal).

Strategies
- refine:      mutate an elite inside its own cell.
- fill_gap:    empty cells at distance 1 from elites, ranked by neighbour score, elite
               neighbours along different axes, under-explored axis values and low
               proposal density.
- extrapolate: along an ordinal axis, fit the score trend over elites of the top evidence
               tier (holding the other axes fixed, or marginalised within one value of the
               group axes, materials: one mechanism class; at least ``min_trend_points`` points
               and r^2 >= ``min_r2``) and target one step beyond the explored edge in the
               improving direction. No trend or end of scale -> no order. The scheduler gives
               extrapolate slots only if a qualifying slice trend exists (``extrapolate_slots``).
- combine:     two elites far apart in descriptor space -> a cell mixing their descriptors
               (under-explored mixtures first), only if the mixture is compatible: its
               compatibility pair (materials: mechanism x length scale) was feasible elsewhere,
               or the cell lies within distance 2 of both parents; pairs only ever reported
               infeasible are skipped. Anchor axes (materials: governing_quantity, and the
               mechanism if it never worked on that quantity) come from the stronger parent.
- diversify:   the least-proposed value of the least diverse axis, one step from an elite.
- explore:     a random cell nobody has proposed or targeted yet, restricted to relevant
               values of the relevance axes (materials: governing_quantity) and weighted
               towards rare values on the other axes and towards cells near feasible entries.

fill_gap and diversify rotate their source elites: an elite's value as a source is
score/100 - source_penalty * log(1 + times it already served as a source), counted over the
archive and within the batch. Orders that change an origin axis (materials:
inspiration_origin) demand a mechanism taken from the new origin (no relabelled copies).
- seed:        no target; samples where the model proposes ideas by itself (cold start).

"Under-explored" is measured per axis value: ``concentration(axis) / (1 + proposals with
that value)``, where concentration is the share of proposals held by the axis' most
common value. A value nobody proposed on an axis where every proposal shares one value
scores 1; a value on a well-mixed axis scores little.

Preferred values (``Archive.preferred_values``; materials: biological origins for a
bio-inspired request) are the only values fill_gap, diversify, combine and extrapolate
target; explore reaches the others at ``nonpreferred_explore_factor``.
"""
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from vectornaut.explorer.archive import EVALUATED, Archive, evidence_rank, rank_key
from vectornaut.explorer.descriptors import DescriptorSpace

REFINE = "refine"
FILL_GAP = "fill_gap"
EXTRAPOLATE = "extrapolate"
COMBINE = "combine"
DIVERSIFY = "diversify"
EXPLORE = "explore"
SEED = "seed"

STRATEGIES = (REFINE, FILL_GAP, EXTRAPOLATE, COMBINE, DIVERSIFY, EXPLORE, SEED)
DEFAULT_WEIGHTS: Dict[str, float] = {
    REFINE: 0.20,
    FILL_GAP: 0.25,
    EXTRAPOLATE: 0.15,
    COMBINE: 0.15,
    DIVERSIFY: 0.15,
    EXPLORE: 0.10,
}
# Strategies claim cells in this order within a batch (so a gap is not also a combine target).
PROCESSING_ORDER = (EXTRAPOLATE, FILL_GAP, DIVERSIFY, COMBINE, REFINE, EXPLORE, SEED)
# Slots a strategy could not use go to these, in this order.
FALLBACK_ORDER = (FILL_GAP, DIVERSIFY, REFINE, EXTRAPOLATE, COMBINE, EXPLORE, SEED)


@dataclass
class StrategyConfig:
    # Minimum |slope| (score points per ordinal step) that counts as a trend.
    min_slope: float = 1.0
    # Priority lost per log(1 + earlier attempts) in a gap cell.
    density_penalty: float = 0.2
    # Bonus per additional axis along which a gap has elite neighbours (up to 3), and for
    # a gap bracketed by elites on both sides of an ordinal axis. Several elites that
    # differ from the gap on the *same* nominal axis earn nothing extra.
    neighbour_bonus: float = 0.05
    # Weight of the under-exploration term of the changed axis value (0..1) in gap and
    # combine ranking.
    rarity_bonus: float = 0.25
    max_context_elites: int = 3
    # Trend fits need this many points (same evidence tier) and at least this r^2.
    min_trend_points: int = 3
    min_r2: float = 0.5
    # Source rotation: fill_gap and diversify value a source elite at score/100 minus this times
    # log(1 + times it already served as a source), counted over the archive and the batch.
    source_penalty: float = 0.1
    # explore: weight factor exp(-decay * (d - 1)) for a cell at distance d from the nearest
    # feasible (evaluated) entry; 0 disables the adjacency bias.
    explore_distance_decay: float = 1.0
    # explore: only cells within this distance of a feasible entry (while any such cell is left);
    # 0 = no limit.
    explore_max_distance: int = 3
    # explore: weight factor for a cell whose compatibility pair was only ever reported infeasible.
    explore_infeasible_pair_factor: float = 0.1
    # Axes whose value pair decides whether a mixture can work (materials: mechanism_class,
    # length_scale); set from the profile by the runner. Empty = no compatibility check.
    compatibility_axes: Tuple[str, ...] = ()
    # combine: an unproven pair is accepted if the child is within this distance of both parents.
    combine_max_parent_distance: int = 2
    # Axes that name where an idea comes from (materials: inspiration_origin); set from the profile.
    origin_axes: Tuple[str, ...] = ()
    # combine: axes taken from the stronger parent (evidence tier, then score). The first always;
    # each further one only if the weaker parent's value was never evaluated together with the
    # anchored value (materials: governing_quantity, mechanism_class); set from the profile.
    combine_anchor_axes: Tuple[str, ...] = ()
    # extrapolate: marginal trends only within one value of these axes (materials:
    # mechanism_class); empty = marginal over all other axes. Set from the profile.
    trend_group_axes: Tuple[str, ...] = ()
    # explore: weight factor for a cell with a non-preferred value (e.g. a non-biological origin for
    # a bio-inspired request); the other strategies never target such values.
    nonpreferred_explore_factor: float = 0.1


def parse_weights(spec: Optional[str]) -> Dict[str, float]:
    """Parses 'refine=0.3,fill_gap=0.3,...'. Unlisted strategies get weight 0."""
    if not spec:
        return dict(DEFAULT_WEIGHTS)
    weights: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition("=")
        name = name.strip()
        if not sep or name not in STRATEGIES:
            raise ValueError(f"Invalid strategy weight '{part}' (strategies: {', '.join(STRATEGIES)}).")
        weight = float(value)
        if weight < 0 or not math.isfinite(weight):
            raise ValueError(f"Strategy weight for '{name}' must be a finite number >= 0.")
        weights[name] = weight
    if sum(weights.values()) <= 0:
        raise ValueError("At least one strategy weight must be > 0.")
    return weights


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def entry_summary(space: DescriptorSpace, entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact view of an entry for order contexts (key fields first)."""
    concept = entry.get("concept") or {}
    summary = concept.get("summary") or concept.get("physical_mechanism") or concept.get("value_proposition") or ""
    breakdown = entry.get("score_breakdown") or {}
    result: Dict[str, Any] = {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "score": entry.get("score"),
        "basis": breakdown.get("basis"),
    }
    if breakdown.get("evidence_tier"):
        result["evidence_tier"] = breakdown["evidence_tier"]
    if breakdown.get("flags"):
        result["flags"] = list(breakdown["flags"])
    if breakdown.get("requirement_coverage") is not None:
        result["requirement_coverage"] = breakdown["requirement_coverage"]
    if breakdown.get("objective_gain_pct") is not None:
        result["objective_gain_pct"] = breakdown["objective_gain_pct"]
    result["main_risk"] = concept.get("main_risk")
    result["cell"] = space.describe(entry.get("descriptors") or {})
    result["summary"] = str(summary)[:300]
    return result


def trend_eligible(entry: Mapping[str, Any]) -> bool:
    """
    Only simulated (or unranked) scores feed trends; estimates, implausible gains and simulated
    gains the critic calls a proxy by construction do not.
    """
    breakdown = entry.get("score_breakdown") or {}
    tier = breakdown.get("evidence_tier")
    if tier not in (None, "simulated"):
        return False
    flags = breakdown.get("flags") or []
    return "implausible_gain" not in flags and "proxy_by_construction" not in flags


class ExplorationStats:
    """Proposal counts per axis value and per-axis concentration (see module docstring)."""

    def __init__(self, archive: Archive):
        self.counts = {axis.name: archive.value_counts(axis.name) for axis in archive.space.axes}
        self.concentration = {}
        for name, counts in self.counts.items():
            total = sum(counts.values())
            self.concentration[name] = max(counts.values()) / total if total else 0.0

    def under_exploration(self, axis: str, value: str) -> float:
        return self.concentration[axis] / (1.0 + self.counts[axis].get(value, 0))


def _weakest_component(entry: Mapping[str, Any]) -> Optional[str]:
    components = (entry.get("score_breakdown") or {}).get("components") or {}
    numeric = {k: v for k, v in components.items() if isinstance(v, (int, float))}
    if not numeric:
        return None
    return min(sorted(numeric), key=lambda k: numeric[k])


def _attempts(archive: Archive, cell: Mapping[str, str]) -> int:
    return max(archive.proposal_count(cell), archive.target_count(cell))


def source_uses(archive: Archive, batch: Sequence[Mapping[str, Any]] = ()) -> Dict[str, int]:
    """Source uses over the archive (Archive.source_uses) plus the orders already scheduled in this batch."""
    uses = dict(archive.source_uses())
    for order in batch:
        if order.get("strategy") == REFINE:
            continue
        parents = list(order.get("parent_ids") or [])
        for parent in parents[:2] if order.get("strategy") == COMBINE else parents[:1]:
            uses[parent] = uses.get(parent, 0) + 1
    return uses


def source_value(entry: Mapping[str, Any], uses: Mapping[str, int], config: StrategyConfig) -> float:
    """How attractive an elite is as the source of a new order: score/100 minus the rotation penalty."""
    return float(entry.get("score") or 0.0) / 100.0 - config.source_penalty * math.log1p(uses.get(entry["id"], 0))


class PairKnowledge:
    """
    What the archive knows about value pairs of the compatibility axes (materials: mechanism x
    length scale): pairs that produced an evaluated concept somewhere (feasible) and pairs that
    only appear in infeasible reports.
    """

    def __init__(self, archive: Archive, axes: Sequence[str]):
        self.axes = tuple(a for a in axes if a in archive.space.names)
        self.feasible: Set[Tuple[str, ...]] = set()
        self.reported_infeasible: Set[Tuple[str, ...]] = set()
        if not self.axes:
            return
        for entry in archive.entries.values():
            if entry.get("status") == EVALUATED and entry.get("descriptors"):
                self.feasible.add(self.pair(entry["descriptors"]))
        for key in archive.data["infeasible"]:
            pattern = archive.space.parse_key(key)
            if all(axis in pattern for axis in self.axes):
                self.reported_infeasible.add(self.pair(pattern))

    def pair(self, cell: Mapping[str, str]) -> Tuple[str, ...]:
        return tuple(cell[axis] for axis in self.axes)

    def only_infeasible(self, cell: Mapping[str, str]) -> bool:
        if not self.axes:
            return False
        pair = self.pair(cell)
        return pair in self.reported_infeasible and pair not in self.feasible

    def describe(self, cell: Mapping[str, str]) -> str:
        return "/".join(f"{axis}={cell[axis]}" for axis in self.axes)


def combine_compatibility(space: DescriptorSpace, child: Mapping[str, str], a: Mapping[str, Any],
                          b: Mapping[str, Any], pairs: PairKnowledge, config: StrategyConfig) -> Tuple[bool, str]:
    """
    Whether a mixed cell is worth asking for. Accepted: the compatibility pair of the child was
    feasible somewhere, or the child lies within ``combine_max_parent_distance`` of both parents.
    Rejected: a pair only ever reported infeasible, or an unproven pair far from a parent (the
    recorded failure: a mm-scale mechanism moved to nm). Without compatibility axes everything
    that is not infeasible is accepted.
    """
    if not pairs.axes:
        return True, "no compatibility check for this profile"
    label = pairs.describe(child)
    if pairs.pair(child) in pairs.feasible:
        return True, f"{label} was feasible elsewhere"
    if pairs.only_infeasible(child):
        return False, f"{label} was reported infeasible and never worked"
    da, db = space.distance(child, a["descriptors"]), space.distance(child, b["descriptors"])
    if max(da, db) <= config.combine_max_parent_distance:
        return True, f"{label} is unproven, but the cell is within distance {max(da, db)} of both parents"
    return False, f"{label} is unproven and the cell is {max(da, db)} steps from a parent"


def origin_note(config: StrategyConfig, axis: str, value: str) -> str:
    """Extra instruction when an order changes an origin axis (against relabelled copies)."""
    if axis not in config.origin_axes:
        return ""
    return (f" The mechanism must really come from a {axis}={value} system: name that system and what it does there. "
            "The source concept's physics with a new origin label is a relabelled analogue and is penalised.")


def _make_order(space: DescriptorSpace, strategy: str, target: Mapping[str, str], context: Dict[str, Any],
                rationale: str, parent_ids: Sequence[str] = ()) -> Dict[str, Any]:
    target = {name: target[name] for name in space.names if name in target}
    return {
        "order_id": None,
        "strategy": strategy,
        "target": target,
        "target_key": space.cell_key(target),
        "context": context,
        "rationale": rationale,
        "parent_ids": list(parent_ids),
    }


def _weighted_pick(rng, items: List[Any], weights: List[float]) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(items))
    point = rng.random() * total
    acc = 0.0
    for idx, weight in enumerate(weights):
        acc += weight
        if point < acc:
            return idx
    return len(items) - 1


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    """Least-squares slope, intercept and r^2. Needs at least two distinct x."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("x values must not all be equal")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    syy = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 if syy == 0 else (sxy * sxy) / (sxx * syy)
    return slope, intercept, r2


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------

def refine(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig,
        batch: Sequence[Mapping[str, Any]] = ()) -> List[Dict[str, Any]]:
    space = archive.space
    elites = [e for e in archive.elite_entries() if e["cell"] not in taken]
    if n <= 0 or not elites:
        return []
    refined = defaultdict(int)
    for entry in archive.entries.values():
        if entry.get("strategy") == REFINE:
            for parent in entry.get("parent_ids") or []:
                refined[parent] += 1
    pool = list(elites)
    weights = [max(float(e.get("score") or 0.0), 1.0) / (1 + refined[e["id"]]) for e in pool]
    orders = []
    while pool and len(orders) < n:
        idx = _weighted_pick(rng, pool, weights)
        elite = pool.pop(idx)
        weights.pop(idx)
        weakest = _weakest_component(elite)
        risk = (elite.get("concept") or {}).get("main_risk")
        focus = f"its weakest score component is '{weakest}'" if weakest else "improve its score"
        if risk:
            focus += f"; its stated main risk is: {risk}"
        breakdown = elite.get("score_breakdown") or {}
        critic = breakdown.get("critic") or {}
        issues = list(critic.get("key_assumption_issues") or []) + list(critic.get("killer_risks") or [])
        if issues:
            focus += f"; the critic's main objection is: {issues[0]}"
        weak_reqs = [r["name"] for r in breakdown.get("requirements") or []
                     if isinstance(r.get("coverage"), (int, float)) and r["coverage"] < 0.5]
        if weak_reqs:
            focus += f"; weakly covered requirements: {', '.join(weak_reqs)}"
        orders.append(_make_order(
            space, REFINE, elite["descriptors"],
            context={"elite": entry_summary(space, elite), "times_refined": refined[elite["id"]]},
            rationale=(
                f"Refine elite '{elite.get('title')}' (score {float(elite.get('score') or 0):.1f}) inside its own cell. "
                f"Keep every descriptor; {focus}. It must be a better variant with a new title, not the same idea."
            ),
            parent_ids=[elite["id"]],
        ))
    return orders


def _gap_source(gap: Dict[str, Any], uses: Mapping[str, int], config: StrategyConfig) -> None:
    """Picks the gap's source elite (best rotation-adjusted value) and sets its priority."""
    ranked = sorted(gap["neighbours"], key=lambda item: (-source_value(item[0], uses, config), item[0]["id"]))
    gap["neighbours"] = ranked
    source, _ = ranked[0]
    gap["source_uses"] = int(uses.get(source["id"], 0))
    gap["priority"] = round(gap["base_priority"] + source_value(source, uses, config), 6)


def gap_candidates(archive: Archive, taken: Iterable[str] = (), config: Optional[StrategyConfig] = None,
                   uses: Optional[Mapping[str, int]] = None) -> List[Dict[str, Any]]:
    """
    Empty, feasible cells at distance 1 from elites, best first (also used by the report). The
    neighbour that serves as the source is the one with the best rotation-adjusted value
    (``source_value``), so an elite that already seeded many orders gives way to others.
    """
    config = config or StrategyConfig()
    space = archive.space
    taken = set(taken)
    uses = archive.source_uses() if uses is None else uses
    neighbours: Dict[str, List[Tuple[Dict[str, Any], str]]] = defaultdict(list)
    cells: Dict[str, Dict[str, str]] = {}
    for elite in archive.elite_entries():
        for cell in space.neighbours(elite["descriptors"]):
            key = space.cell_key(cell)
            if key in archive.elites or key in taken:
                continue
            changed = space.differing_axes(elite["descriptors"], cell)[0]
            neighbours[key].append((elite, changed))
            cells[key] = cell
    stats = ExplorationStats(archive)
    ranked = []
    for key, items in neighbours.items():
        cell = cells[key]
        if archive.is_infeasible(cell) or not archive.is_relevant(cell) or not archive.is_preferred(cell):
            continue
        scores = sorted((float(e.get("score") or 0.0) for e, _ in items), reverse=True)
        attempts = _attempts(archive, cell)
        changed_axes = sorted({axis for _, axis in items}, key=space.names.index)
        bracketed = 0
        for name in changed_axes:
            axis = space.axis(name)
            if axis.is_ordinal:
                sides = {axis.index(e["descriptors"][name]) > axis.index(cell[name]) for e, a in items if a == name}
                bracketed += 1 if len(sides) == 2 else 0
        breadth = min(len(changed_axes) - 1 + bracketed, 3)
        exploration = max(stats.under_exploration(name, cell[name]) for name in changed_axes)
        base_priority = (
            config.neighbour_bonus * breadth
            + config.rarity_bonus * exploration
            - config.density_penalty * math.log1p(attempts)
        )
        gap = {
            "cell": cell, "key": key, "base_priority": base_priority, "best_neighbour_score": scores[0],
            "neighbours": list(items), "attempts": attempts, "proposals": archive.proposal_count(cell),
            "changed_axes": changed_axes, "exploration": round(exploration, 6),
        }
        _gap_source(gap, uses, config)
        ranked.append(gap)
    ranked.sort(key=lambda item: (-item["priority"], item["key"]))
    return ranked


def fill_gap(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig,
        batch: Sequence[Mapping[str, Any]] = ()) -> List[Dict[str, Any]]:
    """
    Best gaps first, picked one at a time: after each pick its source elite counts as used once
    more, so the remaining gaps are re-ranked and the batch rotates over source elites.
    """
    space = archive.space
    uses = source_uses(archive, batch)
    remaining = gap_candidates(archive, taken, config, uses)
    chosen = []
    while remaining and len(chosen) < max(n, 0):
        gap = remaining.pop(0)
        chosen.append(gap)
        source = gap["neighbours"][0][0]
        uses[source["id"]] = uses.get(source["id"], 0) + 1
        for other in remaining:
            _gap_source(other, uses, config)
        remaining.sort(key=lambda item: (-item["priority"], item["key"]))
    orders = []
    for gap in chosen:
        top = gap["neighbours"][: config.max_context_elites]
        best, changed = top[0]
        orders.append(_make_order(
            space, FILL_GAP, gap["cell"],
            context={
                "neighbours": [
                    {**entry_summary(space, e), "differs_in": axis,
                     "neighbour_value": e["descriptors"][axis], "gap_value": gap["cell"][axis]}
                    for e, axis in top
                ],
                "proposals_here": gap["proposals"],
                "attempts_here": gap["attempts"],
                "under_exploration": gap["exploration"],
                "priority": gap["priority"],
                "source_uses": gap["source_uses"],
            },
            rationale=(
                f"Empty cell next to {len(gap['neighbours'])} elite(s); source neighbour '{best.get('title')}' "
                f"scores {float(best.get('score') or 0):.1f} with {changed}={best['descriptors'][changed]} "
                f"(served as a source {gap['source_uses']} time(s) before). "
                f"Only {gap['attempts']} earlier attempt(s) landed or aimed here. "
                f"Find a concept that works with {changed}={gap['cell'][changed]}."
                + origin_note(config, changed, gap["cell"][changed])
            ),
            parent_ids=[e["id"] for e, _ in top],
        ))
    return orders


def find_trends(archive: Archive, config: Optional[StrategyConfig] = None, taken: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """
    Score trends along every ordinal axis. Returns every fitted trend with a status:
    'proposed' (usable target), 'too_few_points', 'flat', 'peaked', 'poor_fit', 'scale_end',
    'infeasible' or 'taken'. Only elites of one evidence tier are fitted (simulated or
    unranked scores; estimates never form a trend). Slice trends (other axes held fixed)
    come first, then marginal ones; within each, the steepest well-fitting trend first.
    """
    config = config or StrategyConfig()
    space = archive.space
    taken = set(taken)
    elites = [e for e in archive.elite_entries() if e.get("score") is not None and trend_eligible(e)]
    trends: List[Dict[str, Any]] = []

    def _trend(axis, mode, fixed, points, target_extra):
        values = axis.values
        xs = [axis.index(v) for v, _ in points]
        ys = [s for _, s in points]
        slope, _, r2 = linear_fit(xs, ys)
        record = {
            "axis": axis.name, "mode": mode, "fixed": dict(fixed),
            "points": [[v, round(s, 4)] for v, s in points],
            "slope": round(slope, 4), "r2": round(r2, 4),
            "direction": "increasing" if slope > 0 else "decreasing",
            "edge": None, "next": None, "target": None, "status": "proposed",
        }
        if len(points) < config.min_trend_points:
            record["status"] = "too_few_points"
            return record
        if abs(slope) < config.min_slope:
            record["status"] = "flat"
            return record
        step = 1 if slope > 0 else -1
        edge_idx = max(xs) if step > 0 else min(xs)
        record["edge"] = values[edge_idx]
        # Points are sorted by index. The trend must still improve at the edge; otherwise
        # the optimum lies inside the explored range and a step beyond it is not a trend.
        edge_pair = (ys[-2], ys[-1]) if step > 0 else (ys[1], ys[0])
        if edge_pair[1] < edge_pair[0]:
            record["status"] = "peaked"
            return record
        if r2 < config.min_r2:
            record["status"] = "poor_fit"
            return record
        next_idx = edge_idx + step
        if not 0 <= next_idx < len(values):
            record["status"] = "scale_end"
            return record
        record["next"] = values[next_idx]
        target = dict(target_extra)
        target[axis.name] = values[next_idx]
        record["target"] = {name: target[name] for name in space.names if name in target}
        key = space.cell_key(record["target"])
        if not archive.is_preferred(record["target"]):
            record["status"] = "not_preferred"
        elif key in taken:
            record["status"] = "taken"
        elif space.is_full(record["target"]):
            if archive.is_infeasible(record["target"]):
                record["status"] = "infeasible"
        elif any(space.parse_key(k) == record["target"] for k in archive.data["infeasible"]):
            record["status"] = "infeasible"
        return record

    for axis in space.ordinal_axes:
        others = [name for name in space.names if name != axis.name]
        groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
        for entry in elites:
            groups[tuple(entry["descriptors"][name] for name in others)].append(entry)
        for rest in sorted(groups):
            members = groups[rest]
            if len({m["descriptors"][axis.name] for m in members}) < 2:
                continue
            members = sorted(members, key=lambda m: axis.index(m["descriptors"][axis.name]))
            points = [(m["descriptors"][axis.name], float(m["score"])) for m in members]
            fixed = dict(zip(others, rest))
            trends.append(_trend(axis, "slice", fixed, points, fixed))

        # Marginal: best score per value, within one value of the group axes (materials: one
        # mechanism class), so points of different mechanisms are never fitted together.
        group_axes = [name for name in config.trend_group_axes if name in space.names and name != axis.name]
        marginal_groups: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)
        for entry in elites:
            group = tuple(entry["descriptors"][name] for name in group_axes)
            value = entry["descriptors"][axis.name]
            best = marginal_groups[group]
            best[value] = max(best.get(value, -math.inf), float(entry["score"]))
        for group in sorted(marginal_groups):
            best_per_value = marginal_groups[group]
            members = [e for e in elites if tuple(e["descriptors"][name] for name in group_axes) == group]
            fixed = dict(zip(group_axes, group))
            # A marginal fit whose points are one slice is that slice (already fitted).
            if len(best_per_value) < 2 or (group_axes and len({json_key({n: e["descriptors"][n] for n in others})
                                                               for e in members}) < 2):
                continue
            points = sorted(best_per_value.items(), key=lambda item: axis.index(item[0]))
            trends.append(_trend(axis, "marginal", fixed, points, fixed))

    def _rank(trend):
        return (0 if trend["mode"] == "slice" else 1, -abs(trend["slope"]) * trend["r2"], trend["axis"],
                json_key(trend["fixed"]))

    trends.sort(key=_rank)
    return trends


def json_key(mapping: Mapping[str, str]) -> str:
    return "|".join(f"{k}={mapping[k]}" for k in sorted(mapping))


def extrapolate(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig,
        batch: Sequence[Mapping[str, Any]] = ()) -> List[Dict[str, Any]]:
    space = archive.space
    orders = []
    used = set(taken)
    elites_by_value: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for entry in archive.elite_entries():
        for name, value in entry["descriptors"].items():
            elites_by_value[(name, value)].append(entry)
    for trend in find_trends(archive, config, taken=used):
        if len(orders) >= n:
            break
        if trend["status"] != "proposed":
            continue
        key = space.cell_key(trend["target"])
        if key in used:
            continue
        edge_elites = sorted(
            (e for e in elites_by_value[(trend["axis"], trend["edge"])]
             if space.matches(e["descriptors"], trend["fixed"])),
            key=lambda e: (-float(e.get("score") or 0.0), e["id"]),
        )[: config.max_context_elites]
        points = ", ".join(f"{v}: {s:.1f}" for v, s in trend["points"])
        scope = f"with {space.describe(trend['fixed'])} held fixed" if trend["fixed"] else "across all other axes (best per value)"
        orders.append(_make_order(
            space, EXTRAPOLATE, trend["target"],
            context={"trend": trend, "edge_elites": [entry_summary(space, e) for e in edge_elites]},
            rationale=(
                f"Scores {trend['direction']} along {trend['axis']} {scope} ({points}; slope "
                f"{trend['slope']:+.1f} per step, r2={trend['r2']:.2f}). Nothing has been tried beyond "
                f"{trend['edge']}; extrapolate one step to {trend['axis']}={trend['next']}."
            ),
            parent_ids=[e["id"] for e in edge_elites],
        ))
        used.add(key)
    return orders


def combine(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig,
        batch: Sequence[Mapping[str, Any]] = ()) -> List[Dict[str, Any]]:
    space = archive.space
    elites = archive.elite_entries()
    if n <= 0 or len(elites) < 2:
        return []
    combined_before = set()
    for entry in archive.entries.values():
        if entry.get("strategy") == COMBINE and len(entry.get("parent_ids") or []) == 2:
            combined_before.add(tuple(sorted(entry["parent_ids"])))
    pairs = []
    for i, a in enumerate(elites):
        for b in elites[i + 1:]:
            d = space.distance(a["descriptors"], b["descriptors"])
            if d < 2 or tuple(sorted((a["id"], b["id"]))) in combined_before:
                continue
            pairs.append((d, float(a.get("score") or 0) + float(b.get("score") or 0), a, b))
    pairs.sort(key=lambda p: (-p[0], -p[1], p[2]["cell"], p[3]["cell"]))

    orders = []
    used = set(taken)
    stats = ExplorationStats(archive)
    known = PairKnowledge(archive, config.compatibility_axes)
    proven = proven_anchor_pairs(archive, config)
    for distance, _, a, b in pairs:
        if len(orders) >= n:
            break
        diff = space.differing_axes(a["descriptors"], b["descriptors"])
        strong, weak = (a, b) if (rank_key(a), b["id"]) >= (rank_key(b), a["id"]) else (b, a)
        anchored = combine_anchors(diff, strong, weak, proven, config)
        free = [name for name in diff if name not in anchored]
        options = []
        for mask in range(1, 2 ** len(free)):
            child = dict(strong["descriptors"])
            for bit, name in enumerate(free):
                if mask >> bit & 1:
                    child[name] = weak["descriptors"][name]
            if child == a["descriptors"] or child == b["descriptors"] or not archive.is_preferred(child):
                continue
            take = {name: ("A" if child[name] == a["descriptors"][name] else "B") for name in diff}
            options.append((space.cell_key(child), child, take))
        options.sort(key=lambda o: o[0])
        rng.shuffle(options)
        # Stable sort: mixtures with under-explored values first, random among equals.
        options.sort(key=lambda o: -sum(stats.under_exploration(name, o[1][name]) for name in diff))
        choice = None
        for key, child, take in options:
            if key in used or key in archive.elites or archive.is_infeasible(child):
                continue
            compatible, why = combine_compatibility(space, child, a, b, known, config)
            if not compatible:
                continue
            choice = (key, child, take, why)
            break
        if choice is None:
            continue
        key, child, take, why = choice
        used.add(key)
        context = {
            "parent_a": entry_summary(space, a), "parent_b": entry_summary(space, b),
            "descriptor_distance": distance, "take_from": take, "compatibility": why,
        }
        anchor_text = ""
        if anchored:
            context["anchored"] = {name: {"from": "A" if strong is a else "B", "why": reason}
                                   for name, reason in anchored.items()}
            anchor_text = (f" {', '.join(f'{name}={child[name]}' for name in anchored)} come(s) from the stronger "
                           f"parent '{strong.get('title')}' ({_tier_text(strong)}): the mixed concept must work on "
                           "that quantity with a mechanism that can actually affect it.")
        orders.append(_make_order(
            space, COMBINE, child, context=context,
            rationale=(
                f"Cross two distant elites (distance {distance}): '{a.get('title')}' ({a.get('score'):.1f}) and "
                f"'{b.get('title')}' ({b.get('score'):.1f}). Combine the working principle of both in the mixed cell "
                f"{space.describe(child)} ({why})." + anchor_text
            ),
            parent_ids=[a["id"], b["id"]],
        ))
    return orders


def proven_anchor_pairs(archive: Archive, config: StrategyConfig) -> Dict[str, Set[Tuple[str, str]]]:
    """
    For every further anchor axis (materials: mechanism_class): the (value, first-anchor value) pairs
    that produced an evaluated, trend-eligible (simulated or unranked) concept somewhere.
    """
    anchors = [name for name in config.combine_anchor_axes if name in archive.space.names]
    proven: Dict[str, Set[Tuple[str, str]]] = {name: set() for name in anchors[1:]}
    if len(anchors) < 2:
        return proven
    first = anchors[0]
    for entry in archive.entries.values():
        if entry.get("status") != EVALUATED or not entry.get("descriptors") or not trend_eligible(entry):
            continue
        for name in anchors[1:]:
            proven[name].add((entry["descriptors"][name], entry["descriptors"][first]))
    return proven


def combine_anchors(diff: Sequence[str], strong: Mapping[str, Any], weak: Mapping[str, Any],
                    proven: Mapping[str, Set[Tuple[str, str]]], config: StrategyConfig) -> Dict[str, str]:
    """
    Axes the child takes from the stronger parent, with the reason. The first anchor axis
    (materials: governing_quantity) always, so a combine never inherits the quantity of an
    estimated or weaker parent; a further anchor axis (mechanism_class) when the weaker parent's
    value never produced a working concept on the anchored quantity.
    """
    anchors = [name for name in config.combine_anchor_axes if name in strong["descriptors"]]
    if not anchors:
        return {}
    first = anchors[0]
    target_value = strong["descriptors"][first]
    result: Dict[str, str] = {}
    if first in diff:
        result[first] = f"from the stronger parent ({_tier_text(strong)} vs {_tier_text(weak)})"
    for name in anchors[1:]:
        value = weak["descriptors"][name]
        if name in diff and (value, target_value) not in proven.get(name, set()):
            result[name] = f"{name}={value} never produced a working concept with {first}={target_value}"
    return result


def _tier_text(entry: Mapping[str, Any]) -> str:
    tier = (entry.get("score_breakdown") or {}).get("evidence_tier")
    score = float(entry.get("score") or 0.0)
    return f"{tier}, {score:.1f}" if tier else f"{score:.1f}"


def _relevance_filter(archive: Archive) -> Optional[Dict[str, Set[str]]]:
    """Allowed values per relevance axis; None if a relevance axis has no known values yet."""
    allowed: Dict[str, Set[str]] = {}
    for axis in archive.relevance_axes():
        values = archive.relevant_values(axis)
        if not values:
            return None
        allowed[axis] = set(values)
    return allowed


def explore(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig,
        batch: Sequence[Mapping[str, Any]] = ()) -> List[Dict[str, Any]]:
    """
    A random never-proposed, never-targeted cell. On relevance axes only relevant values are
    used (none known yet -> no explore order, e.g. at cold start before the first function
    analysis); on the other axes rare values are preferred (weight = product of
    1 / (1 + proposals with that value)). The weight is also multiplied by
    exp(-explore_distance_decay * (d - 1)), d = distance to the nearest feasible (evaluated)
    entry, so explore stays next to regions where concepts worked (cells beyond
    explore_max_distance are skipped while nearer ones are left), and by
    explore_infeasible_pair_factor if the cell's compatibility pair was only ever infeasible.
    """
    space = archive.space
    if n <= 0:
        return []
    allowed = _relevance_filter(archive)
    if allowed is None:
        return []
    stats = ExplorationStats(archive)
    known = PairKnowledge(archive, config.compatibility_axes)
    feasible = archive.elite_entries()          # one per evaluated cell
    free_axes = [name for name in space.names if name not in allowed]
    unvisited, weights, nearest = [], [], []
    for cell in space.all_cells():
        if any(cell[axis] not in values for axis, values in allowed.items()):
            continue
        key = space.cell_key(cell)
        if key in taken or archive.data["proposals"].get(key) or archive.data["targets"].get(key):
            continue
        if archive.is_infeasible(cell):
            continue
        weight = 1.0
        for name in free_axes:
            weight /= 1.0 + stats.counts[name].get(cell[name], 0)
        near = None
        if feasible:
            near = min(((space.distance(cell, e["descriptors"]), e["id"], e) for e in feasible),
                       key=lambda item: (item[0], item[1]))
            if config.explore_distance_decay > 0:
                weight *= math.exp(-config.explore_distance_decay * max(near[0] - 1, 0))
        if known.only_infeasible(cell):
            weight *= config.explore_infeasible_pair_factor
        if not archive.is_preferred(cell):
            weight *= config.nonpreferred_explore_factor
        unvisited.append(cell)
        weights.append(weight)
        nearest.append(near)
    if feasible and config.explore_max_distance > 0:
        keep = [i for i, near in enumerate(nearest) if near[0] <= config.explore_max_distance]
        if keep:
            unvisited = [unvisited[i] for i in keep]
            weights = [weights[i] for i in keep]
            nearest = [nearest[i] for i in keep]
    total = len(unvisited)
    picks = []
    while unvisited and len(picks) < n:
        idx = _weighted_pick(rng, unvisited, weights)
        picks.append((unvisited.pop(idx), nearest.pop(idx)))
        weights.pop(idx)
    scope = "" if not allowed else " among relevant " + "; ".join(
        f"{axis} in ({', '.join(sorted(values, key=space.axis(axis).index))})" for axis, values in allowed.items())
    orders = []
    for cell, near in picks:
        context: Dict[str, Any] = {"unvisited_cells": total, "relevant": {k: sorted(v) for k, v in allowed.items()}}
        near_text = ""
        if near is not None:
            distance, _, entry = near
            context["nearest_feasible"] = {"id": entry["id"], "title": entry.get("title"), "distance": distance,
                                           "cell": space.describe(entry["descriptors"])}
            near_text = (f" The nearest concept that worked is '{entry.get('title')}' at distance {distance} "
                         f"({', '.join(space.differing_axes(cell, entry['descriptors']))} differ).")
        if not archive.is_preferred(cell):
            outside = [f"{axis}={cell[axis]}" for axis in archive.preferred_axes()
                       if cell[axis] not in archive.preferred_values(axis)]
            context["outside_preferred"] = outside
            near_text += (f" Note: {', '.join(outside)} is outside the values the request prefers; propose a concept "
                          "only if it really fits the request.")
        orders.append(_make_order(
            space, EXPLORE, cell, context=context,
            rationale=(
                f"Unexplored cell{scope} ({total} candidate cells have never been proposed or targeted; rare axis "
                "values and cells near concepts that worked are preferred)." + near_text +
                " Try an honest concept here; if the combination cannot work, say so and why."
            ),
        ))
    return orders


def diversify(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig,
        batch: Sequence[Mapping[str, Any]] = ()) -> List[Dict[str, Any]]:
    """
    Least diverse axis first (highest proposal concentration): its least-proposed allowed
    value, placed one change away from a source elite (for ordinal axes: the elite closest in
    steps; then the best evidence tier and the best rotation-adjusted score, ``source_value``,
    so the top elite does not seed every order). One order per axis and pass.
    """
    space = archive.space
    elites = archive.ranked_elites()
    if n <= 0 or not elites:
        return []
    uses = source_uses(archive, batch)
    stats = ExplorationStats(archive)
    allowed = dict(_relevance_filter(archive) or {})
    for name in archive.preferred_axes():
        preferred = set(archive.preferred_values(name))
        allowed[name] = allowed[name] & preferred if name in allowed else preferred
    axes = sorted(space.names, key=lambda name: (-stats.concentration[name], space.names.index(name)))
    orders: List[Dict[str, Any]] = []
    used = set(taken)
    progress = True
    while progress and len(orders) < n:
        progress = False
        for name in axes:
            if len(orders) >= n:
                break
            axis = space.axis(name)
            values = [v for v in axis.values if name not in allowed or v in allowed[name]]
            rng.shuffle(values)
            values.sort(key=lambda v: stats.counts[name].get(v, 0))
            choice = None
            for value in values:
                near = sorted(
                    (e for e in elites if e["descriptors"][name] != value),
                    key=lambda e: (abs(axis.index(e["descriptors"][name]) - axis.index(value)) if axis.is_ordinal else 0,
                                   -evidence_rank(e), -source_value(e, uses, config), e["id"]),
                )
                for elite in near:
                    target = dict(elite["descriptors"])
                    target[name] = value
                    key = space.cell_key(target)
                    if key in used or key in archive.elites or archive.is_infeasible(target) \
                            or not archive.is_preferred(target):
                        continue
                    choice = (value, elite, target, key)
                    break
                if choice:
                    break
            if choice is None:
                continue
            value, elite, target, key = choice
            used.add(key)
            progress = True
            earlier_uses = uses.get(elite["id"], 0)
            uses[elite["id"]] = earlier_uses + 1
            count = stats.counts[name].get(value, 0)
            share = 100.0 * stats.concentration[name]
            orders.append(_make_order(
                space, DIVERSIFY, target,
                context={"elite": entry_summary(space, elite), "axis": name, "value": value,
                         "proposals_with_value": count, "axis_concentration": round(stats.concentration[name], 4),
                         "source_uses": earlier_uses},
                rationale=(
                    f"The archive is least diverse along {name}: {share:.0f} % of all proposals share one value, and "
                    f"{name}={value} has {count} proposal(s). Keep the other descriptors of elite '{elite.get('title')}' "
                    f"and find a concept that really works with {name}={value} (not a relabelled copy)."
                    + origin_note(config, name, value)
                ),
                parent_ids=[elite["id"]],
            ))
    return orders


def seed(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig,
        batch: Sequence[Mapping[str, Any]] = ()) -> List[Dict[str, Any]]:
    space = archive.space
    return [
        _make_order(
            space, SEED, {}, context={},
            rationale=(
                "Free proposal: choose the concept you consider most promising for the request and assign its "
                "descriptors honestly. This samples where ideas are usually proposed."
            ),
        )
        for _ in range(max(n, 0))
    ]


STRATEGY_FUNCS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    REFINE: refine,
    FILL_GAP: fill_gap,
    EXTRAPOLATE: extrapolate,
    COMBINE: combine,
    DIVERSIFY: diversify,
    EXPLORE: explore,
    SEED: seed,
}


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------

def allocate(weights: Mapping[str, float], n: int, rng) -> Dict[str, int]:
    """Largest-remainder allocation; leftover slots are drawn with the remainders as weights."""
    names = sorted(name for name, weight in weights.items() if weight > 0)
    counts = {name: 0 for name in names}
    if n <= 0 or not names:
        return counts
    total = sum(weights[name] for name in names)
    exact = {name: n * weights[name] / total for name in names}
    for name in names:
        counts[name] = int(math.floor(exact[name]))
    remainders = {name: exact[name] - counts[name] for name in names}
    remaining = n - sum(counts.values())
    while remaining > 0:
        pool = [name for name in names if remainders[name] > 0] or names
        idx = _weighted_pick(rng, pool, [remainders[name] if remainders[name] > 0 else 1.0 for name in pool])
        pick = pool[idx]
        counts[pick] += 1
        remainders[pick] = 0.0
        remaining -= 1
    return counts


def explain_trends(archive: Archive, config: Optional[StrategyConfig] = None,
                   taken: Iterable[str] = ()) -> Dict[str, Any]:
    """
    Why extrapolation did or did not produce an order: every fitted trend with its status and a
    one-line reason, and a summary line (also when no trend could be fitted at all).
    """
    config = config or StrategyConfig()
    trends = find_trends(archive, config, taken=taken)
    reasons = {
        "too_few_points": lambda t: f"{len(t['points'])} point(s), needs {config.min_trend_points}",
        "flat": lambda t: f"|slope| {abs(t['slope']):.2f} < {config.min_slope:g} per step",
        "peaked": lambda t: "the score drops at the explored edge (optimum inside the range)",
        "poor_fit": lambda t: f"r2 {t['r2']:.2f} < {config.min_r2:g}",
        "scale_end": lambda t: f"{t['edge']} is the end of the scale",
        "infeasible": lambda t: f"the next cell ({t['next']}) is reported infeasible",
        "taken": lambda t: "the next cell is already targeted in this batch",
        "not_preferred": lambda t: "the next cell uses a value the request does not prefer",
        "proposed": lambda t: f"usable: {t['edge']} -> {t['next']}",
    }
    rows = []
    for t in trends:
        rows.append({"axis": t["axis"], "mode": t["mode"], "fixed": t["fixed"], "points": len(t["points"]),
                     "slope": t["slope"], "r2": t["r2"], "status": t["status"],
                     "reason": reasons.get(t["status"], lambda _t: t["status"])(t)})
    eligible = [e for e in archive.elite_entries() if e.get("score") is not None and trend_eligible(e)]
    if not trends:
        summary = (f"no trend could be fitted: {len(eligible)} simulated elite(s), and no ordinal axis has elites "
                   "at two or more values (slice or marginal)")
    elif any(r["status"] == "proposed" for r in rows):
        summary = f"{sum(r['status'] == 'proposed' for r in rows)} usable trend(s) of {len(rows)}"
    else:
        counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r["status"]] += 1
        summary = f"no usable trend among {len(rows)}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        best = max(rows, key=lambda r: (r["points"], r["r2"]))
        summary += f" (e.g. {best['axis']} {best['mode']}: {best['reason']})"
    return {"summary": summary, "trends": rows}


def extrapolate_slots(archive: Archive, config: StrategyConfig, weights: Mapping[str, float]) -> Dict[str, Any]:
    """
    Whether extrapolate gets slots in this batch: only if a qualifying slice trend exists (other
    axes held fixed, i.e. the same mechanism, origin and quantity; at least ``min_trend_points``
    simulated points, r^2 >= ``min_r2``, improving at the edge, next cell open). Otherwise its
    weight goes to the other strategies, unless it is the only strategy with a weight (its slots
    then fall back as before).
    """
    weight = float(weights.get(EXTRAPOLATE, 0.0) or 0.0)
    qualifying = [t for t in find_trends(archive, config) if t["mode"] == "slice" and t["status"] == "proposed"]
    others = any(float(w or 0.0) > 0 for name, w in weights.items() if name != EXTRAPOLATE)
    if weight <= 0:
        return {"weight": weight, "qualifying_slice_trends": len(qualifying), "allocated": False,
                "reason": "extrapolate has no weight"}
    if qualifying:
        best = qualifying[0]
        return {"weight": weight, "qualifying_slice_trends": len(qualifying), "allocated": True,
                "reason": f"slice trend along {best['axis']} with {json_key(best['fixed'])} "
                          f"({len(best['points'])} points, r2 {best['r2']:.2f}): {best['edge']} -> {best['next']}"}
    return {"weight": weight, "qualifying_slice_trends": 0, "allocated": not others,
            "reason": "no qualifying slice trend: needs the same values on all other axes, "
                      f">= {config.min_trend_points} simulated points and r2 >= {config.min_r2:g}; "
                      + ("its share went to the other strategies" if others else
                         "extrapolate is the only weighted strategy, its slots fall back")}


def schedule(
    archive: Archive,
    batch_size: int,
    rng,
    *,
    round_no: int,
    weights: Optional[Mapping[str, float]] = None,
    config: Optional[StrategyConfig] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    One batch of orders. Without elites (cold start) only 'seed' and 'explore' can
    work, so the batch is seeds plus the explore share (explore yields nothing while the
    relevant values of a relevance axis are unknown; its slot then becomes a seed). Slots a strategy cannot fill
    (no trend, no gap, ...) fall back in FALLBACK_ORDER; 'seed' always succeeds. If ``diagnostics``
    is given, it receives the planned and produced counts per strategy and why extrapolation
    did or did not fire (``explain_trends``); computing them does not consume ``rng``.
    """
    config = config or StrategyConfig()
    weights = dict(weights or DEFAULT_WEIGHTS)
    if not archive.elites:
        explore_w = weights.get(EXPLORE, 0.0)
        weights = {SEED: max(1.0 - explore_w, 0.5), EXPLORE: explore_w}
    slots = extrapolate_slots(archive, config, weights)
    if not slots["allocated"] and weights.get(EXTRAPOLATE, 0) > 0:
        # No qualifying slice trend: the extrapolate share goes to the other strategies.
        weights[EXTRAPOLATE] = 0.0
    counts = allocate(weights, batch_size, rng)
    if diagnostics is not None:
        diagnostics["planned"] = {name: k for name, k in counts.items() if k}
        diagnostics["extrapolate"] = explain_trends(archive, config)
        diagnostics["extrapolate"]["slots"] = slots

    orders: List[Dict[str, Any]] = []
    taken: Set[str] = set()

    def _run(name: str, k: int) -> int:
        if k <= 0:
            return 0
        produced = STRATEGY_FUNCS[name](archive, rng, k, taken, config, batch=list(orders))[:k]
        for order in produced:
            if order["target"]:
                taken.add(order["target_key"])
        orders.extend(produced)
        return len(produced)

    shortfall = 0
    for name in PROCESSING_ORDER:
        k = counts.get(name, 0)
        shortfall += k - _run(name, k)
    for name in FALLBACK_ORDER:
        if shortfall <= 0:
            break
        shortfall -= _run(name, shortfall)

    for idx, order in enumerate(orders, 1):
        order["order_id"] = f"r{round_no:03d}-{idx:02d}"
    if diagnostics is not None:
        produced: Dict[str, int] = defaultdict(int)
        for order in orders:
            produced[order["strategy"]] += 1
        diagnostics["produced"] = dict(sorted(produced.items()))
    return orders
