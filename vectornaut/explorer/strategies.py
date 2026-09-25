# -*- coding: utf-8 -*-
"""
Search strategies: turn the archive into a batch of targeted search orders.

Every function here is pure (it reads the archive, never writes it) and deterministic
for a given ``random.Random``. An order is a JSON-friendly dict::

    {order_id, strategy, target, target_key, context, rationale, parent_ids}

``target`` is a full cell or a partial assignment (``{}`` for a free "seed" proposal).

Strategies
- refine:      mutate an elite inside its own cell.
- fill_gap:    empty cells at distance 1 from elites, ranked by neighbour score and
               low proposal density.
- extrapolate: along an ordinal axis, fit the score trend over elites (holding the other
               axes fixed, or marginalised) and target one step beyond the explored edge
               in the improving direction. No trend or end of scale -> no order.
- combine:     two elites far apart in descriptor space -> a cell mixing their descriptors.
- explore:     a random cell nobody has proposed or targeted yet (small-probability fallback).
- seed:        no target; samples where the model proposes ideas by itself (cold start).
"""
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from vectornaut.explorer.archive import Archive
from vectornaut.explorer.descriptors import DescriptorSpace

REFINE = "refine"
FILL_GAP = "fill_gap"
EXTRAPOLATE = "extrapolate"
COMBINE = "combine"
EXPLORE = "explore"
SEED = "seed"

STRATEGIES = (REFINE, FILL_GAP, EXTRAPOLATE, COMBINE, EXPLORE, SEED)
DEFAULT_WEIGHTS: Dict[str, float] = {
    REFINE: 0.25,
    FILL_GAP: 0.30,
    EXTRAPOLATE: 0.20,
    COMBINE: 0.15,
    EXPLORE: 0.10,
}
# Strategies claim cells in this order within a batch (so a gap is not also a combine target).
PROCESSING_ORDER = (EXTRAPOLATE, FILL_GAP, COMBINE, REFINE, EXPLORE, SEED)
# Slots a strategy could not use go to these, in this order.
FALLBACK_ORDER = (FILL_GAP, REFINE, EXTRAPOLATE, COMBINE, EXPLORE, SEED)


@dataclass
class StrategyConfig:
    # Minimum |slope| (score points per ordinal step) that counts as a trend.
    min_slope: float = 1.0
    # Priority lost per log(1 + earlier attempts) in a gap cell.
    density_penalty: float = 0.2
    # Bonus per additional elite neighbour of a gap (up to 3).
    neighbour_bonus: float = 0.05
    max_context_elites: int = 3


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
    concept = entry.get("concept") or {}
    summary = concept.get("summary") or concept.get("physical_mechanism") or concept.get("value_proposition") or ""
    breakdown = entry.get("score_breakdown") or {}
    return {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "cell": space.describe(entry.get("descriptors") or {}),
        "score": entry.get("score"),
        "basis": breakdown.get("basis"),
        "summary": str(summary)[:300],
        "main_risk": concept.get("main_risk"),
    }


def _weakest_component(entry: Mapping[str, Any]) -> Optional[str]:
    components = (entry.get("score_breakdown") or {}).get("components") or {}
    numeric = {k: v for k, v in components.items() if isinstance(v, (int, float))}
    if not numeric:
        return None
    return min(sorted(numeric), key=lambda k: numeric[k])


def _attempts(archive: Archive, cell: Mapping[str, str]) -> int:
    return max(archive.proposal_count(cell), archive.target_count(cell))


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

def refine(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig) -> List[Dict[str, Any]]:
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


def gap_candidates(archive: Archive, taken: Iterable[str] = (), config: Optional[StrategyConfig] = None) -> List[Dict[str, Any]]:
    """Empty, feasible cells at distance 1 from elites, best first (also used by the report)."""
    config = config or StrategyConfig()
    space = archive.space
    taken = set(taken)
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
    ranked = []
    for key, items in neighbours.items():
        cell = cells[key]
        if archive.is_infeasible(cell):
            continue
        scores = sorted((float(e.get("score") or 0.0) for e, _ in items), reverse=True)
        attempts = _attempts(archive, cell)
        priority = (
            scores[0] / 100.0
            + config.neighbour_bonus * min(len(items) - 1, 3)
            - config.density_penalty * math.log1p(attempts)
        )
        ranked.append({
            "cell": cell, "key": key, "priority": round(priority, 6), "best_neighbour_score": scores[0],
            "neighbours": sorted(items, key=lambda item: (-float(item[0].get("score") or 0.0), item[0]["id"])),
            "attempts": attempts, "proposals": archive.proposal_count(cell),
        })
    ranked.sort(key=lambda item: (-item["priority"], item["key"]))
    return ranked


def fill_gap(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig) -> List[Dict[str, Any]]:
    space = archive.space
    orders = []
    for gap in gap_candidates(archive, taken, config)[: max(n, 0)]:
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
                "priority": gap["priority"],
            },
            rationale=(
                f"Empty cell next to {len(gap['neighbours'])} elite(s); best neighbour '{best.get('title')}' "
                f"scores {float(best.get('score') or 0):.1f} with {changed}={best['descriptors'][changed]}. "
                f"Only {gap['attempts']} earlier attempt(s) landed or aimed here. "
                f"Find a concept that works with {changed}={gap['cell'][changed]}."
            ),
            parent_ids=[e["id"] for e, _ in top],
        ))
    return orders


def find_trends(archive: Archive, config: Optional[StrategyConfig] = None, taken: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """
    Score trends along every ordinal axis. Returns every fitted trend with a status:
    'proposed' (usable target), 'flat', 'peaked', 'scale_end', 'infeasible' or 'taken'. Slice
    trends (other axes held fixed) come first, then marginal ones; within each, the
    steepest well-fitting trend first.
    """
    config = config or StrategyConfig()
    space = archive.space
    taken = set(taken)
    elites = [e for e in archive.elite_entries() if e.get("score") is not None]
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
        next_idx = edge_idx + step
        if not 0 <= next_idx < len(values):
            record["status"] = "scale_end"
            return record
        record["next"] = values[next_idx]
        target = dict(target_extra)
        target[axis.name] = values[next_idx]
        record["target"] = {name: target[name] for name in space.names if name in target}
        key = space.cell_key(record["target"])
        if key in taken:
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

        best_per_value: Dict[str, float] = {}
        for entry in elites:
            value = entry["descriptors"][axis.name]
            best_per_value[value] = max(best_per_value.get(value, -math.inf), float(entry["score"]))
        if len(best_per_value) >= 2:
            points = sorted(best_per_value.items(), key=lambda item: axis.index(item[0]))
            trends.append(_trend(axis, "marginal", {}, points, {}))

    def _rank(trend):
        return (0 if trend["mode"] == "slice" else 1, -abs(trend["slope"]) * trend["r2"], trend["axis"],
                json_key(trend["fixed"]))

    trends.sort(key=_rank)
    return trends


def json_key(mapping: Mapping[str, str]) -> str:
    return "|".join(f"{k}={mapping[k]}" for k in sorted(mapping))


def extrapolate(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig) -> List[Dict[str, Any]]:
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


def combine(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig) -> List[Dict[str, Any]]:
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
    for distance, _, a, b in pairs:
        if len(orders) >= n:
            break
        diff = space.differing_axes(a["descriptors"], b["descriptors"])
        options = []
        for mask in range(1, 2 ** len(diff) - 1):
            child = dict(a["descriptors"])
            take = {}
            for bit, name in enumerate(diff):
                source = b if mask >> bit & 1 else a
                child[name] = source["descriptors"][name]
                take[name] = "B" if source is b else "A"
            options.append((space.cell_key(child), child, take))
        options.sort(key=lambda o: o[0])
        rng.shuffle(options)
        choice = None
        for key, child, take in options:
            if key in used or key in archive.elites or archive.is_infeasible(child):
                continue
            choice = (key, child, take)
            break
        if choice is None:
            continue
        key, child, take = choice
        used.add(key)
        orders.append(_make_order(
            space, COMBINE, child,
            context={
                "parent_a": entry_summary(space, a), "parent_b": entry_summary(space, b),
                "descriptor_distance": distance, "take_from": take,
            },
            rationale=(
                f"Cross two distant elites (distance {distance}): '{a.get('title')}' ({a.get('score'):.1f}) and "
                f"'{b.get('title')}' ({b.get('score'):.1f}). Combine the working principle of both in the mixed cell "
                f"{space.describe(child)}."
            ),
            parent_ids=[a["id"], b["id"]],
        ))
    return orders


def explore(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig) -> List[Dict[str, Any]]:
    space = archive.space
    if n <= 0:
        return []
    unvisited = []
    for cell in space.all_cells():
        key = space.cell_key(cell)
        if key in taken or archive.data["proposals"].get(key) or archive.data["targets"].get(key):
            continue
        if archive.is_infeasible(cell):
            continue
        unvisited.append(cell)
    picks = rng.sample(unvisited, min(n, len(unvisited)))
    return [
        _make_order(
            space, EXPLORE, cell, context={"unvisited_cells": len(unvisited)},
            rationale=(
                f"Random unexplored cell ({len(unvisited)} of {space.size} cells have never been proposed or targeted). "
                "Try an honest concept here; if the combination cannot work, say so and why."
            ),
        )
        for cell in picks
    ]


def seed(archive: Archive, rng, n: int, taken: Set[str], config: StrategyConfig) -> List[Dict[str, Any]]:
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


def schedule(
    archive: Archive,
    batch_size: int,
    rng,
    *,
    round_no: int,
    weights: Optional[Mapping[str, float]] = None,
    config: Optional[StrategyConfig] = None,
) -> List[Dict[str, Any]]:
    """
    One batch of orders. Without elites (cold start) only 'seed' and 'explore' can
    work, so the batch is seeds plus the explore share. Slots a strategy cannot fill
    (no trend, no gap, ...) fall back in FALLBACK_ORDER; 'seed' always succeeds.
    """
    config = config or StrategyConfig()
    weights = dict(weights or DEFAULT_WEIGHTS)
    if not archive.elites:
        explore_w = weights.get(EXPLORE, 0.0)
        weights = {SEED: max(1.0 - explore_w, 0.5), EXPLORE: explore_w}
    counts = allocate(weights, batch_size, rng)

    orders: List[Dict[str, Any]] = []
    taken: Set[str] = set()

    def _run(name: str, k: int) -> int:
        if k <= 0:
            return 0
        produced = STRATEGY_FUNCS[name](archive, rng, k, taken, config)[:k]
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
    return orders
