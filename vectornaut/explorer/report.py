# -*- coding: utf-8 -*-
"""Markdown map reports and JSON export of an explorer archive."""
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from vectornaut.explorer.archive import (
    EVALUATED, FAILED, IMPROVED, INFEASIBLE, INVALID, NEW_ELITE, NOT_BETTER, REJECTED, Archive,
)
from vectornaut.explorer.strategies import STRATEGIES, find_trends, gap_candidates

STAT_KEYS = ("orders", "candidates", "evaluated", "new_elite", "improved", "not_better", "failed",
             "infeasible", "invalid", "rejected", "missing", "off_target", "elites_now")


def _md(text: Any) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def _fmt_score(value: Optional[float]) -> str:
    return "–" if value is None else f"{value:.1f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_md(cell) for cell in row) + " |")
    return "\n".join(lines)


def strategy_stats(archive: Archive, rounds: Optional[Iterable[int]] = None) -> Dict[str, Dict[str, int]]:
    """Yield per strategy from the round logs: how many orders became new elites, improved a cell, ..."""
    wanted = set(rounds) if rounds is not None else None
    stats = {name: {key: 0 for key in STAT_KEYS} for name in STRATEGIES}
    for record in archive.data["rounds"]:
        if wanted is not None and record.get("round") not in wanted:
            continue
        for order in record.get("orders", []):
            s = stats.setdefault(order["strategy"], {key: 0 for key in STAT_KEYS})
            s["orders"] += 1
            status = order.get("entry_status")
            if order.get("item_status") == "missing":
                s["missing"] += 1
                continue
            s["candidates"] += 1
            if status == EVALUATED:
                s["evaluated"] += 1
            for key, match in ((FAILED, "failed"), (INFEASIBLE, "infeasible"), (INVALID, "invalid"), (REJECTED, "rejected")):
                if status == key:
                    s[match] += 1
            outcome = order.get("outcome")
            if outcome in (NEW_ELITE, IMPROVED, NOT_BETTER):
                s[outcome] += 1
            if order.get("on_target") is False:
                s["off_target"] += 1
    elite_ids = set(archive.elites.values())
    for entry_id in elite_ids:
        entry = archive.entries[entry_id]
        if wanted is None or entry.get("round") in wanted:
            stats.setdefault(entry.get("strategy"), {key: 0 for key in STAT_KEYS})["elites_now"] += 1
    return {name: values for name, values in stats.items() if values["orders"] or values["elites_now"]}


def _stats_table(stats: Mapping[str, Mapping[str, int]]) -> str:
    headers = ["strategy", "orders", "evaluated", "new elite", "improved", "not better", "failed", "infeasible",
               "invalid/rejected", "missing", "off target", "elites now"]
    rows = []
    for name in STRATEGIES:
        if name not in stats:
            continue
        s = stats[name]
        rows.append([name, s["orders"], s["evaluated"], s["new_elite"], s["improved"], s["not_better"], s["failed"],
                     s["infeasible"], s["invalid"] + s["rejected"], s["missing"], s["off_target"], s["elites_now"]])
    return _table(headers, rows) if rows else "_No orders yet._"


def _fmt_num(value: Any, digits: int = 2) -> str:
    return "–" if not isinstance(value, (int, float)) or isinstance(value, bool) else f"{value:.{digits}f}"


def _flags(entry: Mapping[str, Any]) -> str:
    return ", ".join((entry.get("score_breakdown") or {}).get("flags") or []) or "–"


def _has_tiers(entries: Iterable[Mapping[str, Any]]) -> bool:
    return any((e.get("score_breakdown") or {}).get("evidence_tier") for e in entries)


def _elites_table(archive: Archive, limit: Optional[int] = None) -> str:
    """Elites best first (evidence tier, then score)."""
    space = archive.space
    elites = archive.ranked_elites()
    if limit:
        elites = elites[:limit]
    if not elites:
        return "_No elites yet._"
    if _has_tiers(elites):
        rows = []
        for i, e in enumerate(elites, 1):
            b = e.get("score_breakdown") or {}
            rows.append([i, _fmt_score(e.get("score")), b.get("evidence_tier") or "–",
                         _fmt_num(b.get("requirement_coverage")), _fmt_num(b.get("used_gain_pct"), 1),
                         _fmt_num(b.get("simulated_gain_pct"), 1), _fmt_num(b.get("critic_plausible_gain_pct"), 1),
                         _flags(e), e.get("title"), space.describe(e.get("descriptors") or {}), e.get("strategy"),
                         e.get("round"), e["id"]])
        return _table(["#", "score", "tier", "req. coverage", "gain used %", "simulated %", "critic %", "flags",
                       "title", "cell", "strategy", "round", "entry"], rows)
    rows = [[i, _fmt_score(e.get("score")), (e.get("score_breakdown") or {}).get("basis", ""), e.get("title"),
             space.describe(e.get("descriptors") or {}), e.get("strategy"), e.get("round"), e["id"]]
            for i, e in enumerate(elites, 1)]
    return _table(["#", "score", "basis", "title", "cell", "strategy", "round", "entry"], rows)


def _requirements_section(archive: Archive) -> str:
    reqs = archive.requirements
    lines = []
    if reqs:
        round_no = archive.request_analysis.get("requirements_round")
        lines.append(f"Extracted by the generator's function analysis (round {round_no}); fixed for this map.\n")
        lines.extend(f"- **{_md(r['name'])}**: {_md(r.get('criterion'))}" for r in reqs)
    else:
        lines.append("_No requirements extracted yet._")
    for axis in archive.relevance_axes():
        stored = (archive.request_analysis.get("relevant") or {}).get(axis) or []
        known = archive.relevant_values(axis) or []
        extra = [v for v in known if v not in stored]
        lines.append(f"\nRelevant `{axis}` values (explore/fill-gap/diversify use only these): "
                     f"{', '.join(stored) or '–'} (function analysis)"
                     + (f" + {', '.join(extra)} (held by elites)" if extra else ""))
    return "\n".join(lines)


def _requirement_coverage_table(archive: Archive, limit: int = 15) -> str:
    reqs = archive.requirements
    elites = [e for e in archive.ranked_elites() if (e.get("score_breakdown") or {}).get("requirements")][:limit]
    if not reqs or not elites:
        return "_No requirement ratings yet (needs the critic)._"
    rows = []
    for e in elites:
        by_name = {r.get("name"): r for r in (e.get("score_breakdown") or {}).get("requirements") or []}
        cells = []
        for req in reqs:
            rating = by_name.get(req["name"]) or {}
            cells.append(_fmt_num(rating.get("coverage")))
        rows.append([e["id"], e.get("title"), *cells, _fmt_num((e.get("score_breakdown") or {}).get("requirement_coverage"))])
    return _table(["entry", "title", *[r["name"] for r in reqs], "mean"], rows)


def _flag_counts(archive: Archive) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for entry in archive.entries.values():
        if entry.get("status") != EVALUATED:
            continue
        for flag in (entry.get("score_breakdown") or {}).get("flags") or []:
            counts[flag] += 1
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none"


def never_proposed(archive: Archive) -> str:
    """Per axis: values no candidate has landed on (the blind spots of the generator)."""
    lines = []
    for axis in archive.space.axes:
        counts = archive.value_counts(axis.name)
        missing = [v for v in axis.values if not counts.get(v)]
        relevant = archive.relevant_values(axis.name)
        note = ""
        if relevant:
            irrelevant = [v for v in missing if v not in relevant]
            missing = [v for v in missing if v in relevant]
            if irrelevant:
                note = f" (not relevant to the request, not searched: {', '.join(irrelevant)})"
        concentration = archive.axis_concentration(axis.name)
        everything = "every relevant value proposed" if note else "every value proposed"
        lines.append(f"- `{axis.name}` (top value holds {100.0 * concentration:.0f} % of proposals): "
                     f"{', '.join(missing) if missing else everything}{note}")
    return "\n".join(lines)


def _infeasible_marker(archive: Archive, axis_a: str, va: str, axis_b: str, vb: str) -> bool:
    for key in archive.data["infeasible"]:
        pattern = archive.space.parse_key(key)
        if (axis_a in pattern or axis_b in pattern) and pattern.get(axis_a, va) == va and pattern.get(axis_b, vb) == vb:
            return True
    return False


def two_axis_view(archive: Archive, axis_a: str, axis_b: str) -> str:
    space = archive.space
    a, b = space.axis(axis_a), space.axis(axis_b)
    best: Dict[tuple, float] = {}
    proposals: Dict[tuple, int] = defaultdict(int)
    for entry in archive.elite_entries():
        key = (entry["descriptors"][axis_a], entry["descriptors"][axis_b])
        best[key] = max(best.get(key, -math.inf), float(entry.get("score") or 0.0))
    for cell_key, count in archive.data["proposals"].items():
        cell = space.parse_key(cell_key)
        proposals[(cell.get(axis_a), cell.get(axis_b))] += int(count)
    rows = []
    for va in a.values:
        row = [va]
        for vb in b.values:
            score = best.get((va, vb))
            text = f"{_fmt_score(score) if score is not None else '·'} ({proposals.get((va, vb), 0)})"
            if score is None and _infeasible_marker(archive, axis_a, va, axis_b, vb):
                text += " x"
            row.append(text)
        rows.append(row)
    legend = (f"Rows: `{axis_a}`, columns: `{axis_b}`. Each cell: best elite score over all other axes "
              f"(number of proposals that landed there). `·` = no elite yet, `x` = contains cells reported infeasible.")
    return legend + "\n\n" + _table([f"{axis_a} \\ {axis_b}", *b.values], rows)


def density_vs_score(archive: Archive) -> str:
    """Per axis value: where ideas are proposed (proposal share) vs where they work (elite scores)."""
    space = archive.space
    total = sum(int(v) for v in archive.data["proposals"].values()) or 1
    parts = []
    for axis in space.axes:
        props: Dict[str, int] = defaultdict(int)
        for cell_key, count in archive.data["proposals"].items():
            props[space.parse_key(cell_key).get(axis.name)] += int(count)
        scores: Dict[str, List[float]] = defaultdict(list)
        for entry in archive.elite_entries():
            scores[entry["descriptors"][axis.name]].append(float(entry.get("score") or 0.0))
        rows = []
        for value in axis.values:
            vals = scores.get(value, [])
            rows.append([value, props.get(value, 0), f"{100.0 * props.get(value, 0) / total:.0f} %", len(vals),
                         _fmt_score(max(vals) if vals else None), _fmt_score(sum(vals) / len(vals) if vals else None)])
        parts.append(f"**{axis.name}** ({axis.kind})\n\n" + _table(
            ["value", "proposals", "share", "elites", "best", "mean elite"], rows))
    return "\n\n".join(parts)


def _gap_table(archive: Archive, limit: int = 8) -> str:
    space = archive.space
    rows = []
    for gap in gap_candidates(archive)[:limit]:
        best, axis = gap["neighbours"][0]
        rows.append([space.describe(gap["cell"]), f"{gap['priority']:.3f}", _fmt_score(gap["best_neighbour_score"]),
                     len(gap["neighbours"]), gap["proposals"], f"{gap.get('exploration', 0.0):.2f}",
                     f"{axis}: {best['descriptors'][axis]} -> {gap['cell'][axis]}"])
    if not rows:
        return "_No gaps next to elites._"
    return _table(["empty cell", "priority", "best neighbour", "elite neighbours", "proposals", "under-explored",
                   "step from best"], rows)


def _trend_table(trends: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for t in trends:
        points = ", ".join(f"{v}: {s:.1f}" for v, s in t["points"])
        fixed = ", ".join(f"{k}={v}" for k, v in t["fixed"].items()) or "(marginal)"
        rows.append([t["axis"], t["mode"], fixed, points, f"{t['slope']:+.2f}", f"{t['r2']:.2f}",
                     t.get("next") or "–", t["status"]])
    if not rows:
        return ("_No ordinal trends yet (needs simulated elites at two or more values of an ordinal axis; "
                "extrapolation needs at least three)._")
    return _table(["axis", "mode", "held fixed", "points (value: score)", "slope/step", "r2", "next", "status"], rows)


def _used_trends(archive: Archive, rounds: Optional[Iterable[int]] = None) -> str:
    wanted = set(rounds) if rounds is not None else None
    rows = []
    for record in archive.data["rounds"]:
        if wanted is not None and record.get("round") not in wanted:
            continue
        for order in record.get("orders", []):
            trend = order.get("trend")
            if not trend:
                continue
            result = order.get("outcome") or order.get("entry_status") or order.get("item_status")
            rows.append([record.get("round"), order["order_id"], trend["axis"], trend["mode"],
                         f"{trend['edge']} -> {trend['next']}", f"{trend['slope']:+.2f}",
                         result, _fmt_score(order.get("score"))])
    if not rows:
        return "_No extrapolation orders yet._"
    return _table(["round", "order", "axis", "mode", "step", "slope/step", "result", "score"], rows)


def _infeasible_table(archive: Archive) -> str:
    rows = []
    for key in sorted(archive.data["infeasible"]):
        record = archive.data["infeasible"][key]
        rows.append([archive.space.describe(archive.space.parse_key(key)), record.get("reason"), record.get("source"),
                     record.get("round"), record.get("count")])
    return _table(["cell / pattern", "reason", "reported by", "round", "times"], rows) if rows else "_None._"


def _coverage(archive: Archive) -> str:
    space = archive.space
    placed = sum(1 for v in archive.data["proposals"].values() if v)
    tiers: Dict[str, int] = defaultdict(int)
    for entry in archive.elite_entries():
        tier = (entry.get("score_breakdown") or {}).get("evidence_tier")
        if tier:
            tiers[tier] += 1
    statuses = defaultdict(int)
    for entry in archive.entries.values():
        statuses[entry["status"]] += 1
    status_text = ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items())) or "none"
    rng = archive.score_range()
    return "\n".join([
        f"- Cells in the grid: {space.size} ({' x '.join(str(len(a.values)) for a in space.axes)})",
        f"- Cells with an elite: {len(archive.elites)} ({100.0 * len(archive.elites) / space.size:.2f} %)",
        f"- Cells with at least one proposal: {placed}",
        f"- Infeasible cells/patterns: {len(archive.data['infeasible'])}",
        f"- Entries: {len(archive.entries)} ({status_text})",
        f"- Elite score range: {_fmt_score(rng[0]) + ' – ' + _fmt_score(rng[1]) if rng else '–'}",
    ] + ([f"- Elites by evidence tier: {', '.join(f'{k}: {v}' for k, v in sorted(tiers.items()))}",
          f"- Flags on evaluated entries: {_flag_counts(archive)}"] if tiers else []))


def render_map(archive: Archive, profile: Any, query: str) -> str:
    axis_a, axis_b = profile.report_axes
    basis_note = ("Scores are self-estimated unit economics (critic-adjusted where available): a structured "
                  "brainstorming aid, not validation." if profile.name == "business" else
                  "Score = 100 * gate * (0.5 * gain_score + 0.5 * requirement coverage). Tier 'simulated' uses the "
                  "simulated gain (the lower of simulation and critic if they differ by more than 2x); tier "
                  "'estimated' (no usable simulated gain, e.g. flagged implausible_gain) uses the lower of the "
                  "critic's and the candidate's estimate at half weight and is capped at 50. Elites are ranked by "
                  "tier first.")
    parts = [
        f"# Explorer map: {profile.name}",
        f"Query: \"{query}\"  \nArchive: `{archive.path}`  \nRounds so far: {archive.round_counter}",
        f"> {basis_note}",
        "## Coverage\n\n" + _coverage(archive),
        "## Requirements of the request\n\n" + _requirements_section(archive),
        "## Elites (best per cell)\n\n" + _elites_table(archive, limit=25),
    ]
    if archive.requirements:
        parts.append("## Requirement coverage (critic ratings, elites)\n\n" + _requirement_coverage_table(archive))
    parts += [
        f"## Map: {axis_a} x {axis_b}\n\n" + two_axis_view(archive, axis_a, axis_b),
        "## Where ideas are proposed vs where they work\n\n" + density_vs_score(archive),
        "## Values never proposed\n\n" + never_proposed(archive),
        "## Promising under-explored cells (next fill-gap targets)\n\n" + _gap_table(archive),
        "## Strategy yield (all rounds)\n\n" + _stats_table(strategy_stats(archive)),
        "## Ordinal trends (current)\n\n" + _trend_table(find_trends(archive)),
        "## Trends used for extrapolation\n\n" + _used_trends(archive),
        "## Infeasible cells\n\n" + _infeasible_table(archive),
    ]
    return "\n\n".join(parts) + "\n"


def render_round(archive: Archive, profile: Any, record: Mapping[str, Any]) -> str:
    space = archive.space
    rows = []
    for order in record.get("orders", []):
        result = order.get("outcome") or order.get("entry_status") or order.get("item_status")
        target = space.describe(space.parse_key(order.get("target_key") or ""))
        rows.append([order["order_id"], order["strategy"], target, result, _fmt_score(order.get("score")),
                     order.get("evidence_tier") or "–", _fmt_num(order.get("requirement_coverage")),
                     ", ".join(order.get("flags") or []) or "–",
                     order.get("title") or "", {True: "yes", False: "no", None: "–"}[order.get("on_target")],
                     order.get("note") or ""])
    notes = record.get("batch_notes") or {}
    parts = [
        f"# Explorer round {record.get('round')} ({profile.name})",
        f"Query: \"{record.get('query')}\"  \nRun: {record.get('run_id')}, seed {record.get('seed')}, "
        f"batch {len(record.get('orders', []))}",
        "## Orders and outcomes\n\n" + _table(
            ["order", "strategy", "target", "result", "score", "tier", "req. coverage", "flags", "title", "on target",
             "note"], rows),
        "## Strategy yield (this round)\n\n" + _stats_table(strategy_stats(archive, rounds=[record.get("round")])),
        "## Extrapolation orders\n\n" + _used_trends(archive, rounds=[record.get("round")]),
    ]
    if notes.get("function_analysis") or notes.get("mechanism_classes_considered") or notes.get("analogues_considered"):
        parts.append("## Generator notes\n\n" + "\n".join([
            f"- Function analysis: {notes.get('function_analysis') or '–'}",
            f"- Requirements: {'; '.join(r['name'] + ': ' + r.get('criterion', '') for r in notes.get('requirements') or []) or '–'}",
            f"- Relevant values: {'; '.join(k + ': ' + ', '.join(v) for k, v in (notes.get('relevant_values') or {}).items()) or '–'}",
            f"- Mechanism classes: {', '.join(notes.get('mechanism_classes_considered') or []) or '–'}",
            f"- Analogues: {', '.join(notes.get('analogues_considered') or []) or '–'}",
        ]))
    if record.get("unmatched"):
        parts.append("## Ignored candidates\n\n" + "\n".join(f"- {_md(u)}" for u in record["unmatched"]))
    return "\n\n".join(parts) + "\n"


def export(archive: Archive, profile: Any, query: str) -> Dict[str, Any]:
    return {
        "profile": profile.name,
        "query": query,
        "archive_path": archive.path,
        "rounds": archive.round_counter,
        "grid_size": archive.space.size,
        "requirements": archive.requirements,
        "relevant_values": {axis: archive.relevant_values(axis) for axis in archive.relevance_axes()},
        "elites": [
            {**{k: e.get(k) for k in ("id", "title", "descriptors", "cell", "score", "score_breakdown", "strategy",
                                      "round", "parent_ids", "raw_result_path")},
             "evidence_tier": (e.get("score_breakdown") or {}).get("evidence_tier"),
             "flags": list((e.get("score_breakdown") or {}).get("flags") or []),
             "requirement_coverage": (e.get("score_breakdown") or {}).get("requirement_coverage")}
            for e in archive.ranked_elites()
        ],
        "strategy_stats": strategy_stats(archive),
        "trends": find_trends(archive),
        "gaps": [
            {"cell": g["cell"], "priority": g["priority"], "best_neighbour_score": g["best_neighbour_score"],
             "proposals": g["proposals"]}
            for g in gap_candidates(archive)[:20]
        ],
        "archive": archive.data,
    }
