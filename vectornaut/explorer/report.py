# -*- coding: utf-8 -*-
"""Markdown map reports and JSON export of an explorer archive."""
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from vectornaut.explorer.analyst import SEED_DIRECT
from vectornaut.explorer.archive import (
    EVALUATED, FAILED, IMPROVED, IMPROVED_ON_TIEBREAK, INFEASIBLE, INVALID, NEW_ELITE, NOT_BETTER, REJECTED, Archive,
    rank_key,
)
from vectornaut.explorer.strategies import STRATEGIES, StrategyConfig, find_trends, gap_candidates

# Strategies in report order (the analyst's direct-answer seeds first).
REPORT_STRATEGIES = (SEED_DIRECT,) + tuple(STRATEGIES)
# Number of best map finds compared with the direct-answer seeds.
COMPARE_TOP = 3

STAT_KEYS = ("orders", "candidates", "evaluated", "new_elite", "improved", "improved_on_tiebreak", "not_better",
             "failed", "infeasible", "invalid", "rejected", "missing", "off_target", "elites_now")


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
    stats = {name: {key: 0 for key in STAT_KEYS} for name in REPORT_STRATEGIES}
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
            if order.get("tiebreak") == IMPROVED_ON_TIEBREAK:
                s["improved_on_tiebreak"] += 1
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
    for name in REPORT_STRATEGIES:
        if name not in stats:
            continue
        s = stats[name]
        improved = s["improved"] if not s.get("improved_on_tiebreak") else \
            f"{s['improved']} ({s['improved_on_tiebreak']} on tiebreak)"
        rows.append([name, s["orders"], s["evaluated"], s["new_elite"], improved, s["not_better"], s["failed"],
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
        novelty = any("novelty" in ((e.get("score_breakdown") or {}).get("contributions") or {}) for e in elites)
        keys = ("objective", "simulated", "requirements") + (("novelty",) if novelty else ())
        for i, e in enumerate(elites, 1):
            b = e.get("score_breakdown") or {}
            parts = b.get("contributions") or {}
            split = " / ".join(_fmt_num(parts.get(k), 1) for k in keys) if parts else "–"
            tier = b.get("evidence_label") or b.get("evidence_tier") or "–"
            if b.get("evidence_label") and b.get("evidence_tier"):
                tier = f"{b['evidence_label']} ({b['evidence_tier']})"
            rows.append([i, _fmt_score(e.get("score")), tier, split,
                         _fmt_num(b.get("objective_gain_pct"), 1), _fmt_num(b.get("critic_objective_gain_pct"), 1),
                         _fmt_num(b.get("conventional_equivalent_gain_pct"), 1),
                         _fmt_num(b.get("simulated_benefit_used_pct"), 1),
                         _fmt_num(b.get("simulated_gain_pct"), 1), b.get("simulated_quantity") or "–",
                         _fmt_num(b.get("requirement_coverage")), _fmt_num(b.get("novelty_rating")),
                         _hard_text(b), _flags(e), e.get("title"), space.describe(e.get("descriptors") or {}),
                         e.get("strategy"), e.get("round"), e["id"]])
        return _table(["#", "score", "evidence (tier)", "points " + " / ".join(
                           {"objective": "obj", "simulated": "sim", "requirements": "req", "novelty": "nov"}[k]
                           for k in keys), "objective gain used %", "critic obj. %",
                       "conv. equivalent %", "sim. benefit used %",
                       "simulated %", "simulated quantity", "req. coverage", "novelty", "hard checks", "flags", "title",
                       "cell", "strategy", "round", "entry"], rows)
    rows = [[i, _fmt_score(e.get("score")), (e.get("score_breakdown") or {}).get("basis", ""), e.get("title"),
             space.describe(e.get("descriptors") or {}), e.get("strategy"), e.get("round"), e["id"]]
            for i, e in enumerate(elites, 1)]
    return _table(["#", "score", "basis", "title", "cell", "strategy", "round", "entry"], rows)


def _hard_text(breakdown: Mapping[str, Any]) -> str:
    failures = breakdown.get("hard_check_failures") or []
    if failures:
        return "FAILED: " + "; ".join(f"{f.get('name')} ({f.get('reason')})" for f in failures)
    checks = (breakdown.get("critic") or {}).get("hard_checks") or []
    return "passed" if checks else "–"


def _analysis_section(archive: Archive) -> str:
    """The analyst's problem analysis (archive version 6): load breakdown, levers, scope, baseline, seeds."""
    analysis = archive.problem_analysis
    if not analysis:
        return ""
    lines = [f"By the analyst before the first search round (round {archive.request_analysis.get('analysis_round')}); "
             "it fixed the framing of this map.\n"]
    if analysis.get("system_analysis"):
        lines.append(_md(analysis["system_analysis"]) + "\n")
    loads = analysis.get("load_breakdown") or []
    if loads:
        lines.append("**Where the load comes from**\n")
        lines.append(_table(["component", "share %", "rough value", "reasoning"],
                            [[x.get("name"), _fmt_num(x.get("share_pct"), 0), x.get("rough_value") or "–",
                              x.get("reasoning") or "–"] for x in loads]) + "\n")
    levers = analysis.get("levers") or []
    if levers:
        lines.append("**Levers, ranked by expected magnitude**\n")
        lines.append(_table(["#", "lever", "acts on", "expected %", "within literal scope", "extension justified",
                             "note"],
                            [[i, x.get("name"), x.get("acts_on") or "–", _fmt_num(x.get("expected_magnitude_pct"), 0),
                              "yes" if x.get("within_literal_scope", True) else "no",
                              {True: "yes", False: "no", None: "–"}[x.get("extension_justified")],
                              x.get("note") or "–"] for i, x in enumerate(levers, 1)]) + "\n")
    verdict = "justified" if analysis.get("scope_extension_justified") else "not justified"
    lines.append(f"- **Scope extension**: {verdict} ({_md(analysis.get('scope_extension_reason') or '–')})")
    if analysis.get("conventional_baseline"):
        lines.append(f"- **Conventional in-service baseline**: {_md(analysis['conventional_baseline'])}")
    origins = analysis.get("relevant_inspiration_origins") or []
    if origins:
        lines.append(f"- **Relevant origins (information)**: {', '.join(origins)}")
    seeds = analysis.get("direct_concepts") or []
    if seeds:
        by_title = {e.get("title"): e for e in archive.entries.values() if e.get("strategy") == SEED_DIRECT}
        rows = []
        for seed in seeds:
            entry = by_title.get(seed.get("title")) or {}
            rows.append([seed.get("title"), _fmt_num(seed.get("realistic_benefit_pct"), 1),
                         "yes" if seed.get("scope_extension") else "no", seed.get("key_physics") or "–",
                         "; ".join(seed.get("risks") or []) or "–", entry.get("id") or "–",
                         entry.get("status") or "–", _fmt_score(entry.get("score"))])
        lines.append("\n**Direct-answer seeds** (the analyst's best 3, evaluated like every other candidate)\n")
        lines.append(_table(["title", "analyst's realistic benefit %", "scope extension", "key physics", "risks",
                             "entry", "status", "score"], rows))
    return "\n".join(lines)


def _is_scored(entry: Mapping[str, Any]) -> bool:
    b = entry.get("score_breakdown") or {}
    return entry.get("status") == EVALUATED and entry.get("score") is not None and \
        "validator_failed" not in (b.get("flags") or [])


def _candidate_row(entry: Mapping[str, Any]) -> Dict[str, Any]:
    b = entry.get("score_breakdown") or {}
    return {
        "id": entry.get("id"), "title": entry.get("title"), "strategy": entry.get("strategy"),
        "score": entry.get("score"), "objective_gain_pct": b.get("objective_gain_pct"),
        "critic_objective_gain_pct": b.get("critic_objective_gain_pct"),
        "novelty_rating": b.get("novelty_rating"), "requirement_coverage": b.get("requirement_coverage"),
        "must_min_coverage": b.get("must_min_coverage"),
        "hard_check_failed": bool(b.get("hard_check_failures")), "evidence_label": b.get("evidence_label"),
        "scope_extension": bool(b.get("scope_extension")), "flags": list(b.get("flags") or []),
    }


def map_vs_direct(archive: Archive, top: int = COMPARE_TOP) -> Optional[Dict[str, Any]]:
    """
    The direct-answer seeds against the best map finds (evaluated, not excluded by the gate, not seeds;
    best by evidence rank and score): best net critic objective gain, best novelty, best requirement
    coverage (mean and weakest must) and best score of each side, with a verdict per measure.
    None when the archive has no seed entries.
    """
    seeds = [e for e in archive.entries.values() if e.get("strategy") == SEED_DIRECT]
    if not seeds:
        return None
    scored_seeds = sorted((e for e in seeds if _is_scored(e)), key=lambda e: (rank_key(e), e["id"]), reverse=True)
    finds = sorted((e for e in archive.entries.values() if e.get("strategy") != SEED_DIRECT and _is_scored(e)),
                   key=lambda e: (rank_key(e), e["id"]), reverse=True)[:top]

    def best(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return max(values) if values else None

    seed_rows = [_candidate_row(e) for e in scored_seeds]
    find_rows = [_candidate_row(e) for e in finds]
    verdicts = {}
    for key, label in (("objective_gain_pct", "objective gain (critic, net of the conventional equivalent)"),
                       ("novelty_rating", "novelty (critic)"), ("requirement_coverage", "requirement coverage (mean)"),
                       ("must_min_coverage", "weakest must-requirement"), ("score", "score")):
        a, b = best(find_rows, key), best(seed_rows, key)
        if a is None or b is None:
            result = "n/a"
        else:
            result = "map" if a > b + 1e-9 else ("direct" if b > a + 1e-9 else "tie")
        verdicts[key] = {"label": label, "map": a, "direct": b, "better": result}
    overall = verdicts["objective_gain_pct"]["better"]
    return {"seeds": seed_rows, "unscored_seeds": [e["id"] for e in seeds if not _is_scored(e)],
            "best_map_finds": find_rows, "verdicts": verdicts,
            "map_beat_direct_on_objective": overall == "map" if overall != "n/a" else None}


def _comparison_section(archive: Archive) -> str:
    data = map_vs_direct(archive)
    if data is None:
        return ""
    lines = [f"The best {COMPARE_TOP} map finds (evaluated, not excluded by the validator gate, seeds left out; "
             "best by evidence rank and score) against the analyst's direct-answer seeds. The objective gain is the "
             "critic's, net of the conventional equivalent.\n"]

    def rows(items: Sequence[Mapping[str, Any]], side: str) -> List[List[Any]]:
        return [[side, r["id"], r["title"], _fmt_score(r["score"]), _fmt_num(r["objective_gain_pct"], 1),
                 _fmt_num(r["novelty_rating"]), _fmt_num(r["requirement_coverage"]), _fmt_num(r["must_min_coverage"]),
                 "FAILED" if r["hard_check_failed"] else "–", "yes" if r["scope_extension"] else "no",
                 r.get("evidence_label") or "–"] for r in items]

    table_rows = rows(data["seeds"], "direct") + rows(data["best_map_finds"], "map")
    lines.append(_table(["side", "entry", "title", "score", "objective gain %", "novelty", "req. coverage",
                         "min must", "hard check", "scope ext.", "evidence"], table_rows) if table_rows
                 else "_Nothing evaluated yet._")
    if data["unscored_seeds"]:
        lines.append(f"\nSeeds without a usable score (failed, rejected or excluded by the gate): "
                     f"{', '.join(data['unscored_seeds'])}")
    lines.append("")
    for key, v in data["verdicts"].items():
        lines.append(f"- {v['label']}: map best {_fmt_num(v['map'], 2)} vs direct best {_fmt_num(v['direct'], 2)} -> "
                     f"**{v['better']}**")
    beat = data["map_beat_direct_on_objective"]
    lines.append("\n**Verdict**: " + (
        "the map found a concept with a larger critic objective gain than the direct answer." if beat else
        "the direct answer is still ahead (or level) on the critic objective gain." if beat is False else
        "not decidable yet (no critic objective gains on one side)."))
    return "\n".join(lines)


def _scope_extension_section(archive: Archive) -> str:
    """Scored concepts that act outside the request's literal wording (flag scope_extension)."""
    rows = []
    for entry in archive.ranked_elites() + [e for e in archive.entries.values() if e.get("id") not in archive.elites.values()]:
        b = entry.get("score_breakdown") or {}
        if entry.get("status") != EVALUATED or "scope_extension" not in (b.get("flags") or []):
            continue
        critic = b.get("critic") or {}
        legit = critic.get("scope_extension_legitimate")
        rows.append([entry["id"], entry.get("title"), entry.get("strategy"), _fmt_score(entry.get("score")),
                     _fmt_num(b.get("objective_gain_pct"), 1), b.get("scope_extension_reason") or "–",
                     {True: "legitimate", False: "NOT legitimate", None: "–"}[legit],
                     critic.get("scope_extension_note") or "–",
                     "yes" if entry["id"] in archive.elites.values() else "no"])
    if not rows:
        return "_No scope extensions._"
    return ("Concepts that act on a lever outside the request's literal wording (justified by the problem analysis). "
            "They are scored like the others (no penalty); the critic states whether the extension is legitimate.\n\n"
            + _table(["entry", "title", "strategy", "score", "objective gain %", "lever / reason", "critic",
                      "critic note", "elite"], rows))


def _framing_section(archive: Archive, profile: Any) -> str:
    objective, baseline = archive.objective_statement, archive.baseline_statement
    if not objective and not baseline:
        return ""
    round_no = archive.request_analysis.get("framing_round")
    lines = [f"Objective and conventional baseline from the function analysis (round {round_no}); fixed for this map.\n",
             f"- **Objective**: {_md(objective or '–')}", f"- **Baseline**: {_md(baseline or '–')}"]
    target = archive.target_gain_pct
    if target is not None:
        lines.append(f"- **Target gain**: {target:g} % (stated by the function analysis, round "
                     f"{archive.request_analysis.get('target_round')})")
    lines.extend(_scale_lines(archive))
    warnings = profile.framing_warnings(archive) if hasattr(profile, "framing_warnings") else []
    lines.extend(f"- Warning: {_md(w)}" for w in warnings)
    return "\n".join(lines) + "\n\n"


def _scale_lines(archive: Archive) -> List[str]:
    scale = archive.objective_scale
    if not scale:
        return []
    history = ", ".join(f"{h.get('pct'):g} % ({h.get('source')}, round {h.get('round') if h.get('round') is not None else '–'})"
                        for h in scale.get("history") or [] if isinstance(h.get("pct"), (int, float)))
    return [f"- **Objective scale**: {scale.get('pct'):g} % ({_md(scale.get('source'))}: {_md(scale.get('detail'))}); "
            f"a gain of this size scores 0.63 of the objective part. History: {history or '–'}"]


def _requirements_section(archive: Archive) -> str:
    reqs = archive.requirements
    lines = []
    if reqs:
        round_no = archive.request_analysis.get("requirements_round")
        lines.append(f"Extracted by the generator's function analysis (round {round_no}); fixed for this map.\n")
        lines.extend(f"- **{_md(r['name'])}** [{_md(r.get('priority') or 'must')}]: {_md(r.get('criterion'))}"
                     for r in reqs)
    else:
        lines.append("_No requirements extracted yet._")
    for axis in archive.relevance_axes():
        stored = (archive.request_analysis.get("relevant") or {}).get(axis) or []
        known = archive.relevant_values(axis) or []
        extra = [v for v in known if v not in stored]
        lines.append(f"\nRelevant `{axis}` values (explore/fill-gap/diversify use only these): "
                     f"{', '.join(stored) or '–'} (function analysis)"
                     + (f" + {', '.join(extra)} (held by elites)" if extra else ""))
    for axis in archive.soft_relevance_axes():
        stored = (archive.request_analysis.get("relevant") or {}).get(axis) or []
        known = archive.relevant_values(axis) or []
        extra = [v for v in known if v not in stored]
        lines.append(f"\nRelevant `{axis}` values (fill-gap/diversify use only these, explore reaches the others at a "
                     f"low weight): {', '.join(stored) or '– (none named: no restriction)'}"
                     + (" (function analysis)" if stored else "")
                     + (f" + {', '.join(extra)} (held by top elites)" if extra else ""))
    reasons = archive.request_analysis.get("preferred_reason") or {}
    for axis in archive.preferred_axes():
        lines.append(f"\nPreferred `{axis}` values (fill-gap/diversify/combine/extrapolate use only these; explore reaches "
                     f"the others at a low weight): {', '.join(archive.preferred_values(axis))}"
                     + (f" ({_md(reasons.get(axis))})" if reasons.get(axis) else ""))
    return "\n".join(lines)


def _requirement_coverage_table(archive: Archive, limit: int = 15) -> str:
    reqs = archive.requirements
    elites = [e for e in archive.ranked_elites() if (e.get("score_breakdown") or {}).get("requirements")][:limit]
    if not reqs or not elites:
        return "_No requirement ratings yet (needs the critic)._"
    rows = []
    for e in elites:
        b = e.get("score_breakdown") or {}
        by_name = {r.get("name"): r for r in b.get("requirements") or []}
        cells = []
        for req in reqs:
            rating = by_name.get(req["name"]) or {}
            cells.append(_fmt_num(rating.get("coverage")))
        rows.append([e["id"], e.get("title"), *cells, _fmt_num(b.get("requirement_coverage")),
                     _fmt_num(b.get("must_min_coverage")), _fmt_num(b.get("requirement_score")),
                     _fmt_num(b.get("must_factor"))])
    headers = [f"{r['name']} [{r.get('priority') or 'must'}]" for r in reqs]
    return _table(["entry", "title", *headers, "mean", "min must", "req. score", "must factor"], rows)


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
                how = "only explore, at a low weight" if axis.name in archive.soft_relevance_axes() else "not searched"
                note = f" (not relevant to the request, {how}: {', '.join(irrelevant)})"
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


def _strategy_config(profile: Any) -> StrategyConfig:
    """The profile's compatibility axes for the gap list (as the runner configures fill_gap)."""
    return StrategyConfig(compatibility_axes=tuple(getattr(profile, "compatibility_axes", ()) or ()))


def _gap_table(archive: Archive, limit: int = 8, config: Optional[StrategyConfig] = None) -> str:
    space = archive.space
    rows = []
    for gap in gap_candidates(archive, config=config)[:limit]:
        best, axis = gap["neighbours"][0]
        rows.append([space.describe(gap["cell"]), f"{gap['priority']:.3f}", _fmt_score(gap["best_neighbour_score"]),
                     len(gap["neighbours"]), gap["proposals"], f"{gap.get('exploration', 0.0):.2f}",
                     f"{axis}: {best['descriptors'][axis]} -> {gap['cell'][axis]}"])
    if not rows:
        return "_No gaps next to elites._"
    return _table(["empty cell", "priority", "best neighbour", "elite neighbours", "proposals", "under-explored",
                   "step from source (rotation-adjusted)"], rows)


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


def _extrapolation_notes(strategy_notes: Mapping[str, Any]) -> str:
    """Why extrapolation did or did not fire in a round (from schedule diagnostics)."""
    info = strategy_notes.get("extrapolate")
    if not info:
        return ""
    planned = (strategy_notes.get("planned") or {}).get("extrapolate", 0)
    produced = (strategy_notes.get("produced") or {}).get("extrapolate", 0)
    text = (f"\n\nExtrapolation (decided at scheduling time, before this round's candidates were evaluated): "
            f"{produced} of {planned} planned slot(s) used; {_md(info.get('summary'))}.")
    slots = info.get("slots") or {}
    if slots.get("weight"):
        text += (f" Slots: {'allocated' if slots.get('allocated') else 'none allocated'} "
                 f"({_md(slots.get('reason'))}).")
    rows = [[t["axis"], t["mode"], ", ".join(f"{k}={v}" for k, v in (t.get("fixed") or {}).items()) or "(marginal)",
             t["points"], f"{t['slope']:+.2f}", f"{t['r2']:.2f}", t["status"], t["reason"]]
            for t in info.get("trends") or []]
    if rows:
        text += "\n\n" + _table(["axis", "mode", "held fixed", "points", "slope/step", "r2", "status", "why"], rows)
    return text


def _infeasible_table(archive: Archive) -> str:
    rows = []
    for key in sorted(archive.data["infeasible"]):
        record = archive.data["infeasible"][key]
        rows.append([archive.space.describe(archive.space.parse_key(key)), record.get("reason"), record.get("source"),
                     "+".join(record.get("scope") or []) or "exact cell", record.get("round"), record.get("count")])
    return _table(["cell / pattern", "reason", "reported by", "scope", "round", "times"], rows) if rows else "_None._"


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
    if hasattr(profile, "score_note"):
        basis_note = profile.score_note(archive)
    else:
        basis_note = ("Scores are self-estimated unit economics (critic-adjusted where available): a structured "
                      "brainstorming aid, not validation.")
    parts = [
        f"# Explorer map: {profile.name}",
        f"Query: \"{query}\"  \nArchive: `{archive.path}`  \nRounds so far: {archive.round_counter}",
        f"> {basis_note}",
        "## Coverage\n\n" + _coverage(archive),
    ]
    analysis = _analysis_section(archive)
    if analysis:
        parts.append("## Problem analysis (analysis first)\n\n" + analysis)
    parts += [
        "## Requirements of the request\n\n" + _framing_section(archive, profile) + _requirements_section(archive),
        "## Elites (best per cell)\n\n" + _elites_table(archive, limit=25),
    ]
    comparison = _comparison_section(archive)
    if comparison:
        parts.append("## Did the map beat the direct answer?\n\n" + comparison)
    if archive.problem_analysis is not None or any(
            "scope_extension" in ((e.get("score_breakdown") or {}).get("flags") or []) for e in archive.entries.values()):
        parts.append("## Scope extensions\n\n" + _scope_extension_section(archive))
    if archive.requirements:
        parts.append("## Requirement coverage (critic ratings, elites)\n\n" + _requirement_coverage_table(archive))
    parts += [
        f"## Map: {axis_a} x {axis_b}\n\n" + two_axis_view(archive, axis_a, axis_b),
        "## Where ideas are proposed vs where they work\n\n" + density_vs_score(archive),
        "## Values never proposed\n\n" + never_proposed(archive),
        "## Promising under-explored cells (next fill-gap targets)\n\n" + _gap_table(archive, config=_strategy_config(profile)),
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
                     order.get("evidence_tier") or "–", _fmt_num(order.get("objective_gain_pct"), 1),
                     _fmt_num(order.get("requirement_coverage")),
                     ", ".join(order.get("flags") or []) or "–",
                     order.get("title") or "", {True: "yes", False: "no", None: "–"}[order.get("on_target")],
                     "; ".join(x for x in (order.get("tiebreak"), order.get("note")) if x)])
    notes = record.get("batch_notes") or {}
    strategy_notes = record.get("strategy_notes") or {}
    scoring_text = ""
    scale = record.get("objective_scale") or {}
    if scale:
        scoring_text = f"Objective scale after this round: {scale.get('pct'):g} % ({_md(scale.get('source'))})."
        update = record.get("scoring_update") or {}
        if update and update.get("previous_objective_scale_pct") is not None:
            scoring_text += (f" Changed from {update['previous_objective_scale_pct']:g} %: {update.get('rescored_entries', 0)} "
                             f"entries re-scored, {len(update.get('elite_changes') or {})} elite(s) changed; scores in "
                             "the table are after re-scoring.")
    kind = " (analysis first: the analyst's direct-answer seeds)" if record.get("kind") == "analysis" else ""
    parts = [
        f"# Explorer round {record.get('round')} ({profile.name}){kind}",
        f"Query: \"{record.get('query')}\"  \nRun: {record.get('run_id')}, seed {record.get('seed')}, "
        f"batch {len(record.get('orders', []))}" + (f", depth {record['depth']}" if record.get("depth") else ""),
        "## Orders and outcomes\n\n" + _table(
            ["order", "strategy", "target", "result", "score", "tier", "objective gain %", "req. coverage", "flags",
             "title", "on target", "note"], rows) + (f"\n\n{scoring_text}" if scoring_text else ""),
        "## Strategy yield (this round)\n\n" + _stats_table(strategy_stats(archive, rounds=[record.get("round")])),
        "## Extrapolation orders\n\n" + _used_trends(archive, rounds=[record.get("round")])
        + _extrapolation_notes(strategy_notes),
    ]
    if notes.get("function_analysis") or notes.get("mechanism_classes_considered") or notes.get("analogues_considered"):
        parts.append("## Generator notes\n\n" + "\n".join([
            f"- Function analysis: {notes.get('function_analysis') or '–'}",
            f"- Objective: {notes.get('objective_statement') or '–'}",
            f"- Baseline: {notes.get('baseline_statement') or '–'}",
            f"- Requirements: {'; '.join(r['name'] + ': ' + r.get('criterion', '') for r in notes.get('requirements') or []) or '–'}",
            f"- Target gain: {notes['target_gain_pct']:g} %" if isinstance(notes.get("target_gain_pct"), (int, float))
            else "- Target gain: –",
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
        "objective_statement": archive.objective_statement,
        "baseline_statement": archive.baseline_statement,
        "relevant_values": {axis: archive.relevant_values(axis) for axis in archive.all_relevance_axes()},
        "target_gain_pct": archive.target_gain_pct,
        "objective_scale": archive.objective_scale,
        "scoring": archive.scoring,
        "problem_analysis": archive.problem_analysis,
        "map_vs_direct": map_vs_direct(archive),
        "elites": [
            {**{k: e.get(k) for k in ("id", "title", "descriptors", "cell", "score", "score_breakdown", "strategy",
                                      "round", "parent_ids", "raw_result_path")},
             "evidence_tier": (e.get("score_breakdown") or {}).get("evidence_tier"),
             "evidence_label": (e.get("score_breakdown") or {}).get("evidence_label"),
             "flags": list((e.get("score_breakdown") or {}).get("flags") or []),
             "requirement_coverage": (e.get("score_breakdown") or {}).get("requirement_coverage")}
            for e in archive.ranked_elites()
        ],
        "strategy_stats": strategy_stats(archive),
        "trends": find_trends(archive),
        "gaps": [
            {"cell": g["cell"], "priority": g["priority"], "best_neighbour_score": g["best_neighbour_score"],
             "proposals": g["proposals"]}
            for g in gap_candidates(archive, config=_strategy_config(profile))[:20]
        ],
        "archive": archive.data,
    }
