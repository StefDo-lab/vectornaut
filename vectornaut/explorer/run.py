# -*- coding: utf-8 -*-
"""
Explorer loop and command line.

    python -m vectornaut.explorer --profile materials --query "..." --rounds 3 --mock
    python -m vectornaut.explorer --profile business  --query "..." --rounds 3 --batch 6 --seed 7

Analysis first (materials, archive version 6; --no-analyst to skip): when a map is created, one
analyst call analyses the request at system level (load breakdown, levers ranked by magnitude, scope
decision, conventional in-service baseline, objective, target, requirements, relevant values) and gives
its own best 3 concepts; they are stored and evaluated as `seed_direct` entries (analysis round 1)
before any search round. The generator then runs in --depth deep (3 deep candidates per round) unless
--depth broad is given; the evaluator is a feasibility/consistency filter (--scoring filter, default).

Each round: schedule search orders from the archive -> generate one candidate per order
(one model call, which also extracts the request's requirements and relevant values) ->
evaluate (materials: pipeline per candidate + one critic call; business: unit economics +
one critic call) -> update the archive -> write a round report, the cumulative map report
(map.md) and a JSON export. The archive persists across runs under
VECTORNAUT_DATA_DIR/explorer/<profile>/archive.json (or .../<profile>/<name>/ with --archive).

Live mode needs GEMINI_API_KEY (or an injected client, e.g. vectornaut.llm_replay --explorer).
"""
import argparse
import dataclasses
import json
import os
import random
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional

from vectornaut.explorer import report
from vectornaut.explorer.analyst import (
    MAX_DIRECT_CONCEPTS, ProblemAnalyst, analysis_brief, analysis_notes, analysis_to_dict, complete_direct_concept,
    seed_orders,
)
from vectornaut.explorer.archive import (
    EVALUATED, FAILED, INFEASIBLE, INVALID, REJECTED, Archive, ArchiveCompatibilityError, default_archive_path,
)
from vectornaut.explorer.generator import (
    DEFAULT_BATCH, DEPTH_BROAD, DEPTH_DEEP, DEPTHS, MISSING, OK, REJECTED as ITEM_REJECTED, INVALID as ITEM_INVALID,
    TARGET_INFEASIBLE, CandidateGenerator, GenerationResult, check_batch,
)
from vectornaut.explorer.profiles import PROFILE_NAMES, get_profile
from vectornaut.explorer.profiles.base import EvaluationContext, ExplorerProfile, PreparedCandidate
from vectornaut.explorer.strategies import StrategyConfig, parse_weights, schedule


class ExplorerError(RuntimeError):
    """Configuration errors reported to the user without a traceback."""


def _live_credentials_available() -> bool:
    if os.environ.get("GEMINI_API_KEY"):
        return True
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true" and bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))


class ExplorerRunner:
    def __init__(
        self,
        profile: ExplorerProfile,
        query: str,
        *,
        seed: int = 0,
        weights: Optional[Mapping[str, float]] = None,
        mock: bool = False,
        client: Any = None,
        archive_path: Optional[str] = None,
        archive_name: Optional[str] = None,
        out_dir: Optional[str] = None,
        epochs: int = 40,
        opt_rounds: int = 1,
        use_critic: bool = True,
        allow_new_query: bool = False,
        verbose: bool = False,
        clock: Optional[Callable[[], str]] = None,
        strategy_config: Optional[StrategyConfig] = None,
        analyst: Optional[bool] = None,
        depth: Optional[str] = None,
        scoring: Optional[str] = None,
        sim_weight: Optional[float] = None,
    ):
        if not (query or "").strip():
            raise ExplorerError("--query must not be empty.")
        if not mock and client is None and not _live_credentials_available():
            raise ExplorerError(
                "Live mode needs GEMINI_API_KEY (or Vertex AI settings). Use --mock for an offline run, or "
                "'python -m vectornaut.llm_replay --explorer ...' to answer the model calls by hand."
            )
        self.profile = profile
        self.query = query.strip()
        self.seed = seed
        self.weights = dict(weights) if weights else None
        self.mock = mock
        self.client = client
        config = strategy_config or StrategyConfig()
        # Profile knowledge the (profile-agnostic) strategies need, unless the caller set it.
        self.config = dataclasses.replace(
            config,
            compatibility_axes=tuple(config.compatibility_axes or getattr(profile, "compatibility_axes", ()) or ()),
            origin_axes=tuple(config.origin_axes or getattr(profile, "origin_axes", ()) or ()),
            combine_anchor_axes=tuple(config.combine_anchor_axes or getattr(profile, "combine_anchor_axes", ()) or ()),
            trend_group_axes=tuple(config.trend_group_axes or getattr(profile, "trend_group_axes", ()) or ()),
        )
        path = archive_path or default_archive_path(profile.name, archive_name)
        try:
            self.archive = Archive.load(profile.name, profile.space, path=path, clock=clock,
                                        migrate=profile.migrate_archive)
        except ArchiveCompatibilityError as err:
            raise ExplorerError(str(err))
        known = self.archive.data.get("queries") or []
        if known and self.query not in known and not allow_new_query:
            raise ExplorerError(
                f"The archive {self.archive.path} was built for the query {known[0]!r}. Scores for a different "
                "request are not comparable: use --archive NAME for a separate map, or --allow-new-query to mix."
            )
        self.archive.declare_relevance_axes(getattr(profile, "relevance_axes", ()) or (),
                                            soft=getattr(profile, "soft_relevance_axes", ()) or ())
        self._update_preferences()
        # Analysis-first stage (archive version 6): on by default where the profile has an analyst.
        self.analyst = ProblemAnalyst(profile, client=client, mock=mock)
        if analyst and not self.analyst.supported:
            raise ExplorerError(f"The {profile.name} profile has no analyst stage; run it without --analyst.")
        self.use_analyst = self.analyst.supported if analyst is None else bool(analyst)
        if depth is not None and depth not in DEPTHS:
            raise ExplorerError(f"--depth must be one of {', '.join(DEPTHS)}.")
        self.depth_requested = depth
        # Scoring configuration (materials: filter / legacy, simulated-benefit weight), stored per archive.
        forced = self._configure_scoring(scoring, sim_weight)
        # Archive-level scoring (materials: the objective scale) is brought up to date on start; a changed
        # scoring configuration re-scores every entry.
        self.startup_scoring = profile.update_scoring(self.archive, None, force=forced)
        self.out_dir = os.path.abspath(out_dir or os.path.join(self.archive.directory, "reports"))
        self.generator = CandidateGenerator(profile, client=client, mock=mock, depth=self.depth)
        self.ctx = EvaluationContext(query=self.query, mock=mock, epochs=epochs, opt_rounds=opt_rounds,
                                     use_critic=use_critic, verbose=verbose)
        self._sync_context()

    @property
    def depth(self) -> str:
        """Generator depth: as requested, else deep when the map has (or is about to get) a problem analysis."""
        if self.depth_requested:
            return self.depth_requested
        analysed = self.archive.problem_analysis is not None
        pending = self.use_analyst and self.archive.round_counter == 0
        return DEPTH_DEEP if analysed or pending else DEPTH_BROAD

    def default_batch(self) -> int:
        return DEFAULT_BATCH[self.depth]

    def _configure_scoring(self, mode: Optional[str], sim_weight: Optional[float]) -> bool:
        """
        Stores the scoring configuration on the archive: the profile's default for a new or unconfigured
        archive, changed by an explicit mode or simulated-benefit weight. Returns True if an existing
        configuration changed (then every entry is re-scored).
        """
        default = self.profile.default_scoring()
        if not default:
            if mode not in (None, "") or sim_weight is not None:
                raise ExplorerError(f"The {self.profile.name} profile has no --scoring / --sim-weight options.")
            return False
        current = self.archive.scoring
        wanted = dict(current or default)
        if mode:
            wanted["mode"] = mode
        if sim_weight is not None:
            wanted["sim_weight"] = sim_weight
        try:
            wanted = self.profile.normalize_scoring(wanted)
        except ValueError as err:
            raise ExplorerError(str(err))
        changed = self.archive.set_scoring(wanted)
        return bool(changed and current)

    def _sync_context(self) -> None:
        """Copies the fixed request analysis (requirements, objective, baseline, relevant values) into the context."""
        archive = self.archive
        self.ctx.requirements = archive.requirements
        self.ctx.objective_statement = archive.objective_statement
        self.ctx.baseline_statement = archive.baseline_statement
        self.ctx.relevant = {axis: archive.stated_relevant_values(axis) for axis in archive.all_relevance_axes()}
        scale = archive.objective_scale
        self.ctx.objective_scale_pct = scale.get("pct")
        self.ctx.objective_scale_source = scale.get("source") or "default"
        self.ctx.scoring = archive.scoring
        self.ctx.analysis_text = analysis_brief(archive.problem_analysis) if archive.problem_analysis else ""

    def _update_preferences(self) -> Dict[str, List[str]]:
        """Preferred axis values derived by the profile from the request and the stored requirements."""
        changed = {}
        for axis, (values, reason) in sorted(self.profile.preferred_values(self.query, self.archive.requirements).items()):
            if self.archive.set_preferred_values(axis, values, reason):
                changed[axis] = self.archive.preferred_values(axis)
        return changed

    def _update_request_analysis(self, notes: Mapping[str, Any], round_no: int) -> Dict[str, Any]:
        """
        Stores the first requirement list, objective, baseline and target gain, and the relevant values
        named by the generator. A newly stored target gain changes the objective scale at once (so this
        round is already scored with it).
        """
        archive = self.archive
        stored = archive.set_requirements(notes.get("requirements") or [], round_no)
        framing = archive.set_framing(notes.get("objective_statement") or "", notes.get("baseline_statement") or "",
                                      round_no)
        target = archive.set_target_gain(notes.get("target_gain_pct"), round_no)
        added: Dict[str, List[str]] = {}
        for axis, values in sorted((notes.get("relevant_values") or {}).items()):
            if axis in archive.all_relevance_axes():
                new = archive.add_relevant_values(axis, values)
                if new:
                    added[axis] = new
        preferred = self._update_preferences()
        scoring = self.profile.update_scoring(archive, round_no) if target else None
        self._sync_context()
        return {"requirements_stored": stored, "objective_stored": framing["objective"],
                "baseline_stored": framing["baseline"], "target_stored": target, "relevant_added": added,
                "preferred_changed": preferred, "scoring_update": scoring}

    # ------------------------------------------------------------------
    def run_analysis(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Analysis-first stage: once, when the map is created (no round yet, no stored analysis) and the
        analyst is on. One analyst call; the analysis is stored (framing, requirements, relevant values,
        scope decision) and its direct concepts are checked and evaluated as ``seed_direct`` entries in
        round 1 (like every other candidate: pipeline + critic). Returns the round record or None.
        """
        archive = self.archive
        if not self.use_analyst or archive.problem_analysis is not None or archive.round_counter > 0:
            return None
        started = archive.clock()
        analysis, prompt = self.analyst.analyse(self.query, archive)
        round_no = archive.next_round()
        archive.set_problem_analysis(analysis_to_dict(analysis), round_no)
        relevant, dropped = self.profile.relevant_values_from_analysis(analysis)
        notes = analysis_notes(analysis, relevant, dropped)
        analysis_update = self._update_request_analysis(notes, round_no)
        concepts = [complete_direct_concept(c) for c in list(getattr(analysis, "direct_concepts", None) or [])]
        ignored = [f"direct concept {i} ignored (only the best {MAX_DIRECT_CONCEPTS} seed the map): "
                   f"{getattr(c, 'title', '')}" for i, c in enumerate(concepts[MAX_DIRECT_CONCEPTS:],
                                                                      MAX_DIRECT_CONCEPTS + 1)]
        concepts = concepts[:MAX_DIRECT_CONCEPTS]
        orders = seed_orders(self.profile.space, len(concepts), round_no)
        for order, concept in zip(orders, concepts):
            concept.order_id = order["order_id"]
        batch = type("DirectAnswer", (), {"candidates": concepts})()
        generation = check_batch(self.profile, orders, batch, archive, prompt=prompt)
        generation.batch_notes = notes
        generation.unmatched.extend(ignored)
        order_logs = self._evaluate_generation(generation, round_no, run_id)
        scoring = self._rescore_after_round(order_logs, round_no)
        self.generator.depth = self.depth
        record = {
            "round": round_no, "kind": "analysis", "run_id": run_id, "seed": self.seed, "query": self.query,
            "mock": self.mock, "orders": order_logs, "batch_notes": notes, "unmatched": generation.unmatched,
            "request_analysis_update": analysis_update, "strategy_notes": {}, "scoring_update": scoring,
            "objective_scale": archive.objective_scale, "started_at": started, "finished_at": archive.clock(),
        }
        archive.log_round(record)
        archive.save()
        return record

    def run_round(self, batch: int, run_id: str) -> Dict[str, Any]:
        archive = self.archive
        started = archive.clock()
        round_no = archive.next_round()
        rng = random.Random(f"{self.seed}:{round_no}")
        diagnostics: Dict[str, Any] = {}
        orders = schedule(archive, batch, rng, round_no=round_no, weights=self.weights, config=self.config,
                          diagnostics=diagnostics)
        for order in orders:
            if order["target"]:
                archive.count_target(order["target"])

        self.generator.depth = self.depth
        generation = self.generator.generate(self.query, orders, archive)
        analysis_update = self._update_request_analysis(generation.batch_notes, round_no)
        order_logs = self._evaluate_generation(generation, round_no, run_id)
        scoring = self._rescore_after_round(order_logs, round_no)

        record = {
            "round": round_no, "run_id": run_id, "seed": self.seed, "query": self.query, "mock": self.mock,
            "depth": self.generator.depth,
            "orders": order_logs, "batch_notes": generation.batch_notes, "unmatched": generation.unmatched,
            "request_analysis_update": analysis_update, "strategy_notes": diagnostics,
            "scoring_update": scoring, "objective_scale": archive.objective_scale,
            "started_at": started, "finished_at": archive.clock(),
        }
        archive.log_round(record)
        archive.save()
        return record

    def _rescore_after_round(self, order_logs: List[Dict[str, Any]], round_no: int) -> Optional[Dict[str, Any]]:
        """The objective scale follows the archive (materials): re-score everything if it changed."""
        archive = self.archive
        scoring = self.profile.update_scoring(archive, round_no)
        if scoring:
            for log in order_logs:
                entry = archive.entries.get(log.get("entry_id") or "")
                if entry is not None and entry.get("status") == EVALUATED:
                    breakdown = entry.get("score_breakdown") or {}
                    log.update({"score": entry["score"], "flags": list(breakdown.get("flags") or []),
                                "objective_gain_pct": breakdown.get("objective_gain_pct"),
                                "tiebreak": entry.get("tiebreak")})
            self._sync_context()
        return scoring

    def _evaluate_generation(self, generation: GenerationResult, round_no: int, run_id: str) -> List[Dict[str, Any]]:
        """Evaluates the checked candidates of one batch and stores every item in the archive; returns the order logs."""
        archive = self.archive
        space = self.profile.space
        prepared: List[PreparedCandidate] = []
        for item in generation.items:
            if item.status == OK:
                prepared.append(PreparedCandidate(order=item.order, candidate=item.candidate,
                                                  descriptors=item.descriptors,
                                                  concept=self.profile.concept_payload(item.candidate)))
        results = self.profile.evaluate(prepared, self.ctx) if prepared else []
        result_by_order = {p.order["order_id"]: r for p, r in zip(prepared, results)}

        order_logs = []
        for item in generation.items:
            order = item.order
            log: Dict[str, Any] = {
                "order_id": order["order_id"], "strategy": order["strategy"], "target_key": order["target_key"],
                "rationale": order["rationale"], "parent_ids": order["parent_ids"],
                "trend": (order.get("context") or {}).get("trend"), "item_status": item.status,
                "entry_id": None, "entry_status": None, "outcome": None, "on_target": None, "score": None,
                "evidence_tier": None, "flags": [], "requirement_coverage": None, "objective_gain_pct": None,
                "tiebreak": None,
                "title": getattr(item.candidate, "title", None) if item.candidate is not None else None,
                "note": "; ".join(item.issues)[:300],
            }
            entry = None
            common = dict(round_no=round_no, run_id=run_id, order=order,
                          title=(getattr(item.candidate, "title", "") or "") if item.candidate is not None else "",
                          concept=self.profile.concept_payload(item.candidate) if item.candidate is not None else {})
            if item.status == MISSING:
                pass
            elif item.status == TARGET_INFEASIBLE:
                reason = item.issues[0] if item.issues else "reported infeasible"
                # The generator may say which axes the impossibility depends on: the pattern is closed.
                scope = list(getattr(item.candidate, "infeasibility_scope", None) or [])
                pattern, used, scope_note = archive.infeasibility_pattern(order["target"], scope)
                entry = archive.add_entry(**common, descriptors=None, status=INFEASIBLE, reason=reason,
                                          score_breakdown={"basis": "target reported infeasible by the generator",
                                                           "infeasibility_scope": used,
                                                           "infeasible_pattern": space.cell_key(pattern)})
                archive.mark_infeasible(pattern, reason, entry_id=entry["id"], round_no=round_no,
                                        source="generator", scope=used, note=scope_note)
                if scope_note:
                    log["note"] = f"{log['note']}; {scope_note}".strip("; ")[:300]
            elif item.status == ITEM_INVALID:
                entry = archive.add_entry(**common, descriptors=None, status=INVALID, reason="; ".join(item.issues))
            elif item.status == ITEM_REJECTED:
                entry = archive.add_entry(**common, descriptors=item.descriptors, status=REJECTED,
                                          reason="; ".join(item.issues))
            else:
                result = result_by_order[order["order_id"]]
                entry = archive.add_entry(**common, descriptors=item.descriptors, status=result.status,
                                          score=result.score, score_breakdown=result.breakdown,
                                          reason=result.reason, raw_result=result.raw)
                if result.status == INFEASIBLE:
                    archive.mark_infeasible(item.descriptors, result.reason, entry_id=entry["id"],
                                            round_no=round_no, source="evaluator")
                if result.reason:
                    log["note"] = result.reason[:300]
            if entry is not None:
                breakdown = entry.get("score_breakdown") or {}
                log.update({"entry_id": entry["id"], "entry_status": entry["status"], "outcome": entry["outcome"],
                            "on_target": entry["on_target"], "score": entry["score"],
                            "evidence_tier": breakdown.get("evidence_tier"), "flags": list(breakdown.get("flags") or []),
                            "requirement_coverage": breakdown.get("requirement_coverage"),
                            "objective_gain_pct": breakdown.get("objective_gain_pct"),
                            "tiebreak": entry.get("tiebreak")})
            order_logs.append(log)
        return order_logs

    def write_reports(self, record: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
        os.makedirs(self.out_dir, exist_ok=True)
        paths = {}
        if record is not None:
            path = os.path.join(self.out_dir, f"round_{int(record['round']):03d}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(report.render_round(self.archive, self.profile, record))
            paths["round"] = path
        map_path = os.path.join(self.out_dir, "map.md")
        with open(map_path, "w", encoding="utf-8") as f:
            f.write(report.render_map(self.archive, self.profile, self.query))
        export_path = os.path.join(self.out_dir, "archive_export.json")
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(report.export(self.archive, self.profile, self.query), f, indent=2, sort_keys=True,
                      ensure_ascii=False, default=str)
        paths.update({"map": map_path, "export": export_path})
        return paths

    def run(self, rounds: int, batch: Optional[int] = None) -> Dict[str, Any]:
        """
        ``rounds`` search rounds with ``batch`` orders each (default: 3 in deep mode, 6 in broad mode).
        A new map starts with the analysis round (analyst on), which does not count towards ``rounds``.
        """
        if batch is None:
            batch = self.default_batch()
        if rounds < 1 or batch < 1:
            raise ExplorerError("--rounds and --batch must be >= 1.")
        self.archive.register_query(self.query)
        run_id = f"run-{self.archive.round_counter + 1:03d}"
        records = []
        paths: Dict[str, str] = {}
        analysis_record = self.run_analysis(run_id)
        if analysis_record is not None:
            records.append(analysis_record)
            paths = self.write_reports(analysis_record)
            print(f"[explorer] round {analysis_record['round']} (analysis): " + ", ".join(
                f"{o['order_id']} seed_direct -> {o['outcome'] or o['entry_status'] or o['item_status']}"
                for o in analysis_record["orders"]))
        for _ in range(rounds):
            record = self.run_round(batch, run_id)
            records.append(record)
            paths = self.write_reports(record)
            print(f"[explorer] round {record['round']}: " + ", ".join(
                f"{o['order_id']} {o['strategy']} -> {o['outcome'] or o['entry_status'] or o['item_status']}"
                for o in record["orders"]))
        return {
            "status": "complete",
            "profile": self.profile.name,
            "run_id": run_id,
            "rounds": [r["round"] for r in records],
            "archive": self.archive.path,
            "elites": len(self.archive.elites),
            "entries": len(self.archive.entries),
            "analysis": self.archive.problem_analysis is not None,
            "depth": self.depth,
            "batch": batch,
            "reports": paths,
            "strategy_stats": report.strategy_stats(self.archive, rounds=[r["round"] for r in records]),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vectornaut.explorer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, choices=PROFILE_NAMES)
    parser.add_argument("--query", required=True, help="The request the map is built for")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=None,
                        help="Search orders (= candidates) per round (default: 3 with --depth deep, 6 with broad)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy-weights", default=None,
                        help="e.g. refine=0.2,fill_gap=0.25,extrapolate=0.15,combine=0.15,diversify=0.15,explore=0.1")
    parser.add_argument("--mock", action="store_true", help="Deterministic fake generator and mock evaluation, offline")
    parser.add_argument("--out", default=None, help="Report folder (default: <archive folder>/reports)")
    parser.add_argument("--archive", default=None, help="Separate archive name under explorer/<profile>/<name>/")
    parser.add_argument("--allow-new-query", action="store_true", help="Add a different query to an existing archive")
    parser.add_argument("--epochs", type=int, default=40, help="PINN epochs per materials evaluation")
    parser.add_argument("--opt-rounds", type=int, default=1, help="Pipeline optimization rounds per materials candidate")
    parser.add_argument("--no-critic", action="store_true",
                        help="Skip the critic call (materials: no plausible-gain check, requirements unrated)")
    parser.add_argument("--min-slope", type=float, default=StrategyConfig.min_slope,
                        help="Smallest score change per ordinal step that counts as a trend")
    parser.add_argument("--verbose", action="store_true", help="Show pipeline output")
    analyst = parser.add_mutually_exclusive_group()
    analyst.add_argument("--analyst", dest="analyst", action="store_const", const=True, default=None,
                         help="Analysis-first stage for a new map (default where the profile has one: materials)")
    analyst.add_argument("--no-analyst", dest="analyst", action="store_const", const=False,
                         help="Skip the analysis-first stage (the generator frames the request, as before version 6)")
    parser.add_argument("--depth", choices=DEPTHS, default=None,
                        help="deep: few candidates with a quantitative estimate and a self-critique each (default "
                             "when the map has a problem analysis); broad: more, shorter candidates")
    parser.add_argument("--scoring", choices=("filter", "legacy"), default=None,
                        help="materials: 'filter' (default: the simulation is a feasibility/consistency gate) or "
                             "'legacy' (version-5 score); changing it re-scores the archive")
    parser.add_argument("--sim-weight", type=float, default=None,
                        help="materials, filter scoring: weight of the critic-capped simulated benefit (default 0; "
                             "the weights are renormalised)")
    return parser


def runner_from_args(args: argparse.Namespace, client: Any = None) -> ExplorerRunner:
    try:
        weights = parse_weights(args.strategy_weights)
    except ValueError as err:
        raise ExplorerError(str(err))
    return ExplorerRunner(
        get_profile(args.profile), args.query, seed=args.seed, weights=weights, mock=args.mock, client=client,
        archive_name=args.archive, out_dir=args.out, epochs=args.epochs, opt_rounds=args.opt_rounds,
        use_critic=not args.no_critic, allow_new_query=args.allow_new_query, verbose=args.verbose,
        strategy_config=StrategyConfig(min_slope=args.min_slope), analyst=args.analyst, depth=args.depth,
        scoring=args.scoring, sim_weight=args.sim_weight,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = runner_from_args(args).run(args.rounds, args.batch)
    except ExplorerError as err:
        print(f"[explorer] {err}", file=sys.stderr)
        return 2
    print(json.dumps({k: summary[k] for k in ("status", "profile", "run_id", "rounds", "archive", "elites",
                                               "entries", "analysis", "depth", "batch", "reports")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
