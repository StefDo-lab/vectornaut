# -*- coding: utf-8 -*-
"""
Explorer loop and command line.

    python -m vectornaut.explorer --profile materials --query "..." --rounds 3 --batch 6 --mock
    python -m vectornaut.explorer --profile business  --query "..." --rounds 3 --batch 6 --seed 7

Each round: schedule search orders from the archive -> generate one candidate per order
(one model call, which also extracts the request's requirements and relevant values) ->
evaluate (materials: pipeline per candidate + one critic call; business: unit economics +
one critic call) -> update the archive -> write a round report, the cumulative map report
(map.md) and a JSON export. The archive persists across runs under
VECTORNAUT_DATA_DIR/explorer/<profile>/archive.json (or .../<profile>/<name>/ with --archive).

Live mode needs GEMINI_API_KEY (or an injected client, e.g. vectornaut.llm_replay --explorer).
"""
import argparse
import json
import os
import random
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional

from vectornaut.explorer import report
from vectornaut.explorer.archive import (
    EVALUATED, FAILED, INFEASIBLE, INVALID, REJECTED, Archive, default_archive_path,
)
from vectornaut.explorer.generator import (
    MISSING, OK, REJECTED as ITEM_REJECTED, INVALID as ITEM_INVALID, TARGET_INFEASIBLE, CandidateGenerator,
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
        self.config = strategy_config or StrategyConfig()
        path = archive_path or default_archive_path(profile.name, archive_name)
        self.archive = Archive.load(profile.name, profile.space, path=path, clock=clock)
        known = self.archive.data.get("queries") or []
        if known and self.query not in known and not allow_new_query:
            raise ExplorerError(
                f"The archive {self.archive.path} was built for the query {known[0]!r}. Scores for a different "
                "request are not comparable: use --archive NAME for a separate map, or --allow-new-query to mix."
            )
        self.archive.declare_relevance_axes(getattr(profile, "relevance_axes", ()) or ())
        self.out_dir = os.path.abspath(out_dir or os.path.join(self.archive.directory, "reports"))
        self.generator = CandidateGenerator(profile, client=client, mock=mock)
        self.ctx = EvaluationContext(query=self.query, mock=mock, epochs=epochs, opt_rounds=opt_rounds,
                                     use_critic=use_critic, verbose=verbose, requirements=self.archive.requirements)

    def _update_request_analysis(self, notes: Mapping[str, Any], round_no: int) -> Dict[str, Any]:
        """Stores the first requirement list and the relevant values named by the generator."""
        archive = self.archive
        stored = archive.set_requirements(notes.get("requirements") or [], round_no)
        added: Dict[str, List[str]] = {}
        for axis, values in sorted((notes.get("relevant_values") or {}).items()):
            if axis in archive.relevance_axes():
                new = archive.add_relevant_values(axis, values)
                if new:
                    added[axis] = new
        self.ctx.requirements = archive.requirements
        return {"requirements_stored": stored, "relevant_added": added}

    # ------------------------------------------------------------------
    def run_round(self, batch: int, run_id: str) -> Dict[str, Any]:
        archive = self.archive
        space = self.profile.space
        started = archive.clock()
        round_no = archive.next_round()
        rng = random.Random(f"{self.seed}:{round_no}")
        orders = schedule(archive, batch, rng, round_no=round_no, weights=self.weights, config=self.config)
        for order in orders:
            if order["target"]:
                archive.count_target(order["target"])

        generation = self.generator.generate(self.query, orders, archive)
        analysis_update = self._update_request_analysis(generation.batch_notes, round_no)
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
                "evidence_tier": None, "flags": [], "requirement_coverage": None,
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
                entry = archive.add_entry(**common, descriptors=None, status=INFEASIBLE, reason=reason,
                                          score_breakdown={"basis": "target reported infeasible by the generator"})
                archive.mark_infeasible(order["target"], reason, entry_id=entry["id"], round_no=round_no,
                                        source="generator")
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
                            "requirement_coverage": breakdown.get("requirement_coverage")})
            order_logs.append(log)

        record = {
            "round": round_no, "run_id": run_id, "seed": self.seed, "query": self.query, "mock": self.mock,
            "orders": order_logs, "batch_notes": generation.batch_notes, "unmatched": generation.unmatched,
            "request_analysis_update": analysis_update,
            "started_at": started, "finished_at": archive.clock(),
        }
        archive.log_round(record)
        archive.save()
        return record

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

    def run(self, rounds: int, batch: int) -> Dict[str, Any]:
        if rounds < 1 or batch < 1:
            raise ExplorerError("--rounds and --batch must be >= 1.")
        self.archive.register_query(self.query)
        run_id = f"run-{self.archive.round_counter + 1:03d}"
        records = []
        paths: Dict[str, str] = {}
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
            "reports": paths,
            "strategy_stats": report.strategy_stats(self.archive, rounds=[r["round"] for r in records]),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vectornaut.explorer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, choices=PROFILE_NAMES)
    parser.add_argument("--query", required=True, help="The request the map is built for")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=6, help="Search orders (= candidates) per round")
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
        strategy_config=StrategyConfig(min_slope=args.min_slope),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = runner_from_args(args).run(args.rounds, args.batch)
    except ExplorerError as err:
        print(f"[explorer] {err}", file=sys.stderr)
        return 2
    print(json.dumps({k: summary[k] for k in ("status", "profile", "run_id", "rounds", "archive", "elites",
                                               "entries", "reports")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
