# -*- coding: utf-8 -*-
"""Explorer end to end: mock loops for both profiles, replay pause/resume, pipeline concept hook."""
import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from vectornaut.explorer.archive import (
    ARCHIVE_VERSION, EVALUATED, FAILED, IMPROVED, IMPROVED_ON_TIEBREAK, INFEASIBLE, TIE_EPSILON, Archive,
)
from vectornaut.explorer.generator import CandidateGenerator, build_prompt
from vectornaut.explorer.profiles.base import EvaluationContext, PreparedCandidate
from vectornaut.explorer.profiles.business import BusinessProfile
from vectornaut.explorer.profiles.materials import (
    MATERIALS_SPACE, OBJECTIVE_SCALE_PCT, SIM_SCALE_PCT, W_OBJ, W_REQ, W_SIM, MaterialsProfile, ScoringConfig,
    score_pipeline_result,
)
from vectornaut.explorer.run import ExplorerError, ExplorerRunner, main
from vectornaut.explorer.schemas import (
    BackOfEnvelope, DescriptorAssignment, MaterialsCandidate, MaterialsCandidateBatch,
)
from vectornaut.config import ParameterProposal
from vectornaut.pipeline import ConceptsExhaustedError, PipelineRunner, PipelineRunRequest

CREDENTIALS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT")
FROZEN = "2026-01-01T00:00:00Z"
MATERIAL_CELL = {"mechanism_class": "interfacial_slip", "length_scale": "10_um",
                 "inspiration_origin": "plant", "governing_quantity": "wall_shear"}


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data_dir = tmp.name
        env = {k: v for k, v in os.environ.items() if k not in CREDENTIALS}
        env["VECTORNAUT_DATA_DIR"] = self.data_dir
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_quiet(self, fn, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)


def assert_elite_invariants(test, archive):
    """
    Elites are the best evaluated entry of their cell (within the tie epsilon, where the tiebreak
    decides); replacements were strict improvements or won a tie on the tiebreak key.
    """
    for key, elite_id in archive.elites.items():
        elite = archive.entries[elite_id]
        test.assertEqual(elite["status"], EVALUATED)
        same_cell = [e for e in archive.entries.values() if e.get("cell") == key and e["status"] == EVALUATED]
        test.assertLessEqual(max(e["score"] for e in same_cell) - elite["score"], TIE_EPSILON)
    for entry in archive.entries.values():
        if entry.get("outcome") == IMPROVED and entry.get("tiebreak") != IMPROVED_ON_TIEBREAK:
            test.assertGreater(entry["score"], archive.entries[entry["replaced"]]["score"])


class BusinessLoopTest(OfflineTestCase):
    def test_three_mock_rounds_grow_the_archive_and_write_reports(self):
        runner = ExplorerRunner(BusinessProfile(), "Reduce food waste in cities", seed=3, mock=True, clock=lambda: FROZEN)
        summary = self.run_quiet(runner.run, 3, 6)
        archive = runner.archive
        self.assertEqual(summary["rounds"], [1, 2, 3])
        self.assertEqual(len(archive.entries), 18)
        self.assertGreater(len(archive.elites), 5)
        strategies = {e["strategy"] for e in archive.entries.values()}
        self.assertTrue({"seed", "refine", "fill_gap"} <= strategies, strategies)
        assert_elite_invariants(self, archive)
        # Every evaluated entry is critic-adjusted and carries its breakdown and raw result.
        for entry in archive.entries.values():
            if entry["status"] == EVALUATED:
                self.assertEqual(entry["score_breakdown"]["inputs_source"], "critic")
                self.assertTrue(os.path.exists(os.path.join(archive.directory, entry["raw_result_path"])))

        report_dir = os.path.join(self.data_dir, "explorer", "business", "reports")
        for name in ("round_001.md", "round_002.md", "round_003.md", "map.md", "archive_export.json"):
            self.assertTrue(os.path.exists(os.path.join(report_dir, name)), name)
        with open(os.path.join(report_dir, "map.md"), encoding="utf-8") as f:
            map_md = f.read()
        for heading in ("## Elites (best per cell)", "## Map: customer_segment x revenue_model",
                        "## Where ideas are proposed vs where they work", "## Strategy yield (all rounds)",
                        "## Trends used for extrapolation", "## Infeasible cells", "not validation"):
            self.assertIn(heading, map_md)
        with open(os.path.join(report_dir, "archive_export.json"), encoding="utf-8") as f:
            export = json.load(f)
        self.assertEqual(len(export["elites"]), len(archive.elites))

        # A second run continues the same archive and round numbering.
        again = ExplorerRunner(BusinessProfile(), "Reduce food waste in cities", seed=3, mock=True, clock=lambda: FROZEN)
        summary = self.run_quiet(again.run, 1, 4)
        self.assertEqual(summary["rounds"], [4])
        self.assertEqual(len(again.archive.entries), 22)
        assert_elite_invariants(self, again.archive)

    def test_runs_are_deterministic_for_a_seed(self):
        def run(seed, sub):
            path = os.path.join(self.data_dir, sub, "archive.json")
            runner = ExplorerRunner(BusinessProfile(), "q", seed=seed, mock=True, archive_path=path, clock=lambda: FROZEN,
                                    out_dir=os.path.join(self.data_dir, sub, "reports"))
            self.run_quiet(runner.run, 3, 5)
            with open(path, encoding="utf-8") as f:
                return f.read()

        self.assertEqual(run(11, "a"), run(11, "b"))
        self.assertNotEqual(run(11, "c"), run(12, "d"))

    def test_infeasible_target_is_marked_and_skipped(self):
        runner = ExplorerRunner(BusinessProfile(), "q", mock=True, clock=lambda: FROZEN)
        target = {"customer_segment": "public_sector", "revenue_model": "advertising", "market_scale": "national",
                  "advantage_type": "cost", "capital_intensity": "seed"}
        order = {"order_id": "r001-01", "strategy": "explore", "target": target,
                 "target_key": runner.archive.space.cell_key(target), "context": {}, "rationale": "", "parent_ids": []}
        with mock.patch("vectornaut.explorer.run.schedule", return_value=[order]):
            record = self.run_quiet(runner.run_round, 1, "run-001")
        self.assertEqual(record["orders"][0]["entry_status"], INFEASIBLE)
        self.assertTrue(runner.archive.is_infeasible(target))
        self.assertIn("advertising", runner.archive.infeasible_reason(target))
        self.assertEqual(runner.archive.proposal_count(target), 0)

    def test_live_mode_refuses_without_api_key(self):
        with self.assertRaises(ExplorerError):
            ExplorerRunner(BusinessProfile(), "q", mock=False)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["--profile", "business", "--query", "q", "--rounds", "1", "--batch", "1"])
        self.assertEqual(code, 2)
        self.assertIn("GEMINI_API_KEY", err.getvalue())

    def test_other_query_needs_its_own_archive(self):
        self.run_quiet(ExplorerRunner(BusinessProfile(), "first", mock=True).run, 1, 2)
        with self.assertRaises(ExplorerError):
            ExplorerRunner(BusinessProfile(), "second", mock=True)
        ExplorerRunner(BusinessProfile(), "second", mock=True, archive_name="second")
        ExplorerRunner(BusinessProfile(), "second", mock=True, allow_new_query=True)

    def test_cli_mock_run(self):
        out = os.path.join(self.data_dir, "cli_reports")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["--profile", "business", "--query", "q", "--rounds", "2", "--batch", "3", "--mock",
                         "--seed", "5", "--strategy-weights", "refine=1,fill_gap=1", "--out", out])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(os.path.join(out, "map.md")))
        self.assertTrue(os.path.exists(os.path.join(out, "round_002.md")))


class MaterialsLoopTest(OfflineTestCase):
    def test_three_mock_rounds_through_the_mock_pipeline(self):
        # Version 6: this test pins the pre-analyst path (the generator frames the request in round 1,
        # cold-start seeds) and the version-5 formula; the analysis-first loop is in test_explorer_v6.py.
        runner = ExplorerRunner(MaterialsProfile(), "Reduce drag of a ship hull coating", seed=1, mock=True,
                                epochs=5, clock=lambda: FROZEN, analyst=False, scoring="legacy")
        summary = self.run_quiet(runner.run, 3, 3)
        archive = runner.archive
        self.assertEqual(summary["rounds"], [1, 2, 3])
        self.assertEqual(len(archive.entries), 9)
        assert_elite_invariants(self, archive)
        evaluated = [e for e in archive.entries.values() if e["status"] == EVALUATED]
        self.assertTrue(evaluated)
        entry = evaluated[0]
        self.assertIn(entry["score_breakdown"]["basis"], ("simulated", "simulated, critic-checked"))
        self.assertTrue(entry["score_breakdown"]["critic_used"])
        self.assertEqual(len(entry["score_breakdown"]["requirements"]), 4)
        self.assertTrue(entry["score_breakdown"]["objective_gain_source"].startswith("critic"))
        # Requirements, objective and baseline come from the (mock) function analysis and are stored once.
        self.assertEqual([(r["name"], r["priority"]) for r in archive.requirements],
                         [("low_friction_drag", "must"), ("non_toxic_antifouling", "must"),
                          ("multi_year_durability", "must"), ("low_application_cost", "nice")])
        self.assertIn("time-averaged", archive.objective_statement)
        self.assertIn("after 12 months in service", archive.baseline_statement)
        self.assertEqual(runner.ctx.baseline_statement, archive.baseline_statement)
        self.assertEqual(archive.relevant_values("governing_quantity"), ["wall_shear", "flow_rate", "fouling_adhesion"])
        self.assertEqual(runner.config.compatibility_axes, ("mechanism_class", "length_scale"))
        self.assertEqual(runner.config.origin_axes, ("inspiration_origin",))
        for record in archive.data["rounds"]:
            self.assertIn("summary", record["strategy_notes"]["extrapolate"])
        # From round 2 on, the generator prompt shows the stored framing.
        prompt = build_prompt(runner.profile, runner.query, [], archive)
        self.assertIn(f"- baseline: {archive.baseline_statement}", prompt)
        # Cold start: the explore slot has no relevant quantities yet and becomes a seed.
        self.assertEqual({o["strategy"] for o in archive.data["rounds"][0]["orders"]}, {"seed"})
        for entry_ in archive.entries.values():
            if entry_.get("strategy") == "explore":
                where = entry_["descriptors"] or archive.space.parse_key(entry_["target_key"])
                self.assertIn(where["governing_quantity"], ("wall_shear", "flow_rate", "fouling_adhesion"))
        with open(os.path.join(archive.directory, entry["raw_result_path"]), encoding="utf-8") as f:
            raw = json.load(f)
        # The pipeline evaluated exactly the generated concept (no miner call, no re-mining).
        self.assertEqual(raw["pipeline_result"]["miner"]["design_name"], entry["title"])
        self.assertEqual(raw["pipeline_result"]["failed_concepts"], [])
        with open(os.path.join(archive.directory, "reports", "map.md"), encoding="utf-8") as f:
            map_md = f.read()
        for heading in ("## Map: mechanism_class x length_scale", "## Requirements of the request",
                        "## Requirement coverage (critic ratings, elites)", "## Values never proposed",
                        "| evidence (tier) | points obj / sim / req | objective gain used % |",
                        "Elites by evidence tier",
                        "- **Objective**:", "- **Baseline**:", "0.45 * objective_score + 0.1 * simulated_score"):
            self.assertIn(heading, map_md)
        with open(os.path.join(archive.directory, "reports", "round_002.md"), encoding="utf-8") as f:
            round_md = f.read()
        self.assertIn("Extrapolation (decided at scheduling time", round_md)
        self.assertIn("- Baseline: ", round_md)
        with open(os.path.join(archive.directory, "reports", "archive_export.json"), encoding="utf-8") as f:
            export = json.load(f)
        self.assertEqual(len(export["requirements"]), 4)
        self.assertEqual(export["baseline_statement"], archive.baseline_statement)
        self.assertTrue(all("evidence_tier" in e and "flags" in e for e in export["elites"]))

    def test_materials_mock_runs_are_deterministic(self):
        def run(sub):
            path = os.path.join(self.data_dir, sub, "archive.json")
            runner = ExplorerRunner(MaterialsProfile(), "q", seed=4, mock=True, epochs=5, archive_path=path,
                                    clock=lambda: FROZEN, out_dir=os.path.join(self.data_dir, sub, "reports"))
            self.run_quiet(runner.run, 2, 3)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            # Raw result paths and pipeline internals (plots, scripts) may carry run-specific paths.
            data = json.loads(text)
            return {k: {kk: vv for kk, vv in v.items() if kk != "raw_result_path"} for k, v in data["entries"].items()}

        self.assertEqual(run("a"), run("b"))

    def test_bionic_request_keeps_targeted_orders_on_biological_origins(self):
        query = "Develop a bio-inspired hull coating that lowers friction drag for years"
        runner = ExplorerRunner(MaterialsProfile(), query, seed=2, mock=True, epochs=5, clock=lambda: FROZEN)
        self.run_quiet(runner.run, 3, 6)
        archive = runner.archive
        self.assertEqual(archive.preferred_values("inspiration_origin"), ["plant", "animal", "microbe"])
        self.assertEqual(runner.config.combine_anchor_axes, ("governing_quantity", "mechanism_class"))
        self.assertEqual(runner.config.trend_group_axes, ("mechanism_class",))
        targeted = [e for e in archive.entries.values() if e["strategy"] in ("fill_gap", "diversify", "combine",
                                                                            "extrapolate")]
        self.assertTrue(targeted)
        for entry in targeted:
            origin = archive.space.parse_key(entry["target_key"]).get("inspiration_origin")
            self.assertIn(origin, (None, "plant", "animal", "microbe"), entry["target_key"])
        for record in archive.data["rounds"][1:]:
            self.assertIn("slots", record["strategy_notes"]["extrapolate"])
            if not record["strategy_notes"]["extrapolate"]["slots"]["allocated"]:
                self.assertNotIn("extrapolate", record["strategy_notes"]["planned"])
        with open(os.path.join(archive.directory, "reports", "map.md"), encoding="utf-8") as f:
            self.assertIn("Preferred `inspiration_origin` values", f.read())
        prompt = build_prompt(runner.profile, runner.query, [], archive)
        self.assertIn("PREFERRED INSPIRATION ORIGINS", prompt)

    def test_version_3_archive_is_migrated_and_rescored_by_the_runner(self):
        runner = ExplorerRunner(MaterialsProfile(), "q", seed=1, mock=True, epochs=5, clock=lambda: FROZEN)
        self.run_quiet(runner.run, 1, 3)
        path = runner.archive.path
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["version"] = 3
        scores = {k: e["score"] for k, e in data["entries"].items()}
        for entry in data["entries"].values():
            if entry["score"] is not None:
                entry["score"] = 1.0                      # as scored by the old formula
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        again = ExplorerRunner(MaterialsProfile(), "q", seed=1, mock=True, epochs=5, clock=lambda: FROZEN)
        self.assertEqual(again.archive.data["version"], ARCHIVE_VERSION)
        self.assertEqual(again.archive.data["migrations"][0]["from_version"], 3)
        for entry_id, score in scores.items():
            if score is not None:
                self.assertAlmostEqual(again.archive.entries[entry_id]["score"], score, places=3)
        self.run_quiet(again.run, 1, 3)
        assert_elite_invariants(self, again.archive)

    def test_archive_with_the_old_vocabulary_is_refused_with_a_clear_message(self):
        path = os.path.join(self.data_dir, "explorer", "materials", "archive.json")
        os.makedirs(os.path.dirname(path))
        axes = MATERIALS_SPACE.to_dict()
        axes[3]["values"] = [v for v in axes[3]["values"] if v not in ("fouling_adhesion", "degradation_rate")]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 2, "profile": "materials", "axes": axes, "queries": ["q"]}, f)
        with self.assertRaises(ExplorerError) as ctx:
            ExplorerRunner(MaterialsProfile(), "q", mock=True)
        self.assertIn("fouling_adhesion", str(ctx.exception))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["--profile", "materials", "--query", "q", "--rounds", "1", "--batch", "1", "--mock"])
        self.assertEqual(code, 2)
        self.assertIn("--archive NAME", err.getvalue())
        # A separate map works.
        ExplorerRunner(MaterialsProfile(), "q", mock=True, archive_name="v3")


def _materials_candidate(order_id="r001-01", cell=None, **changes):
    fields = dict(
        order_id=order_id, title=f"Concept {order_id}", summary="s",
        descriptors=[DescriptorAssignment(axis=k, value=v) for k, v in (cell or MATERIAL_CELL).items()],
        back_of_envelope=BackOfEnvelope(quantity="drag", formula="1/(1+b/h)", value=12.0, unit="%"),
        main_risk="wear", novelty_vs_known="n", inspiration_source="kelp", domain="Fluid Dynamics",
        physical_mechanism="slip", parameters=[ParameterProposal(name="slip_length", value=1e-5, min_bound=1e-6,
                                                                 max_bound=1e-4, justification="j")],
    )
    fields.update(changes)
    return MaterialsCandidate(**fields)


class _Runner:
    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []

    def run(self, req):
        self.requests.append(req)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class MaterialsEvaluatorTest(OfflineTestCase):
    def evaluate(self, outcome, use_critic=False):
        runner = _Runner(outcome)
        # Version 6: pins the version-5 formula (the filter scoring is tested in test_explorer_v6.py).
        profile = MaterialsProfile(runner_factory=lambda: runner, scoring="legacy")
        item = PreparedCandidate(order={"order_id": "r001-01"}, candidate=_materials_candidate(),
                                 descriptors=dict(MATERIAL_CELL), concept={})
        ctx = EvaluationContext(query="q", mock=True, epochs=7, opt_rounds=2, use_critic=use_critic)
        result = profile.evaluate([item], ctx)[0]
        return result, runner.requests[0]

    def test_request_carries_the_concept_and_disables_remining(self):
        result, req = self.evaluate({"status": "completed", "simulator": {"performance_gain_pct": 20.0},
                                     "validation": {"status": "pass", "score": 1.0}})
        self.assertEqual(req.concept.design_name, "Concept r001-01")
        self.assertEqual(req.max_concept_attempts, 1)
        self.assertEqual((req.epochs, req.max_optimization_rounds, req.is_mock), (7, 2, True))
        # No critic: 100 * gate * (W_OBJ * 0.5 * (1 - exp(-12/OBJECTIVE_SCALE)) [own estimate, unchecked]
        #                         + W_SIM * (1 - exp(-20/SIM_SCALE)) + W_REQ * 0.5 [unrated requirements]), gate = 1
        expected = 100 * (W_OBJ * 0.5 * (1 - math.exp(-12.0 / OBJECTIVE_SCALE_PCT))
                          + W_SIM * (1 - math.exp(-20.0 / SIM_SCALE_PCT)) + W_REQ * 0.5)
        self.assertAlmostEqual(result.score, expected, places=3)
        self.assertEqual(result.breakdown["basis"], "simulated")
        self.assertEqual(result.breakdown["evidence_tier"], "simulated")
        self.assertIn("requirements_unrated", result.breakdown["flags"])
        self.assertIn("objective_gain_unchecked", result.breakdown["flags"])

    def test_rejection_marks_infeasible_and_failure_is_failed(self):
        result, _ = self.evaluate({"status": "rejected", "rejection": {"stage": "auditor", "reason": "2nd law"}})
        self.assertEqual((result.status, result.reason), (INFEASIBLE, "2nd law"))
        result, _ = self.evaluate(ConceptsExhaustedError([{"reason": "Audit failed: unsafe", "stage": "auditor"}], 1))
        self.assertEqual(result.status, FAILED)
        self.assertIn("Audit failed", result.reason)
        result, _ = self.evaluate(RuntimeError("boom"))
        self.assertEqual(result.status, FAILED)

    def test_unavailable_gain_falls_back_to_the_marked_estimate(self):
        cand = _materials_candidate()
        for simulator, flag in (({"performance_gain_pct": 0.0, "gain_basis": "n/a"}, "gain_unavailable"),
                                ({"performance_gain_pct": None}, "gain_unavailable"),
                                ({"performance_gain_pct": float("nan")}, "implausible_gain")):
            result = score_pipeline_result({"status": "completed", "simulator": simulator,
                                            "validation": {"status": "warn", "score": 0.5}}, cand,
                                           scoring=ScoringConfig.legacy())
            self.assertEqual(result.breakdown["basis"], "estimated, not simulated")
            self.assertEqual(result.breakdown["evidence_tier"], "estimated")
            self.assertIn(flag, result.breakdown["flags"])
            objective_score = (1 - math.exp(-12.0 / OBJECTIVE_SCALE_PCT)) * 0.5
            # gate = 0.5 * 0.8 (warn); no simulated part; requirements unrated -> 0.5
            self.assertAlmostEqual(result.score, 100 * 0.5 * 0.8 * (W_OBJ * objective_score + W_REQ * 0.5), places=3)
        result = score_pipeline_result({"status": "completed", "simulator": {"performance_gain_pct": 5.0,
                                                                             "gain_basis": "baseline design"},
                                        "validation": {"status": "fail", "score": 0.9}}, cand,
                                       scoring=ScoringConfig.legacy())
        self.assertEqual(result.breakdown["basis"], "simulated")
        self.assertEqual(result.breakdown["components"]["gate"], 0.0)
        self.assertEqual(result.score, 0.0)

    def test_sanity_checks(self):
        profile = MaterialsProfile()
        bad = _materials_candidate(parameters=[ParameterProposal(name="h", value=5.0, min_bound=0.0, max_bound=1.0,
                                                                 justification="")])
        self.assertTrue(any("outside" in issue for issue in profile.sanity_issues(bad, MATERIAL_CELL)))
        self.assertTrue(profile.sanity_issues(_materials_candidate(parameters=[]), MATERIAL_CELL))
        self.assertEqual(profile.sanity_issues(_materials_candidate(), MATERIAL_CELL), [])


class GeneratorTest(OfflineTestCase):
    class Client:
        def __init__(self, batch):
            self.batch = batch
            self.calls = []
            self.models = self

        def generate_content(self, model=None, contents=None, config=None):
            self.calls.append((model, contents, config))
            return SimpleNamespace(parsed=self.batch, text=self.batch.model_dump_json())

    def test_live_call_uses_the_explorer_stage_and_a_structured_prompt(self):
        archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(self.data_dir, "a.json"))
        archive.add_entry(round_no=1, run_id="t", order={"order_id": "x", "strategy": "seed", "target": {}},
                          title="Kelp mucus film", concept={}, descriptors=dict(MATERIAL_CELL), status=EVALUATED,
                          score=50.0)
        target = dict(MATERIAL_CELL, length_scale="100_um")
        orders = [{"order_id": "r002-01", "strategy": "fill_gap", "target": target,
                   "target_key": MATERIALS_SPACE.cell_key(target), "context": {"proposals_here": 0},
                   "rationale": "Empty cell next to 1 elite", "parent_ids": []}]
        client = self.Client(MaterialsCandidateBatch(candidates=[_materials_candidate("r002-01", cell=target)]))
        with mock.patch.dict(os.environ, {"VECTORNAUT_MODEL_EXPLORER": "explorer-model"}):
            result = CandidateGenerator(MaterialsProfile(), client=client).generate("hull drag", orders, archive)
        model, prompt, config = client.calls[0]
        self.assertEqual(model, "explorer-model")
        self.assertIs(config.response_schema, MaterialsCandidateBatch)
        self.assertEqual(result.items[0].status, "ok")
        self.assertEqual(result.items[0].descriptors, target)
        for text in ("REQUEST: \"hull drag\"", "[r002-01] strategy=fill_gap", "length_scale=100_um",
                     "(all axes fixed)", "Kelp mucus film", "Shark-skin riblets", "1. Function analysis",
                     "target_feasible=false", "nm < sub_um < um"):
            self.assertIn(text, prompt)
        self.assertEqual(prompt, build_prompt(MaterialsProfile(), "hull drag", orders, archive))


class ReplayExplorerTest(OfflineTestCase):
    def _answer(self, session, index, payload):
        path = os.path.join(session, "responses", f"{index:02d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def test_replay_pauses_and_resumes(self):
        from vectornaut.llm_replay import run_explorer_session

        session = os.path.join(self.data_dir, "session")
        args = dict(profile="business", query="Reduce food waste in cities", rounds=1, batch=2, seed=0)
        status = self.run_quiet(run_explorer_session, session, **args)
        self.assertEqual((status["status"], status["index"]), ("needs_response", 1))
        with open(status["request"], encoding="utf-8") as f:
            request = f.read()
        self.assertIn("[r001-01]", request)
        self.assertIn("[r001-02]", request)
        self.assertTrue(status["request"].endswith("01_BusinessCandidateBatch.md"))

        def cand(order_id, segment):
            return {
                "order_id": order_id, "title": f"Idea {order_id}", "summary": "s",
                "descriptors": [{"axis": "customer_segment", "value": segment},
                                {"axis": "revenue_model", "value": "subscription"},
                                {"axis": "market_scale", "value": "national"},
                                {"axis": "advantage_type", "value": "cost"},
                                {"axis": "capital_intensity", "value": "seed"}],
                "back_of_envelope": {"quantity": "rev", "formula": "a*b", "value": 1.0, "unit": "EUR"},
                "main_risk": "r", "novelty_vs_known": "n", "value_proposition": "v", "target_customer": "t",
                "advantage_mechanism": "m", "input_assumptions": "a",
                "inputs": {"monthly_revenue_per_customer": 100.0, "gross_margin": 0.8, "cac": 400.0,
                           "monthly_churn": 0.05, "addressable_customers": 100000.0, "reachable_share_3y": 0.05,
                           "fixed_costs_per_year": 200000.0, "upfront_capex": 100000.0},
            }

        self._answer(session, 1, {"function_analysis": "f", "candidates": [cand("r001-01", "smb"),
                                                                            cand("r001-02", "consumers")]})
        status = self.run_quiet(run_explorer_session, session, **args)
        self.assertEqual((status["status"], status["index"]), ("needs_response", 2))
        self.assertTrue(status["request"].endswith("02_CriticBatch.md"))

        self._answer(session, 2, {"reviews": [
            {"order_id": "r001-01", "verdict": "survives", "killer_risks": []},
            {"order_id": "r001-02", "verdict": "refuted", "killer_risks": ["incumbents"]},
        ]})
        status = self.run_quiet(run_explorer_session, session, **args)
        self.assertEqual(status["status"], "complete", status)
        self.assertEqual(status["calls"], 2)
        with open(os.path.join(session, "result.json"), encoding="utf-8") as f:
            first = json.load(f)
        self.assertEqual(first["entries"], 2)
        self.assertTrue(os.path.exists(os.path.join(session, "report.md")))

        archive_path = os.path.join(session, "data", "explorer", "business", "archive.json")
        with open(archive_path, encoding="utf-8") as f:
            data = json.load(f)
        scores = sorted(e["score"] for e in data["entries"].values())
        self.assertAlmostEqual(scores[1], 73.4934, places=3)            # survives, no adjustment
        self.assertAlmostEqual(scores[0], 73.4934 * (1 - 0.4 - 0.05), places=3)  # refuted + one killer risk

        # Rerunning replays everything from scratch and gives the same archive (except timestamps).
        status = self.run_quiet(run_explorer_session, session, **args)
        self.assertEqual(status["status"], "complete")
        with open(archive_path, encoding="utf-8") as f:
            again = json.load(f)
        strip = lambda d: {k: {kk: vv for kk, vv in v.items() if kk not in ("created_at", "updated_at")}
                           for k, v in d["entries"].items()}
        self.assertEqual(strip(again), strip(data))
        self.assertEqual(len(again["rounds"]), 1)


class PipelineConceptHookTest(unittest.TestCase):
    """PipelineRunRequest.concept / max_concept_attempts (used by the materials evaluator)."""

    class Dumpable(SimpleNamespace):
        def model_dump(self):
            return dict(self.__dict__)

    def runner(self, audit_passed=True):
        test = self
        seen = {"mined": 0, "formulated": []}

        class Miner:
            def mock_mine_design(self, query):
                seen["mined"] += 1
                return test.Dumpable(design_name="Mined", inspiration_source="s", physical_mechanism="m")

        class Formulator:
            def mock_formulate_model(self, query, concept):
                seen["formulated"].append(concept.design_name)
                return test.Dumpable(design_name=concept.design_name, inspiration_source="s", domain="Fluid Dynamics",
                                     physical_mechanism="m", parameters=[], governing_equation="d2u_dy2 = 0",
                                     boundary_conditions=["u(0) = 0", "u(1) = 1"], independent_variables=["y"],
                                     dependent_variables=["u"], svg_schematic="")

        class Audit(test.Dumpable):
            audited_parameters_dict = {}
            dimensionless_numbers_dict = {}

        class Auditor:
            def mock_audit_design(self, miner_output, override_parameters=None, user_query=None):
                return Audit(audit_passed=audit_passed, audit_notes="unsafe" if not audit_passed else "ok",
                             audited_parameters=[], dimensionless_numbers=[], simulation_coefficient=0.0,
                             solver_method="analytical", ui_metadata={})

        class Synth:
            def mock_generate_synthesis(self):
                return test.Dumpable(executive_summary="ok")

        def solver(miner_output, auditor_output, epochs):
            return test.Dumpable(solver_method="analytical", epochs_trained=0, final_loss=0.0, loss_history=[],
                                 performance_gain_pct=10.0, relative_error=0.0, sample_points=[0.0, 1.0],
                                 solution_primary=[0.0, 1.0], solution_reference=[0.0, 1.0],
                                 primary_metric_value=1.0, reference_metric_value=1.0)

        runner = PipelineRunner(miner=Miner(), formulator=Formulator(), auditor=Auditor(), optimizer=None,
                                synthesizer=Synth(), solver=solver)
        return runner, seen

    def request(self, **kwargs):
        return PipelineRunRequest(query="q", is_mock=True, max_optimization_rounds=1, **kwargs)

    def test_supplied_concept_replaces_the_miner(self):
        runner, seen = self.runner()
        concept = self.Dumpable(design_name="Supplied", inspiration_source="s", physical_mechanism="m")
        with contextlib.redirect_stdout(io.StringIO()):
            with mock.patch("vectornaut.pipeline.validate_run_output") as validate:
                validate.return_value = SimpleNamespace(status="pass", score=1.0, recommended_action="accept",
                                                        checks=[], model_dump=lambda: {"status": "pass"})
                result = runner.run(self.request(concept=concept, max_concept_attempts=1))
        self.assertEqual(seen["mined"], 0)
        self.assertEqual(seen["formulated"], ["Supplied"])
        self.assertEqual(result["miner"]["design_name"], "Supplied")

    def test_single_attempt_raises_instead_of_remining(self):
        runner, seen = self.runner(audit_passed=False)
        concept = self.Dumpable(design_name="Supplied", inspiration_source="s", physical_mechanism="m")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ConceptsExhaustedError) as ctx:
                runner.run(self.request(concept=concept, max_concept_attempts=1))
        self.assertEqual(ctx.exception.attempts, 1)
        self.assertEqual(len(ctx.exception.failed_concepts), 1)
        self.assertEqual(seen["mined"], 0)

    def test_default_request_still_mines_and_remines(self):
        runner, seen = self.runner(audit_passed=False)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ConceptsExhaustedError) as ctx:
                runner.run(self.request())
        self.assertEqual(ctx.exception.attempts, 3)
        self.assertEqual(seen["mined"], 3)
        self.assertEqual(PipelineRunRequest(query="q").max_concept_attempts, 3)
        self.assertIsNone(PipelineRunRequest(query="q").concept)


if __name__ == "__main__":
    unittest.main()
