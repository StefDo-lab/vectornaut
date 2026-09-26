# -*- coding: utf-8 -*-
"""
Explorer archive version 6: analysis first (analyst stage, ProblemAnalysis, direct-answer seeds and the
"did the map beat the direct answer?" comparison), scope extensions, the evaluator as a filter (simulated
weight 0, validator gate, consistency check), the critic's hard checks and novelty, generator depth,
migration of a version-5 archive, the replay patch of the analyst, and a mock end-to-end loop.
"""
import contextlib
import io
import json
import math
import os
import random
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from vectornaut.config import MODEL_STAGES, ParameterProposal, get_model_name
from vectornaut.explorer import report
from vectornaut.explorer import strategies as st
from vectornaut.explorer.analyst import (
    SEED_DIRECT, ProblemAnalyst, analysis_brief, analysis_to_dict, complete_direct_concept,
)
from vectornaut.explorer.archive import ARCHIVE_VERSION, EVALUATED, NOT_BETTER, Archive
from vectornaut.explorer.generator import DEFAULT_BATCH, build_prompt
from vectornaut.explorer.profiles.business import BusinessProfile
from vectornaut.explorer.profiles.materials import (
    FILTER_W_NOV, HARD_CHECK_FACTOR, HARD_CHECK_NAMES, LABEL_CONSISTENT, LABEL_CONTRADICTED, LABEL_REVIEWED,
    LABEL_UNREVIEWED, MATERIALS_SPACE, MaterialsProfile, ScoringConfig, critic_prompt, rescore_breakdown,
    score_pipeline_result,
)
from vectornaut.explorer.profiles.base import PreparedCandidate
from vectornaut.explorer.run import ExplorerError, ExplorerRunner, build_parser, main
from vectornaut.explorer.schemas import (
    BackOfEnvelope, DescriptorAssignment, DirectConcept, HardCheck, MaterialsCandidate, MaterialsCriticReview,
    ProblemAnalysis, RequirementRating,
)

CREDENTIALS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT")
FROZEN = "2026-01-01T00:00:00Z"
QUERY = "Reduce drag of a ship hull coating"
CELL = {"mechanism_class": "graded_stiffness", "length_scale": "100_um", "inspiration_origin": "animal",
        "governing_quantity": "fouling_adhesion"}
REQUIREMENTS = [{"name": "low_drag", "criterion": "less drag", "priority": "must"},
                {"name": "non_toxic", "criterion": "no biocide", "priority": "must"},
                {"name": "cheap", "criterion": "standard yard", "priority": "nice"}]
V6_CRITIC_FIELDS = ("hard_checks", "simulation_contradicts_claim", "simulation_consistency_note", "novelty_rating",
                    "novelty_reason", "scope_extension_legitimate", "scope_extension_note")
GEMINI_ARCHIVE = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "explorer_trials",
                              "2026-09-27_facade_gemini_live", "archive.json")


def candidate(order_id="r002-01", estimate=5.0, cell=None, **changes):
    fields = dict(
        order_id=order_id, title=f"Concept {order_id}", summary="A graded skin.",
        descriptors=[DescriptorAssignment(axis=k, value=v) for k, v in (cell or CELL).items()],
        back_of_envelope=BackOfEnvelope(quantity="drag", formula="share x effect", value=estimate, unit="%"),
        main_risk="wear", novelty_vs_known="n", baseline="silicone FRC after 12 months",
        inspiration_source="whale skin", domain="Structural Mechanics", physical_mechanism="graded modulus",
        parameters=[ParameterProposal(name="modulus", value=1e6, min_bound=1e5, max_bound=1e7, justification="j")],
    )
    fields.update(changes)
    return MaterialsCandidate(**fields)


def pipeline_result(gain=30.0, status="pass", score=1.0, basis="baseline_parameters"):
    return {"status": "completed",
            "simulator": {"performance_gain_pct": gain, "gain_basis": basis, "solver_method": "analytical"},
            "validation": {"status": status, "score": score, "checks": []},
            "miner": {"governing_equation": "d2u_dy2 = 0", "boundary_conditions": ["u(0) = 0"]},
            "auditor": {"audited_parameters": []}}


def review(order_id="r002-01", obj=6.0, coverage=0.8, **changes):
    fields = dict(order_id=order_id, plausible_simulated_benefit_pct=20.0, plausible_objective_gain_pct=obj,
                  killer_risks=["wear"], novelty_rating=0.5,
                  hard_checks=[HardCheck(name=n, passed=True, reason="ok") for n in HARD_CHECK_NAMES],
                  requirement_coverage=[RequirementRating(name=r["name"], coverage=coverage) for r in REQUIREMENTS])
    fields.update(changes)
    return MaterialsCriticReview(**fields)


def score(result=None, cand=None, rev="default", **kwargs):
    rev = review() if rev == "default" else rev
    return score_pipeline_result(result or pipeline_result(), cand or candidate(), review=rev,
                                 requirements=REQUIREMENTS, relevant_quantities=["fouling_adhesion", "wall_shear"],
                                 **kwargs)


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

    def quiet(self, fn, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)

    def runner(self, sub="a", **kwargs):
        path = os.path.join(self.data_dir, sub, "archive.json")
        kwargs.setdefault("seed", 1)
        return ExplorerRunner(MaterialsProfile(), QUERY, mock=True, epochs=5, archive_path=path, clock=lambda: FROZEN,
                              out_dir=os.path.join(self.data_dir, sub, "reports"), **kwargs)


# ---------------------------------------------------------------------------
# 1. analyst: schema, storage, prompt flow
# ---------------------------------------------------------------------------

class AnalystSchemaTest(unittest.TestCase):
    def test_stage_and_schema(self):
        self.assertIn("analyst", MODEL_STAGES)
        with mock.patch.dict(os.environ, {"VECTORNAUT_MODEL_ANALYST": "gemini-3.1-pro-preview"}):
            self.assertEqual(get_model_name("analyst"), "gemini-3.1-pro-preview")
        # Everything is optional, so a partial answer still parses; a direct concept needs no order_id.
        analysis = ProblemAnalysis.model_validate({"system_analysis": "s", "direct_concepts": [{"title": "t"}]})
        self.assertEqual(analysis.direct_concepts[0].order_id, "")
        self.assertIsInstance(analysis.direct_concepts[0], MaterialsCandidate)
        props = ProblemAnalysis.model_json_schema()["properties"]
        for name in ("load_breakdown", "levers", "scope_extension_justified", "conventional_baseline",
                     "objective_statement", "baseline_statement", "target_gain_pct", "requirements",
                     "relevant_mechanism_classes", "relevant_governing_quantities", "relevant_inspiration_origins",
                     "direct_concepts"):
            self.assertIn(name, props)

    def test_prompt_asks_for_system_analysis_levers_scope_and_three_deep_concepts(self):
        prompt = MaterialsProfile().analyst_prompt("Entwickle eine bionische Fassade", None)
        for text in ('REQUEST: "Entwickle eine bionische Fassade"', "load_breakdown", "levers", "within_literal_scope",
                     "extension_justified", "conventional_baseline", "direct_concepts", "best 3 concepts",
                     "realistic_benefit_pct", "self_critique", "processing and thermal stability",
                     "biological model", "nm < sub_um < um"):
            self.assertIn(text, prompt)

    def test_live_call_uses_the_analyst_stage_with_high_thinking(self):
        payload = MaterialsProfile().mock_analysis(QUERY)

        class Client:
            def __init__(self):
                self.calls, self.models = [], self

            def generate_content(self, model=None, contents=None, config=None):
                self.calls.append((model, contents, config))
                return SimpleNamespace(parsed=payload, text=payload.model_dump_json())

        client = Client()
        with mock.patch.dict(os.environ, {"VECTORNAUT_MODEL_ANALYST": "analyst-model"}):
            analysis, prompt = ProblemAnalyst(MaterialsProfile(), client=client).analyse(QUERY, None)
        model, contents, config = client.calls[0]
        self.assertEqual(model, "analyst-model")
        self.assertIs(config.response_schema, ProblemAnalysis)
        self.assertEqual(str(getattr(config.thinking_config.thinking_level, "value",
                                     config.thinking_config.thinking_level)).lower(), "high")
        self.assertEqual(contents, prompt)
        self.assertEqual(len(analysis.direct_concepts), 3)

    def test_business_has_no_analyst(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ExplorerError):
                ExplorerRunner(BusinessProfile(), "q", mock=True, archive_path=os.path.join(tmp, "a.json"),
                               analyst=True)
            runner = ExplorerRunner(BusinessProfile(), "q", mock=True, archive_path=os.path.join(tmp, "b.json"))
            self.assertFalse(runner.use_analyst)
            self.assertEqual(runner.depth, "broad")
            with self.assertRaises(ExplorerError):
                ExplorerRunner(BusinessProfile(), "q", mock=True, archive_path=os.path.join(tmp, "c.json"),
                               scoring="legacy")

    def test_direct_concepts_are_completed_from_their_deep_fields(self):
        concept = DirectConcept(title="Louvre screen", key_physics="g-value 0.15 x 45 % glazing share",
                                realistic_benefit_pct=25.0, benefit_reasoning="45 % x 60 %", risks=["wind load"])
        complete_direct_concept(concept)
        self.assertEqual((concept.back_of_envelope.value, concept.back_of_envelope.unit), (25.0, "%"))
        self.assertEqual(concept.back_of_envelope.formula, "45 % x 60 %")
        self.assertEqual(concept.main_risk, "wind load")
        self.assertEqual(concept.physical_mechanism, "g-value 0.15 x 45 % glazing share")
        self.assertTrue(concept.summary)

    def test_brief_is_compact_and_names_levers_scope_and_seeds(self):
        stored = analysis_to_dict(MaterialsProfile().mock_analysis(QUERY))
        self.assertEqual(len(stored["direct_concepts"]), 3)
        self.assertNotIn("parameters", stored["direct_concepts"][0])        # the full seeds are archive entries
        self.assertEqual(stored["direct_concepts"][0]["descriptors"]["mechanism_class"], "graded_stiffness")
        text = analysis_brief(stored)
        for part in ("Load breakdown:", "Levers (largest first): 1. fouling control ~10 %",
                     "[outside the literal wording; extension justified]", "Scope extension: justified",
                     "Direct-answer seeds", "[scope extension]"):
            self.assertIn(part, text)
        self.assertLessEqual(len(analysis_brief(dict(stored, system_analysis="x" * 20000), limit=800)), 800)


class AnalysisFlowTest(OfflineTestCase):
    def test_analysis_round_stores_the_framing_and_seeds_before_any_search_round(self):
        runner = self.runner()
        self.assertTrue(runner.use_analyst)
        self.assertEqual((runner.depth, runner.default_batch()), ("deep", DEFAULT_BATCH["deep"]))
        summary = self.quiet(runner.run, 1)
        archive = runner.archive
        self.assertEqual(summary["rounds"], [1, 2])
        self.assertEqual((summary["batch"], summary["depth"], summary["analysis"]), (3, "deep", True))
        analysis = archive.problem_analysis
        self.assertEqual(archive.request_analysis["analysis_round"], 1)
        self.assertEqual(len(analysis["load_breakdown"]), 3)
        self.assertEqual(analysis["levers"][1]["within_literal_scope"], False)
        # The framing comes from the analyst (round 1), not from the generator's step 1.
        self.assertEqual(archive.request_analysis["framing_round"], 1)
        self.assertEqual(archive.request_analysis["requirements_round"], 1)
        self.assertEqual(len(archive.requirements), 4)
        self.assertEqual(archive.stated_relevant_values("mechanism_class"),
                         ["interfacial_slip", "flow_redirection", "trapped_gas_or_liquid", "graded_stiffness"])
        first = archive.data["rounds"][0]
        self.assertEqual(first["kind"], "analysis")
        self.assertEqual({o["strategy"] for o in first["orders"]}, {SEED_DIRECT})
        seeds = [e for e in archive.entries.values() if e["strategy"] == SEED_DIRECT]
        self.assertEqual(len(seeds), 3)
        for seed in seeds:
            self.assertEqual(seed["status"], EVALUATED)                # pipeline + critic like any candidate
            self.assertTrue(seed["score_breakdown"]["critic_used"])
            self.assertEqual(seed["round"], 1)
            self.assertIn("key_physics", seed["concept"])
        # Later prompts see the analysis compactly; the critic gets it through the context.
        prompt = build_prompt(runner.profile, QUERY, [], archive, depth="deep")
        self.assertIn("PROBLEM ANALYSIS (by the analyst", prompt)
        self.assertIn("the problem analysis above fixes the framing", prompt)
        self.assertIn("Scope extensions are allowed", prompt)
        self.assertIn("Levers (largest first)", runner.ctx.analysis_text)
        self.assertEqual(archive.data["rounds"][1]["depth"], "deep")
        # A second run continues without a second analysis.
        summary = self.quiet(self.runner().run, 1)
        self.assertEqual(summary["rounds"], [3])

    def test_critic_prompt_carries_the_analysis(self):
        text = critic_prompt(MATERIALS_SPACE, QUERY, [], [], REQUIREMENTS, analysis_text="Levers (largest first): 1. x")
        self.assertIn("PROBLEM ANALYSIS (by the analyst", text)
        self.assertIn("Levers (largest first): 1. x", text)
        self.assertNotIn("PROBLEM ANALYSIS", critic_prompt(MATERIALS_SPACE, QUERY, [], [], REQUIREMENTS))

    def test_no_analyst_keeps_the_generator_framing_path(self):
        runner = self.runner(analyst=False)
        self.assertEqual(runner.depth, "broad")
        summary = self.quiet(runner.run, 1, 3)
        archive = runner.archive
        self.assertIsNone(archive.problem_analysis)
        self.assertEqual(summary["rounds"], [1])
        self.assertEqual({o["strategy"] for o in archive.data["rounds"][0]["orders"]}, {"seed"})
        self.assertTrue(archive.objective_statement)                     # framed by the (mock) generator
        prompt = build_prompt(runner.profile, QUERY, [], archive)
        self.assertNotIn("PROBLEM ANALYSIS", prompt)
        self.assertNotIn("DEPTH MODE", prompt)


# ---------------------------------------------------------------------------
# 2. seeds, anchoring and the comparison section
# ---------------------------------------------------------------------------

def _entry(archive, title, strategy, cell, result):
    return archive.add_entry(round_no=1, run_id="t", order={"order_id": title, "strategy": strategy, "target": {}},
                             title=title, concept={}, descriptors=cell, status=EVALUATED, score=result.score,
                             score_breakdown=result.breakdown)


class ComparisonTest(unittest.TestCase):
    def archive(self, tmp):
        archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
        archive.set_problem_analysis({"direct_concepts": [{"title": "seed"}]}, 1)
        return archive

    def test_map_find_with_a_larger_objective_gain_beats_the_direct_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = self.archive(tmp)
            _entry(archive, "seed", SEED_DIRECT, CELL, score(rev=review(obj=5.0, novelty_rating=0.2)))
            _entry(archive, "find", "fill_gap", dict(CELL, length_scale="mm"), score(rev=review(obj=8.0,
                                                                                               novelty_rating=0.7)))
            _entry(archive, "failed gate", "refine", dict(CELL, length_scale="um"),
                   score(result=pipeline_result(status="fail"), rev=review(obj=50.0)))
            data = report.map_vs_direct(archive)
            self.assertTrue(data["map_beat_direct_on_objective"])
            self.assertEqual(data["verdicts"]["objective_gain_pct"]["map"], 8.0)
            self.assertEqual(data["verdicts"]["novelty_rating"]["better"], "map")
            self.assertEqual([r["title"] for r in data["best_map_finds"]], ["find"])   # gate failures excluded
            text = report.render_map(archive, MaterialsProfile(), "q")
            self.assertIn("## Did the map beat the direct answer?", text)
            self.assertIn("the map found a concept with a larger critic objective gain", text)

    def test_direct_answer_ahead_and_no_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = self.archive(tmp)
            _entry(archive, "seed", SEED_DIRECT, CELL, score(rev=review(obj=12.0)))
            _entry(archive, "find", "fill_gap", dict(CELL, length_scale="mm"), score(rev=review(obj=3.0)))
            data = report.map_vs_direct(archive)
            self.assertFalse(data["map_beat_direct_on_objective"])
            self.assertIn("the direct answer is still ahead", report.render_map(archive, MaterialsProfile(), "q"))
            plain = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "b.json"))
            self.assertIsNone(report.map_vs_direct(plain))
            self.assertNotIn("Did the map beat", report.render_map(plain, MaterialsProfile(), "q"))

    def test_strategies_anchor_on_the_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = self.archive(tmp)
            seed = _entry(archive, "seed", SEED_DIRECT, CELL, score(rev=review(obj=5.0)))
            other = _entry(archive, "other", "fill_gap", dict(CELL, mechanism_class="flow_redirection",
                                                              length_scale="mm", inspiration_origin="plant"),
                           score(rev=review(obj=5.0)))
            third = _entry(archive, "third", "fill_gap", dict(CELL, mechanism_class="interfacial_slip",
                                                              length_scale="nm", inspiration_origin="microbe",
                                                              governing_quantity="wall_shear"),
                           score(rev=review(obj=5.0)))
            config = st.StrategyConfig()
            self.assertAlmostEqual(st.source_value(seed, {}, config) - st.source_value(other, {}, config),
                                   config.seed_anchor_bonus + (seed["score"] - other["score"]) / 100, places=6)
            # combine prefers a pair with the seed although other/third are further apart.
            orders = st.combine(archive, random.Random(0), 1, set(), config)
            self.assertIn(seed["id"], orders[0]["parent_ids"])
            self.assertIn("one of the analyst's direct answers", orders[0]["rationale"])
            # fill_gap starts from the seed (bonus) and says so.
            gap = st.fill_gap(archive, random.Random(0), 1, set(), config)[0]
            self.assertEqual(gap["parent_ids"][0], seed["id"])
            self.assertIn("find a concept that beats it", gap["rationale"])
            # refine picks the seed twice as often (weight factor).
            picks = [st.refine(archive, random.Random(i), 1, set(), config)[0]["parent_ids"][0] for i in range(300)]
            self.assertGreater(picks.count(seed["id"]), picks.count(other["id"]) * 1.5)
            self.assertIsNotNone(third)


# ---------------------------------------------------------------------------
# 3. scope extensions
# ---------------------------------------------------------------------------

class ScopeExtensionTest(OfflineTestCase):
    def test_flagged_not_penalised_and_judged_by_the_critic(self):
        plain = score()
        extended = score(cand=candidate(scope_extension=True, scope_extension_reason="windows carry the load"))
        self.assertIn("scope_extension", extended.breakdown["flags"])
        self.assertEqual(extended.breakdown["scope_extension_reason"], "windows carry the load")
        self.assertAlmostEqual(extended.score, plain.score, places=6)
        illegit = score(cand=candidate(scope_extension=True),
                        rev=review(scope_extension_legitimate=False, scope_extension_note="sidesteps the request"))
        self.assertIn("scope_extension_not_legitimate", illegit.breakdown["flags"])
        self.assertAlmostEqual(illegit.score, plain.score, places=6)
        self.assertEqual(illegit.breakdown["critic"]["scope_extension_note"], "sidesteps the request")

    def test_report_lists_scope_extensions_and_the_critic_sees_them(self):
        runner = self.runner()
        self.quiet(runner.run, 1)
        with open(os.path.join(self.data_dir, "a", "reports", "map.md"), encoding="utf-8") as f:
            text = f.read()
        section = text.split("## Scope extensions")[1].split("\n## ")[0]
        self.assertIn("Bottom air-lubrication cavity", section)
        self.assertIn("legitimate", section)
        seed = [e for e in runner.archive.entries.values() if "air-lubrication" in e["title"]][0]
        profile = runner.profile
        item = PreparedCandidate(order={"order_id": "x", "strategy": SEED_DIRECT},
                                 candidate=DirectConcept.model_validate(dict(seed["concept"], order_id="x")),
                                 descriptors=seed["descriptors"], concept={})
        block = critic_prompt(profile.space, QUERY, [item], [{}], REQUIREMENTS)
        self.assertIn("SCOPE EXTENSION", block)
        self.assertIn("the analyst's careful direct answer", block)


# ---------------------------------------------------------------------------
# 4. the evaluator as a filter
# ---------------------------------------------------------------------------

class FilterScoringTest(unittest.TestCase):
    def test_simulated_gain_does_not_rank_by_default(self):
        low, high = score(result=pipeline_result(gain=2.0)), score(result=pipeline_result(gain=80.0))
        self.assertAlmostEqual(low.score, high.score, places=6)
        self.assertEqual(high.breakdown["weights"]["simulated"], 0.0)
        self.assertEqual(high.breakdown["contributions"]["simulated"], 0.0)
        self.assertEqual(high.breakdown["scoring_mode"], "filter")
        self.assertIn("[filter scoring]", high.breakdown["formula"])
        # Ranking = objective + requirements + novelty (contributions add up to the score).
        b = high.breakdown
        self.assertAlmostEqual(sum(b["contributions"].values()), high.score, places=3)
        self.assertEqual(set(b["contributions"]), {"objective", "simulated", "requirements", "novelty"})

    def test_validator_is_a_gate(self):
        passed, warned = score(), score(result=pipeline_result(status="warn", score=0.55))
        self.assertAlmostEqual(passed.score, warned.score, places=6)             # the validator score no longer scales
        self.assertIn("validator_warning", warned.breakdown["flags"])
        failed = score(result=pipeline_result(status="fail", score=0.9))
        self.assertEqual(failed.score, 0.0)
        self.assertIn("validator_failed", failed.breakdown["flags"])
        # Legacy scoring keeps the old factor.
        legacy = score(result=pipeline_result(status="warn", score=0.5), scoring=ScoringConfig.legacy())
        self.assertAlmostEqual(legacy.breakdown["components"]["gate"], 0.4)

    def test_sim_weight_is_configurable(self):
        config = ScoringConfig(sim_weight=0.1)
        weights = config.weights()
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertAlmostEqual(weights["simulated"], 0.1 / 1.1, places=5)
        low = score(result=pipeline_result(gain=2.0), scoring=config)
        high = score(result=pipeline_result(gain=80.0), scoring=config)
        self.assertGreater(high.score, low.score)
        self.assertEqual(ScoringConfig.from_dict(config.to_dict()), config)
        self.assertEqual(ScoringConfig.from_dict({"mode": "nonsense"}).mode, "filter")

    def test_evidence_labels_and_ranks(self):
        consistent = score()
        self.assertEqual((consistent.breakdown["evidence_label"], consistent.breakdown["evidence_tier"],
                          consistent.breakdown["evidence_rank"]), (LABEL_CONSISTENT, "simulated", 2))
        heat = score(cand=candidate(cell=dict(CELL, governing_quantity="heat_flux")))    # irrelevant quantity
        self.assertEqual((heat.breakdown["evidence_label"], heat.breakdown["evidence_tier"],
                          heat.breakdown["evidence_rank"]), (LABEL_REVIEWED, "estimated", 2))
        self.assertEqual(heat.breakdown["objective_discount"], 1.0)       # the critic's gain counts fully
        self.assertAlmostEqual(heat.score, consistent.score, places=6)
        unreviewed = score(rev=None, cand=candidate(estimate=400.0))
        self.assertEqual((unreviewed.breakdown["evidence_label"], unreviewed.breakdown["evidence_rank"]),
                         ("simulation only (no critic)", 1))
        self.assertLessEqual(unreviewed.score, 50.0)
        none = score(rev=None, result=pipeline_result(gain=None, basis="none"))
        self.assertEqual(none.breakdown["evidence_label"], LABEL_UNREVIEWED)

    def test_contradicting_simulation_is_flagged_and_loses_the_cell(self):
        contradicted = score(rev=review(obj=9.0, simulation_contradicts_claim=True,
                                        simulation_consistency_note="the model shows no release benefit"))
        b = contradicted.breakdown
        self.assertIn("simulation_contradicts_claim", b["flags"])
        self.assertEqual((b["evidence_label"], b["evidence_rank"]), (LABEL_CONTRADICTED, 1))
        sound = score(rev=review(obj=5.0))
        self.assertGreater(contradicted.score, sound.score)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
            first = _entry(archive, "sound", "seed", CELL, sound)
            self.assertEqual(_entry(archive, "contradicted", "seed", CELL, contradicted)["outcome"], NOT_BETTER)
            self.assertEqual(archive.elite_for(CELL)["id"], first["id"])

    def test_novelty_term(self):
        new, known = score(rev=review(novelty_rating=1.0)), score(rev=review(novelty_rating=0.0))
        self.assertAlmostEqual(new.score - known.score, 100 * FILTER_W_NOV, places=4)
        unrated = score(rev=review(novelty_rating=None))
        self.assertIn("novelty_unrated", unrated.breakdown["flags"])
        self.assertAlmostEqual(unrated.score, score(rev=review(novelty_rating=0.5)).score, places=6)

    def test_rescoring_matches_scoring_in_both_modes(self):
        for config in (ScoringConfig(), ScoringConfig.legacy(), ScoringConfig(sim_weight=0.2)):
            for res in (score(scoring=config), score(rev=review(simulation_contradicts_claim=True), scoring=config),
                        score(rev=review(hard_checks=[HardCheck(name="physical_bounds", passed=False, reason="r")]),
                              scoring=config),
                        score(rev=None, scoring=config)):
                again, new = rescore_breakdown(res.breakdown, REQUIREMENTS, scoring=config)
                self.assertAlmostEqual(again, res.score, places=4)
                self.assertEqual(new["flags"], res.breakdown["flags"])
                self.assertEqual(new["evidence_rank"], res.breakdown["evidence_rank"])


# ---------------------------------------------------------------------------
# 5. stricter critic: hard checks
# ---------------------------------------------------------------------------

class HardCheckTest(unittest.TestCase):
    def test_failed_hard_check_multiplies_the_score(self):
        plain = score()
        failed = score(rev=review(hard_checks=[
            HardCheck(name="processing_stability", passed=False,
                      reason="calcite decomposes above ~825 C, the glaze fires at 1100 C"),
            HardCheck(name="physical_bounds", passed=None)]))
        self.assertIn("hard_check_failed", failed.breakdown["flags"])
        self.assertEqual(failed.breakdown["hard_check_factor"], HARD_CHECK_FACTOR)
        self.assertAlmostEqual(failed.score, plain.score * HARD_CHECK_FACTOR, places=4)
        self.assertEqual(failed.breakdown["hard_check_failures"][0]["name"], "processing_stability")
        self.assertIn("825", failed.breakdown["hard_check_failures"][0]["reason"])
        unknown = score(rev=review(hard_checks=[HardCheck(name="field_record", passed=None)]))
        self.assertNotIn("hard_check_failed", unknown.breakdown["flags"])
        # Also in legacy mode.
        legacy = score(rev=review(hard_checks=[HardCheck(name="physical_bounds", passed=False)]),
                       scoring=ScoringConfig.legacy())
        self.assertEqual(legacy.breakdown["hard_check_factor"], HARD_CHECK_FACTOR)

    def test_prompt_has_the_checklist_and_calibration(self):
        text = critic_prompt(MATERIALS_SPACE, QUERY, [], [], REQUIREMENTS)
        for part in ("HARD CHECKS", *HARD_CHECK_NAMES, "calcite", "sub-ambient", "vertical facade", "CALIBRATION",
                     "cool/white roof", "External shading of windows", "fouling-release coating",
                     f"multiplies the candidate's score by {HARD_CHECK_FACTOR:g}", "simulation_contradicts_claim",
                     "novelty_rating", "scope_extension_legitimate", "consistency check"):
            self.assertIn(part, text)
        props = MaterialsCriticReview.model_json_schema()["properties"]
        for name in ("hard_checks", "simulation_contradicts_claim", "novelty_rating", "scope_extension_legitimate"):
            self.assertIn(name, props)
        # Older answers without the new fields still parse.
        self.assertEqual(MaterialsCriticReview(order_id="x").hard_checks, [])

    def test_mock_critic_answers_the_checklist(self):
        profile = MaterialsProfile()
        item = PreparedCandidate(order={"order_id": "r001-01", "strategy": "fill_gap"}, candidate=candidate("r001-01"),
                                 descriptors=dict(CELL), concept={})
        rev = profile.mock_review(item, pipeline_result(), REQUIREMENTS)
        self.assertEqual([c.name for c in rev.hard_checks], list(HARD_CHECK_NAMES))
        self.assertIsNotNone(rev.novelty_rating)
        self.assertIsNotNone(rev.simulation_contradicts_claim)


# ---------------------------------------------------------------------------
# 6. depth
# ---------------------------------------------------------------------------

class DepthTest(OfflineTestCase):
    def test_deep_prompt_and_defaults(self):
        archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(self.data_dir, "x.json"))
        deep = build_prompt(MaterialsProfile(), QUERY, [], archive, depth="deep")
        broad = build_prompt(MaterialsProfile(), QUERY, [], archive, depth="broad")
        self.assertIn("DEPTH MODE", deep)
        self.assertIn("self_critique", deep)
        self.assertNotIn("DEPTH MODE", broad)
        self.assertIn("self_critique", MaterialsCandidate.model_json_schema()["properties"])
        self.assertEqual(DEFAULT_BATCH, {"deep": 3, "broad": 6})
        self.assertEqual(self.runner("b", depth="broad").depth, "broad")
        self.assertEqual(self.runner("c", analyst=False, depth="deep").depth, "deep")
        with self.assertRaises(ExplorerError):
            self.runner("d", depth="medium")

    def test_cli_flags(self):
        args = build_parser().parse_args(["--profile", "materials", "--query", "q", "--no-analyst", "--depth", "broad",
                                          "--scoring", "legacy", "--sim-weight", "0.1"])
        self.assertEqual((args.analyst, args.depth, args.scoring, args.sim_weight, args.batch),
                         (False, "broad", "legacy", 0.1, None))
        self.assertIsNone(build_parser().parse_args(["--profile", "materials", "--query", "q"]).analyst)
        out = os.path.join(self.data_dir, "cli")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["--profile", "materials", "--query", QUERY, "--rounds", "1", "--mock", "--epochs", "5",
                         "--out", out, "--archive", "cli"])
        self.assertEqual(code, 0)
        summary = json.loads(buf.getvalue()[buf.getvalue().index("{"):])
        self.assertEqual((summary["depth"], summary["batch"], summary["analysis"], summary["rounds"]),
                         ("deep", 3, True, [1, 2]))


# ---------------------------------------------------------------------------
# 7. migration of a version-5 archive
# ---------------------------------------------------------------------------

class MigrationTest(OfflineTestCase):
    def test_v5_archive_is_rescored_as_a_filter_and_keeps_the_old_framing_path(self):
        runner = self.runner("v5", analyst=False, scoring="legacy")
        self.quiet(runner.run, 1, 3)
        path = runner.archive.path
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["version"] = 5
        for key in ("scoring", "problem_analysis", "analysis_round"):
            data["request_analysis"].pop(key, None)
        legacy = {k: e["score"] for k, e in data["entries"].items()}
        penalised = {k for k, e in data["entries"].items() if (e.get("score_breakdown") or {}).get("hard_check_failures")}
        for entry in data["entries"].values():         # a version-5 review had no checklist, novelty, ...
            b = entry.get("score_breakdown") or {}
            for key in V6_CRITIC_FIELDS:
                (b.get("critic") or {}).pop(key, None)
            for key in ("hard_check_failures", "novelty_rating", "simulation_contradicts_claim", "evidence_label"):
                b.pop(key, None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        again = self.runner("v5")
        archive = again.archive
        self.assertEqual((archive.data["version"], archive.data["migrated_from"]), (ARCHIVE_VERSION, 5))
        self.assertEqual(archive.scoring["mode"], "filter")
        migration = archive.data["migrations"][-1]
        self.assertEqual((migration["from_version"], migration["scoring_mode"]), (5, "filter"))
        changed = 0
        for entry_id, entry in archive.entries.items():
            if entry["score"] is None:
                continue
            b = entry["score_breakdown"]
            self.assertEqual(b["rescored_from"]["score"], legacy[entry_id])
            self.assertEqual(b["scoring_mode"], "filter")
            self.assertEqual(b["contributions"]["simulated"], 0.0)
            self.assertNotIn("hard_check_failed", b["flags"])               # no checklist in version-5 reviews
            changed += abs(entry["score"] - legacy[entry_id]) > 1e-6
        self.assertGreater(changed, 0)
        # (Not saved yet.) Asking for legacy scoring re-scores back to the recorded numbers.
        back = self.runner("v5", scoring="legacy")
        for entry_id, score_ in legacy.items():
            if score_ is not None and entry_id not in penalised:
                self.assertAlmostEqual(back.archive.entries[entry_id]["score"], score_, places=3)
        # No analysis in the old archive: no analyst call later, broad depth, the generator frames as before.
        self.assertIsNone(archive.problem_analysis)
        self.assertEqual(again.depth, "broad")
        summary = self.quiet(again.run, 1, 2)
        self.assertEqual(summary["rounds"], [2])
        self.assertEqual(summary["batch"], 2)
        self.assertFalse(any(e["strategy"] == SEED_DIRECT for e in again.archive.entries.values()))

    def test_live_gemini_facade_archive_rescored_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "archive.json")
            shutil.copy(GEMINI_ARCHIVE, path)
            archive = Archive.load("materials", MATERIALS_SPACE, path=path, migrate=MaterialsProfile().migrate_archive)
            self.assertEqual(archive.data["migrated_from"], 5)
            self.assertEqual(archive.objective_scale["pct"], 18.0)          # the stated target of 35 % stays
            evaluated = [e for e in archive.entries.values() if e["score"] is not None]
            self.assertEqual(len(evaluated), 15)
            for entry in evaluated:
                b = entry["score_breakdown"]
                self.assertEqual(b["evidence_rank"], 2)                     # all critic-reviewed
                self.assertEqual(b["contributions"]["simulated"], 0.0)
                self.assertIn("novelty_unrated", b["flags"])
            # The recorded reviews predate the checklist: the calcite glaze (which decomposes at its own
            # firing temperature) still ranks first, the sealed closed-cell tile second.
            ranked = [e["title"] for e in archive.ranked_elites()]
            self.assertTrue(ranked[0].startswith("Tillandsia-Trichome-Mimetic Sintered Calcite"), ranked[:3])
            self.assertTrue(ranked[1].startswith("Hermetically Sealed Closed-Cell"), ranked[:3])
            # With the processing-stability failure the stricter critic is asked for (calcite/aragonite
            # decompose or transform at glaze and sintering temperatures), the sealed tile leads.
            for entry in archive.entries.values():
                if "Calcit" in entry["title"] or "Carbonate" in entry["title"]:
                    critic = entry["score_breakdown"]["critic"]
                    critic["hard_checks"] = [{"name": "processing_stability", "passed": False,
                                              "reason": "CaCO3 decomposes below the firing temperature"}]
            profile = MaterialsProfile()
            profile.update_scoring(archive, None, force=True)
            ranked = [e["title"] for e in archive.ranked_elites()]
            self.assertTrue(ranked[0].startswith("Hermetically Sealed Closed-Cell"), ranked[:3])


# ---------------------------------------------------------------------------
# 8. replay
# ---------------------------------------------------------------------------

class ReplayTest(OfflineTestCase):
    def test_replay_patches_and_answers_the_analyst(self):
        from vectornaut.llm_replay import EXPLORER_CLIENT_MODULES, run_explorer_session

        self.assertIn("vectornaut.explorer.analyst", EXPLORER_CLIENT_MODULES)
        session = os.path.join(self.data_dir, "session")
        args = dict(profile="materials", query=QUERY, rounds=1, batch=2, seed=0, epochs=5)
        status = self.quiet(run_explorer_session, session, **args)
        self.assertEqual((status["status"], status["index"]), ("needs_response", 1))
        self.assertTrue(status["request"].endswith("01_ProblemAnalysis.md"), status)
        with open(status["request"], encoding="utf-8") as f:
            self.assertIn("You are the Analyst", f.read())
        with open(os.path.join(session, "responses", "01.json"), "w", encoding="utf-8") as f:
            f.write(MaterialsProfile().mock_analysis(QUERY).model_dump_json())
        with mock.patch("vectornaut.explorer.analyst.get_client", side_effect=AssertionError("not patched")):
            status = self.quiet(run_explorer_session, session, **args)
        self.assertEqual((status["status"], status["index"]), ("needs_response", 2), status)
        self.assertNotIn("ProblemAnalysis", os.path.basename(status["request"]))   # a pipeline stage of seed 1
        other = os.path.join(self.data_dir, "session2")
        status = self.quiet(run_explorer_session, other, analyst=False, **args)
        self.assertTrue(status["request"].endswith("01_MaterialsCandidateBatch.md"), status)


# ---------------------------------------------------------------------------
# 9. mock end-to-end loop
# ---------------------------------------------------------------------------

class EndToEndTest(OfflineTestCase):
    def test_mock_loop_with_analysis_seeds_and_comparison(self):
        runner = self.runner("e2e", seed=3)
        summary = self.quiet(runner.run, 2)
        archive = runner.archive
        self.assertEqual(summary["rounds"], [1, 2, 3])
        self.assertEqual(len(archive.entries), 9)
        for key, elite_id in archive.elites.items():
            elite = archive.entries[elite_id]
            same = [e for e in archive.entries.values() if e.get("cell") == key and e["status"] == EVALUATED
                    and e["score_breakdown"]["evidence_rank"] == elite["score_breakdown"]["evidence_rank"]]
            self.assertLessEqual(max(e["score"] for e in same) - elite["score"], 1.0)
        seed_ids = {e["id"] for e in archive.entries.values() if e["strategy"] == SEED_DIRECT}
        anchored = [e for e in archive.entries.values() if set(e.get("parent_ids") or []) & seed_ids]
        self.assertTrue(anchored)                                            # strategies search around the seeds
        for entry in archive.entries.values():
            if entry["status"] == EVALUATED:
                self.assertEqual(entry["score_breakdown"]["scoring_mode"], "filter")
                self.assertIn(entry["score_breakdown"]["evidence_label"], (LABEL_CONSISTENT, LABEL_REVIEWED,
                                                                           LABEL_CONTRADICTED))
        reports = os.path.join(self.data_dir, "e2e", "reports")
        with open(os.path.join(reports, "map.md"), encoding="utf-8") as f:
            text = f.read()
        for heading in ("## Problem analysis (analysis first)", "**Where the load comes from**",
                        "**Levers, ranked by expected magnitude**", "**Direct-answer seeds**",
                        "## Did the map beat the direct answer?", "**Verdict**", "## Scope extensions",
                        "| evidence (tier) | points obj / sim / req / nov |", "[filter scoring]", "| seed_direct |"):
            self.assertIn(heading, text)
        with open(os.path.join(reports, "round_001.md"), encoding="utf-8") as f:
            self.assertIn("(analysis first: the analyst's direct-answer seeds)", f.read())
        with open(os.path.join(reports, "archive_export.json"), encoding="utf-8") as f:
            export = json.load(f)
        self.assertIn("verdicts", export["map_vs_direct"])
        self.assertEqual(export["scoring"]["mode"], "filter")
        self.assertEqual(len(export["problem_analysis"]["direct_concepts"]), 3)
        # Deterministic for a seed.
        other = self.runner("e2e_b", seed=3)
        self.quiet(other.run, 2)
        strip = lambda a: {k: (v["title"], v["score"], v["cell"]) for k, v in a.entries.items()}
        self.assertEqual(strip(other.archive), strip(archive))
        self.assertTrue(math.isfinite(sum(e["score"] or 0 for e in archive.entries.values())))


if __name__ == "__main__":
    unittest.main()
