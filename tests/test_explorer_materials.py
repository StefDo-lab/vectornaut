# -*- coding: utf-8 -*-
"""Materials explorer: evidence tiers, gain sanity, critic combination, requirements, prompts."""
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from vectornaut.config import ParameterProposal
from vectornaut.explorer.archive import EVALUATED, IMPROVED, NOT_BETTER, Archive
from vectornaut.explorer.generator import CandidateGenerator, build_prompt, check_batch, compact_context
from vectornaut.explorer.profiles.base import EvaluationContext, PreparedCandidate
from vectornaut.explorer.profiles.materials import (
    ESTIMATED_SCORE_CAP, MATERIALS_SPACE, MaterialsProfile, critic_prompt, score_pipeline_result, to_miner_concept,
)
from vectornaut.explorer.schemas import (
    BackOfEnvelope, DescriptorAssignment, MaterialsCandidate, MaterialsCandidateBatch, MaterialsCriticBatch,
    MaterialsCriticReview, Requirement, RequirementRating,
)
from vectornaut.explorer import strategies as st

CELL = {"mechanism_class": "graded_stiffness", "length_scale": "100_um",
        "inspiration_origin": "animal", "governing_quantity": "stress"}
REQUIREMENTS = [{"name": "low_friction_drag", "criterion": "less drag than a standard coating"},
                {"name": "non_toxic_antifouling", "criterion": "no biocide release"},
                {"name": "multi_year_durability", "criterion": "several years in seawater"}]


def candidate(order_id="r001-01", estimate=85.0, cell=None, **changes):
    fields = dict(
        order_id=order_id, title=f"Concept {order_id}", summary="A graded laminate that sheds fouling.",
        descriptors=[DescriptorAssignment(axis=k, value=v) for k, v in (cell or CELL).items()],
        back_of_envelope=BackOfEnvelope(quantity="drag reduction", formula="d = 1 - tau/tau0", value=estimate, unit="%"),
        main_risk="delamination", novelty_vs_known="graded instead of uniform silicone",
        baseline="clean standard foul-release hull coating",
        inspiration_source="whale skin", domain="Structural Mechanics", physical_mechanism="graded modulus",
        parameters=[ParameterProposal(name="modulus", value=1e6, min_bound=1e5, max_bound=1e7, justification="j")],
    )
    fields.update(changes)
    return MaterialsCandidate(**fields)


def pipeline_result(gain, status="pass", score=1.0, sanity_passed=None, basis="baseline_parameters"):
    checks = []
    if sanity_passed is not None:
        checks.append({"name": "physics_performance_gain_sanity", "passed": sanity_passed, "score": 0.35,
                       "detail": f"performance_gain_pct={gain}, allowed absolute max=500.0.", "severity": "warning"})
    return {
        "status": "completed",
        "simulator": {"performance_gain_pct": gain, "gain_basis": basis, "solver_method": "analytical",
                      "metric_spec": {"kind": "value_at", "label": "surface displacement", "unit": "m"}},
        "validation": {"status": status, "score": score, "checks": checks},
        "miner": {"governing_equation": "d2u_dy2 = 0", "boundary_conditions": ["u(0) = 0", "u(1) = 1"]},
        "auditor": {"audited_parameters": [{"name": "calibrated_gap", "value": 8.8e-05}],
                    "baseline_description": "uniform silicone"},
    }


def review(order_id="r001-01", plausible=None, coverage=(), **changes):
    fields = dict(order_id=order_id, plausible_gain_pct=plausible, plausible_gain_reasoning="literature",
                  key_assumption_issues=["gap chosen to match"], killer_risks=["wear"],
                  requirement_coverage=[RequirementRating(name=n, coverage=c, reason="r") for n, c in coverage])
    fields.update(changes)
    return MaterialsCriticReview(**fields)


class GainSanityTest(unittest.TestCase):
    def test_sanity_failed_gain_does_not_count(self):
        # The recorded 68,858 % artefact: validator warn 0.948 with a failed gain-sanity check.
        result = score_pipeline_result(pipeline_result(68857.88, status="warn", score=0.948, sanity_passed=False),
                                       candidate(estimate=85.0))
        b = result.breakdown
        self.assertIn("implausible_gain", b["flags"])
        self.assertEqual(b["evidence_tier"], "estimated")
        self.assertEqual(b["evidence_rank"], 0)
        self.assertIsNone(b["performance_gain_pct"])
        self.assertAlmostEqual(b["simulated_gain_pct"], 68857.88)      # kept for the record
        self.assertEqual(b["used_gain_pct"], 85.0)                      # the candidate's estimate, discounted
        self.assertLessEqual(result.score, ESTIMATED_SCORE_CAP)
        self.assertEqual(b["gain_sanity"]["passed"], False)

    def test_absurd_or_non_finite_gain_counts_as_implausible_even_without_the_check(self):
        for gain in (5000.0, -900.0, float("inf"), "nan"):
            b = score_pipeline_result(pipeline_result(gain), candidate()).breakdown
            self.assertIn("implausible_gain", b["flags"], gain)
            self.assertEqual(b["evidence_tier"], "estimated")
        b = score_pipeline_result(pipeline_result(19.0, sanity_passed=True), candidate()).breakdown
        self.assertEqual((b["evidence_tier"], b["flags"]), ("simulated", ["requirements_unrated"]))

    def test_implausible_entry_never_becomes_elite_over_a_sound_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))

            def add(result):
                return archive.add_entry(round_no=1, run_id="t", order={"order_id": "o", "strategy": "seed", "target": {}},
                                         title="t", concept={}, descriptors=dict(CELL), status=EVALUATED,
                                         score=result.score, score_breakdown=result.breakdown)

            sound = score_pipeline_result(pipeline_result(3.0), candidate(estimate=3.0))
            bogus = score_pipeline_result(pipeline_result(68857.88, "warn", 0.948, sanity_passed=False), candidate())
            first = add(sound)
            self.assertEqual(add(bogus)["outcome"], NOT_BETTER)
            self.assertEqual(archive.elite_for(CELL)["id"], first["id"])
            # Reverse order: the sound entry replaces the artefact even with a lower score.
            archive2 = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "b.json"))
            archive = archive2
            add(bogus)
            self.assertEqual(add(sound)["outcome"], IMPROVED)


class TierAndScoreTest(unittest.TestCase):
    def test_estimated_entry_has_a_lower_ceiling(self):
        huge = score_pipeline_result(pipeline_result(None, basis="none"), candidate(estimate=1000.0),
                                     review=review(plausible=900.0, coverage=[(r["name"], 1.0) for r in REQUIREMENTS]),
                                     requirements=REQUIREMENTS)
        self.assertEqual(huge.score, ESTIMATED_SCORE_CAP)
        self.assertTrue(huge.breakdown["capped"])
        simulated = score_pipeline_result(pipeline_result(40.0), candidate(),
                                          review=review(plausible=35.0, coverage=[(r["name"], 1.0) for r in REQUIREMENTS]),
                                          requirements=REQUIREMENTS)
        self.assertGreater(simulated.score, huge.score)
        self.assertEqual(simulated.breakdown["evidence_rank"], 2)
        self.assertGreater(simulated.breakdown["evidence_rank"], huge.breakdown["evidence_rank"])

    def test_breakdown_formula_matches_the_score(self):
        for gain, cov in ((19.0, (1.0, 0.0, 0.5)), (0.13, (0.2, 0.2, 0.2)), (None, (0.9, 0.9, 0.9))):
            res = score_pipeline_result(pipeline_result(gain, basis="none" if gain is None else "baseline_parameters"),
                                        candidate(estimate=8.0), review=review(coverage=zip([r["name"] for r in REQUIREMENTS], cov)),
                                        requirements=REQUIREMENTS)
            c = res.breakdown["components"]
            expected = 100 * c["gate"] * (0.5 * c["gain_score"] + 0.5 * c["requirement_coverage"])
            if res.breakdown["evidence_tier"] == "estimated":
                expected = min(expected, ESTIMATED_SCORE_CAP)
            self.assertAlmostEqual(res.score, expected, places=3)
            self.assertIn("score = 100 * gate * (0.5 * gain_score + 0.5 * requirement_coverage)", res.breakdown["formula"])
        # The old validity floor is gone: a 0.13 % gain with poor coverage scores little.
        tiny = score_pipeline_result(pipeline_result(0.13, "warn", 0.956), candidate(estimate=50.0),
                                     review=review(coverage=[(r["name"], 0.2) for r in REQUIREMENTS]),
                                     requirements=REQUIREMENTS)
        self.assertLess(tiny.score, 10.0)

    def test_requirement_coverage_drives_the_score(self):
        def score(cov):
            return score_pipeline_result(pipeline_result(19.0), candidate(),
                                         review=review(plausible=19.0, coverage=cov), requirements=REQUIREMENTS)
        good = score([("low_friction_drag", 0.9), ("non_toxic_antifouling", 0.9), ("multi_year_durability", 0.9)])
        poor = score([("low_friction_drag", 0.9), ("non_toxic_antifouling", 0.1), ("Multi Year Durability", 0.1)])
        self.assertGreater(good.score - poor.score, 25.0)
        self.assertAlmostEqual(poor.breakdown["requirement_coverage"], (0.9 + 0.1 + 0.1) / 3, places=4)
        self.assertEqual([r["coverage"] for r in poor.breakdown["requirements"]], [0.9, 0.1, 0.1])
        partial = score([("low_friction_drag", 1.0), ("unknown", 0.0)])
        self.assertIn("requirements_partially_rated", partial.breakdown["flags"])
        self.assertAlmostEqual(partial.breakdown["requirement_coverage"], (1.0 + 0.5 + 0.5) / 3, places=4)
        unrated = score_pipeline_result(pipeline_result(19.0), candidate(), requirements=REQUIREMENTS)
        self.assertIn("requirements_unrated", unrated.breakdown["flags"])
        self.assertEqual(unrated.breakdown["requirement_coverage"], 0.5)


class CriticCombinationTest(unittest.TestCase):
    def score(self, simulated, plausible):
        return score_pipeline_result(pipeline_result(simulated), candidate(estimate=7.0),
                                     review=review(plausible=plausible), requirements=REQUIREMENTS).breakdown

    def test_lower_value_is_used_when_simulation_and_critic_differ_by_more_than_2x(self):
        b = self.score(30.0, 10.0)
        self.assertEqual(b["used_gain_pct"], 10.0)
        self.assertIn("model_assumption_sensitive", b["flags"])
        self.assertEqual((b["simulated_gain_pct"], b["critic_plausible_gain_pct"], b["estimated_gain_pct"]), (30.0, 10.0, 7.0))
        self.assertEqual(b["evidence_tier"], "simulated")
        b = self.score(5.0, 20.0)
        self.assertEqual(b["used_gain_pct"], 5.0)
        self.assertIn("model_assumption_sensitive", b["flags"])
        b = self.score(5.0, -1.0)           # sign disagreement
        self.assertEqual(b["used_gain_pct"], -1.0)
        self.assertIn("model_assumption_sensitive", b["flags"])

    def test_simulation_is_used_when_they_agree_within_2x(self):
        b = self.score(12.0, 10.0)
        self.assertEqual(b["used_gain_pct"], 12.0)
        self.assertNotIn("model_assumption_sensitive", b["flags"])
        self.assertEqual(b["critic"]["key_assumption_issues"], ["gap chosen to match"])

    def test_without_simulated_gain_the_lower_estimate_counts(self):
        b = score_pipeline_result(pipeline_result(0.0, basis="none"), candidate(estimate=15.0),
                                  review=review(plausible=4.0)).breakdown
        self.assertEqual((b["evidence_tier"], b["used_gain_pct"]), ("estimated", 4.0))
        self.assertIn("critic", b["gain_source"])


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, []
        self.models = self

    def generate_content(self, model=None, contents=None, config=None):
        self.calls.append((model, contents, config))
        if self.error:
            raise self.error
        return SimpleNamespace(parsed=self.payload, text=self.payload.model_dump_json())


class Runner:
    def __init__(self, results):
        self.results = list(results)

    def run(self, request):
        return self.results.pop(0)


class CriticStageTest(unittest.TestCase):
    def items(self):
        return [PreparedCandidate(order={"order_id": f"r001-0{i}"}, candidate=candidate(f"r001-0{i}"),
                                  descriptors=dict(CELL), concept={}) for i in (1, 2)]

    def test_one_critic_call_per_batch_with_formulation_and_gains(self):
        batch = MaterialsCriticBatch(reviews=[
            review("r001-01", plausible=6.0, coverage=[("low_friction_drag", 0.8)]),
            review("r001-02", plausible=40.0),
        ])
        client = FakeClient(batch)
        profile = MaterialsProfile(runner_factory=lambda: runner, critic_client=client)
        runner = Runner([pipeline_result(19.35), pipeline_result(68857.88, "warn", 0.948, sanity_passed=False)])
        ctx = EvaluationContext(query="hull coating", mock=False, requirements=REQUIREMENTS)
        with mock.patch.dict(os.environ, {"VECTORNAUT_MODEL_CRITIC": "critic-model"}):
            first, second = profile.evaluate(self.items(), ctx)
        self.assertEqual(len(client.calls), 1)
        model, prompt, config = client.calls[0]
        self.assertEqual(model, "critic-model")
        self.assertIs(config.response_schema, MaterialsCriticBatch)
        for text in ("REQUEST: \"hull coating\"", "non_toxic_antifouling: no biocide release", "[r001-01]", "[r001-02]",
                     "formulation: d2u_dy2 = 0", "calibrated_gap=8.8e-05", "simulated gain: 19.35 %",
                     "REJECTED as implausible", "candidate's estimate: 85 %", "clean standard foul-release",
                     "surface displacement", "uniform silicone"):
            self.assertIn(text, prompt)
        self.assertEqual(first.breakdown["used_gain_pct"], 6.0)          # 19.35 vs 6 -> lower, flagged
        self.assertIn("model_assumption_sensitive", first.breakdown["flags"])
        self.assertEqual(second.breakdown["evidence_tier"], "estimated")
        self.assertEqual(second.breakdown["used_gain_pct"], 40.0)        # min(critic 40, estimate 85)
        self.assertEqual(first.raw["critic_review"]["plausible_gain_pct"], 6.0)

    def test_failed_critic_keeps_the_simulation_and_flags_it(self):
        client = FakeClient(error=RuntimeError("quota"))
        runner = Runner([pipeline_result(10.0), pipeline_result(12.0)])
        profile = MaterialsProfile(runner_factory=lambda: runner, critic_client=client)
        results = profile.evaluate(self.items(), EvaluationContext(query="q", mock=False))
        for res in results:
            self.assertEqual(res.status, EVALUATED)
            self.assertIn("critic_failed", res.breakdown["flags"])
            self.assertIn("quota", res.breakdown["critic_error"])

    def test_no_critic_option_skips_the_call(self):
        client = FakeClient(error=AssertionError("must not be called"))
        runner = Runner([pipeline_result(10.0), pipeline_result(12.0)])
        profile = MaterialsProfile(runner_factory=lambda: runner, critic_client=client)
        results = profile.evaluate(self.items(), EvaluationContext(query="q", mock=False, use_critic=False))
        self.assertEqual(client.calls, [])
        self.assertNotIn("critic_missing", results[0].breakdown["flags"])

    def test_critic_prompt_without_requirements_asks_for_them(self):
        text = critic_prompt(MATERIALS_SPACE, "q", self.items()[:1], [pipeline_result(5.0)], [])
        self.assertIn("none extracted", text)

    def test_replay_patches_the_materials_critic_client(self):
        from vectornaut.llm_replay import EXPLORER_CLIENT_MODULES
        self.assertIn("vectornaut.explorer.profiles.materials", EXPLORER_CLIENT_MODULES)


class ConceptHandoverTest(unittest.TestCase):
    def test_candidate_context_reaches_the_pipeline(self):
        concept = to_miner_concept(candidate())
        text = concept.physical_mechanism
        for part in ("graded modulus.", "Summary: A graded laminate that sheds fouling.",
                     "Comparison baseline: clean standard foul-release hull coating.",
                     "Candidate estimate: drag reduction = 85 % (d = 1 - tau/tau0).", "Main risk: delamination.",
                     "Novelty vs known: graded instead of uniform silicone."):
            self.assertIn(part, text)
        self.assertEqual(concept.design_name, "Concept r001-01")
        bare = to_miner_concept(candidate(summary="", baseline="", main_risk="", novelty_vs_known="",
                                          back_of_envelope=None))
        self.assertEqual(bare.physical_mechanism, "graded modulus.")


class ContextTruncationTest(unittest.TestCase):
    def context(self):
        long = "x" * 900
        return {"neighbours": [
            {"id": f"e0000{i}", "title": f"Title {i} " + "t" * 50, "score": 77.7 + i, "basis": "simulated",
             "main_risk": "Lubricant depletion: " + "r" * 400, "cell": "mechanism_class=other, " + "c" * 120,
             "summary": long, "differs_in": "mechanism_class"}
            for i in range(3)
        ], "attempts_here": 0}

    def test_truncated_context_is_valid_json_with_key_fields(self):
        for limit in (2000, 1200, 700):
            text = compact_context(self.context(), limit)
            data = json.loads(text)
            self.assertLessEqual(len(text), limit)
            for i, n in enumerate(data["neighbours"]):
                self.assertEqual(n["id"], f"e0000{i}")
                self.assertEqual(n["score"], round(77.7 + i, 4))
                self.assertTrue(n["title"].startswith(f"Title {i}"))
                self.assertTrue(n["main_risk"].startswith("Lubricant depletion"))
            self.assertEqual(data["attempts_here"], 0)
        small = {"a": 1, "b": "short"}
        self.assertEqual(json.loads(compact_context(small)), small)

    def test_prompt_contexts_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
            orders = [{"order_id": "r002-01", "strategy": "fill_gap", "target": dict(CELL),
                       "target_key": MATERIALS_SPACE.cell_key(CELL), "context": self.context(), "rationale": "r",
                       "parent_ids": []}]
            prompt = build_prompt(MaterialsProfile(), "q", orders, archive)
        line = next(l for l in prompt.splitlines() if l.strip().startswith("context:"))
        data = json.loads(line.split("context:", 1)[1])
        self.assertEqual(len(data["neighbours"]), 3)


class GeneratorAnalysisTest(unittest.TestCase):
    def test_prompt_asks_for_baseline_requirements_and_relevant_quantities(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
            archive.declare_relevance_axes(["governing_quantity"])
            orders = [{"order_id": "r001-01", "strategy": "seed", "target": {}, "target_key": "", "context": {},
                       "rationale": "free", "parent_ids": []}]
            prompt = build_prompt(MaterialsProfile(), "q", orders, archive)
            for text in ("baseline: name the baseline", "not a fouled or untreated", "REQUIREMENTS OF THE REQUEST: not extracted",
                         "relevant_governing_quantities"):
                self.assertIn(text, prompt)
            archive.set_requirements(REQUIREMENTS, 1)
            archive.add_relevant_values("governing_quantity", ["wall_shear"])
            prompt = build_prompt(MaterialsProfile(), "q", orders, archive)
            self.assertIn("- non_toxic_antifouling: no biocide release", prompt)
            self.assertIn("RELEVANT GOVERNING QUANTITIES", prompt)

    def test_check_batch_reads_requirements_and_validates_relevant_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
            orders = [{"order_id": "r001-01", "strategy": "seed", "target": {}, "target_key": "", "context": {},
                       "rationale": "", "parent_ids": []}]
            batch = MaterialsCandidateBatch(
                function_analysis="f", requirements=[Requirement(name="low_friction_drag", criterion="c"),
                                                     Requirement(name="", criterion="dropped")],
                relevant_governing_quantities=["Wall Shear", "drag", "stress"],
                candidates=[candidate("r001-01")])
            result = check_batch(MaterialsProfile(), orders, batch, archive)
        self.assertEqual(result.items[0].status, "ok")
        self.assertEqual(result.batch_notes["requirements"], [{"name": "low_friction_drag", "criterion": "c"}])
        self.assertEqual(result.batch_notes["relevant_values"], {"governing_quantity": ["wall_shear", "stress"]})
        self.assertEqual(result.batch_notes["relevant_values_rejected"], ["drag"])

    def test_missing_baseline_is_flagged_not_rejected(self):
        b = score_pipeline_result(pipeline_result(10.0), candidate(baseline="")).breakdown
        self.assertIn("baseline_unstated", b["flags"])
        self.assertEqual(MaterialsProfile().sanity_issues(candidate(baseline=""), CELL), [])


if __name__ == "__main__":
    unittest.main()
