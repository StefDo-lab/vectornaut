# -*- coding: utf-8 -*-
"""Materials explorer: evidence tiers, gain sanity, two-number critic scoring, requirements, framing, prompts."""
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from vectornaut.config import ParameterProposal
from vectornaut.explorer.archive import ARCHIVE_VERSION, EVALUATED, IMPROVED, NOT_BETTER, TIE_EPSILON, Archive
from vectornaut.explorer.generator import CandidateGenerator, build_prompt, check_batch, compact_context
from vectornaut.explorer.profiles.base import EvaluationContext, PreparedCandidate
from vectornaut.explorer.profiles.materials import (
    ESTIMATE_DISCOUNT, ESTIMATED_SCORE_CAP, MATERIALS_SPACE, OBJECTIVE_SCALE_PCT, RELABEL_FACTOR, SIM_SCALE_PCT,
    W_OBJ, W_REQ, W_SIM, MaterialsProfile, critic_prompt, differs_strongly, framing_warnings, rescore_breakdown,
    score_pipeline_result, to_miner_concept,
)
from vectornaut.explorer.schemas import (
    BackOfEnvelope, DescriptorAssignment, MaterialsCandidate, MaterialsCandidateBatch, MaterialsCriticBatch,
    MaterialsCriticReview, Requirement, RequirementRating,
)
from vectornaut.explorer import strategies as st

CELL = {"mechanism_class": "graded_stiffness", "length_scale": "100_um",
        "inspiration_origin": "animal", "governing_quantity": "stress"}
FOULING_CELL = dict(CELL, governing_quantity="fouling_adhesion")
REQUIREMENTS = [{"name": "low_friction_drag", "criterion": "less drag than a standard coating"},
                {"name": "non_toxic_antifouling", "criterion": "no biocide release"},
                {"name": "multi_year_durability", "criterion": "several years in seawater"}]
OBJECTIVE = "time-averaged hull friction drag over a 5-year docking interval, including fouling"
BASELINE = "conventional biocide-free silicone foul-release coating after 12 months in service"


def f(gain, scale):
    return 1 - math.exp(-gain / scale) if gain > 0 else 0.0


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


def review(order_id="r001-01", sim=None, obj=None, coverage=(), **changes):
    fields = dict(order_id=order_id, plausible_simulated_benefit_pct=sim, plausible_objective_gain_pct=obj,
                  plausible_gain_reasoning="literature",
                  key_assumption_issues=["gap chosen to match"], killer_risks=["wear"],
                  requirement_coverage=[RequirementRating(name=n, coverage=c, reason="r") for n, c in coverage])
    fields.update(changes)
    return MaterialsCriticReview(**fields)


def full_coverage(value=1.0):
    return [(r["name"], value) for r in REQUIREMENTS]


class VocabularyTest(unittest.TestCase):
    def test_fouling_control_quantities_are_in_the_vocabulary(self):
        axis = MATERIALS_SPACE.axis("governing_quantity")
        for value in ("fouling_adhesion", "degradation_rate"):
            self.assertIn(value, axis.values)
            self.assertIn(value, axis.value_help)
        self.assertEqual(axis.values[-1], "other")
        self.assertEqual(MATERIALS_SPACE.validate(dict(CELL, governing_quantity="Fouling Adhesion"))["governing_quantity"],
                         "fouling_adhesion")
        self.assertIn("fouling_adhesion", MATERIALS_SPACE.vocabulary_text())


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
        self.assertIsNone(b["simulated_benefit_used_pct"])
        self.assertAlmostEqual(b["simulated_gain_pct"], 68857.88)      # kept for the record
        self.assertEqual(b["objective_gain_pct"], 85.0)                 # the candidate's estimate, discounted
        self.assertIn("objective_gain_unchecked", b["flags"])
        self.assertEqual(b["objective_discount"], ESTIMATE_DISCOUNT)
        self.assertLessEqual(result.score, ESTIMATED_SCORE_CAP)
        self.assertEqual(b["gain_sanity"]["passed"], False)

    def test_absurd_or_non_finite_gain_counts_as_implausible_even_without_the_check(self):
        for gain in (5000.0, -900.0, float("inf"), "nan"):
            b = score_pipeline_result(pipeline_result(gain), candidate()).breakdown
            self.assertIn("implausible_gain", b["flags"], gain)
            self.assertEqual(b["evidence_tier"], "estimated")
        b = score_pipeline_result(pipeline_result(19.0, sanity_passed=True), candidate()).breakdown
        self.assertEqual((b["evidence_tier"], b["flags"]), ("simulated", ["objective_gain_unchecked", "requirements_unrated"]))

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
                                     review=review(obj=900.0, coverage=full_coverage()), requirements=REQUIREMENTS)
        self.assertEqual(huge.score, ESTIMATED_SCORE_CAP)
        self.assertTrue(huge.breakdown["capped"])
        simulated = score_pipeline_result(pipeline_result(40.0), candidate(),
                                          review=review(sim=35.0, obj=35.0, coverage=full_coverage()),
                                          requirements=REQUIREMENTS)
        self.assertGreater(simulated.score, huge.score)
        self.assertEqual(simulated.breakdown["evidence_rank"], 2)
        self.assertGreater(simulated.breakdown["evidence_rank"], huge.breakdown["evidence_rank"])

    def test_breakdown_formula_matches_the_score(self):
        cases = ((19.0, (1.0, 0.0, 0.5), False), (0.13, (0.2, 0.2, 0.2), False), (None, (0.9, 0.9, 0.9), False),
                 (19.0, (0.8, 0.8, 0.8), True))
        for gain, cov, relabelled in cases:
            res = score_pipeline_result(pipeline_result(gain, basis="none" if gain is None else "baseline_parameters"),
                                        candidate(estimate=8.0),
                                        review=review(sim=gain, obj=3.0, relabelled_analogue=relabelled,
                                                      coverage=zip([r["name"] for r in REQUIREMENTS], cov)),
                                        requirements=REQUIREMENTS)
            b = res.breakdown
            c = b["components"]
            expected = 100 * c["gate"] * (W_OBJ * c["objective_score"] + W_SIM * c["simulated_score"]
                                          + W_REQ * c["requirement_score"]) * b["relabel_factor"] \
                * b["baseline_factor"] * b["must_factor"]
            if b["evidence_tier"] == "estimated":
                expected = min(expected, ESTIMATED_SCORE_CAP)
            self.assertAlmostEqual(res.score, expected, places=3)
            if not b["capped"]:
                self.assertAlmostEqual(sum(b["contributions"].values()), res.score, places=2)
            self.assertIn("score = 100 * gate * (0.45 * objective_score + 0.1 * simulated_score + "
                          "0.45 * requirement_score) * relabel_factor * baseline_factor * must_factor", b["formula"])
            self.assertIn("objective scale 2 % (default)", b["formula"])
            self.assertEqual(b["weights"], {"objective": W_OBJ, "simulated": W_SIM, "requirements": W_REQ})
        # No validity floor: a 0.13 % gain with poor coverage scores little.
        tiny = score_pipeline_result(pipeline_result(0.13, "warn", 0.956), candidate(estimate=50.0),
                                     review=review(sim=0.1, obj=0.1, coverage=[(r["name"], 0.2) for r in REQUIREMENTS]),
                                     requirements=REQUIREMENTS)
        self.assertLess(tiny.score, 10.0)

    def test_requirement_coverage_drives_the_score(self):
        def score(cov):
            return score_pipeline_result(pipeline_result(19.0), candidate(),
                                         review=review(sim=19.0, obj=5.0, coverage=cov), requirements=REQUIREMENTS)
        good = score([("low_friction_drag", 0.9), ("non_toxic_antifouling", 0.9), ("multi_year_durability", 0.9)])
        poor = score([("low_friction_drag", 0.9), ("non_toxic_antifouling", 0.1), ("Multi Year Durability", 0.1)])
        self.assertAlmostEqual(good.breakdown["requirement_score"], 0.9, places=4)
        self.assertEqual(good.breakdown["must_factor"], 1.0)
        self.assertGreater(good.score - poor.score, 30.0)
        self.assertAlmostEqual(poor.breakdown["requirement_coverage"], (0.9 + 0.1 + 0.1) / 3, places=4)
        # Soft minimum over the must-requirements (all must by default) and the soft gate below 0.3.
        self.assertAlmostEqual(poor.breakdown["requirement_score"], 0.5 * 1.1 / 3 + 0.5 * 0.1, places=4)
        self.assertAlmostEqual(poor.breakdown["must_min_coverage"], 0.1)
        self.assertAlmostEqual(poor.breakdown["must_factor"], 1 - 0.5 * (0.3 - 0.1) / 0.3, places=5)
        self.assertIn("must_requirement_unmet", poor.breakdown["flags"])
        self.assertEqual([r["coverage"] for r in poor.breakdown["requirements"]], [0.9, 0.1, 0.1])
        self.assertEqual({r["priority"] for r in poor.breakdown["requirements"]}, {"must"})
        partial = score([("low_friction_drag", 1.0), ("unknown", 0.0)])
        self.assertIn("requirements_partially_rated", partial.breakdown["flags"])
        self.assertAlmostEqual(partial.breakdown["requirement_coverage"], (1.0 + 0.5 + 0.5) / 3, places=4)
        unrated = score_pipeline_result(pipeline_result(19.0), candidate(), requirements=REQUIREMENTS)
        self.assertIn("requirements_unrated", unrated.breakdown["flags"])
        self.assertEqual(unrated.breakdown["requirement_coverage"], 0.5)


class TwoNumberScoringTest(unittest.TestCase):
    """The recorded failure: fouling-release concepts simulated 45-84 % lower release stress but scored ~0 gain."""

    def score(self, gain=60.0, cell=None, relevant=("wall_shear", "fouling_adhesion"), **review_fields):
        fields = dict(sim=50.0, obj=6.0, coverage=full_coverage(0.6))
        fields.update(review_fields)
        return score_pipeline_result(pipeline_result(gain), candidate(estimate=4.0, cell=cell or FOULING_CELL),
                                     review=review(**fields), requirements=REQUIREMENTS,
                                     relevant_quantities=list(relevant) if relevant is not None else None)

    def test_simulated_fouling_release_benefit_and_objective_gain_both_count(self):
        res = self.score()
        b = res.breakdown
        self.assertEqual(b["evidence_tier"], "simulated")
        self.assertEqual(b["simulated_quantity"], "fouling_adhesion")
        self.assertTrue(b["simulated_quantity_relevant"])
        self.assertEqual(b["simulated_benefit_used_pct"], 50.0)          # 60 vs critic 50: the lower one
        self.assertNotIn("model_assumption_sensitive", b["flags"])      # ... but they do not differ strongly
        self.assertEqual(b["objective_gain_pct"], 6.0)
        self.assertEqual(b["objective_gain_source"], "critic")
        self.assertEqual(b["objective_discount"], 1.0)
        expected = 100 * (W_OBJ * f(6.0, OBJECTIVE_SCALE_PCT) + W_SIM * f(50.0, SIM_SCALE_PCT) + W_REQ * 0.6)
        self.assertAlmostEqual(res.score, expected, places=3)
        self.assertAlmostEqual(b["contributions"]["objective"], 100 * W_OBJ * f(6.0, OBJECTIVE_SCALE_PCT), places=3)
        self.assertAlmostEqual(b["contributions"]["simulated"], 100 * W_SIM * f(50.0, SIM_SCALE_PCT), places=3)
        # Zero objective gain still leaves the simulated benefit in the score (the old formula gave ~0).
        zero = self.score(obj=0.0)
        self.assertGreater(zero.breakdown["contributions"]["simulated"], 9.0)

    def test_simulation_is_critic_checked(self):
        b = self.score(gain=84.0, sim=20.0).breakdown
        self.assertEqual(b["simulated_benefit_used_pct"], 20.0)
        self.assertIn("model_assumption_sensitive", b["flags"])
        self.assertEqual(b["basis"], "simulated, critic-checked")

    def test_irrelevant_quantity_does_not_count_and_is_not_tier_simulated(self):
        heat = dict(FOULING_CELL, governing_quantity="heat_flux")
        b = self.score(cell=heat).breakdown
        self.assertEqual(b["evidence_tier"], "estimated")
        self.assertIn("simulated_quantity_irrelevant", b["flags"])
        self.assertEqual(b["components"]["simulated_score"], 0.0)
        self.assertIsNone(b["simulated_benefit_used_pct"])
        self.assertEqual(b["simulated_gain_pct"], 60.0)                  # shown, not scored
        self.assertEqual(b["objective_discount"], ESTIMATE_DISCOUNT)
        self.assertIn("not among the relevant quantities", b["simulated_quantity_relevance"])
        # The critic can also rule the quantity out ...
        b = self.score(simulated_quantity_relevant=False).breakdown
        self.assertEqual(b["evidence_tier"], "estimated")
        self.assertIn("critic", b["simulated_quantity_relevance"])
        # ... and without any relevance information the quantity counts.
        b = self.score(cell=heat, relevant=None).breakdown
        self.assertEqual(b["evidence_tier"], "simulated")

    def test_relabelled_analogue_is_penalised(self):
        plain = self.score()
        relabelled = self.score(relabelled_analogue=True, relabel_reason="same as parent")
        self.assertIn("relabelled_analogue", relabelled.breakdown["flags"])
        self.assertAlmostEqual(relabelled.score, plain.score * RELABEL_FACTOR, places=3)
        self.assertEqual(relabelled.breakdown["critic"]["relabel_reason"], "same as parent")

    def test_non_conventional_baseline_is_flagged(self):
        b = self.score(baseline_conventional=False, baseline_issue="parent concept").breakdown
        self.assertIn("baseline_not_conventional", b["flags"])
        self.assertEqual(b["critic"]["baseline_issue"], "parent concept")
        self.assertNotIn("baseline_not_conventional", self.score(baseline_conventional=True).breakdown["flags"])

    def test_without_critic_the_candidate_estimate_counts_half(self):
        res = score_pipeline_result(pipeline_result(60.0), candidate(estimate=4.0, cell=FOULING_CELL))
        b = res.breakdown
        self.assertEqual((b["objective_gain_pct"], b["objective_discount"]), (4.0, ESTIMATE_DISCOUNT))
        self.assertIn("objective_gain_unchecked", b["flags"])
        self.assertAlmostEqual(b["components"]["objective_score"], f(4.0, OBJECTIVE_SCALE_PCT) * ESTIMATE_DISCOUNT,
                               places=5)
        none = score_pipeline_result(pipeline_result(60.0), candidate(back_of_envelope=None, cell=FOULING_CELL))
        self.assertIn("objective_gain_missing", none.breakdown["flags"])
        self.assertEqual(none.breakdown["components"]["objective_score"], 0.0)


class DiffersStronglyTest(unittest.TestCase):
    def test_cases_from_the_role_play(self):
        self.assertTrue(differs_strongly(-28.7, -5.0))       # both negative, far apart (was not flagged)
        self.assertFalse(differs_strongly(0.02, 0.0))        # noise below 1 pp (was flagged)
        self.assertFalse(differs_strongly(0.5, 0.0))
        self.assertTrue(differs_strongly(66.7, 0.5))
        self.assertTrue(differs_strongly(30.0, 10.0))
        self.assertFalse(differs_strongly(12.0, 10.0))
        self.assertTrue(differs_strongly(5.0, -1.0))         # signs differ, |a - b| > 1 pp
        self.assertFalse(differs_strongly(0.4, -0.4))        # signs differ, but within 1 pp
        self.assertFalse(differs_strongly(-10.0, -6.0))      # 4 pp < 50 % of 10
        self.assertTrue(differs_strongly(-10.0, -4.0))


class CriticCombinationTest(unittest.TestCase):
    def score(self, simulated, plausible):
        return score_pipeline_result(pipeline_result(simulated), candidate(estimate=7.0),
                                     review=review(sim=plausible, obj=1.0), requirements=REQUIREMENTS).breakdown

    def test_lower_value_is_used_when_simulation_and_critic_differ_strongly(self):
        b = self.score(30.0, 10.0)
        self.assertEqual(b["simulated_benefit_used_pct"], 10.0)
        self.assertIn("model_assumption_sensitive", b["flags"])
        self.assertEqual((b["simulated_gain_pct"], b["critic_simulated_benefit_pct"], b["estimated_gain_pct"]),
                         (30.0, 10.0, 7.0))
        self.assertEqual(b["evidence_tier"], "simulated")
        b = self.score(5.0, 20.0)
        self.assertEqual(b["simulated_benefit_used_pct"], 5.0)
        self.assertIn("model_assumption_sensitive", b["flags"])
        b = self.score(5.0, -1.0)           # sign disagreement
        self.assertEqual(b["simulated_benefit_used_pct"], -1.0)
        self.assertIn("model_assumption_sensitive", b["flags"])
        b = self.score(-28.7, -5.0)         # both negative: now a disagreement, the lower one is used
        self.assertEqual(b["simulated_benefit_used_pct"], -28.7)
        self.assertIn("model_assumption_sensitive", b["flags"])

    def test_the_lower_value_is_used_even_when_they_agree(self):
        # v3 role-play: 8 of 13 elites used the higher simulated number (critic at 50-100 % of it).
        b = self.score(12.0, 10.0)
        self.assertEqual(b["simulated_benefit_used_pct"], 10.0)
        self.assertNotIn("model_assumption_sensitive", b["flags"])     # no strong disagreement: no flag
        self.assertIn("critic", b["simulated_benefit_source"])
        self.assertEqual(b["critic"]["key_assumption_issues"], ["gap chosen to match"])
        b = self.score(49.9, 30.0)          # the recorded pilot-whale case
        self.assertEqual(b["simulated_benefit_used_pct"], 30.0)
        b = self.score(10.0, 12.0)          # the critic above the simulation: the simulation counts
        self.assertEqual(b["simulated_benefit_used_pct"], 10.0)
        self.assertIn("not above the critic", b["simulated_benefit_source"])
        b = self.score(0.02, 0.0)           # noise is agreement
        self.assertNotIn("model_assumption_sensitive", b["flags"])
        self.assertEqual(b["simulated_benefit_used_pct"], 0.0)
        # Without a critic value the simulation counts as it is.
        b = score_pipeline_result(pipeline_result(12.0), candidate(), review=review(sim=None, obj=1.0)).breakdown
        self.assertEqual(b["simulated_benefit_used_pct"], 12.0)

    def test_without_simulated_gain_the_critic_objective_counts(self):
        b = score_pipeline_result(pipeline_result(0.0, basis="none"), candidate(estimate=15.0),
                                  review=review(obj=4.0)).breakdown
        self.assertEqual((b["evidence_tier"], b["objective_gain_pct"]), ("estimated", 4.0))
        self.assertIn("critic", b["objective_gain_source"])
        self.assertEqual(b["objective_discount"], ESTIMATE_DISCOUNT)


class FramingTest(unittest.TestCase):
    QUERY = ("Entwickle eine bionisch inspirierte Beschichtung für Schiffsrümpfe, die den Reibungswiderstand senkt, "
             "ohne giftige Antifouling-Wirkstoffe auszukommen, und mehrere Jahre im Salzwasser hält.")

    def test_service_life_requests_need_an_in_service_baseline(self):
        warnings = framing_warnings(self.QUERY, "hull friction drag", "clean standard foul-release coating")
        self.assertEqual(len(warnings), 2)
        self.assertEqual(framing_warnings(self.QUERY, OBJECTIVE, BASELINE), [])
        self.assertEqual(framing_warnings("reduce pipe friction", "friction", "smooth pipe"), [])

    def test_archive_stores_objective_and_baseline_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
            self.assertEqual(archive.set_framing("", BASELINE, 1), {"objective": False, "baseline": True})
            self.assertEqual(archive.set_framing(OBJECTIVE, "another baseline", 2), {"objective": True, "baseline": False})
            self.assertEqual((archive.objective_statement, archive.baseline_statement), (OBJECTIVE, BASELINE))
            self.assertEqual(archive.request_analysis["framing_round"], 1)
            archive.save()
            loaded = Archive.load("materials", MATERIALS_SPACE, path=archive.path)
            self.assertEqual((loaded.objective_statement, loaded.baseline_statement), (OBJECTIVE, BASELINE))


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

    def test_one_critic_call_per_batch_with_formulation_gains_objective_and_parents(self):
        batch = MaterialsCriticBatch(reviews=[
            review("r001-01", sim=6.0, obj=2.0, coverage=[("low_friction_drag", 0.8)]),
            review("r001-02", obj=40.0),
        ])
        client = FakeClient(batch)
        profile = MaterialsProfile(runner_factory=lambda: runner, critic_client=client)
        runner = Runner([pipeline_result(19.35), pipeline_result(68857.88, "warn", 0.948, sanity_passed=False)])
        ctx = EvaluationContext(query="hull coating", mock=False, requirements=REQUIREMENTS,
                                objective_statement=OBJECTIVE, baseline_statement=BASELINE,
                                relevant={"governing_quantity": ["wall_shear", "stress"]})
        items = self.items()
        items[0].order.update(strategy="diversify", context={"elite": {
            "id": "e00001", "title": "Dolphin-skin graded elastomer", "summary": "graded modulus skin",
            "cell": "mechanism_class=graded_stiffness, inspiration_origin=geology"}})
        with mock.patch.dict(os.environ, {"VECTORNAUT_MODEL_CRITIC": "critic-model"}):
            first, second = profile.evaluate(items, ctx)
        self.assertEqual(len(client.calls), 1)
        model, prompt, config = client.calls[0]
        self.assertEqual(model, "critic-model")
        self.assertIs(config.response_schema, MaterialsCriticBatch)
        self.assertNotIn("into the sea", prompt)
        for text in ("REQUEST: \"hull coating\"", "non_toxic_antifouling [must]: no biocide release", "[r001-01]", "[r001-02]",
                     "formulation: d2u_dy2 = 0", "calibrated_gap=8.8e-05", "simulated gain: 19.35 %",
                     "conventional_equivalent_gain_pct", "equal-R", "into the environment",
                     "REJECTED as implausible", "candidate's estimate: 85 %", "clean standard foul-release",
                     "surface displacement", "uniform silicone", f"OBJECTIVE (fixed for this map): {OBJECTIVE}",
                     f"CONVENTIONAL BASELINE (fixed for this map): {BASELINE}",
                     "simulated quantity: stress (named by the function analysis: wall_shear, stress)",
                     "search order: diversify; parent(s): 'Dolphin-skin graded elastomer'",
                     "plausible_simulated_benefit_pct", "plausible_objective_gain_pct", "relabelled_analogue",
                     "baseline_conventional", "inspiration: whale skin", "proxy_by_construction",
                     "baseline that contains the"):
            self.assertIn(text, prompt)
        self.assertEqual(first.breakdown["simulated_benefit_used_pct"], 6.0)   # 19.35 vs 6 -> lower, flagged
        self.assertEqual(first.breakdown["objective_gain_pct"], 2.0)
        self.assertIn("model_assumption_sensitive", first.breakdown["flags"])
        self.assertEqual(second.breakdown["evidence_tier"], "estimated")
        self.assertEqual(second.breakdown["objective_gain_pct"], 40.0)         # the critic's objective gain
        self.assertEqual(first.raw["critic_review"]["plausible_simulated_benefit_pct"], 6.0)
        self.assertEqual(first.raw["critic_review"]["plausible_objective_gain_pct"], 2.0)

    def test_irrelevant_quantity_from_the_context_reaches_the_score(self):
        client = FakeClient(MaterialsCriticBatch(reviews=[review("r001-01", sim=19.0, obj=1.0),
                                                          review("r001-02", sim=19.0, obj=1.0)]))
        runner = Runner([pipeline_result(19.0), pipeline_result(19.0)])
        profile = MaterialsProfile(runner_factory=lambda: runner, critic_client=client)
        ctx = EvaluationContext(query="q", mock=False, relevant={"governing_quantity": ["wall_shear"]})
        results = profile.evaluate(self.items(), ctx)
        self.assertEqual(results[0].breakdown["evidence_tier"], "estimated")      # stress not relevant here
        self.assertIn("simulated_quantity_irrelevant", results[0].breakdown["flags"])
        self.assertIn("NOT named by the function analysis", client.calls[0][1])

    def test_mock_critic_returns_both_numbers_and_flags_relabelled_copies(self):
        profile = MaterialsProfile()
        flagged = 0
        for i in range(12):
            order = {"order_id": f"r002-{i:02d}", "strategy": "diversify",
                     "context": {"elite": {"title": "Parent", "cell": "mechanism_class=graded_stiffness, "
                                           "inspiration_origin=plant"}}}
            item = PreparedCandidate(order=order, candidate=candidate(order["order_id"]), descriptors=dict(CELL),
                                     concept={})
            rev = profile.mock_review(item, pipeline_result(10.0), REQUIREMENTS)
            self.assertIsNotNone(rev.plausible_simulated_benefit_pct)
            self.assertIsNotNone(rev.plausible_objective_gain_pct)
            flagged += rev.relabelled_analogue
        self.assertTrue(0 < flagged < 12)

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

    def test_objective_and_conventional_baseline_reach_the_formulator_and_auditor(self):
        text = to_miner_concept(candidate(), OBJECTIVE, BASELINE).physical_mechanism
        self.assertIn(f"Objective of the request: {OBJECTIVE}.", text)
        self.assertIn("Conventional baseline for the simulation (compare against this, never against a parent or "
                      f"sibling concept): {BASELINE}.", text)

    def test_pipeline_request_carries_the_framing_from_the_context(self):
        seen = []

        class Capture:
            def run(self, request):
                seen.append(request)
                return pipeline_result(5.0)

        profile = MaterialsProfile(runner_factory=Capture)
        item = PreparedCandidate(order={"order_id": "r001-01"}, candidate=candidate(), descriptors=dict(CELL), concept={})
        profile.evaluate([item], EvaluationContext(query="q", mock=True, use_critic=False,
                                                   objective_statement=OBJECTIVE, baseline_statement=BASELINE))
        self.assertIn(BASELINE, seen[0].concept.physical_mechanism)


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
            for text in ("baseline: name the baseline", "never a fouled or untreated", "never a parent",
                         "REQUIREMENTS OF THE REQUEST: not extracted", "relevant_governing_quantities",
                         "OBJECTIVE AND BASELINE (step 1", "objective_statement", "baseline_statement",
                         "after 12-24\n  months in service", "fouling_adhesion, degradation_rate", "relabelled analogue"):
                self.assertIn(text, prompt)
            archive.set_requirements(REQUIREMENTS, 1)
            archive.add_relevant_values("governing_quantity", ["wall_shear"])
            archive.set_framing(OBJECTIVE, "", 1)
            prompt = build_prompt(MaterialsProfile(), "q", orders, archive)
            self.assertIn("- non_toxic_antifouling [must]: no biocide release", prompt)
            self.assertIn("RELEVANT GOVERNING QUANTITIES", prompt)
            self.assertIn(f"Already stored: objective = {OBJECTIVE}; baseline = (none)", prompt)
            archive.set_framing("", "clean standard foul-release coating", 2)
            archive.register_query("hull coating lasting several years")
            prompt = build_prompt(MaterialsProfile(), "q", orders, archive)
            self.assertIn("OBJECTIVE AND BASELINE (fixed for this map", prompt)
            self.assertIn(f"- objective: {OBJECTIVE}", prompt)
            self.assertIn("- baseline: clean standard foul-release coating", prompt)
            self.assertIn("Note: the request asks for years of service, but the baseline statement", prompt)

    def test_check_batch_reads_requirements_and_validates_relevant_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
            orders = [{"order_id": "r001-01", "strategy": "seed", "target": {}, "target_key": "", "context": {},
                       "rationale": "", "parent_ids": []}]
            batch = MaterialsCandidateBatch(
                function_analysis="f", requirements=[Requirement(name="low_friction_drag", criterion="c"),
                                                     Requirement(name="", criterion="dropped")],
                relevant_governing_quantities=["Wall Shear", "drag", "stress"],
                objective_statement="  time-averaged drag\n over 5 years ", baseline_statement=BASELINE,
                candidates=[candidate("r001-01")])
            result = check_batch(MaterialsProfile(), orders, batch, archive)
        self.assertEqual(result.items[0].status, "ok")
        self.assertEqual(result.batch_notes["objective_statement"], "time-averaged drag over 5 years")
        self.assertEqual(result.batch_notes["baseline_statement"], BASELINE)
        self.assertEqual(result.batch_notes["requirements"],
                         [{"name": "low_friction_drag", "criterion": "c", "priority": "must"}])
        self.assertEqual(result.batch_notes["relevant_values"], {"governing_quantity": ["wall_shear", "stress"]})
        self.assertEqual(result.batch_notes["relevant_values_rejected"], ["drag"])

    def test_missing_baseline_is_flagged_not_rejected(self):
        b = score_pipeline_result(pipeline_result(10.0), candidate(baseline="")).breakdown
        self.assertIn("baseline_unstated", b["flags"])
        self.assertEqual(MaterialsProfile().sanity_issues(candidate(baseline=""), CELL), [])


# The v3 role-play (hull coating): simulated %, critic simulated %, critic objective %, requirement coverage.
V3_CASES = {
    "glacier_soft_bed": (79.3, 55.0, 1.5, 0.425),
    "pack_ice_tiles": (78.3, 60.0, 1.5, 0.4167),
    "pilot_whale_soft_skin": (49.9, 30.0, 1.0, 0.50),
    "palm_fibres": (51.3, 35.0, 1.0, 0.4833),
    "oil_free_palm_leaching": (77.8, 60.0, 0.8, 0.45),
    "bound_hydration_leaching": (99.2, 80.0, 0.7, 0.4417),
    "graded_tie_layer": (55.3, 35.0, 0.3, 0.4917),
}
V3_PROXIES = ("oil_free_palm_leaching", "bound_hydration_leaching")


class ObjectiveDrivenScoringTest(unittest.TestCase):
    """P2/P3: the objective term dominates differences; proxies by construction lose the simulated term."""

    def score(self, name, proxy=False, scale=None):
        sim, critic_sim, objective, coverage = V3_CASES[name]
        return score_pipeline_result(pipeline_result(sim), candidate(estimate=2.0, cell=FOULING_CELL),
                                     review=review(sim=critic_sim, obj=objective, coverage=full_coverage(coverage),
                                                   proxy_by_construction=proxy,
                                                   proxy_reason="leaching flux vs an oil-containing baseline"),
                                     requirements=REQUIREMENTS, relevant_quantities=["fouling_adhesion"],
                                     objective_scale_pct=scale)

    def test_one_percent_objective_outweighs_requirement_noise_and_saturated_simulation(self):
        gain = 100 * W_OBJ * (f(1.5, OBJECTIVE_SCALE_PCT) - f(0.5, OBJECTIVE_SCALE_PCT))
        req_noise = 100 * W_REQ * (0.50 - 0.42)                 # spread of the v3 coverage ratings
        sim_spread = 100 * W_SIM * (f(80.0, SIM_SCALE_PCT) - f(25.0, SIM_SCALE_PCT))
        self.assertGreater(gain, 10.0)
        self.assertGreater(gain, 2 * req_noise)
        self.assertGreater(gain, 2 * sim_spread)

    def test_v3_ranking_tracks_the_critic_objective_gain(self):
        scores = {name: self.score(name, proxy=name in V3_PROXIES).score for name in V3_CASES}
        order = sorted(scores, key=lambda n: -scores[n])
        self.assertEqual(set(order[:2]), {"glacier_soft_bed", "pack_ice_tiles"})
        self.assertEqual(set(order[2:4]), {"pilot_whale_soft_skin", "palm_fibres"})
        for proxy in V3_PROXIES:          # the leaching proxies rank below every compliance-release concept
            self.assertLess(scores[proxy], min(scores[n] for n in order[:4]))
        self.assertLess(scores["graded_tie_layer"], min(scores[n] for n in order[:4]))   # 0.3 % objective
        # Glacier and pack-ice are a tie (< TIE_EPSILON); the tiebreak prefers pack-ice's higher critic benefit.
        self.assertLess(abs(scores["glacier_soft_bed"] - scores["pack_ice_tiles"]), TIE_EPSILON)

    def test_proxy_by_construction_zeroes_the_simulated_term(self):
        plain = self.score("bound_hydration_leaching")
        proxy = self.score("bound_hydration_leaching", proxy=True)
        b = proxy.breakdown
        self.assertIn("proxy_by_construction", b["flags"])
        self.assertEqual(b["components"]["simulated_score"], 0.0)
        self.assertIsNone(b["simulated_benefit_used_pct"])
        self.assertIn("proxy", b["simulated_benefit_source"])
        self.assertEqual(b["evidence_tier"], "simulated")            # the tier and the objective stay
        self.assertEqual(b["objective_discount"], 1.0)
        self.assertEqual(b["critic"]["proxy_reason"], "leaching flux vs an oil-containing baseline")
        self.assertAlmostEqual(plain.score - proxy.score, plain.breakdown["contributions"]["simulated"], places=3)
        self.assertEqual(b["contributions"]["simulated"], 0.0)
        self.assertFalse(MaterialsCriticReview(order_id="x").proxy_by_construction)
        self.assertIn("proxy_by_construction", MaterialsCriticReview.model_json_schema()["properties"])

    def test_tiebreak_key_is_stored(self):
        b = self.score("pack_ice_tiles").breakdown
        self.assertEqual(b["tiebreak"], [1.5, -1.0, 60.0])           # objective gain, -killer risks, sim. used
        self.assertIsNone(score_pipeline_result(pipeline_result(10.0), candidate()).breakdown["tiebreak"])


class PreferredOriginTest(unittest.TestCase):
    """P4: a bionic / bio-inspired request prefers biological origins."""

    def test_bio_keywords_in_the_request_or_requirements(self):
        profile = MaterialsProfile()
        german = profile.preferred_values(FramingTest.QUERY, [])
        self.assertEqual(german["inspiration_origin"][0], ["plant", "animal", "microbe"])
        self.assertIn("bionisch", german["inspiration_origin"][1])
        for query in ("A bio-inspired hull coating", "Biomimetic riblets", "bionic skin", "Bionik-Beschichtung"):
            self.assertIn("inspiration_origin", profile.preferred_values(query, []), query)
        req = [{"name": "bionic_mechanism", "criterion": "The mechanism is genuinely derived from a biological model."}]
        self.assertIn("bionic_mechanism", profile.preferred_values("Reduce hull drag", req)["inspiration_origin"][1])
        # Fouling is biological, but that is not a request for a biological model.
        self.assertEqual(profile.preferred_values("Reduce drag of a ship hull coating",
                                                  [{"name": "fouling_control", "criterion": "limits biological fouling"}]),
                         {})

    def test_prompt_names_the_preferred_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
            archive.set_preferred_values("inspiration_origin", ["plant", "animal", "microbe"], "the request asks ...")
            prompt = build_prompt(MaterialsProfile(), "q", [], archive)
        self.assertIn("PREFERRED INSPIRATION ORIGINS: plant, animal, microbe", prompt)


class ArchiveV3MigrationTest(unittest.TestCase):
    """Version 3 -> 4: entries are re-scored from their breakdowns and the elites rebuilt (tiebreak)."""

    def test_v3_archive_is_rescored_and_the_tie_goes_to_pack_ice(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "archive.json"))
            profile = MaterialsProfile()
            ids = {}
            for name in ("glacier_soft_bed", "pack_ice_tiles", "pilot_whale_soft_skin"):
                res = ObjectiveDrivenScoringTest.score(ObjectiveDrivenScoringTest(), name)
                breakdown = dict(res.breakdown)
                breakdown.pop("tiebreak")                      # version 3 had no tiebreak key ...
                cell = dict(FOULING_CELL, inspiration_origin="atmosphere_ocean") if name != "pilot_whale_soft_skin" \
                    else dict(FOULING_CELL)
                entry = archive.add_entry(round_no=1, run_id="t", order={"order_id": name, "strategy": "seed"},
                                          title=name, concept={}, descriptors=cell, status=EVALUATED,
                                          score=res.score, score_breakdown=breakdown)
                ids[name] = entry["id"]
            key = MATERIALS_SPACE.cell_key(dict(FOULING_CELL, inspiration_origin="atmosphere_ocean"))
            archive.data["elites"][key] = ids["glacier_soft_bed"]         # ... and kept the older entry
            for entry in archive.entries.values():                         # ... and scored with other constants
                entry["score"] = 40.0
            archive.data["version"] = 3
            archive.save()
            loaded = Archive.load("materials", MATERIALS_SPACE, path=archive.path, migrate=profile.migrate_archive)
            self.assertEqual(loaded.data["version"], ARCHIVE_VERSION)
            self.assertEqual(loaded.data["migrated_from"], 3)
            migration = loaded.data["migrations"][-1]
            self.assertEqual(migration["rescored_entries"], 3)
            self.assertEqual(migration["elite_changes"][key],
                             {"from": ids["glacier_soft_bed"], "to": ids["pack_ice_tiles"], "tiebreak": True})
            # No target was stored before version 5: the scale comes from the archive's critic objective
            # gains (1.5, 1.5, 1.0 -> 2 x median 1.5 = 3 %).
            self.assertEqual((loaded.objective_scale["pct"], loaded.objective_scale["source"]), (3.0, "archive"))
            self.assertEqual(migration["objective_scale_pct"], 3.0)
            pack = loaded.entries[ids["pack_ice_tiles"]]
            self.assertAlmostEqual(pack["score"], ObjectiveDrivenScoringTest.score(
                ObjectiveDrivenScoringTest(), "pack_ice_tiles", scale=3.0).score, places=3)
            self.assertEqual(pack["score_breakdown"]["rescored_from"]["score"], 40.0)
            self.assertEqual(pack["score_breakdown"]["tiebreak"], [1.5, -1.0, 60.0])
            self.assertIn("0.45 * objective_score", pack["score_breakdown"]["formula"])
            # Loading again does not migrate twice.
            loaded.save()
            again = Archive.load("materials", MATERIALS_SPACE, path=archive.path, migrate=profile.migrate_archive)
            self.assertEqual(len(again.data["migrations"]), 1)

    def test_rescore_uses_the_lower_number_and_keeps_the_estimated_cap(self):
        # The recorded pilot-whale elite: simulated 49.9 %, critic 30 %: version 3 used 49.9.
        res = ObjectiveDrivenScoringTest.score(ObjectiveDrivenScoringTest(), "pilot_whale_soft_skin")
        old = dict(res.breakdown, simulated_benefit_used_pct=49.9, flags=[])
        score, new = rescore_breakdown(old)
        self.assertEqual(new["simulated_benefit_used_pct"], 30.0)
        self.assertAlmostEqual(score, res.score, places=3)
        estimated = score_pipeline_result(pipeline_result(None, basis="none"), candidate(estimate=1000.0),
                                          review=review(obj=900.0, coverage=full_coverage()), requirements=REQUIREMENTS)
        self.assertEqual(rescore_breakdown(estimated.breakdown)[0], ESTIMATED_SCORE_CAP)
        self.assertIsNone(rescore_breakdown({"basis": "business"}))


if __name__ == "__main__":
    unittest.main()
