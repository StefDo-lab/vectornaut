# -*- coding: utf-8 -*-
"""
Explorer fixes from the facade-cooling role-play (2026-09-27, archive version 5):

1. objective scale per archive (stated target, else the archive's critic objective gains), with
   re-scoring so scores stay comparable within a map;
2. must/nice requirements: soft minimum and soft gate over the must-requirements, and a factor for a
   non-conventional baseline;
3. the gain of a conventional measure with the same physical effect is subtracted;
4. generator infeasibility with a scope closes a pattern; fill_gap/diversify skip pairs only reported
   infeasible;
5. mechanism classes named by the function analysis steer fill_gap/diversify/explore;
7. extrapolation diagnostics say they describe the scheduling-time state.
"""
import contextlib
import io
import math
import os
import random
import tempfile
import unittest
from unittest import mock

from vectornaut.config import ParameterProposal
from vectornaut.explorer import strategies as st
from vectornaut.explorer.archive import EVALUATED, Archive, normalize_priority
from vectornaut.explorer.descriptors import NOMINAL, ORDINAL, Axis, DescriptorSpace
from vectornaut.explorer.generator import build_prompt, check_batch
from vectornaut.explorer.profiles.materials import (
    BASELINE_FACTOR, MATERIALS_SPACE, OBJECTIVE_SCALE_PCT, MaterialsProfile, archive_objective_gains,
    net_objective_gain, objective_scale_for, requirement_terms, rescore_archive, rescore_breakdown,
    score_pipeline_result,
)
from vectornaut.explorer.run import ExplorerRunner
from vectornaut.explorer.schemas import (
    BackOfEnvelope, DescriptorAssignment, MaterialsCandidate, MaterialsCandidateBatch, MaterialsCriticReview,
    Requirement, RequirementRating,
)

FROZEN = "2026-01-01T00:00:00Z"
HEAT_CELL = {"mechanism_class": "radiative_control", "length_scale": "sub_um", "inspiration_origin": "animal",
             "governing_quantity": "heat_flux"}
FACADE_REQUIREMENTS = [
    {"name": "indoor_heat_gain_reduction", "criterion": ">= 20 % less heat gain than an aged cool paint"},
    {"name": "passive_no_power_no_water", "criterion": "no electricity or water"},
    {"name": "weather_durability_20y", "criterion": "20 years outdoors"},
    {"name": "bionic_mechanism", "criterion": "mechanism derived from a biological model"},
    {"name": "building_practicality", "criterion": "cost, weight, fire safety", "priority": "nice"},
]
# The facade baseline run: Cork-oak cladding (e00012) and the Cyphochilus-scale white coat (e00001):
# simulated %, critic simulated %, critic objective %, ratings in FACADE_REQUIREMENTS order.
CORK = (44.47, 40.0, 35.0, (0.9, 1.0, 0.7, 0.15, 0.55))
CYPHOCHILUS = (35.58, 25.0, 17.0, (0.65, 1.0, 0.5, 0.55, 0.7))


def f(gain, scale):
    return 1 - math.exp(-gain / scale) if gain > 0 else 0.0


def candidate(order_id="r001-01", cell=None, estimate=10.0, **changes):
    fields = dict(
        order_id=order_id, title=f"Concept {order_id}", summary="white scattering coat",
        descriptors=[DescriptorAssignment(axis=k, value=v) for k, v in (cell or HEAT_CELL).items()],
        back_of_envelope=BackOfEnvelope(quantity="heat gain", formula="dq = dR * I", value=estimate, unit="%"),
        main_risk="soiling", novelty_vs_known="n", baseline="aged white cool paint",
        inspiration_source="beetle scales", domain="Heat Transfer", physical_mechanism="scattering",
        parameters=[ParameterProposal(name="k", value=0.2, min_bound=0.01, max_bound=1.0, justification="j")],
    )
    fields.update(changes)
    return MaterialsCandidate(**fields)


def pipeline_result(gain):
    return {"status": "completed",
            "simulator": {"performance_gain_pct": gain, "gain_basis": "baseline_parameters", "solver_method": "scipy"},
            "validation": {"status": "pass", "score": 1.0, "checks": []}, "miner": {}, "auditor": {}}


def review(sim=None, obj=None, ratings=(), requirements=FACADE_REQUIREMENTS, **changes):
    fields = dict(order_id="r001-01", plausible_simulated_benefit_pct=sim, plausible_objective_gain_pct=obj,
                  killer_risks=["soiling"],
                  requirement_coverage=[RequirementRating(name=r["name"], coverage=c)
                                        for r, c in zip(requirements, ratings)])
    fields.update(changes)
    return MaterialsCriticReview(**fields)


def score(case, scale=None, requirements=FACADE_REQUIREMENTS, **review_changes):
    sim, critic_sim, obj, ratings = case
    return score_pipeline_result(pipeline_result(sim), candidate(),
                                 review=review(critic_sim, obj, ratings, requirements, **review_changes),
                                 requirements=requirements, relevant_quantities=["heat_flux", "temperature"],
                                 objective_scale_pct=scale)


class ObjectiveScaleTest(unittest.TestCase):
    def test_sources_and_bounds(self):
        self.assertEqual(objective_scale_for(20.0, [1.0, 2.0, 3.0])[:2], (10.0, "target"))
        self.assertIn("reaching the target scores 0.86", objective_scale_for(20.0, [])[2])
        # The hull-coating archive: median 0.8 % -> 1.6 %, bounded to the version-4 value of 2 %.
        self.assertEqual(objective_scale_for(None, [0.3, 0.5, 0.8, 1.0, 1.5, -0.5])[:2], (2.0, "archive"))
        # The facade guided archive: median 10 % -> 20 %.
        facade = [20, 8, 12, 10, 12, 7, 3, 5, 18, 1, 18, 5, 12]
        self.assertEqual(objective_scale_for(None, facade)[:2], (20.0, "archive"))
        self.assertEqual(objective_scale_for(None, [5.0, 10.0, 0.0, -3.0])[:2], (OBJECTIVE_SCALE_PCT, "default"))
        self.assertEqual(objective_scale_for(None, [400.0] * 5)[0], 50.0)
        self.assertEqual(objective_scale_for(0.1, [])[0], 0.5)
        self.assertEqual(objective_scale_for(None, [3.3, 3.3, 3.3])[0], 6.6)

    def test_facade_gains_no_longer_saturate(self):
        gains = (35.0, 20.0, 12.0, 5.0)
        at_v4 = [score((40.0, 30.0, g, (0.7,) * 5), scale=2.0).breakdown["contributions"]["objective"] for g in gains]
        at_v5 = [score((40.0, 30.0, g, (0.7,) * 5), scale=20.0).breakdown["contributions"]["objective"] for g in gains]
        self.assertLess(at_v4[0] - at_v4[2], 0.2)               # 35 % vs 12 %: the same 45 points at 2 %
        self.assertGreater(at_v5[0] - at_v5[2], 15.0)           # now 36 vs 20 points
        self.assertGreater(at_v5[1] - at_v5[3], 10.0)
        b = score((40.0, 30.0, 20.0, (0.7,) * 5), scale=20.0).breakdown
        self.assertEqual(b["scales_pct"]["objective"], 20.0)
        self.assertIn("objective scale 20 %", b["formula"])

    def test_archive_is_rescored_when_the_scale_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
            archive.set_requirements(FACADE_REQUIREMENTS, 1)
            ids = []
            for i, obj in enumerate((20.0, 10.0, 5.0)):
                res = score((40.0, 30.0, obj, (0.7,) * 5))                     # scored at the default 2 %
                cell = dict(HEAT_CELL, length_scale=MATERIALS_SPACE.axis("length_scale").values[i])
                ids.append(archive.add_entry(round_no=1, run_id="t", order={"order_id": f"o{i}", "strategy": "seed"},
                                             title=f"c{i}", concept={}, descriptors=cell, status=EVALUATED,
                                             score=res.score, score_breakdown=res.breakdown)["id"])
            self.assertEqual(archive_objective_gains(archive), [20.0, 10.0, 5.0])
            summary = rescore_archive(archive, round_no=1)
            self.assertEqual((summary["objective_scale_pct"], summary["objective_scale_source"]), (20.0, "archive"))
            self.assertEqual(summary["rescored_entries"], 3)
            for entry_id, obj in zip(ids, (20.0, 10.0, 5.0)):
                expected = score((40.0, 30.0, obj, (0.7,) * 5), scale=20.0).score
                self.assertAlmostEqual(archive.entries[entry_id]["score"], expected, places=3)
                self.assertEqual(archive.entries[entry_id]["score_breakdown"]["objective_scale_source"], "archive")
            self.assertEqual(archive.objective_scale["history"], [{"pct": 20.0, "source": "archive", "round": 1}])
            self.assertIsNone(rescore_archive(archive, round_no=2))              # unchanged: nothing to do
            # A stated target takes over.
            self.assertTrue(archive.set_target_gain(20.0, 3))
            self.assertFalse(archive.set_target_gain(30.0, 4))                    # stored once
            summary = MaterialsProfile().update_scoring(archive, 3)
            self.assertEqual((summary["objective_scale_pct"], summary["objective_scale_source"]), (10.0, "target"))
            self.assertEqual(summary["previous_objective_scale_pct"], 20.0)
            archive.save()
            loaded = Archive.load("materials", MATERIALS_SPACE, path=archive.path)
            self.assertEqual(loaded.objective_scale["pct"], 10.0)
            self.assertEqual(loaded.target_gain_pct, 20.0)
            text = MaterialsProfile().score_note(loaded)
            self.assertIn("Objective scale 10 % (target", text)


class ConventionalEquivalentTest(unittest.TestCase):
    def test_net_gain_and_flag(self):
        self.assertEqual(net_objective_gain(35.0, 30.0), (5.0, True))
        self.assertEqual(net_objective_gain(35.0, 5.0), (30.0, False))
        self.assertEqual(net_objective_gain(10.0, 12.0), (0.0, True))
        self.assertEqual(net_objective_gain(10.0, None), (10.0, False))
        self.assertEqual(net_objective_gain(-2.0, 3.0), (-2.0, False))
        self.assertEqual(net_objective_gain(None, 3.0), (None, False))

    def test_insulation_in_disguise_scores_its_own_contribution_only(self):
        plain = score(CORK, scale=12.0)
        disguised = score(CORK, scale=12.0, conventional_equivalent_gain_pct=30.0)
        b = disguised.breakdown
        self.assertEqual((b["critic_objective_gain_pct"], b["conventional_equivalent_gain_pct"], b["objective_gain_pct"]),
                         (35.0, 30.0, 5.0))
        self.assertIn("mostly_conventional_effect", b["flags"])
        self.assertIn("net of the conventional equivalent", b["objective_gain_source"])
        self.assertEqual(b["critic"]["conventional_equivalent_gain_pct"], 30.0)
        self.assertEqual(b["tiebreak"][0], 5.0)
        self.assertLess(disguised.score, plain.score - 15.0)
        self.assertNotIn("mostly_conventional_effect", plain.breakdown["flags"])
        self.assertIsNone(plain.breakdown["conventional_equivalent_gain_pct"])
        self.assertIn("conventional_equivalent_gain_pct", MaterialsCriticReview.model_json_schema()["properties"])
        # The archive's scale is derived from the net gains.
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
            archive.add_entry(round_no=1, run_id="t", order={"order_id": "o", "strategy": "seed"}, title="t",
                              concept={}, descriptors=dict(HEAT_CELL), status=EVALUATED, score=disguised.score,
                              score_breakdown=disguised.breakdown)
            self.assertEqual(archive_objective_gains(archive), [5.0])

    def test_mock_critic_reports_a_conventional_equivalent(self):
        from vectornaut.explorer.profiles.base import PreparedCandidate
        cell = dict(HEAT_CELL, mechanism_class="trapped_gas_or_liquid", governing_quantity="wall_shear")
        item = PreparedCandidate(order={"order_id": "r001-01"}, candidate=candidate(cell=cell), descriptors=cell,
                                 concept={})
        rev = MaterialsProfile.mock_review(item, pipeline_result(10.0), FACADE_REQUIREMENTS)
        self.assertAlmostEqual(rev.conventional_equivalent_gain_pct, 0.6 * rev.plausible_objective_gain_pct, places=3)


class MustRequirementTest(unittest.TestCase):
    def test_priorities(self):
        self.assertEqual([normalize_priority(v) for v in (None, "", "must", "MUST", "nice", "Nice-to-have",
                                                           "optional", "should", "hard")],
                         ["must", "must", "must", "must", "nice", "nice", "nice", "nice", "must"])
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
            archive.set_requirements(FACADE_REQUIREMENTS, 1)
            self.assertEqual([r["priority"] for r in archive.requirements], ["must"] * 4 + ["nice"])

    def test_soft_minimum_and_gate(self):
        rows = [{"name": "a", "coverage": 0.9}, {"name": "b", "coverage": 0.9},
                {"name": "c", "coverage": 0.1, "priority": "nice"}]
        terms = requirement_terms(rows)
        self.assertAlmostEqual(terms["coverage"], 1.9 / 3)
        self.assertEqual((terms["must_min"], terms["must_factor"], terms["must_unmet"]), (0.9, 1.0, []))
        self.assertAlmostEqual(terms["score"], 0.5 * 1.9 / 3 + 0.5 * 0.9)
        rows[2]["priority"] = "must"
        terms = requirement_terms(rows)
        self.assertEqual(terms["must_min"], 0.1)
        self.assertAlmostEqual(terms["must_factor"], 1 - 0.5 * 0.2 / 0.3)
        self.assertEqual(terms["must_unmet"], ["c"])
        self.assertEqual(requirement_terms([{"name": "a", "coverage": 0.0}])["must_factor"], 0.5)
        self.assertEqual(requirement_terms([{"name": "a", "coverage": 0.3}])["must_factor"], 1.0)
        # Unrated requirements count 0.5, and no ratings at all keep the old 0.5.
        self.assertEqual(requirement_terms([{"name": "a", "coverage": None}])["score"], 0.5)
        self.assertEqual(requirement_terms([])["score"], 0.5)

    def test_non_bionic_concept_no_longer_wins_a_bionic_request(self):
        # Facade baseline run: the cork cladding (critic 35 %, bionic 0.15, non-conventional baseline) ranked
        # first under version 4; the archive's scale was 12 % (2 x median 6 %).
        cork = score(CORK, scale=12.0, baseline_conventional=False, baseline_issue="uninsulated wall")
        cyph = score(CYPHOCHILUS, scale=12.0)
        self.assertGreater(cyph.score, cork.score + 10.0)
        b = cork.breakdown
        self.assertEqual(b["must_min_coverage"], 0.15)
        self.assertAlmostEqual(b["must_factor"], 0.75)
        self.assertEqual(b["baseline_factor"], BASELINE_FACTOR)
        self.assertIn("must_requirement_unmet", b["flags"])
        self.assertIn("baseline_not_conventional", b["flags"])
        # The same concept with a met bionic requirement and a conventional baseline would still win.
        honest = score((44.47, 40.0, 35.0, (0.9, 1.0, 0.7, 0.6, 0.55)), scale=12.0)
        self.assertGreater(honest.score, cyph.score)

    def test_non_conventional_baseline_costs_twenty_percent(self):
        plain = score(CYPHOCHILUS, scale=12.0)
        flagged = score(CYPHOCHILUS, scale=12.0, baseline_conventional=False)
        self.assertAlmostEqual(flagged.score, plain.score * BASELINE_FACTOR, places=3)

    def test_rescoring_matches_scoring_and_old_breakdowns_migrate(self):
        res = score(CORK, scale=12.0, conventional_equivalent_gain_pct=20.0, baseline_conventional=False)
        again, _ = rescore_breakdown(res.breakdown, FACADE_REQUIREMENTS, 12.0, "archive")
        self.assertAlmostEqual(again, res.score, places=4)
        # A version-4 breakdown: no conventional equivalent, no priorities, old component names.
        v4 = score(CORK, scale=12.0, baseline_conventional=False).breakdown
        old = dict(v4, components=dict(v4["components"], requirement_coverage=v4["requirement_coverage"]),
                   requirements=[{k: v for k, v in r.items() if k != "priority"} for r in v4["requirements"]])
        for key in ("conventional_equivalent_gain_pct", "requirement_score", "must_factor", "baseline_factor",
                    "must_min_coverage"):
            old.pop(key)
        old["components"].pop("requirement_score")
        old["critic"] = {k: v for k, v in old["critic"].items() if k != "conventional_equivalent_gain_pct"}
        new_score, new = rescore_breakdown(old, FACADE_REQUIREMENTS, 12.0, "archive")
        self.assertAlmostEqual(new_score, score(CORK, scale=12.0, baseline_conventional=False).score, places=4)
        self.assertEqual(new["requirements"][-1]["priority"], "nice")          # from the stored requirement list


class GeneratorFieldsTest(unittest.TestCase):
    def test_batch_notes_and_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(tmp, "a.json"))
            archive.declare_relevance_axes(["governing_quantity"], soft=["mechanism_class"])
            orders = [{"order_id": "r001-01", "strategy": "seed", "target": {}, "target_key": "", "context": {},
                       "rationale": "", "parent_ids": []}]
            prompt = build_prompt(MaterialsProfile(), "q", orders, archive)
            for text in ("relevant_mechanism_classes", "target_gain_pct", "infeasibility_scope", "priority 'must'",
                         "'nice' for implied"):
                self.assertIn(text, prompt)
            batch = MaterialsCandidateBatch(
                requirements=[Requirement(name="bionic_mechanism", criterion="c"),
                              Requirement(name="low_cost", criterion="c", priority="nice")],
                relevant_mechanism_classes=["Radiative Control", "levitation", "phase_change"],
                relevant_governing_quantities=["heat_flux"], target_gain_pct=20.0,
                candidates=[candidate("r001-01")])
            result = check_batch(MaterialsProfile(), orders, batch, archive)
            self.assertEqual(result.batch_notes["target_gain_pct"], 20.0)
            self.assertEqual([r["priority"] for r in result.batch_notes["requirements"]], ["must", "nice"])
            self.assertEqual(result.batch_notes["relevant_values"],
                             {"governing_quantity": ["heat_flux"], "mechanism_class": ["radiative_control", "phase_change"]})
            self.assertEqual(result.batch_notes["relevant_values_rejected"], ["levitation"])
            archive.set_requirements(result.batch_notes["requirements"], 1)
            archive.add_relevant_values("mechanism_class", ["radiative_control"])
            archive.set_target_gain(20.0, 1)
            archive.set_framing("heat gain over 20 years", "aged cool paint after 3 years", 1)
            prompt = build_prompt(MaterialsProfile(), "q", orders, archive)
            for text in ("- low_cost [nice]: c", "RELEVANT MECHANISM CLASSES", "radiative_control",
                         "- target gain: 20 % (stored"):
                self.assertIn(text, prompt)

    def test_candidate_schema_has_the_scope(self):
        c = MaterialsCandidate(order_id="x", target_feasible=False, infeasibility_reason="Stokes flow",
                               infeasibility_scope=["mechanism_class", "length_scale"])
        self.assertEqual(c.infeasibility_scope, ["mechanism_class", "length_scale"])
        self.assertEqual(MaterialsCandidate(order_id="x").infeasibility_scope, [])


SPACE = DescriptorSpace([
    Axis("mech", NOMINAL, ("a", "b", "c")),
    Axis("size", ORDINAL, ("s1", "s2", "s3", "s4", "s5")),
    Axis("origin", NOMINAL, ("x", "y")),
])


def cell(mech, size, origin):
    return {"mech": mech, "size": size, "origin": origin}


class StrategyTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.archive = Archive("test", SPACE, path=os.path.join(tmp.name, "a.json"), clock=lambda: FROZEN)
        self.config = st.StrategyConfig(compatibility_axes=("mech", "size"), origin_axes=("origin",))
        self.n = 0

    def add(self, descriptors, score=50.0, status=EVALUATED):
        self.n += 1
        return self.archive.add_entry(round_no=1, run_id="t", title=f"c{self.n}", concept={},
                                      order={"order_id": f"o{self.n}", "strategy": "seed", "target": descriptors},
                                      descriptors=descriptors, status=status, score=score)


class InfeasibilityScopeTest(StrategyTestCase):
    def test_pattern_from_the_scope(self):
        target = cell("b", "s1", "x")
        self.assertEqual(self.archive.infeasibility_pattern(target, ["mech+size"])[:2],
                         ({"mech": "b", "size": "s1"}, ["mech", "size"]))
        pattern, used, note = self.archive.infeasibility_pattern(target, ["size"])
        self.assertEqual((pattern, used), (target, []))
        self.assertIn("fewer than 2", note)
        pattern, used, note = self.archive.infeasibility_pattern(target, ["mech", "Size", "colour"])
        self.assertEqual(pattern, {"mech": "b", "size": "s1"})
        self.assertIn("unknown scope axes ignored: colour", note)
        self.assertEqual(self.archive.infeasibility_pattern(target, [])[:2], (target, []))
        self.assertEqual(self.archive.infeasibility_pattern(target, ["mech", "size", "origin"])[:2],
                         (target, ["mech", "size", "origin"]))
        self.archive.mark_infeasible({"mech": "b", "size": "s1"}, "Stokes flow", entry_id=None, round_no=1,
                                     source="generator", scope=["mech", "size"], note="generalised")
        self.assertTrue(self.archive.is_infeasible(cell("b", "s1", "y")))
        key = SPACE.cell_key({"mech": "b", "size": "s1"})
        self.assertEqual(self.archive.data["infeasible"][key]["scope"], ["mech", "size"])

    def test_pairs_only_reported_infeasible_are_not_targeted_by_fill_gap_or_diversify(self):
        # The facade case: flow_redirection/sub_um/plant reported infeasible for the exact cell; the next
        # round's diversify targeted flow_redirection/sub_um with another origin.
        self.add(cell("a", "s1", "x"), 60.0)
        self.add(cell("b", "s2", "x"), 55.0)
        self.add(cell("b", "s2", "y"), 54.0)
        self.archive.mark_infeasible(cell("b", "s1", "x"), "Stokes flow", entry_id=None, round_no=1, source="generator")
        gaps = st.gap_candidates(self.archive, config=self.config)
        self.assertFalse([g for g in gaps if (g["cell"]["mech"], g["cell"]["size"]) == ("b", "s1")])
        for seed in range(10):
            orders = st.diversify(self.archive, random.Random(seed), 6, set(), self.config)
            orders += st.fill_gap(self.archive, random.Random(seed), 10, set(), self.config)
            for order in orders:
                self.assertNotEqual((order["target"]["mech"], order["target"]["size"]), ("b", "s1"), order)
        # Without the compatibility axes the other origin is still a gap.
        plain = st.gap_candidates(self.archive, config=st.StrategyConfig())
        self.assertTrue([g for g in plain if g["cell"] == cell("b", "s1", "y")])

    def test_runner_stores_the_generalised_pattern(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VECTORNAUT_DATA_DIR": tmp}):
            profile = MaterialsProfile()
            runner = ExplorerRunner(profile, "q", mock=True, clock=lambda: FROZEN)
            target = {"mechanism_class": "flow_redirection", "length_scale": "sub_um", "inspiration_origin": "plant",
                      "governing_quantity": "heat_flux"}
            order = {"order_id": "r001-01", "strategy": "diversify", "target": target,
                     "target_key": MATERIALS_SPACE.cell_key(target), "context": {}, "rationale": "", "parent_ids": []}
            answer = MaterialsCandidate(order_id="r001-01", target_feasible=False,
                                        infeasibility_reason="Stokes flow in the viscous sublayer",
                                        infeasibility_scope=["mechanism_class", "length_scale"])
            with mock.patch("vectornaut.explorer.run.schedule", return_value=[order]), \
                    mock.patch.object(profile, "mock_candidate", return_value=answer), \
                    contextlib.redirect_stdout(io.StringIO()):
                record = runner.run_round(1, "run-001")
            archive = runner.archive
            key = MATERIALS_SPACE.cell_key({"mechanism_class": "flow_redirection", "length_scale": "sub_um"})
            self.assertIn(key, archive.data["infeasible"])
            self.assertTrue(archive.is_infeasible(dict(target, inspiration_origin="animal", governing_quantity="temperature")))
            self.assertIn("generalised to mechanism_class+length_scale", record["orders"][0]["note"])
            entry = archive.entries[record["orders"][0]["entry_id"]]
            self.assertEqual(entry["score_breakdown"]["infeasible_pattern"], key)
            runner.write_reports(record)
            with open(os.path.join(runner.out_dir, "map.md"), encoding="utf-8") as fh:
                self.assertIn("mechanism_class+length_scale", fh.read())


class MechanismRelevanceTest(StrategyTestCase):
    def setUp(self):
        super().setUp()
        self.archive.declare_relevance_axes([], soft=["mech"])

    def test_nothing_named_means_no_restriction(self):
        self.add(cell("a", "s2", "x"))
        self.assertEqual(self.archive.soft_relevance_axes(), ["mech"])
        self.assertEqual(self.archive.relevance_axes(), [])
        self.assertTrue(self.archive.is_soft_relevant(cell("c", "s2", "x")))
        self.assertTrue([g for g in st.gap_candidates(self.archive, config=self.config) if g["cell"]["mech"] != "a"])

    def test_fill_gap_diversify_and_explore_follow_the_named_mechanisms(self):
        self.archive.add_relevant_values("mech", ["a"])
        self.add(cell("a", "s2", "x"), 60.0)
        self.add(cell("a", "s4", "y"), 50.0)
        gaps = st.gap_candidates(self.archive, config=self.config)
        self.assertTrue(gaps)
        self.assertEqual({g["cell"]["mech"] for g in gaps}, {"a"})
        for seed in range(5):
            for order in st.diversify(self.archive, random.Random(seed), 6, set(), self.config):
                self.assertEqual(order["target"]["mech"], "a", order)
        picks = []
        for seed in range(200):
            orders = st.explore(self.archive, random.Random(seed), 1, set(), self.config)
            picks.append(orders[0])
        outside = [o for o in picks if o["target"]["mech"] != "a"]
        self.assertTrue(outside)                                  # explore still reaches the others ...
        self.assertLess(len(outside) / len(picks), 0.4)           # ... at a low weight (0.1 per cell)
        for order in outside:
            self.assertEqual(order["context"]["outside_relevant"], [f"mech={order['target']['mech']}"])
            self.assertIn("was not named as relevant", order["rationale"])

    def test_top_elites_widen_the_relevant_mechanisms(self):
        self.archive.add_relevant_values("mech", ["a"])
        self.add(cell("a", "s2", "x"), 60.0)
        self.add(cell("b", "s4", "x"), 70.0)              # a strong elite with another mechanism
        self.assertEqual(self.archive.relevant_values("mech"), ["a", "b"])
        self.assertTrue(self.archive.is_soft_relevant(cell("b", "s5", "x")))
        self.assertFalse(self.archive.is_soft_relevant(cell("c", "s5", "x")))
        for i in range(5):                                 # five better elites push it out of the top five
            self.add(cell("a", "s1", "y") if i == 0 else cell("a", ["s3", "s5", "s4", "s2"][i - 1], "y"), 80.0 + i)
        self.assertEqual(self.archive.relevant_values("mech"), ["a"])

    def test_map_report_names_the_relevant_mechanisms(self):
        from vectornaut.explorer import report
        archive = Archive("materials", MATERIALS_SPACE, path=os.path.join(self.archive.directory, "m.json"))
        archive.declare_relevance_axes(["governing_quantity"], soft=["mechanism_class"])
        archive.add_relevant_values("mechanism_class", ["radiative_control"])
        archive.set_requirements(FACADE_REQUIREMENTS, 1)
        archive.set_objective_scale(20.0, "archive", "2 x median 10 %", 1)
        archive.set_framing("heat gain over 20 years", "aged cool paint", 1)
        text = report.render_map(archive, MaterialsProfile(), "q")
        self.assertIn("Relevant `mechanism_class` values (fill-gap/diversify use only these", text)
        self.assertIn("**building_practicality** [nice]", text)
        self.assertIn("- **Objective scale**: 20 % (archive: 2 x median 10 %)", text)
        self.assertIn("objective scale 20 % (archive)", text)
        self.assertIn("(not relevant to the request, only explore, at a low weight", text)


class SchedulingTimeWordingTest(StrategyTestCase):
    def test_no_trend_summary_names_the_scheduling_time_state(self):
        summary = st.explain_trends(self.archive)["summary"]
        self.assertIn("at scheduling time (before this round's candidates)", summary)
        self.assertIn("0 trend-eligible (simulated) elite(s)", summary)


class MaterialsLoopV5Test(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        patcher = mock.patch.dict(os.environ, {"VECTORNAUT_DATA_DIR": tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_scores_stay_comparable_within_the_map(self):
        runner = ExplorerRunner(MaterialsProfile(), "Reduce drag of a ship hull coating", seed=1, mock=True, epochs=5,
                                clock=lambda: FROZEN)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run(3, 4)
        archive = runner.archive
        scale = archive.objective_scale
        self.assertIn(scale["source"], ("archive", "default"))
        self.assertEqual(runner.ctx.objective_scale_pct, scale["pct"])
        for entry in archive.entries.values():
            if entry["status"] != EVALUATED:
                continue
            b = entry["score_breakdown"]
            self.assertEqual(b["scales_pct"]["objective"], scale["pct"])
            again, _ = rescore_breakdown(b, archive.requirements, scale["pct"], scale["source"])
            self.assertAlmostEqual(again, entry["score"], places=3)
        for record in archive.data["rounds"]:
            self.assertEqual(record["objective_scale"]["pct"] > 0, True)
        # Mechanism classes from the (mock) function analysis steer the targeted strategies.
        relevant = set(archive.relevant_values("mechanism_class"))
        self.assertTrue({"interfacial_slip", "flow_redirection"} <= relevant)
        for entry in archive.entries.values():
            if entry["strategy"] in ("fill_gap", "diversify"):
                mech = archive.space.parse_key(entry["target_key"]).get("mechanism_class")
                self.assertIn(mech, relevant | {None}, entry["target_key"])
        with open(os.path.join(archive.directory, "reports", "map.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("- **Objective scale**:", text)
        self.assertIn("min must", text)

    def test_a_stated_target_sets_the_scale_in_the_first_round(self):
        profile = MaterialsProfile()
        original = profile.mock_batch

        def with_target(query, orders):
            batch = original(query, orders)
            batch.target_gain_pct = 20.0
            return batch

        runner = ExplorerRunner(profile, "q", seed=1, mock=True, epochs=5, clock=lambda: FROZEN)
        with mock.patch.object(profile, "mock_batch", side_effect=with_target), contextlib.redirect_stdout(io.StringIO()):
            record = runner.run_round(3, "run-001")
        archive = runner.archive
        self.assertEqual(archive.target_gain_pct, 20.0)
        self.assertEqual((archive.objective_scale["pct"], archive.objective_scale["source"]), (10.0, "target"))
        self.assertTrue(record["request_analysis_update"]["target_stored"])
        for entry in archive.entries.values():
            if entry["status"] == EVALUATED:
                self.assertEqual(entry["score_breakdown"]["scales_pct"]["objective"], 10.0)
                self.assertEqual(entry["score_breakdown"]["objective_scale_source"], "target")


if __name__ == "__main__":
    unittest.main()
