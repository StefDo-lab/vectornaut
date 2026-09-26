# -*- coding: utf-8 -*-
"""Explorer descriptors, archive and search strategies (no model calls)."""
import json
import os
import random
import tempfile
import unittest

from vectornaut.explorer.archive import (
    ARCHIVE_VERSION, EVALUATED, FAILED, IMPROVED, INVALID, NEW_ELITE, NOT_BETTER, Archive, ArchiveCompatibilityError,
)
from vectornaut.explorer.descriptors import NOMINAL, ORDINAL, Axis, DescriptorError, DescriptorSpace
from vectornaut.explorer.profiles.business import BUSINESS_SPACE
from vectornaut.explorer.profiles.materials import MATERIALS_SPACE
from vectornaut.explorer.schemas import DescriptorAssignment
from vectornaut.explorer import strategies as st

SPACE = DescriptorSpace([
    Axis("mech", NOMINAL, ("a", "b", "c")),
    Axis("size", ORDINAL, ("s1", "s2", "s3", "s4", "s5")),
    Axis("origin", NOMINAL, ("x", "y")),
])


def cell(mech, size, origin):
    return {"mech": mech, "size": size, "origin": origin}


class ArchiveTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.archive = self.new_archive()
        self._n = 0

    def new_archive(self, space=SPACE, name="archive.json"):
        return Archive("test", space, path=os.path.join(self._tmp.name, name), clock=lambda: "2026-01-01T00:00:00Z")

    def add(self, descriptors, score, strategy="seed", status=EVALUATED, parents=(), archive=None, breakdown=None):
        self._n += 1
        archive = archive or self.archive
        return archive.add_entry(
            round_no=1, run_id="t", title=f"concept {self._n}", concept={"summary": f"idea {self._n}"},
            order={"order_id": f"o{self._n}", "strategy": strategy, "target": descriptors, "parent_ids": list(parents)},
            descriptors=descriptors, status=status, score=score, score_breakdown=breakdown,
        )


SIMULATED = {"evidence_tier": "simulated", "evidence_rank": 2}
ESTIMATED = {"evidence_tier": "estimated", "evidence_rank": 1}
IMPLAUSIBLE = {"evidence_tier": "estimated", "evidence_rank": 0, "flags": ["implausible_gain"]}


class DescriptorTest(unittest.TestCase):
    def test_normalisation_is_case_and_spacing_tolerant(self):
        result = MATERIALS_SPACE.validate({
            "Mechanism Class": "Interfacial Slip",
            "length_scale": " 10 um ",
            "inspiration-origin": "PLANT",
            "governing_quantity": "wall-shear",
        })
        self.assertEqual(result, {
            "mechanism_class": "interfacial_slip", "length_scale": "10_um",
            "inspiration_origin": "plant", "governing_quantity": "wall_shear",
        })
        self.assertEqual(MATERIALS_SPACE.axis("length_scale").normalize("10 µm"), "10_um")

    def test_unknown_value_is_rejected_with_reason_not_guessed(self):
        with self.assertRaises(DescriptorError) as ctx:
            MATERIALS_SPACE.validate({
                "mechanism_class": "slippery", "length_scale": "mm",
                "inspiration_origin": "plant", "governing_quantity": "wall_shear",
            })
        self.assertIn("mechanism_class: 'slippery' is not an allowed value", str(ctx.exception))
        self.assertIn("interfacial_slip", str(ctx.exception))

    def test_missing_unknown_and_duplicate_axes_are_problems(self):
        with self.assertRaises(DescriptorError) as ctx:
            SPACE.validate({"mech": "a", "colour": "red"})
        problems = ctx.exception.problems
        self.assertTrue(any("unknown axis 'colour'" in p for p in problems))
        self.assertIn("size: missing", problems)
        self.assertIn("origin: missing", problems)
        with self.assertRaises(DescriptorError) as ctx:
            SPACE.validate_pairs([
                DescriptorAssignment(axis="mech", value="a"), DescriptorAssignment(axis="mech", value="b"),
                DescriptorAssignment(axis="size", value="s1"), DescriptorAssignment(axis="origin", value="x"),
            ])
        self.assertTrue(any("assigned twice" in p for p in ctx.exception.problems))

    def test_partial_validation_and_pairs(self):
        self.assertEqual(SPACE.validate({"Size": "S3"}, partial=True), {"size": "s3"})
        self.assertEqual(
            SPACE.validate_pairs([{"axis": "origin", "value": "Y"}, {"axis": "mech", "value": "c"},
                                  {"axis": "size", "value": "s2"}]),
            {"mech": "c", "size": "s2", "origin": "y"},
        )

    def test_cell_keys_patterns_and_matching(self):
        key = SPACE.cell_key(cell("a", "s2", "x"))
        self.assertEqual(key, "mech=a|size=s2|origin=x")
        self.assertEqual(SPACE.parse_key(key), cell("a", "s2", "x"))
        pattern_key = SPACE.cell_key({"size": "s2"})
        self.assertEqual(pattern_key, "mech=*|size=s2|origin=*")
        self.assertEqual(SPACE.parse_key(pattern_key), {"size": "s2"})
        self.assertTrue(SPACE.matches(cell("c", "s2", "y"), {"size": "s2"}))
        self.assertFalse(SPACE.matches(cell("c", "s3", "y"), {"size": "s2"}))

    def test_distance_and_neighbours(self):
        self.assertEqual(SPACE.distance(cell("a", "s1", "x"), cell("b", "s4", "y")), 1 + 3 + 1)
        neighbours = SPACE.neighbours(cell("a", "s1", "x"))
        # mech: 2 other values; size: only s2 (s1 is the end of the scale); origin: 1 other value.
        self.assertEqual(len(neighbours), 4)
        self.assertIn(cell("a", "s2", "x"), neighbours)
        self.assertNotIn(cell("a", "s3", "x"), neighbours)
        self.assertEqual(len(SPACE.neighbours(cell("b", "s3", "y"))), 2 + 2 + 1)
        self.assertTrue(all(SPACE.distance(cell("b", "s3", "y"), n) == 1 for n in SPACE.neighbours(cell("b", "s3", "y"))))

    def test_profile_spaces_have_the_agreed_axes(self):
        self.assertEqual(MATERIALS_SPACE.names, ["mechanism_class", "length_scale", "inspiration_origin", "governing_quantity"])
        self.assertEqual([a.name for a in MATERIALS_SPACE.ordinal_axes], ["length_scale"])
        self.assertEqual([a.name for a in BUSINESS_SPACE.ordinal_axes], ["market_scale", "capital_intensity"])
        self.assertEqual(BUSINESS_SPACE.axis("market_scale").values, ("niche", "regional", "national", "continental", "global"))

    def test_axis_rejects_unnormalised_values(self):
        with self.assertRaises(ValueError):
            Axis("bad", NOMINAL, ("Foo Bar", "baz"))


class ArchiveTest(ArchiveTestCase):
    def test_elite_replaced_only_by_strictly_better_score(self):
        first = self.add(cell("a", "s1", "x"), 50.0)
        self.assertEqual(first["outcome"], NEW_ELITE)
        tie = self.add(cell("a", "s1", "x"), 50.0)
        self.assertEqual(tie["outcome"], NOT_BETTER)
        worse = self.add(cell("a", "s1", "x"), 10.0)
        self.assertEqual(worse["outcome"], NOT_BETTER)
        better = self.add(cell("a", "s1", "x"), 60.0)
        self.assertEqual(better["outcome"], IMPROVED)
        self.assertEqual(better["replaced"], first["id"])
        self.assertEqual(self.archive.elite_for(cell("a", "s1", "x"))["id"], better["id"])
        self.assertEqual(self.archive.proposal_count(cell("a", "s1", "x")), 4)

    def test_failed_entries_count_as_proposals_but_never_become_elites(self):
        failed = self.add(cell("b", "s2", "x"), None, status=FAILED)
        self.assertIsNone(failed["outcome"])
        self.assertEqual(self.archive.proposal_count(cell("b", "s2", "x")), 1)
        self.assertEqual(self.archive.elites, {})

    def test_save_load_roundtrip_is_atomic_and_checks_profile(self):
        self.add(cell("a", "s1", "x"), 50.0)
        self.archive.mark_infeasible({"mech": "c"}, "impossible", entry_id=None, round_no=1, source="test")
        self.archive.save()
        leftovers = [name for name in os.listdir(self._tmp.name) if name.startswith(".archive-")]
        self.assertEqual(leftovers, [])
        loaded = Archive.load("test", SPACE, path=self.archive.path)
        self.assertEqual(loaded.data, self.archive.data)
        self.assertTrue(loaded.is_infeasible(cell("c", "s5", "y")))
        with self.assertRaises(ValueError):
            Archive.load("other", SPACE, path=self.archive.path)
        with open(self.archive.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["entries"]["e00001"]["title"], "concept 1")

    def test_evidence_tier_beats_score_within_a_cell(self):
        where = cell("a", "s1", "x")
        simulated = self.add(where, 20.0, breakdown=SIMULATED)
        estimated = self.add(where, 45.0, breakdown=ESTIMATED)
        self.assertEqual(estimated["outcome"], NOT_BETTER)
        self.assertEqual(self.archive.elite_for(where)["id"], simulated["id"])
        better = self.add(where, 21.0, breakdown=SIMULATED)
        self.assertEqual(better["outcome"], IMPROVED)
        # The other way round: a simulated entry replaces an estimated elite whatever its score.
        other = cell("b", "s1", "x")
        self.add(other, 45.0, breakdown=ESTIMATED)
        replaced = self.add(other, 5.0, breakdown=SIMULATED)
        self.assertEqual(replaced["outcome"], IMPROVED)
        # An estimate after an implausible simulation never replaces a sound estimate.
        third = cell("c", "s1", "x")
        sound = self.add(third, 10.0, breakdown=ESTIMATED)
        self.assertEqual(self.add(third, 40.0, breakdown=IMPLAUSIBLE)["outcome"], NOT_BETTER)
        self.assertEqual(self.archive.elite_for(third)["id"], sound["id"])
        ranked = [e["id"] for e in self.archive.ranked_elites()]
        self.assertEqual(ranked[-1], sound["id"])     # estimated tier last despite its score

    def test_requirements_are_stored_once_and_survive_a_reload(self):
        self.assertTrue(self.archive.set_requirements(
            [{"name": "low_drag", "criterion": "less drag"}, {"name": "low_drag", "criterion": "dup"},
             {"name": " ", "criterion": "empty"}, {"name": "non_toxic"}], round_no=1))
        self.assertFalse(self.archive.set_requirements([{"name": "other", "criterion": "x"}], round_no=2))
        self.assertEqual([r["name"] for r in self.archive.requirements], ["low_drag", "non_toxic"])
        self.archive.declare_relevance_axes(["origin"])
        self.assertEqual(self.archive.relevant_values("origin"), [])
        self.archive.add_relevant_values("origin", ["y"])
        self.add(cell("a", "s1", "x"), 5.0)
        self.assertEqual(self.archive.relevant_values("origin"), ["x", "y"])   # elites count as relevant
        self.assertIsNone(self.archive.relevant_values("mech"))
        self.archive.save()
        loaded = Archive.load("test", SPACE, path=self.archive.path)
        self.assertEqual(loaded.requirements, self.archive.requirements)
        self.assertEqual(loaded.relevance_axes(), ["origin"])

    def test_old_archive_without_request_analysis_loads(self):
        self.add(cell("a", "s1", "x"), 5.0)
        self.archive.save()
        with open(self.archive.path, encoding="utf-8") as f:
            data = json.load(f)
        data.pop("request_analysis")
        with open(self.archive.path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        loaded = Archive.load("test", SPACE, path=self.archive.path)
        self.assertEqual(loaded.requirements, [])
        self.assertEqual(loaded.relevance_axes(), [])

    def _rewrite(self, **changes):
        self.archive.save()
        with open(self.archive.path, encoding="utf-8") as f:
            data = json.load(f)
        data.update(changes)
        with open(self.archive.path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_older_archive_with_the_same_vocabulary_is_migrated(self):
        self.add(cell("a", "s1", "x"), 5.0)
        self._rewrite(version=2)
        loaded = Archive.load("test", SPACE, path=self.archive.path)
        self.assertEqual(loaded.data["version"], ARCHIVE_VERSION)
        self.assertEqual(loaded.data["migrated_from"], 2)
        self.assertEqual(len(loaded.elites), 1)
        self.assertEqual(loaded.objective_statement, "")

    def test_archive_with_another_vocabulary_or_a_newer_version_is_refused(self):
        self.add(cell("a", "s1", "x"), 5.0)
        self._rewrite(version=ARCHIVE_VERSION + 1)
        with self.assertRaises(ArchiveCompatibilityError) as ctx:
            Archive.load("test", SPACE, path=self.archive.path)
        self.assertIn("newer than this explorer", str(ctx.exception))
        # A version-2 materials archive: governing_quantity had no fouling values yet.
        old_axes = MATERIALS_SPACE.to_dict()
        old_axes[3] = dict(old_axes[3], values=["wall_shear", "flow_rate", "heat_flux", "temperature", "deflection",
                                                "stress", "field_strength", "other"])
        path = os.path.join(self._tmp.name, "materials.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 2, "profile": "materials", "axes": old_axes, "entries": {}, "elites": {}}, f)
        with self.assertRaises(ArchiveCompatibilityError) as ctx:
            Archive.load("materials", MATERIALS_SPACE, path=path)
        message = str(ctx.exception)
        for text in ("version 2", "governing_quantity: new values fouling_adhesion, degradation_rate",
                     "not comparable", "--archive NAME"):
            self.assertIn(text, message)
        self.assertIsInstance(ctx.exception, ValueError)

    def test_source_uses_count_sources_not_context_neighbours(self):
        a = self.add(cell("a", "s1", "x"), 50.0)
        b = self.add(cell("b", "s1", "x"), 40.0)
        c = self.add(cell("c", "s1", "x"), 30.0)
        self.add(cell("a", "s2", "x"), 1.0, strategy="fill_gap", parents=[a["id"], b["id"]])
        self.add(cell("a", "s3", "x"), 1.0, strategy="combine", parents=[b["id"], c["id"]])
        self.add(cell("a", "s4", "x"), 1.0, strategy="refine", parents=[c["id"]])
        self.assertEqual(self.archive.source_uses(), {a["id"]: 1, b["id"]: 1, c["id"]: 1})
        batch = [{"strategy": "diversify", "parent_ids": [a["id"]]}, {"strategy": "refine", "parent_ids": [a["id"]]}]
        self.assertEqual(st.source_uses(self.archive, batch)[a["id"]], 2)

    def test_raw_result_is_stored_relative_to_the_archive(self):
        entry = self.archive.add_entry(
            round_no=1, run_id="t", order={"order_id": "o", "strategy": "seed", "target": {}}, title="t",
            concept={}, descriptors=cell("a", "s1", "x"), status=EVALUATED, score=1.0, raw_result={"x": 1},
        )
        self.assertEqual(entry["raw_result_path"], os.path.join("results", "e00001.json"))
        self.assertTrue(os.path.exists(os.path.join(self.archive.directory, entry["raw_result_path"])))


class FillGapTest(ArchiveTestCase):
    def test_gaps_next_to_good_elites_come_first(self):
        good = self.add(cell("a", "s3", "x"), 80.0)
        self.add(cell("c", "s1", "y"), 20.0)
        orders = st.fill_gap(self.archive, random.Random(0), 3, set(), st.StrategyConfig())
        self.assertEqual(len(orders), 3)
        for order in orders:
            self.assertEqual(order["strategy"], st.FILL_GAP)
            self.assertEqual(SPACE.distance(order["target"], good["descriptors"]), 1)
            self.assertEqual(order["parent_ids"][0], good["id"])
            self.assertNotIn(order["target_key"], self.archive.elites)

    def test_low_proposal_density_is_preferred(self):
        self.add(cell("a", "s3", "x"), 80.0)
        # Two failed attempts already landed in one neighbour; it must drop behind untouched ones.
        crowded = cell("a", "s3", "y")
        self.add(crowded, None, status=FAILED)
        self.add(crowded, None, status=FAILED)
        gaps = st.gap_candidates(self.archive)
        keys = [g["key"] for g in gaps]
        self.assertEqual(keys[-1], SPACE.cell_key(crowded))
        self.assertEqual(gaps[-1]["proposals"], 2)

    def test_gap_bracketed_on_an_ordinal_axis_gets_the_neighbour_bonus(self):
        self.add(cell("a", "s2", "x"), 60.0)
        self.add(cell("a", "s4", "x"), 60.0)
        top = st.gap_candidates(self.archive, config=st.StrategyConfig(rarity_bonus=0.0))[0]
        self.assertEqual(top["cell"], cell("a", "s3", "x"))
        self.assertEqual(len(top["neighbours"]), 2)

    def test_neighbour_bonus_counts_distinct_axes_not_elites(self):
        config = st.StrategyConfig(rarity_bonus=0.0)
        # Two elites that differ from the gap along two different axes earn the bonus ...
        self.add(cell("a", "s1", "x"), 60.0)
        self.add(cell("b", "s2", "x"), 60.0)
        gaps = {g["key"]: g for g in st.gap_candidates(self.archive, config=config)}
        two_axes = gaps[SPACE.cell_key(cell("b", "s1", "x"))]
        single = gaps[SPACE.cell_key(cell("c", "s1", "x"))]
        self.assertEqual(two_axes["changed_axes"], ["mech", "size"])
        self.assertGreater(two_axes["priority"], single["priority"])
        # ... several elites that differ from it only in the same nominal axis earn nothing.
        archive = self.new_archive(name="same_axis.json")
        self.add(cell("a", "s1", "x"), 60.0, archive=archive)
        self.add(cell("b", "s1", "x"), 60.0, archive=archive)
        gaps = {g["key"]: g for g in st.gap_candidates(archive, config=config)}
        crowd = gaps[SPACE.cell_key(cell("c", "s1", "x"))]
        lone = gaps[SPACE.cell_key(cell("a", "s1", "y"))]
        self.assertEqual(len(crowd["neighbours"]), 2)
        self.assertEqual(crowd["priority"], lone["priority"])

    def test_under_explored_axis_values_rank_first(self):
        # Every proposal so far has origin=x; mechanisms are mixed.
        self.add(cell("a", "s3", "x"), 50.0)
        self.add(cell("b", "s3", "x"), 50.0)
        self.add(cell("c", "s1", "x"), 50.0)
        gaps = st.gap_candidates(self.archive)
        self.assertEqual(gaps[0]["changed_axes"], ["origin"])
        self.assertEqual(gaps[0]["cell"]["origin"], "y")
        self.assertAlmostEqual(gaps[0]["exploration"], 1.0)
        mech_gap = next(g for g in gaps if g["changed_axes"] == ["mech"])
        self.assertLess(mech_gap["exploration"], gaps[0]["exploration"])
        orders = st.fill_gap(self.archive, random.Random(0), 1, set(), st.StrategyConfig())
        self.assertEqual(orders[0]["target"]["origin"], "y")
        self.assertIn("under_exploration", orders[0]["context"])

    def test_irrelevant_values_of_a_relevance_axis_are_not_gap_targets(self):
        self.archive.declare_relevance_axes(["origin"])
        self.add(cell("a", "s3", "x"), 80.0)
        targets = [g["cell"] for g in st.gap_candidates(self.archive)]
        self.assertTrue(targets)
        self.assertFalse(any(t["origin"] == "y" for t in targets))
        self.archive.add_relevant_values("origin", ["y"])
        self.assertIn(cell("a", "s3", "y"), [g["cell"] for g in st.gap_candidates(self.archive)])

    def test_infeasible_and_taken_cells_are_skipped(self):
        self.add(cell("a", "s3", "x"), 80.0)
        self.archive.mark_infeasible({"mech": "b"}, "no", entry_id=None, round_no=1, source="test")
        taken = {SPACE.cell_key(cell("a", "s2", "x"))}
        orders = st.fill_gap(self.archive, random.Random(0), 10, taken, st.StrategyConfig())
        targets = [o["target"] for o in orders]
        self.assertNotIn(cell("b", "s3", "x"), targets)
        self.assertNotIn(cell("a", "s2", "x"), targets)
        self.assertIn(cell("a", "s4", "x"), targets)


class ExtrapolateTest(ArchiveTestCase):
    def _run(self, n=5, **config):
        return st.extrapolate(self.archive, random.Random(0), n, set(), st.StrategyConfig(**config))

    def test_monotone_trend_proposes_the_next_ordinal_value(self):
        for size, score in (("s1", 10.0), ("s2", 20.0), ("s3", 30.0)):
            self.add(cell("a", size, "x"), score)
        orders = self._run()
        self.assertEqual(orders[0]["target"], cell("a", "s4", "x"))
        trend = orders[0]["context"]["trend"]
        self.assertEqual((trend["mode"], trend["axis"], trend["edge"], trend["next"]), ("slice", "size", "s3", "s4"))
        self.assertAlmostEqual(trend["slope"], 10.0)
        self.assertAlmostEqual(trend["r2"], 1.0)
        self.assertIn("extrapolate one step to size=s4", orders[0]["rationale"])

    def test_decreasing_trend_extrapolates_downwards(self):
        for size, score in (("s3", 30.0), ("s4", 20.0), ("s5", 10.0)):
            self.add(cell("b", size, "y"), score)
        orders = self._run()
        self.assertEqual(orders[0]["target"], cell("b", "s2", "y"))

    def test_no_proposal_at_the_end_of_the_scale(self):
        for size, score in (("s3", 10.0), ("s4", 20.0), ("s5", 30.0)):
            self.add(cell("a", size, "x"), score)
        self.assertEqual(self._run(), [])
        statuses = {(t["mode"], t["status"]) for t in st.find_trends(self.archive)}
        self.assertEqual(statuses, {("slice", "scale_end"), ("marginal", "scale_end")})

    def test_flat_and_peaked_trends_are_skipped(self):
        for size, score in (("s1", 10.0), ("s2", 30.0), ("s3", 20.0)):
            self.add(cell("a", size, "x"), score)
        self.assertEqual(self._run(), [])
        self.assertEqual({t["status"] for t in st.find_trends(self.archive)}, {"peaked"})
        for size, score in (("s1", 10.0), ("s2", 10.2), ("s3", 10.4)):
            self.add(cell("c", size, "y"), score)
        trends = [t for t in st.find_trends(self.archive, st.StrategyConfig(min_slope=1.0)) if t["fixed"].get("mech") == "c"]
        self.assertEqual(trends[0]["status"], "flat")

    def test_marginal_trend_gives_a_partial_target(self):
        # No slice has two elites, but the best score per size value rises.
        self.add(cell("a", "s1", "x"), 10.0)
        self.add(cell("b", "s2", "y"), 25.0)
        self.add(cell("c", "s3", "x"), 38.0)
        orders = self._run()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["target"], {"size": "s4"})
        self.assertEqual(orders[0]["target_key"], "mech=*|size=s4|origin=*")
        self.assertEqual(orders[0]["context"]["trend"]["mode"], "marginal")

    def test_two_points_are_not_a_trend(self):
        self.add(cell("a", "s1", "x"), 10.0)
        self.add(cell("a", "s2", "x"), 30.0)
        self.assertEqual(self._run(), [])
        self.assertEqual({t["status"] for t in st.find_trends(self.archive)}, {"too_few_points"})
        # With min_trend_points=2 the same two points would be extrapolated.
        self.assertEqual(self._run(min_trend_points=2)[0]["target"], cell("a", "s3", "x"))

    def test_trends_use_only_simulated_scores(self):
        # Two simulated points plus one estimate would make three, but tiers are never mixed.
        self.add(cell("a", "s1", "x"), 10.0, breakdown=SIMULATED)
        self.add(cell("a", "s2", "x"), 20.0, breakdown=SIMULATED)
        self.add(cell("a", "s3", "x"), 45.0, breakdown=ESTIMATED)
        self.assertEqual(self._run(), [])
        trends = st.find_trends(self.archive)
        self.assertTrue(all(len(t["points"]) == 2 for t in trends), trends)
        self.assertEqual({t["status"] for t in trends}, {"too_few_points"})
        # Three self-estimates never trigger an extrapolation either.
        archive = self.new_archive(name="estimates.json")
        for size, score in (("s1", 10.0), ("s2", 20.0), ("s3", 30.0)):
            self.add(cell("b", size, "y"), score, archive=archive, breakdown=ESTIMATED)
        self.assertEqual(st.find_trends(archive), [])
        self.add(cell("b", "s3", "x"), 30.0, archive=archive, breakdown=SIMULATED)
        self.add(cell("b", "s2", "x"), 20.0, archive=archive, breakdown=SIMULATED)
        self.add(cell("b", "s1", "x"), 10.0, archive=archive, breakdown=SIMULATED)
        orders = st.extrapolate(archive, random.Random(0), 5, set(), st.StrategyConfig())
        self.assertEqual(orders[0]["target"], cell("b", "s4", "x"))

    def test_poor_fit_is_not_extrapolated(self):
        for size, score in (("s1", 10.0), ("s2", 40.0), ("s3", 12.0), ("s4", 45.0)):
            self.add(cell("a", size, "x"), score)
        statuses = {t["status"] for t in st.find_trends(self.archive, st.StrategyConfig(min_r2=0.9))}
        self.assertEqual(statuses, {"poor_fit"})
        self.assertEqual(self._run(min_r2=0.9), [])

    def test_infeasible_target_is_skipped(self):
        for size, score in (("s1", 10.0), ("s2", 20.0), ("s3", 30.0)):
            self.add(cell("a", size, "x"), score)
        self.archive.mark_infeasible(cell("a", "s4", "x"), "too big", entry_id=None, round_no=1, source="test")
        self.archive.mark_infeasible({"size": "s4"}, "too big", entry_id=None, round_no=1, source="test")
        self.assertEqual(self._run(), [])
        self.assertIn("infeasible", {t["status"] for t in st.find_trends(self.archive)})

    def test_linear_fit(self):
        slope, intercept, r2 = st.linear_fit([0, 1, 2, 3], [1, 3, 5, 7])
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 1.0)
        self.assertAlmostEqual(r2, 1.0)
        with self.assertRaises(ValueError):
            st.linear_fit([1, 1], [0, 1])


class CombineRefineExploreTest(ArchiveTestCase):
    def test_combine_picks_the_most_distant_elites_and_mixes_them(self):
        a = self.add(cell("a", "s1", "x"), 50.0)
        self.add(cell("a", "s2", "x"), 90.0)
        c = self.add(cell("c", "s5", "y"), 40.0)
        orders = st.combine(self.archive, random.Random(3), 1, set(), st.StrategyConfig())
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertEqual(sorted(order["parent_ids"]), sorted([a["id"], c["id"]]))
        target = order["target"]
        self.assertNotIn(order["target_key"], self.archive.elites)
        self.assertNotEqual(target, a["descriptors"])
        self.assertNotEqual(target, c["descriptors"])
        for name in SPACE.names:
            self.assertIn(target[name], (a["descriptors"][name], c["descriptors"][name]))

    def test_combine_needs_distance_two(self):
        self.add(cell("a", "s1", "x"), 50.0)
        self.add(cell("a", "s2", "x"), 50.0)
        self.assertEqual(st.combine(self.archive, random.Random(0), 2, set(), st.StrategyConfig()), [])

    def test_refine_targets_elite_cells_and_prefers_good_unrefined_elites(self):
        low = self.add(cell("a", "s1", "x"), 1.0)
        high = self.add(cell("b", "s1", "x"), 99.0)
        picks = [st.refine(self.archive, random.Random(seed), 1, set(), st.StrategyConfig())[0]["parent_ids"][0]
                 for seed in range(20)]
        self.assertGreater(picks.count(high["id"]), picks.count(low["id"]))
        orders = st.refine(self.archive, random.Random(0), 5, set(), st.StrategyConfig())
        self.assertEqual(len(orders), 2)
        self.assertEqual({o["target_key"] for o in orders}, set(self.archive.elites))

    def test_explore_picks_unvisited_feasible_cells(self):
        self.add(cell("a", "s1", "x"), 5.0)
        self.archive.mark_infeasible({"mech": "b"}, "no", entry_id=None, round_no=1, source="test")
        orders = st.explore(self.archive, random.Random(1), 100, set(), st.StrategyConfig(explore_max_distance=0))
        targets = [o["target"] for o in orders]
        self.assertEqual(len(targets), 3 * 5 * 2 - 1 - 10)
        self.assertNotIn(cell("a", "s1", "x"), targets)
        self.assertFalse(any(t["mech"] == "b" for t in targets))
        # By default only cells within distance 3 of a feasible entry.
        near = st.explore(self.archive, random.Random(1), 100, set(), st.StrategyConfig())
        self.assertTrue(near)
        self.assertLess(len(near), len(targets))
        self.assertTrue(all(SPACE.distance(o["target"], cell("a", "s1", "x")) <= 3 for o in near))

    def test_explore_only_uses_relevant_values_of_relevance_axes(self):
        self.archive.declare_relevance_axes(["origin"])
        # Nothing known yet (cold start before any function analysis): no explore order.
        self.assertEqual(st.explore(self.archive, random.Random(0), 3, set(), st.StrategyConfig()), [])
        orders = st.schedule(self.archive, 4, random.Random(0), round_no=1)
        self.assertEqual({o["strategy"] for o in orders}, {st.SEED})
        self.archive.add_relevant_values("origin", ["y"])
        orders = st.explore(self.archive, random.Random(0), 100, set(), st.StrategyConfig())
        self.assertEqual(len(orders), 3 * 5)
        self.assertEqual({o["target"]["origin"] for o in orders}, {"y"})
        self.assertIn("relevant", orders[0]["rationale"])

    def test_explore_prefers_rare_values_on_other_axes(self):
        for size in ("s1", "s2", "s3", "s4", "s5"):
            for _ in range(3):
                self.add(cell("a", size, "x"), None, status=FAILED)
        picks = {"a": 0, "b": 0, "c": 0}
        for seed in range(30):
            for order in st.explore(self.archive, random.Random(seed), 2, set(), st.StrategyConfig()):
                picks[order["target"]["mech"]] += 1
        self.assertLess(picks["a"], picks["b"])
        self.assertLess(picks["a"], picks["c"])

    def test_diversify_targets_the_least_proposed_value_of_the_least_diverse_axis(self):
        best = self.add(cell("a", "s3", "x"), 60.0)
        self.add(cell("b", "s3", "x"), 40.0)
        self.add(cell("c", "s1", "x"), 30.0)
        orders = st.diversify(self.archive, random.Random(0), 1, set(), st.StrategyConfig())
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["strategy"], st.DIVERSIFY)
        self.assertEqual(orders[0]["target"], cell("a", "s3", "y"))
        self.assertEqual(orders[0]["parent_ids"], [best["id"]])
        self.assertIn("least diverse along origin", orders[0]["rationale"])
        # Several orders go to different axes and never to taken or elite cells.
        orders = st.diversify(self.archive, random.Random(0), 3, {SPACE.cell_key(cell("a", "s3", "y"))}, st.StrategyConfig())
        keys = [o["target_key"] for o in orders]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertNotIn(SPACE.cell_key(cell("a", "s3", "y")), keys)
        self.assertFalse(set(keys) & set(self.archive.elites))


COMPAT = st.StrategyConfig(compatibility_axes=("mech", "size"))


class CompatibilityRotationTest(ArchiveTestCase):
    def _no_source(self, parent_id, times):
        """Records that an elite already served as a source (entries without a cell change no statistics)."""
        for _ in range(times):
            self.archive.add_entry(round_no=1, run_id="t", title="x", concept={}, descriptors=None, status=INVALID,
                                   order={"order_id": "o", "strategy": "fill_gap", "target": {},
                                          "parent_ids": [parent_id]})

    def test_combine_keeps_a_proven_mechanism_scale_pair(self):
        # The recorded failure: a mm-scale mechanism combined with nm (distance 4 on the size axis alone).
        a = self.add(cell("a", "s1", "x"), 50.0)
        b = self.add(cell("c", "s5", "y"), 40.0)
        for seed in range(10):
            orders = st.combine(self.archive, random.Random(seed), 1, set(), COMPAT)
            self.assertEqual(len(orders), 1)
            target = orders[0]["target"]
            self.assertIn((target["mech"], target["size"]), {("a", "s1"), ("c", "s5")})
            self.assertIn("feasible elsewhere", orders[0]["context"]["compatibility"])
        # Without compatibility axes (e.g. business) any mixture that is not infeasible may be chosen.
        mixed = {(o["target"]["mech"], o["target"]["size"])
                 for seed in range(20) for o in st.combine(self.archive, random.Random(seed), 1, set(), st.StrategyConfig())}
        self.assertTrue(mixed - {("a", "s1"), ("c", "s5")})
        self.assertEqual(sorted(orders[0]["parent_ids"]), sorted([a["id"], b["id"]]))

    def test_combine_accepts_unproven_pairs_close_to_both_parents_but_not_infeasible_ones(self):
        self.add(cell("a", "s2", "x"), 50.0)
        self.add(cell("b", "s3", "y"), 40.0)
        targets = {o["target_key"] for seed in range(20) for o in st.combine(self.archive, random.Random(seed), 1, set(), COMPAT)}
        pairs = {(SPACE.parse_key(k)["mech"], SPACE.parse_key(k)["size"]) for k in targets}
        self.assertTrue(pairs & {("a", "s3"), ("b", "s2")})              # unproven, within distance 2 of both
        self.archive.mark_infeasible({"mech": "a", "size": "s3"}, "no", entry_id=None, round_no=1, source="test")
        self.archive.mark_infeasible(cell("b", "s2", "x"), "no", entry_id=None, round_no=1, source="test")
        for seed in range(20):
            for order in st.combine(self.archive, random.Random(seed), 1, set(), COMPAT):
                pair = (order["target"]["mech"], order["target"]["size"])
                self.assertNotIn(pair, {("a", "s3"), ("b", "s2")})

    def test_explore_prefers_cells_next_to_feasible_regions(self):
        self.add(cell("a", "s1", "x"), 50.0)
        self.archive.add_entry(round_no=1, run_id="t", title="bad", concept={}, descriptors=cell("c", "s5", "y"),
                               status=FAILED, order={"order_id": "o", "strategy": "seed", "target": {}})

        def far_share(config):
            far = total = 0
            for seed in range(40):
                for order in st.explore(self.archive, random.Random(seed), 3, set(), config):
                    total += 1
                    far += SPACE.distance(order["target"], cell("a", "s1", "x")) >= 3
            return far / total

        biased = far_share(st.StrategyConfig())
        flat = far_share(st.StrategyConfig(explore_distance_decay=0.0))
        self.assertLess(biased, flat)
        order = st.explore(self.archive, random.Random(0), 1, set(), st.StrategyConfig())[0]
        self.assertEqual(order["context"]["nearest_feasible"]["distance"],
                         SPACE.distance(order["target"], cell("a", "s1", "x")))
        self.assertIn("nearest concept that worked", order["rationale"])

    def test_explore_avoids_pairs_that_were_only_infeasible(self):
        self.add(cell("a", "s1", "x"), 50.0)
        self.archive.mark_infeasible(cell("b", "s1", "x"), "no", entry_id=None, round_no=1, source="test")
        config = st.StrategyConfig(compatibility_axes=("mech", "size"), explore_distance_decay=0.0,
                                   explore_infeasible_pair_factor=0.0)
        for seed in range(10):
            for order in st.explore(self.archive, random.Random(seed), 5, set(), config):
                self.assertNotEqual((order["target"]["mech"], order["target"]["size"]), ("b", "s1"))

    def _two_elite_archive(self):
        top = self.add(cell("a", "s3", "x"), 60.0)
        low = self.add(cell("c", "s1", "y"), 55.0)
        return top, low

    def test_fill_gap_rotates_source_elites_within_a_batch(self):
        top, low = self._two_elite_archive()

        def sources(penalty):
            orders = st.fill_gap(self.archive, random.Random(0), 4, set(), st.StrategyConfig(source_penalty=penalty))
            return [o["parent_ids"][0] for o in orders]

        self.assertEqual(sources(0.0).count(top["id"]), 3)
        rotated = sources(0.3)
        self.assertEqual((rotated.count(top["id"]), rotated.count(low["id"])), (2, 2))

    def test_fill_gap_penalises_elites_that_already_served_as_source(self):
        top, low = self._two_elite_archive()
        first = st.fill_gap(self.archive, random.Random(0), 1, set(), st.StrategyConfig())[0]
        self.assertEqual(first["parent_ids"][0], top["id"])
        self._no_source(top["id"], 3)
        first = st.fill_gap(self.archive, random.Random(0), 1, set(), st.StrategyConfig())[0]
        self.assertEqual(first["parent_ids"][0], low["id"])
        self.assertEqual(first["context"]["source_uses"], 0)
        gap = st.gap_candidates(self.archive)[0]
        self.assertIn("source_uses", gap)
        # The batch counts too: a fallback call sees the sources of orders already scheduled.
        batch = [{"strategy": "fill_gap", "parent_ids": [low["id"]]}] * 3
        first = st.fill_gap(self.archive, random.Random(0), 1, set(), st.StrategyConfig(), batch=batch)[0]
        self.assertEqual(first["parent_ids"][0], top["id"])

    def test_diversify_rotates_its_source_elite(self):
        best = self.add(cell("a", "s3", "x"), 60.0)
        second = self.add(cell("b", "s3", "x"), 58.0)
        self.add(cell("c", "s1", "x"), 30.0)
        self.assertEqual(st.diversify(self.archive, random.Random(0), 1, set(), st.StrategyConfig())[0]["parent_ids"],
                         [best["id"]])
        self._no_source(best["id"], 2)
        order = st.diversify(self.archive, random.Random(0), 1, set(), st.StrategyConfig())[0]
        self.assertEqual(order["parent_ids"], [second["id"]])
        self.assertEqual(order["target"], cell("b", "s3", "y"))

    def test_orders_that_change_an_origin_axis_demand_a_real_origin(self):
        self.add(cell("a", "s3", "x"), 60.0)
        self.add(cell("b", "s3", "x"), 40.0)
        self.add(cell("c", "s1", "x"), 30.0)          # every proposal has origin=x: the least diverse axis
        config = st.StrategyConfig(origin_axes=("origin",))
        order = st.diversify(self.archive, random.Random(0), 1, set(), config)[0]
        self.assertEqual(order["context"]["axis"], "origin")
        self.assertIn("must really come from a origin=y system", order["rationale"])
        self.assertIn("relabelled analogue", order["rationale"])
        gaps = st.fill_gap(self.archive, random.Random(0), 10, set(), config)
        by_axis = {o["context"]["neighbours"][0]["differs_in"]: o for o in gaps}
        self.assertIn("relabelled analogue", by_axis["origin"]["rationale"])
        self.assertNotIn("relabelled analogue", by_axis["size"]["rationale"])
        plain = st.diversify(self.archive, random.Random(0), 1, set(), st.StrategyConfig())[0]
        self.assertNotIn("relabelled analogue", plain["rationale"])

    def test_schedule_reports_why_extrapolation_did_not_fire(self):
        diagnostics = {}
        st.schedule(self.archive, 3, random.Random(0), round_no=1, diagnostics=diagnostics)
        self.assertIn("no trend could be fitted", diagnostics["extrapolate"]["summary"])
        # The recorded case: three marginal points with r2 = 0.20.
        for size, score in (("s1", 14.9), ("s3", 4.6), ("s5", 27.7)):
            self.add(cell("a", size, "x"), score, breakdown={"evidence_tier": "simulated", "evidence_rank": 2})
        diagnostics = {}
        orders = st.schedule(self.archive, 6, random.Random(0), round_no=2, diagnostics=diagnostics)
        info = diagnostics["extrapolate"]
        self.assertNotIn(st.EXTRAPOLATE, {o["strategy"] for o in orders})
        self.assertIn("poor_fit", info["summary"])
        row = next(t for t in info["trends"] if t["status"] == "poor_fit")
        self.assertIn("< 0.5", row["reason"])
        self.assertEqual(diagnostics["produced"].get(st.EXTRAPOLATE, 0), 0)
        self.assertEqual(sum(diagnostics["produced"].values()), 6)
        # Diagnostics do not change the schedule.
        again = st.schedule(self.archive, 6, random.Random(0), round_no=2)
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(orders, sort_keys=True))


class SchedulerTest(ArchiveTestCase):
    def _populate(self):
        for size, score in (("s1", 10.0), ("s2", 20.0), ("s3", 30.0)):
            self.add(cell("a", size, "x"), score)
        self.add(cell("c", "s5", "y"), 45.0)

    def test_schedule_is_deterministic_for_a_seed(self):
        self._populate()
        first = st.schedule(self.archive, 6, random.Random("7:2"), round_no=2)
        second = st.schedule(self.archive, 6, random.Random("7:2"), round_no=2)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual([o["order_id"] for o in first], [f"r002-0{i}" for i in range(1, 7)])
        full_targets = [o["target_key"] for o in first if o["target"]]
        self.assertEqual(len(full_targets), len(set(full_targets)))
        weighted = st.schedule(self.archive, 6, random.Random("7:2"), round_no=2,
                               weights={"extrapolate": 1.0, "fill_gap": 1.0, "diversify": 1.0})
        self.assertIn(st.EXTRAPOLATE, {o["strategy"] for o in weighted})
        self.assertIn(st.DIVERSIFY, {o["strategy"] for o in weighted})
        seeds = {json.dumps(st.schedule(self.archive, 6, random.Random(seed), round_no=2), sort_keys=True)
                 for seed in range(8)}
        self.assertGreater(len(seeds), 1)

    def test_cold_start_uses_seed_and_explore_only(self):
        orders = st.schedule(self.archive, 5, random.Random(0), round_no=1)
        self.assertEqual(len(orders), 5)
        self.assertTrue({o["strategy"] for o in orders} <= {st.SEED, st.EXPLORE})
        self.assertIn(st.SEED, {o["strategy"] for o in orders})

    def test_unavailable_strategies_fall_back(self):
        self._populate()
        orders = st.schedule(self.archive, 4, random.Random(0), round_no=3, weights={"combine": 1.0})
        self.assertEqual(len(orders), 4)
        self.assertEqual(orders[0]["strategy"], st.COMBINE)

    def test_infeasible_cells_are_never_scheduled(self):
        self._populate()
        self.archive.mark_infeasible({"origin": "y"}, "no", entry_id=None, round_no=1, source="test")
        for seed in range(5):
            for order in st.schedule(self.archive, 8, random.Random(seed), round_no=4):
                if order["target"] and SPACE.is_full(order["target"]) and order["strategy"] != st.REFINE:
                    self.assertNotEqual(order["target"].get("origin"), "y", order)

    def test_allocate_and_parse_weights(self):
        counts = st.allocate({"refine": 0.5, "fill_gap": 0.5, "explore": 0.1}, 7, random.Random(0))
        self.assertEqual(sum(counts.values()), 7)
        self.assertGreaterEqual(counts["refine"], 3)
        self.assertEqual(st.parse_weights("refine=1,explore=0.5"), {"refine": 1.0, "explore": 0.5})
        self.assertEqual(st.parse_weights(None), st.DEFAULT_WEIGHTS)
        for bad in ("magic=1", "refine", "refine=-1", "refine=0"):
            with self.assertRaises(ValueError):
                st.parse_weights(bad)


if __name__ == "__main__":
    unittest.main()
