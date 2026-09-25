# -*- coding: utf-8 -*-
"""Explorer descriptors, archive and search strategies (no model calls)."""
import json
import os
import random
import tempfile
import unittest

from vectornaut.explorer.archive import EVALUATED, FAILED, IMPROVED, NEW_ELITE, NOT_BETTER, Archive
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

    def add(self, descriptors, score, strategy="seed", status=EVALUATED, parents=(), archive=None):
        self._n += 1
        archive = archive or self.archive
        return archive.add_entry(
            round_no=1, run_id="t", title=f"concept {self._n}", concept={"summary": f"idea {self._n}"},
            order={"order_id": f"o{self._n}", "strategy": strategy, "target": descriptors, "parent_ids": list(parents)},
            descriptors=descriptors, status=status, score=score,
        )


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

    def test_more_elite_neighbours_rank_higher(self):
        self.add(cell("a", "s2", "x"), 60.0)
        self.add(cell("a", "s4", "x"), 60.0)
        top = st.gap_candidates(self.archive)[0]
        self.assertEqual(top["cell"], cell("a", "s3", "x"))
        self.assertEqual(len(top["neighbours"]), 2)

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
        for size, score in (("s1", 10.0), ("s2", 10.2)):
            self.add(cell("c", size, "y"), score)
        trends = [t for t in st.find_trends(self.archive, st.StrategyConfig(min_slope=1.0)) if t["fixed"].get("mech") == "c"]
        self.assertEqual(trends[0]["status"], "flat")

    def test_marginal_trend_gives_a_partial_target(self):
        # No slice has two elites, but the best score per size value rises.
        self.add(cell("a", "s1", "x"), 10.0)
        self.add(cell("b", "s2", "y"), 25.0)
        orders = self._run()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["target"], {"size": "s3"})
        self.assertEqual(orders[0]["target_key"], "mech=*|size=s3|origin=*")
        self.assertEqual(orders[0]["context"]["trend"]["mode"], "marginal")

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
        orders = st.explore(self.archive, random.Random(1), 100, set(), st.StrategyConfig())
        targets = [o["target"] for o in orders]
        self.assertEqual(len(targets), 3 * 5 * 2 - 1 - 10)
        self.assertNotIn(cell("a", "s1", "x"), targets)
        self.assertFalse(any(t["mech"] == "b" for t in targets))


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
        self.assertIn(st.EXTRAPOLATE, {o["strategy"] for o in first})
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
