# -*- coding: utf-8 -*-
"""Offline ports of the mock-mode /api/run scripts in tests/live/.

Source scripts (kept unchanged, they still need a running server):
    tests/live/test_api.py
    tests/live/test_optimizer_live.py
    tests/live/test_remining_live.py
    tests/live/test_parameter_propagation.py
"""
import math
import os
import unittest

from tests.offline_support import FAST_EPOCHS, OfflineApiTestCase


PLASTRON_QUERY = (
    "PlastronGlide Hydrophobic Ski Base inspired by Collembola cuticle "
    "with slip_length=0.00002 and film_thickness=0.00001"
)


class RunEndpointMockTest(OfflineApiTestCase):
    """tests/live/test_api.py: a plain mock run succeeds end to end."""

    def test_mock_run_returns_complete_result(self):
        body = self.run_pipeline(
            query="design a microtextured biomimetic drag reduction surface for high-velocity water craft",
        )

        self.assertIs(body["success"], True)
        self.assertIs(body["is_mock"], True)
        self.assertEqual(body["epochs"], FAST_EPOCHS)
        self.assertTrue(body["miner"]["design_name"])
        self.assertTrue(body["simulator"]["solver_method"])
        gain = body["simulator"]["performance_gain_pct"]
        self.assertIsInstance(gain, (int, float))
        self.assertTrue(math.isfinite(gain))
        for key in ("auditor", "validation", "optimization_history", "synthesis", "failed_concepts", "report_md"):
            self.assertIn(key, body)
        self.assertIn(body["validation"]["status"], ("pass", "warn"))

    def test_missing_api_key_falls_back_to_mock_mode(self):
        # is_mock=False without GEMINI_API_KEY must not try to reach Gemini.
        body = self.run_pipeline(query="design a drag reducing surface", is_mock=False, max_optimization_rounds=1)

        self.assertIs(body["success"], True)
        self.assertIs(body["is_mock"], True)

    def test_run_is_archived_in_data_dir(self):
        body = self.run_pipeline(query="design a drag reducing surface", max_optimization_rounds=1)

        self.assertTrue(body["report_md"].startswith("#"))
        self.assertTrue(os.listdir(os.path.join(self.data_dir, "history")))
        self.assertTrue(os.listdir(os.path.join(self.data_dir, "reports")))


class OptimizerLoopMockTest(OfflineApiTestCase):
    """tests/live/test_optimizer_live.py: closed-loop optimizer with 2 rounds."""

    def test_two_optimization_rounds_are_recorded_and_reported(self):
        body = self.run_pipeline(query="Bionic Hydro-Riblet Ski Base", max_optimization_rounds=2)

        self.assertIs(body["success"], True)
        self.assertTrue(body["miner"]["design_name"])

        history = body["optimization_history"]
        self.assertEqual(len(history), 2)
        self.assertEqual([entry["round"] for entry in history], [1, 2])
        for entry in history:
            self.assertIsInstance(entry["parameters"], dict)
            self.assertTrue(entry["parameters"])
            self.assertIn("performance_gain_pct", entry["simulator"])
            self.assertIn("validation", entry)

        # Round 1 was evaluated by the optimizer, which changed the parameters for round 2.
        self.assertTrue(history[0]["optimizer_reasoning"])
        self.assertNotEqual(history[0]["parameters"], history[1]["parameters"])

        self.assertIn("## 7. Autonome Optimierungshistorie (Closed-Loop)", body["report_md"])

    def test_single_round_skips_optimizer(self):
        body = self.run_pipeline(query="Bionic Hydro-Riblet Ski Base", max_optimization_rounds=1)

        self.assertEqual(len(body["optimization_history"]), 1)
        self.assertEqual(body["optimization_history"][0]["optimizer_reasoning"], "")


class ReminingMockTest(OfflineApiTestCase):
    """tests/live/test_remining_live.py: a failing concept triggers re-mining."""

    def test_failed_concept_is_replaced_by_alternative(self):
        # "fail" makes the mock miner return "PlastronGlide Fail-Prone Ski Base",
        # which the mock optimizer rejects (structural collapse) after round 1.
        body = self.run_pipeline(query="fail ski base", max_optimization_rounds=2)

        self.assertIs(body["success"], True)

        failed = body["failed_concepts"]
        self.assertGreaterEqual(len(failed), 1)
        self.assertEqual(failed[0]["design_name"], "PlastronGlide Fail-Prone Ski Base")
        self.assertTrue(failed[0]["reason"].startswith("Optimizer rejected:"))

        self.assertIn("Alternative", body["miner"]["design_name"])

        synthesis = body["synthesis"]
        for key in ("executive_summary", "mechanical_limits", "manufacturing_methods"):
            self.assertTrue(synthesis.get(key), f"synthesis.{key} is empty")

        report_md = body["report_md"]
        self.assertIn("## 8. Kommerzielle & Praktische Synthese", report_md)
        self.assertIn("## 9. Verlauf gescheiterter Konzepte (Re-Mining)", report_md)
        self.assertIn("PlastronGlide Fail-Prone Ski Base", report_md)


class ParameterPropagationMockTest(OfflineApiTestCase):
    """tests/live/test_parameter_propagation.py: override_parameters across runs."""

    def _baseline_and_override(self, **extra):
        baseline = self.run_pipeline(query=PLASTRON_QUERY, override_parameters=None, **extra)
        optimized = self.run_pipeline(
            query=PLASTRON_QUERY,
            override_parameters={"slip_length": 0.00004},
            previous_miner_output=baseline["miner"],
            **extra,
        )
        self.assertIs(baseline["success"], True)
        self.assertIs(optimized["success"], True)
        return baseline, optimized

    def test_override_changes_coefficient_and_gain(self):
        # Same request shape as the live script (default max_optimization_rounds).
        baseline, optimized = self._baseline_and_override()

        self.assertNotEqual(
            optimized["auditor"]["simulation_coefficient"],
            baseline["auditor"]["simulation_coefficient"],
        )
        self.assertNotEqual(
            optimized["simulator"]["performance_gain_pct"],
            baseline["simulator"]["performance_gain_pct"],
        )

    def test_previous_miner_output_is_reused(self):
        baseline, optimized = self._baseline_and_override(max_optimization_rounds=1)

        self.assertEqual(optimized["miner"]["design_name"], baseline["miner"]["design_name"])
        self.assertEqual(optimized["miner"]["parameters"], baseline["miner"]["parameters"])

    def test_override_reaches_auditor_in_first_round(self):
        _, optimized = self._baseline_and_override()

        first_round = optimized["optimization_history"][0]
        self.assertEqual(first_round["parameters"]["slip_length"], 0.00004)
        self.assertEqual(first_round["simulation_coefficient"], 0.00004)

    def test_override_is_final_audited_value_with_single_round(self):
        baseline, optimized = self._baseline_and_override(max_optimization_rounds=1)

        self.assertEqual(baseline["auditor"]["audited_parameters_dict"]["slip_length"], 0.00002)
        self.assertEqual(optimized["auditor"]["audited_parameters_dict"]["slip_length"], 0.00004)
        self.assertEqual(optimized["auditor"]["simulation_coefficient"], 0.00004)
        self.assertNotEqual(
            optimized["simulator"]["performance_gain_pct"],
            baseline["simulator"]["performance_gain_pct"],
        )

    # BUG (documented, not fixed): with the default max_optimization_rounds=3 --
    # which is what both the live script and the frontend's "Apply & Run Next
    # Optimization Round" button send -- the mock optimizer runs after round 1 and
    # rewrites *every* parameter (+10% of its bound range), including the one the
    # user just overrode. The final auditor output then reports slip_length
    # 4.99e-05 instead of the requested 4e-05, so tests/live/test_parameter_propagation.py
    # fails its first assertion against the current server as well.
    @unittest.expectedFailure
    def test_override_is_final_audited_value_with_default_rounds(self):
        _, optimized = self._baseline_and_override()

        self.assertEqual(optimized["auditor"]["audited_parameters_dict"]["slip_length"], 0.00004)


if __name__ == "__main__":
    unittest.main()
