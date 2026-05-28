# -*- coding: utf-8 -*-
import unittest

from vectornaut.benchmark import load_benchmark_cases, run_benchmark


class BenchmarkUnitTest(unittest.TestCase):
    def test_default_benchmark_cases_match_expectations(self):
        result = run_benchmark()

        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["total"], 3)
        self.assertEqual(result["failed"], 0)

    def test_default_benchmark_cases_have_ids(self):
        cases = load_benchmark_cases()

        ids = [case.get("id") for case in cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(ids))


if __name__ == "__main__":
    unittest.main()
