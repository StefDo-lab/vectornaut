# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

import web_server


def sample_run():
    return {
        "miner": {
            "design_name": "Smoke Concept",
            "domain": "Fluid Dynamics",
            "governing_equation": "d2u_dy2 = 0",
        },
        "auditor": {"audit_passed": True},
        "simulator": {
            "solver_method": "analytical",
            "relative_error": 0.0,
            "performance_gain_pct": 10.0,
            "sample_points": [0.0, 1.0],
            "solution_primary": [0.0, 1.0],
            "solution_reference": [0.0, 1.0],
            "primary_metric_value": 0.8,
            "reference_metric_value": 1.0,
        },
        "synthesis": {"executive_summary": "ok"},
        "report_md": "# ok",
    }


class ApiSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous_data_dir = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = self.tmp.name
        self.client = TestClient(web_server.app)

    def tearDown(self):
        if self.previous_data_dir is None:
            os.environ.pop("VECTORNAUT_DATA_DIR", None)
        else:
            os.environ["VECTORNAUT_DATA_DIR"] = self.previous_data_dir
        self.tmp.cleanup()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_history_endpoint_starts_empty(self):
        response = self.client.get("/api/history?limit=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["runs"], [])

    def test_eval_judge_endpoint(self):
        response = self.client.post("/api/eval/judge", json={"run": sample_run(), "criteria": {}})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["passed"])

    def test_chat_endpoint_mock(self):
        response = self.client.post(
            "/api/chat",
            json={
                "message": "Erklär mir die Testergebnisse",
                "is_mock": True,
                "current_run": sample_run(),
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("reply", response.json())

    def test_run_endpoint_mock_archives_to_temp_storage(self):
        response = self.client.post(
            "/api/run",
            json={
                "query": "design drag reducing surface",
                "is_mock": True,
                "epochs": 1,
                "max_optimization_rounds": 1,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertIn(body["validation"]["status"], ["pass", "warn"])

        history_response = self.client.get("/api/history?limit=5")
        self.assertEqual(history_response.status_code, 200)
        self.assertGreaterEqual(len(history_response.json()["runs"]), 1)

    def test_solver_compare_endpoint_uses_current_run(self):
        run_response = self.client.post(
            "/api/run",
            json={
                "query": "design drag reducing surface",
                "is_mock": True,
                "epochs": 1,
                "max_optimization_rounds": 1,
            },
        )
        self.assertEqual(run_response.status_code, 200)

        compare_response = self.client.post(
            "/api/run/solver-compare",
            json={
                "current_run": run_response.json(),
                "is_mock": True,
                "epochs": 1,
                "methods": ["analytical", "scipy"],
            },
        )
        self.assertEqual(compare_response.status_code, 200)
        body = compare_response.json()
        self.assertTrue(body["success"])
        self.assertGreaterEqual(len(body["results"]), 1)
        self.assertIn("solver_method", body["results"][0])


if __name__ == "__main__":
    unittest.main()
