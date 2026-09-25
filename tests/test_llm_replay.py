# -*- coding: utf-8 -*-
import json
import os
import tempfile
import unittest

from vectornaut.config import ModelFormulation
from vectornaut.llm_replay import NeedResponse, ReplayClient


class _Config:
    def __init__(self, schema):
        self.response_schema = schema
        self.thinking_config = None


class ReplayClientTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.session = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_answer_writes_request_and_pauses(self):
        client = ReplayClient(self.session)

        with self.assertRaises(NeedResponse) as ctx:
            client.models.generate_content(model="m", contents="Löse u'' = 0", config=_Config(ModelFormulation))

        self.assertEqual(ctx.exception.index, 1)
        with open(ctx.exception.request_path, encoding="utf-8") as f:
            request = f.read()
        self.assertIn("Löse u'' = 0", request)
        self.assertIn('"governing_equation"', request)
        self.assertTrue(ctx.exception.request_path.endswith("01_ModelFormulation.md"))

    def test_pause_is_not_swallowed_by_broad_exception_handlers(self):
        client = ReplayClient(self.session)

        def stage_with_fallback():
            try:
                client.models.generate_content(model="m", contents="x", config=_Config(ModelFormulation))
            except Exception:
                return "fallback"

        with self.assertRaises(NeedResponse):
            stage_with_fallback()

    def test_existing_answer_is_parsed_into_the_schema(self):
        os.makedirs(os.path.join(self.session, "responses"))
        with open(os.path.join(self.session, "responses", "01.json"), "w", encoding="utf-8") as f:
            json.dump({
                "governing_equation": "d2u_dy2 = 0",
                "boundary_conditions": ["u(0) = 0", "u(1) = 1"],
                "independent_variables": ["y"],
                "dependent_variables": ["u"],
            }, f)
        client = ReplayClient(self.session)

        response = client.models.generate_content(model="m", contents="x", config=_Config(ModelFormulation))

        self.assertIsInstance(response.parsed, ModelFormulation)
        self.assertEqual(response.parsed.boundary_conditions, ["u(0) = 0", "u(1) = 1"])
        self.assertEqual(client.log[0]["stage"], "ModelFormulation")


if __name__ == "__main__":
    unittest.main()
