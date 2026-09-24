# -*- coding: utf-8 -*-
"""Offline ports of the mock-mode /api/chat scripts in tests/live/.

Source scripts (kept unchanged, they still need a running server):
    tests/live/test_chat.py            run with override -> explain -> optimize
    tests/live/test_server_encoding.py UTF-8 / umlaut handling of chat responses
"""
import json
import unittest

from tests.offline_support import FAST_EPOCHS, OfflineApiTestCase, miner_param


SKI_QUERY = "design a hydrophobic ski base inspired by collembola cuticle"
OVERRIDE = {"slip_length": 0.000035}


class ChatAfterOverrideRunTest(OfflineApiTestCase):
    """tests/live/test_chat.py"""

    def _override_run(self, **extra):
        # max_optimization_rounds=1 keeps the mock optimizer from rewriting the
        # override (see ParameterPropagationMockTest in test_api_mock_flows.py).
        payload = {"query": SKI_QUERY, "override_parameters": dict(OVERRIDE), "max_optimization_rounds": 1}
        payload.update(extra)
        run = self.run_pipeline(**payload)
        self.assertIs(run["success"], True)
        self.assertEqual(run["miner"]["design_name"], "PlastronGlide Hydrophobic Ski Base")
        return run

    def test_override_reaches_audited_parameters(self):
        run = self._override_run()

        self.assertEqual(run["auditor"]["audited_parameters_dict"]["slip_length"], 0.000035)

    # BUG (documented, not fixed): tests/live/test_chat.py fails its first check
    # against the current server. It expects the override in miner.parameters,
    # which the pre-refactor web_server.py wrote into the miner output;
    # PipelineRunner now only hands overrides to the auditor, so
    # miner.parameters keeps slip_length=2e-05. (The UI shows proposed and
    # audited values side by side, so this may be an intentional change that the
    # live script never caught up with.)
    @unittest.expectedFailure
    def test_override_is_reflected_in_miner_parameters(self):
        run = self._override_run()

        self.assertEqual(miner_param(run, "slip_length"), 0.000035)

    def test_explanation_uses_run_context(self):
        run = self._override_run()

        body = self.chat("Warum ist die Reibungsreduktion 66.67%?", current_run=run).json()

        reply = body["reply"]
        # Live script: reply must contain the physics explanation ("exakt", "66.67" or "1/3").
        self.assertTrue(any(marker in reply for marker in ("exakt", "66.67", "1/3")), reply)
        # The explanation is built from the audited (overridden) slip length and
        # the simulated gain of this run, not from hard-coded defaults.
        self.assertIn("λ = 35.0 µm", reply)
        self.assertIn("h = 10.0 µm", reply)
        self.assertIn(f"{run['simulator']['performance_gain_pct']:.2f}%", reply)
        self.assertEqual(body["suggested_params"], {})

    def test_default_explanation_is_exactly_two_thirds(self):
        # Without an override (λ = 20 µm, h = 10 µm) the friction drops to exactly 1/3.
        body = self.chat("Warum ist die Reibungsreduktion 66.67%?", current_run={
            "miner": {"design_name": "PlastronGlide Hydrophobic Ski Base", "parameters": []},
            "auditor": {"audited_parameters_dict": {"slip_length": 0.00002, "film_thickness": 0.00001}},
            "simulator": {"performance_gain_pct": 66.67},
        }).json()

        self.assertIn("1/3", body["reply"])
        self.assertIn("exakt 66.67%", body["reply"])

    def test_optimization_request_suggests_parameters(self):
        run = self._override_run()

        body = self.chat("Optimiere das Design", current_run=run).json()

        self.assertIn("slip_length", body["suggested_params"])
        self.assertEqual(body["suggested_params"], {"slip_length": 0.00004, "film_thickness": 0.000005})
        self.assertIn("Apply & Run Next Optimization Round", body["reply"])

    def test_suggested_params_feed_next_run(self):
        # Full UI round trip: run -> chat suggestion -> run with the suggestion.
        run = self._override_run()
        suggestion = self.chat("Optimiere das Design", current_run=run).json()["suggested_params"]

        next_run = self.run_pipeline(
            query=SKI_QUERY,
            override_parameters=suggestion,
            previous_miner_output=run["miner"],
            max_optimization_rounds=1,
        )

        audited = next_run["auditor"]["audited_parameters_dict"]
        self.assertEqual(audited["slip_length"], 0.00004)
        self.assertEqual(audited["film_thickness"], 0.000005)

        again = self.chat("Optimiere das Design", current_run=next_run).json()
        self.assertIn("bereits maximal optimiert", again["reply"])
        self.assertEqual(again["suggested_params"], {})

    def test_missing_api_key_falls_back_to_mock_chat(self):
        body = self.chat("Hallo", current_run=None, is_mock=False).json()

        self.assertIn("reply", body)
        self.assertIn("Shark-Skin", body["reply"])

    # BUG (documented, not fixed): the mock chat decides "ski design?" with
    # `"ski" in design_name.lower()`, and "Shark-Skin ..." contains "ski". A
    # riblet design (the default mock concept) therefore gets the PlastronGlide
    # slip_length/film_thickness suggestion instead of the riblet branch, which
    # scales the run's own height/thickness and spacing/length parameters.
    # Repro: POST /api/run {"query": "design drag reducing surface", "is_mock": true}
    # (-> "Shark-Skin Inspired Riblet Foil"), then POST /api/chat
    # {"message": "Optimiere das Design", "current_run": <run>, "is_mock": true}
    # -> suggested_params == {"slip_length": 4e-05, "film_thickness": 5e-06}.
    @unittest.expectedFailure
    def test_riblet_design_gets_riblet_optimization(self):
        run = self.run_pipeline(query="design drag reducing surface", max_optimization_rounds=1)
        self.assertIn("Riblet", run["miner"]["design_name"])

        body = self.chat("Optimiere das Design", current_run=run).json()

        self.assertNotIn("slip_length", body["suggested_params"])
        riblet_names = {p["name"] for p in run["miner"]["parameters"]}
        self.assertTrue(set(body["suggested_params"]) <= riblet_names)


class ChatEncodingTest(OfflineApiTestCase):
    """tests/live/test_server_encoding.py"""

    def test_chat_response_is_utf8_json_with_umlauts(self):
        response = self.chat("Erkläre das bitte", current_run=None)

        self.assertEqual(response.headers["content-type"], "application/json; charset=utf-8")
        raw = response.content
        text = raw.decode("utf-8")  # must not raise
        body = json.loads(text)
        reply = body["reply"]
        self.assertTrue(reply)
        # Umlauts, Greek and micro signs travel as real UTF-8, not \u escapes
        # and not the "$" corruption the chat prompt warns about.
        self.assertIn("Länge".encode("utf-8"), raw)
        self.assertIn("λ".encode("utf-8"), raw)
        self.assertIn("µm".encode("utf-8"), raw)
        self.assertNotIn(b"\\u00e4", raw)
        self.assertNotIn("L$nge", reply)

    def test_umlaut_in_request_is_understood(self):
        # "Erkläre" hits the explanation branch only if the ä survives decoding.
        reply = self.chat("Erkläre das bitte", current_run=None).json()["reply"]

        self.assertIn("Reibungsreduktion", reply)
        self.assertIn("Formel für die Reibungskraft", reply)

    def test_run_response_is_utf8_json_with_umlauts(self):
        response = self.post_json("/api/run", {
            "query": "fail ski base",
            "epochs": FAST_EPOCHS,
            "is_mock": True,
            "max_optimization_rounds": 2,
        })
        self.assertEqual(response.headers["content-type"], "application/json; charset=utf-8")
        raw = response.content
        body = json.loads(raw.decode("utf-8"))
        self.assertIn("Großflächiges", body["synthesis"]["manufacturing_methods"])
        self.assertIn("Großflächiges".encode("utf-8"), raw)
        self.assertIn("Verlauf gescheiterter Konzepte", body["report_md"])


if __name__ == "__main__":
    unittest.main()
