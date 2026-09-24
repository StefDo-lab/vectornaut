# -*- coding: utf-8 -*-
"""Offline tests for the model-comparison harness (vectornaut.model_eval)."""
import contextlib
import copy
import io
import json
import os
import re
import tempfile
import unittest
from unittest import mock

from vectornaut import model_eval
from vectornaut.model_eval import (
    ConfigSpecError,
    ModelConfig,
    build_markdown_report,
    env_overrides,
    extract_run_facts,
    load_reference_cases,
    parse_config_spec,
    parse_configs,
    run_case,
    run_model_eval,
    score_run,
    select_cases,
)


SUCCESS_EXPECT = {
    "outcome": "success",
    "audit_passed": True,
    "max_relative_error": 0.1,
    "max_remines": 1,
    "domain_keywords": ["fluid", "drag"],
    "dimensionality": 1,
    "solver_methods": ["analytical", "scipy", "pinn"],
    "parameter_ranges": [{"match": "viscosity", "min": 1e-7, "max": 10.0}],
}

REJECT_EXPECT = {"outcome": "rejected", "audit_passed": False, "max_remines": 2}


def synthetic_result(**changes):
    result = {
        "success": True,
        "query": "reduce drag",
        "is_mock": False,
        "miner": {
            "design_name": "Synthetic Riblet",
            "domain": "Fluid Dynamics",
            "physical_mechanism": "Riblets reduce drag.",
            "governing_equation": "d2u_dy2 = 0",
            "boundary_conditions": ["u(0) = 0", "u(1) = 1"],
            "independent_variables": ["y"],
            "dependent_variables": ["u"],
        },
        "auditor": {
            "audit_passed": True,
            "solver_method": "analytical",
            "audited_parameters_dict": {"viscosity": 0.001, "free_stream_velocity": 1.5},
            "ui_metadata": {"domain_name": "Fluid Dynamics"},
        },
        "simulator": {
            "solver_method": "analytical",
            "relative_error": 0.01,
            "performance_gain_pct": 12.0,
            "sample_points": [0.0, 0.5, 1.0],
            "solution_primary": [0.0, 0.5, 1.0],
            "solution_reference": [0.0, 0.5, 1.0],
            "primary_metric_value": 0.88,
            "reference_metric_value": 1.0,
        },
        "validation": {"status": "pass", "score": 1.0, "recommended_action": "accept", "reliability": "high"},
        "optimization_history": [{"round": 1}],
        "synthesis": {"executive_summary": "ok"},
        "failed_concepts": [],
        "models": {"miner": "m-1", "auditor": "m-1"},
    }
    for path, value in changes.items():
        target = result
        parts = path.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return result


def exhausted_error(*reasons):
    history = [{"design_name": f"C{i}", "inspiration_source": "x", "reason": r} for i, r in enumerate(reasons)]
    return f"{model_eval.EXHAUSTED_PREFIX}. Failed history: {history}"


def check_score(scored, name):
    for check in scored["checks"]:
        if check["name"] == name:
            return check["score"]
    return "absent"


class ConfigParsingTest(unittest.TestCase):
    def test_baseline_without_overrides(self):
        config = parse_config_spec("baseline=")
        self.assertEqual(config.name, "baseline")
        self.assertEqual(config.overrides, {})

    def test_multiple_assignments_and_unset(self):
        config = parse_config_spec(
            "strong=VECTORNAUT_MODEL_FORMULATOR=gemini-x-pro, VECTORNAUT_THINKING_AUDITOR=high,VECTORNAUT_MODEL="
        )
        self.assertEqual(config.overrides, {
            "VECTORNAUT_MODEL_FORMULATOR": "gemini-x-pro",
            "VECTORNAUT_THINKING_AUDITOR": "high",
            "VECTORNAUT_MODEL": None,
        })

    def test_invalid_specs_are_rejected(self):
        for spec in [
            "no_equals_sign",
            "bad name=",
            "x=VECTORNAUT_MODEL_FORMULATOR",
            "x=GEMINI_API_KEY=abc",
            "x=VECTORNAUT_DATA_DIR=/tmp",
            "x=VECTORNAUT_MODEL_FORMULATR=typo",
            "x=VECTORNAUT_THINKING_AUDITOR=extreme",
            "x=VECTORNAUT_MODEL=a,VECTORNAUT_MODEL=b",
        ]:
            with self.subTest(spec=spec), self.assertRaises(ConfigSpecError):
                parse_config_spec(spec)

    def test_parse_configs_defaults_and_duplicates(self):
        self.assertEqual([c.name for c in parse_configs([])], ["baseline"])
        with self.assertRaises(ConfigSpecError):
            parse_configs(["a=", "a=VECTORNAUT_MODEL=x"])


class EnvOverrideTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ["VECTORNAUT_MODEL"] = "ambient-model"
        os.environ.pop("VECTORNAUT_MODEL_AUDITOR", None)

    def test_overrides_applied_and_restored(self):
        with env_overrides({"VECTORNAUT_MODEL_AUDITOR": "strong", "VECTORNAUT_MODEL": None}):
            self.assertEqual(os.environ["VECTORNAUT_MODEL_AUDITOR"], "strong")
            self.assertNotIn("VECTORNAUT_MODEL", os.environ)
        self.assertNotIn("VECTORNAUT_MODEL_AUDITOR", os.environ)
        self.assertEqual(os.environ["VECTORNAUT_MODEL"], "ambient-model")

    def test_restored_on_exception(self):
        with self.assertRaises(RuntimeError):
            with env_overrides({"VECTORNAUT_MODEL": "temp", "VECTORNAUT_MODEL_AUDITOR": "temp"}):
                raise RuntimeError("boom")
        self.assertEqual(os.environ["VECTORNAUT_MODEL"], "ambient-model")
        self.assertNotIn("VECTORNAUT_MODEL_AUDITOR", os.environ)

    def test_run_case_applies_env_and_records_failure(self):
        seen = {}

        def failing_run(request):
            seen["auditor_model"] = model_eval.get_model_name("auditor")
            seen["data_dir"] = os.environ["VECTORNAUT_DATA_DIR"]
            print("[*] Starting Concept Attempt 1...")
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")

        config = parse_config_spec("strong=VECTORNAUT_MODEL_AUDITOR=strong-auditor")
        case = {"id": "c1", "query": "q", "expect": dict(SUCCESS_EXPECT)}
        previous_data_dir = os.environ.get("VECTORNAUT_DATA_DIR")
        with tempfile.TemporaryDirectory() as tmp:
            record = run_case(case, config, 1, mock=True, epochs=1, max_rounds=1, run_dir=tmp, run_fn=failing_run)
            self.assertTrue(os.path.exists(os.path.join(tmp, "pipeline.log")))
            self.assertEqual(seen["data_dir"], tmp)
        self.assertEqual(seen["auditor_model"], "strong-auditor")
        self.assertEqual(record["resolved_models"]["auditor"], "strong-auditor")
        self.assertEqual(record["resolved_models"]["miner"], "ambient-model")
        self.assertEqual(record["facts"]["category"], "api_error")
        self.assertEqual(record["facts"]["concept_attempts"], 1)
        self.assertEqual(record["score"], 0.0)
        self.assertNotIn("VECTORNAUT_MODEL_AUDITOR", os.environ)
        self.assertEqual(os.environ.get("VECTORNAUT_DATA_DIR"), previous_data_dir)

    def test_live_run_that_fell_back_to_mock_is_flagged(self):
        case = {"id": "c1", "query": "q", "expect": dict(SUCCESS_EXPECT)}
        with tempfile.TemporaryDirectory() as tmp:
            record = run_case(
                case, ModelConfig("baseline"), 1, mock=False, epochs=1, max_rounds=1, run_dir=tmp,
                run_fn=lambda request: synthetic_result(is_mock=True),
            )
        self.assertEqual(record["facts"]["category"], "mock_fallback")
        self.assertFalse(record["outcome_met"])


class ScoringTest(unittest.TestCase):
    def score(self, expect, result=None, error_type=None, error=None, log=""):
        return score_run(expect, extract_run_facts(result, error_type, error, log))

    def test_clean_pass_scores_full(self):
        scored = self.score(SUCCESS_EXPECT, synthetic_result())
        self.assertEqual(scored["score"], 1.0, scored["failed_checks"])
        self.assertTrue(scored["outcome_met"])

    def test_validator_warn_is_partial(self):
        result = synthetic_result(validation__status="warn", validation__score=0.8, simulator__relative_error=0.5)
        scored = self.score(SUCCESS_EXPECT, result)
        self.assertTrue(scored["outcome_met"])
        self.assertEqual(check_score(scored, "validator_status"), 0.5)
        self.assertEqual(check_score(scored, "relative_error"), 0.5)
        self.assertLess(scored["score"], 1.0)
        self.assertGreater(scored["score"], 0.7)

    def test_solver_failure_after_all_concepts(self):
        error = exhausted_error("Simulation solver error: a", "Simulation solver error: b", "Simulation solver error: c")
        log = "\n".join(f"[*] Starting Concept Attempt {i}..." for i in (1, 2, 3))
        facts = extract_run_facts(None, "ValueError", error, log)
        self.assertEqual(facts["category"], "solver_error")
        self.assertEqual(facts["concept_attempts"], 3)
        self.assertEqual(facts["remine_reasons"], ["solver_error"] * 3)
        scored = score_run(SUCCESS_EXPECT, facts)
        self.assertEqual(scored["score"], 0.0)
        self.assertFalse(scored["outcome_met"])

    def test_audit_failure_on_feasible_case(self):
        error = exhausted_error("Audit failed: a", "Audit failed: b", "Audit failed: c")
        scored = self.score(SUCCESS_EXPECT, None, "ValueError", error)
        self.assertFalse(scored["outcome_met"])
        self.assertEqual(check_score(scored, "audit_passed"), 0.0)

    def test_audit_failure_on_infeasible_case_is_rewarded(self):
        error = exhausted_error("Audit failed: a", "Audit failed: b", "Audit failed: c")
        scored = self.score(REJECT_EXPECT, None, "ValueError", error)
        self.assertTrue(scored["outcome_met"])
        self.assertEqual(scored["score"], 1.0)

        accepted = self.score(REJECT_EXPECT, synthetic_result())
        self.assertFalse(accepted["outcome_met"])
        self.assertEqual(accepted["score"], 0.0)

        remined = synthetic_result(failed_concepts=[{"reason": "Audit failed: impossible"}])
        partial = self.score(REJECT_EXPECT, remined)
        self.assertFalse(partial["outcome_met"])
        self.assertGreater(partial["score"], 0.0)
        self.assertLess(partial["score"], 1.0)

    def test_wrong_dimensionality_and_solver(self):
        expect = dict(SUCCESS_EXPECT, dimensionality=2, solver_methods=["pinn", "fdm"])
        scored = self.score(expect, synthetic_result())
        self.assertEqual(check_score(scored, "dimensionality"), 0.0)
        self.assertEqual(check_score(scored, "solver_method"), 0.0)
        self.assertIn("dimensionality", scored["failed_checks"])

    def test_parameter_ranges_and_domain_keywords(self):
        result = synthetic_result(auditor__audited_parameters_dict={"viscosity": 5000.0}, miner__domain="Thermodynamics",
                                  miner__design_name="X", miner__physical_mechanism="heat",
                                  auditor__ui_metadata={"domain_name": "Thermodynamics"})
        scored = self.score(SUCCESS_EXPECT, result)
        self.assertEqual(check_score(scored, "parameter_ranges"), 0.0)
        self.assertEqual(check_score(scored, "domain_keywords"), 0.0)

        unmatched = synthetic_result(auditor__audited_parameters_dict={"something_else": 1.0})
        self.assertEqual(check_score(self.score(SUCCESS_EXPECT, unmatched), "parameter_ranges"), "absent")

    def test_remines_and_dynamic_script(self):
        result = synthetic_result(
            failed_concepts=[{"reason": "Validator failed: x"}, {"reason": "Optimizer rejected: y"}],
            simulator__solver_method="dynamic_script",
            simulator__validation_passed=False,
        )
        log = "[*] Generating script iteration 1...\n[*] Generating script iteration 2...\n"
        facts = extract_run_facts(result, log_text=log)
        self.assertEqual(facts["remines"], 2)
        self.assertEqual(facts["script_iterations"], 2)
        scored = score_run(dict(SUCCESS_EXPECT, solver_methods=["dynamic_script"]), facts)
        self.assertEqual(check_score(scored, "remines"), 0.5)
        self.assertEqual(check_score(scored, "dynamic_script_validation"), 0.0)
        self.assertEqual(check_score(scored, "solver_method"), 1.0)

    def test_error_classification(self):
        self.assertEqual(model_eval.classify_error("ClientError", "400 INVALID_ARGUMENT", []), "api_error")
        self.assertEqual(model_eval.classify_error("KeyError", "'x'", []), "exception:KeyError")
        self.assertEqual(
            model_eval.classify_error("ValueError", exhausted_error("Optimizer rejected: kollaps"), []),
            "optimizer_rejected",
        )


class ReferenceCasesTest(unittest.TestCase):
    def test_reference_case_file_is_consistent(self):
        cases = load_reference_cases()
        self.assertGreaterEqual(len(cases), 8)
        known_solvers = {"analytical", "scipy", "pinn", "fdm", "dynamic_script"}
        outcomes = set()
        for case in cases:
            expect = case["expect"]
            outcomes.add(expect["outcome"])
            self.assertIn(expect.get("dimensionality"), (None, 1, 2), case["id"])
            for method in expect.get("solver_methods") or []:
                self.assertIn(method, known_solvers, case["id"])
            for spec in expect.get("parameter_ranges") or []:
                re.compile(spec["match"])
                self.assertLess(spec["min"], spec["max"], case["id"])
        self.assertEqual(outcomes, {"success", "rejected"})
        self.assertTrue(any("dynamic_script" in (c["expect"].get("solver_methods") or []) for c in cases))
        self.assertTrue(any(c["expect"].get("dimensionality") == 2 for c in cases))

    def test_select_cases(self):
        cases = load_reference_cases()
        selected = select_cases(cases, f"{cases[1]['id']},{cases[0]['id']}")
        self.assertEqual([c["id"] for c in selected], [cases[1]["id"], cases[0]["id"]])
        with self.assertRaises(ValueError):
            select_cases(cases, "does_not_exist")


class HarnessRunTest(unittest.TestCase):
    """Mirrors tests/test_api_smoke.py: every test runs against a temporary VECTORNAUT_DATA_DIR."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous_data_dir = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = self.tmp.name

    def tearDown(self):
        if self.previous_data_dir is None:
            os.environ.pop("VECTORNAUT_DATA_DIR", None)
        else:
            os.environ["VECTORNAUT_DATA_DIR"] = self.previous_data_dir
        self.tmp.cleanup()

    def _cases(self):
        return [
            {"id": "ok_case", "title": "ok", "query": "reduce drag", "expect": dict(SUCCESS_EXPECT)},
            {"id": "reject_case", "title": "reject", "query": "impossible", "expect": dict(REJECT_EXPECT)},
        ]

    def test_report_generation_with_fake_pipeline(self):
        def fake_run(request):
            if request.query == "impossible":
                if model_eval.get_model_name("auditor") == "strong":
                    raise ValueError(exhausted_error("Audit failed: a", "Audit failed: b", "Audit failed: c"))
                return synthetic_result()
            return synthetic_result(validation__status="warn", validation__score=0.7)

        configs = parse_configs(["baseline=", "strong=VECTORNAUT_MODEL_AUDITOR=strong"])
        out_dir = os.path.join(self.tmp.name, "out")
        report = run_model_eval(
            configs, self._cases(), repeats=2, mock=True, out_dir=out_dir, run_fn=fake_run, progress=io.StringIO(),
        )
        self.assertEqual(report["meta"]["completed_runs"], 8)
        summary = report["summary"]["configs"]
        self.assertGreater(summary["strong"]["mean_score"], summary["baseline"]["mean_score"])
        self.assertEqual(summary["baseline"]["failures"], {"accepted_infeasible": 2})
        self.assertEqual(summary["strong"]["outcome_rate"], 1.0)
        self.assertEqual(summary["strong"]["remine_reasons"], {"audit_rejected": 6})

        with open(report["paths"]["json"], encoding="utf-8") as fh:
            self.assertEqual(len(json.load(fh)["runs"]), 8)
        with open(report["paths"]["markdown"], encoding="utf-8") as fh:
            markdown = fh.read()
        self.assertEqual(markdown, build_markdown_report(copy.deepcopy(report)))
        for heading in ["## Summary per config", "## Per case (mean score)", "## Check breakdown", "## Notable failures"]:
            self.assertIn(heading, markdown)
        self.assertIn("| strong |", markdown)
        self.assertIn("accepted_infeasible", markdown)
        self.assertIn("MOCK", markdown)
        with open(report["paths"]["runs_jsonl"], encoding="utf-8") as fh:
            self.assertEqual(len(fh.readlines()), 8)

    def test_refuses_live_run_without_api_key(self):
        called = []
        stderr, stdout = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ):
            os.environ.pop("GEMINI_API_KEY", None)
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                code = model_eval.main(["--config", "baseline=", "--cases", "riblet_hose_de"], run_fn=called.append)
        self.assertEqual(code, 2)
        self.assertEqual(called, [])
        self.assertIn("GEMINI_API_KEY", stderr.getvalue())
        self.assertIn("1 pipeline run(s)", stdout.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "model_evals")))

    def test_dry_run_prints_live_estimate(self):
        stdout = io.StringIO()
        with mock.patch.dict(os.environ):
            os.environ.pop("GEMINI_API_KEY", None)
            with contextlib.redirect_stdout(stdout):
                code = model_eval.main(["--config", "a=", "--config", "b=VECTORNAUT_MODEL=x", "--repeats", "3", "--dry-run"])
        self.assertEqual(code, 0)
        n_cases = len(load_reference_cases())
        self.assertIn(f"= {2 * n_cases * 3} pipeline run(s)", stdout.getvalue())
        self.assertIn("Gemini calls", stdout.getvalue())

    def test_end_to_end_mock_run(self):
        stdout = io.StringIO()
        with mock.patch.dict(os.environ):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("VECTORNAUT_MODEL", None)
            with contextlib.redirect_stdout(stdout):
                code = model_eval.main([
                    "--mock", "--cases", "riblet_watercraft_en,plastron_ski",
                    "--config", "baseline=", "--config", "strong=VECTORNAUT_MODEL=some-stronger-model",
                    "--epochs", "5", "--max-rounds", "1",
                ])
            self.assertNotIn("VECTORNAUT_MODEL", os.environ)
        self.assertEqual(code, 0, stdout.getvalue())

        eval_root = os.path.join(self.tmp.name, "model_evals")
        (run_dir_name,) = os.listdir(eval_root)
        with open(os.path.join(eval_root, run_dir_name, "model_eval.json"), encoding="utf-8") as fh:
            report = json.load(fh)
        self.assertTrue(report["meta"]["mock"])
        self.assertEqual(len(report["runs"]), 4)
        for run in report["runs"]:
            self.assertTrue(run["facts"]["success"], run["facts"])
            self.assertEqual(run["facts"]["dimensionality"], 1)
            self.assertEqual(run["models"], {})
            self.assertTrue(os.path.exists(os.path.join(run["run_dir"], "result.json")))
        strong_runs = [run for run in report["runs"] if run["config"] == "strong"]
        self.assertEqual(strong_runs[0]["resolved_models"]["formulator"], "some-stronger-model")
        self.assertTrue(os.path.exists(os.path.join(eval_root, run_dir_name, "model_eval.md")))
        self.assertIn("mock mode", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
