# -*- coding: utf-8 -*-
"""Sandbox for model-generated scripts (LIVE_PATH_FINDINGS 6 and 12)."""
import json
import os
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock
from unittest.mock import MagicMock

from vectornaut.config import (
    AuditedParameter, AuditorOutput, AxisMetadata, DimensionlessNumber, MetricMetadata,
    MinerOutput, ParameterProposal, UIMetadata,
)
from vectornaut.sandbox import check_generated_code, run_generated_script, sandbox_workdir
from vectornaut.script_generator import GeneratedScriptResponse, ScriptGenerator
from vectornaut.test_generator import GeneratedTestScriptResponse, TestScriptGenerator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSIX = os.name == "posix"


def _write(workdir, name, code):
    with open(os.path.join(workdir, name), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(code))


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


class RunGeneratedScriptTest(unittest.TestCase):
    def test_environment_is_scrubbed(self):
        secrets = {
            "GEMINI_API_KEY": "gemini-secret",
            "GOOGLE_APPLICATION_CREDENTIALS": "/secret.json",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "AZURE_CLIENT_SECRET": "azure-secret",
            "GITHUB_TOKEN": "gh-secret",
            "DB_PASSWORD": "pw",
            "VECTORNAUT_DATA_DIR": "/data",
        }
        with mock.patch.dict(os.environ, secrets), sandbox_workdir() as wd:
            _write(wd, "env.py", """
                import json, os
                print(json.dumps(dict(os.environ)))
            """)
            run = run_generated_script(["env.py"], timeout=30, workdir=wd)
            self.assertEqual(run.returncode, 0, run.stderr)
            env = json.loads(run.stdout)
            home = env["HOME"]
            self.assertTrue(os.path.realpath(home).startswith(os.path.realpath(wd)))

        for name, value in secrets.items():
            self.assertNotIn(name, env)
            self.assertNotIn(value, json.dumps(env))
        self.assertFalse([k for k in env if k.startswith("VECTORNAUT_")])
        self.assertEqual(env["MPLBACKEND"], "Agg")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["HTTPS_PROXY"], "http://127.0.0.1:9")

    def test_runs_in_fresh_workdir_outside_repo(self):
        with sandbox_workdir() as wd:
            _write(wd, "cwd.py", "import os\nprint(os.getcwd())\n")
            run = run_generated_script(["cwd.py"], timeout=30, workdir=wd)
            cwd = run.stdout.strip()
            self.assertEqual(os.path.realpath(cwd), os.path.realpath(wd))
        self.assertFalse(os.path.realpath(cwd).startswith(os.path.realpath(REPO_ROOT) + os.sep))
        self.assertFalse(os.path.exists(wd), "workdir is removed afterwards")

    def test_timeout_kills_the_whole_process_group(self):
        with sandbox_workdir() as wd:
            # parent -> child -> grandchild, all sleeping; pids are written to files.
            _write(wd, "spawn.py", """
                import os, subprocess, sys, time
                level = int(sys.argv[1])
                with open(f"pid{level}.txt", "w") as f:
                    f.write(str(os.getpid()))
                if level < 2:
                    subprocess.Popen([sys.executable, "spawn.py", str(level + 1)])
                time.sleep(60)
            """)
            start = time.monotonic()
            run = run_generated_script(["spawn.py", "0"], timeout=3, workdir=wd)
            elapsed = time.monotonic() - start
            pids = []
            for level in range(3):
                path = os.path.join(wd, f"pid{level}.txt")
                if os.path.exists(path):
                    with open(path) as f:
                        pids.append(int(f.read()))

        self.assertTrue(run.timed_out)
        self.assertIsNone(run.returncode)
        self.assertLess(elapsed, 30)
        self.assertEqual(len(pids), 3, "child and grandchild were started")
        if POSIX:
            deadline = time.monotonic() + 5
            while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertFalse([pid for pid in pids if _alive(pid)], "grandchildren survived the timeout")

    @unittest.skipUnless(POSIX, "resource limits are POSIX-only")
    def test_memory_and_file_size_limits(self):
        with mock.patch.dict(os.environ, {"VECTORNAUT_SANDBOX_MEMORY_MB": "512", "VECTORNAUT_SANDBOX_FILE_SIZE_MB": "1"}), \
                sandbox_workdir() as wd:
            _write(wd, "limits.py", """
                import resource, sys
                print(resource.getrlimit(resource.RLIMIT_AS)[0], resource.getrlimit(resource.RLIMIT_CPU)[0])
                try:
                    block = bytearray(1024 * 1024 * 1024)
                    print("allocated")
                except MemoryError:
                    print("memory-limited")
                try:
                    with open("big.bin", "wb") as f:
                        f.write(b"0" * (2 * 1024 * 1024))
                    print("wrote")
                except OSError:
                    print("file-limited")
            """)
            run = run_generated_script(["limits.py"], timeout=20, workdir=wd)
        lines = run.stdout.split()
        self.assertEqual(int(lines[0]), 512 * 1024 * 1024)
        self.assertEqual(int(lines[1]), 20 + 5)
        self.assertIn("memory-limited", lines)
        self.assertIn("file-limited", lines)


class StaticCheckTest(unittest.TestCase):
    def test_forbidden_code_is_rejected(self):
        forbidden = [
            "import socket",
            "import urllib.request",
            "from http.client import HTTPConnection",
            "import requests",
            "import ctypes",
            "from subprocess import run",
            "import subprocess as sp",
            "import os\nos.system('ls')",
            "import os as o\no.popen('ls')",
            "from os import system",
            "import shutil\nshutil.rmtree('/')",
            "__import__('socket')",
            "eval('1+1')",
            "import os\ngetattr(os, 'system')('ls')",
        ]
        for code in forbidden:
            with self.subTest(code=code):
                self.assertTrue(check_generated_code(code))

    def test_numerical_code_passes(self):
        code = textwrap.dedent("""
            import argparse, json, math, os, sys
            import numpy as np
            from scipy.integrate import solve_ivp
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.remove("x") if os.path.exists("x") else None
            print(os.path.join("a", "b"), sys.argv)
        """)
        self.assertEqual(check_generated_code(code), [])


def _miner():
    return MinerOutput(
        design_name="Sandbox Toy",
        inspiration_source="test",
        domain="Fluid Dynamics",
        physical_mechanism="test",
        parameters=[ParameterProposal(name="viscosity", value=0.001, min_bound=0.0001, max_bound=0.01, justification="t")],
        governing_equation="d2u_dy2 = -1.0",
        boundary_conditions=["u(0) = 0"],
        independent_variables=["y"],
        dependent_variables=["u"],
        svg_schematic="<svg></svg>",
    )


def _auditor():
    return AuditorOutput(
        audit_passed=True,
        audit_notes="ok",
        audited_parameters=[AuditedParameter(name="viscosity", value=0.001)],
        dimensionless_numbers=[DimensionlessNumber(name="Re", value=1.0)],
        simulation_coefficient=0.001,
        solver_method="dynamic_script",
        ui_metadata=UIMetadata(
            domain_name="Fluid",
            independent_var=AxisMetadata(label="y", unit="m"),
            dependent_var=AxisMetadata(label="u", unit="m/s"),
            primary_metric=MetricMetadata(label="P"),
            reference_metric=MetricMetadata(label="R"),
            performance_gain=MetricMetadata(label="G"),
        ),
    )


def _client(*codes, schema=GeneratedScriptResponse):
    responses = []
    for code in codes:
        response = MagicMock()
        response.parsed = schema(explanation="x", code=code)
        responses.append(response)
    client = MagicMock()
    client.models.generate_content.side_effect = responses
    return client


GOOD_SOLVER = """
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--params"); ap.add_argument("--output"); ap.add_argument("--plot")
a = ap.parse_args()
out = {"success": SUCCESS, "performance_gain_pct": 10.0, "relative_error": 0.001,
       "sample_points": [0.0, 1.0], "solution_primary": [1.0, 2.0], "solution_reference": [1.0, 1.5],
       "primary_metric_value": 2.0, "reference_metric_value": 1.5}
with open(a.output, "w", encoding="utf-8") as f:
    json.dump(out, f)
with open(a.plot, "wb") as f:
    f.write(b"PNG")
"""


class GeneratorSandboxIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"VECTORNAUT_DATA_DIR": self._tmp.name})
        self.env.start()
        self.plot = os.path.join(self._tmp.name, "plot.png")

    def tearDown(self):
        self.env.stop()
        self._tmp.cleanup()

    def test_forbidden_import_triggers_self_correction(self):
        client = _client("import socket\n" + GOOD_SOLVER.replace("SUCCESS", "True"), GOOD_SOLVER.replace("SUCCESS", "True"))
        result = ScriptGenerator(client=client).generate_and_execute(_miner(), _auditor(), plot_png_path=self.plot)

        self.assertTrue(result["success"])
        self.assertEqual(client.models.generate_content.call_count, 2)
        correction_prompt = client.models.generate_content.call_args_list[1].kwargs["contents"]
        self.assertIn("REJECTED", correction_prompt)
        self.assertIn("socket", correction_prompt)

    def test_exit_code_zero_with_success_false_triggers_self_correction(self):
        client = _client(GOOD_SOLVER.replace("SUCCESS", "False"), GOOD_SOLVER.replace("SUCCESS", "True"))
        result = ScriptGenerator(client=client).generate_and_execute(_miner(), _auditor(), plot_png_path=self.plot)

        self.assertTrue(result["success"])
        self.assertEqual(client.models.generate_content.call_count, 2)
        correction_prompt = client.models.generate_content.call_args_list[1].kwargs["contents"]
        self.assertIn('"success" is False', correction_prompt)
        self.assertTrue(os.path.exists(self.plot))

    def _run_tests(self, *codes):
        solver = os.path.join(self._tmp.name, "solver.py")
        params = os.path.join(self._tmp.name, "params.json")
        with open(solver, "w", encoding="utf-8") as f:
            f.write(GOOD_SOLVER.replace("SUCCESS", "True"))
        with open(params, "w", encoding="utf-8") as f:
            json.dump({"viscosity": 0.001}, f)
        client = _client(*codes, schema=GeneratedTestScriptResponse)
        result = TestScriptGenerator(client=client).generate_and_execute_tests(_miner(), _auditor(), solver, params)
        return result, client

    def test_pass_fail_comes_from_unittest_not_self_report(self):
        # The module claims success in its own JSON, but one unittest fails.
        code = textwrap.dedent("""
            import json, unittest
            from solver_harness import run_solver, nominal_params

            class SolverPhysicalValidation(unittest.TestCase):
                def test_nominal_run(self):
                    run = run_solver(nominal_params())
                    self.assertEqual(run.returncode, 0)
                    self.assertTrue(run.output["success"])

                def test_physical_invariants(self):
                    self.assertGreater(run_solver({"viscosity": 0.0}).output["performance_gain_pct"], 50.0)

            if __name__ == "__main__":
                with open("results.json", "w", encoding="utf-8") as f:
                    json.dump({"success": True, "total_run": 2, "total_failures": 0}, f)
        """)
        result, client = self._run_tests(code)

        self.assertEqual(client.models.generate_content.call_count, 1)
        self.assertFalse(result["success"])
        self.assertEqual(result["total_run"], 2)
        self.assertEqual(result["total_failures"], 1)
        by_name = {t["name"]: t for t in result["test_results"]}
        self.assertTrue(by_name["test_nominal_run"]["passed"])
        self.assertFalse(by_name["test_physical_invariants"]["passed"])
        with open(result["test_output_path"], encoding="utf-8") as f:
            self.assertFalse(json.load(f)["success"])

    def test_old_subprocess_contract_is_rejected_and_corrected(self):
        old = "import subprocess, unittest\nclass T(unittest.TestCase):\n    def test_nominal_run(self):\n        pass\n"
        new = "import unittest\nclass T(unittest.TestCase):\n    def test_nominal_run(self):\n        pass\n"
        result, client = self._run_tests(old, new)

        self.assertTrue(result["success"])
        self.assertEqual(client.models.generate_content.call_count, 2)
        correction_prompt = client.models.generate_content.call_args_list[1].kwargs["contents"]
        self.assertIn("solver_harness", correction_prompt)

    def test_module_without_tests_triggers_correction(self):
        result, client = self._run_tests("x = 1\n", "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n")

        self.assertTrue(result["success"])
        self.assertEqual(client.models.generate_content.call_count, 2)


if __name__ == "__main__":
    unittest.main()
