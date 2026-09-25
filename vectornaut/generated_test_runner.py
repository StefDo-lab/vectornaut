# -*- coding: utf-8 -*-
"""
Trusted runner for model-generated validation test scripts.

This file is copied into the sandbox working directory and executed there (it does
not import vectornaut). It

1. registers a module ``solver_harness`` that gives the tests a safe way to run the
   generated solver: ``run_solver(params)`` and ``nominal_params()``; the test script
   itself must not spawn processes,
2. imports the generated test script as a module (its ``__main__`` block does not run),
3. runs every ``unittest.TestCase`` in it and writes a JSON summary derived from the
   unittest result object - the test script's own claims are not used.

Exit codes: 0 all tests passed, 1 tests failed/errored, 2 the test script could not be
imported or contained no tests.

Usage:
    python generated_test_runner.py --tests T.py --solver S.py --params P.json
        --summary OUT.json --solver-timeout 60
"""
import argparse
import collections
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
import types
import unittest

SolverRun = collections.namedtuple("SolverRun", ["returncode", "output", "stderr"])


def _build_harness(solver_path, nominal, solver_timeout, runs_dir):
    counter = {"n": 0}

    def nominal_params():
        """Fresh copy of the nominal (audited) parameters."""
        return dict(nominal)

    def run_solver(params=None, timeout=None):
        """
        Run the solver with ``params`` (dict; defaults to the nominal parameters).
        Returns SolverRun(returncode, output, stderr); ``output`` is the parsed
        --output JSON or None if the solver wrote none. Temporary files are handled here.
        """
        counter["n"] += 1
        run_dir = tempfile.mkdtemp(prefix=f"run{counter['n']:03d}_", dir=runs_dir)
        params_path = os.path.join(run_dir, "params.json")
        output_path = os.path.join(run_dir, "results.json")
        plot_path = os.path.join(run_dir, "plot.png")
        with open(params_path, "w", encoding="utf-8") as fh:
            json.dump(dict(nominal) if params is None else dict(params), fh)
        try:
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", solver_path,
                 "--params", params_path, "--output", output_path, "--plot", plot_path],
                cwd=run_dir, stdin=subprocess.DEVNULL, capture_output=True,
                encoding="utf-8", errors="replace",
                timeout=float(timeout or solver_timeout),
            )
            returncode, stderr = proc.returncode, proc.stderr or proc.stdout or ""
        except subprocess.TimeoutExpired:
            returncode, stderr = -9, f"solver timed out after {timeout or solver_timeout} s"
        output = None
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as fh:
                    output = json.load(fh)
            except (OSError, ValueError) as err:
                stderr += f"\ninvalid output JSON: {err}"
        return SolverRun(returncode, output, stderr)

    def is_finite_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

    module = types.ModuleType("solver_harness")
    module.SolverRun = SolverRun
    module.run_solver = run_solver
    module.nominal_params = nominal_params
    module.NOMINAL_PARAMS = dict(nominal)
    module.is_finite_number = is_finite_number
    return module


def _short(tb_text):
    lines = [line for line in (tb_text or "").strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _collect(result, suite_names):
    outcome = {}
    for test, tb in result.failures:
        outcome[test.id()] = ("failed", _short(tb))
    for test, tb in result.errors:
        outcome[test.id()] = ("error", _short(tb))
    for test, reason in result.skipped:
        outcome[test.id()] = ("skipped", str(reason))
    for test, tb in result.expectedFailures:
        outcome[test.id()] = ("expected_failure", _short(tb))
    for test in result.unexpectedSuccesses:
        outcome[test.id()] = ("unexpected_success", "")
    tests = []
    for test_id in suite_names:
        status, message = outcome.get(test_id, ("passed", ""))
        tests.append({
            "name": test_id.split(".")[-1],
            "id": test_id,
            "status": status,
            "passed": status in ("passed", "expected_failure"),
            "message": message,
        })
    # Errors raised outside individual tests (setUpClass, module fixtures).
    for test_id, (status, message) in outcome.items():
        if test_id not in suite_names:
            tests.append({"name": test_id, "id": test_id, "status": status, "passed": False, "message": message})
    return tests


def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--solver-timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    summary_path = os.path.abspath(args.summary)
    tests_path = os.path.abspath(args.tests)
    solver_path = os.path.abspath(args.solver)
    runs_dir = os.path.abspath("solver_runs")
    os.makedirs(runs_dir, exist_ok=True)

    with open(args.params, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    nominal = {k: v for k, v in raw.items() if isinstance(v, (int, float)) and not isinstance(v, bool)} if isinstance(raw, dict) else {}

    def write_summary(data):
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    sys.modules["solver_harness"] = _build_harness(solver_path, nominal, args.solver_timeout, runs_dir)
    # The test module sees neither the runner's arguments nor the summary path.
    sys.argv = [tests_path]

    try:
        spec = importlib.util.spec_from_file_location("generated_validation_tests", tests_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_validation_tests"] = module
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    except BaseException:
        write_summary({"collected": False, "import_error": traceback.format_exc()[-6000:]})
        return 2

    tests = list(_iter_tests(suite))
    names = [t.id() for t in tests]
    if not tests:
        write_summary({"collected": False, "import_error": "The test script defines no unittest.TestCase test methods."})
        return 2

    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    details = _collect(result, names)
    skipped = len(result.skipped)
    executed = result.testsRun - skipped
    success = bool(result.wasSuccessful() and executed > 0 and not result.unexpectedSuccesses)
    write_summary({
        "collected": True,
        "success": success,
        "total_run": result.testsRun,
        "total_failures": len(result.failures),
        "total_errors": len(result.errors),
        "total_skipped": skipped,
        "test_results": details,
    })
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
