# -*- coding: utf-8 -*-
"""
Time budget of the symbolic (SymPy) solve and the on-disk solver memo.

The recorded case (facade-cooling explorer run, 2026-09-27, e00007 "Cactus-spine pile facade shade"):
an exponential source with two Robin boundary conditions; sp.dsolve/sp.solve ran ~20 min although the
auditor had asked for SciPy. The symbolic solve now runs in a worker process under
VECTORNAUT_SYMBOLIC_TIMEOUT_S (default 20 s) and is terminated when it runs over; the SciPy fallback
then serves, and solver_note says why.
"""
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from tests.solver_accuracy_helpers import Case, run_case
from vectornaut.solvers import solvers_1d as s1
from vectornaut.solvers.parsing import get_domain_bounds, parse_equation_and_bcs

CACTUS_EQ = ("d2T_dx2 = pile_depth**2 / pile_effective_conductivity * (volumetric_air_exchange_coefficient * "
             "(T - ambient_temperature) - (1 - rod_solar_reflectance) * extinction_coefficient * "
             "solar_irradiance_daily_mean * exp(-extinction_coefficient * pile_depth * (1 - x)))")
CACTUS_BCS = [
    "pile_effective_conductivity / pile_depth * dT_dx(0) + wall_solar_absorptance * solar_irradiance_daily_mean * "
    "exp(-extinction_coefficient * pile_depth) - wall_thermal_emittance * net_sky_longwave * "
    "exp(-extinction_coefficient * pile_depth) = wall_u_value * (T(0) - indoor_temperature)",
    "pile_effective_conductivity / pile_depth * dT_dx(1) = external_heat_transfer_coefficient * "
    "(ambient_temperature - T(1)) - pile_top_emittance * net_sky_longwave",
]
CACTUS_PARAMS = {
    "pile_depth": 0.04, "extinction_coefficient": 60.0, "rod_solar_reflectance": 0.8,
    "pile_effective_conductivity": 0.2, "volumetric_air_exchange_coefficient": 500.0, "wall_u_value": 0.5,
    "solar_irradiance_daily_mean": 180.0, "ambient_temperature": 308.15, "indoor_temperature": 298.15,
    "wall_solar_absorptance": 0.33, "wall_thermal_emittance": 0.9, "net_sky_longwave": 40.0,
    "external_heat_transfer_coefficient": 15.0, "pile_top_emittance": 0.5,
}
COUETTE_SLIP = Case("couette_slip", "d2u_dy2 = 0", ["u(0) = slip_length * du_dy(0)", "u(1) = U"], ["y"], "u",
                    exact=lambda y: 2.0 * (y + 0.25) / 1.25, params={"U": 2.0}, coefficient=0.25)


def _parse(eq, bcs, x_name, y_name, params):
    lo, hi = get_domain_bounds(bcs, params, dependent_var=y_name)
    rhs, _, x, y, syms = parse_equation_and_bcs(eq, bcs, x_name, y_name, params)
    return rhs, bcs, x, y, syms, params, lo, hi


def _slow_impl(*args):
    time.sleep(60)
    raise AssertionError("not reached")


class _Env(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vectornaut_symbolic_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = {"VECTORNAUT_DATA_DIR": self.tmp}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(s1.SOLVER_MEMO_ENV, None)

    def budget(self, seconds):
        return mock.patch.dict(os.environ, {s1.SYMBOLIC_TIMEOUT_ENV: str(seconds)})


class SymbolicTimeoutTest(_Env):
    def test_budget_from_the_environment(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(s1.SYMBOLIC_TIMEOUT_ENV, None)
            self.assertEqual(s1.symbolic_timeout_s(), 20.0)
        for raw, expected in (("5", 5.0), ("0.5", 0.5), ("0", 0.0), ("-1", 0.0), ("junk", 20.0), ("", 20.0)):
            with self.budget(raw):
                self.assertEqual(s1.symbolic_timeout_s(), expected, raw)

    def test_recorded_cactus_model_is_cut_off_and_scipy_still_solves_it(self):
        args = _parse(CACTUS_EQ, CACTUS_BCS, "x", "T", CACTUS_PARAMS)
        start = time.perf_counter()
        with self.budget(1.0), self.assertRaises(s1.SymbolicTimeoutError) as ctx:
            s1.solve_analytical(*args)
        self.assertLess(time.perf_counter() - start, 15.0)
        self.assertIsInstance(ctx.exception, s1.SolverResultError)       # the dispatcher's fallback catches it
        self.assertIn("time budget of 1 s", str(ctx.exception))
        _, deriv = s1.solve_scipy_bvp(*args)
        self.assertTrue(abs(deriv) < 1e3)

    def test_results_and_errors_come_back_from_the_worker(self):
        args = _parse("d2u_dy2 = -G / mu", ["u(0) = 0", "u(1) = 0"], "y", "u", {"G": 2.0, "mu": 0.5})
        with self.budget(30):
            expr, deriv = s1.solve_analytical(*args)
        self.assertAlmostEqual(float(expr.subs(args[2], 0.5)), 0.5, places=12)
        self.assertAlmostEqual(deriv, 2.0, places=12)
        with self.budget(30), mock.patch.object(s1, "_solve_analytical_impl",
                                                side_effect=s1.SolverResultError("closed form is not finite")):
            with self.assertRaises(s1.SolverResultError) as ctx:
                s1.solve_analytical(*args)
        self.assertIn("closed form is not finite", str(ctx.exception))
        # Budget 0: in-process, as before (no worker).
        with self.budget(0), mock.patch.object(s1, "_run_with_budget", side_effect=AssertionError("no worker")):
            self.assertAlmostEqual(s1.solve_analytical(*args)[1], 2.0, places=12)

    def test_dispatcher_falls_back_to_scipy_and_skips_the_baseline_symbolic_solve(self):
        with self.budget(0.5), mock.patch.object(s1, "_solve_analytical_impl", _slow_impl):
            start = time.perf_counter()
            run = run_case(COUETTE_SLIP, "scipy", record=False)
            elapsed = time.perf_counter() - start
        # One budget for the design; the baseline (same equation) goes straight to SciPy.
        self.assertLess(elapsed, 10.0)
        self.assertEqual(run.sim.solver_method, "scipy")
        note = run.sim.solver_note or ""
        self.assertIn("analytical: symbolic solve exceeded the time budget of 0.5 s", note)
        self.assertIn("baseline: symbolic solve not attempted: it exceeded the time budget for the design", note)
        self.assertAlmostEqual(run.sim.performance_gain_pct, 20.0, delta=1e-4)
        self.assertLess(run.errors()["rel_max"], 1e-6)

    def test_analytical_request_reports_the_timeout(self):
        with self.budget(0.5), mock.patch.object(s1, "_solve_analytical_impl", _slow_impl):
            run = run_case(COUETTE_SLIP, "analytical", record=False)
        self.assertEqual(run.sim.solver_method, "scipy")
        self.assertIn("time budget", run.sim.solver_note or "")

    def test_pinn_uses_scipy_as_reference_without_a_symbolic_solve(self):
        from vectornaut import solver_dispatcher
        with mock.patch.object(solver_dispatcher, "solve_analytical",
                               side_effect=AssertionError("symbolic solve must be skipped")):
            run = run_case(COUETTE_SLIP, "pinn", epochs=20, record=False)
        self.assertEqual(run.sim.solver_method, "pinn")
        self.assertNotIn("analytical", run.sim.solver_note or "")
        self.assertAlmostEqual(run.sim.reference_metric_value, 1.6, delta=1e-6)   # SciPy: du/dy(0) = 2 / 1.25


class SolverMemoTest(_Env):
    def memo(self):
        return mock.patch.dict(os.environ, {s1.SOLVER_MEMO_ENV: os.path.join(self.tmp, "memo")})

    def test_memo_is_off_by_default(self):
        self.assertIsNone(s1.solver_memo_dir())
        args = _parse("d2u_dy2 = -G / mu", ["u(0) = 0", "u(1) = 0"], "y", "u", {"G": 2.0, "mu": 0.5})
        s1.solve_analytical(*args)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "memo")))

    def test_results_failures_and_timeouts_are_memoised(self):
        args = _parse("d2u_dy2 = -G / mu", ["u(0) = 0", "u(1) = 0"], "y", "u", {"G": 2.0, "mu": 0.5})
        with self.memo():
            expr, deriv = s1.solve_analytical(*args)
            files = os.listdir(os.path.join(self.tmp, "memo", "analytical"))
            self.assertEqual(len(files), 1)
            with mock.patch.object(s1, "_solve_analytical_impl", side_effect=AssertionError("must come from the memo")), \
                    mock.patch.object(s1, "_run_with_budget", side_effect=AssertionError("must come from the memo")):
                again, deriv_again = s1.solve_analytical(*args)
            self.assertEqual((str(again), deriv_again), (str(expr), deriv))
            # Other parameter values are another key.
            other = _parse("d2u_dy2 = -G / mu", ["u(0) = 0", "u(1) = 0"], "y", "u", {"G": 4.0, "mu": 0.5})
            self.assertAlmostEqual(s1.solve_analytical(*other)[1], 4.0, places=12)
            self.assertEqual(len(os.listdir(os.path.join(self.tmp, "memo", "analytical"))), 2)

            slow = _parse("d2u_dy2 = -G / mu * y", ["u(0) = 0", "u(1) = 0"], "y", "u", {"G": 2.0, "mu": 0.5})
            with self.budget(0.3), mock.patch.object(s1, "_solve_analytical_impl", _slow_impl):
                with self.assertRaises(s1.SymbolicTimeoutError):
                    s1.solve_analytical(*slow)
            # A memoised timeout is reused while the budget is not larger ...
            start = time.perf_counter()
            with self.budget(0.3), self.assertRaises(s1.SymbolicTimeoutError) as ctx:
                s1.solve_analytical(*slow)
            self.assertLess(time.perf_counter() - start, 0.25)
            self.assertIn("memoised", str(ctx.exception))
            # ... and a larger budget tries again (here it solves).
            with self.budget(30):
                self.assertTrue(s1.solve_analytical(*slow)[0] is not None)

    def test_replay_sessions_use_the_memo_by_default(self):
        from vectornaut import llm_replay
        session = os.path.join(self.tmp, "session")
        self.assertEqual(llm_replay.solver_memo_env(session), {s1.SOLVER_MEMO_ENV: os.path.join(session, "solver_memo")})
        self.assertEqual(llm_replay.solver_memo_env(session, enabled=False), {})
        seen = {}

        class Runner:
            def __init__(self, *args, **kwargs):
                seen["memo"] = os.environ.get(s1.SOLVER_MEMO_ENV)

            def run(self, rounds, batch):
                raise RuntimeError("stop")

        with mock.patch("vectornaut.explorer.run.ExplorerRunner", Runner):
            status = llm_replay.run_explorer_session(session, "business", "q", 1, 1)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(seen["memo"], os.path.join(session, "solver_memo"))
        with mock.patch("vectornaut.explorer.run.ExplorerRunner", Runner):
            llm_replay.run_explorer_session(session, "business", "q", 1, 1, solver_memo=False)
        self.assertIsNone(seen["memo"])


if __name__ == "__main__":
    unittest.main()
