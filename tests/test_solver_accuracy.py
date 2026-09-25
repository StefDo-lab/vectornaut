# -*- coding: utf-8 -*-
"""Accuracy tests for the deterministic solvers behind ``dispatch_and_solve``.

Every case here has a closed-form solution, and the solver output
(``solution_primary`` at ``sample_points``, the wall/mean metric and
``performance_gain_pct``) is compared against that exact answer. This is
independent of the app's own ``solution_reference``, which is produced by the
same code paths.

Solver selection (see vectornaut/solver_dispatcher.py):
  1D: 'analytical' (SymPy dsolve), 'scipy' (solve_bvp), 'pinn' (1D PINN, falls
      back to scipy on error). The reference is analytical if dsolve works, else
      scipy. sample_points = 20 equispaced points of the domain inferred from the
      BCs; primary_metric_value = du/dx at the lower wall.
  2D: 'fdm' (Jacobi, fixed 20x20 grid on the unit square, h = 1/19) or 'pinn'
      (falls back to fdm). The reference is always FDM; primary_metric_value is
      the mean of the field over the 400 grid points.

Tolerances: 1e-6 (relative to max|u|) for analytical/scipy; O(h^2) or O(h)
bounds for FDM depending on whether first-order Neumann edges are involved;
loose, measured bounds for the PINNs (fixed seed, a few hundred epochs).

Tests decorated with ``@unittest.expectedFailure`` assert the *correct*
behaviour for a solver bug found while writing this module; each one carries a
BUG comment with the suspected cause. When a bug is fixed, the corresponding
test starts to "unexpectedly succeed" and the decorator should be removed.

Run alone with ``python -m unittest tests.test_solver_accuracy``; set
``VECTORNAUT_ACCURACY_TABLE=1`` to print the measured error table.
"""
import math
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from tests.solver_accuracy_helpers import Case, maybe_print_table, run_case

PINN_EPOCHS_1D = 300
PINN_EPOCHS_2D = 300
EXACT_RTOL = 1e-6


# Solver results are memoised per (case, method, epochs) across all test classes so each
# configuration is solved once per module run.
_RUNS = {}


_PREVIOUS_TORCH_THREADS = None


def setUpModule():
    # The PINNs are tiny; one intra-op thread gives identical results and avoids large
    # slowdowns when the machine is busy.
    global _PREVIOUS_TORCH_THREADS
    import torch
    _PREVIOUS_TORCH_THREADS = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    import torch
    if _PREVIOUS_TORCH_THREADS:
        torch.set_num_threads(_PREVIOUS_TORCH_THREADS)
    maybe_print_table()


class _TempDataDirTestCase(unittest.TestCase):
    """Points VECTORNAUT_DATA_DIR at a fresh temp dir so PINN model caches never
    land in the repo and are never reused from other tests or runs."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="vectornaut_accuracy_")
        cls._previous_data_dir = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = cls._tmp

    @classmethod
    def tearDownClass(cls):
        if cls._previous_data_dir is None:
            os.environ.pop("VECTORNAUT_DATA_DIR", None)
        else:
            os.environ["VECTORNAUT_DATA_DIR"] = cls._previous_data_dir
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def solve(self, case, method, epochs=200, **kwargs):
        key = (case.name, method, epochs)
        if key not in _RUNS:
            _RUNS[key] = run_case(case, method, epochs=epochs, **kwargs)
        return _RUNS[key]

    def validate(self, run, sim=None):
        from vectornaut.validator import validate_run_output
        return validate_run_output(run.miner, run.auditor, sim if sim is not None else run.sim)

    def assertValidatorPasses(self, run):
        result = self.validate(run)
        failed = [(c.name, c.detail) for c in result.checks if not c.passed]
        self.assertEqual(result.status, "pass", f"validator flagged a correct solution: {failed}")


# ---------------------------------------------------------------------------
# 1D cases
# ---------------------------------------------------------------------------

SIN2 = math.sin(2.0)
ADV = 1.0 - math.exp(-2.0)

CASES_1D = {
    "linear_dirichlet": Case(
        "linear_dirichlet", "d2u_dy2 = 0", ["u(0) = 1", "u(1) = 3"], ["y"], "u",
        exact=lambda y: 1.0 + 2.0 * y, exact_metric=2.0),
    "poiseuille_params": Case(
        "poiseuille_params", "d2u_dy2 = -G / mu", ["u(0) = 0", "u(1) = 0"], ["y"], "u",
        exact=lambda y: 2.0 * y * (1.0 - y), params={"G": 2.0, "mu": 0.5}, exact_metric=2.0),
    "neumann_outlet": Case(
        "neumann_outlet", "d2u_dy2 = 2", ["u(0) = 0", "du_dy(1) = 0"], ["y"], "u",
        exact=lambda y: y ** 2 - 2.0 * y, exact_metric=-2.0),
    "prime_notation": Case(
        "prime_notation", "u'' = 2", ["u(0) = 0", "u'(1) = 0"], ["y"], "u",
        exact=lambda y: y ** 2 - 2.0 * y, exact_metric=-2.0),
    "lhs_coefficient_domain_param": Case(
        "lhs_coefficient_domain_param", "mu * d2u_dy2 + G = 0", ["u(0) = 0", "u(H) = 0"], ["y"], "u",
        exact=lambda y: 2.0 * y * (2.0 - y), params={"G": 2.0, "mu": 0.5, "H": 2.0},
        exact_metric=4.0, exact_domain=(0.0, 2.0)),
    # Navier-slip Couette flow: u = U (y + b) / (H + b), drag reduction b / (H + b) = 20 %.
    "couette_navier_slip": Case(
        "couette_navier_slip", "d2u_dy2 = 0", ["u(0) = slip_length * du_dy(0)", "u(1) = U"], ["y"], "u",
        exact=lambda y: 2.0 * (y + 0.25) / 1.25, params={"U": 2.0}, coefficient=0.25, exact_metric=1.6),
    "robin_convective": Case(
        "robin_convective", "d2T_dx2 = 0", ["T(0) = 1", "dT_dx(1) = -Bi * T(1)"], ["x"], "T",
        exact=lambda x: 1.0 - 0.5 * x, params={"Bi": 1.0}, exact_metric=-0.5),
    "sine_source": Case(
        "sine_source", "d2u_dy2 = -pi**2 * sin(pi * y)", ["u(0) = 0", "u(1) = 0"], ["y"], "u",
        exact=lambda y: np.sin(np.pi * y), exact_metric=math.pi),
    "helmholtz": Case(
        "helmholtz", "d2u_dy2 = -k**2 * u(y)", ["u(0) = 0", "u(1) = 1"], ["y"], "u",
        exact=lambda y: np.sin(2.0 * y) / SIN2, params={"k": 2.0}, exact_metric=2.0 / SIN2),
    "advection_diffusion": Case(
        "advection_diffusion", "d2u_dy2 + Pe * du_dy = 0", ["u(0) = 0", "u(1) = 1"], ["y"], "u",
        exact=lambda y: (1.0 - np.exp(-2.0 * y)) / ADV, params={"Pe": 2.0}, exact_metric=2.0 / ADV),
    "shifted_domain": Case(
        "shifted_domain", "d2u_dy2 = 2", ["u(1) = 1", "u(2) = 4"], ["y"], "u",
        exact=lambda y: y ** 2, exact_metric=2.0, exact_domain=(1.0, 2.0)),
    # Plane Poiseuille flow in a 10 um film (the app's typical film_thickness scale):
    # u = G/(2 mu) y (h - y), wall shear rate G h / (2 mu) = 50 1/s.
    "thin_film_poiseuille": Case(
        "thin_film_poiseuille", "d2u_dy2 = -G / mu", ["u(0) = 0", "u(h) = 0"], ["y"], "u",
        exact=lambda y: 1.0e4 / (2.0e-3) * y * (1.0e-5 - y), params={"G": 1.0e4, "mu": 1.0e-3, "h": 1.0e-5},
        exact_metric=50.0, exact_domain=(0.0, 1.0e-5)),
    "offset_temperature": Case(
        "offset_temperature", "d2T_dx2 = 0", ["T(0) = T_hot", "T(1) = T_cold"], ["x"], "T",
        exact=lambda x: 310.0 - 50.0 * x, params={"T_hot": 310.0, "T_cold": 260.0}, exact_metric=-50.0),
}

EXACT_GAIN_1D = {"couette_navier_slip": 20.0}

# (case, method, max-norm relative tolerance, metric relative tolerance)
PASSING_1D = [(name, m, EXACT_RTOL, 1e-4) for name in CASES_1D for m in ("analytical", "scipy")
              if not (name == "thin_film_poiseuille" and m == "scipy")]
PASSING_1D.append(("thin_film_poiseuille", "scipy_profile_only", EXACT_RTOL, None))
# PINN: measured max relative errors are 1e-4..7e-4 (3.7e-3 for helmholtz) at 300 epochs.
PASSING_1D += [(name, "pinn", 5e-3, 2e-2) for name in (
    "linear_dirichlet", "poiseuille_params", "neumann_outlet", "prime_notation",
    "lhs_coefficient_domain_param", "couette_navier_slip", "robin_convective",
    "advection_diffusion", "shifted_domain")]
PASSING_1D.append(("helmholtz", "pinn", 3e-2, 2e-2))


def _make_1d_test(case_name, method, rtol, metric_rtol):
    def test(self):
        case = CASES_1D[case_name]
        solver = "scipy" if method == "scipy_profile_only" else method
        epochs = PINN_EPOCHS_1D if solver == "pinn" else 200
        run = self.solve(case, solver, epochs=epochs)
        sim = run.sim
        self.assertEqual(sim.solver_method, solver, run.log)
        self.assertEqual(len(sim.sample_points), 20)
        lo, hi = case.exact_domain
        self.assertAlmostEqual(sim.sample_points[0], lo, delta=1e-12 + 1e-9 * abs(hi - lo))
        self.assertAlmostEqual(sim.sample_points[-1], hi, delta=1e-12 + 1e-9 * abs(hi - lo))

        errs = run.errors()
        self.assertLess(errs["rel_max"], rtol, f"{case_name}/{method}: {errs}")

        # relative_error is computed against the app's reference (analytical here), which is
        # exact for these cases, so it must agree with the error against the true solution.
        self.assertAlmostEqual(sim.relative_error, errs["rel_l1_like_app"],
                               delta=1e-9 + 1e-3 * errs["rel_l1_like_app"])

        if metric_rtol is not None:
            scale = abs(case.exact_metric)
            self.assertAlmostEqual(sim.primary_metric_value, case.exact_metric, delta=metric_rtol * scale)
            self.assertAlmostEqual(sim.reference_metric_value, case.exact_metric, delta=1e-6 * scale)
            gain_tol = 1e-2 if solver != "pinn" else 0.5
            self.assertAlmostEqual(sim.performance_gain_pct, EXACT_GAIN_1D.get(case_name, 0.0), delta=gain_tol)

        self.assertValidatorPasses(run)
    test.__name__ = f"test_{case_name}_{method}"
    return test


class OneDimensionalAccuracyTest(_TempDataDirTestCase):
    """Correct 1D behaviour: profiles, wall metric, relative_error and gain vs closed form."""

    def test_pinn_relative_error_matches_true_error(self):
        # For the PINN the reported relative_error is the only accuracy signal the UI shows.
        run = self.solve(CASES_1D["poiseuille_params"], "pinn", epochs=PINN_EPOCHS_1D)
        errs = run.errors()
        self.assertGreater(run.sim.relative_error, 0.0)
        self.assertAlmostEqual(run.sim.relative_error, errs["rel_l1_like_app"], delta=1e-3 * errs["rel_l1_like_app"])

    def test_validator_flags_boundary_violation(self):
        run = self.solve(CASES_1D["poiseuille_params"], "analytical")
        broken = run.sim.model_copy(update={"solution_primary": [v + 1.0 for v in run.sim.solution_primary]})
        result = self.validate(run, broken)
        self.assertNotEqual(result.status, "pass")
        self.assertIn("physics_boundary_conditions", [c.name for c in result.checks if not c.passed])

    def test_validator_flags_thin_film_pinn_garbage(self):
        run = self.solve(CASES_1D["thin_film_poiseuille"], "pinn", epochs=PINN_EPOCHS_1D)
        self.assertNotEqual(self.validate(run).status, "pass")


for _args in PASSING_1D:
    _t = _make_1d_test(*_args)
    setattr(OneDimensionalAccuracyTest, _t.__name__, _t)


class OneDimensionalKnownBugsTest(_TempDataDirTestCase):
    """Each test asserts the correct behaviour; the BUG comment explains why it currently fails."""

    # FIXED (kept as regression test). Was: a bare dependent variable in the equation ("... * u" instead of "u(y)") crashes
    # every 1D solver with TypeError("unsupported operand type(s) for *: 'Mul' and
    # 'UndefinedFunction'"). parsing.py:43 binds the name `u` to the *class*
    # sp.Function('u') and clean_expr_str (parsing.py:26-33) only rewrites derivatives,
    # so "k**2 * u" multiplies by an undefined function class. Natural LLM output for
    # reaction-diffusion / fin equations ("d2T_dx2 = m**2 * (T - T_inf)") hits this.
    def test_bare_dependent_variable_in_equation(self):
        case = Case("helmholtz_bare_u", "d2u_dy2 = -k**2 * u", ["u(0) = 0", "u(1) = 1"], ["y"], "u",
                    exact=lambda y: np.sin(2.0 * y) / SIN2, params={"k": 2.0}, exact_metric=2.0 / SIN2)
        run = self.solve(case, "analytical")
        self.assertLess(run.errors()["rel_max"], EXACT_RTOL)

    # FIXED (kept as regression test). Was: the slip BC used as the example in config.py:62/73 ("u(0) = lambda * du_dy(0)")
    # cannot be parsed: `lambda` is a Python keyword, so sp.parse_expr in
    # solvers_1d.py:191/279/378 raises "Starred arguments in lambda not supported" and
    # dispatch_and_solve raises RuntimeError("All primary analytical solvers failed.") --
    # even though _merged_params (solver_dispatcher.py:55) injects a 'lambda' parameter.
    def test_lambda_slip_bc_from_schema_example(self):
        case = Case("couette_lambda_slip", "d2u_dy2 = 0", ["u(0) = lambda * du_dy(0)", "u(1) = u_free"], ["y"], "u",
                    exact=lambda y: 2.0 * (y + 0.25) / 1.25, params={"u_free": 2.0}, coefficient=0.25)
        run = self.solve(case, "analytical")
        self.assertLess(run.errors()["rel_max"], EXACT_RTOL)

    # FIXED (kept as regression test). Was: get_domain_bounds (parsing.py:132) matches *any* "name(number)" in the BC text,
    # so the value "exp(-2)" in "c(1) = exp(-2)" is taken as a BC location and the domain
    # becomes [-2, 1]. The analytical profile is then sampled on the wrong interval, the
    # wall metric is evaluated at y=-2 (-109.2 instead of -2), and scipy/PINN fail
    # (both BCs map to the same end -> singular Jacobian).
    def test_domain_not_polluted_by_function_calls_in_bc_values(self):
        case = Case("decay_exp_bc", "d2c_dx2 = c(x) / Lc**2", ["c(0) = 1", "c(1) = exp(-2)"], ["x"], "c",
                    exact=lambda x: np.exp(-2.0 * x), params={"Lc": 0.5}, exact_metric=-2.0)
        run = self.solve(case, "analytical")
        self.assertEqual((run.sim.sample_points[0], run.sim.sample_points[-1]), (0.0, 1.0))
        self.assertAlmostEqual(run.sim.primary_metric_value, -2.0, delta=1e-6)

    # FIXED (kept as regression test). Was: get_domain_bounds (parsing.py:137) silently falls back to [0, 1] when the
    # domain length is <= 1e-6, i.e. for sub-micron films (here 500 nm). The profile is
    # sampled on [0, 1] (values down to -5e6 m/s), and scipy fails because both BCs map
    # to the lower end (solvers_1d.py:260) -- the analytical fallback is then used but
    # still reported as 'scipy' (solver_dispatcher.py:447-451).
    def test_submicron_domain_is_respected(self):
        case = Case("nano_film_poiseuille", "d2u_dy2 = -G / mu", ["u(0) = 0", "u(h) = 0"], ["y"], "u",
                    exact=lambda y: 1.0e4 / 2.0e-3 * y * (5.0e-7 - y), params={"G": 1.0e4, "mu": 1.0e-3, "h": 5.0e-7},
                    exact_metric=2.5)
        run = self.solve(case, "scipy")
        self.assertAlmostEqual(run.sim.sample_points[-1], 5.0e-7, delta=1e-15)
        self.assertLess(run.errors()["rel_max"], EXACT_RTOL)

    # FIXED (kept as regression test). Was: solve_scipy_bvp estimates the wall derivative with a forward difference of fixed
    # absolute step dx = 1e-5 (solvers_1d.py:311-312). On a 10 um film this step spans the
    # whole domain, so the "wall shear rate" is the secant slope across the channel:
    # ~-2e-11 instead of 50, and performance_gain_pct reports 100 % drag reduction for a
    # no-slip channel. The profile itself is correct (see test_thin_film_poiseuille_scipy_profile_only).
    def test_scipy_wall_derivative_on_thin_film(self):
        run = self.solve(CASES_1D["thin_film_poiseuille"], "scipy")
        self.assertAlmostEqual(run.sim.primary_metric_value, 50.0, delta=0.05)
        self.assertAlmostEqual(run.sim.performance_gain_pct, 0.0, delta=0.1)

    # FIXED (kept as regression test). Was: solve_analytical treats every free symbol whose name starts with 'C' as an
    # integration constant (solvers_1d.py:171), so a parameter such as 'Cf' (or C_p, Cd,
    # C0) is solved for as an unknown and float() fails ("Cannot convert expression to
    # float"). The dispatcher then silently uses scipy while still reporting
    # solver_method='analytical' (see test_solver_method_reports_actual_solver).
    # The failure is NON-DETERMINISTIC across processes: `constants` comes from a set of
    # Symbols, whose iteration order depends on PYTHONHASHSEED, and
    # sp.solve(2 equations, [C1, C2, Cf]) only works when C1 and C2 come first ('Cf' fails
    # for PYTHONHASHSEED=1,3,5,9 and works for 0,2,4,6,7,8 on CPython 3.11). The order is
    # fixed within a process, so the check runs in subprocesses with pinned hash seeds.
    def test_parameter_named_like_integration_constant(self):
        import subprocess
        import sys
        code = (
            "from vectornaut.solvers.parsing import parse_equation_and_bcs\n"
            "from vectornaut.solvers.solvers_1d import solve_analytical\n"
            "p = {'Cf': 2.0}; b = ['u(0) = 0', 'u(1) = 0']\n"
            "rhs, _, y, u, syms = parse_equation_and_bcs('d2u_dy2 = -Cf', b, 'y', 'u', p)\n"
            "try:\n"
            "    expr, d = solve_analytical(rhs, b, y, u, syms, p, 0.0, 1.0)\n"
            "    ok = abs(float(expr.subs(y, 0.5)) - 0.25) < 1e-12 and abs(d - 1.0) < 1e-12\n"
            "    print('OK' if ok else 'WRONG ' + str(expr))\n"
            "except Exception as exc:\n"
            "    print('ERROR ' + repr(exc))\n"
        )
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        procs = {}
        for seed in ("0", "1", "2", "3"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=repo_root)
            procs[seed] = subprocess.Popen([sys.executable, "-c", code], cwd=repo_root, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        results = {seed: proc.communicate(timeout=120)[0].strip().splitlines()[-1:] for seed, proc in procs.items()}
        self.assertEqual({seed: out for seed, out in results.items() if out != ["OK"]}, {})

    # FIXED (kept as regression test). Was: when SymPy fails and solution_primary comes from the SciPy fallback
    # (solver_dispatcher.py:461-464), method_requested is left as 'analytical', so the
    # output (and the validator's "deterministic validation coverage" info) claims an
    # analytical solution. relative_error is then scipy-vs-scipy = 0. The mirror case
    # (scipy requested, analytical used, reported 'scipy') is at solver_dispatcher.py:447-451.
    # SymPy failure is simulated with a mock: real triggers are the 'C*' parameter bug above
    # or nonlinear ODEs, where dsolve takes ~45 s before failing.
    def test_solver_method_reports_actual_solver(self):
        with mock.patch("vectornaut.solver_dispatcher.solve_analytical", side_effect=NotImplementedError("dsolve")):
            run = run_case(CASES_1D["poiseuille_params"], "analytical", record=False)
        self.assertLess(run.errors()["rel_max"], EXACT_RTOL)  # scipy fallback is accurate ...
        self.assertEqual(run.sim.solver_method, "scipy")      # ... but is reported as 'analytical'

    # FIXED (kept as regression test). Was: _merged_params (solver_dispatcher.py:53-56) unconditionally overwrites
    # 'slip_length' (and 'lambda', 'slippage_coefficient') with simulation_coefficient,
    # silently discarding an explicitly audited slip_length. Here the audited
    # slip_length=0.1 is replaced by 0.5, giving 33 % instead of 9.1 % drag reduction.
    def test_audited_slip_length_not_overridden(self):
        case = Case("couette_audited_slip", "d2u_dy2 = 0", ["u(0) = slip_length * du_dy(0)", "u(1) = U"], ["y"], "u",
                    exact=lambda y: 2.0 * (y + 0.1) / 1.1, params={"U": 2.0, "slip_length": 0.1}, coefficient=0.5,
                    exact_metric=2.0 / 1.1)
        run = self.solve(case, "analytical")
        self.assertLess(run.errors()["rel_max"], EXACT_RTOL)
        self.assertAlmostEqual(run.sim.performance_gain_pct, 100.0 * 0.1 / 1.1, delta=1e-6)

    # Regression: auto-injected parameters used to be lost before reaching the 1D solvers.
    def test_auto_injected_parameters_reach_1d_solver(self):
        case = Case("poiseuille_no_params", "d2u_dy2 = -G / mu", ["u(0) = 0", "u(1) = 0"], ["y"], "u",
                    exact=lambda y: 0.5 * y * (1.0 - y))  # defaults G = mu = 1.0
        run = self.solve(case, "analytical")
        self.assertLess(run.errors()["rel_max"], EXACT_RTOL)

    # FIXED (kept as regression test). Was: the performance-gain baseline zeroes every parameter whose name contains
    # "thick" unless it is one of five whitelisted names (solver_dispatcher.py:346-354).
    # For a plain no-slip Couette flow over a "coating_thickness" gap both BCs collapse to
    # y=0, the baseline solve fails, and the fallback baseline_deriv = 1.5
    # (solver_dispatcher.py:368) yields a -33 % "gain" for a design with no bionic feature.
    def test_baseline_gain_zero_without_bionic_feature(self):
        case = Case("couette_coating_gap", "d2u_dy2 = 0", ["u(0) = 0", "u(coating_thickness) = U"], ["y"], "u",
                    exact=lambda y: 2.0 * y, params={"U": 1.0, "coating_thickness": 0.5},
                    exact_metric=2.0, exact_domain=(0.0, 0.5))
        run = self.solve(case, "analytical")
        self.assertLess(run.errors()["rel_max"], EXACT_RTOL)  # the solution itself is right
        self.assertAlmostEqual(run.sim.performance_gain_pct, 0.0, delta=1e-6)

    # FIXED (kept as regression test). Was: SymPy's torch printer emits the builtin names pow/exp for purely numeric
    # sub-expressions (pi**2 -> pow(3.14159..., 2), exp(-2)), which the torch namespace
    # maps to torch.pow/torch.exp; these reject Python floats. solve_pytorch_pinn
    # (solvers_1d.py:342/385) therefore raises for any equation or BC with such a constant
    # and the dispatcher silently falls back to scipy (reported as 'scipy').
    def test_pinn_handles_numeric_constants(self):
        run = self.solve(CASES_1D["sine_source"], "pinn", epochs=PINN_EPOCHS_1D)
        self.assertEqual(run.sim.solver_method, "pinn", run.log)

    # FIXED (kept as regression test). Was: the 1D PINN has no input/output scaling. For a temperature field of
    # 260-310 K it starts at ~0 and after 300 epochs is off by ~175 K (1000 epochs: ~39 K),
    # reporting a 100 % "insulation gain" for plain conduction. The 2D PINN initialises the
    # output bias to the mean Dirichlet value (solvers_2d.py:157-161); the 1D one does not
    # (solvers_1d.py:331-333). Tolerance here: 5 K.
    def test_pinn_offset_temperature(self):
        run = self.solve(CASES_1D["offset_temperature"], "pinn", epochs=PINN_EPOCHS_1D)
        self.assertLess(run.errors()["max_abs"], 5.0)

    # BUG (accuracy): on the 10 um film the PINN trains on raw coordinates ~1e-5 and a
    # second derivative of -1e7, so the loss is ~1e14 and the output is garbage (max error
    # ~67 m/s for a 1.25e-4 m/s profile, wall derivative 0). No nondimensionalisation in
    # solve_pytorch_pinn (solvers_1d.py:388-420). Validator does flag this run.
    @unittest.expectedFailure
    def test_pinn_thin_film(self):
        run = self.solve(CASES_1D["thin_film_poiseuille"], "pinn", epochs=PINN_EPOCHS_1D)
        self.assertLess(run.errors()["rel_max"], 0.5)

    # FIXED (kept as regression test). Was: validator false positive on a correct solution. _estimate_derivative
    # (validator.py:97-131) uses a first-order one-sided difference between the 20 sample
    # points, with an absolute tolerance of 0.1, so the exact solution of u''=20,
    # du_dy(1)=0 (discrete slope 10*h = 0.53 at y=1) is flagged 'warn'/'rerun_solver'.
    def test_validator_accepts_correct_steep_neumann_solution(self):
        case = Case("steep_neumann", "d2u_dy2 = 20", ["u(0) = 0", "du_dy(1) = 0"], ["y"], "u",
                    exact=lambda y: 10.0 * y ** 2 - 20.0 * y)
        run = self.solve(case, "analytical")
        self.assertLess(run.errors()["rel_max"], EXACT_RTOL)
        self.assertValidatorPasses(run)

    # FIXED (kept as regression test). Was: validate_run_output only checks simple Dirichlet/Neumann values at
    # the boundary and trusts the reported relative_error; it never compares
    # solution_primary with solution_reference (or checks the PDE residual). An all-zero
    # "Poiseuille" profile satisfies u(0)=u(1)=0 and is accepted with status 'pass'.
    def test_validator_flags_primary_reference_mismatch(self):
        run = self.solve(CASES_1D["poiseuille_params"], "analytical")
        broken = run.sim.model_copy(update={"solution_primary": [0.0] * len(run.sim.solution_primary)})
        self.assertNotEqual(self.validate(run, broken).status, "pass")


class PinnCacheTest(unittest.TestCase):
    """Uses its own data dir: the PINN model cache must start empty."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = self.tmp.name

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("VECTORNAUT_DATA_DIR", None)
        else:
            os.environ["VECTORNAUT_DATA_DIR"] = self.previous
        self.tmp.cleanup()

    # FIXED (kept as regression test). Was: the PINN cache key (model_cache.py:299-313) is equation + BCs + params (+ domain)
    # but not the requested epochs, so after a quick 10-epoch run, asking for 300 epochs
    # silently returns the stale 10-epoch model (epochs_trained=10, max error ~1.1).
    def test_cache_respects_requested_epochs(self):
        case = CASES_1D["poiseuille_params"]
        run_case(case, "pinn", epochs=10, record=False)
        run = run_case(case, "pinn", epochs=PINN_EPOCHS_1D, record=False)
        self.assertEqual(run.sim.epochs_trained, PINN_EPOCHS_1D)
        self.assertLess(run.errors()["rel_max"], 5e-3)


# ---------------------------------------------------------------------------
# 2D cases (unit square, 20x20 grid, h = 1/19)
# ---------------------------------------------------------------------------

def _lid_driven_laplace(x, y, n_terms=399):
    """u = 1 on y=1, 0 on the other edges: sum over odd n of 4/(n pi) sin(n pi x) sinh(n pi y)/sinh(n pi)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    total = np.zeros(np.broadcast(x, y).shape)
    for n in range(1, n_terms + 1, 2):
        k = n * np.pi
        ratio = np.exp(k * (y - 1.0)) * (1.0 - np.exp(-2.0 * k * y)) / (1.0 - np.exp(-2.0 * k))
        total += 4.0 / k * np.sin(k * x) * ratio
    return total


TWO_PI_SQ = 2.0 * math.pi ** 2

CASES_2D = {
    "laplace_linear_x": Case(
        "laplace_linear_x", "d2T_dx2 + d2T_dy2 = 0",
        ["T(0, y) = 0", "T(1, y) = 1", "dT_dy(x, 0) = 0", "dT_dy(x, 1) = 0"], ["x", "y"], "T",
        exact=lambda x, y: x),
    "laplace_linear_y": Case(
        "laplace_linear_y", "d2T_dx2 + d2T_dy2 = 0",
        ["T(x, 0) = 0", "T(x, 1) = 1", "dT_dx(0, y) = 0", "dT_dx(1, y) = 0"], ["x", "y"], "T",
        exact=lambda x, y: y),
    "laplace_hot_cold_params": Case(
        "laplace_hot_cold_params", "d2T_dx2 + d2T_dy2 = 0",
        ["T(0, y) = T_hot", "T(1, y) = T_cold", "dT_dy(x, 0) = 0", "dT_dy(x, 1) = 0"], ["x", "y"], "T",
        exact=lambda x, y: 300.0 - 40.0 * x, params={"T_hot": 300.0, "T_cold": 260.0}),
    "laplace_lid_series": Case(
        "laplace_lid_series", "d2u_dx2 + d2u_dy2 = 0",
        ["u(0, y) = 0", "u(1, y) = 0", "u(x, 0) = 0", "u(x, 1) = 1"], ["x", "y"], "u",
        exact=_lid_driven_laplace),
    "poisson_quadratic": Case(
        "poisson_quadratic", "d2u_dx2 + d2u_dy2 = 2",
        ["u(0, y) = 0", "u(1, y) = 1", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"], ["x", "y"], "u",
        exact=lambda x, y: x ** 2),
    "poisson_neumann_flux": Case(
        "poisson_neumann_flux", "d2u_dx2 + d2u_dy2 = 1",
        ["u(0, y) = 0", "du_dx(1, y) = 1", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"], ["x", "y"], "u",
        exact=lambda x, y: 0.5 * x ** 2),
    # Manufactured u = sin(pi x) sin(pi y); coefficient -2 pi^2 written as a number so the
    # FDM result is not affected by the pi**2 PINN bug.
    "poisson_sin_sin": Case(
        "poisson_sin_sin", f"d2u_dx2 + d2u_dy2 = -{TWO_PI_SQ!r} * sin(pi * x) * sin(pi * y)",
        ["u(0, y) = 0", "u(1, y) = 0", "u(x, 0) = 0", "u(x, 1) = 0"], ["x", "y"], "u",
        exact=lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)),
}


def _interior(points, margin=0.2):
    return ((points[:, 0] > margin) & (points[:, 0] < 1 - margin)
            & (points[:, 1] > margin) & (points[:, 1] < 1 - margin))


# (case, method, tolerance on max|err|/max|u|, interior-only?)
# FDM: Jacobi stops at max update < 1e-6 (measured error ~1.5e-4 for linear fields);
# O(h^2) = 2.2e-3 for sin*sin; first-order Neumann edge in poisson_neumann_flux gives O(h)
# (measured 3.6e-2 relative, h = 0.053). The lid problem is checked away from the
# discontinuous corners.
PASSING_2D = [
    ("laplace_linear_x", "fdm", 1e-3, False),
    ("laplace_linear_y", "fdm", 1e-3, False),
    ("laplace_hot_cold_params", "fdm", 1e-5, False),
    ("laplace_lid_series", "fdm", 5e-3, True),
    ("poisson_quadratic", "fdm", 1e-3, False),
    ("poisson_neumann_flux", "fdm", 0.06, False),
    ("poisson_sin_sin", "fdm", 5e-3, False),
    # PINN, 300 epochs; measured 7.6e-4 .. 5.4e-3 relative (lid interior: 4.8e-2 absolute).
    ("laplace_linear_x", "pinn", 2e-2, False),
    ("laplace_linear_y", "pinn", 2e-2, False),
    ("laplace_hot_cold_params", "pinn", 1e-3, False),
    ("laplace_lid_series", "pinn", 0.1, True),
    ("poisson_quadratic", "pinn", 2e-2, False),
    ("poisson_neumann_flux", "pinn", 2e-2, False),
]


def _make_2d_test(case_name, method, rtol, interior_only):
    def test(self):
        case = CASES_2D[case_name]
        run = self.solve(case, method, epochs=PINN_EPOCHS_2D if method == "pinn" else 200)
        sim = run.sim
        self.assertEqual(sim.solver_method, method, run.log)
        points = run.points
        self.assertEqual(points.shape, (400, 2))
        self.assertAlmostEqual(points.min(), 0.0)
        self.assertAlmostEqual(points.max(), 1.0)
        mask = _interior(points) if interior_only else None
        errs = run.errors(mask)
        self.assertLess(errs["rel_max"], rtol, f"{case_name}/{method}: {errs}")
        # primary_metric_value is the mean field value over the sample points.
        exact_mean = float(np.mean(run.exact_values()))
        scale = float(np.max(np.abs(run.exact_values())))
        mean_tol = (0.05 if interior_only else rtol) * scale
        self.assertAlmostEqual(sim.primary_metric_value, exact_mean, delta=mean_tol)
        self.assertAlmostEqual(sim.primary_metric_value, float(np.mean(sim.solution_primary)), delta=1e-9 * scale)
        # relative_error is primary vs FDM reference; recompute it from the arrays.
        primary = np.asarray(sim.solution_primary)
        reference = np.asarray(sim.solution_reference)
        recomputed = np.sum(np.abs(primary - reference)) / np.sum(np.abs(reference) + 1e-8)
        self.assertAlmostEqual(sim.relative_error, recomputed, delta=1e-9 + 1e-6 * recomputed)
        self.assertNotEqual(self.validate(run).status, "fail")
    test.__name__ = f"test_{case_name}_{method}"
    return test


class TwoDimensionalAccuracyTest(_TempDataDirTestCase):
    """Correct 2D behaviour vs closed-form harmonic / manufactured Poisson solutions."""


for _args in PASSING_2D:
    _t = _make_2d_test(*_args)
    setattr(TwoDimensionalAccuracyTest, _t.__name__, _t)


class TwoDimensionalKnownBugsTest(_TempDataDirTestCase):

    # BUG: solve_pytorch_pinn_2d evaluates rhs_val = f(xy_pde) once, before the training
    # loop, on a tensor with requires_grad=True (solvers_2d.py:170-175). For any source term
    # that depends on x or y, rhs_val carries a graph that is freed by the first
    # backward(), so epoch 2 raises "Trying to backward through the graph a second time"
    # and the dispatcher silently falls back to FDM. Only constant sources work.
    @unittest.expectedFailure
    def test_pinn_spatially_varying_source(self):
        run = self.solve(CASES_2D["poisson_sin_sin"], "pinn", epochs=PINN_EPOCHS_2D)
        self.assertEqual(run.sim.solver_method, "pinn", run.log)
        self.assertLess(run.errors()["rel_max"], 5e-2)

    # FIXED (kept as regression test). Was: same torch-printer issue as the 1D PINN (solvers_2d.py:172): a constant source
    # "pi**2 / 4" is printed as pow(3.14159, 2) -> torch.pow(float, int) -> TypeError ->
    # silent FDM fallback. Exact solution u = pi^2 x^2 / 8.
    def test_pinn_numeric_constant_source(self):
        case = Case("poisson_pi_constant", "d2u_dx2 + d2u_dy2 = pi**2 / 4",
                    ["u(0, y) = 0", "u(1, y) = pi**2 / 8", "du_dy(x, 0) = 0", "du_dy(x, 1) = 0"], ["x", "y"], "u",
                    exact=lambda x, y: math.pi ** 2 / 8.0 * x ** 2)
        run = self.solve(case, "pinn", epochs=PINN_EPOCHS_2D)
        self.assertEqual(run.sim.solver_method, "pinn", run.log)

    # FIXED (kept as regression test). Was: parse_bc_2d_string only supports constant edge values (float(...) at
    # solvers_2d.py:38-40). "u(x, 1) = sin(pi * x)" raises, the exception is swallowed at
    # solver_dispatcher.py:191-192 and the edge is replaced by the default homogeneous
    # Neumann BC (solver_dispatcher.py:194-201). Result: u == 0 everywhere instead of
    # sin(pi x) sinh(pi y)/sinh(pi), with relative_error 0 and no error surfaced.
    def test_non_constant_dirichlet_bc(self):
        case = Case("laplace_sin_top", "d2u_dx2 + d2u_dy2 = 0",
                    ["u(0, y) = 0", "u(1, y) = 0", "u(x, 0) = 0", "u(x, 1) = sin(pi * x)"], ["x", "y"], "u",
                    exact=lambda x, y: np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi))
        run = self.solve(case, "fdm")
        self.assertLess(run.errors()["rel_max"], 2e-2)

    # FIXED (kept as regression test). Was: the 2D solvers hard-code the unit square (solvers_2d.py:91-93, 166-189) and edge
    # detection only recognises coordinates 0 and 1 (solvers_2d.py:47-68). "T(2, y) = 2"
    # falls through to edge='left' (overwriting T(0, y) = 0), and the now-missing right
    # edge is filled with the default T = 273 K (solver_dispatcher.py:199). Exact: T = x on
    # [0, 2] x [0, 1]; the solver returns ~137 K on [0, 1]^2.
    def test_non_unit_domain(self):
        case = Case("laplace_wide_slab", "d2T_dx2 + d2T_dy2 = 0",
                    ["T(0, y) = 0", "T(2, y) = 2", "dT_dy(x, 0) = 0", "dT_dy(x, 1) = 0"], ["x", "y"], "T",
                    exact=lambda x, y: x)
        run = self.solve(case, "fdm")
        self.assertAlmostEqual(float(run.points[:, 0].max()), 2.0)
        self.assertLess(run.errors()["rel_max"], 1e-2)

    # FIXED (kept as regression test). Was: for solver_method='fdm' the primary solution *is* the
    # reference (solver_dispatcher.py:273-275), so relative_error is identically 0 however
    # wrong the field is -- here the O(h) Neumann edge leaves a 4 % L2 error vs the exact
    # solution, and in test_non_constant_dirichlet_bc a 100 % error, both reported as 0.
    def test_fdm_relative_error_reflects_discretisation_error(self):
        run = self.solve(CASES_2D["poisson_neumann_flux"], "fdm")
        true_err = run.errors()["rel_l1_like_app"]
        self.assertGreater(true_err, 0.01)
        self.assertGreaterEqual(run.sim.relative_error, 0.1 * true_err)

    # FIXED (kept as regression test). Was: in 2D, performance_gain_pct = clamp(simulation_coefficient *
    # 100, 0, 99) (solver_dispatcher.py:279) regardless of the solution. The coefficient
    # does not appear in this problem, yet the "gain" goes 0 % -> 90 %.
    def test_performance_gain_independent_of_unused_coefficient(self):
        base = CASES_2D["laplace_linear_x"]
        gains = []
        for coeff in (0.0, 0.9):
            case = Case(f"{base.name}_c{coeff}", base.equation, base.bcs, base.independent, base.dependent,
                        exact=base.exact, coefficient=coeff)
            gains.append(self.solve(case, "fdm").sim.performance_gain_pct)
        self.assertAlmostEqual(gains[0], gains[1], delta=1e-9)


if __name__ == "__main__":
    unittest.main()
