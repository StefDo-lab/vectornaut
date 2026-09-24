# -*- coding: utf-8 -*-
"""Offline regression tests for the 1D dispatcher metrics, the PINN model cache and
the deterministic validator.

Covers: audited slip aliases are not overridden by simulation_coefficient, the
no-effect baseline only neutralises bionic-effect parameters (never geometric
lengths) and has no magic fallback value, solver_method reports the solver that
actually produced solution_primary, the PINN cache honours the requested epochs,
and the validator's derivative BC / primary-vs-reference / PINN divergence checks.
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from tests.solver_accuracy_helpers import UI_METADATA, Case, run_case
from vectornaut.config import AuditedParameter, AuditorOutput
from vectornaut.validator import validate_run_data


class _TempDataDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="vectornaut_metrics_")
        self._previous = os.environ.get("VECTORNAUT_DATA_DIR")
        os.environ["VECTORNAUT_DATA_DIR"] = self._tmp

    def tearDown(self):
        if self._previous is None:
            os.environ.pop("VECTORNAUT_DATA_DIR", None)
        else:
            os.environ["VECTORNAUT_DATA_DIR"] = self._previous
        shutil.rmtree(self._tmp, ignore_errors=True)


POISEUILLE = Case("poiseuille", "d2u_dy2 = -G / mu", ["u(0) = 0", "u(1) = 0"], ["y"], "u",
                  exact=lambda y: 2.0 * y * (1.0 - y), params={"G": 2.0, "mu": 0.5}, exact_metric=2.0)


class MergedParamsTest(unittest.TestCase):
    def _auditor(self, audited, coefficient):
        return AuditorOutput(
            audit_passed=True,
            audit_notes="ok",
            audited_parameters=[AuditedParameter(name=k, value=v) for k, v in audited.items()],
            dimensionless_numbers=[],
            simulation_coefficient=coefficient,
            solver_method="analytical",
            ui_metadata=UI_METADATA,
        )

    def test_audited_aliases_are_kept(self):
        from vectornaut.solver_dispatcher import _merged_params
        params = _merged_params(self._auditor({"slip_length": 0.1, "lambda": 0.2}, 0.5))
        self.assertEqual(params["slip_length"], 0.1)
        self.assertEqual(params["lambda"], 0.2)
        self.assertEqual(params["slippage_coefficient"], 0.5)
        self.assertEqual(params["simulation_coefficient"], 0.5)

    def test_missing_aliases_are_filled_from_coefficient(self):
        from vectornaut.solver_dispatcher import _merged_params
        params = _merged_params(self._auditor({"U": 2.0}, 0.25))
        for alias in ("slip_length", "lambda", "slippage_coefficient", "simulation_coefficient"):
            self.assertEqual(params[alias], 0.25)


class BaselineGainTest(_TempDataDir):
    def test_navier_slip_gain_matches_closed_form(self):
        case = Case("couette_slip", "d2u_dy2 = 0", ["u(0) = slip_length * du_dy(0)", "u(1) = U"], ["y"], "u",
                    exact=lambda y: 2.0 * (y + 0.25) / 1.25, params={"U": 2.0}, coefficient=0.25)
        run = run_case(case, "analytical", record=False)
        self.assertAlmostEqual(run.sim.performance_gain_pct, 20.0, delta=1e-6)

    def test_geometric_thickness_is_not_zeroed(self):
        case = Case("coating_gap", "d2u_dy2 = 0", ["u(0) = 0", "u(coating_thickness) = U"], ["y"], "u",
                    exact=lambda y: 2.0 * y, params={"U": 1.0, "coating_thickness": 0.5}, coefficient=0.3)
        run = run_case(case, "scipy", record=False)
        self.assertEqual(run.sim.solver_method, "scipy")
        self.assertAlmostEqual(run.sim.performance_gain_pct, 0.0, delta=1e-4)

    def test_failed_baseline_reports_zero_gain_instead_of_magic_value(self):
        from vectornaut import solver_dispatcher
        real_analytical = solver_dispatcher.solve_analytical
        real_scipy = solver_dispatcher.solve_scipy_bvp

        def _fail_for_baseline(real):
            def wrapper(*args):
                if args[5].get("slip_length") == 0.0:  # params of the no-effect baseline
                    raise ValueError("baseline not solvable")
                return real(*args)
            return wrapper

        # (The old fallback baseline_deriv = 1.5 gave -6.7 % here.)
        case = Case("couette_slip_nobase", "d2u_dy2 = 0", ["u(0) = slip_length * du_dy(0)", "u(1) = U"],
                    ["y"], "u", exact=lambda y: 2.0 * (y + 0.25) / 1.25, params={"U": 2.0}, coefficient=0.25)
        with mock.patch.object(solver_dispatcher, "solve_analytical", _fail_for_baseline(real_analytical)), \
                mock.patch.object(solver_dispatcher, "solve_scipy_bvp", _fail_for_baseline(real_scipy)):
            run = run_case(case, "analytical", record=False)
        self.assertLess(run.errors()["rel_max"], 1e-9)
        self.assertEqual(run.sim.performance_gain_pct, 0.0)


class SolverMethodReportingTest(_TempDataDir):
    def test_analytical_requested_scipy_used(self):
        with mock.patch("vectornaut.solver_dispatcher.solve_analytical", side_effect=NotImplementedError("dsolve")):
            run = run_case(POISEUILLE, "analytical", record=False)
        self.assertEqual(run.sim.solver_method, "scipy")
        self.assertLess(run.errors()["rel_max"], 1e-6)

    def test_scipy_requested_analytical_used(self):
        with mock.patch("vectornaut.solver_dispatcher.solve_scipy_bvp", side_effect=ValueError("no convergence")):
            run = run_case(POISEUILLE, "scipy", record=False)
        self.assertEqual(run.sim.solver_method, "analytical")
        self.assertLess(run.errors()["rel_max"], 1e-9)

    def test_pinn_failure_reports_scipy(self):
        with mock.patch("vectornaut.solver_dispatcher.solve_pytorch_pinn", side_effect=RuntimeError("pinn")):
            run = run_case(POISEUILLE, "pinn", epochs=5, record=False)
        self.assertEqual(run.sim.solver_method, "scipy")

    def test_unknown_1d_method_uses_analytical(self):
        run = run_case(POISEUILLE, "fdm", record=False)
        self.assertEqual(run.sim.solver_method, "analytical")
        self.assertEqual(len(run.sim.solution_primary), 20)
        self.assertLess(run.errors()["rel_max"], 1e-9)


class PinnCacheEpochsTest(_TempDataDir):
    def test_longer_trained_model_is_reused_shorter_is_not(self):
        from vectornaut.solvers.solvers_1d import solve_pytorch_pinn
        from vectornaut.storage import models_dir
        with mock.patch("vectornaut.solver_dispatcher.solve_pytorch_pinn", wraps=solve_pytorch_pinn) as trainer:
            first = run_case(POISEUILLE, "pinn", epochs=20, record=False)
            self.assertEqual(first.sim.epochs_trained, 20)
            self.assertEqual(trainer.call_count, 1)
            # Fewer epochs requested: the 20-epoch model is good enough and is reused.
            reused = run_case(POISEUILLE, "pinn", epochs=10, record=False)
            self.assertEqual(trainer.call_count, 1)
            self.assertEqual(reused.sim.epochs_trained, 20)
            # More epochs requested: the cached model is too short, train again.
            longer = run_case(POISEUILLE, "pinn", epochs=30, record=False)
            self.assertEqual(trainer.call_count, 2)
            self.assertEqual(longer.sim.epochs_trained, 30)
        metas = sorted(f for f in os.listdir(models_dir()) if f.startswith("pinn_") and f.endswith(".json"))
        with open(os.path.join(models_dir(), metas[-1]), "r", encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["epochs_trained"], 30)

    def test_legacy_metadata_without_epochs(self):
        from vectornaut.solvers.model_cache import _cached_epochs
        self.assertEqual(_cached_epochs({"loss_history": [1.0, 0.5, 0.1]}), 3)
        self.assertIsNone(_cached_epochs({"final_loss": 0.1}))
        self.assertIsNone(_cached_epochs({"loss_history": []}))
        self.assertEqual(_cached_epochs({"epochs_trained": 300, "loss_history": [1.0]}), 300)


def _validator_run(solution_primary, solution_reference=None, bcs=None, solver_method="analytical", final_loss=0.0,
                   points=None):
    points = list(np.linspace(0.0, 1.0, 20)) if points is None else points
    reference = solution_primary if solution_reference is None else solution_reference
    return {
        "miner": {
            "governing_equation": "d2u_dy2 = -G / mu",
            "boundary_conditions": bcs or ["u(0) = 0", "u(1) = 0"],
            "dependent_variables": ["u"],
            "independent_variables": ["y"],
        },
        "auditor": {"audit_passed": True},
        "simulator": {
            "solver_method": solver_method,
            "final_loss": final_loss,
            "relative_error": 0.0,
            "performance_gain_pct": 0.0,
            "sample_points": [float(p) for p in points],
            "solution_primary": [float(v) for v in solution_primary],
            "solution_reference": [float(v) for v in reference],
            "primary_metric_value": 2.0,
            "reference_metric_value": 2.0,
        },
    }


class ValidatorChecksTest(unittest.TestCase):
    y = np.linspace(0.0, 1.0, 20)

    def _failed(self, result):
        return {check.name for check in result.checks if not check.passed}

    def test_steep_exact_neumann_solution_passes(self):
        u = 10.0 * self.y ** 2 - 20.0 * self.y
        result = validate_run_data(_validator_run(u, bcs=["u(0) = 0", "du_dy(1) = 0"]))
        self.assertEqual(result.status, "pass", self._failed(result))

    def test_neumann_tolerance_scales_with_units(self):
        # Same steep profile on a 10 um film with values ~1e-4: still exact, still passes.
        y = self.y * 1.0e-5
        u = 1.0e6 * y ** 2 - 20.0 * y
        result = validate_run_data(_validator_run(u, bcs=["u(0) = 0", "du_dy(1e-05) = 0"], points=list(y)))
        self.assertEqual(result.status, "pass", self._failed(result))

    def test_wrong_neumann_value_is_flagged(self):
        u = self.y ** 2 - 2.0 * self.y  # du/dy(1) = 0, BC claims 0.5
        result = validate_run_data(_validator_run(u, bcs=["u(0) = 0", "du_dy(1) = 0.5"]))
        self.assertIn("physics_derivative_boundary_conditions", self._failed(result))
        self.assertEqual(result.recommended_action, "rerun_solver")

    def test_zero_and_sign_flipped_profiles_are_flagged(self):
        exact = 2.0 * self.y * (1.0 - self.y)
        for wrong in (np.zeros_like(exact), -exact):
            result = validate_run_data(_validator_run(wrong, exact))
            self.assertEqual(result.status, "warn")
            self.assertEqual(result.recommended_action, "rerun_solver")
            self.assertIn("numeric_primary_reference_agreement", self._failed(result))

    def test_small_primary_reference_deviation_passes(self):
        exact = 2.0 * self.y * (1.0 - self.y)
        result = validate_run_data(_validator_run(exact * 1.02, exact))
        self.assertEqual(result.status, "pass", self._failed(result))

    def test_dynamic_script_baseline_reference_is_not_compared(self):
        # A generated script's solution_reference is the no-effect baseline, not a second solve.
        slip = 2.0 * (self.y + 0.25) / 1.25
        run = _validator_run(slip, 2.0 * self.y, bcs=["u(1) = 2"], solver_method="dynamic_script")
        run["simulator"]["validation_passed"] = True
        result = validate_run_data(run)
        self.assertEqual(result.status, "pass", self._failed(result))
        self.assertNotIn("numeric_primary_reference_agreement", {check.name for check in result.checks})

    def test_offset_field_agreement_uses_variation_not_magnitude(self):
        # Flat 285 K instead of a 310 -> 260 K profile: only ~5 % of the magnitude, but wrong.
        exact = 310.0 - 50.0 * self.y
        result = validate_run_data(_validator_run(np.full_like(exact, 285.0), exact, bcs=["T(0) = 310", "T(1) = 260"]))
        self.assertIn("numeric_primary_reference_agreement", self._failed(result))

    def test_diverged_pinn_requests_solver_rerun(self):
        exact = 2.0 * self.y * (1.0 - self.y)
        result = validate_run_data(_validator_run(exact, solver_method="pinn", final_loss=1.5e6))
        self.assertEqual(result.status, "warn")
        self.assertEqual(result.recommended_action, "rerun_solver")
        self.assertIn("solver_pinn_divergence", self._failed(result))

    def test_moderate_pinn_loss_only_requests_inspection(self):
        exact = 2.0 * self.y * (1.0 - self.y)
        result = validate_run_data(_validator_run(exact, solver_method="pinn", final_loss=5.0))
        self.assertEqual(result.status, "warn")
        self.assertEqual(result.recommended_action, "inspect")
        self.assertEqual(self._failed(result), {"solver_pinn_loss"})


class PipelineFallbackOnDivergedPinnTest(unittest.TestCase):
    def test_diverged_pinn_falls_back_to_analytical(self):
        from tests.test_pipeline_unit import Dumpable, FakeAuditor, FakeFormulator, FakeMiner, FakeSynthesizer
        from vectornaut.pipeline import PipelineRunner, PipelineRunRequest

        def solver(miner_output, auditor_output, epochs):
            pinn = auditor_output.solver_method == "pinn"
            return Dumpable(
                solver_method=auditor_output.solver_method,
                epochs_trained=10 if pinn else 0,
                final_loss=1.5e6 if pinn else 0.0,
                loss_history=[1.5e6] if pinn else [],
                performance_gain_pct=10.0,
                relative_error=0.0,
                sample_points=[0.0, 0.5, 1.0],
                solution_primary=[0.0, 0.5, 1.0],
                solution_reference=[0.0, 0.5, 1.0],
                primary_metric_value=1.0,
                reference_metric_value=1.0,
            )

        runner = PipelineRunner(miner=FakeMiner(), formulator=FakeFormulator(), auditor=FakeAuditor(solver_method="pinn"),
                                optimizer=object(), synthesizer=FakeSynthesizer(), solver=solver)
        output = runner.run(PipelineRunRequest(query="test", is_mock=True, epochs=5, max_optimization_rounds=1))
        self.assertEqual(output["auditor"]["solver_method"], "analytical")
        self.assertEqual(output["simulator"]["solver_method"], "analytical")
        self.assertEqual(output["validation"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
