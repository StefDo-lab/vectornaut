# -*- coding: utf-8 -*-
"""Helpers for tests/test_solver_accuracy.py.

Builds MinerOutput/AuditorOutput pairs for problems with known closed-form
solutions, runs ``dispatch_and_solve`` quietly, and measures the error of the
returned ``solution_primary`` against the exact solution (not against the
app's own ``solution_reference``, which is produced by the same code).

Set ``VECTORNAUT_ACCURACY_TABLE=1`` to print a case x method measurement table
at the end of the test run.
"""
import contextlib
import io
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from vectornaut.config import AuditedParameter, AuditorOutput, MinerOutput, SimulatorOutput

UI_METADATA = {
    "domain_name": "Accuracy",
    "independent_var": {"label": "coordinate", "unit": "m"},
    "dependent_var": {"label": "field", "unit": "-"},
    "primary_metric": {"label": "primary"},
    "reference_metric": {"label": "reference"},
    "performance_gain": {"label": "gain"},
}


@dataclass
class Case:
    """A boundary value problem with a known exact solution."""
    name: str
    equation: str
    bcs: List[str]
    independent: List[str]
    dependent: str
    exact: Callable[..., np.ndarray]
    params: Dict[str, float] = field(default_factory=dict)
    coefficient: float = 0.0
    # Exact value of primary_metric_value: du/dx at the lower wall (1D) or the
    # mean of the field over the sample points (2D, computed from `exact`).
    exact_metric: Optional[float] = None
    exact_domain: Tuple[float, float] = (0.0, 1.0)

    @property
    def is_2d(self) -> bool:
        return len(self.independent) == 2


def build_inputs(case: Case, method: str, design_name: str = "Accuracy Case") -> Tuple[MinerOutput, AuditorOutput]:
    miner = MinerOutput(
        design_name=design_name,
        inspiration_source="manufactured solution",
        domain="Verification",
        physical_mechanism="closed-form reference problem",
        parameters=[],
        governing_equation=case.equation,
        boundary_conditions=list(case.bcs),
        independent_variables=list(case.independent),
        dependent_variables=[case.dependent],
        svg_schematic="<svg></svg>",
    )
    auditor = AuditorOutput(
        audit_passed=True,
        audit_notes="accuracy test",
        audited_parameters=[AuditedParameter(name=k, value=v) for k, v in case.params.items()],
        dimensionless_numbers=[],
        simulation_coefficient=case.coefficient,
        solver_method=method,
        ui_metadata=UI_METADATA,
    )
    return miner, auditor


@dataclass
class Run:
    case: Case
    method: str
    epochs: int
    sim: SimulatorOutput
    runtime_s: float
    log: str
    miner: MinerOutput
    auditor: AuditorOutput

    @property
    def points(self) -> np.ndarray:
        return np.asarray(self.sim.sample_points, dtype=float)

    def exact_values(self) -> np.ndarray:
        pts = self.points
        if pts.ndim == 2:
            return np.asarray(self.case.exact(pts[:, 0], pts[:, 1]), dtype=float) * np.ones(len(pts))
        return np.asarray(self.case.exact(pts), dtype=float) * np.ones(len(pts))

    def errors(self, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        exact = self.exact_values()
        primary = np.asarray(self.sim.solution_primary, dtype=float)
        if mask is not None:
            exact, primary = exact[mask], primary[mask]
        diff = primary - exact
        scale = float(np.max(np.abs(exact))) or 1.0
        return {
            "max_abs": float(np.max(np.abs(diff))),
            "rel_max": float(np.max(np.abs(diff)) / scale),
            "rel_l2": float(np.linalg.norm(diff) / (np.linalg.norm(exact) or 1.0)),
            # Same formula the dispatcher uses for `relative_error`, but against the exact solution.
            "rel_l1_like_app": float(np.sum(np.abs(diff)) / np.sum(np.abs(exact) + 1e-8)),
        }


MEASUREMENTS: List[Dict[str, object]] = []


def run_case(case: Case, method: str, epochs: int = 200, record: bool = True, label: Optional[str] = None) -> Run:
    """Runs dispatch_and_solve with stdout captured. VECTORNAUT_DATA_DIR must already point at a temp dir."""
    # Import lazily so the data dir env var set by the test class is what the solver sees.
    from vectornaut.solver_dispatcher import dispatch_and_solve

    miner, auditor = build_inputs(case, method)
    torch.manual_seed(0)
    buf = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = dispatch_and_solve(miner, auditor, epochs=epochs)
    runtime = time.perf_counter() - start
    run = Run(case, method, epochs, sim, runtime, buf.getvalue(), miner, auditor)
    if record:
        try:
            errs = run.errors()
        except Exception:  # pragma: no cover - diagnostic only
            errs = {"max_abs": float("nan"), "rel_l2": float("nan")}
        MEASUREMENTS.append({
            "case": label or case.name,
            "requested": method,
            "used": sim.solver_method,
            "max_abs": errs["max_abs"],
            "rel_l2": errs["rel_l2"],
            "metric": sim.primary_metric_value,
            "exact_metric": exact_metric(run),
            "runtime_s": runtime,
        })
    return run


def exact_metric(run: Run) -> Optional[float]:
    if run.case.is_2d:
        return float(np.mean(run.exact_values()))
    return run.case.exact_metric


def format_table(rows: Sequence[Dict[str, object]]) -> str:
    header = f"{'case':34s} {'requested':10s} {'used':10s} {'max_abs_err':>12s} {'rel_L2_err':>11s} {'metric':>12s} {'exact_metric':>12s} {'time_s':>7s}"
    lines = [header, "-" * len(header)]
    for r in rows:
        em = r["exact_metric"]
        lines.append(
            f"{str(r['case']):34s} {str(r['requested']):10s} {str(r['used']):10s} {r['max_abs']:12.3e} {r['rel_l2']:11.3e} "
            f"{r['metric']:12.5g} {(em if em is not None else float('nan')):12.5g} {r['runtime_s']:7.2f}"
        )
    return "\n".join(lines)


def maybe_print_table() -> None:
    if os.environ.get("VECTORNAUT_ACCURACY_TABLE") and MEASUREMENTS:
        print("\n\nSolver accuracy measurements (errors vs exact solution at sample_points):")
        print(format_table(MEASUREMENTS))
