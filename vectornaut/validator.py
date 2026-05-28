# -*- coding: utf-8 -*-
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    severity: str = Field(description="info, warning, or error")
    score: float
    detail: str


class ValidationResult(BaseModel):
    status: str
    reliability: str
    score: float
    checks: List[ValidationCheck]
    warnings: List[str]
    recommended_action: str


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_numeric_sequence(values: Any) -> bool:
    return isinstance(values, list) and all(_is_finite_number(item) for item in values)


def _coordinate_distance(a: Any, b: float) -> Optional[float]:
    if _is_finite_number(a):
        return abs(float(a) - b)
    if isinstance(a, Sequence) and not isinstance(a, (str, bytes)) and a and _is_finite_number(a[0]):
        return abs(float(a[0]) - b)
    return None


def _nearest_solution_value(sample_points: List[Any], solution: List[float], target: float) -> Optional[float]:
    best_idx = None
    best_dist = None
    for idx, point in enumerate(sample_points):
        dist = _coordinate_distance(point, target)
        if dist is None:
            continue
        if best_dist is None or dist < best_dist:
            best_idx = idx
            best_dist = dist
    if best_idx is None or best_idx >= len(solution):
        return None
    return float(solution[best_idx])


def _parse_simple_dirichlet_bc(boundary_condition: str, dependent_var: str) -> Optional[tuple[float, float]]:
    pattern = rf"^\s*{re.escape(dependent_var)}\s*\(\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
    match = re.match(pattern, boundary_condition)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _score_status(checks: List[ValidationCheck]) -> tuple[str, str, float, str]:
    error_checks = [check for check in checks if check.severity == "error"]
    warning_checks = [check for check in checks if check.severity == "warning"]
    failed_errors = [check for check in error_checks if not check.passed]
    failed_warnings = [check for check in warning_checks if not check.passed]

    if failed_errors:
        status = "fail"
        recommended_action = "remine"
    elif failed_warnings:
        status = "warn"
        recommended_action = "inspect"
    else:
        status = "pass"
        recommended_action = "accept"

    if failed_errors and any("solver" in check.name or "relative_error" in check.name for check in failed_errors):
        recommended_action = "rerun_solver"
    if failed_warnings and any("boundary" in check.name or "relative_error" in check.name for check in failed_warnings):
        recommended_action = "rerun_solver"

    if not checks:
        return "fail", "low", 0.0, "inspect"

    weighted_total = 0.0
    weighted_score = 0.0
    weights = {"error": 1.0, "warning": 0.6, "info": 0.25}
    for check in checks:
        weight = weights.get(check.severity, 0.5)
        weighted_total += weight
        weighted_score += max(0.0, min(1.0, check.score)) * weight

    score = round(weighted_score / weighted_total, 3) if weighted_total else 0.0
    if status == "fail" or score < 0.55:
        reliability = "low"
    elif status == "warn" or score < 0.82:
        reliability = "medium"
    else:
        reliability = "high"
    return status, reliability, score, recommended_action


def validate_run_output(
    miner_output: Any,
    auditor_output: Any,
    simulator_output: Any,
    optimization_history: Optional[List[Dict[str, Any]]] = None,
    criteria: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    criteria = criteria or {}
    miner = _as_dict(miner_output)
    auditor = _as_dict(auditor_output)
    simulator = _as_dict(simulator_output)
    checks: List[ValidationCheck] = []
    warnings: List[str] = []

    def add(name: str, passed: bool, detail: str, severity: str = "error", score: Optional[float] = None) -> None:
        checks.append(ValidationCheck(
            name=name,
            passed=bool(passed),
            severity=severity,
            score=1.0 if passed else 0.0 if score is None else score,
            detail=detail,
        ))

    required_sim = [
        "solver_method",
        "relative_error",
        "performance_gain_pct",
        "sample_points",
        "solution_primary",
        "solution_reference",
        "primary_metric_value",
        "reference_metric_value",
    ]
    missing = [name for name in required_sim if simulator.get(name) is None]
    add("schema_required_fields", not missing, "Required simulator fields present." if not missing else f"Missing simulator fields: {missing}.")

    sample_points = simulator.get("sample_points")
    primary = simulator.get("solution_primary")
    reference = simulator.get("solution_reference")
    aligned = (
        isinstance(sample_points, list)
        and isinstance(primary, list)
        and isinstance(reference, list)
        and len(sample_points) > 0
        and len(sample_points) == len(primary) == len(reference)
    )
    add(
        "schema_aligned_solution_arrays",
        aligned,
        f"samples={len(sample_points) if isinstance(sample_points, list) else 'n/a'}, "
        f"primary={len(primary) if isinstance(primary, list) else 'n/a'}, "
        f"reference={len(reference) if isinstance(reference, list) else 'n/a'}.",
    )

    primary_finite = _is_numeric_sequence(primary)
    reference_finite = _is_numeric_sequence(reference)
    metrics = ["relative_error", "performance_gain_pct", "primary_metric_value", "reference_metric_value"]
    invalid_metrics = [name for name in metrics if not _is_finite_number(simulator.get(name))]
    add(
        "numeric_finite_values",
        primary_finite and reference_finite and not invalid_metrics,
        "All solution values and key metrics are finite." if not invalid_metrics else f"Invalid metrics: {invalid_metrics}.",
    )

    relative_error = simulator.get("relative_error")
    max_relative_error = float(criteria.get("max_relative_error", 1.0))
    add(
        "numeric_relative_error",
        _is_finite_number(relative_error) and float(relative_error) <= max_relative_error,
        f"relative_error={relative_error!r}, threshold<={max_relative_error}.",
        severity="warning",
        score=0.4,
    )

    gain = simulator.get("performance_gain_pct")
    max_gain = float(criteria.get("max_abs_performance_gain_pct", 500.0))
    add(
        "physics_performance_gain_sanity",
        _is_finite_number(gain) and abs(float(gain)) <= max_gain,
        f"performance_gain_pct={gain!r}, allowed absolute max={max_gain}.",
        severity="warning",
        score=0.35,
    )

    solver_method = str(simulator.get("solver_method") or "").lower()
    final_loss = simulator.get("final_loss", 0.0)
    if solver_method == "pinn":
        max_pinn_loss = float(criteria.get("max_pinn_final_loss", 1.0))
        add(
            "solver_pinn_loss",
            _is_finite_number(final_loss) and float(final_loss) <= max_pinn_loss,
            f"PINN final_loss={final_loss!r}, threshold<={max_pinn_loss}.",
            severity="warning",
            score=0.4,
        )
    elif solver_method == "dynamic_script":
        validation_passed = simulator.get("validation_passed")
        add(
            "solver_dynamic_script_validation",
            validation_passed is True,
            f"Generated-script validation status is {validation_passed!r}.",
            severity="warning",
            score=0.3,
        )
    elif solver_method in {"analytical", "scipy", "fdm", "mock"}:
        add("solver_method_confidence", True, f"Solver method {solver_method!r} has deterministic validation coverage.", severity="info")
    else:
        add("solver_method_known", False, f"Unknown solver method: {solver_method!r}.", severity="warning", score=0.3)

    if aligned and primary_finite:
        dependent_vars = miner.get("dependent_variables") or []
        dependent_var = dependent_vars[0] if dependent_vars else "u"
        simple_bcs = [
            parsed for parsed in (
                _parse_simple_dirichlet_bc(str(bc), dependent_var)
                for bc in miner.get("boundary_conditions", []) or []
            )
            if parsed is not None
        ]
        if simple_bcs:
            tolerance = float(criteria.get("boundary_tolerance", 0.1))
            residuals = []
            for location, expected in simple_bcs:
                observed = _nearest_solution_value(sample_points, primary, location)
                if observed is None:
                    residuals.append(float("inf"))
                else:
                    residuals.append(abs(observed - expected))
            max_residual = max(residuals) if residuals else 0.0
            add(
                "physics_boundary_conditions",
                math.isfinite(max_residual) and max_residual <= tolerance,
                f"max simple Dirichlet residual={max_residual:.4g}, tolerance<={tolerance}.",
                severity="warning",
                score=0.45,
            )
        else:
            warnings.append("No simple Dirichlet boundary conditions could be validated deterministically.")
            add("physics_boundary_conditions_supported", True, "No simple boundary condition check was applicable.", severity="info", score=0.75)

    if auditor.get("audit_passed") is not None:
        add(
            "auditor_passed",
            auditor.get("audit_passed") is True,
            f"Auditor status is {auditor.get('audit_passed')!r}.",
        )

    if optimization_history is not None:
        add(
            "optimization_history_present",
            isinstance(optimization_history, list),
            f"optimization_history entries={len(optimization_history) if isinstance(optimization_history, list) else 'n/a'}.",
            severity="info",
        )

    status, reliability, score, recommended_action = _score_status(checks)
    warnings.extend([check.detail for check in checks if check.severity == "warning" and not check.passed])

    return ValidationResult(
        status=status,
        reliability=reliability,
        score=score,
        checks=checks,
        warnings=warnings,
        recommended_action=recommended_action,
    )


def validate_run_data(run_data: Dict[str, Any], criteria: Optional[Dict[str, Any]] = None) -> ValidationResult:
    return validate_run_output(
        miner_output=run_data.get("miner", {}),
        auditor_output=run_data.get("auditor", {}),
        simulator_output=run_data.get("simulator", {}),
        optimization_history=run_data.get("optimization_history"),
        criteria=criteria,
    )
