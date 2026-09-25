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
    coordinate = _coordinate_value(a)
    if coordinate is None:
        return None
    return abs(coordinate - b)


def _coordinate_value(a: Any) -> Optional[float]:
    if _is_finite_number(a):
        return float(a)
    if isinstance(a, Sequence) and not isinstance(a, (str, bytes)) and a and _is_finite_number(a[0]):
        return float(a[0])
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


def _parse_simple_neumann_bc(boundary_condition: str, dependent_var: str, independent_var: str) -> Optional[tuple[float, float]]:
    number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    patterns = [
        rf"^\s*{re.escape(dependent_var)}\s*'\s*\(\s*{number}\s*\)\s*=\s*{number}\s*$",
        rf"^\s*d{re.escape(dependent_var)}_d{re.escape(independent_var)}\s*\(\s*{number}\s*\)\s*=\s*{number}\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, boundary_condition)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None


def _numeric_pairs(sample_points: List[Any], solution: List[float]) -> List[tuple[float, float]]:
    pairs = []
    for point, value in zip(sample_points, solution):
        coordinate = _coordinate_value(point)
        if coordinate is not None and _is_finite_number(value):
            pairs.append((coordinate, float(value)))
    return sorted(set(pairs), key=lambda item: item[0])


def _estimate_derivative(sample_points: List[Any], solution: List[float], target: float) -> Optional[float]:
    """
    Estimates du/dx at target from the sampled solution with a second-order stencil:
    the derivative of the quadratic through the three samples nearest to target
    (one-sided 3-point formula at the domain ends, central difference inside; also
    valid for non-uniform spacing). Falls back to the secant slope with two samples.
    """
    pairs = _numeric_pairs(sample_points, solution)
    if len(pairs) < 2:
        return None
    if len(pairs) == 2:
        (x0, y0), (x1, y1) = pairs
        dx = x1 - x0
        if abs(dx) < 1e-300:
            return None
        return (y1 - y0) / dx

    # Index of the sample closest to target; the stencil is centred on it where possible.
    nearest = min(range(len(pairs)), key=lambda idx: abs(pairs[idx][0] - target))
    start = min(max(nearest - 1, 0), len(pairs) - 3)
    (x0, y0), (x1, y1), (x2, y2) = pairs[start:start + 3]
    d01, d02, d12 = x0 - x1, x0 - x2, x1 - x2
    if d01 == 0.0 or d02 == 0.0 or d12 == 0.0:
        return None
    # Derivative of the Lagrange interpolant through (x0, y0), (x1, y1), (x2, y2) at target.
    derivative = (
        y0 * ((target - x1) + (target - x2)) / (d01 * d02)
        - y1 * ((target - x0) + (target - x2)) / (d01 * d12)
        + y2 * ((target - x0) + (target - x1)) / (d02 * d12)
    )
    return derivative if math.isfinite(derivative) else None


def _derivative_scale(sample_points: List[Any], solution: List[float]) -> tuple[float, float]:
    """
    Returns (slope_scale, value_scale) of a sampled 1D solution: the mean slope
    |range(u)| / |domain length| and max|u| / |domain length|. Used to make the
    derivative boundary tolerance independent of units and grid size.
    """
    pairs = _numeric_pairs(sample_points, solution)
    if len(pairs) < 2:
        return 0.0, 0.0
    length = pairs[-1][0] - pairs[0][0]
    if not length > 0.0:
        return 0.0, 0.0
    values = [value for _, value in pairs]
    return (max(values) - min(values)) / length, max(abs(value) for value in values) / length


def _primary_reference_deviation(primary: List[float], reference: List[float]) -> tuple[float, float]:
    """
    Scale-aware disagreement between the primary and the reference solution.
    Returns (rms deviation, max deviation), both divided by the variation of the fields
    (the larger of their ranges, floored at 1 % of their magnitude so that nearly
    constant fields do not turn round-off into large relative errors). E.g. the max
    deviation is 0 for identical arrays, 1 for an all-zero primary against a
    non-negative profile that touches 0, and 2 for a sign-flipped one.
    """
    diffs = [p - r for p, r in zip(primary, reference)]
    if not diffs:
        return 0.0, 0.0
    magnitude = max(max(abs(v) for v in primary), max(abs(v) for v in reference))
    variation = max(max(reference) - min(reference), max(primary) - min(primary))
    scale = max(variation, 0.01 * magnitude, 1e-12)
    rms = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    return rms / scale, max(abs(d) for d in diffs) / scale


# Failed error checks that mean the request itself is physically impossible as stated;
# the pipeline answers reject_request by stopping with a rejection instead of re-mining.
_REQUEST_REJECTION_CHECKS = {"physics_closed_system_efficiency"}

# Phrases that describe a closed or adiabatic system without energy input. Deliberately
# narrow: "adiabatic tip", "closed loop" or "closed refrigerant circuit" do not match.
_CLOSED_SYSTEM_PATTERNS = (
    re.compile(r"adiabat\w*\s+(?:\w+\s+)?(?:system|gehäuse|gehaeuse|enclosure|housing|container|behälter|behaelter)", re.IGNORECASE),
    re.compile(r"(?:isolated|abgeschlossen\w*)\s+(?:\w+\s+)?system", re.IGNORECASE),
    re.compile(r"(?:ohne|without|no)\s+(?:jede\w*\s+|any\s+|external\s+|externe\w*\s+)?(?:energiezufuhr|energieeintrag|energy\s+input|energy\s+supply|power\s+input)", re.IGNORECASE),
)
_EFFICIENCY_NAME = re.compile(r"efficien|wirkungsgrad|(?:^|_)cop(?:$|_)|coefficient_of_performance|leistungszahl", re.IGNORECASE)
_AMPLIFICATION_NAME = re.compile(r"amplif|verstärk|verstaerk|multiplication|multiplier|(?:^|_)gain_(?:factor|ratio)", re.IGNORECASE)
_PERCENT_NAME = re.compile(r"pct|percent|prozent", re.IGNORECASE)
_RATIO_NAME = re.compile(r"ratio|factor|faktor|fraction|(?:^|_)cop(?:$|_)|coefficient_of_performance|leistungszahl", re.IGNORECASE)
_ENERGY_CONTEXT = re.compile(r"therm|heat|wärm|waerm|energ|temperat", re.IGNORECASE)


def _run_parameters(miner: Dict[str, Any], auditor: Dict[str, Any]) -> Dict[str, Any]:
    params = auditor.get("audited_parameters_dict")
    if isinstance(params, dict) and params:
        return params
    listed = auditor.get("audited_parameters") or miner.get("parameters") or []
    return {
        item.get("name"): item.get("value")
        for item in (_as_dict(entry) for entry in listed)
        if item.get("name")
    }


def check_closed_system_efficiency(
    miner_output: Any,
    auditor_output: Any,
    user_query: Optional[str] = None,
) -> Optional[ValidationCheck]:
    """
    Energy-conservation guard: an efficiency, COP or heat amplification factor above 1
    (above 100 for percent values) is impossible in a closed/adiabatic system without
    energy input. Returns a failed error check when the request or concept describes
    such a system and a parameter violates it; otherwise None (no check recorded).
    """
    miner = _as_dict(miner_output)
    auditor = _as_dict(auditor_output)
    text = " ".join(str(part or "") for part in (
        user_query,
        miner.get("design_name"),
        miner.get("domain"),
        miner.get("physical_mechanism"),
    ))
    closed_match = next((m for m in (p.search(text) for p in _CLOSED_SYSTEM_PATTERNS) if m), None)
    if closed_match is None:
        return None
    energy_context = bool(_ENERGY_CONTEXT.search(text))

    violations = []
    for name, value in _run_parameters(miner, auditor).items():
        if not _is_finite_number(value):
            continue
        name_text = str(name)
        if _EFFICIENCY_NAME.search(name_text):
            limit = 100.0 if _PERCENT_NAME.search(name_text) or not _RATIO_NAME.search(name_text) else 1.0
        elif _AMPLIFICATION_NAME.search(name_text) and energy_context:
            limit = 100.0 if _PERCENT_NAME.search(name_text) else 1.0
        else:
            continue
        if float(value) > limit * (1.0 + 1e-9):
            violations.append(f"{name_text}={float(value):g} > {limit:g}")

    if not violations:
        return None
    return ValidationCheck(
        name="physics_closed_system_efficiency",
        passed=False,
        severity="error",
        score=0.0,
        detail=(
            f"The system is described as closed/adiabatic without energy input (\"{closed_match.group(0)}\"), "
            f"but {', '.join(violations)}: more energy out than in violates energy conservation "
            "(first law of thermodynamics). The request is not physically feasible as stated."
        ),
    )


# Failed warning checks that indicate the solver (not the model) produced a bad result;
# the pipeline answers rerun_solver by trying the other deterministic solvers.
_RERUN_SOLVER_WARNINGS = {"numeric_primary_reference_agreement", "solver_pinn_divergence"}

# Solver methods whose solution_reference is an independent solution of the same problem.
_REFERENCE_SOLVER_METHODS = {"analytical", "scipy", "pinn", "fdm"}


def _dynamic_script_result_warnings(simulator: Dict[str, Any]) -> List[tuple[str, str]]:
    """(check name, detail) for each result warning recorded by the dynamic_script path
    (noise-level metrics, baseline hard-constraint violations; see solvers/dynamic_script.py)."""
    sweep = simulator.get("parameter_sweep") or {}
    warnings = sweep.get("warnings") if isinstance(sweep, dict) else None
    return [
        (f"dynamic_script_{item.get('code', 'result_warning')}", str(item.get("message", "")))
        for item in (warnings or [])
        if isinstance(item, dict)
    ]


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
    if failed_warnings and any(
        "boundary" in check.name or "relative_error" in check.name or check.name in _RERUN_SOLVER_WARNINGS
        for check in failed_warnings
    ):
        recommended_action = "rerun_solver"
    if any(check.name in _REQUEST_REJECTION_CHECKS for check in failed_errors):
        recommended_action = "reject_request"

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
    user_query: Optional[str] = None,
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

    # The reported relative_error is computed by the solver itself; independently check
    # that the primary solution agrees with the reference solution it is compared to.
    # Only for the built-in solvers, whose reference solves the same problem (analytical /
    # SciPy in 1D, FDM in 2D). A generated dynamic script's reference is the unperturbed
    # no-effect baseline, which is expected to differ from the primary solution.
    if (aligned and primary_finite and reference_finite
            and str(simulator.get("solver_method") or "").lower() in _REFERENCE_SOLVER_METHODS):
        rms_deviation, max_deviation = _primary_reference_deviation(
            [float(value) for value in primary], [float(value) for value in reference]
        )
        # The RMS (not the max) deviation is thresholded: local spikes, e.g. a PINN at the
        # discontinuous corners of a lid-driven cavity, should not trigger a rerun alone.
        max_rms_deviation = float(criteria.get("max_primary_reference_deviation", 0.1))
        add(
            "numeric_primary_reference_agreement",
            rms_deviation <= max_rms_deviation,
            f"primary vs reference: rms deviation={rms_deviation:.4g} (<= {max_rms_deviation}), "
            f"max deviation={max_deviation:.4g}, relative to the field variation.",
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

    objective = simulator.get("objective_metric") or auditor.get("objective_metric") or {}
    score_field = objective.get("score_field", "performance_gain_pct")
    direction = objective.get("direction", "maximize")
    threshold = float(objective.get("acceptance_threshold", 0.0) or 0.0)
    objective_score = simulator.get(score_field)
    if objective:
        objective_ok = _is_finite_number(objective_score) and (
            float(objective_score) >= threshold if direction != "minimize" else float(objective_score) <= threshold
        )
        add(
            "objective_metric_contract",
            objective_ok,
            f"{score_field}={objective_score!r}, direction={direction}, threshold={threshold}.",
            severity="warning",
            score=0.45,
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
        # A PINN whose loss is orders of magnitude above the threshold has clearly not
        # converged; request a solver rerun (fallback to the deterministic solvers)
        # instead of only flagging it for inspection.
        divergence_loss = float(criteria.get("pinn_divergence_final_loss", 1000.0 * max_pinn_loss))
        if not (_is_finite_number(final_loss) and float(final_loss) <= divergence_loss):
            add(
                "solver_pinn_divergence",
                False,
                f"PINN final_loss={final_loss!r} exceeds divergence threshold {divergence_loss}; training did not converge.",
                severity="warning",
                score=0.2,
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
        for name, detail in _dynamic_script_result_warnings(simulator):
            add(name, False, detail, severity="warning", score=0.4)
    elif solver_method in {"analytical", "scipy", "fdm", "mock"}:
        add("solver_method_confidence", True, f"Solver method {solver_method!r} has deterministic validation coverage.", severity="info")
    else:
        add("solver_method_known", False, f"Unknown solver method: {solver_method!r}.", severity="warning", score=0.3)

    if aligned and primary_finite:
        dependent_vars = miner.get("dependent_variables") or []
        dependent_var = dependent_vars[0] if dependent_vars else "u"
        independent_vars = miner.get("independent_variables") or []
        independent_var = independent_vars[0] if independent_vars else "x"
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

        simple_derivative_bcs = [
            parsed for parsed in (
                _parse_simple_neumann_bc(str(bc), dependent_var, independent_var)
                for bc in miner.get("boundary_conditions", []) or []
            )
            if parsed is not None
        ]
        if simple_derivative_bcs:
            # An explicit derivative_boundary_tolerance is an absolute tolerance. Otherwise
            # the tolerance scales with the slope of the solution (range(u) / domain length)
            # and the prescribed derivative, so it does not depend on units or grid size.
            explicit_tolerance = criteria.get("derivative_boundary_tolerance")
            relative_tolerance = float(criteria.get("derivative_boundary_rel_tolerance", 0.05))
            slope_scale, value_scale = _derivative_scale(sample_points, primary)
            derivative_residuals = []
            derivative_tolerances = []
            for location, expected in simple_derivative_bcs:
                if explicit_tolerance is not None:
                    tolerance = float(explicit_tolerance)
                else:
                    tolerance = relative_tolerance * max(slope_scale, abs(expected)) + 1e-6 * value_scale + 1e-12
                observed = _estimate_derivative(sample_points, primary, location)
                if observed is None:
                    derivative_residuals.append(float("inf"))
                else:
                    derivative_residuals.append(abs(observed - expected))
                derivative_tolerances.append(tolerance)
            worst = max(
                range(len(derivative_residuals)),
                key=lambda idx: derivative_residuals[idx] / derivative_tolerances[idx] if derivative_tolerances[idx] > 0 else float("inf"),
            )
            max_derivative_residual = derivative_residuals[worst]
            derivative_tolerance = derivative_tolerances[worst]
            add(
                "physics_derivative_boundary_conditions",
                all(
                    math.isfinite(residual) and residual <= tolerance
                    for residual, tolerance in zip(derivative_residuals, derivative_tolerances)
                ),
                f"max simple derivative residual={max_derivative_residual:.4g}, tolerance<={derivative_tolerance:.4g} "
                f"(3-point second-order difference).",
                severity="warning",
                score=0.45,
            )

    if auditor.get("audit_passed") is not None:
        add(
            "auditor_passed",
            auditor.get("audit_passed") is True,
            f"Auditor status is {auditor.get('audit_passed')!r}.",
        )

    closed_system_check = check_closed_system_efficiency(miner, auditor, user_query=user_query or criteria.get("user_query"))
    if closed_system_check is not None:
        checks.append(closed_system_check)

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
        user_query=run_data.get("query"),
    )
