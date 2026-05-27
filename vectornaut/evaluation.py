# -*- coding: utf-8 -*-
import json
import math
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


EVAL_RUNS_DIR = "eval_runs"


def make_eval_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{timestamp}_{uuid.uuid4().hex[:8]}"


def utcish_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def collect_non_finite_numbers(value: Any, path: str = "$") -> List[str]:
    problems: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            problems.extend(collect_non_finite_numbers(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            problems.extend(collect_non_finite_numbers(child, f"{path}[{idx}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        problems.append(path)
    return problems


def get_nested(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def numeric_list(values: Any) -> bool:
    return isinstance(values, list) and all(is_finite_number(item) for item in values)


def evaluate_run_output(run_data: Dict[str, Any], criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    criteria = criteria or {}
    max_relative_error = float(criteria.get("max_relative_error", 1.0))
    min_sample_count = int(criteria.get("min_sample_count", 2))
    min_performance_gain = criteria.get("min_performance_gain_pct")
    expected_solver = criteria.get("expected_solver_method")
    required_paths = criteria.get("required_paths") or [
        "miner.design_name",
        "miner.governing_equation",
        "auditor.audit_passed",
        "simulator.solver_method",
        "simulator.relative_error",
        "simulator.sample_points",
        "simulator.solution_primary",
        "simulator.solution_reference",
    ]

    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def add_check(name: str, passed: bool, detail: str, severity: str = "error") -> None:
        checks.append({
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
        })

    missing_paths = [path for path in required_paths if get_nested(run_data, path) is None]
    add_check(
        "required_fields",
        not missing_paths,
        "All required paths are present." if not missing_paths else f"Missing paths: {missing_paths}",
    )

    non_finite_paths = collect_non_finite_numbers(run_data)
    add_check(
        "finite_numbers",
        not non_finite_paths,
        "All numeric values are finite." if not non_finite_paths else f"Non-finite numeric values at: {non_finite_paths[:25]}",
    )

    audit_passed = get_nested(run_data, "auditor.audit_passed")
    add_check(
        "audit_passed",
        audit_passed is True,
        "Auditor passed the design." if audit_passed is True else f"Auditor status is {audit_passed!r}.",
    )

    sim = run_data.get("simulator", {}) if isinstance(run_data.get("simulator"), dict) else {}
    solver_method = sim.get("solver_method")
    if expected_solver:
        add_check(
            "expected_solver",
            solver_method == expected_solver,
            f"Solver is {solver_method!r}, expected {expected_solver!r}.",
        )

    relative_error = sim.get("relative_error")
    add_check(
        "relative_error_threshold",
        is_finite_number(relative_error) and float(relative_error) <= max_relative_error,
        f"Relative error is {relative_error!r}; threshold is <= {max_relative_error}.",
    )

    sample_points = sim.get("sample_points", [])
    primary = sim.get("solution_primary", [])
    reference = sim.get("solution_reference", [])
    enough_samples = isinstance(sample_points, list) and len(sample_points) >= min_sample_count
    aligned_lengths = isinstance(primary, list) and isinstance(reference, list) and len(primary) == len(reference) and len(primary) > 0
    add_check(
        "solution_arrays",
        enough_samples and aligned_lengths,
        f"samples={len(sample_points) if isinstance(sample_points, list) else 'n/a'}, primary={len(primary) if isinstance(primary, list) else 'n/a'}, reference={len(reference) if isinstance(reference, list) else 'n/a'}",
    )

    add_check(
        "solution_values_finite",
        numeric_list(primary) and numeric_list(reference),
        "Primary and reference solutions are finite numeric arrays.",
    )

    if min_performance_gain is not None:
        gain = sim.get("performance_gain_pct")
        add_check(
            "minimum_performance_gain",
            is_finite_number(gain) and float(gain) >= float(min_performance_gain),
            f"Performance gain is {gain!r}; expected >= {min_performance_gain}.",
        )

    validation_passed = sim.get("validation_passed")
    if validation_passed is False:
        add_check("generated_validation", False, "Generated validation tests explicitly failed.")
    elif validation_passed is True:
        add_check("generated_validation", True, "Generated validation tests passed.", severity="info")
    else:
        warnings.append("No generated validation result was attached to the simulator output.")

    report_complete = bool(run_data.get("report_md")) and bool(run_data.get("synthesis"))
    add_check(
        "report_complete",
        report_complete,
        "Report markdown and synthesis are present." if report_complete else "Report markdown or synthesis is missing.",
        severity="warning",
    )

    error_checks = [check for check in checks if check["severity"] == "error"]
    warning_checks = [check for check in checks if check["severity"] == "warning"]
    errors_passed = all(check["passed"] for check in error_checks)

    score_total = len(checks) or 1
    score_passed = sum(1 for check in checks if check["passed"])
    scores = {
        "overall": round(score_passed / score_total, 3),
        "schema_valid": not missing_paths,
        "physics_plausible": audit_passed is True and not non_finite_paths,
        "solver_stable": any(check["name"] == "relative_error_threshold" and check["passed"] for check in checks)
        and any(check["name"] == "solution_arrays" and check["passed"] for check in checks),
        "report_complete": report_complete,
    }

    return {
        "passed": errors_passed,
        "scores": scores,
        "checks": checks,
        "warnings": warnings,
        "summary": {
            "passed_checks": score_passed,
            "total_checks": score_total,
            "failed_errors": [check for check in error_checks if not check["passed"]],
            "failed_warnings": [check for check in warning_checks if not check["passed"]],
        },
    }


def build_eval_trace(
    run_id: str,
    request_data: Dict[str, Any],
    run_data: Optional[Dict[str, Any]],
    started_at: str,
    finished_at: str,
    duration_ms: int,
    status: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    run_data = run_data or {}
    stages: Dict[str, Dict[str, Any]] = {}

    def stage(name: str, output_key: str, ok_detail: str) -> None:
        output = run_data.get(output_key)
        stages[name] = {
            "status": "ok" if output else "missing",
            "detail": ok_detail if output else f"No {output_key} output present.",
            "output": output,
        }

    stage("miner", "miner", "Concept mining/formulation output captured.")
    stage("auditor", "auditor", "Auditor output captured.")
    stage("solver", "simulator", "Solver output captured.")
    stage("synthesizer", "synthesis", "Synthesis output captured.")

    optimization_history = run_data.get("optimization_history")
    stages["optimizer"] = {
        "status": "ok" if optimization_history else "skipped_or_missing",
        "detail": f"{len(optimization_history)} optimization rounds captured." if isinstance(optimization_history, list) else "No optimization history present.",
        "output": optimization_history,
    }

    if error:
        stages["error"] = {
            "status": "failed",
            "detail": error,
            "output": None,
        }

    return {
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "request": request_data,
        "stages": stages,
    }


def persist_eval_record(record: Dict[str, Any]) -> str:
    os.makedirs(EVAL_RUNS_DIR, exist_ok=True)
    run_id = record["run_id"]
    path = os.path.abspath(os.path.join(EVAL_RUNS_DIR, f"{run_id}.json"))
    record["storage_path"] = path
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    return path


def load_eval_record(run_id: str) -> Dict[str, Any]:
    safe_id = os.path.basename(run_id).replace(".json", "")
    path = os.path.join(EVAL_RUNS_DIR, f"{safe_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Eval run not found: {run_id}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_eval_records(limit: int = 50) -> List[Dict[str, Any]]:
    if not os.path.exists(EVAL_RUNS_DIR):
        return []

    files = [
        os.path.join(EVAL_RUNS_DIR, name)
        for name in os.listdir(EVAL_RUNS_DIR)
        if name.endswith(".json")
    ]
    files.sort(key=lambda path: os.path.getmtime(path), reverse=True)

    records: List[Dict[str, Any]] = []
    for path in files[: max(1, limit)]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            records.append({
                "run_id": data.get("run_id"),
                "name": data.get("name"),
                "status": data.get("status"),
                "passed": data.get("evaluation", {}).get("passed"),
                "started_at": data.get("trace", {}).get("started_at"),
                "duration_ms": data.get("trace", {}).get("duration_ms"),
                "storage_path": os.path.abspath(path),
            })
        except Exception:
            records.append({
                "run_id": os.path.basename(path).replace(".json", ""),
                "status": "unreadable",
                "storage_path": os.path.abspath(path),
            })
    return records


def load_history_record(history_file: str) -> Dict[str, Any]:
    safe_name = os.path.basename(history_file)
    path = os.path.join("history", safe_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"History file not found: {history_file}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
