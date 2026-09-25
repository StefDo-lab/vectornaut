# -*- coding: utf-8 -*-
"""
Objective contract, result checks, parameter sweep and validation for the
dynamic_script path (model-generated solver + test scripts).

Reporting rule: the baseline run (the audited parameters) is the reported result.
The sweep is a recommendation only; its best accepted candidate and parameters are
returned in ``parameter_sweep["recommendation"]`` and are never mixed into the
headline numbers.
"""
import math
import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple

from ..config import MinerOutput, AuditorOutput
from ..sandbox import copy_if_exists, run_generated_script, sandbox_workdir, script_timeout_seconds

# Sweep candidates whose numerical error estimate exceeds this are discarded.
MAX_SWEEP_RELATIVE_ERROR = 0.05
# Metrics below NOISE_LEVEL_REL * max|solution| are treated as numerical noise.
NOISE_LEVEL_REL = 1e-6
# Number of parameters swept when the auditor names no design variables.
FALLBACK_SWEEP_PARAMETER_COUNT = 4

REQUIRED_NUMERIC_FIELDS = ("performance_gain_pct", "relative_error", "primary_metric_value", "reference_metric_value")
REQUIRED_LIST_FIELDS = ("sample_points", "solution_primary", "solution_reference")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def dynamic_plots_dir() -> str:
    """
    Where dynamic-script plots are written. Default: the repository's static/plots,
    which the web UI serves as /plots (independent of the working directory).
    VECTORNAUT_PLOTS_DIR overrides it (the UI only shows plots from static/plots).
    """
    return os.path.abspath(os.environ.get("VECTORNAUT_PLOTS_DIR") or os.path.join(_REPO_ROOT, "static", "plots"))


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def solver_output_problems(data: Any) -> List[str]:
    """Contract violations of a solver's output JSON (empty list = acceptable)."""
    if not isinstance(data, dict):
        return ["output JSON is not an object"]
    problems = []
    if data.get("success") is not True:
        problems.append(f'"success" is {data.get("success")!r}, expected true')
    for name in REQUIRED_NUMERIC_FIELDS:
        if not _is_finite_number(data.get(name)):
            problems.append(f"{name}={data.get(name)!r} is missing or not a finite number")
    lengths = {}
    for name in REQUIRED_LIST_FIELDS:
        value = data.get(name)
        if not isinstance(value, list) or not value:
            problems.append(f"{name} is missing or empty")
        else:
            lengths[name] = len(value)
    if len(lengths) == len(REQUIRED_LIST_FIELDS) and len(set(lengths.values())) > 1:
        problems.append(f"solution arrays are not aligned: {lengths}")
    for name in ("solution_primary", "solution_reference"):
        value = data.get(name)
        if isinstance(value, list) and value and not all(_is_finite_number(v) for v in value):
            problems.append(f"{name} contains non-finite or non-numeric values")
    return problems


def _dynamic_objective_contract(auditor_output: AuditorOutput) -> Dict[str, Any]:
    if getattr(auditor_output, "objective_metric", None):
        contract = auditor_output.objective_metric
        return contract.model_dump() if hasattr(contract, "model_dump") else dict(contract)
    ui_meta = auditor_output.ui_metadata.model_dump() if hasattr(auditor_output.ui_metadata, "model_dump") else {}
    return {
        "objective_name": ui_meta.get("performance_gain", {}).get("label", "Performance Gain"),
        "score_field": "performance_gain_pct",
        "direction": "maximize",
        "primary_metric": ui_meta.get("primary_metric", {}).get("label", "Primary Metric"),
        "reference_metric": ui_meta.get("reference_metric", {}).get("label", "Reference Metric"),
        "lower_is_better": False,
        "acceptance_threshold": 0.0,
        "hard_constraints": [f"relative_error <= {MAX_SWEEP_RELATIVE_ERROR}", "parameters within bounds", "finite numeric outputs"],
        "design_variables": None,
        "note": "Dynamic-script variants are ranked by performance_gain_pct. The generated script must keep this metric definition stable across variants.",
    }


# --- hard constraints ---------------------------------------------------------------

_OPS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
}
_CONSTRAINT_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|<|>)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(%?)(.*)$"
)


def parse_hard_constraint(text: str) -> Optional[Tuple[str, str, float]]:
    """
    Parse ``<name> <op> <number>[%] [unit]`` (op one of <=, >=, <, >). Parenthetical
    remarks are ignored. Returns None when the constraint has another form.
    ``1%`` is read as 0.01 unless the name is a percentage (``*_pct``).
    """
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"\([^)]*\)", " ", text.replace("≤", "<=").replace("≥", ">="))
    match = _CONSTRAINT_RE.match(cleaned)
    if not match:
        return None
    name, op, number, percent, rest = match.groups()
    rest = rest.strip()
    # Only a unit or a remark may follow the number, not further arithmetic.
    if rest and (rest[0] in "*/+-^(" or re.match(r"^[A-Za-z_]\w*\s*[*/+\-^]", rest)):
        return None
    value = float(number)
    if percent and not (name.lower().endswith("_pct") or "percent" in name.lower()):
        value /= 100.0
    return name, op, value


def evaluate_hard_constraints(
    constraints: Optional[List[str]], result: Dict[str, Any], params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate the parseable hard constraints against the script output fields and the
    parameters. Returns {"violations": [...], "evaluated": [...], "not_evaluated": [...]}.
    """
    lookup: Dict[str, float] = {}
    for source in (params or {}, result or {}):
        for key, value in source.items():
            if _is_finite_number(value):
                lookup[str(key)] = float(value)
    lowered = {key.lower(): key for key in lookup}
    report = {"violations": [], "evaluated": [], "not_evaluated": []}
    for constraint in constraints or []:
        parsed = parse_hard_constraint(constraint)
        if parsed is None:
            report["not_evaluated"].append({"constraint": constraint, "reason": "unparseable"})
            continue
        name, op, limit = parsed
        key = name if name in lookup else lowered.get(name.lower())
        if key is None:
            report["not_evaluated"].append({"constraint": constraint, "reason": f"unknown quantity '{name}'"})
            continue
        observed = lookup[key]
        passed = _OPS[op](observed, limit)
        entry = {"constraint": constraint, "quantity": key, "value": observed, "op": op, "limit": limit, "passed": passed}
        report["evaluated"].append(entry)
        if not passed:
            report["violations"].append(f"{key}={observed:.6g} violates '{constraint}'")
    return report


# --- noise guard ------------------------------------------------------------------

def noise_level_warning(result: Dict[str, Any]) -> Optional[str]:
    """
    Warning text when primary and reference metric are both at numerical-noise level
    compared with the solution field (|m| <= NOISE_LEVEL_REL * max|solution|); a
    percentage gain computed from such values is meaningless.
    """
    primary = result.get("primary_metric_value")
    reference = result.get("reference_metric_value")
    if not (_is_finite_number(primary) and _is_finite_number(reference)):
        return None
    values = [
        abs(float(v))
        for name in ("solution_primary", "solution_reference")
        for v in (result.get(name) or [])
        if _is_finite_number(v)
    ]
    scale = max(values) if values else 0.0
    tolerance = NOISE_LEVEL_REL * scale
    magnitude = max(abs(float(primary)), abs(float(reference)))
    if magnitude <= tolerance and (values or magnitude == 0.0):
        return (
            f"primary_metric_value={float(primary):.3g} and reference_metric_value={float(reference):.3g} are at "
            f"numerical-noise level (<= {NOISE_LEVEL_REL:g} x max|solution| = {tolerance:.3g}); "
            f"performance_gain_pct={result.get('performance_gain_pct')!r} is not meaningful and is not counted as a gain."
        )
    return None


def evaluate_candidate(result: Dict[str, Any], params: Dict[str, Any], objective: Dict[str, Any]) -> Dict[str, Any]:
    """Acceptance checks for one solver result (baseline or sweep variant)."""
    score_field = objective.get("score_field", "performance_gain_pct")
    reasons: List[str] = []
    if result.get("success") is not True:
        reasons.append(f"script reported success={result.get('success')!r}")
    for name in dict.fromkeys((score_field,) + REQUIRED_NUMERIC_FIELDS):
        if not _is_finite_number(result.get(name)):
            reasons.append(f"{name}={result.get(name)!r} is not a finite number")
    relative_error = result.get("relative_error")
    if _is_finite_number(relative_error) and float(relative_error) > MAX_SWEEP_RELATIVE_ERROR:
        reasons.append(f"relative_error={float(relative_error):.3g} > {MAX_SWEEP_RELATIVE_ERROR}")
    constraints = evaluate_hard_constraints(objective.get("hard_constraints"), result, params)
    reasons.extend(constraints["violations"])
    noise = noise_level_warning(result)
    if noise:
        reasons.append("noise-level metrics")
    return {
        "status": "ok" if not reasons else "rejected",
        "rejection_reasons": reasons,
        "constraint_violations": constraints["violations"],
        "constraints_evaluated": constraints["evaluated"],
        "constraints_not_evaluated": constraints["not_evaluated"],
        "noise_level": bool(noise),
        "noise_warning": noise,
    }


# --- sweep ------------------------------------------------------------------------

def _candidate_parameter_values(current: float, minimum: float, maximum: float) -> List[float]:
    values = [current]
    span = maximum - minimum
    if span <= 0:
        return values
    values.extend([
        current - span * 0.25,
        current + span * 0.25,
        minimum + span * 0.25,
        minimum + span * 0.5,
        minimum + span * 0.75,
    ])
    clean = []
    for value in values:
        clipped = max(minimum, min(maximum, float(value)))
        if not any(abs(clipped - existing) <= max(1e-12, abs(clipped) * 1e-9) for existing in clean):
            clean.append(clipped)
    return clean[:5]


def _run_dynamic_script_once(script_path: str, params: Dict[str, float], output_path: str, plot_path: str) -> Dict[str, Any]:
    """Run the generated solver once in the sandbox; the params/output/plot files are kept at the given paths."""
    params_path = output_path.replace("results_", "params_")
    with open(params_path, "w", encoding="utf-8") as pf:
        json.dump(params, pf, indent=4)
    with sandbox_workdir("vectornaut_sweep_") as workdir:
        copy_if_exists(script_path, os.path.join(workdir, "solver.py"))
        copy_if_exists(params_path, os.path.join(workdir, "params.json"))
        run = run_generated_script(
            ["solver.py", "--params", "params.json", "--output", "results.json", "--plot", "plot.png"],
            timeout=script_timeout_seconds(),
            workdir=workdir,
        )
        has_output = copy_if_exists(os.path.join(workdir, "results.json"), output_path)
        copy_if_exists(os.path.join(workdir, "plot.png"), plot_path)
    if not run.ok:
        raise RuntimeError(run.error_text(2000) or f"Dynamic script variant failed (exit code {run.returncode}).")
    if not has_output:
        raise RuntimeError("Dynamic script variant wrote no output JSON.")
    with open(output_path, "r", encoding="utf-8") as rf:
        return json.load(rf)


def select_sweep_parameters(
    miner_output: MinerOutput, auditor_output: AuditorOutput, objective: Dict[str, Any]
) -> Tuple[List[str], str, str]:
    """
    Parameters to vary: the auditor's design_variables (audited, numeric, with miner
    bounds). Without design_variables, fall back to the first audited parameters.
    Returns (names, source, note).
    """
    audited = auditor_output.audited_parameters_dict
    params_by_name = {p.name: p for p in miner_output.parameters}
    sweepable = [
        name for name in audited.keys()
        if name in params_by_name and _is_finite_number(audited.get(name))
    ]
    design_variables = objective.get("design_variables")
    if design_variables:
        names = [name for name in design_variables if name in sweepable]
        skipped = [name for name in design_variables if name not in sweepable]
        note = f"Swept the auditor's design_variables {names}."
        if skipped:
            note += f" Not sweepable (not audited, not numeric or no bounds): {skipped}."
        return names, "auditor_design_variables", note
    names = sweepable[:FALLBACK_SWEEP_PARAMETER_COUNT]
    note = (
        "The auditor named no design_variables; fell back to the first "
        f"{FALLBACK_SWEEP_PARAMETER_COUNT} audited parameters {names}, which may include operating conditions."
    )
    return names, "fallback_first_parameters", note


def _candidate_record(label: str, params: Dict[str, Any], result: Dict[str, Any], objective: Dict[str, Any], **extra) -> Dict[str, Any]:
    evaluation = evaluate_candidate(result, params, objective)
    record = {
        "label": label,
        **extra,
        "parameters": params,
        "success": result.get("success"),
        "performance_gain_pct": result.get("performance_gain_pct"),
        "relative_error": result.get("relative_error"),
        "primary_metric_value": result.get("primary_metric_value"),
        "reference_metric_value": result.get("reference_metric_value"),
        "status": evaluation["status"],
        "rejection_reasons": evaluation["rejection_reasons"],
        "constraint_violations": evaluation["constraint_violations"],
        "noise_level": evaluation["noise_level"],
    }
    if evaluation["noise_warning"]:
        record["noise_warning"] = evaluation["noise_warning"]
    return record, evaluation


def _dynamic_parameter_sweep(
    miner_output: MinerOutput,
    auditor_output: AuditorOutput,
    script_path: str,
    base_params: Dict[str, float],
    base_result: Dict[str, Any],
    slug: str,
    timestamp: str,
    skip_reason: Optional[str] = None,
) -> Dict[str, Any]:
    objective = _dynamic_objective_contract(auditor_output)
    score_field = objective.get("score_field", "performance_gain_pct")
    direction = objective.get("direction", "maximize")
    threshold = float(objective.get("acceptance_threshold", 0.0) or 0.0)

    baseline, baseline_eval = _candidate_record("baseline", base_params.copy(), base_result, objective)
    candidates = [baseline]
    warnings: List[Dict[str, str]] = []
    if baseline_eval["noise_warning"]:
        warnings.append({"code": "noise_level_metric", "message": f"Baseline: {baseline_eval['noise_warning']}"})
    if baseline_eval["constraint_violations"]:
        warnings.append({
            "code": "hard_constraint_violation",
            "message": "Baseline violates hard constraints: " + "; ".join(baseline_eval["constraint_violations"]),
        })
    relative_error = base_result.get("relative_error")
    if _is_finite_number(relative_error) and float(relative_error) > MAX_SWEEP_RELATIVE_ERROR:
        warnings.append({
            "code": "baseline_error_estimate_high",
            "message": f"Baseline relative_error={float(relative_error):.3g} exceeds {MAX_SWEEP_RELATIVE_ERROR}.",
        })

    swept, source, note = select_sweep_parameters(miner_output, auditor_output, objective)
    if skip_reason:
        swept = []
        note = f"Sweep skipped: {skip_reason}."
    else:
        generated_dir = os.path.dirname(script_path)
        sweep_dir = os.path.join(generated_dir, "sweeps")
        os.makedirs(sweep_dir, exist_ok=True)
        audited = auditor_output.audited_parameters_dict
        params_by_name = {p.name: p for p in miner_output.parameters}
        for name in swept:
            proposal = params_by_name[name]
            current = float(audited[name])
            for value in _candidate_parameter_values(current, float(proposal.min_bound), float(proposal.max_bound)):
                if abs(value - current) <= max(1e-12, abs(current) * 1e-9):
                    continue
                variant_params = base_params.copy()
                variant_params[name] = value
                variant_params.setdefault("simulation_coefficient", auditor_output.simulation_coefficient)
                label = f"{name}={value:.6g}"
                safe_label = re.sub(r"[^a-zA-Z0-9_]+", "_", label)
                output_path = os.path.abspath(os.path.join(sweep_dir, f"results_{slug}_{timestamp}_{safe_label}.json"))
                plot_path = os.path.abspath(os.path.join(sweep_dir, f"plot_{slug}_{timestamp}_{safe_label}.png"))
                try:
                    result = _run_dynamic_script_once(script_path, variant_params, output_path, plot_path)
                    record, _ = _candidate_record(
                        label, variant_params, result, objective,
                        changed_parameter=name, changed_value=value,
                    )
                    record["output_path"] = output_path
                    candidates.append(record)
                except Exception as err:
                    candidates.append({
                        "label": label,
                        "changed_parameter": name,
                        "changed_value": value,
                        "parameters": variant_params,
                        "status": "failed",
                        "error": str(err)[-2000:],
                    })

    valid = [item for item in candidates if item.get("status") == "ok"]
    if direction == "minimize":
        accepted = [item for item in valid if float(item.get(score_field)) <= threshold]
        best = min(accepted, key=lambda item: float(item.get(score_field))) if accepted else None
    else:
        accepted = [item for item in valid if float(item.get(score_field)) >= threshold]
        best = max(accepted, key=lambda item: float(item.get(score_field))) if accepted else None

    recommendation = None
    if best is not None and best is not baseline:
        base_score = baseline.get(score_field)
        improves = baseline["status"] != "ok" or not _is_finite_number(base_score) or (
            float(best[score_field]) < float(base_score) if direction == "minimize" else float(best[score_field]) > float(base_score)
        )
        if improves:
            recommendation = {
                "label": best["label"],
                "changed_parameter": best.get("changed_parameter"),
                "changed_value": best.get("changed_value"),
                "parameters": best["parameters"],
                score_field: best[score_field],
                "baseline_" + score_field: base_score,
                "note": "Not adopted: the reported result is the baseline run with the audited parameters. "
                        "The optimizer may take this variant up in the next round.",
            }

    return {
        "objective": objective,
        "headline_source": "baseline",
        "baseline": baseline,
        "best": best,
        "recommendation": recommendation,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid),
        "accepted_candidate_count": len(accepted),
        "swept_parameters": swept,
        "sweep_parameter_source": source,
        "note": note,
        "skipped": skip_reason,
        "max_relative_error": MAX_SWEEP_RELATIVE_ERROR,
        "hard_constraints_not_evaluated": baseline_eval["constraints_not_evaluated"],
        "warnings": warnings,
    }


def _run_generated_validation(
    miner_output: MinerOutput,
    auditor_output: AuditorOutput,
    res: Dict[str, Any],
) -> Tuple[Optional[bool], Optional[str], Optional[List[Dict[str, Any]]], Optional[str], Optional[str]]:
    """
    Runs the generated validation test script against the generated solver script.
    Returns (validation_passed, validation_report, validation_tests, test_script_path, test_output_path).
    """
    # Run test script validation on the fly
    validation_passed = None
    validation_report = None
    validation_tests = None

    try:
        from ..test_generator import TestScriptGenerator
        test_gen = TestScriptGenerator()
        solver_path = res.get("script_path")
        params_json_path = res.get("params_json_path") or solver_path.replace("solver_", "params_").replace(".py", ".json")

        print(f"[*] Running automated validation on generated script: {solver_path}")
        test_res = test_gen.generate_and_execute_tests(
            miner_output=miner_output,
            auditor_output=auditor_output,
            solver_script_path=solver_path,
            params_json_path=params_json_path
        )

        validation_passed = test_res.get("success", False)
        validation_tests = test_res.get("test_results", [])
        test_script_path = test_res.get("test_script_path")
        test_output_path = test_res.get("test_output_path")

        # Format a simple markdown report of the validation runs
        report_lines = []
        report_lines.append(f"### Status der Testausführung: {'✅ ERFOLGREICH' if validation_passed else '❌ FEHLGESCHLAGEN'}")
        report_lines.append(f"- **Testskript:** `{os.path.basename(test_res.get('test_script_path', ''))}`")
        report_lines.append(f"- **Ergebnisse:** `{os.path.basename(test_res.get('test_output_path', ''))}`")
        report_lines.append(f"- **Auswertung:** {test_res.get('result_source', 'unittest runner (harness)')}")
        report_lines.append(f"- **Ausgeführte Tests:** {test_res.get('total_run', 0)}")
        report_lines.append(f"- **Fehlgeschlagene Tests:** {test_res.get('total_failures', 0)}")
        report_lines.append(f"- **Test-Fehler (Errors):** {test_res.get('total_errors', 0)}\n")

        report_lines.append("| Testfall | Status | Details |")
        report_lines.append("| --- | --- | --- |")
        for t in validation_tests:
            status_emoji = "✅ Passed" if t.get("passed") else "❌ Failed"
            message = t.get("message", "").replace("\n", " ").strip()
            report_lines.append(f"| `{t.get('name')}` | {status_emoji} | {message} |")

        validation_report = "\n".join(report_lines)
        print(f"[+] Automated validation complete. Passed: {validation_passed}")
    except Exception as test_err:
        print(f"[-] Automated validation crashed: {test_err}")
        validation_passed = False
        validation_report = f"### Status der Testausführung: ❌ CRASHED\n\nFehler bei der Testgenerierung/-ausführung: `{str(test_err)}`"
        validation_tests = [{"name": "validation_runner", "passed": False, "message": str(test_err)}]
        test_script_path = None
        test_output_path = None

    return validation_passed, validation_report, validation_tests, test_script_path, test_output_path
