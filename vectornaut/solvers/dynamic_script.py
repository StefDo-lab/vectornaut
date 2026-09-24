# -*- coding: utf-8 -*-
import os
import json
import re
import sys
import subprocess
from typing import Dict, List, Any, Optional, Tuple

from ..config import MinerOutput, AuditorOutput


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
        "hard_constraints": ["relative_error <= 1.0", "parameters within bounds", "finite numeric outputs"],
        "note": "Dynamic-script variants are ranked by performance_gain_pct. The generated script must keep this metric definition stable across variants.",
    }


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
    params_path = output_path.replace("results_", "params_")
    with open(params_path, "w", encoding="utf-8") as pf:
        json.dump(params, pf, indent=4)
    result = subprocess.run(
        [sys.executable, script_path, "--params", params_path, "--output", output_path, "--plot", plot_path],
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
        timeout=int(os.environ.get("VECTORNAUT_SCRIPT_TIMEOUT_SECONDS", "60")),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Dynamic script variant failed.")
    with open(output_path, "r", encoding="utf-8") as rf:
        return json.load(rf)


def _dynamic_parameter_sweep(
    miner_output: MinerOutput,
    auditor_output: AuditorOutput,
    script_path: str,
    base_params: Dict[str, float],
    base_result: Dict[str, Any],
    slug: str,
    timestamp: str,
) -> Dict[str, Any]:
    generated_dir = os.path.dirname(script_path)
    sweep_dir = os.path.join(generated_dir, "sweeps")
    os.makedirs(sweep_dir, exist_ok=True)
    candidates = [{
        "label": "baseline",
        "parameters": base_params.copy(),
        "performance_gain_pct": base_result.get("performance_gain_pct"),
        "relative_error": base_result.get("relative_error"),
        "primary_metric_value": base_result.get("primary_metric_value"),
        "reference_metric_value": base_result.get("reference_metric_value"),
        "status": "ok",
    }]

    audited = auditor_output.audited_parameters_dict.copy()
    params_by_name = {p.name: p for p in miner_output.parameters}
    sweep_params = [
        name for name in audited.keys()
        if name in params_by_name and isinstance(audited.get(name), (int, float))
    ][:4]

    for name in sweep_params:
        proposal = params_by_name[name]
        for value in _candidate_parameter_values(float(audited[name]), float(proposal.min_bound), float(proposal.max_bound)):
            if abs(value - float(audited[name])) <= max(1e-12, abs(float(audited[name])) * 1e-9):
                continue
            variant_params = base_params.copy()
            variant_params[name] = value
            variant_params.setdefault("simulation_coefficient", auditor_output.simulation_coefficient)
            variant_params.setdefault("slippage_coefficient", auditor_output.simulation_coefficient)
            variant_params.setdefault("lambda", auditor_output.simulation_coefficient)
            variant_params.setdefault("slip_length", auditor_output.simulation_coefficient)
            label = f"{name}={value:.6g}"
            safe_label = re.sub(r"[^a-zA-Z0-9_]+", "_", label)
            output_path = os.path.abspath(os.path.join(sweep_dir, f"results_{slug}_{timestamp}_{safe_label}.json"))
            plot_path = os.path.abspath(os.path.join(sweep_dir, f"plot_{slug}_{timestamp}_{safe_label}.png"))
            try:
                result = _run_dynamic_script_once(script_path, variant_params, output_path, plot_path)
                candidates.append({
                    "label": label,
                    "changed_parameter": name,
                    "changed_value": value,
                    "parameters": variant_params,
                    "performance_gain_pct": result.get("performance_gain_pct"),
                    "relative_error": result.get("relative_error"),
                    "primary_metric_value": result.get("primary_metric_value"),
                    "reference_metric_value": result.get("reference_metric_value"),
                    "status": "ok",
                    "output_path": output_path,
                })
            except Exception as err:
                candidates.append({
                    "label": label,
                    "changed_parameter": name,
                    "changed_value": value,
                    "parameters": variant_params,
                    "status": "failed",
                    "error": str(err),
                })

    objective = _dynamic_objective_contract(auditor_output)
    score_field = objective.get("score_field", "performance_gain_pct")
    direction = objective.get("direction", "maximize")
    threshold = float(objective.get("acceptance_threshold", 0.0) or 0.0)
    valid = [
        item for item in candidates
        if item.get("status") == "ok" and isinstance(item.get(score_field), (int, float))
        and (float(item.get("relative_error", 0.0) or 0.0) <= 1.0)
    ]
    if direction == "minimize":
        accepted = [item for item in valid if float(item.get(score_field)) <= threshold]
        best = min(valid, key=lambda item: float(item.get(score_field))) if valid else None
    else:
        accepted = [item for item in valid if float(item.get(score_field)) >= threshold]
        best = max(valid, key=lambda item: float(item.get(score_field))) if valid else None
    return {
        "objective": objective,
        "baseline": candidates[0],
        "best": best,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid),
        "accepted_candidate_count": len(accepted),
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
        # Reconstruct params_json_path from solver_path
        solver_path = res.get("script_path")
        params_json_path = solver_path.replace("solver_", "params_").replace(".py", ".json")

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
