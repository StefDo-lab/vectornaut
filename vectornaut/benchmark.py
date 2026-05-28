# -*- coding: utf-8 -*-
import argparse
import json
import os
from typing import Any, Dict, List, Optional

from vectornaut.evaluation import evaluate_run_output


DEFAULT_BENCHMARK_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "benchmarks", "eval_cases.json")
)


def load_benchmark_cases(path: Optional[str] = None) -> List[Dict[str, Any]]:
    case_path = path or DEFAULT_BENCHMARK_PATH
    with open(case_path, "r", encoding="utf-8") as fh:
        cases = json.load(fh)
    if not isinstance(cases, list):
        raise ValueError(f"Benchmark file must contain a list of cases: {case_path}")
    return cases


def run_benchmark_case(case: Dict[str, Any]) -> Dict[str, Any]:
    result = evaluate_run_output(case.get("run", {}), case.get("criteria", {}))
    expected = case.get("expected", {})
    validator = result.get("validator", {})

    expectation_checks = []
    if "passed" in expected:
        expectation_checks.append({
            "name": "expected_passed",
            "matched": result.get("passed") is expected["passed"],
            "expected": expected["passed"],
            "actual": result.get("passed"),
        })
    if "validator_status" in expected:
        expectation_checks.append({
            "name": "expected_validator_status",
            "matched": validator.get("status") == expected["validator_status"],
            "expected": expected["validator_status"],
            "actual": validator.get("status"),
        })
    if "recommended_action" in expected:
        expectation_checks.append({
            "name": "expected_recommended_action",
            "matched": validator.get("recommended_action") == expected["recommended_action"],
            "expected": expected["recommended_action"],
            "actual": validator.get("recommended_action"),
        })

    return {
        "id": case.get("id"),
        "description": case.get("description", ""),
        "matched_expectations": all(check["matched"] for check in expectation_checks),
        "expectation_checks": expectation_checks,
        "passed": result.get("passed"),
        "validator_status": validator.get("status"),
        "reliability": validator.get("reliability"),
        "recommended_action": validator.get("recommended_action"),
        "score": validator.get("score"),
        "failed_errors": result.get("summary", {}).get("failed_errors", []),
        "warnings": result.get("warnings", []) + validator.get("warnings", []),
    }


def run_benchmark(path: Optional[str] = None) -> Dict[str, Any]:
    cases = load_benchmark_cases(path)
    results = [run_benchmark_case(case) for case in cases]
    matched = sum(1 for result in results if result["matched_expectations"])
    return {
        "total": len(results),
        "matched": matched,
        "failed": len(results) - matched,
        "success": matched == len(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Vectornaut deterministic benchmark cases.")
    parser.add_argument("--cases", default=None, help="Path to benchmark JSON cases.")
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    args = parser.parse_args()

    result = run_benchmark(args.cases)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Vectornaut benchmark: {result['matched']}/{result['total']} matched expectations")
        for item in result["results"]:
            marker = "OK" if item["matched_expectations"] else "FAIL"
            print(
                f"- {marker} {item['id']}: "
                f"eval_passed={item['passed']} validator={item['validator_status']} "
                f"action={item['recommended_action']}"
            )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
