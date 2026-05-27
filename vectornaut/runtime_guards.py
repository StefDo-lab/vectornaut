# -*- coding: utf-8 -*-
import math
import os
from typing import Any, Dict, Iterable, Tuple


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def run_limits(epochs: Any, max_optimization_rounds: Any) -> Tuple[int, int]:
    max_epochs = _env_int("VECTORNAUT_MAX_EPOCHS", 1000)
    max_rounds = _env_int("VECTORNAUT_MAX_OPTIMIZATION_ROUNDS", 10)
    safe_epochs = clamp_int(epochs, default=200, minimum=1, maximum=max_epochs)
    safe_rounds = clamp_int(max_optimization_rounds, default=3, minimum=1, maximum=max_rounds)
    return safe_epochs, safe_rounds


def _finite_values(values: Iterable[Any]) -> bool:
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            return False
    return True


def validate_simulator_output(sim_output: Any) -> None:
    data: Dict[str, Any] = sim_output.model_dump() if hasattr(sim_output, "model_dump") else dict(sim_output)
    sample_points = data.get("sample_points")
    primary = data.get("solution_primary")
    reference = data.get("solution_reference")

    if not isinstance(sample_points, list) or not isinstance(primary, list) or not isinstance(reference, list):
        raise ValueError("Solver output must contain sample_points, solution_primary, and solution_reference lists.")
    if not sample_points:
        raise ValueError("Solver output contains no sample points.")
    if len(primary) != len(reference) or len(primary) != len(sample_points):
        raise ValueError(
            "Solver output arrays must have aligned lengths: "
            f"samples={len(sample_points)}, primary={len(primary)}, reference={len(reference)}."
        )
    if not _finite_values(primary) or not _finite_values(reference):
        raise ValueError("Solver output contains non-finite solution values.")

    metric_names = ["performance_gain_pct", "relative_error", "primary_metric_value", "reference_metric_value"]
    invalid_metrics = [
        name for name in metric_names
        if not isinstance(data.get(name), (int, float)) or not math.isfinite(float(data.get(name)))
    ]
    if invalid_metrics:
        raise ValueError(f"Solver output contains invalid metrics: {invalid_metrics}.")
