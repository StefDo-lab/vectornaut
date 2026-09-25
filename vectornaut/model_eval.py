# -*- coding: utf-8 -*-
"""
Model-comparison harness for Vectornaut.

Runs the full pipeline for several AI model configurations on a fixed set of
reference prompts (benchmarks/ai_reference_cases.json) and scores every run with
objective, deterministic checks (pipeline success, audit outcome, solver routing,
validator result, relative error, dimensionality, parameter plausibility, ...).

    python -m vectornaut.model_eval --mock
    python -m vectornaut.model_eval \
        --config baseline= \
        --config strong_formulator=VECTORNAUT_MODEL_FORMULATOR=<model>,VECTORNAUT_MODEL_AUDITOR=<model> \
        --repeats 2

Each configuration is a set of environment overrides (VECTORNAUT_MODEL*,
VECTORNAUT_THINKING*, ...) that is applied for one pipeline run and restored
afterwards. See docs/OPERATIONS.md, section "Comparing AI models".
"""
import argparse
import ast
import io
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from vectornaut.config import MODEL_STAGES, get_model_name, get_thinking_level, normalize_thinking_level
from vectornaut.evaluation import collect_non_finite_numbers, evaluate_run_output, is_finite_number
from vectornaut.storage import get_data_dir


DEFAULT_CASES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "benchmarks", "ai_reference_cases.json")
)
DEFAULT_EPOCHS = 80
DEFAULT_MAX_ROUNDS = 2
MAX_CONCEPT_ATTEMPTS = 3  # mirrors PipelineRunner.run
PIPELINE_STAGES = tuple(stage for stage in MODEL_STAGES if stage != "chat")

# Keys the harness controls itself and that a --config must not override.
PROTECTED_ENV_KEYS = {"VECTORNAUT_DATA_DIR"}

# Check weights for the per-run score (0..1). Checks that do not apply to a case
# (e.g. no expected dimensionality) are left out of the weighted mean.
CHECK_WEIGHTS: Dict[str, float] = {
    "pipeline_success": 3.0,
    "audit_passed": 2.0,
    "validator_status": 2.0,
    "validator_score": 1.0,
    "solver_method": 1.0,
    "dimensionality": 1.0,
    "domain_keywords": 1.0,
    "relative_error": 1.0,
    "finite_outputs": 1.0,
    "parameter_ranges": 1.0,
    "remines": 1.0,
    "eval_passed": 1.0,
    "dynamic_script_validation": 1.0,
    # outcome == "rejected"
    "rejected": 3.0,
    "audit_rejected": 2.0,
}

# Pipeline failure reasons (failed_concepts[*].reason prefixes) -> category.
REASON_CATEGORIES = (
    ("Audit failed", "audit_rejected"),
    ("Simulation solver error", "solver_error"),
    ("Validator failed", "validator_fail"),
    ("Optimizer rejected", "optimizer_rejected"),
)
# "request_rejected": the pipeline returned status="rejected" (miner, auditor or the
# deterministic physics check declared the request physically infeasible).
PIPELINE_REJECTION_CATEGORIES = {"request_rejected", "audit_rejected", "solver_error", "validator_fail", "optimizer_rejected"}
# Rejections that reflect a judgement by an AI stage (full credit for infeasible cases).
JUDGEMENT_REJECTION_CATEGORIES = {"request_rejected", "audit_rejected", "optimizer_rejected"}
EXHAUSTED_PREFIX = "All bionic concepts failed validation"
_API_ERROR_PATTERN = re.compile(
    r"api[ _]?key|quota|resource[_ ]exhausted|rate limit|\b429\b|\b401\b|\b403\b|\b500\b|\b503\b|"
    r"unavailable|deadline|permission[_ ]denied|unauthenticated",
    re.IGNORECASE,
)


class ConfigSpecError(ValueError):
    """Raised for malformed --config specifications."""


# ==========================================
# Configurations and environment handling
# ==========================================

@dataclass
class ModelConfig:
    name: str
    # env var -> value; None means "unset this variable for the run".
    overrides: Dict[str, Optional[str]] = field(default_factory=dict)

    def describe(self) -> str:
        if not self.overrides:
            return "(ambient environment)"
        return ", ".join(f"{key}={'<unset>' if value is None else value}" for key, value in self.overrides.items())


_CONFIG_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENV_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _validate_override(key: str, value: str) -> None:
    if not _ENV_KEY_PATTERN.match(key):
        raise ConfigSpecError(f"Invalid environment variable name '{key}'.")
    if not key.startswith("VECTORNAUT_"):
        raise ConfigSpecError(f"Only VECTORNAUT_* variables can be overridden per config (got '{key}').")
    if key in PROTECTED_ENV_KEYS:
        raise ConfigSpecError(f"'{key}' is managed by the harness and cannot be overridden per config.")
    for prefix in ("VECTORNAUT_MODEL_", "VECTORNAUT_THINKING_"):
        if key.startswith(prefix):
            stage = key[len(prefix):].lower()
            if stage not in MODEL_STAGES:
                raise ConfigSpecError(
                    f"Unknown stage in '{key}'. Expected one of: {', '.join(s.upper() for s in MODEL_STAGES)}."
                )
            if prefix == "VECTORNAUT_THINKING_" and value and normalize_thinking_level(value) is None:
                raise ConfigSpecError(f"Invalid thinking level '{value}' for {key} (allowed: minimal, low, medium, high).")


def parse_config_spec(spec: str) -> ModelConfig:
    """
    Parses 'NAME=KEY=VALUE,KEY2=VALUE2'. 'NAME=' is a config without overrides
    (ambient environment). 'KEY=' (empty value) unsets KEY for that config.
    """
    name, sep, assignments = (spec or "").partition("=")
    name = name.strip()
    if not sep:
        raise ConfigSpecError(f"Config '{spec}' must look like NAME=ENV_ASSIGNMENTS (use 'NAME=' for no overrides).")
    if not _CONFIG_NAME_PATTERN.match(name):
        raise ConfigSpecError(f"Invalid config name '{name}' (letters, digits, '_', '-', '.').")

    overrides: Dict[str, Optional[str]] = {}
    for part in assignments.split(","):
        part = part.strip()
        if not part:
            continue
        key, eq, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not eq:
            raise ConfigSpecError(f"Assignment '{part}' in config '{name}' must look like KEY=VALUE.")
        _validate_override(key, value)
        if key in overrides:
            raise ConfigSpecError(f"'{key}' is assigned twice in config '{name}'.")
        overrides[key] = value or None
    return ModelConfig(name=name, overrides=overrides)


def parse_configs(specs: Optional[Sequence[str]]) -> List[ModelConfig]:
    configs = [parse_config_spec(spec) for spec in (specs or [])] or [ModelConfig(name="baseline")]
    names = [config.name for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigSpecError(f"Duplicate config names: {', '.join(duplicates)}.")
    return configs


@contextmanager
def env_overrides(overrides: Mapping[str, Optional[str]]) -> Iterator[None]:
    """Applies env overrides (None = unset) and restores the previous state afterwards, also on errors."""
    saved = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def resolved_model_settings() -> Dict[str, Dict[str, Optional[str]]]:
    """Model and thinking override per pipeline stage in the *current* environment."""
    return {
        "models": {stage: get_model_name(stage) for stage in PIPELINE_STAGES},
        # None = the stage's built-in default (see docs/OPERATIONS.md).
        "thinking": {stage: get_thinking_level(stage) for stage in PIPELINE_STAGES},
    }


def resolve_config_settings(config: ModelConfig) -> Dict[str, Dict[str, Optional[str]]]:
    with env_overrides(config.overrides):
        return resolved_model_settings()


# ==========================================
# Reference cases
# ==========================================

def load_reference_cases(path: Optional[str] = None) -> List[Dict[str, Any]]:
    case_path = path or DEFAULT_CASES_PATH
    with open(case_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    defaults = data.get("defaults", {}) if isinstance(data, dict) else {}
    raw_cases = data.get("cases", []) if isinstance(data, dict) else data
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"Reference case file contains no cases: {case_path}")

    cases = []
    seen = set()
    for raw in raw_cases:
        case_id = raw.get("id")
        if not case_id or not raw.get("query"):
            raise ValueError(f"Every reference case needs an 'id' and a 'query': {raw!r}")
        if case_id in seen:
            raise ValueError(f"Duplicate reference case id: {case_id}")
        seen.add(case_id)
        case = dict(raw)
        case["expect"] = {**defaults, **(raw.get("expect") or {})}
        cases.append(case)
    return cases


def select_cases(cases: List[Dict[str, Any]], case_filter: Optional[str]) -> List[Dict[str, Any]]:
    if not case_filter:
        return cases
    wanted = [item.strip() for item in case_filter.split(",") if item.strip()]
    by_id = {case["id"]: case for case in cases}
    unknown = [item for item in wanted if item not in by_id]
    if unknown:
        raise ValueError(f"Unknown case id(s): {', '.join(unknown)}. Available: {', '.join(by_id)}")
    return [by_id[item] for item in wanted]


# ==========================================
# Fact extraction and scoring (pure functions)
# ==========================================

def classify_reason(reason: str) -> str:
    text = str(reason or "")
    for prefix, category in REASON_CATEGORIES:
        if text.startswith(prefix):
            return category
    return "other"


def parse_failed_history(error_message: str) -> List[Dict[str, Any]]:
    """Recovers failed_concepts from the pipeline's 'All bionic concepts failed' error message."""
    marker = "Failed history:"
    if marker not in (error_message or ""):
        return []
    try:
        parsed = ast.literal_eval(error_message.split(marker, 1)[1].strip())
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def classify_error(error_type: Optional[str], error_message: Optional[str], failed_concepts: List[Dict[str, Any]]) -> str:
    message = error_message or ""
    if message.startswith(EXHAUSTED_PREFIX):
        if failed_concepts:
            return classify_reason(failed_concepts[-1].get("reason", ""))
        reasons = [category for prefix, category in REASON_CATEGORIES if prefix in message]
        return reasons[-1] if reasons else "concepts_exhausted"
    if _API_ERROR_PATTERN.search(message) or (error_type or "").lower().startswith(("clienterror", "servererror", "apierror")):
        return "api_error"
    return f"exception:{error_type or 'Unknown'}"


def _dig(data: Any, *path: str) -> Any:
    current = data
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def extract_run_facts(
    result: Optional[Dict[str, Any]],
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    log_text: str = "",
    error_failed_concepts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Reduces a pipeline result (or failure) to the deterministic facts that are scored."""
    result = result if isinstance(result, dict) else None
    success = bool(result and result.get("success"))
    rejected = bool(result and result.get("status") == "rejected")
    if result is not None:
        failed_concepts = list(result.get("failed_concepts") or [])
    elif error_failed_concepts is not None:
        # ConceptsExhaustedError carries the structured history.
        failed_concepts = list(error_failed_concepts)
    else:
        failed_concepts = parse_failed_history(error_message or "")

    logged_attempts = log_text.count("Starting Concept Attempt")
    if success:
        concept_attempts: Optional[int] = len(failed_concepts) + 1
    elif logged_attempts:
        concept_attempts = logged_attempts
    elif failed_concepts:
        concept_attempts = len(failed_concepts)
    else:
        concept_attempts = None

    remine_reasons = [classify_reason(item.get("reason", "")) for item in failed_concepts]
    simulator = _dig(result, "simulator") or {}
    auditor = _dig(result, "auditor") or {}
    miner = _dig(result, "miner") or {}
    validation = _dig(result, "validation") or {}
    history = (result or {}).get("optimization_history") or []

    solver_method = str(simulator.get("solver_method") or "").lower() or None
    script_iterations = log_text.count("Generating script iteration")
    params = auditor.get("audited_parameters_dict")
    if not isinstance(params, dict):
        params = {p.get("name"): p.get("value") for p in auditor.get("audited_parameters") or [] if isinstance(p, dict)}

    independent = miner.get("independent_variables")
    domain_text = " ".join(
        str(part or "") for part in (
            miner.get("domain"),
            miner.get("design_name"),
            miner.get("physical_mechanism"),
            miner.get("governing_equation"),
            _dig(auditor, "ui_metadata", "domain_name"),
        )
    ).lower()

    evaluation = None
    if result is not None:
        try:
            evaluation = evaluate_run_output(result, {"max_relative_error": 1.0})
        except Exception as eval_err:  # scoring must never crash the harness
            evaluation = {"passed": False, "error": str(eval_err)}

    if success:
        category = "ok"
    elif rejected:
        category = "request_rejected"
    elif result is not None:
        category = "unsuccessful_result"
    else:
        category = classify_error(error_type, error_message, failed_concepts)

    return {
        "success": success,
        "category": category,
        "is_mock": (result or {}).get("is_mock"),
        "models": (result or {}).get("models") or {},
        "error_type": error_type,
        "error": (error_message or "")[:2000] or None,
        "concept_attempts": concept_attempts,
        "remines": None if concept_attempts is None else max(0, concept_attempts - 1),
        "remine_reasons": remine_reasons,
        "audit_failures": remine_reasons.count("audit_rejected"),
        "rejection_stage": ((result or {}).get("rejection") or {}).get("stage") if rejected else None,
        "audit_passed": auditor.get("audit_passed") if result is not None else None,
        "auditor_solver_method": str(auditor.get("solver_method") or "").lower() or None,
        "solver_method": solver_method,
        "solver_fallbacks": sum(len(round_data.get("solver_fallbacks") or []) for round_data in history if isinstance(round_data, dict)),
        "optimization_rounds": len(history),
        "validator_status": validation.get("status"),
        "validator_score": validation.get("score"),
        "validator_action": validation.get("recommended_action"),
        "validator_reliability": validation.get("reliability"),
        "relative_error": simulator.get("relative_error"),
        "performance_gain_pct": simulator.get("performance_gain_pct"),
        "non_finite_paths": collect_non_finite_numbers(simulator)[:10] if simulator else [],
        "eval_passed": evaluation.get("passed") if evaluation else None,
        "eval_failed_checks": [c.get("name") for c in (evaluation or {}).get("summary", {}).get("failed_errors", [])],
        "dimensionality": len(independent) if isinstance(independent, list) else None,
        "independent_variables": independent,
        "domain": miner.get("domain"),
        "design_name": miner.get("design_name"),
        "governing_equation": miner.get("governing_equation"),
        "domain_text": domain_text,
        "parameters": params,
        "script_iterations": script_iterations if (script_iterations or solver_method == "dynamic_script") else None,
        "dynamic_validation_passed": simulator.get("validation_passed") if solver_method == "dynamic_script" else None,
    }


def check_parameter_ranges(ranges: Sequence[Dict[str, Any]], params: Optional[Dict[str, Any]]) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    """Returns (fraction of matched parameters inside their range, details). None if nothing matched."""
    params = params or {}
    details: List[Dict[str, Any]] = []
    total = 0
    passed = 0
    for spec in ranges or []:
        pattern = re.compile(spec.get("match", "$^"), re.IGNORECASE)
        low = float(spec.get("min", -math.inf))
        high = float(spec.get("max", math.inf))
        matched = {name: value for name, value in params.items() if name and pattern.search(str(name))}
        if not matched:
            details.append({"match": spec.get("match"), "status": "n/a"})
            continue
        for name, value in matched.items():
            ok = is_finite_number(value) and low <= float(value) <= high
            total += 1
            passed += int(ok)
            details.append({"match": spec.get("match"), "parameter": name, "value": value, "min": low, "max": high, "ok": ok})
    return (round(passed / total, 3) if total else None), details


def _check(name: str, score: Optional[float], detail: str) -> Dict[str, Any]:
    return {
        "name": name,
        "score": None if score is None else round(max(0.0, min(1.0, float(score))), 3),
        "weight": CHECK_WEIGHTS.get(name, 1.0),
        "detail": detail,
    }


def score_run(expect: Dict[str, Any], facts: Dict[str, Any]) -> Dict[str, Any]:
    """Scores one run against a case's expectations. Returns score (0..1), outcome_met and checks."""
    checks: List[Dict[str, Any]] = []
    outcome = expect.get("outcome", "success")

    if outcome == "rejected":
        if not facts["success"] and facts["category"] in JUDGEMENT_REJECTION_CATEGORIES:
            rejected, detail = 1.0, f"Pipeline rejected the request ({facts['category']})."
        elif not facts["success"] and facts["category"] in PIPELINE_REJECTION_CATEGORIES:
            rejected, detail = 0.5, f"Pipeline gave up for a numerical reason ({facts['category']}), not an explicit rejection."
        elif facts["success"] and facts["audit_failures"]:
            rejected, detail = 0.5, "A concept was audit-rejected, but a re-mined concept was accepted."
        elif facts["success"]:
            rejected, detail = 0.0, "Pipeline accepted the infeasible request."
        else:
            rejected, detail = 0.0, f"Run failed for an unrelated reason ({facts['category']})."
        checks.append(_check("rejected", rejected, detail))
        checks.append(_check(
            "audit_rejected",
            1.0 if facts["audit_failures"] or facts["category"] == "request_rejected" else 0.0,
            f"audit rejections={facts['audit_failures']}, rejection stage={facts.get('rejection_stage')!r}",
        ))
        outcome_met = not facts["success"] and facts["category"] in PIPELINE_REJECTION_CATEGORIES
    else:
        success = facts["success"]
        checks.append(_check("pipeline_success", 1.0 if success else 0.0, f"category={facts['category']}"))

        expected_audit = expect.get("audit_passed", True)
        checks.append(_check(
            "audit_passed",
            1.0 if facts["audit_passed"] is expected_audit else 0.0,
            f"audit_passed={facts['audit_passed']!r}, expected {expected_audit!r}",
        ))

        status = facts["validator_status"]
        checks.append(_check(
            "validator_status",
            {"pass": 1.0, "warn": 0.5}.get(status, 0.0),
            f"status={status!r}, action={facts['validator_action']!r}",
        ))
        v_score = facts["validator_score"]
        checks.append(_check("validator_score", float(v_score) if is_finite_number(v_score) else 0.0, f"score={v_score!r}"))

        allowed_solvers = expect.get("solver_methods")
        if allowed_solvers:
            checks.append(_check(
                "solver_method",
                1.0 if facts["solver_method"] in allowed_solvers else 0.0,
                f"solver={facts['solver_method']!r} (auditor={facts['auditor_solver_method']!r}), expected one of {allowed_solvers}",
            ))

        expected_dim = expect.get("dimensionality")
        if expected_dim:
            checks.append(_check(
                "dimensionality",
                1.0 if facts["dimensionality"] == expected_dim else 0.0,
                f"{facts['dimensionality']!r}D {facts['independent_variables']!r}, expected {expected_dim}D",
            ))

        keywords = [str(k).lower() for k in expect.get("domain_keywords") or []]
        if keywords:
            hits = [k for k in keywords if k in facts["domain_text"]]
            checks.append(_check(
                "domain_keywords",
                1.0 if hits else 0.0,
                f"domain={facts['domain']!r}, matched={hits[:4]}",
            ))

        max_rel = float(expect.get("max_relative_error", 0.1))
        rel = facts["relative_error"]
        if is_finite_number(rel):
            rel_score = 1.0 if float(rel) <= max_rel else 0.5 if float(rel) <= 1.0 else 0.0
        else:
            rel_score = 0.0
        checks.append(_check("relative_error", rel_score, f"relative_error={rel!r}, target<={max_rel} (0.5 credit up to 1.0)"))

        finite = success and not facts["non_finite_paths"]
        checks.append(_check("finite_outputs", 1.0 if finite else 0.0, f"non-finite={facts['non_finite_paths'][:3]}"))

        range_score, range_details = check_parameter_ranges(expect.get("parameter_ranges") or [], facts["parameters"] if success else {})
        if range_score is not None:
            bad = [f"{d['parameter']}={d['value']!r}" for d in range_details if d.get("ok") is False]
            checks.append(_check("parameter_ranges", range_score, f"out of range: {bad}" if bad else "all matched parameters in range"))

        max_remines = int(expect.get("max_remines", 1))
        remines = facts["remines"]
        if remines is None:
            remine_score = 0.0
        else:
            remine_score = 1.0 if remines <= max_remines else max(0.0, 1.0 - 0.5 * (remines - max_remines))
        checks.append(_check("remines", remine_score if success else 0.0, f"remines={remines!r} ({facts['remine_reasons']}), max {max_remines}"))

        checks.append(_check(
            "eval_passed",
            1.0 if facts["eval_passed"] else 0.0,
            f"evaluate_run_output failed errors: {facts['eval_failed_checks']}" if facts["eval_failed_checks"] else f"passed={facts['eval_passed']!r}",
        ))

        if facts["solver_method"] == "dynamic_script":
            passed = facts["dynamic_validation_passed"]
            checks.append(_check(
                "dynamic_script_validation",
                1.0 if passed is True else 0.0,
                f"generated validation={passed!r}, script iterations={facts['script_iterations']!r}",
            ))
        outcome_met = success

    applicable = [c for c in checks if c["score"] is not None]
    total_weight = sum(c["weight"] for c in applicable)
    score = sum(c["score"] * c["weight"] for c in applicable) / total_weight if total_weight else 0.0
    return {
        "score": round(score, 3),
        "outcome_met": bool(outcome_met),
        "checks": checks,
        "failed_checks": [c["name"] for c in applicable if c["score"] < 1.0],
    }


# ==========================================
# Running
# ==========================================

class _Tee(io.TextIOBase):
    def __init__(self, buffer: io.StringIO, echo: Optional[Any] = None):
        self._buffer = buffer
        self._echo = echo

    def write(self, text: str) -> int:
        self._buffer.write(text)
        if self._echo is not None:
            self._echo.write(text)
        return len(text)

    def flush(self) -> None:
        if self._echo is not None:
            self._echo.flush()

    @property
    def encoding(self) -> str:  # some libraries inspect sys.stdout.encoding
        return "utf-8"


def _default_run_fn(request: Any) -> Dict[str, Any]:
    from vectornaut.pipeline import PipelineRunner
    return PipelineRunner().run(request)


def run_case(
    case: Dict[str, Any],
    config: ModelConfig,
    repeat: int,
    *,
    mock: bool,
    epochs: int,
    max_rounds: int,
    run_dir: str,
    run_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    echo: Optional[Any] = None,
) -> Dict[str, Any]:
    """Runs one pipeline (config x case x repeat) in an isolated data dir and scores it."""
    from vectornaut.pipeline import PipelineRunRequest

    run_fn = run_fn or _default_run_fn
    os.makedirs(run_dir, exist_ok=True)
    request = PipelineRunRequest(
        query=case["query"],
        epochs=epochs,
        is_mock=mock,
        max_optimization_rounds=max_rounds,
    )

    buffer = io.StringIO()
    result: Optional[Dict[str, Any]] = None
    error_type = error_message = error_trace = None
    error_failed_concepts = None
    # Each run gets its own data dir so that cached PINN models, generated scripts
    # and history from one config cannot leak into another.
    with env_overrides({**config.overrides, "VECTORNAUT_DATA_DIR": run_dir}):
        settings = resolved_model_settings()
        started_at = datetime.now().isoformat(timespec="seconds")
        start = time.perf_counter()
        try:
            with redirect_stdout(_Tee(buffer, echo)):
                result = run_fn(request)
        except Exception as exc:  # the harness records failures instead of aborting
            error_type = type(exc).__name__
            error_message = str(exc)
            error_failed_concepts = getattr(exc, "failed_concepts", None)
            error_trace = traceback.format_exc()
        duration_s = round(time.perf_counter() - start, 3)

    log_text = buffer.getvalue()
    with open(os.path.join(run_dir, "pipeline.log"), "w", encoding="utf-8") as fh:
        fh.write(log_text)
        if error_trace:
            fh.write("\n\n--- harness caught exception ---\n" + error_trace)
    if result is not None:
        with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False, default=str)

    facts = extract_run_facts(result, error_type, error_message, log_text, error_failed_concepts=error_failed_concepts)
    if not mock and facts["is_mock"]:
        # The pipeline silently falls back to mock mode without an API key.
        facts["category"] = "mock_fallback"
        facts["success"] = False
    scored = score_run(case["expect"], facts)
    if not scored["outcome_met"] and facts["category"] == "ok":
        facts["category"] = "accepted_infeasible"
    facts.pop("domain_text", None)

    return {
        "config": config.name,
        "case": case["id"],
        "repeat": repeat,
        "started_at": started_at,
        "duration_s": duration_s,
        "resolved_models": settings["models"],
        "resolved_thinking": settings["thinking"],
        "models": facts["models"],
        "score": scored["score"],
        "outcome_met": scored["outcome_met"],
        "failed_checks": scored["failed_checks"],
        "checks": scored["checks"],
        "facts": facts,
        "run_dir": os.path.abspath(run_dir),
    }


def estimate_model_calls(n_runs: int, max_rounds: int) -> Tuple[int, int]:
    """Rough (typical, worst-case) number of Gemini calls for n_runs live pipeline runs."""
    rounds = max(1, max_rounds)
    typical = 2 + rounds + (rounds - 1) + 1  # miner+formulator, auditor/round, optimizer, synthesizer
    # worst case: every concept attempt re-mined, dynamic_script with 3 self-corrections + test generator per round
    per_concept_worst = 2 + rounds + (rounds - 1) + 4 * rounds
    worst = MAX_CONCEPT_ATTEMPTS * per_concept_worst + 1
    return n_runs * typical, n_runs * worst


# ==========================================
# Aggregation and reporting
# ==========================================

def _mean(values: Sequence[float]) -> Optional[float]:
    values = [float(v) for v in values if is_finite_number(v)]
    return round(sum(values) / len(values), 3) if values else None


def _std(values: Sequence[float]) -> float:
    values = [float(v) for v in values if is_finite_number(v)]
    return round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0


def summarize(runs: List[Dict[str, Any]], configs: List[ModelConfig], cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_config: Dict[str, Any] = {}
    per_case: Dict[str, Dict[str, Any]] = {case["id"]: {} for case in cases}
    success_case_ids = {case["id"] for case in cases if case["expect"].get("outcome", "success") != "rejected"}
    baseline_score = None

    for config in configs:
        config_runs = [run for run in runs if run["config"] == config.name]
        case_stds = []
        for case in cases:
            case_runs = [run for run in config_runs if run["case"] == case["id"]]
            if not case_runs:
                continue
            scores = [run["score"] for run in case_runs]
            if len(scores) > 1:
                case_stds.append(_std(scores))
            per_case[case["id"]][config.name] = {
                "n": len(case_runs),
                "mean_score": _mean(scores),
                "std": _std(scores),
                "outcome_rate": _mean([1.0 if run["outcome_met"] else 0.0 for run in case_runs]),
                "mean_runtime_s": _mean([run["duration_s"] for run in case_runs]),
                "solver_methods": sorted({str(run["facts"]["solver_method"]) for run in case_runs}),
                "categories": sorted({run["facts"]["category"] for run in case_runs}),
            }

        failures: Dict[str, int] = {}
        remine_reasons: Dict[str, int] = {}
        check_values: Dict[str, List[float]] = {}
        models_used: Dict[str, List[str]] = {}
        for run in config_runs:
            if not run["outcome_met"]:
                category = run["facts"]["category"]
                failures[category] = failures.get(category, 0) + 1
            for reason in run["facts"]["remine_reasons"]:
                remine_reasons[reason] = remine_reasons.get(reason, 0) + 1
            for check in run["checks"]:
                if check["score"] is not None:
                    check_values.setdefault(check["name"], []).append(check["score"])
            for stage, model in (run["models"] or {}).items():
                models_used.setdefault(stage, [])
                if model not in models_used[stage]:
                    models_used[stage].append(model)

        success_runs = [run for run in config_runs if run["case"] in success_case_ids]
        mean_score = _mean([run["score"] for run in config_runs])
        if baseline_score is None:
            baseline_score = mean_score
        by_config[config.name] = {
            "overrides": config.overrides,
            "runs": len(config_runs),
            "mean_score": mean_score,
            "delta_vs_first": None if mean_score is None or baseline_score is None else round(mean_score - baseline_score, 3),
            "mean_repeat_std": _mean(case_stds) if case_stds else None,
            "outcome_rate": _mean([1.0 if run["outcome_met"] else 0.0 for run in config_runs]),
            "pipeline_success_rate": _mean([1.0 if run["facts"]["success"] else 0.0 for run in success_runs]),
            "mean_validator_score": _mean([run["facts"]["validator_score"] for run in config_runs]),
            "mean_concept_attempts": _mean([run["facts"]["concept_attempts"] for run in config_runs]),
            "mean_runtime_s": _mean([run["duration_s"] for run in config_runs]),
            "total_runtime_s": round(sum(run["duration_s"] for run in config_runs), 1),
            "failures": failures,
            "remine_reasons": remine_reasons,
            "check_means": {name: _mean(values) for name, values in check_values.items()},
            "models_used": models_used,
        }
    return {"configs": by_config, "cases": per_case}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_signed(value: Optional[float]) -> str:
    return "–" if value is None else f"{value:+.3f}"


def _md_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _compact_models(settings: Dict[str, Dict[str, Optional[str]]]) -> str:
    models = settings.get("models", {})
    thinking = settings.get("thinking", {})
    distinct = sorted(set(models.values()))
    if len(distinct) == 1:
        text = f"all: {distinct[0]}"
    else:
        default = max(distinct, key=lambda model: list(models.values()).count(model))
        others = ", ".join(f"{stage}: {model}" for stage, model in models.items() if model != default)
        text = f"{default}; {others}"
    overrides = ", ".join(f"{stage}={level}" for stage, level in thinking.items() if level)
    return f"{text}; thinking {overrides}" if overrides else text


def build_markdown_report(report: Dict[str, Any]) -> str:
    meta = report["meta"]
    summary = report["summary"]
    configs = [config["name"] for config in meta["configs"]]
    lines: List[str] = []
    lines.append("# Vectornaut model comparison")
    lines.append("")
    lines.append(
        f"- Started: {meta['started_at']} · Mode: **{'MOCK (no AI calls)' if meta['mock'] else 'LIVE'}** · "
        f"Runs: {meta['completed_runs']}/{meta['planned_runs']} · Cases: {len(meta['cases'])} · "
        f"Repeats: {meta['repeats']} · Epochs: {meta['epochs']} · Optimization rounds: {meta['max_rounds']}"
    )
    lines.append(f"- Case file: `{meta['cases_file']}`")
    if meta.get("interrupted"):
        lines.append("- **Interrupted** before all planned runs completed.")
    if meta["mock"]:
        lines.append("")
        lines.append(
            "> Mock mode does not call any AI model: every stage returns the canned mock concept "
            "(riblet foil or Collembola plastron ski base, 1D, PINN solver, audit always passes). "
            "Scores therefore only exercise the harness, the deterministic solvers and the validator; "
            "they say nothing about model quality and are identical across configs by construction."
        )
    lines.append("")

    lines.append("## Configurations")
    lines.append("")
    lines.append("| Config | Overrides | Resolved models |")
    lines.append("|---|---|---|")
    for config in meta["configs"]:
        lines.append(f"| {config['name']} | {_md_cell(config['description'])} | {_md_cell(_compact_models(config['resolved']))} |")
    lines.append("")

    lines.append("## Summary per config")
    lines.append("")
    lines.append("| Config | Mean score | Δ vs first | Outcome met | Pipeline success | Validator score | Concept attempts | Repeat σ | Mean runtime (s) | Failures |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in configs:
        item = summary["configs"].get(name, {})
        failures = ", ".join(f"{cat}: {count}" for cat, count in sorted(item.get("failures", {}).items())) or "none"
        lines.append(
            f"| {name} | {_fmt(item.get('mean_score'))} | {_fmt_signed(item.get('delta_vs_first'))} | "
            f"{_fmt(item.get('outcome_rate'), 2)} | {_fmt(item.get('pipeline_success_rate'), 2)} | "
            f"{_fmt(item.get('mean_validator_score'))} | {_fmt(item.get('mean_concept_attempts'), 2)} | "
            f"{_fmt(item.get('mean_repeat_std'))} | {_fmt(item.get('mean_runtime_s'), 1)} | {_md_cell(failures)} |"
        )
    lines.append("")

    lines.append("## Per case (mean score)")
    lines.append("")
    lines.append("| Case | Expected | " + " | ".join(configs) + " |")
    lines.append("|---|---|" + "---|" * len(configs))
    case_by_id = {case["id"]: case for case in meta["cases"]}
    for case_id, per_config in summary["cases"].items():
        expect = case_by_id.get(case_id, {}).get("expect", {})
        if expect.get("outcome") == "rejected":
            expected = "rejected"
        else:
            dim = f"{expect.get('dimensionality')}D" if expect.get("dimensionality") else "any-D"
            expected = f"{dim}, {'/'.join(expect.get('solver_methods') or ['any'])}"
        cells = []
        for name in configs:
            item = per_config.get(name)
            if not item:
                cells.append("–")
                continue
            text = _fmt(item["mean_score"])
            if item["n"] > 1:
                text += f" ±{item['std']:.2f}"
            text += f" ({'/'.join(item['solver_methods'])})"
            if item["outcome_rate"] is not None and item["outcome_rate"] < 1.0:
                text += f" [{'/'.join(item['categories'])}]"
            cells.append(_md_cell(text))
        lines.append(f"| {case_id} | {_md_cell(expected)} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Check breakdown (mean over applicable runs)")
    lines.append("")
    lines.append("| Check | Weight | " + " | ".join(configs) + " |")
    lines.append("|---|---|" + "---|" * len(configs))
    for check_name, weight in CHECK_WEIGHTS.items():
        values = [summary["configs"].get(name, {}).get("check_means", {}).get(check_name) for name in configs]
        if all(value is None for value in values):
            continue
        lines.append(f"| {check_name} | {weight:g} | " + " | ".join(_fmt(value) for value in values) + " |")
    lines.append("")

    notable = [run for run in report["runs"] if not run["outcome_met"] or run["score"] < 0.6]
    lines.append("## Notable failures")
    lines.append("")
    if not notable:
        lines.append("None: every run met its expected outcome with a score of at least 0.6.")
    for run in notable:
        facts = run["facts"]
        failed = [c for c in run["checks"] if c["score"] is not None and c["score"] < 1.0]
        lines.append(
            f"- **{run['config']} / {run['case']} / r{run['repeat']}**: score {run['score']:.3f}, "
            f"category `{facts['category']}`, solver `{facts['solver_method']}`, "
            f"design {_md_cell(facts.get('design_name') or '–')}"
        )
        if facts.get("error"):
            lines.append(f"  - error: {_md_cell(facts['error'][:300])}")
        for check in failed[:6]:
            lines.append(f"  - {check['name']} = {check['score']:.2f}: {_md_cell(check['detail'])}")
    lines.append("")

    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "Each run scores 0–1 as the weighted mean of the applicable checks (weights above). "
        "`Outcome met` means the pipeline produced a result (or, for `rejected` cases, refused the request). "
        "Differences between configs smaller than the repeat σ are noise; use `--repeats 3` or more "
        "before drawing conclusions. Full pipeline output and logs of every run are under `runs/`."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(report: Dict[str, Any], out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "model_eval.json")
    md_path = os.path.join(out_dir, "model_eval.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(build_markdown_report(report))
    return json_path, md_path


# ==========================================
# Orchestration / CLI
# ==========================================

def run_model_eval(
    configs: List[ModelConfig],
    cases: List[Dict[str, Any]],
    *,
    repeats: int = 1,
    epochs: int = DEFAULT_EPOCHS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    mock: bool = False,
    out_dir: str,
    cases_file: str = DEFAULT_CASES_PATH,
    run_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    verbose: bool = False,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    progress = progress if progress is not None else sys.stdout
    planned = len(configs) * len(cases) * repeats
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "runs.jsonl")

    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "mock": mock,
        "repeats": repeats,
        "epochs": epochs,
        "max_rounds": max_rounds,
        "cases_file": cases_file,
        "planned_runs": planned,
        "completed_runs": 0,
        "interrupted": False,
        "configs": [
            {"name": c.name, "overrides": c.overrides, "description": c.describe(), "resolved": resolve_config_settings(c)}
            for c in configs
        ],
        "cases": [{"id": c["id"], "title": c.get("title"), "query": c["query"], "expect": c["expect"]} for c in cases],
    }

    try:
        # Warm up heavy imports (torch, sympy, scipy) so the first run's wall time is comparable.
        import vectornaut.pipeline  # noqa: F401
        import vectornaut.solver_dispatcher  # noqa: F401
    except ImportError:
        pass

    runs: List[Dict[str, Any]] = []
    index = 0
    try:
        # Configs are interleaved per case so that drifting API conditions
        # (rate limits, latency) affect every config alike.
        for repeat in range(1, repeats + 1):
            for case in cases:
                for config in configs:
                    index += 1
                    run_dir = os.path.join(out_dir, "runs", config.name, case["id"], f"r{repeat}")
                    progress.write(f"[{index}/{planned}] {config.name} | {case['id']} | r{repeat} ... ")
                    progress.flush()
                    record = run_case(
                        case, config, repeat,
                        mock=mock, epochs=epochs, max_rounds=max_rounds, run_dir=run_dir,
                        run_fn=run_fn, echo=progress if verbose else None,
                    )
                    runs.append(record)
                    with open(jsonl_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    progress.write(
                        f"score {record['score']:.3f} ({record['facts']['category']}, "
                        f"{record['facts']['solver_method']}, {record['duration_s']:.1f}s)\n"
                    )
                    progress.flush()
    except KeyboardInterrupt:
        meta["interrupted"] = True
        progress.write("\n[!] Interrupted - writing partial report.\n")

    meta["completed_runs"] = len(runs)
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    report = {"meta": meta, "summary": summarize(runs, configs, cases), "runs": runs}
    json_path, md_path = write_report(report, out_dir)
    report["paths"] = {"json": json_path, "markdown": md_path, "runs_jsonl": jsonl_path}
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vectornaut.model_eval",
        description="Compare AI model configurations on the Vectornaut reference prompts with deterministic scores.",
    )
    parser.add_argument(
        "--config", action="append", default=[], metavar="NAME=ENV_ASSIGNMENTS",
        help="Model configuration, repeatable. E.g. 'baseline=' or "
             "'strong=VECTORNAUT_MODEL_FORMULATOR=<model>,VECTORNAUT_THINKING_AUDITOR=high'. "
             "'KEY=' unsets KEY. Default: a single 'baseline=' config.",
    )
    parser.add_argument("--cases", default=None, help="Comma-separated case ids (default: all).")
    parser.add_argument("--cases-file", default=DEFAULT_CASES_PATH, help="Reference case JSON file.")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per config and case (LLM output varies). Default 1.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"PINN epochs per solve. Default {DEFAULT_EPOCHS}.")
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS, help=f"Optimization rounds per concept. Default {DEFAULT_MAX_ROUNDS}.")
    parser.add_argument("--mock", action="store_true", help="Run the pipeline in mock mode (no API calls, offline).")
    parser.add_argument("--out", default=None, help="Output directory. Default: $VECTORNAUT_DATA_DIR/model_evals/<timestamp>.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and cost estimate, then exit.")
    parser.add_argument("--list-cases", action="store_true", help="List the reference cases and exit.")
    parser.add_argument("--verbose", action="store_true", help="Echo pipeline output while running.")
    return parser


def main(argv: Optional[Sequence[str]] = None, run_fn: Optional[Callable[[Any], Dict[str, Any]]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        configs = parse_configs(args.config)
        cases = select_cases(load_reference_cases(args.cases_file), args.cases)
    except (ConfigSpecError, ValueError, OSError) as err:
        print(f"[!] {err}", file=sys.stderr)
        return 2
    if args.repeats < 1:
        print("[!] --repeats must be >= 1.", file=sys.stderr)
        return 2

    if args.list_cases:
        for case in cases:
            expect = case["expect"]
            print(f"{case['id']:28s} outcome={expect.get('outcome')} dim={expect.get('dimensionality')} "
                  f"solvers={expect.get('solver_methods')}  {case.get('title', '')}")
        return 0

    n_runs = len(configs) * len(cases) * args.repeats
    print(f"Plan: {len(configs)} config(s) x {len(cases)} case(s) x {args.repeats} repeat(s) = {n_runs} pipeline run(s)"
          f" [{'mock' if args.mock else 'LIVE'}]")
    for config in configs:
        print(f"  - {config.name}: {config.describe()} -> {_compact_models(resolve_config_settings(config))}")
    if not args.mock:
        typical, worst = estimate_model_calls(n_runs, args.max_rounds)
        print(f"  Live cost estimate: ~{typical} Gemini calls typical, up to ~{worst} with re-mining/dynamic_script "
              f"self-correction. Each run is billed to GEMINI_API_KEY.")

    if args.dry_run:
        return 0
    if not args.mock and not os.environ.get("GEMINI_API_KEY"):
        print(
            "[!] GEMINI_API_KEY is not set. Refusing to run a live model comparison: the pipeline would silently "
            "fall back to mock mode and every config would get identical scores. Set GEMINI_API_KEY "
            "(e.g. in .env) or pass --mock to exercise the harness offline.",
            file=sys.stderr,
        )
        return 2

    out_dir = args.out or os.path.join(
        get_data_dir(), "model_evals", datetime.now().strftime("%Y%m%d_%H%M%S") + ("_mock" if args.mock else "")
    )
    report = run_model_eval(
        configs, cases,
        repeats=args.repeats, epochs=args.epochs, max_rounds=args.max_rounds, mock=args.mock,
        out_dir=out_dir, cases_file=os.path.abspath(args.cases_file), run_fn=run_fn, verbose=args.verbose,
    )

    print("")
    print("Summary:")
    for name, item in report["summary"]["configs"].items():
        failures = ", ".join(f"{cat}={count}" for cat, count in sorted(item["failures"].items())) or "none"
        print(f"  {name:20s} score={_fmt(item['mean_score'])} ({_fmt_signed(item['delta_vs_first'])})  "
              f"outcome={_fmt(item['outcome_rate'], 2)}  runtime={_fmt(item['mean_runtime_s'], 1)}s  failures: {failures}")
    if args.mock:
        print("  (mock mode: scores reflect the canned mock concepts, not the AI models)")
    print(f"Report: {report['paths']['markdown']}")
    print(f"JSON:   {report['paths']['json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
