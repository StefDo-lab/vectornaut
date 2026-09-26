import json
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

def _is_transient_model_error(exc: Exception) -> bool:
    """Server errors (5xx) and rate limits (429) are worth retrying; other client errors are not."""
    from google.genai import errors

    try:
        import anthropic

        if isinstance(exc, (anthropic.InternalServerError, anthropic.RateLimitError, anthropic.APIConnectionError)):
            return True
    except ImportError:
        pass

    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError):
        return getattr(exc, "code", None) == 429
    return False


# Token usage per model name, summed over all generate_content calls in this process.
MODEL_USAGE: Dict[str, Dict[str, int]] = {}


def _record_usage(model: Optional[str], response: Any) -> None:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return
    entry = MODEL_USAGE.setdefault(model or "unknown", {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0})
    entry["calls"] += 1
    entry["prompt_tokens"] += int(getattr(meta, "prompt_token_count", 0) or 0)
    entry["output_tokens"] += int(getattr(meta, "candidates_token_count", 0) or 0)
    entry["thinking_tokens"] += int(getattr(meta, "thoughts_token_count", 0) or 0)


def _print_usage_summary() -> None:
    for model, entry in sorted(MODEL_USAGE.items()):
        print(f"[model usage] {model}: {entry['calls']} calls, {entry['prompt_tokens']} prompt, "
              f"{entry['output_tokens']} output, {entry['thinking_tokens']} thinking tokens")


if os.environ.get("VECTORNAUT_LOG_USAGE", "").strip().lower() in ("1", "true", "yes"):
    import atexit

    atexit.register(_print_usage_summary)


class _StreamedResponse:
    """Minimal stand-in for GenerateContentResponse assembled from a stream."""

    def __init__(self, text: str, parsed: Any, usage_metadata: Any, candidates: Any):
        self.text = text
        self.parsed = parsed
        self.usage_metadata = usage_metadata
        self.candidates = candidates


def _streaming_config(config: Any) -> Any:
    """Copy of the request config that also streams thought summaries (keeps long requests alive)."""
    from google.genai import types

    if config is None:
        return types.GenerateContentConfig(thinking_config=types.ThinkingConfig(include_thoughts=True))
    thinking = getattr(config, "thinking_config", None)
    thinking = (thinking.model_copy(update={"include_thoughts": True}) if thinking is not None
                else types.ThinkingConfig(include_thoughts=True))
    return config.model_copy(update={"thinking_config": thinking})


def _parse_streamed_text(text: str, schema: Any) -> Any:
    if schema is None or not text.strip():
        return None
    if isinstance(schema, type) and hasattr(schema, "model_validate_json"):
        return schema.model_validate_json(text)
    return json.loads(text)


# ------------------------------------------------------------------
# Claude (Anthropic) adapter: stages whose model name starts with "claude-"
# are sent to the Anthropic Messages API instead of Gemini.
# ------------------------------------------------------------------

ANTHROPIC_API_BASE_URL = "https://api.anthropic.com"
_THINKING_TO_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}


def _is_claude_model(model: Optional[str]) -> bool:
    return bool(model) and str(model).startswith("claude-")


def _anthropic_client():
    import anthropic

    # Always target the public API explicitly: ANTHROPIC_BASE_URL may point at another service
    # (e.g. the host running this process). A key injected by the environment's proxy works with
    # any placeholder value.
    api_key = os.environ.get("VECTORNAUT_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "injected-by-environment"
    return anthropic.Anthropic(api_key=api_key, base_url=ANTHROPIC_API_BASE_URL, max_retries=4)


def _effort_from_config(config: Any) -> str:
    thinking = getattr(config, "thinking_config", None) if config is not None else None
    level = getattr(thinking, "thinking_level", None) if thinking is not None else None
    level = str(getattr(level, "value", level) or "").lower()
    return _THINKING_TO_EFFORT.get(level, "high")


def _generate_with_claude(model: str, contents: Any, config: Any, client_factory=None) -> "_StreamedResponse":
    """Streams one Claude request and returns a Gemini-like response (text, parsed, usage_metadata)."""
    from types import SimpleNamespace

    client = (client_factory or _anthropic_client)()
    prompt = contents if isinstance(contents, str) else json.dumps(contents, ensure_ascii=False, default=str)
    schema = getattr(config, "response_schema", None) if config is not None else None
    kwargs = {
        "model": model,
        "max_tokens": 64000,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": _effort_from_config(config)},
        "messages": [{"role": "user", "content": prompt}],
    }
    if isinstance(schema, type):
        kwargs["output_format"] = schema
    with client.messages.stream(**kwargs) as stream:
        message = stream.get_final_message()
    if getattr(message, "stop_reason", None) == "refusal":
        raise RuntimeError(f"Claude declined the request ({getattr(message, 'stop_details', None)})")
    text = "".join(getattr(b, "text", "") for b in message.content if getattr(b, "type", None) == "text")
    usage = getattr(message, "usage", None)
    usage_metadata = SimpleNamespace(
        prompt_token_count=int(getattr(usage, "input_tokens", 0) or 0),
        candidates_token_count=int(getattr(usage, "output_tokens", 0) or 0),
        thoughts_token_count=0,  # Claude bills thinking as output; it is included in output_tokens
    )
    return _StreamedResponse(text=text, parsed=_parse_streamed_text(text, schema),
                             usage_metadata=usage_metadata, candidates=None)


class _RetryingModels:
    """
    Wraps client.models: generate_content retries transient failures with exponential backoff.
    With streaming enabled (default; VECTORNAUT_MODEL_STREAM=0 disables it) the request is sent via
    generate_content_stream with thought summaries included, so data flows while the model thinks.
    Some egress proxies cut requests that stay silent for ~30 s, which long-thinking calls exceed.
    Thought parts are dropped; the answer text is parsed into the request's response_schema.
    """

    def __init__(self, models: Any, retries: int, base_delay: float, sleep=None, stream: bool = False):
        self._models = models
        self._retries = retries
        self._base_delay = base_delay
        self._sleep = sleep
        self._stream = stream

    def _generate_streamed(self, model: Any, contents: Any, config: Any) -> _StreamedResponse:
        text_parts = []
        last = None
        for chunk in self._models.generate_content_stream(model=model, contents=contents, config=_streaming_config(config)):
            last = chunk
            candidates = getattr(chunk, "candidates", None) or []
            content = getattr(candidates[0], "content", None) if candidates else None
            for part in (getattr(content, "parts", None) or []):
                if getattr(part, "thought", False):
                    continue
                if getattr(part, "text", None):
                    text_parts.append(part.text)
        text = "".join(text_parts)
        schema = getattr(config, "response_schema", None) if config is not None else None
        return _StreamedResponse(
            text=text,
            parsed=_parse_streamed_text(text, schema),
            usage_metadata=getattr(last, "usage_metadata", None),
            candidates=getattr(last, "candidates", None),
        )

    def generate_content(self, *args, **kwargs):
        import time

        sleep = self._sleep or time.sleep
        model = kwargs.get("model") or (args[0] if args else None)
        for attempt in range(self._retries + 1):
            try:
                if _is_claude_model(model):
                    response = _generate_with_claude(model, kwargs.get("contents"), kwargs.get("config"))
                elif self._stream and hasattr(self._models, "generate_content_stream"):
                    response = self._generate_streamed(model, kwargs.get("contents"), kwargs.get("config"))
                else:
                    response = self._models.generate_content(*args, **kwargs)
                _record_usage(model, response)
                return response
            except Exception as exc:
                if attempt >= self._retries or not _is_transient_model_error(exc):
                    raise
                delay = self._base_delay * (2 ** attempt)
                print(f"[model] Transient error ({str(exc)[:120]}); retry {attempt + 1}/{self._retries} in {delay:.0f}s")
                sleep(delay)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


class _RetryingClient:
    def __init__(self, client: Any, retries: int, base_delay: float, stream: bool = True):
        self._client = client
        self.models = _RetryingModels(client.models, retries, base_delay, stream=stream)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def get_client():
    """
    Initializes and returns the unified google-genai Client.
    Uses AI Studio (GEMINI_API_KEY) by default, or Vertex AI if configured.
    generate_content retries transient errors (5xx, 429): VECTORNAUT_MODEL_RETRIES (default 4)
    attempts with exponential backoff starting at VECTORNAUT_MODEL_RETRY_DELAY_S (default 2 s).
    """
    from google import genai

    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    if use_vertex:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        client = genai.Client(vertexai=True, project=project, location=location)
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            # The client will check GEMINI_API_KEY env variable automatically
            client = genai.Client()

    try:
        retries = max(0, int(os.environ.get("VECTORNAUT_MODEL_RETRIES", "4")))
    except ValueError:
        retries = 4
    try:
        base_delay = max(0.0, float(os.environ.get("VECTORNAUT_MODEL_RETRY_DELAY_S", "2")))
    except ValueError:
        base_delay = 2.0
    stream = os.environ.get("VECTORNAUT_MODEL_STREAM", "1").strip().lower() not in ("0", "false", "no")
    return _RetryingClient(client, retries, base_delay, stream=stream)

# ==========================================
# Per-stage model / thinking configuration
# ==========================================

DEFAULT_MODEL_NAME = "gemini-3.5-flash"

# Pipeline stages that call the model. Each can be configured via
# VECTORNAUT_MODEL_<STAGE> / VECTORNAUT_THINKING_<STAGE> (stage name upper-cased).
MODEL_STAGES = (
    "miner",
    "formulator",
    "auditor",
    "optimizer",
    "synthesizer",
    "script_generator",
    "test_generator",
    "chat",
    # Idea-space explorer (vectornaut/explorer): candidate generator, critic (both profiles) and the
    # analysis-first stage (one deep system-level analysis + direct-answer seeds per archive).
    # A stronger model can be set per stage, e.g. VECTORNAUT_MODEL_CRITIC=gemini-3.1-pro-preview.
    "explorer",
    "critic",
    "analyst",
)

# Thinking levels accepted by google-genai's ThinkingConfig.thinking_level.
THINKING_LEVELS = ("minimal", "low", "medium", "high")

def _stage_env_suffix(stage: str) -> str:
    normalized = (stage or "").strip().lower()
    if normalized not in MODEL_STAGES:
        raise ValueError(f"Unknown model stage '{stage}'. Expected one of: {', '.join(MODEL_STAGES)}")
    return normalized.upper()

def get_model_name(stage: str) -> str:
    """
    Resolves the Gemini model for a pipeline stage at call time:
    VECTORNAUT_MODEL_<STAGE> -> VECTORNAUT_MODEL -> DEFAULT_MODEL_NAME.
    """
    suffix = _stage_env_suffix(stage)
    for env_name in (f"VECTORNAUT_MODEL_{suffix}", "VECTORNAUT_MODEL"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return DEFAULT_MODEL_NAME

def normalize_thinking_level(value: Optional[str]) -> Optional[str]:
    """Returns the lower-cased thinking level if it is valid, otherwise None."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in THINKING_LEVELS else None

def get_thinking_level(stage: str, default: Optional[str] = None) -> Optional[str]:
    """
    Resolves the thinking level for a pipeline stage at call time:
    VECTORNAUT_THINKING_<STAGE> -> default. Invalid values fall back to default.
    A default of None means the stage sends no thinking config.
    """
    env_name = f"VECTORNAUT_THINKING_{_stage_env_suffix(stage)}"
    raw = os.environ.get(env_name, "").strip()
    if raw:
        level = normalize_thinking_level(raw)
        if level:
            return level
        print(f"[!] Ignoring invalid {env_name}='{raw}' (allowed: {', '.join(THINKING_LEVELS)}). Using default: {default}")
    return default

def get_thinking_config(stage: str, default: Optional[str] = None, override: Optional[str] = None):
    """
    Builds the google-genai ThinkingConfig for a stage, or None if the stage runs
    without thinking. An explicit override (e.g. a constructor/call argument) wins
    over the environment.
    """
    level = override or get_thinking_level(stage, default)
    if not level:
        return None
    from google.genai import types
    return types.ThinkingConfig(thinking_level=level)

class ParameterProposal(BaseModel):
    name: str = Field(description="Name of the parameter, e.g., riblet_height, riblet_spacing, viscosity")
    value: float = Field(description="Proposed initial value of the parameter")
    min_bound: float = Field(description="Minimum physically plausible bound")
    max_bound: float = Field(description="Maximum physically plausible bound")
    justification: str = Field(description="Explanation for the value and bounds based on bionic inspiration")

class MinerConceptOutput(BaseModel):
    design_name: str = Field(description="Name of the biomimetic design concept")
    inspiration_source: str = Field(description="The biological or natural organism/phenomenon that inspires the design")
    domain: str = Field(description="The scientific domain, e.g., Fluid Dynamics, Thermodynamics, Electromagnetics")
    physical_mechanism: str = Field(description="Explanation of the physical mechanism being mimicked")
    parameters: List[ParameterProposal] = Field(description="List of proposed physical parameters")
    svg_schematic: str = Field(description="Self-contained responsive 2D SVG markup illustrating the micro/nanostructure of the proposed material, scaled dynamically according to the parameter values. Uses dark-mode aligned accent colors (neon purple, neon cyan, dark fills) and technical labels.")
    # Optional with defaults so that recorded answers without these fields still parse.
    request_feasible: bool = Field(default=True, description="False only if the user's request itself is physically impossible as stated (e.g. violates energy conservation or the second law of thermodynamics, or contains mutually contradictory requirements), so that no concept can fulfil it. True if at least one physically valid concept can fulfil the request.")
    infeasibility_reason: Optional[str] = Field(default=None, description="If request_feasible is false: a short explanation for the user naming the violated physical law or the contradictory requirements. Otherwise null.")

    @property
    def proposed_parameters(self) -> Dict[str, float]:
        return {p.name: p.value for p in self.parameters}

    @property
    def suggested_bounds(self) -> Dict[str, List[float]]:
        return {p.name: [p.min_bound, p.max_bound] for p in self.parameters}

    @property
    def parameter_justifications(self) -> Dict[str, str]:
        return {p.name: p.justification for p in self.parameters}

class ModelFormulation(BaseModel):
    governing_equation: str = Field(description="Symbolic ordinary/partial differential equation, e.g., 'd2u_dy2 = -c_pg' or 'd2T_dx2 = 0'")
    boundary_conditions: List[str] = Field(description="List of symbolic boundary conditions, e.g., ['u(0) = lambda * du_dy(0)', 'u(1) = u_free']")
    independent_variables: List[str] = Field(description="Independent variables (coordinates) for the system, e.g., ['y'] or ['x']")
    dependent_variables: List[str] = Field(description="Dependent variables (fields) for the system, e.g., ['u'] or ['T']")
    # Optional so that recorded answers without this field still parse.
    parameters: Optional[List[ParameterProposal]] = Field(default=None, description="Only parameters to add or change: every NEW parameter the equations or boundary conditions need that is not in the proposed list (e.g. boundary values, ambient temperatures, loads, source terms), and any EXISTING parameter (same name) whose value, bounds or justification you change, e.g. to widen its bounds. Proposed parameters that are not listed are kept unchanged. Empty or null if nothing is added or changed.")

class MinerOutput(BaseModel):
    design_name: str = Field(description="Name of the biomimetic design concept")
    inspiration_source: str = Field(description="The biological or natural organism/phenomenon that inspires the design")
    domain: str = Field(description="The scientific domain, e.g., Fluid Dynamics, Thermodynamics, Electromagnetics")
    physical_mechanism: str = Field(description="Explanation of the physical mechanism being mimicked")
    parameters: List[ParameterProposal] = Field(description="List of proposed physical parameters")
    governing_equation: str = Field(description="Symbolic ordinary/partial differential equation, e.g., 'd2u_dy2 = -c_pg' or 'd2T_dx2 = 0'")
    boundary_conditions: List[str] = Field(description="List of symbolic boundary conditions, e.g., ['u(0) = lambda * du_dy(0)', 'u(1) = u_free']")
    independent_variables: List[str] = Field(description="Independent variables (coordinates) for the system, e.g., ['y'] or ['x']")
    dependent_variables: List[str] = Field(description="Dependent variables (fields) for the system, e.g., ['u'] or ['T']")
    svg_schematic: str = Field(description="Self-contained responsive 2D SVG markup illustrating the micro/nanostructure of the proposed material, scaled dynamically according to the parameter values. Uses dark-mode aligned accent colors (neon purple, neon cyan, dark fills) and technical labels.")

    @property
    def proposed_parameters(self) -> Dict[str, float]:
        return {p.name: p.value for p in self.parameters}

    @property
    def suggested_bounds(self) -> Dict[str, List[float]]:
        return {p.name: [p.min_bound, p.max_bound] for p in self.parameters}

    @property
    def parameter_justifications(self) -> Dict[str, str]:
        return {p.name: p.justification for p in self.parameters}

class AuditedParameter(BaseModel):
    name: str = Field(description="Name of the audited parameter")
    value: float = Field(description="The sanitized, audited value")

class DimensionlessNumber(BaseModel):
    name: str = Field(description="Name of the dimensionless number, e.g., ReynoldsNumber")
    value: float = Field(description="Calculated value")

class BaselineParameter(BaseModel):
    name: str = Field(description="Name of a parameter (as in the equations/audited parameters)")
    value: float = Field(description="Its value in the conventional (non-bionic) reference design")

class MetricSpec(BaseModel):
    """Declarative description of the quantity the solver reports as primary/reference metric."""
    kind: str = Field(description="One of: value_at, derivative_at, max, min, max_abs, mean, integral (of the dependent variable)")
    location: Optional[str] = Field(default=None, description="Where to evaluate, in the coordinates of the boundary conditions: 1D a coordinate such as '0', '1' or a parameter name; 2D a line/edge such as 'x=1' or 'y=0' (averaged along it, derivative normal to it) or a point '(0.5, 0.5)'. Empty for max/min/max_abs/mean/integral over the whole domain.")
    scale: Optional[str] = Field(default=None, description="Optional factor as an expression in parameter names that converts the (normalized) result into the physical quantity, e.g. 'viscosity/channel_half_height' or 'thermal_conductivity/L'")
    unit: Optional[str] = Field(default=None, description="Unit of the scaled metric, e.g. Pa, m, W/m^2, K")
    label: Optional[str] = Field(default=None, description="Short name of the metric, e.g. 'Maximum deflection'")
    transform: Optional[str] = Field(default=None, description="Optional figure of merit as a SymPy expression in m (the scaled raw metric) and parameter names, applied to the design, reference and baseline metric before the gain is computed, when the objective is a nonlinear function of the metric, e.g. 'sqrt(2*adhesion_energy/m)' (detachment stress from the compliance m). unit, label and objective_metric.lower_is_better then refer to the transformed quantity. Empty = the metric itself.")

class AxisMetadata(BaseModel):
    label: str = Field(description="Label for the coordinate or field")
    unit: str = Field(description="Unit of measurement, e.g., m, m/s, K, V")

class MetricMetadata(BaseModel):
    label: str = Field(description="Label for the metric, e.g., Wall Shear Stress")

class UIMetadata(BaseModel):
    domain_name: str = Field(description="Name of the physical domain")
    independent_var: AxisMetadata = Field(description="Metadata for independent variable")
    dependent_var: AxisMetadata = Field(description="Metadata for dependent variable")
    primary_metric: MetricMetadata = Field(description="Metadata for primary solver metric")
    reference_metric: MetricMetadata = Field(description="Metadata for reference solver metric")
    performance_gain: MetricMetadata = Field(description="Metadata for performance gain metric")

class ObjectiveMetricContract(BaseModel):
    objective_name: str = Field(description="Human-readable optimization objective name")
    score_field: str = Field(default="performance_gain_pct", description="Simulator output field used for ranking")
    direction: str = Field(default="maximize", description="Either 'maximize' or 'minimize'")
    primary_metric: str = Field(description="Name of the primary solver metric")
    reference_metric: str = Field(description="Name of the reference metric")
    lower_is_better: bool = Field(default=False, description="Whether lower primary metric values are preferable")
    acceptance_threshold: float = Field(default=0.0, description="Minimum acceptable score when maximizing, maximum acceptable score when minimizing")
    hard_constraints: List[str] = Field(default_factory=lambda: [
        "relative_error <= 1.0",
        "parameters within bounds",
        "finite numeric outputs",
    ], description="Non-negotiable constraints for accepting solver or sweep results")
    design_variables: Optional[List[str]] = Field(default=None, description="Names of audited parameters that are free design choices (geometry, material) which optimizer and parameter sweeps may vary; excludes operating conditions, loads and environment")

class AuditorOutput(BaseModel):
    audit_passed: bool = Field(description="True if the parameters are physically plausible and safe for simulator execution")
    audit_notes: str = Field(description="Detailed notes explaining the checks performed, reasons for success/failure, or adjustments made")
    # Optional with defaults so that recorded answers without these fields still parse.
    request_feasible: bool = Field(default=True, description="False only if the user's request itself is physically impossible as stated (e.g. violates energy conservation or the second law of thermodynamics, or contains mutually contradictory requirements), so that no other concept could pass either. A concept that merely fails (unsafe, unrealistic parameters) keeps this true and sets audit_passed=false.")
    infeasibility_reason: Optional[str] = Field(default=None, description="If request_feasible is false: a short explanation for the user naming the violated physical law or the contradictory requirements. Otherwise null.")
    audited_parameters: List[AuditedParameter] = Field(description="The finalized, sanitized parameters to be passed to the simulator")
    dimensionless_numbers: List[DimensionlessNumber] = Field(description="Calculated key dimensionless values for verification")
    simulation_coefficient: float = Field(description="Derived core coefficient (e.g., slip length, heat transfer coefficient) used directly in the simulator boundary/PDE equations")
    solver_method: str = Field(description="The selected solver method: 'analytical', 'scipy', or 'pinn'")
    ui_metadata: UIMetadata = Field(description="Metadata describing labels, units, and axes for dynamic UI rendering")
    objective_metric: Optional[ObjectiveMetricContract] = Field(default=None, description="Contract defining how solver outputs are ranked and accepted")

    @property
    def audited_parameters_dict(self) -> Dict[str, float]:
        # Read-only view: a fresh dict on every access, writes to it are discarded.
        # Use add_injected_parameters() to record additional parameters.
        return {p.name: p.value for p in self.audited_parameters}

    def add_injected_parameters(self, params: Dict[str, float]) -> List[str]:
        """
        Records parameters that were not audited (e.g. defaults auto-injected by the
        solver for symbols missing in the equations/BCs) so they show up in history,
        synthesis and the API response. Existing audited values are never overwritten.
        Returns the names that were added.
        """
        existing = {p.name for p in self.audited_parameters}
        added = []
        for name, value in params.items():
            if name in existing:
                continue
            self.audited_parameters.append(AuditedParameter(name=name, value=float(value)))
            existing.add(name)
            added.append(name)
        if added:
            injected_desc = ", ".join(f"{name}={float(params[name]):g}" for name in added)
            note = f"Automatisch ergänzte Standardparameter (nicht auditiert): {injected_desc}"
            self.audit_notes = f"{self.audit_notes}\n{note}" if self.audit_notes else note
        return added

    @property
    def dimensionless_numbers_dict(self) -> Dict[str, float]:
        return {p.name: p.value for p in self.dimensionless_numbers}

    # Metric and baseline (declared after the methods; pydantic collects them all the same).
    metric: Optional[MetricSpec] = Field(default=None, description="Which quantity of the solved field the solver reports as primary/reference metric. Without it the solver reports du/dx at the lower boundary (1D) or the field mean (2D).")
    baseline_parameters: Optional[List[BaselineParameter]] = Field(default=None, description="Parameter values of the conventional (non-bionic) reference design that differ from the audited ones. The solver re-solves with them and reports performance_gain_pct as the relative improvement of the metric over this baseline.")
    baseline_description: Optional[str] = Field(default=None, description="Short name of the baseline design, e.g. 'uncoated double glazing'")

    @field_validator("baseline_parameters", mode="before")
    @classmethod
    def _baseline_parameters_from_mapping(cls, value: Any) -> Any:
        # Accept {"name": value} as well as the list-of-pairs form of the schema.
        if isinstance(value, dict):
            return [{"name": str(k), "value": v} for k, v in value.items()]
        return value

    @property
    def baseline_parameters_dict(self) -> Dict[str, float]:
        return {p.name: p.value for p in (self.baseline_parameters or [])}

class SimulatorOutput(BaseModel):
    solver_method: str = Field(description="The solver method used: analytical, scipy, pinn")
    epochs_trained: int = Field(description="Number of epochs the PINN was trained")
    final_loss: float = Field(description="The final loss value achieved by the PINN (0 if not trained)")
    loss_history: List[float] = Field(description="History of loss values recorded during training")
    performance_gain_pct: float = Field(description="Calculated performance improvement in percent")
    relative_error: float = Field(description="Relative error between primary and reference solution")
    sample_points: List[Any] = Field(description="Coordinate values where the field was evaluated")
    solution_primary: List[float] = Field(description="Primary solver predicted field values at sample points")
    solution_reference: List[float] = Field(description="Reference solver field values at sample points")
    primary_metric_value: float = Field(description="Derived performance metric from the primary solver")
    reference_metric_value: float = Field(description="Derived performance metric from the reference solver")
    custom_plot_url: Optional[str] = Field(description="Relative URL to the dynamically generated plot image", default=None)
    validation_passed: Optional[bool] = Field(description="True if automated validation tests passed", default=None)
    validation_report: Optional[str] = Field(description="Markdown report of the automated validation runs", default=None)
    validation_tests: Optional[List[Dict[str, Any]]] = Field(description="Details of each run validation test", default=None)
    script_path: Optional[str] = Field(description="Absolute path to the generated dynamic solver script", default=None)
    params_json_path: Optional[str] = Field(description="Absolute path to the generated solver parameter JSON", default=None)
    test_script_path: Optional[str] = Field(description="Absolute path to the generated validation test script", default=None)
    test_output_path: Optional[str] = Field(description="Absolute path to the generated validation test result JSON", default=None)
    execution_mode: Optional[str] = Field(description="Execution mode metadata for generated or deterministic solvers", default=None)
    objective_metric: Optional[Dict[str, Any]] = Field(description="Objective metric contract used to score solver results", default=None)
    parameter_sweep: Optional[Dict[str, Any]] = Field(description="Dynamic-script parameter sweep summary and candidate results", default=None)
    # Metric and gain provenance (None for results produced before these fields existed).
    metric_spec: Optional[Dict[str, Any]] = Field(description="Metric spec used for primary/reference/baseline metric values (kind, location, scale, unit, label); None = default metric", default=None)
    metric_unit: Optional[str] = Field(description="Unit of primary/reference/baseline metric values", default=None)
    baseline_metric_value: Optional[float] = Field(description="Metric of the baseline design the gain is computed against", default=None)
    gain_basis: Optional[str] = Field(description="What performance_gain_pct compares against: 'baseline_parameters' (auditor's reference design), 'bionic_effect' (same design without slip/simulation coefficient) or 'none' (no baseline: gain not computable, performance_gain_pct is 0 and means n/a)", default=None)
    gain_note: Optional[str] = Field(description="Why the metric spec or the gain could not be used as requested", default=None)
    solver_note: Optional[str] = Field(description="Solver fallbacks and their reasons (e.g. analytical solution failed, SciPy BVP used); None = the requested solvers were used", default=None)
    boundary_derivatives: Optional[List[Dict[str, float]]] = Field(description="du/dx of the primary 1D solution at the domain ends from the solver itself ([{'location': x, 'value': du/dx}]), for the validator's derivative BC check", default=None)
