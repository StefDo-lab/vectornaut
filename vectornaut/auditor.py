import math
from .config import get_client, get_model_name, get_thinking_config, MinerOutput, AuditorOutput, AuditedParameter, DimensionlessNumber, ObjectiveMetricContract

def extract_param(params_dict, keys, default):
    for key in keys:
        if key in params_dict:
            return params_dict[key]
    return default


def _close(a, b, rel=1e-9) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return a == b or abs(a - b) <= rel * max(abs(a), abs(b))


def _fmt(value) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


# UI default domain -> keywords in the model's domain name (checked in this order, so
# "Fluid Mechanics" is fluid and "Structural Mechanics" structural).
_UI_DOMAIN_KEYWORDS = (
    ("fluid", ("fluid", "flow", "hydrodynam", "aerodynam", "strömung", "stroemung")),
    ("thermal", ("thermo", "thermal", "heat", "wärme", "waerme", "temperat")),
    ("electro", ("electr", "elektr", "magnet", "dielectric")),
    ("structural", ("struct", "solid", "elastic", "mechanic", "mechanik", "beam", "bending", "biege", "statik")),
)

_UI_DEFAULT_METADATA = {
    "fluid": {
        "domain_name": "Fluid Dynamics",
        "independent_var": { "label": "Channel Height", "unit": "m" },
        "dependent_var": { "label": "Flow Velocity", "unit": "m/s" },
        "primary_metric": { "label": "PINN Wall Shear Stress" },
        "reference_metric": { "label": "Analytical Wall Shear Stress" },
        "performance_gain": { "label": "Drag Reduction Efficiency" }
    },
    "thermal": {
        "domain_name": "Thermodynamics",
        "independent_var": { "label": "Plate Position (x)", "unit": "m" },
        "dependent_var": { "label": "Temperature (T)", "unit": "K" },
        "primary_metric": { "label": "PINN Thermal Gradient" },
        "reference_metric": { "label": "Analytical Thermal Gradient" },
        "performance_gain": { "label": "Thermal Insulation Efficiency" }
    },
    "electro": {
        "domain_name": "Electromagnetics",
        "independent_var": { "label": "Gap Distance (y)", "unit": "m" },
        "dependent_var": { "label": "Electrostatic Potential (V)", "unit": "V" },
        "primary_metric": { "label": "PINN Electric Field" },
        "reference_metric": { "label": "Analytical Electric Field" },
        "performance_gain": { "label": "Field Attenuation Efficiency" }
    },
    "structural": {
        "domain_name": "Structural Mechanics",
        "independent_var": { "label": "Position along the Member (x)", "unit": "m" },
        "dependent_var": { "label": "Deflection (w)", "unit": "m" },
        "primary_metric": { "label": "Numerical Deflection" },
        "reference_metric": { "label": "Analytical Deflection" },
        "performance_gain": { "label": "Structural Performance Gain" }
    },
}


def match_ui_domain(*domain_texts):
    """Returns 'fluid', 'thermal', 'electro' or 'structural' for the first text that names one, else None."""
    for text in domain_texts:
        lowered = str(text or "").lower()
        if not lowered:
            continue
        for key, keywords in _UI_DOMAIN_KEYWORDS:
            if any(keyword in lowered for keyword in keywords):
                return key
    return None


def _default_ui_metadata(matched_domain, miner_output) -> dict:
    if matched_domain in _UI_DEFAULT_METADATA:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _UI_DEFAULT_METADATA[matched_domain].items()}
    # Unknown domain: neutral labels instead of pretending it is a fluid problem.
    independent = (getattr(miner_output, "independent_variables", None) or ["x"])[0]
    dependent = (getattr(miner_output, "dependent_variables", None) or ["u"])[0]
    return {
        "domain_name": getattr(miner_output, "domain", None) or "Scientific Domain",
        "independent_var": { "label": independent, "unit": "" },
        "dependent_var": { "label": dependent, "unit": "" },
        "primary_metric": { "label": "Primary Metric" },
        "reference_metric": { "label": "Reference Metric" },
        "performance_gain": { "label": "Performance Gain" },
    }


def _tracked_parameter(coefficient, params):
    """Name of the single parameter whose value equals the coefficient, else None."""
    matches = [name for name, value in params.items() if _close(value, coefficient)]
    return matches[0] if len(matches) == 1 else None


def _is_reynolds_name(name) -> bool:
    lowered = str(name or "").strip().lower()
    return "reynolds" in lowered or lowered == "re" or lowered.startswith("re_")


_VELOCITY_KEYS = ["free_stream_velocity", "design_flow_velocity", "flow_velocity", "mean_velocity", "inlet_velocity", "velocity", "U_ski", "ski_velocity"]
_LENGTH_KEYS = ["characteristic_length", "length", "hydraulic_diameter", "diameter", "pipe_diameter", "hose_diameter", "channel_height", "channel_width", "film_thickness", "gap_height", "chord_length"]
_KINEMATIC_VISCOSITY_KEYS = ["kinematic_viscosity", "nu"]
_DYNAMIC_VISCOSITY_KEYS = ["viscosity", "dynamic_viscosity", "mu"]
_DENSITY_KEYS = ["density", "fluid_density", "rho"]


def _positive_param(params, keys):
    for key in keys:
        value = params.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value != 0:
            return key, abs(float(value))
    return None


def compute_reynolds_number(params):
    """
    Re = v L / nu (kinematic viscosity) or rho v L / mu (dynamic viscosity and density),
    only from parameters that are actually present. Returns (value, description) or None.
    """
    velocity = _positive_param(params, _VELOCITY_KEYS)
    length = _positive_param(params, _LENGTH_KEYS)
    if not velocity or not length:
        return None
    nu = _positive_param(params, _KINEMATIC_VISCOSITY_KEYS)
    if nu:
        return velocity[1] * length[1] / nu[1], f"{velocity[0]}, {length[0]}, {nu[0]}"
    mu = _positive_param(params, _DYNAMIC_VISCOSITY_KEYS)
    rho = _positive_param(params, _DENSITY_KEYS)
    if mu and rho:
        return rho[1] * velocity[1] * length[1] / mu[1], f"{rho[0]}, {velocity[0]}, {length[0]}, {mu[0]}"
    return None

class Auditor:
    def __init__(self, client=None):
        self.client = client

    def audit_design(self, miner_output: MinerOutput, override_parameters: dict = None, user_query: str = None) -> AuditorOutput:
        """
        Audits proposed parameters against physical constraints.
        Uses thinking_level="high" by default for deep physical reasoning
        (override via VECTORNAUT_MODEL_AUDITOR / VECTORNAUT_THINKING_AUDITOR).
        """
        from google.genai import types

        # The values to audit this round: the Miner's proposal with the user's or the
        # optimizer's overrides applied. Without this the model would audit (and derive the
        # simulation coefficient and dimensionless numbers from) the previous parameters.
        overrides = {k: v for k, v in (override_parameters or {}).items()}
        proposed_parameters = {**miner_output.proposed_parameters, **overrides}
        override_block = ""
        if overrides:
            override_block = f"""
        PARAMETER VALUES TO AUDIT THIS ROUND:
        The following values were set by the user or by the optimizer after the previous round. They replace the Miner's proposal and are already included in "Proposed Parameters" above:
        {overrides}
        Audit exactly these values. Base audited_parameters, the dimensionless numbers, the simulation_coefficient and your audit_notes on them, not on the Miner's original values.
"""

        prompt = f"""
        You are a senior physical auditor verifying design specifications for a biomimetic material.
        Here is the proposal from the Miner stage:
        
        Design Name: {miner_output.design_name}
        Original User Query: {user_query or "None"}
        Inspiration Source: {miner_output.inspiration_source}
        Domain: {miner_output.domain}
        Physical Mechanism: {miner_output.physical_mechanism}
        Proposed Parameters: {proposed_parameters}
        Suggested Bounds: {miner_output.suggested_bounds}
        Parameter Justifications: {miner_output.parameter_justifications}
        Governing Equation: {miner_output.governing_equation}
        Boundary Conditions: {miner_output.boundary_conditions}
{override_block}
        FEASIBILITY OF THE REQUEST ITSELF (request_feasible, infeasibility_reason):
        Distinguish two kinds of failure:
        - The CONCEPT fails (unsafe, structurally unstable, parameters unrealistic and not fixable, model unsuitable): set audit_passed=false and keep request_feasible=true. The pipeline will then search for a different concept.
        - The REQUEST itself is physically impossible as stated, so that no other concept could pass either: it violates a conservation law (e.g. more energy out than in, efficiency above 100 % in a closed or adiabatic system without energy input, perpetual motion), the second law of thermodynamics (e.g. heat flowing on its own from cold to hot, exceeding the Carnot efficiency), or its requirements contradict each other. Then set request_feasible=false, audit_passed=false, and explain in infeasibility_reason, in one or two sentences for the user, which law or requirement is violated and, if there is one, the closest physically possible alternative. The pipeline stops and shows this reason instead of searching for new concepts.
        Do not set request_feasible=false for requests that are merely difficult or ambitious, or when only this concept's parameters are the problem.

        Perform a rigorous audit of these parameters:
        1. Check if the parameter ranges make physical sense (e.g., density, viscosity, spacing, height must be positive).
        2. Adjust any parameters that are out of bounds or physically unrealistic (e.g., riblet spacing should be larger than or equal to height, viscosity must be reasonable).
        3. Calculate the key dimensionless numbers of this problem (e.g. Reynolds, Péclet, Biot, Nusselt numbers). Use the actual characteristic length, velocity and material properties from the parameters above; where a needed quantity is not a parameter, state the value you assumed and why in audit_notes. Do not report numbers that do not apply to this problem.
        4. Derive the core simulation coefficient of this mechanism (simulation_coefficient), e.g. an effective slip length, heat transfer coefficient or effective stiffness, from the parameters and the physics of the mechanism, and explain the derivation in audit_notes.
           - Only for riblet geometries (parameters riblet_height and riblet_spacing), a common estimate of the slip length is
             lambda = 0.2 * riblet_spacing * (1 - exp(-2.0 * riblet_height / riblet_spacing)); the pipeline recomputes the coefficient with this formula when both parameters are present.
           - For every other mechanism derive the coefficient yourself (it is not the riblet formula). If the mechanism has no such single coefficient, return 0.0 and say so in audit_notes.
           - The equations use simulation_coefficient only if they refer to it (or to slip_length, lambda, slippage_coefficient); otherwise it is informational.
        5. Select the best solver_method out of:
           - 'analytical': If the equation is linear and has a simple, closed-form solution.
           - 'scipy': For standard 1D ODE BVPs that might have minor nonlinearities but are fast to solve. (Do NOT select 'scipy' if the system has multiple independent variables/2D coordinates).
           - 'pinn': If the user wants standard neural network solver, or for typical 2D PDE systems.
           - 'dynamic_script': If the system is transient (time-dependent), multi-domain, highly non-linear, if the equations are non-standard, or if the user explicitly requested a custom script solver/simulation.
        6. Define the ui_metadata schema for dynamic rendering. It MUST follow this structure:
           {{
             "domain_name": "Scientific Domain Name",
             "independent_var": {{ "label": "Label of independent coordinate (e.g., Channel Height)", "unit": "m" }},
             "dependent_var": {{ "label": "Label of dependent field (e.g., Flow Velocity)", "unit": "m/s" }},
             "primary_metric": {{ "label": "Label of primary solver metric (e.g., PINN Wall Shear Stress)" }},
             "reference_metric": {{ "label": "Label of reference solver metric (e.g., Analytical Wall Shear Stress)" }},
             "performance_gain": {{ "label": "Label of performance gain metric (e.g., Drag Reduction Efficiency)" }}
           }}

        7. Define objective_metric as a strict optimization contract:
           - objective_name: human-readable target, e.g. "Drag Reduction Efficiency" or "Fatigue Life Gain".
           - score_field: use "performance_gain_pct" unless there is a concrete reason not to.
           - direction: "maximize" if higher score is better, "minimize" if lower score is better.
           - primary_metric/reference_metric: must match the physical meaning of ui_metadata metrics.
           - lower_is_better: true if the raw primary metric is a loss/risk/stress/friction metric.
           - acceptance_threshold: minimum acceptable score for maximize, maximum acceptable score for minimize.
           - hard_constraints: include relative error, finite outputs, parameter bounds, and any domain-specific safety limits.
           - design_variables: names of the audited parameters that are free design choices (geometry, material) and may be varied by sweeps/optimization; exclude operating conditions, loads and environment (e.g. heat flux, pulse duration, inlet velocity, ambient temperature).

        7b. Define metric (what the solver measures) and baseline_parameters (what the design is compared with):
           Without a metric the solver reports du/dx at the lower boundary (1D) or the mean of the field (2D), which is rarely the quantity the user asked for.
           - metric.kind: one of "value_at", "derivative_at", "max", "min", "max_abs", "mean", "integral" of the dependent variable.
           - metric.location: in the coordinates of the boundary conditions (usually normalized to [0, 1]).
             1D: a coordinate such as "0", "1" or a parameter name. 2D: a line/edge such as "x=1" or "y=0" (values and derivatives are averaged along it,
             derivatives are taken normal to it) or a point "(0.5, 0.5)" for value_at. Leave empty for max/min/max_abs/mean/integral over the whole domain.
           - metric.scale: optional expression in parameter names that turns the normalized result into the physical quantity (empty = 1).
           - metric.unit and metric.label: unit and short name of the scaled metric; ui_metadata.primary_metric/reference_metric must describe the same quantity.
           - metric.transform (optional): if the objective is a NONLINEAR function of the metric, a SymPy expression in m (the scaled metric) and parameter names
             that turns m into the figure of merit. It is applied to the design, reference and baseline metric before performance_gain_pct is computed, so
             the gain is the relative change of the figure of merit, not of m. metric.unit, metric.label and objective_metric.lower_is_better then refer to the
             transformed quantity. Allowed: numbers, m, parameter names, + - * / ** and sqrt, exp, log, sin, cos, tan, sinh, cosh, tanh, abs, min, max, pi.
             Example: foul-release coating where the solver gives the compliance C of the coating and the detachment stress is sigma_c = sqrt(2*w/C) with the
             adhesion energy w: metric {{"kind": "max_abs", "unit": "Pa", "label": "Critical detachment stress", "transform": "sqrt(2*adhesion_energy/m)"}}
             with objective_metric.lower_is_better = true (a lower detachment stress releases fouling more easily). Leave transform empty when the metric itself is the objective.
           Examples:
             Wall shear stress on the lower wall of a channel normalized by its half height: {{"kind": "derivative_at", "location": "0", "scale": "viscosity/channel_half_height", "unit": "Pa", "label": "Wall shear stress"}}
             Maximum deflection of a beam whose deflection w is already in metres: {{"kind": "max_abs", "unit": "m", "label": "Maximum deflection"}}
             Heat flux into the room through a wall normalized by its thickness L (outdoor side at x=0, room at x=1, q = k dT/dx / L): {{"kind": "derivative_at", "location": "1", "scale": "thermal_conductivity/L", "unit": "W/m^2", "label": "Heat flux to the room"}}
           - baseline_parameters: the parameter values of the conventional (non-bionic) reference design, as a list of {{"name": ..., "value": ...}} pairs that override
             the audited parameters (list only the ones that differ), plus a short baseline_description. The solver solves the same equations again with them and
             computes performance_gain_pct as the relative improvement of the metric: (baseline - design) / |baseline| * 100 if objective_metric.lower_is_better,
             otherwise (design - baseline) / |baseline| * 100. Only parameters used in the equations, boundary conditions or metric.scale change the result.
             Example: low-e coated glazing compared with uncoated glazing: baseline_parameters [{{"name": "coating_emissivity", "value": 0.84}}], baseline_description "uncoated double glazing".
           - If there is no sensible conventional baseline, leave baseline_parameters empty: the gain is then reported as not available (n/a), not as 0 %.
             (For slip-flow models whose equations use the slip length or simulation_coefficient, the design without slip is used as baseline automatically.)
           - Benefits outside the solved field (mass, cost, manufacturability) cannot be computed by the solver. Describe them in audit_notes and choose metric and
             baseline for what the field shows (e.g. deflection against the solid beam), then set objective_metric.lower_is_better to match the metric
             (true for heat loss, deflection, drag, stress) and acceptance_threshold in percent of that improvement (it may be negative if a loss is acceptable).

         8. Perform a structural and material stress check:
            - Identify likely operational stresses and load conditions from the user query (e.g. pressure, compression, bending, friction, shear, thermal load).
            - Estimate the material limits (e.g. yield strength, maximum temperature) of the bionic material.
            - If the proposed parameters would cause a structural collapse (e.g. if microstructures are too thin and tall under high pressure, or if temperatures exceed material limits), explain this failure in 'audit_notes'.
            - If these parameters are physically dangerous and cannot be resolved, set 'audit_passed' to False, otherwise True.

         Output the results structured strictly according to the AuditorOutput schema.
         """

        client = self.client or get_client()
        response = client.models.generate_content(
            model=get_model_name("auditor"),
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=get_thinking_config("auditor", default="high"),
                response_mime_type="application/json",
                response_schema=AuditorOutput,
            )
        )
        audited: AuditorOutput = response.parsed

        # Programmatic safety guardrails applied on top of LLM outputs. Every change is
        # recorded in `changes` and reported in the audit notes.
        model_params = audited.audited_parameters_dict
        sanitized_params = model_params.copy()
        changes = []

        # Explicitly apply user/optimizer overrides (the model saw them in the prompt)
        for k, v in overrides.items():
            if k in model_params and not _close(model_params[k], v):
                changes.append(f"{k}: {_fmt(model_params[k])} -> {_fmt(v)} (Vorgabe aus Override übernommen)")
            sanitized_params[k] = v

        suggested_bounds = miner_output.suggested_bounds
        for param, val in sanitized_params.items():
            if param in suggested_bounds:
                bounds = suggested_bounds[param]
                if len(bounds) == 2:
                    min_val, max_val = bounds[0], bounds[1]
                    original = val
                    if val < min_val:
                        val = min_val
                    elif val > max_val:
                        val = max_val
                    if val != original:
                        changes.append(f"{param}: {_fmt(original)} -> {_fmt(val)} (auf Grenzen [{_fmt(min_val)}, {_fmt(max_val)}] begrenzt)")
                    sanitized_params[param] = val
        
        # Enforce positive values for key dimensions
        for key in ["viscosity", "kinematic_viscosity", "riblet_height", "riblet_spacing", "free_stream_velocity", "design_flow_velocity", "conductivity"]:
            if key in sanitized_params:
                positive = max(1e-6, sanitized_params[key])
                if positive != sanitized_params[key]:
                    changes.append(f"{key}: {_fmt(sanitized_params[key])} -> {_fmt(positive)} (muss positiv sein)")
                sanitized_params[key] = positive

        # Domain-aware recalculations. "Thermodynamics", "Heat Transfer", "Structural
        # Mechanics", "Electrostatics" etc. map to the UI default domains by keyword.
        matched_domain = match_ui_domain(
            miner_output.domain,
            audited.ui_metadata.domain_name if audited.ui_metadata else None,
        )
        recalculated_coeff = audited.simulation_coefficient
        coeff_source = None
        
        if matched_domain == "fluid":
            # Check if slip length is explicitly overridden
            slip_keys = ["slip_length", "slippage_coefficient", "lambda", "simulation_coefficient"]
            overridden_slip = None
            for k in slip_keys:
                if k in overrides:
                    overridden_slip = overrides[k]
                    coeff_source = f"Override {k}"
                    break

            if overridden_slip is not None:
                recalculated_coeff = overridden_slip
                if "riblet_spacing" in sanitized_params or "spacing" in sanitized_params:
                    s = extract_param(sanitized_params, ["riblet_spacing", "spacing"], 0.03)
                    if s < 0.005:
                        char_height = 10.0 * s
                        recalculated_coeff = recalculated_coeff / char_height
            elif "riblet_height" in sanitized_params and "riblet_spacing" in sanitized_params:
                # Riblet geometry: slip length from the riblet formula (only when both
                # riblet parameters exist; other fluid mechanisms keep the model's coefficient).
                h = sanitized_params["riblet_height"]
                s = sanitized_params["riblet_spacing"]
                # Recalculate lambda (dimensional)
                recalculated_coeff = 0.2 * s * (1.0 - math.exp(-2.0 * h / s))
                if s < 0.005:
                    char_height = 10.0 * s
                    recalculated_coeff = recalculated_coeff / char_height
                coeff_source = "Riblet-Geometrie (lambda = 0.2 s (1 - exp(-2 h / s)))"
            else:
                recalculated_coeff = extract_param(sanitized_params, ["slip_length", "slippage_coefficient"], audited.simulation_coefficient)
                coeff_source = "slip_length"
        elif "simulation_coefficient" in overrides:
            recalculated_coeff = overrides["simulation_coefficient"]
            coeff_source = "Override simulation_coefficient"
        else:
            # A coefficient that is simply one of the audited parameters follows that
            # parameter when an override or bound clamp changed it.
            tracked = _tracked_parameter(audited.simulation_coefficient, model_params)
            if tracked and not _close(sanitized_params.get(tracked, model_params[tracked]), model_params[tracked]):
                recalculated_coeff = sanitized_params[tracked]
                coeff_source = f"Parameter {tracked}"
        if not _close(recalculated_coeff, audited.simulation_coefficient):
            changes.append(
                f"simulation_coefficient: {_fmt(audited.simulation_coefficient)} -> {_fmt(recalculated_coeff)} (neu berechnet aus {coeff_source})"
            )

        # Reconstruct list counterparts for output validation
        audited_params_list = [
            AuditedParameter(name=k, value=v) for k, v in sanitized_params.items()
        ]
        # Keep the model's dimensionless numbers. A Reynolds number is only added when it
        # can be computed from actual parameters and the model did not give one.
        dimensionless_list = list(audited.dimensionless_numbers or [])
        if not any(_is_reynolds_name(item.name) for item in dimensionless_list):
            reynolds = compute_reynolds_number(sanitized_params)
            if reynolds is not None:
                re_value, re_inputs = reynolds
                dimensionless_list.append(DimensionlessNumber(name="ReynoldsNumber", value=re_value))
                changes.append(f"ReynoldsNumber = {_fmt(re_value)} ergänzt (aus {re_inputs})")

        # Dynamic UI metadata fallback
        ui_meta = audited.ui_metadata.model_dump() if audited.ui_metadata else {}
        final_ui_meta = _default_ui_metadata(matched_domain, miner_output)
        if ui_meta:
            for k, v in ui_meta.items():
                if isinstance(v, dict) and k in final_ui_meta:
                    final_ui_meta[k].update(v)
                else:
                    final_ui_meta[k] = v

        objective_contract = audited.objective_metric or build_objective_contract(final_ui_meta)

        solver_method = audited.solver_method or "pinn"
        if user_query and any(k in user_query.lower() for k in ["script", "skript", "dynamic solver", "custom solver"]):
            solver_method = "dynamic_script"
        if solver_method != (audited.solver_method or ""):
            changes.append(f"solver_method: {audited.solver_method or 'leer'} -> {solver_method}")

        request_feasible = audited.request_feasible is not False
        audit_notes = audited.audit_notes or ""
        # Only a passed audit reports corrections; for a rejected one they are irrelevant.
        if changes and audited.audit_passed and request_feasible:
            audit_notes = f"{audit_notes}\nProgrammatische Korrekturen: " + "; ".join(changes)

        result = AuditorOutput(
            audit_passed=audited.audit_passed and request_feasible,
            audit_notes=audit_notes,
            request_feasible=request_feasible,
            infeasibility_reason=audited.infeasibility_reason if not request_feasible else None,
            audited_parameters=audited_params_list,
            dimensionless_numbers=dimensionless_list,
            simulation_coefficient=recalculated_coeff,
            solver_method=solver_method,
            ui_metadata=final_ui_meta,
            objective_metric=objective_contract
        )
        return _with_metric_and_baseline(result, audited)

    def mock_audit_design(self, miner_output: MinerOutput, override_parameters: dict = None, user_query: str = None) -> AuditorOutput:
        """
        Returns a mock AuditorOutput for testing without active API credentials.
        """
        params = miner_output.proposed_parameters.copy()
        
        # Explicitly apply user overrides if provided
        if override_parameters:
            for k, v in override_parameters.items():
                params[k] = v

        v = extract_param(params, ["free_stream_velocity", "design_flow_velocity", "velocity", "U_ski"], 1.5)
        mu = extract_param(params, ["viscosity", "kinematic_viscosity"], 0.001)

        if mu < 1e-4:
            re = (v * 1.0) / mu
        else:
            re = (1000.0 * v * 1.0) / mu

        # Check if slip length is explicitly overridden
        slip_keys = ["slip_length", "slippage_coefficient", "lambda", "simulation_coefficient"]
        overridden_slip = None
        if override_parameters:
            for k in slip_keys:
                if k in override_parameters:
                    overridden_slip = override_parameters[k]
                    break

        if overridden_slip is not None:
            coeff = overridden_slip
            if "riblet_spacing" in params or "spacing" in params:
                s = extract_param(params, ["riblet_spacing", "spacing"], 0.03)
                if s < 0.005:
                    char_height = 10.0 * s
                    coeff = coeff / char_height
        elif "riblet_height" in params or "riblet_spacing" in params:
            h = extract_param(params, ["riblet_height", "height"], 0.015)
            s = extract_param(params, ["riblet_spacing", "spacing"], 0.03)
            coeff = 0.2 * s * (1.0 - math.exp(-2.0 * h / s))
            if s < 0.005:
                char_height = 10.0 * s
                coeff = coeff / char_height
        else:
            coeff = extract_param(params, ["slip_length", "slippage_coefficient"], 0.0)

        audited_params_list = [
            AuditedParameter(name=k, value=v) for k, v in params.items()
        ]
        dimensionless_list = [
            DimensionlessNumber(name="ReynoldsNumber", value=re)
        ]

        solver_method = "pinn"
        if user_query and any(k in user_query.lower() for k in ["script", "skript", "dynamic solver", "custom solver"]):
            solver_method = "dynamic_script"

        final_ui_meta = {
            "domain_name": "Fluid Dynamics",
            "independent_var": { "label": "Channel Height", "unit": "m" },
            "dependent_var": { "label": "Flow Velocity", "unit": "m/s" },
            "primary_metric": { "label": "PINN Wall Shear Stress" },
            "reference_metric": { "label": "Analytical Wall Shear Stress" },
            "performance_gain": { "label": "Drag Reduction Efficiency" }
        }

        return AuditorOutput(
            audit_passed=True,
            audit_notes="Mock audit: proposed parameters are within safe operational bounds. Analytical dimensionless numbers and boundary conditions validated.",
            audited_parameters=audited_params_list,
            dimensionless_numbers=dimensionless_list,
            simulation_coefficient=coeff,
            solver_method=solver_method,
            ui_metadata=final_ui_meta,
            objective_metric=build_objective_contract(final_ui_meta)
        )


def _with_metric_and_baseline(result: AuditorOutput, audited: AuditorOutput) -> AuditorOutput:
    """Carries the model's metric spec and baseline design over to the post-processed output unchanged."""
    result.metric = getattr(audited, "metric", None)
    result.baseline_parameters = getattr(audited, "baseline_parameters", None)
    result.baseline_description = getattr(audited, "baseline_description", None)
    return result


def build_objective_contract(ui_metadata: dict) -> ObjectiveMetricContract:
    primary = (ui_metadata.get("primary_metric") or {}).get("label", "Primary Metric")
    reference = (ui_metadata.get("reference_metric") or {}).get("label", "Reference Metric")
    objective = (ui_metadata.get("performance_gain") or {}).get("label", "Performance Gain")
    primary_lower = primary.lower()
    lower_is_better = any(term in primary_lower for term in [
        "stress", "friction", "drag", "loss", "risk", "error", "damage", "shear"
    ])
    return ObjectiveMetricContract(
        objective_name=objective,
        score_field="performance_gain_pct",
        direction="maximize",
        primary_metric=primary,
        reference_metric=reference,
        lower_is_better=lower_is_better,
        acceptance_threshold=0.0,
        hard_constraints=[
            "relative_error <= 1.0",
            "parameters within bounds",
            "finite numeric outputs",
            "validation_passed == true for generated scripts",
        ],
    )
