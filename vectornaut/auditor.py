import math
from google import genai
from google.genai import types
from .config import get_client, MinerOutput, AuditorOutput, AuditedParameter, DimensionlessNumber

def extract_param(params_dict, keys, default):
    for key in keys:
        if key in params_dict:
            return params_dict[key]
    return default

class Auditor:
    def __init__(self, client: genai.Client = None):
        self.client = client or get_client()

    def audit_design(self, miner_output: MinerOutput, override_parameters: dict = None, user_query: str = None) -> AuditorOutput:
        """
        Audits proposed parameters against physical constraints.
        Enforces thinking_level="high" for deep physical reasoning.
        """
        prompt = f"""
        You are a senior physical auditor verifying design specifications for a biomimetic material.
        Here is the proposal from the Miner stage:
        
        Design Name: {miner_output.design_name}
        Original User Query: {user_query or "None"}
        Inspiration Source: {miner_output.inspiration_source}
        Domain: {miner_output.domain}
        Physical Mechanism: {miner_output.physical_mechanism}
        Proposed Parameters: {miner_output.proposed_parameters}
        Suggested Bounds: {miner_output.suggested_bounds}
        Parameter Justifications: {miner_output.parameter_justifications}
        Governing Equation: {miner_output.governing_equation}
        Boundary Conditions: {miner_output.boundary_conditions}

        Perform a rigorous audit of these parameters:
        1. Check if the parameter ranges make physical sense (e.g., density, viscosity, spacing, height must be positive).
        2. Adjust any parameters that are out of bounds or physically unrealistic (e.g., riblet spacing should be larger than or equal to height, viscosity must be reasonable).
        3. Calculate key dimensionless numbers (e.g., Reynolds number assuming density = 1000 kg/m^3 for water and L = 1.0 m, or relevant thermal/electromagnetic dimensionless numbers).
        4. Derive a simulation coefficient used directly in the simulator boundary/PDE equations.
           - For Fluid Dynamics (drag reduction), calculate the slippage length lambda:
             lambda = 0.2 * riblet_spacing * (1 - exp(-2.0 * riblet_height / riblet_spacing))
           - For Thermodynamics or Electromagnetics, determine a corresponding physical/design parameter and return it as the 'simulation_coefficient'.
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

        Output the results structured strictly according to the AuditorOutput schema.
        """

        response = self.client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="high"
                ),
                response_mime_type="application/json",
                response_schema=AuditorOutput,
            )
        )
        audited: AuditorOutput = response.parsed

        # Programmatic safety guardrails applied on top of LLM outputs:
        sanitized_params = audited.audited_parameters_dict.copy()
        
        # Explicitly apply user overrides if provided
        if override_parameters:
            for k, v in override_parameters.items():
                sanitized_params[k] = v

        suggested_bounds = miner_output.suggested_bounds
        for param, val in sanitized_params.items():
            if param in suggested_bounds:
                bounds = suggested_bounds[param]
                if len(bounds) == 2:
                    min_val, max_val = bounds[0], bounds[1]
                    if val < min_val:
                        val = min_val
                    elif val > max_val:
                        val = max_val
                    sanitized_params[param] = val
        
        # Enforce positive values for key dimensions
        for key in ["viscosity", "kinematic_viscosity", "riblet_height", "riblet_spacing", "free_stream_velocity", "design_flow_velocity", "conductivity"]:
            if key in sanitized_params:
                sanitized_params[key] = max(1e-6, sanitized_params[key])

        # Domain-aware recalculations and fallbacks
        domain = miner_output.domain.lower()
        recalculated_coeff = audited.simulation_coefficient
        re = 100.0 # Default dimensionless number placeholder
        
        if "fluid" in domain:
            v = extract_param(sanitized_params, ["free_stream_velocity", "design_flow_velocity", "velocity", "U_ski"], 1.5)
            mu = extract_param(sanitized_params, ["viscosity", "kinematic_viscosity"], 0.001)

            # Check if slip length is explicitly overridden
            slip_keys = ["slip_length", "slippage_coefficient", "lambda", "simulation_coefficient"]
            overridden_slip = None
            if override_parameters:
                for k in slip_keys:
                    if k in override_parameters:
                        overridden_slip = override_parameters[k]
                        break

            if overridden_slip is not None:
                recalculated_coeff = overridden_slip
                if "riblet_spacing" in sanitized_params or "spacing" in sanitized_params:
                    s = extract_param(sanitized_params, ["riblet_spacing", "spacing"], 0.03)
                    if s < 0.005:
                        char_height = 10.0 * s
                        recalculated_coeff = recalculated_coeff / char_height
            elif "riblet_height" in sanitized_params or "riblet_spacing" in sanitized_params:
                h = extract_param(sanitized_params, ["riblet_height", "height"], 0.015)
                s = extract_param(sanitized_params, ["riblet_spacing", "spacing"], 0.03)
                # Recalculate lambda (dimensional)
                recalculated_coeff = 0.2 * s * (1.0 - math.exp(-2.0 * h / s))
                if s < 0.005:
                    char_height = 10.0 * s
                    recalculated_coeff = recalculated_coeff / char_height
            else:
                recalculated_coeff = extract_param(sanitized_params, ["slip_length", "slippage_coefficient"], audited.simulation_coefficient)

            # Calculate Reynolds number dynamically based on viscosity type
            if mu < 1e-4:
                re = (v * 1.0) / mu
            else:
                re = (1000.0 * v * 1.0) / mu
                
        # Reconstruct list counterparts for output validation
        audited_params_list = [
            AuditedParameter(name=k, value=v) for k, v in sanitized_params.items()
        ]
        dimensionless_list = [
            DimensionlessNumber(name="ReynoldsNumber" if "fluid" in domain else "DimensionlessParameter", value=re)
        ]

        # Dynamic UI metadata fallback
        ui_meta = audited.ui_metadata.model_dump() if audited.ui_metadata else {}
        default_meta = {
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
            }
        }
        
        # Match domain for fallback
        matched_domain = "fluid"
        for k in default_meta:
            if k in domain:
                matched_domain = k
                break
                
        final_ui_meta = default_meta[matched_domain].copy()
        if ui_meta:
            for k, v in ui_meta.items():
                if isinstance(v, dict) and k in final_ui_meta:
                    final_ui_meta[k].update(v)
                else:
                    final_ui_meta[k] = v

        solver_method = audited.solver_method or "pinn"
        if user_query and any(k in user_query.lower() for k in ["script", "skript", "dynamic solver", "custom solver"]):
            solver_method = "dynamic_script"

        return AuditorOutput(
            audit_passed=True,
            audit_notes=audited.audit_notes + " (Programmatic safety checks applied.)",
            audited_parameters=audited_params_list,
            dimensionless_numbers=dimensionless_list,
            simulation_coefficient=recalculated_coeff,
            solver_method=solver_method,
            ui_metadata=final_ui_meta
        )

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

        return AuditorOutput(
            audit_passed=True,
            audit_notes="Mock audit: proposed parameters are within safe operational bounds. Analytical dimensionless numbers and boundary conditions validated.",
            audited_parameters=audited_params_list,
            dimensionless_numbers=dimensionless_list,
            simulation_coefficient=coeff,
            solver_method=solver_method,
            ui_metadata={
                "domain_name": "Fluid Dynamics",
                "independent_var": { "label": "Channel Height", "unit": "m" },
                "dependent_var": { "label": "Flow Velocity", "unit": "m/s" },
                "primary_metric": { "label": "PINN Wall Shear Stress" },
                "reference_metric": { "label": "Analytical Wall Shear Stress" },
                "performance_gain": { "label": "Drag Reduction Efficiency" }
            }
        )
