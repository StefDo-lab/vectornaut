# -*- coding: utf-8 -*-
import json
from typing import List, Optional, Tuple

from .config import get_client, get_model_name, get_thinking_config, MinerConceptOutput, ModelFormulation, MinerOutput, ParameterProposal


def _fmt(value) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def merge_formulated_parameters(
    proposed: List[ParameterProposal],
    formulated: Optional[List[ParameterProposal]],
) -> Tuple[List[ParameterProposal], List[str]]:
    """
    Merges the formulator's `parameters` into the concept's proposed parameters.
    New names are appended; an existing name takes the value, bounds and justification the
    formulator gave for it; proposed parameters the formulator did not list stay unchanged.
    Returns (merged parameters, human-readable list of changes).
    """
    merged = [p.model_copy() for p in proposed or []]
    index = {p.name: i for i, p in enumerate(merged)}
    changes: List[str] = []
    for item in formulated or []:
        name = str(getattr(item, "name", "") or "").strip()
        if not name:
            continue
        item = item.model_copy(update={"name": name})
        if name not in index:
            merged.append(item)
            index[name] = len(merged) - 1
            changes.append(
                f"{name} added (value={_fmt(item.value)}, bounds=[{_fmt(item.min_bound)}, {_fmt(item.max_bound)}])"
            )
            continue
        current = merged[index[name]]
        updates = {}
        diffs = []
        if item.value != current.value:
            updates["value"] = item.value
            diffs.append(f"value {_fmt(current.value)} -> {_fmt(item.value)}")
        if item.min_bound != current.min_bound or item.max_bound != current.max_bound:
            updates["min_bound"] = item.min_bound
            updates["max_bound"] = item.max_bound
            diffs.append(
                f"bounds [{_fmt(current.min_bound)}, {_fmt(current.max_bound)}] -> [{_fmt(item.min_bound)}, {_fmt(item.max_bound)}]"
            )
        justification = (item.justification or "").strip()
        if justification and justification != (current.justification or "").strip():
            updates["justification"] = item.justification
            diffs.append("justification updated")
        if updates:
            merged[index[name]] = current.model_copy(update=updates)
            changes.append(f"{name}: " + ", ".join(diffs))
    return merged, changes

class ModelFormulator:
    def __init__(self, client=None, thinking_level: str = None):
        """
        Initializes the ModelFormulator agent.
        The thinking_level can be set to "low", "medium", or "high" to customize Gemini's planning depth.
        If omitted, VECTORNAUT_THINKING_FORMULATOR is used, falling back to "medium".
        """
        self.client = client
        self.thinking_level = thinking_level
        # Changes the last formulate_model call made to the concept's parameters.
        self.last_parameter_changes: List[str] = []

    def formulate_model(self, query: str, concept: MinerConceptOutput, thinking_level: str = None) -> MinerOutput:
        """
        Formulates a rigorous, mathematically consistent physical model (ODE/PDE, boundary conditions, vars)
        based on the provided bionic concept using the configured formulator model with thinking.
        An explicit thinking_level (call argument, then constructor) wins over VECTORNAUT_THINKING_FORMULATOR.
        """
        from google.genai import types

        active_thinking_level = thinking_level or self.thinking_level
        
        prompt = f"""
        You are the Model Formulator for Vectornaut. Your job is to translate a biomimetic concept and its parameters into a rigorous, mathematically consistent and SOLVABLE physical model represented as a second-order 1D boundary value problem (ODE) or 2D partial differential equation (PDE).

        Here is the design request that initiated the search:
        "{query}"

        Here is the biomimetic concept proposed by the Miner:
        - Design Name: {concept.design_name}
        - Inspiration Source: {concept.inspiration_source}
        - Domain: {concept.domain}
        - Physical Mechanism: {concept.physical_mechanism}
        
        Proposed Parameters:
        """
        
        for p in concept.parameters:
            prompt += f"- {p.name}: value={p.value}, bounds=[{p.min_bound}, {p.max_bound}]"
            justification = (p.justification or "").strip()
            if justification:
                prompt += f" -- {justification}"
            prompt += "\n"

        prompt += """
        CRITICAL RULES FOR MATHEMATICAL SOLVABILITY:
        1. PARAMETER CONSISTENCY & SELF-EXTENSION:
           - Every single symbol used in your governing equation or boundary conditions (besides the independent variables and the dependent variable) MUST be a parameter: either one of the Proposed Parameters above or one you return in the 'parameters' field.
           - The 'parameters' field of the returned JSON is a list of ParameterProposal objects (name, value, min_bound, max_bound, justification). Put in it ONLY:
             a) every NEW parameter your model needs that is NOT in the Proposed Parameters list, e.g. boundary values, external temperatures, concentrations, loads or source terms ('C_body', 'C_ambient', 'T_hot', 'T_cold', ...), with a physically justified value and bounds;
             b) an EXISTING parameter whose value, bounds or justification you need to change (e.g. widen bounds that are too narrow for the model), under exactly the same name and with all fields filled in.
           - Proposed parameters you do not list are kept unchanged. Leave 'parameters' empty if nothing needs to be added or changed.
           - Do not use any undefined symbols: a symbol that is neither proposed nor returned in 'parameters' is filled in with an unaudited default value (usually 1.0).
        2. 1D BOUNDARY CONDITIONS (BVP NORMALIZATION):
           - If 1D, you must provide EXACTLY TWO boundary conditions: one evaluated at the numerical value 0 and one evaluated at the numerical value 1.
           - Example format for independent variable 'y' and dependent variable 'u':
             ['u(0) = 0', 'u(1) = free_stream_velocity']
           - NEVER evaluate boundary conditions at a parameter name like 'u(crypt_depth) = ...' or 'u(h) = ...'. Always use the normalized unit interval [0, 1] and evaluate at the literal numbers '0' and '1'.
           - DO NOT write initial value problems at a single point (e.g., do not write 'w(0) = 0' and 'dw_dx(0) = 0'). The boundary coordinates must be on opposite boundaries (0 and 1).
        3. 2D BOUNDARY CONDITIONS:
           - If 2D, you must provide exactly four boundary conditions at the edges of the unit domain [0, 1] x [0, 1]: for x=0, x=1, y=0, y=1.
           - Example for independent variables ['x', 'y'] and dependent variable 'T':
             ['T(0, y) = T_hot', 'T(1, y) = T_cold', 'dT_dy(x, 0) = 0', 'dT_dy(x, 1) = 0']
        4. SIMPLICITY & LINEARITY:
           - Keep the differential equations as simple as possible to ensure convergence of symbolic (SymPy) and numerical (SciPy, PyTorch PINN, FDM) solvers.
           - The equation must be LINEAR in the dependent variable and its derivatives (no products such as u*du_dx, no u**2, no sin(u)).
           - For 1D: Use the format 'd2[dep]_d[ind]2 = RHS'. The RHS may contain [dep] and d[dep]_d[ind] multiplied by parameter expressions, plus a source term that may depend on the independent variable and the parameters (polynomials, exp, sin, ... of [ind] are fine).
             Example: 'd2u_dy2 = -pressure_gradient / viscosity'
             Example: 'd2T_dy2 = heat_source / thermal_conductivity'
             Example (beam bending moment under a uniform load on the normalized span): 'd2w_dx2 = -uniform_load * span_length**4 * x * (1 - x) / (2 * youngs_modulus * second_moment_of_area)'
           - For 2D: any linear second-order elliptic operator is supported, e.g. 'k_x * d2T_dx2 + k_y * d2T_dy2 = -heat_source'; coefficients and the source term may depend on x, y and the parameters.
        """

        client = self.client or get_client()
        response = client.models.generate_content(
            model=get_model_name("formulator"),
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=get_thinking_config("formulator", default="medium", override=active_thinking_level),
                response_mime_type="application/json",
                response_schema=ModelFormulation,
            )
        )
        
        formulation = response.parsed
        
        print(f"\n[*] ModelFormulator output:")
        print(f"    Governing: {formulation.governing_equation}")
        print(f"    Boundary Conditions: {formulation.boundary_conditions}")
        print(f"    Independent Variables: {formulation.independent_variables}")
        print(f"    Dependent Variables: {formulation.dependent_variables}")
        print(f"    Concept Parameters: {[p.name for p in concept.parameters]}")

        # Parameters the formulator added or changed (new boundary values, loads, widened bounds).
        parameters, changes = merge_formulated_parameters(concept.parameters, getattr(formulation, "parameters", None))
        self.last_parameter_changes = changes
        for change in changes:
            print(f"    Formulator parameter change: {change}")

        # Combine the Concept and the Formulation into MinerOutput for full pipeline compatibility
        return MinerOutput(
            design_name=concept.design_name,
            inspiration_source=concept.inspiration_source,
            domain=concept.domain,
            physical_mechanism=concept.physical_mechanism,
            parameters=parameters,
            svg_schematic=concept.svg_schematic,
            governing_equation=formulation.governing_equation,
            boundary_conditions=formulation.boundary_conditions,
            independent_variables=formulation.independent_variables,
            dependent_variables=formulation.dependent_variables
        )

    def mock_formulate_model(self, query: str, concept: MinerConceptOutput) -> MinerOutput:
        """
        Returns a mock formulated model (MinerOutput) based on the concept name
        to keep integration tests running without active credentials.
        """
        name_lower = concept.design_name.lower()
        
        # Avoid matching 'skin' as 'ski'
        is_ski_related = "ski " in name_lower or "skibelag" in name_lower or "ski-base" in name_lower
        if "plastron" in name_lower or is_ski_related or "collembola" in name_lower:
            gov_eq = "d2u_dy2 = pressure_gradient * film_thickness^2 / viscosity"
            bcs = [
                "u(0) = (slip_length / film_thickness) * du_dy(0)",
                "u(1) = ski_velocity"
            ]
            ind_vars = ["y"]
            dep_vars = ["u"]
        else:
            # Default to Shark-Skin / Riblet
            gov_eq = "d2u_dy2 = -pressure_gradient / viscosity"
            bcs = [
                "u(0) = slippage_coefficient * du_dy(0)",
                "u(1) = free_stream_velocity"
            ]
            ind_vars = ["y"]
            dep_vars = ["u"]
            
        return MinerOutput(
            design_name=concept.design_name,
            inspiration_source=concept.inspiration_source,
            domain=concept.domain,
            physical_mechanism=concept.physical_mechanism,
            parameters=concept.parameters,
            svg_schematic=concept.svg_schematic,
            governing_equation=gov_eq,
            boundary_conditions=bcs,
            independent_variables=ind_vars,
            dependent_variables=dep_vars
        )
