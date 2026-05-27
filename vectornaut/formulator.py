# -*- coding: utf-8 -*-
import json
from .config import get_client, MinerConceptOutput, ModelFormulation, MinerOutput

class ModelFormulator:
    def __init__(self, client=None, thinking_level: str = "medium"):
        """
        Initializes the ModelFormulator agent.
        The thinking_level can be set to "low", "medium", or "high" to customize Gemini's planning depth.
        """
        self.client = client
        self.thinking_level = thinking_level

    def formulate_model(self, query: str, concept: MinerConceptOutput, thinking_level: str = None) -> MinerOutput:
        """
        Formulates a rigorous, mathematically consistent physical model (ODE/PDE, boundary conditions, vars)
        based on the provided bionic concept using gemini-3.5-flash with thinking.
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
            prompt += f"- {p.name}: value={p.value}, bounds=[{p.min_bound}, {p.max_bound}]\n"
            
        prompt += """
        CRITICAL RULES FOR MATHEMATICAL SOLVABILITY:
        1. PARAMETER CONSISTENCY & SELF-EXTENSION:
           - Every single symbol used in your governing equation or boundary conditions (besides the independent variables and the dependent variable) MUST be a parameter.
           - If your physical model requires boundary values, external temperatures, concentrations, or source terms that are NOT present in the Proposed Parameters list above (e.g. 'C_body', 'C_ambient', 'T_hot', 'T_cold', etc.), you MUST define and append them as new ParameterProposal objects in the 'parameters' field in the returned JSON!
           - Do not use any undefined symbols. All symbols in equations must exist in the final parameters list.
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
           - Keep the differential equations as simple and linear as possible to ensure convergence of symbolic (SymPy) and numerical (SciPy, PyTorch PINN) solvers.
           - For 1D: Use the format 'd2[dep]_d[ind]2 = RHS', where RHS is a linear combination of parameters (and optionally [dep] or d[dep]_d[ind]).
             Example: 'd2u_dy2 = -pressure_gradient / viscosity'
             Example: 'd2T_dy2 = heat_source / thermal_conductivity'
        """

        client = self.client or get_client()
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level=active_thinking_level
                ),
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
        
        # Combine the Concept and the Formulation into MinerOutput for full pipeline compatibility
        return MinerOutput(
            design_name=concept.design_name,
            inspiration_source=concept.inspiration_source,
            domain=concept.domain,
            physical_mechanism=concept.physical_mechanism,
            parameters=concept.parameters,
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
