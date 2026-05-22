from google import genai
from google.genai import types
from .config import get_client, MinerOutput, ParameterProposal

class Miner:
    def __init__(self, client: genai.Client = None):
        self.client = client or get_client()

    def mine_design(self, query: str) -> MinerOutput:
        """
        Explores bionics/materials spaces and outputs structured design targets.
        Uses gemini-3.5-flash with thinking_level="medium".
        """
        prompt = f"""
        Explore the semantic and physical space of biomimetic material design for the following request:
        "{query}"

        Analyze natural systems (e.g., plants, animals, structures, physics) that solve this physical problem.
        Identify a specific natural inspiration source, describe the underlying physical mechanism, and extract concrete physical parameters that can be modeled.
        Propose key parameters (such as height, spacing, viscosity, velocity, conductivity, density, etc.) with suggested initial values and reasonable physical bounds [min, max] that can be audited.

        In addition, formulate a 1D or 2D second-order differential equation (BVP or PDE) that models this system:
        - Decide if the system is best modeled in 1D (a single coordinate like y, e.g., velocity across a channel) or 2D (spatial coordinates x and y over a cross-section, e.g., temperature profile in a plate or electrostatic field in a 2D coaxial geometry). Choose 2D if the query explicitly mentions 2D, or if the physical setup inherently requires two independent dimensions.
        
        1. governing_equation:
           - For 1D: A second-order ODE in the format 'd2[dep]_d[ind]2 = RHS', where [dep] is the dependent variable (e.g., u, T, V) and [ind] is the independent variable (e.g., y, x).
             Example: 'd2u_dy2 = -pressure_gradient / viscosity'
           - For 2D: A second-order PDE in the format 'd2[dep]_d[ind1]2 + d2[dep]_d[ind2]2 = RHS', where [dep] is the dependent variable (e.g., T, V) and [ind1], [ind2] are the two independent variables (e.g., x, y).
             Example for 2D heat conduction: 'd2T_dx2 + d2T_dy2 = 0'
             Example for 2D Poisson electrostatics: 'd2V_dx2 + d2V_dy2 = -charge_density / permittivity'
             Ensure all symbols in RHS are defined in your proposed parameters.
        2. boundary_conditions: A list of symbolic boundary equations at the boundaries.
           - For 1D: exactly two boundary conditions at 0 and 1. Example: ['T(0) = T_hot', 'T(1) = T_cold']
           - For 2D: boundary conditions at the edges of the unit domain [0, 1] x [0, 1]. Provide boundary conditions for the four edges: x=0, x=1, y=0, y=1.
             Example: ['T(0, y) = T_hot', 'T(1, y) = T_cold', 'dT_dy(x, 0) = 0', 'dT_dy(x, 1) = 0'] (which represent hot/cold walls and insulated bottom/top).
        3. independent_variables: A list containing the independent variables:
           - For 1D: a single element, e.g., ['y'] or ['x'].
           - For 2D: two elements, e.g., ['x', 'y'].
             Make sure it matches the governing equation and BCs.
        4. dependent_variables: A list containing the single dependent variable, e.g., ['u'] or ['T'] or ['V'].
             Make sure it matches the governing equation and BCs.
        5. svg_schematic:
           - A clean, self-contained SVG code snippet (using <svg viewBox="0 0 400 200" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">) illustrating the proposed microstructure.
           - Do NOT wrap this field in markdown code blocks inside the JSON; it must be a raw string. All inner double quotes must be properly escaped if necessary.
           - Draw a clean 2D cross-section or diagram showing the geometry of the proposed microstructure (e.g., V-shaped riblets, micro-posts, honeycomb pores, fiber grids) scaled visually to match your proposed parameter values.
           - The style must fit a premium dark-themed dashboard. Use a dark background (fill="#0f172a") with rounded corners (rx="8" ry="8") or transparent, and draw using neon purple (#7b2cbf), neon cyan (#00f5d4), glowing white, and subtle grays.
           - Add technical annotations like dimension lines, arrows, and parameter labels (e.g. 's' or 'spacing' for spacing, 'h' or 'height' for height, etc.) to show how the proposed parameters map to the geometry. Make it look like a high-tech scientific blueprint.
        """

        response = self.client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="medium"
                ),
                response_mime_type="application/json",
                response_schema=MinerOutput,
            )
        )
        return response.parsed

    def mock_mine_design(self, query: str) -> MinerOutput:
        """
        Returns a mock MinerOutput for testing without active API credentials.
        """
        q_lower = query.lower()
        if any(keyword in q_lower for keyword in ["plastron", "ski", "collembola"]):
            return MinerOutput(
                design_name="PlastronGlide Hydrophobic Ski Base",
                inspiration_source="Collembola cuticle (springtail)",
                domain="Fluid Dynamics",
                physical_mechanism="The hierarchical micro- and nanostructures of the Collembola cuticle trap a persistent layer of air (plastron) when in contact with water. This plastron layer changes the boundary condition at the liquid-solid interface from no-slip to shear-free (slip), significantly reducing viscous drag in the meltwater film under the ski.",
                parameters=[
                    ParameterProposal(
                        name="film_thickness",
                        value=0.00001,
                        min_bound=0.000001,
                        max_bound=0.0001,
                        justification="Thickness of the liquid meltwater film generated by friction between the ski and snow."
                    ),
                    ParameterProposal(
                        name="slip_length",
                        value=0.00002,
                        min_bound=0.000001,
                        max_bound=0.0001,
                        justification="Effective Navier slip length enabled by the trapped plastron within the Collembola-inspired structures."
                    ),
                    ParameterProposal(
                        name="viscosity",
                        value=0.00179,
                        min_bound=0.001,
                        max_bound=0.002,
                        justification="Dynamic viscosity of water near 0 degrees Celsius."
                    ),
                    ParameterProposal(
                        name="pressure_gradient",
                        value=-10000.0,
                        min_bound=-1000000.0,
                        max_bound=0.0,
                        justification="Pressure gradient driving the fluid flow in the meltwater film."
                    ),
                    ParameterProposal(
                        name="ski_velocity",
                        value=10.0,
                        min_bound=0.0,
                        max_bound=50.0,
                        justification="Relative velocity of the ski base with respect to the snow surface."
                    )
                ],
                governing_equation="d2u_dy2 = pressure_gradient * film_thickness^2 / viscosity",
                boundary_conditions=[
                    "u(0) = (slip_length / film_thickness) * du_dy(0)",
                    "u(1) = ski_velocity"
                ],
                independent_variables=["y"],
                dependent_variables=["u"],
                svg_schematic='''<svg viewBox="0 0 400 200" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="200" rx="8" ry="8" fill="#0f172a" />
  <path d="M 0,50 L 400,50 M 0,100 L 400,100 M 0,150 L 400,150 M 100,0 L 100,200 M 200,0 L 200,200 M 300,0 L 300,200" stroke="#1e293b" stroke-width="1" stroke-dasharray="4 4" />
  <rect x="0" y="160" width="400" height="40" fill="#1e1b4b" stroke="#7b2cbf" stroke-width="2" />
  <rect x="50" y="110" width="20" height="50" fill="#7b2cbf" opacity="0.8" />
  <circle cx="60" cy="110" r="12" fill="#7b2cbf" />
  <rect x="150" y="110" width="20" height="50" fill="#7b2cbf" opacity="0.8" />
  <circle cx="160" cy="110" r="12" fill="#7b2cbf" />
  <rect x="250" y="110" width="20" height="50" fill="#7b2cbf" opacity="0.8" />
  <circle cx="260" cy="110" r="12" fill="#7b2cbf" />
  <rect x="350" y="110" width="20" height="50" fill="#7b2cbf" opacity="0.8" />
  <circle cx="360" cy="110" r="12" fill="#7b2cbf" />
  <rect x="72" y="110" width="66" height="50" fill="rgba(123, 44, 191, 0.15)" />
  <rect x="172" y="110" width="66" height="50" fill="rgba(123, 44, 191, 0.15)" />
  <rect x="272" y="110" width="66" height="50" fill="rgba(123, 44, 191, 0.15)" />
  <text x="105" y="140" fill="#7b2cbf" font-size="10" font-family="monospace" text-anchor="middle" opacity="0.6">Air Plastron</text>
  <text x="205" y="140" fill="#7b2cbf" font-size="10" font-family="monospace" text-anchor="middle" opacity="0.6">Air Plastron</text>
  <text x="305" y="140" fill="#7b2cbf" font-size="10" font-family="monospace" text-anchor="middle" opacity="0.6">Air Plastron</text>
  <path d="M 0,98 Q 60,105 100,98 T 200,98 T 300,98 T 400,98" fill="none" stroke="#00f5d4" stroke-width="3" />
  <rect x="0" y="40" width="400" height="58" fill="rgba(0, 245, 212, 0.08)" />
  <text x="20" y="65" fill="#00f5d4" font-size="12" font-family="sans-serif" font-weight="bold">Water Flow (Meltwater)</text>
  <line x1="160" y1="80" x2="260" y2="80" stroke="#f1f3f9" stroke-width="1" />
  <polygon points="160,80 165,77 165,83" fill="#f1f3f9" />
  <polygon points="260,80 255,77 255,83" fill="#f1f3f9" />
  <text x="210" y="75" fill="#f1f3f9" font-size="11" font-family="monospace" text-anchor="middle">s (spacing)</text>
  <line x1="28" y1="110" x2="28" y2="160" stroke="#f1f3f9" stroke-width="1" />
  <polygon points="28,110 25,115 31,115" fill="#f1f3f9" />
  <polygon points="28,160 25,155 31,155" fill="#f1f3f9" />
  <text x="20" y="138" fill="#f1f3f9" font-size="11" font-family="monospace" text-anchor="end">h (height)</text>
  <path d="M 0,98 L 400,98" stroke="#00f5d4" stroke-dasharray="2 2" />
  <line x1="385" y1="98" x2="385" y2="120" stroke="#00f5d4" stroke-width="1" />
  <polygon points="385,98 382,103 388,103" fill="#00f5d4" />
  <text x="378" y="115" fill="#00f5d4" font-size="11" font-family="monospace" text-anchor="end">λ (slip length)</text>
</svg>'''
            )

        return MinerOutput(
            design_name="Shark-Skin Inspired Riblet Foil",
            inspiration_source="Galeocerdo cuvier (Tiger Shark)",
            domain="Fluid Dynamics",
            physical_mechanism="Micro-grooves aligned with flow direction reduce viscous drag by lifting turbulent vortices off the wall.",
            parameters=[
                ParameterProposal(
                    name="riblet_height",
                    value=0.015,
                    min_bound=0.001,
                    max_bound=0.1,
                    justification="Optimal height to stay inside the viscous sublayer."
                ),
                ParameterProposal(
                    name="riblet_spacing",
                    value=0.03,
                    min_bound=0.005,
                    max_bound=0.2,
                    justification="Optimizes vortex spacing control, typically s+ of around 15."
                ),
                ParameterProposal(
                    name="viscosity",
                    value=0.001,
                    min_bound=0.0001,
                    max_bound=0.01,
                    justification="Water viscosity at standard conditions."
                ),
                ParameterProposal(
                    name="free_stream_velocity",
                    value=1.5,
                    min_bound=0.1,
                    max_bound=5.0,
                    justification="Operating velocity for testing conditions."
                ),
                ParameterProposal(
                    name="pressure_gradient",
                    value=2.0,
                    min_bound=0.0,
                    max_bound=10.0,
                    justification="Simulates external pressure driven flow."
                )
            ],
            governing_equation="d2u_dy2 = -pressure_gradient / viscosity",
            boundary_conditions=[
                "u(0) = slippage_coefficient * du_dy(0)",
                "u(1) = free_stream_velocity"
            ],
            independent_variables=["y"],
            dependent_variables=["u"],
            svg_schematic='''<svg viewBox="0 0 400 200" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="200" rx="8" ry="8" fill="#0f172a" />
  <path d="M 0,50 L 400,50 M 0,100 L 400,100 M 0,150 L 400,150 M 100,0 L 100,200 M 200,0 L 200,200 M 300,0 L 300,200" stroke="#1e293b" stroke-width="1" stroke-dasharray="4 4" />
  <path d="M 0,160 L 20,160 L 40,110 L 60,160 L 100,160 L 120,110 L 140,160 L 180,160 L 200,110 L 220,160 L 260,160 L 280,110 L 300,160 L 340,160 L 360,110 L 380,160 L 400,160 L 400,200 L 0,200 Z" fill="#1e1b4b" stroke="#7b2cbf" stroke-width="2" />
  <rect x="0" y="30" width="400" height="80" fill="rgba(0, 245, 212, 0.05)" />
  <path d="M 0,110 C 40,100 120,100 120,110 C 120,100 200,100 200,110 C 200,100 280,100 280,110 C 280,100 360,100 360,110" fill="none" stroke="#00f5d4" stroke-width="2" stroke-dasharray="3 3" />
  <circle cx="120" cy="95" r="10" fill="none" stroke="#00f5d4" stroke-width="1.5" stroke-dasharray="2 2" />
  <path d="M 127,95 A 7,7 0 1,1 120,88" fill="none" stroke="#00f5d4" stroke-width="1.5" />
  <circle cx="280" cy="95" r="10" fill="none" stroke="#00f5d4" stroke-width="1.5" stroke-dasharray="2 2" />
  <path d="M 287,95 A 7,7 0 1,1 280,88" fill="none" stroke="#00f5d4" stroke-width="1.5" />
  <text x="200" y="55" fill="#00f5d4" font-size="12" font-family="sans-serif" font-weight="bold" text-anchor="middle">Laminar Flow / Vortex Lifting</text>
  <text x="120" y="80" fill="#00f5d4" font-size="9" font-family="sans-serif" text-anchor="middle" opacity="0.8">Lifted Vortex</text>
  <text x="280" y="80" fill="#00f5d4" font-size="9" font-family="sans-serif" text-anchor="middle" opacity="0.8">Lifted Vortex</text>
  <line x1="120" y1="125" x2="280" y2="125" stroke="#f1f3f9" stroke-width="1" />
  <polygon points="120,125 125,122 125,128" fill="#f1f3f9" />
  <polygon points="280,125 275,122 275,128" fill="#f1f3f9" />
  <text x="200" y="140" fill="#f1f3f9" font-size="11" font-family="monospace" text-anchor="middle">s (spacing)</text>
  <line x1="90" y1="110" x2="90" y2="160" stroke="#f1f3f9" stroke-width="1" />
  <polygon points="90,110 87,115 93,115" fill="#f1f3f9" />
  <polygon points="90,160 87,155 93,155" fill="#f1f3f9" />
  <text x="82" y="140" fill="#f1f3f9" font-size="11" font-family="monospace" text-anchor="end">h (height)</text>
</svg>'''
        )
