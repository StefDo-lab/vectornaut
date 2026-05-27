import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

def get_client():
    """
    Initializes and returns the unified google-genai Client.
    Uses AI Studio (GEMINI_API_KEY) by default, or Vertex AI if configured.
    """
    from google import genai

    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    if use_vertex:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        return genai.Client(vertexai=True, project=project, location=location)
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            return genai.Client(api_key=api_key)
        else:
            # The client will check GEMINI_API_KEY env variable automatically
            return genai.Client()

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

class AuditorOutput(BaseModel):
    audit_passed: bool = Field(description="True if the parameters are physically plausible and safe for simulator execution")
    audit_notes: str = Field(description="Detailed notes explaining the checks performed, reasons for success/failure, or adjustments made")
    audited_parameters: List[AuditedParameter] = Field(description="The finalized, sanitized parameters to be passed to the simulator")
    dimensionless_numbers: List[DimensionlessNumber] = Field(description="Calculated key dimensionless values for verification")
    simulation_coefficient: float = Field(description="Derived core coefficient (e.g., slip length, heat transfer coefficient) used directly in the simulator boundary/PDE equations")
    solver_method: str = Field(description="The selected solver method: 'analytical', 'scipy', or 'pinn'")
    ui_metadata: UIMetadata = Field(description="Metadata describing labels, units, and axes for dynamic UI rendering")

    @property
    def audited_parameters_dict(self) -> Dict[str, float]:
        return {p.name: p.value for p in self.audited_parameters}

    @property
    def dimensionless_numbers_dict(self) -> Dict[str, float]:
        return {p.name: p.value for p in self.dimensionless_numbers}

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
