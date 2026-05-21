import sys
import os

# Adjust path to import vectornaut
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vectornaut.config import MinerOutput, ParameterProposal, AuditorOutput, AuditedParameter, DimensionlessNumber
from vectornaut.solver_dispatcher import dispatch_and_solve

def test_plastron():
    print("Testing PlastronGlide Hydrophobic Ski Base simulation...")
    
    miner_out = MinerOutput(
        design_name="PlastronGlide Hydrophobic Ski Base",
        inspiration_source="Collembola (Springtail) cuticle",
        domain="Fluid Dynamics",
        physical_mechanism="The springtail's skin possesses a hierarchical nanostructure...",
        parameters=[
            ParameterProposal(name="slip_length", value=0.00002, min_bound=0.000001, max_bound=0.0001, justification="..."),
            ParameterProposal(name="film_thickness", value=0.00001, min_bound=0.000001, max_bound=0.00005, justification="..."),
            ParameterProposal(name="pressure_gradient", value=-1000.0, min_bound=-50000.0, max_bound=0.0, justification="..."),
            ParameterProposal(name="viscosity", value=0.00179, min_bound=0.001, max_bound=0.002, justification="..."),
            ParameterProposal(name="U_ski", value=10.0, min_bound=0.5, max_bound=40.0, justification="...")
        ],
        governing_equation="d2u_dy2 = (film_thickness^2 * pressure_gradient) / viscosity",
        boundary_conditions=[
            "u(0) = (slip_length / film_thickness) * du_dy(0)",
            "u(1) = U_ski"
        ],
        independent_variables=["y"],
        dependent_variables=["u"]
    )
    
    auditor_out = AuditorOutput(
        audit_passed=True,
        audit_notes="...",
        audited_parameters=[
            AuditedParameter(name="slip_length", value=0.00002),
            AuditedParameter(name="film_thickness", value=0.00001),
            AuditedParameter(name="pressure_gradient", value=-1000.0),
            AuditedParameter(name="viscosity", value=0.00179),
            AuditedParameter(name="U_ski", value=10.0)
        ],
        dimensionless_numbers=[DimensionlessNumber(name="ReynoldsNumber", value=837988.8)],
        simulation_coefficient=0.003792723,
        solver_method="analytical",
        ui_metadata={
            "domain_name": "Fluid Dynamics",
            "independent_var": { "label": "Film Coordinate", "unit": "m" },
            "dependent_var": { "label": "Meltwater Velocity", "unit": "m/s" },
            "primary_metric": { "label": "Frictional Shear Stress" },
            "reference_metric": { "label": "No-Slip Shear Stress" },
            "performance_gain": { "label": "Drag Reduction Ratio" }
        }
    )
    
    sim_out = dispatch_and_solve(miner_out, auditor_out)
    print(f"Solver used: {sim_out.solver_method}")
    print(f"Performance gain: {sim_out.performance_gain_pct:.4f}%")
    print(f"Primary metric (shear stress/deriv): {sim_out.primary_metric_value:.6f}")
    print(f"Reference metric: {sim_out.reference_metric_value:.6f}")
    print(f"Relative error: {sim_out.relative_error:.4e}")
    
    # We expect a positive drag reduction percentage now!
    assert sim_out.performance_gain_pct > 0.0, f"Performance gain should be > 0%, got {sim_out.performance_gain_pct}%"
    print("PlastronGlide test passed successfully!")

if __name__ == "__main__":
    test_plastron()
