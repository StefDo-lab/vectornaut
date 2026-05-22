import sys
import os
import sympy as sp
import numpy as np

# Adjust path to import vectornaut
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vectornaut.config import MinerOutput, ParameterProposal, AuditorOutput, AuditedParameter, DimensionlessNumber
from vectornaut.solver_dispatcher import dispatch_and_solve

def test_fluid_dynamics():
    print("\n=== Testing Fluid Dynamics (Shark-Skin Riblets) ===")
    miner_out = MinerOutput(
        design_name="Shark-Skin Inspired Riblet Foil",
        inspiration_source="Galeocerdo cuvier (Tiger Shark)",
        domain="Fluid Dynamics",
        physical_mechanism="Micro-grooves aligned with flow direction reduce viscous drag.",
        parameters=[
            ParameterProposal(name="viscosity", value=0.001, min_bound=0.0001, max_bound=0.01, justification="Water viscosity"),
            ParameterProposal(name="free_stream_velocity", value=1.5, min_bound=0.1, max_bound=5.0, justification="Operating velocity"),
            ParameterProposal(name="pressure_gradient", value=2.0, min_bound=0.0, max_bound=10.0, justification="Pressure gradient")
        ],
        governing_equation="d2u_dy2 = -pressure_gradient / viscosity",
        boundary_conditions=[
            "u(0) = slippage_coefficient * du_dy(0)",
            "u(1) = free_stream_velocity"
        ],
        independent_variables=["y"],
        dependent_variables=["u"],
        svg_schematic="<svg></svg>"
    )
    
    auditor_out = AuditorOutput(
        audit_passed=True,
        audit_notes="Audit passed",
        audited_parameters=[
            AuditedParameter(name="viscosity", value=0.001),
            AuditedParameter(name="free_stream_velocity", value=1.5),
            AuditedParameter(name="pressure_gradient", value=2.0)
        ],
        dimensionless_numbers=[DimensionlessNumber(name="ReynoldsNumber", value=1500.0)],
        simulation_coefficient=0.005, # slippage length
        solver_method="pinn",
        ui_metadata={
            "domain_name": "Fluid Dynamics",
            "independent_var": { "label": "Channel Height", "unit": "m" },
            "dependent_var": { "label": "Flow Velocity", "unit": "m/s" },
            "primary_metric": { "label": "PINN Wall Shear Stress" },
            "reference_metric": { "label": "Analytical Wall Shear Stress" },
            "performance_gain": { "label": "Drag Reduction Efficiency" }
        }
    )

    print("Running PINN solver...")
    sim_out = dispatch_and_solve(miner_out, auditor_out, epochs=100)
    print(f"Solver used: {sim_out.solver_method}")
    print(f"Final PINN loss: {sim_out.final_loss:.4e}")
    print(f"Performance gain: {sim_out.performance_gain_pct:.2f}%")
    print(f"Relative error: {sim_out.relative_error:.4e}")
    assert len(sim_out.solution_primary) == 20
    assert len(sim_out.solution_reference) == 20
    print("Fluid Dynamics PINN test passed!")

    print("\nRunning SciPy solver...")
    auditor_out.solver_method = "scipy"
    sim_out_scipy = dispatch_and_solve(miner_out, auditor_out)
    print(f"Solver used: {sim_out_scipy.solver_method}")
    print(f"Performance gain: {sim_out_scipy.performance_gain_pct:.2f}%")
    print(f"Relative error: {sim_out_scipy.relative_error:.4e}")
    print("Fluid Dynamics SciPy test passed!")

def test_thermodynamics():
    print("\n=== Testing Thermodynamics (Polar Bear Insulation) ===")
    miner_out = MinerOutput(
        design_name="Polar Bear Fur Insulator",
        inspiration_source="Polar Bear Hair",
        domain="Thermodynamics",
        physical_mechanism="Hollow hair traps air to minimize conductive thermal loss.",
        parameters=[
            ParameterProposal(name="T_hot", value=310.0, min_bound=273.0, max_bound=400.0, justification="Body temp K"),
            ParameterProposal(name="T_cold", value=260.0, min_bound=200.0, max_bound=300.0, justification="Ambient temp K"),
            ParameterProposal(name="insulation_thickness", value=0.02, min_bound=0.005, max_bound=0.1, justification="Fur thickness")
        ],
        governing_equation="d2T_dx2 = 0",
        boundary_conditions=[
            "T(0) = T_hot",
            "T(1) = T_cold"
        ],
        independent_variables=["x"],
        dependent_variables=["T"],
        svg_schematic="<svg></svg>"
    )
    
    auditor_out = AuditorOutput(
        audit_passed=True,
        audit_notes="Audit passed",
        audited_parameters=[
            AuditedParameter(name="T_hot", value=310.0),
            AuditedParameter(name="T_cold", value=260.0),
            AuditedParameter(name="insulation_thickness", value=0.02)
        ],
        dimensionless_numbers=[DimensionlessNumber(name="BiotNumber", value=0.15)],
        simulation_coefficient=0.02, # insulation thickness
        solver_method="analytical",
        ui_metadata={
            "domain_name": "Thermodynamics",
            "independent_var": { "label": "Plate Position (x)", "unit": "m" },
            "dependent_var": { "label": "Temperature (T)", "unit": "K" },
            "primary_metric": { "label": "PINN Thermal Gradient" },
            "reference_metric": { "label": "Analytical Thermal Gradient" },
            "performance_gain": { "label": "Thermal Insulation Efficiency" }
        }
    )

    print("Running Analytical solver...")
    sim_out = dispatch_and_solve(miner_out, auditor_out)
    print(f"Solver used: {sim_out.solver_method}")
    print(f"Performance gain: {sim_out.performance_gain_pct:.2f}%")
    print(f"Relative error: {sim_out.relative_error:.4e}")
    print(f"Analytical primary solution profile: {[round(v, 2) for v in sim_out.solution_primary[:5]]}...")
    assert len(sim_out.solution_primary) == 20
    print("Thermodynamics Analytical test passed!")

def test_electromagnetics():
    print("\n=== Testing Electromagnetics (Electrostatics) ===")
    miner_out = MinerOutput(
        design_name="Shielded Electrostatic Plate",
        inspiration_source="Coaxial Shielding",
        domain="Electromagnetics",
        physical_mechanism="Shielded potential boundaries to maintain linear drop.",
        parameters=[
            ParameterProposal(name="V_0", value=100.0, min_bound=0.0, max_bound=1000.0, justification="Potential at 0"),
            ParameterProposal(name="V_1", value=0.0, min_bound=0.0, max_bound=1000.0, justification="Potential at 1")
        ],
        governing_equation="d2V_dy2 = 0",
        boundary_conditions=[
            "V(0) = V_0",
            "V(1) = V_1"
        ],
        independent_variables=["y"],
        dependent_variables=["V"],
        svg_schematic="<svg></svg>"
    )
    
    auditor_out = AuditorOutput(
        audit_passed=True,
        audit_notes="Audit passed",
        audited_parameters=[
            AuditedParameter(name="V_0", value=100.0),
            AuditedParameter(name="V_1", value=0.0)
        ],
        dimensionless_numbers=[DimensionlessNumber(name="DimensionlessVoltage", value=1.0)],
        simulation_coefficient=1.0,
        solver_method="scipy",
        ui_metadata={
            "domain_name": "Electromagnetics",
            "independent_var": { "label": "Gap Distance (y)", "unit": "m" },
            "dependent_var": { "label": "Electrostatic Potential (V)", "unit": "V" },
            "primary_metric": { "label": "PINN Electric Field" },
            "reference_metric": { "label": "Analytical Electric Field" },
            "performance_gain": { "label": "Field Attenuation Efficiency" }
        }
    )

    print("Running SciPy solver...")
    sim_out = dispatch_and_solve(miner_out, auditor_out)
    print(f"Solver used: {sim_out.solver_method}")
    print(f"Performance gain: {sim_out.performance_gain_pct:.2f}%")
    print(f"Relative error: {sim_out.relative_error:.4e}")
    print(f"SciPy solution profile: {[round(v, 2) for v in sim_out.solution_primary[:5]]}...")
    assert len(sim_out.solution_primary) == 20
    print("Electromagnetics SciPy test passed!")

if __name__ == "__main__":
    try:
        test_fluid_dynamics()
        test_thermodynamics()
        test_electromagnetics()
        print("\nAll integration tests passed successfully!")
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
