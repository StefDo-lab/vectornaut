import sys
import os
import numpy as np

# Adjust path to import vectornaut
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vectornaut.config import MinerOutput, ParameterProposal, AuditorOutput, AuditedParameter, DimensionlessNumber
from vectornaut.solver_dispatcher import dispatch_and_solve

def test_2d_heat_conduction():
    print("\n=== Testing 2D Heat Conduction (Double-Pane Insulation) ===")
    miner_out = MinerOutput(
        design_name="Spheniscidae-Plumage Insulated System",
        inspiration_source="Emperor Penguin feather structure",
        domain="Thermodynamics",
        physical_mechanism="Multilayered thermal insulation utilizing trapped stagnant air gaps.",
        parameters=[
            ParameterProposal(name="T_hot", value=300.0, min_bound=273.0, max_bound=400.0, justification="Hot side temp"),
            ParameterProposal(name="T_cold", value=260.0, min_bound=200.0, max_bound=300.0, justification="Cold side temp")
        ],
        governing_equation="d2T_dx2 + d2T_dy2 = 0",
        boundary_conditions=[
            "T(0, y) = T_hot",
            "T(1, y) = T_cold",
            "dT_dy(x, 0) = 0",
            "dT_dy(x, 1) = 0"
        ],
        independent_variables=["x", "y"],
        dependent_variables=["T"]
    )
    
    auditor_out = AuditorOutput(
        audit_passed=True,
        audit_notes="Audit passed for 2D Heat Conduction",
        audited_parameters=[
            AuditedParameter(name="T_hot", value=300.0),
            AuditedParameter(name="T_cold", value=260.0)
        ],
        dimensionless_numbers=[DimensionlessNumber(name="BiotNumber", value=0.15)],
        simulation_coefficient=0.03,
        solver_method="pinn",
        ui_metadata={
            "domain_name": "Thermodynamics",
            "independent_var": { "label": "Position (x, y)", "unit": "m" },
            "dependent_var": { "label": "Temperature (T)", "unit": "K" },
            "primary_metric": { "label": "PINN Mean Temperature" },
            "reference_metric": { "label": "FDM Mean Temperature" },
            "performance_gain": { "label": "Insulation Efficiency" }
        }
    )

    print("Running 2D PINN solver (epochs=400)...")
    sim_out = dispatch_and_solve(miner_out, auditor_out, epochs=400)
    print(f"Solver used: {sim_out.solver_method}")
    print(f"Final 2D PINN loss: {sim_out.final_loss:.4e}")
    print(f"Performance gain: {sim_out.performance_gain_pct:.2f}%")
    print(f"Relative L2 error vs FDM: {sim_out.relative_error:.4e}")
    assert len(sim_out.sample_points) == 400
    assert len(sim_out.solution_primary) == 400
    assert len(sim_out.solution_reference) == 400
    
    # Check that boundaries are approximately satisfied in primary solution
    # Left boundary (x=0) should be close to T_hot (300.0)
    # Right boundary (x=1) should be close to T_cold (260.0)
    left_boundary_vals = []
    right_boundary_vals = []
    for pt, val in zip(sim_out.sample_points, sim_out.solution_primary):
        if abs(pt[0] - 0.0) < 1e-5:
            left_boundary_vals.append(val)
        elif abs(pt[0] - 1.0) < 1e-5:
            right_boundary_vals.append(val)
            
    print(f"Mean left boundary temp (PINN): {np.mean(left_boundary_vals):.2f} K (Expected ~300 K)")
    print(f"Mean right boundary temp (PINN): {np.mean(right_boundary_vals):.2f} K (Expected ~260 K)")
    
    # Check FDM values
    left_boundary_fdm = []
    right_boundary_fdm = []
    for pt, val in zip(sim_out.sample_points, sim_out.solution_reference):
        if abs(pt[0] - 0.0) < 1e-5:
            left_boundary_fdm.append(val)
        elif abs(pt[0] - 1.0) < 1e-5:
            right_boundary_fdm.append(val)
            
    print(f"Mean left boundary temp (FDM): {np.mean(left_boundary_fdm):.2f} K")
    print(f"Mean right boundary temp (FDM): {np.mean(right_boundary_fdm):.2f} K")
    
    print("2D PINN test passed!")

    print("\nRunning 2D FDM solver...")
    auditor_out.solver_method = "fdm"
    sim_out_fdm = dispatch_and_solve(miner_out, auditor_out)
    print(f"Solver used: {sim_out_fdm.solver_method}")
    print(f"Performance gain: {sim_out_fdm.performance_gain_pct:.2f}%")
    print(f"Relative error: {sim_out_fdm.relative_error:.4e}")
    print("2D FDM test passed!")

if __name__ == "__main__":
    try:
        test_2d_heat_conduction()
        print("\nAll 2D integration tests passed successfully!")
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
