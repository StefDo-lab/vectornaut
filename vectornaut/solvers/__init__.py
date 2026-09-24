"""
Solver building blocks used by vectornaut.solver_dispatcher:

- parsing:        1D equation/BC parsing, missing-parameter auto-detection, domain bounds
- pinn_model:     GenericPINN network shared by the 1D and 2D PINN solvers
- solvers_1d:     SymPy analytical, SciPy BVP and 1D PINN solvers
- solvers_2d:     2D parsing, 2D FDM and 2D PINN solvers
- model_cache:    save/load of trained PINN models for reuse
- dynamic_script: objective contract, parameter sweep and validation of generated scripts
"""
