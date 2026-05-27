# -*- coding: utf-8 -*-
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from vectornaut.miner import Miner
from vectornaut.formulator import ModelFormulator
from vectornaut.auditor import Auditor
from vectornaut.solver_dispatcher import dispatch_and_solve

def test_pipeline():
    miner = Miner()
    formulator = ModelFormulator()
    auditor = Auditor()
    
    query = "breathable omniphobic membrane"
    print(f"[*] Querying Miner for query: '{query}'...")
    concept = miner.mine_design(query)
    print(f"[+] Concept mined: {concept.design_name}")
    print(f"    Inspiration: {concept.inspiration_source}")
    print(f"    Parameters: {[p.name for p in concept.parameters]}")
    
    print("\n[*] Formulating model...")
    miner_output = formulator.formulate_model(query, concept)
    print(f"[+] Formulated model:")
    print(f"    Governing: {miner_output.governing_equation}")
    print(f"    BCs: {miner_output.boundary_conditions}")
    print(f"    Variables: {miner_output.independent_variables} -> {miner_output.dependent_variables}")
    
    print("\n[*] Auditing model...")
    auditor_output = auditor.audit_design(miner_output, user_query=query)
    print(f"[+] Audit passed: {auditor_output.audit_passed}")
    print(f"    Audited params: {auditor_output.audited_parameters_dict}")
    print(f"    Solver method: {auditor_output.solver_method}")
    print(f"    Simulation coefficient: {auditor_output.simulation_coefficient}")
    
    print("\n[*] Solving model...")
    try:
        sim_output = dispatch_and_solve(
            miner_output=miner_output,
            auditor_output=auditor_output,
            epochs=200
        )
        print(f"[+] Solve successful!")
        print(f"    Solver method used: {sim_output.solver_method}")
        print(f"    Performance gain: {sim_output.performance_gain_pct:.4f}%")
        print(f"    Relative error: {sim_output.relative_error:.4e}")
        print(f"    Primary metric: {sim_output.primary_metric_value}")
        print(f"    Reference metric: {sim_output.reference_metric_value}")
    except Exception as solve_err:
        print(f"[-] Solve failed!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
