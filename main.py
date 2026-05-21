import os
import sys
import json
import argparse
import datetime
from dotenv import load_dotenv

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vectornaut.config import get_client
from vectornaut.miner import Miner
from vectornaut.auditor import Auditor
from vectornaut.simulator import run_simulation

# ASCII Logo for Vectornaut 2.0
LOGO = r"""
======================================================================
  _   __         _                                  _     ___   ___ 
 | | / /        | |                                | |   |__ \ / _ \\
 | |/ /  ___  __| |_ _ __ ___  _ __   __ _ _   _ __| |_     ) | | | |
 |    \ / _ \/ _` | '__/ _` | '_ \ / _` | | | | __| __|   / /| | | |
 | |\  \  __/ (_| | | | (_| | | | | (_| | |_| | |_| |_   / /_| |_| |
 \_| \_/\___|\__,_|_|  \__,_|_| |_|\__,_|\__,_|\__|\__| |____|\___/ 
                                                                    
            Autonomous Biomimetic Material Design Loop
======================================================================
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Vectornaut 2.0: Decoupled Agentic Pipeline Orchestrator")
    parser.add_argument(
        "--query",
        type=str,
        default="design a microtextured biomimetic drag reduction surface for high-velocity water craft",
        help="Design request for the Miner stage."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to train the PINN simulator."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run using mock LLM responses (useful for offline testing or without API keys)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="run_report.json",
        help="Filename to save the structured run report."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    load_dotenv()
    
    print(LOGO)
    print(f"[*] Starting run at: {datetime.datetime.now().isoformat()}")
    print(f"[*] Design Query: '{args.query}'")
    print(f"[*] PINN Epochs: {args.epochs}")
    
    # Check API key configuration
    api_key = os.environ.get("GEMINI_API_KEY")
    is_mock = args.mock
    if not api_key and not is_mock:
        print("[!] WARNING: GEMINI_API_KEY environment variable not set. Falling back to --mock mode.")
        is_mock = True
    elif is_mock:
        print("[*] Running in MOCK mode as requested.")
    else:
        print("[*] Gemini API key found. Initializing AI Studio client...")

    # Stage 1: Miner
    print("\n" + "="*70)
    print(" [STAGE 1] MINER: Semantic Space Exploration")
    print("="*70)
    print("[*] Querying Gemini to extract biomimetic analogues...")
    
    try:
        miner = Miner()
        if is_mock:
            miner_output = miner.mock_mine_design(args.query)
        else:
            miner_output = miner.mine_design(args.query)
            
        print(f"[+] Design Concept: {miner_output.design_name}")
        print(f"[+] Inspiration Source: {miner_output.inspiration_source}")
        print(f"[+] Domain: {miner_output.domain}")
        print(f"[+] Physical Mechanism: {miner_output.physical_mechanism}")
        print("\n[*] Mined Parameters Proposed:")
        for k, v in miner_output.proposed_parameters.items():
            bounds = miner_output.suggested_bounds.get(k, [0.0, 0.0])
            just = miner_output.parameter_justifications.get(k, "No justification provided.")
            print(f"    - {k:25} : {v:<8} (Bounds: {bounds}) -> {just}")
            
    except Exception as e:
        print(f"[!] Miner stage failed: {e}")
        print("[!] Please check your API key, network connection, or run with --mock.")
        sys.exit(1)

    # Stage 2: Auditor
    print("\n" + "="*70)
    print(" [STAGE 2] AUDITOR: Physical Verification Loop")
    print("="*70)
    print("[*] Querying Auditor to sanity check parameters and calculate slip boundary variables...")
    
    try:
        auditor = Auditor()
        if is_mock:
            auditor_output = auditor.mock_audit_design(miner_output)
        else:
            auditor_output = auditor.audit_design(miner_output)
            
        print(f"[+] Audit Passed: {auditor_output.audit_passed}")
        print(f"[+] Audit Notes: {auditor_output.audit_notes}")
        print("\n[*] Audited parameters (sanitized and safe for simulation):")
        for k, v in auditor_output.audited_parameters_dict.items():
            print(f"    - {k:25} : {v}")
            
        print("\n[*] Dimensionless Numbers computed:")
        for k, v in auditor_output.dimensionless_numbers_dict.items():
            print(f"    - {k:25} : {v:.4f}")
            
        print(f"\n[+] Derived Simulation Coefficient (slip length lambda): {auditor_output.simulation_coefficient:.6f}")
        
    except Exception as e:
        print(f"[!] Auditor stage failed: {e}")
        sys.exit(1)

    # Stage 3: Simulator
    print("\n" + "="*70)
    print(" [STAGE 3] SIMULATOR: Physics-Informed Neural Network (PINN-lite)")
    print("="*70)
    print(f"[*] Initializing PyTorch neural network solver...")
    print(f"[*] Solving differential equation u''(y) = -c_pg using autograd over {args.epochs} epochs...")
    
    try:
        # Extract required simulation values from audited parameters
        audited_dict = auditor_output.audited_parameters_dict
        u_free = next((audited_dict[k] for k in ["free_stream_velocity", "design_flow_velocity", "velocity"] if k in audited_dict), 1.5)
        c_pg = next((audited_dict[k] for k in ["pressure_gradient", "pressure_drop"] if k in audited_dict), 2.0)
        
        sim_output = run_simulation(
            slippage_coefficient=auditor_output.simulation_coefficient,
            u_free=u_free,
            c_pg=c_pg,
            epochs=args.epochs
        )
        
        print(f"[+] Training completed successfully.")
        print(f"[+] Final loss: {sim_output.final_loss:.6e}")
        print(f"[+] Autograd-derived wall shear stress (PINN): {sim_output.wall_shear_stress_pinn:.6f}")
        print(f"[+] Analytical wall shear stress (Exact):    {sim_output.wall_shear_stress_analytical:.6f}")
        print(f"[+] Relative solver L2 error:               {sim_output.relative_error:.6f}")
        
        print("\n" + "-"*50)
        print(f" PERFORMANCE METRIC: Biomimetic Drag Reduction")
        print(f" Drag Reduction Efficiency: {sim_output.performance_gain_pct:.2f}%")
        print("-"*50)
        
        # Visualizing velocity field comparison
        print("\n[*] Simulated Flow Velocity Field Comparison (y vs u):")
        print(f"    {'y-coord':<10} | {'u_PINN':<12} | {'u_Analytical':<12} | {'Difference':<12}")
        print("    " + "-"*55)
        
        # Output 5 representative points for readability
        idxs = [0, 4, 9, 14, 19]
        for idx in idxs:
            y = sim_output.sample_points_y[idx]
            u_p = sim_output.sample_points_u_pinn[idx]
            u_a = sim_output.sample_points_u_analytical[idx]
            diff = abs(u_p - u_a)
            print(f"    {y:<10.3f} | {u_p:<12.6f} | {u_a:<12.6f} | {diff:<12.6f}")
            
    except Exception as e:
        print(f"[!] Simulator stage failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save Run Report
    print("\n" + "="*70)
    print(" [STAGE 4] COMPILING FINAL REPORT")
    print("="*70)
    
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": args.query,
        "is_mock": is_mock,
        "miner_stage": miner_output.model_dump(),
        "auditor_stage": auditor_output.model_dump(),
        "simulator_stage": sim_output.model_dump()
    }
    
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"[+] Run report saved successfully to: '{args.output}'")
    except Exception as e:
        print(f"[!] Failed to write run report: {e}")
        
    print("\n[*] Vectornaut 2.0 loop completed successfully!\n")

if __name__ == "__main__":
    main()
