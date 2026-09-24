import urllib.request
import urllib.error
import json
import sys

def run_test():
    url = "http://127.0.0.1:8080/api/run"
    headers = {"Content-Type": "application/json"}
    
    # 1. Baseline run with is_mock=True
    baseline_payload = {
        "query": "PlastronGlide Hydrophobic Ski Base inspired by Collembola cuticle with slip_length=0.00002 and film_thickness=0.00001",
        "epochs": 100,
        "is_mock": True,
        "override_parameters": None
    }
    
    print("[*] Sending baseline request...")
    req_baseline = urllib.request.Request(
        url,
        data=json.dumps(baseline_payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req_baseline) as response:
            res_body = response.read().decode("utf-8")
            baseline_res = json.loads(res_body)
    except urllib.error.URLError as e:
        print(f"[-] Failed to connect to server: {e}")
        sys.exit(1)
        
    if not baseline_res.get("success"):
        print("[-] Baseline run failed:", baseline_res)
        sys.exit(1)
        
    baseline_miner = baseline_res.get("miner", {})
    baseline_auditor = baseline_res.get("auditor", {})
    baseline_sim = baseline_res.get("simulator", {})
    
    baseline_coeff = baseline_auditor.get("simulation_coefficient")
    baseline_gain = baseline_sim.get("performance_gain_pct")
    
    # Extract actual slip_length in auditor output
    baseline_audited_params = baseline_auditor.get("audited_parameters_dict", {})
    baseline_slip = baseline_audited_params.get("slip_length")
    
    print(f"[+] Baseline run completed successfully:")
    print(f"    - Audited slip_length: {baseline_slip}")
    print(f"    - Simulation coefficient: {baseline_coeff}")
    print(f"    - Performance gain: {baseline_gain}%")
    
    # 2. Optimization run with override_parameters: {"slip_length": 0.00004}
    opt_payload = {
        "query": "PlastronGlide Hydrophobic Ski Base inspired by Collembola cuticle with slip_length=0.00002 and film_thickness=0.00001",
        "epochs": 100,
        "is_mock": True,
        "override_parameters": {"slip_length": 0.00004},
        "previous_miner_output": baseline_miner
    }
    
    print("[*] Sending optimization request with override_parameters: {'slip_length': 0.00004}...")
    req_opt = urllib.request.Request(
        url,
        data=json.dumps(opt_payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req_opt) as response:
            res_body = response.read().decode("utf-8")
            opt_res = json.loads(res_body)
    except urllib.error.URLError as e:
        print(f"[-] Failed to connect to server: {e}")
        sys.exit(1)
        
    if not opt_res.get("success"):
        print("[-] Optimization run failed:", opt_res)
        sys.exit(1)
        
    opt_auditor = opt_res.get("auditor", {})
    opt_sim = opt_res.get("simulator", {})
    
    opt_coeff = opt_auditor.get("simulation_coefficient")
    opt_gain = opt_sim.get("performance_gain_pct")
    
    opt_audited_params = opt_auditor.get("audited_parameters_dict", {})
    opt_slip = opt_audited_params.get("slip_length")
    
    print(f"[+] Optimization run completed successfully:")
    print(f"    - Audited slip_length: {opt_slip}")
    print(f"    - Simulation coefficient: {opt_coeff}")
    print(f"    - Performance gain: {opt_gain}%")
    
    # 3. Assertions
    print("[*] Running assertions...")
    
    # Verify that the overridden parameter was updated in auditor output parameters dict
    assert opt_slip == 0.00004, f"Expected audited slip_length to be 0.00004, but got {opt_slip}"
    print("[+] Assertion passed: audited slip_length was successfully overridden to 0.00004.")
    
    # Verify that the simulation coefficient changed due to the override
    assert opt_coeff != baseline_coeff, f"Simulation coefficient did not change: baseline={baseline_coeff}, opt={opt_coeff}"
    print(f"[+] Assertion passed: simulation coefficient changed from {baseline_coeff} to {opt_coeff}.")
    
    # Verify that the performance gain changed due to the override
    assert opt_gain != baseline_gain, f"Performance gain did not change: baseline={baseline_gain}%, opt={opt_gain}%"
    print(f"[+] Assertion passed: performance gain changed from {baseline_gain}% to {opt_gain}%.")
    
    print("\n[SUCCESS] Integration test passed! Parameter propagation and override works correctly.")

if __name__ == "__main__":
    run_test()
