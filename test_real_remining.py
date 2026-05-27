# -*- coding: utf-8 -*-
import urllib.request
import json
import time

def test_real_discovery():
    url = "http://127.0.0.1:8080/api/run"
    
    # We will trigger a real discovery run with a query that previously failed
    payload = {
        "query": "design a breathable water-repellent membrane",
        "epochs": 100,
        "is_mock": False,
        "max_optimization_rounds": 2
    }
    
    print(f"[*] Sending request to {url} with query: '{payload['query']}'...")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            print(f"[+] API Request completed in {time.time() - start_time:.2f}s!")
            print(f"[+] Success: {res_data.get('success')}")
            if res_data.get('success'):
                print(f"    Design Name: {res_data['miner']['design_name']}")
                print(f"    Inspiration: {res_data['miner']['inspiration_source']}")
                print(f"    Solver used: {res_data['simulator']['solver_method']}")
                print(f"    Performance gain: {res_data['simulator']['performance_gain_pct']:.4f}%")
                print(f"    Relative error: {res_data['simulator']['relative_error']:.4e}")
                print(f"    Audited params: {res_data['auditor']['audited_parameters_dict']}")
                if res_data.get("failed_concepts"):
                    print(f"    Failed concepts in history: {len(res_data['failed_concepts'])}")
                    for idx, fc in enumerate(res_data['failed_concepts'], 1):
                        print(f"      - {idx}: {fc['design_name']} -> {fc['reason']}")
            else:
                print(f"[-] API returned success=False: {res_data.get('error')}")
    except Exception as e:
        print(f"[!] Request failed: {e}")
        if hasattr(e, 'read'):
            print(f"[!] Server Error Details: {e.read().decode('utf-8')}")

if __name__ == "__main__":
    test_real_discovery()
