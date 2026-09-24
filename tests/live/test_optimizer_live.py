# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import json
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def test_optimizer_live():
    print("Testing running web server API with max_optimization_rounds=2 (Mock Mode)...")
    url = "http://127.0.0.1:8080/api/run"
    
    data = {
        "query": "Bionic Hydro-Riblet Ski Base",
        "epochs": 50,
        "is_mock": True,
        "max_optimization_rounds": 2
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            
            print(f"API Response Success: {res_json.get('success')}")
            print(f"Design discovered: {res_json.get('miner', {}).get('design_name')}")
            
            opt_history = res_json.get("optimization_history", [])
            print(f"Optimization History rounds count: {len(opt_history)}")
            
            for run in opt_history:
                print(f"  Round {run.get('round')}: Parameters = {run.get('parameters')}, Gain = {run.get('simulator', {}).get('performance_gain_pct')}%")
                print(f"  Reasoning: {run.get('optimizer_reasoning')}")
                
            report_md = res_json.get("report_md", "")
            if "## 7. Autonome Optimierungshistorie (Closed-Loop)" in report_md:
                print("[+] Success: Report contains section '## 7. Autonome Optimierungshistorie (Closed-Loop)'")
            else:
                print("[-] Failure: Report does NOT contain the optimization history section!")
                sys.exit(1)
                
            if len(opt_history) == 2:
                print("[+] Live optimizer integration test passed successfully!")
                sys.exit(0)
            else:
                print(f"[-] Expected 2 optimization rounds, got {len(opt_history)}")
                sys.exit(1)
                
    except urllib.error.URLError as e:
        print(f"Failed to connect to local server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_optimizer_live()
