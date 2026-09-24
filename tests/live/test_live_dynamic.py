# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import json
import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def test_live_dynamic():
    print("Testing running web server API with a query designed to trigger dynamic_script...")
    url = "http://127.0.0.1:8080/api/run"
    
    # We explicitly request dynamic_script solver method in the query to guide the LLM Auditor.
    data = {
        "query": "A highly non-linear transient heat shield with temperature-dependent conductivity. Please select solver_method = 'dynamic_script' to write a custom simulation script.",
        "epochs": 100,
        "is_mock": False
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            
            print(f"API Response Success: {res_json.get('success')}")
            sim = res_json.get('simulator', {})
            print(f"Design discovered: {res_json.get('miner', {}).get('design_name')}")
            print(f"Solver used: {sim.get('solver_method')}")
            
            # Print validation fields
            print(f"Validation Passed: {sim.get('validation_passed')}")
            print("Validation Report:")
            print(sim.get('validation_report'))
            
            # Check if report_md has the section
            report_md = res_json.get("report_md", "")
            if "## 6. Automatische Validierung (AI-Generated Tests)" in report_md:
                print("[+] Success: Report contains section '## 6. Automatische Validierung (AI-Generated Tests)'")
            else:
                print("[-] Failure: Report does NOT contain the validation section!")
                sys.exit(1)
                
            if res_json.get("success") is True and sim.get('solver_method') == 'dynamic_script':
                print("[+] Live dynamic solver test passed successfully!")
                sys.exit(0)
            else:
                print(f"[-] Live dynamic solver test failed. Solver used: {sim.get('solver_method')}")
                sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Live dynamic solver test failed to connect: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_live_dynamic()
