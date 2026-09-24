# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import json
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def test_remining_live():
    print("Testing running web server API with a query designed to trigger Re-Mining (Mock Mode)...")
    url = "http://127.0.0.1:8080/api/run"
    
    # We query with 'fail ski' so that:
    # 1. The mock miner triggers Plastron output.
    # 2. It will name it 'PlastronGlide Fail-Prone Ski Base' because 'fail' is in the query.
    # 3. This triggers mock_optimize to fail in round 1.
    # 4. The server performs Re-Mining and gets a second concept named 'Alternative...' which succeeds.
    data = {
        "query": "fail ski base",
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
            
            # Check failed concepts history
            failed_c = res_json.get("failed_concepts", [])
            print(f"Failed concepts count: {len(failed_c)}")
            for idx, fc in enumerate(failed_c, 1):
                print(f"  Failed Concept {idx}: {fc.get('design_name')} -> Reason: {fc.get('reason')}")
                
            # Check final miner concept (should be alternative)
            final_design = res_json.get("miner", {}).get("design_name")
            print(f"Final Design: {final_design}")
            
            # Check synthesis report
            synthesis = res_json.get("synthesis", {})
            print("Synthesis Report keys:", list(synthesis.keys()))
            print("Synthesis Limits:", synthesis.get("mechanical_limits"))
            print("Synthesis Methods:", synthesis.get("manufacturing_methods"))
            
            report_md = res_json.get("report_md", "")
            
            # Check sections in report
            has_synthesis_section = "## 8. Kommerzielle & Praktische Synthese" in report_md
            has_failed_section = "## 9. Verlauf gescheiterter Konzepte (Re-Mining)" in report_md
            
            print(f"Report has Synthesis Section: {has_synthesis_section}")
            print(f"Report has Failed Section: {has_failed_section}")
            
            if len(failed_c) >= 1 and "Alternative" in final_design and has_synthesis_section and has_failed_section:
                print("[+] Success: Re-Mining and Synthesis live integration tests passed successfully!")
                sys.exit(0)
            else:
                print("[-] Failure: Some expected integration test elements are missing.")
                sys.exit(1)
                
    except urllib.error.URLError as e:
        print(f"Failed to connect to local server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_remining_live()
