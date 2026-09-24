import urllib.request
import urllib.error
import json
import sys

def test_live_plastron():
    print("Testing running web server API with PlastronGlide at http://127.0.0.1:8080/api/run...")
    url = "http://127.0.0.1:8080/api/run"
    data = {
        "query": "PlastronGlide Hydrophobic Ski Base inspired by Collembola cuticle with slip_length=0.00002 and film_thickness=0.00001",
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
            print(f"Primary Metric (Wall Shear Stress): {sim.get('primary_metric_value'):.6f}")
            print(f"Reference Metric (Wall Shear Stress): {sim.get('reference_metric_value'):.6f}")
            print(f"Performance Gain (Drag Reduction Ratio): {sim.get('performance_gain_pct'):.4f}%")
            
            if res_json.get("success") is True and sim.get('performance_gain_pct', 0.0) > 0.0:
                print("Live PlastronGlide API test passed!")
                sys.exit(0)
            else:
                print("Live PlastronGlide API test failed (gain was 0 or not successful)")
                sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Live PlastronGlide API test failed to connect: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_live_plastron()
