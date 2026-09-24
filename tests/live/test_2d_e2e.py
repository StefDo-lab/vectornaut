import urllib.request
import urllib.error
import json
import sys

def test_2d_e2e():
    print("Testing running web server API with 2D query at http://127.0.0.1:8080/api/run...")
    url = "http://127.0.0.1:8080/api/run"
    data = {
        "query": "Determine the temperature distribution through a double-pane glass window under winter conditions using 2D heat conduction model",
        "epochs": 100, # 100 epochs is quick and enough for verification
        "is_mock": False
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            
            print(f"API Response Success: {res_json.get('success')}")
            print(f"Design discovered: {res_json.get('miner', {}).get('design_name')}")
            print(f"Variables chosen: {res_json.get('miner', {}).get('independent_variables')}")
            print(f"Governing Equation: {res_json.get('miner', {}).get('governing_equation')}")
            print(f"Solver used: {res_json.get('simulator', {}).get('solver_method')}")
            print(f"Relative error: {res_json.get('simulator', {}).get('relative_error')}")
            print(f"Sample points count: {len(res_json.get('simulator', {}).get('sample_points', []))}")
            
            if res_json.get("success") is True and len(res_json.get('simulator', {}).get('sample_points', [])) == 400:
                print("E2E 2D API test passed!")
                sys.exit(0)
            else:
                print("E2E 2D API test failed (success field not True or sample points not 400)")
                sys.exit(1)
    except urllib.error.URLError as e:
        print(f"E2E 2D API test failed to connect to server or request failed: {e}")
        # Print response body if available for debugging
        if hasattr(e, 'read'):
            try:
                print("Error Response Body:", e.read().decode('utf-8'))
            except Exception:
                pass
        sys.exit(1)

if __name__ == "__main__":
    test_2d_e2e()
