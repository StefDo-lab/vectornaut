import urllib.request
import urllib.error
import json
import sys

def test_api():
    print("Testing running web server API at http://127.0.0.1:8080/api/run...")
    url = "http://127.0.0.1:8080/api/run"
    data = {
        "query": "design a microtextured biomimetic drag reduction surface for high-velocity water craft",
        "epochs": 100,
        "is_mock": True
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            
            print(f"API Response Success: {res_json.get('success')}")
            print(f"Design discovered: {res_json.get('miner', {}).get('design_name')}")
            print(f"Solver used: {res_json.get('simulator', {}).get('solver_method')}")
            print(f"Performance Gain: {res_json.get('simulator', {}).get('performance_gain_pct'):.2f}%")
            
            if res_json.get("success") is True:
                print("E2E API test passed!")
                sys.exit(0)
            else:
                print("E2E API test failed (success field not True)")
                sys.exit(1)
    except urllib.error.URLError as e:
        print(f"E2E API test failed to connect to server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_api()
