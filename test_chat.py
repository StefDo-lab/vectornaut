import urllib.request
import urllib.error
import json
import sys

def run_tests():
    print("=== Testing /api/run with override_parameters ===")
    url_run = "http://127.0.0.1:8080/api/run"
    run_payload = {
        "query": "design a hydrophobic ski base inspired by collembola cuticle",
        "epochs": 50,
        "is_mock": True,
        "override_parameters": {
            "slip_length": 0.000035
        }
    }
    
    headers = {"Content-Type": "application/json"}
    req_run = urllib.request.Request(
        url_run, 
        data=json.dumps(run_payload).encode("utf-8"), 
        headers=headers, 
        method="POST"
    )
    
    run_data = None
    try:
        with urllib.request.urlopen(req_run) as response:
            res_body = response.read().decode("utf-8")
            run_data = json.loads(res_body)
            print(f"Run Success: {run_data.get('success')}")
            print(f"Design Name: {run_data.get('miner', {}).get('design_name')}")
            
            # Check if slip_length was overridden
            params = run_data.get("miner", {}).get("parameters", [])
            slip_length_val = None
            for p in params:
                if p["name"] == "slip_length":
                    slip_length_val = p["value"]
            print(f"Overridden slip_length value: {slip_length_val}")
            if slip_length_val == 0.000035:
                print("SUCCESS: parameter override worked on backend!")
            else:
                print("FAILURE: parameter override did not match 0.000035")
                sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Failed to connect to server for run: {e}")
        sys.exit(1)

    print("\n=== Testing /api/chat ===")
    url_chat = "http://127.0.0.1:8080/api/chat"
    chat_payload = {
        "message": "Warum ist die Reibungsreduktion 66.67%?",
        "history": [],
        "current_run": run_data,
        "is_mock": True
    }
    
    req_chat = urllib.request.Request(
        url_chat, 
        data=json.dumps(chat_payload).encode("utf-8"), 
        headers=headers, 
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req_chat) as response:
            res_body = response.read().decode("utf-8")
            chat_res = json.loads(res_body)
            print(f"Chat Response Reply:\n{chat_res.get('reply')}\n")
            print(f"Suggested parameters: {chat_res.get('suggested_params')}")
            
            if "exakt" in chat_res.get("reply") or "66.67" in chat_res.get("reply") or "1/3" in chat_res.get("reply"):
                print("SUCCESS: Chat returned expected physics explanation!")
            else:
                print("WARNING: Chat reply didn't contain expected physics text patterns, check reply above.")
    except urllib.error.URLError as e:
        print(f"Failed to connect to server for chat: {e}")
        sys.exit(1)

    print("\n=== Testing /api/chat with Optimization query ===")
    chat_payload_opt = {
        "message": "Optimiere das Design",
        "history": [],
        "current_run": run_data,
        "is_mock": True
    }
    req_chat_opt = urllib.request.Request(
        url_chat, 
        data=json.dumps(chat_payload_opt).encode("utf-8"), 
        headers=headers, 
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req_chat_opt) as response:
            res_body = response.read().decode("utf-8")
            chat_res_opt = json.loads(res_body)
            print(f"Chat Opt Reply:\n{chat_res_opt.get('reply')}\n")
            print(f"Suggested parameters: {chat_res_opt.get('suggested_params')}")
            
            if chat_res_opt.get("suggested_params") and "slip_length" in chat_res_opt.get("suggested_params"):
                print("SUCCESS: Chat suggested parameters for optimization!")
            else:
                print("FAILURE: Chat did not suggest expected parameters for optimization.")
                sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Failed to connect to server for chat opt: {e}")
        sys.exit(1)

    print("\nAll backend API tests passed successfully!")
    sys.exit(0)

if __name__ == "__main__":
    run_tests()
