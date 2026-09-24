import urllib.request
import urllib.error
import json

def query_myelin():
    url = "http://127.0.0.1:8080/api/run"
    data = {
        "query": "design a myelin-inspired coaxial electrostatic shield",
        "epochs": 100,
        "is_mock": False
    }
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            with open("myelin_response.json", "w") as f:
                json.dump(res_json, f, indent=4)
            print("Successfully saved response to myelin_response.json")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    query_myelin()
