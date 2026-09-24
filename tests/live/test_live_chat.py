import urllib.request
import urllib.error
import json
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def test_live():
    url_chat = "http://127.0.0.1:8080/api/chat"
    headers = {"Content-Type": "application/json"}
    
    # We pass a simple message and current_run = None, is_mock = False
    chat_payload = {
        "message": "Warum ist die Reibungsreduktion 66.67%?",
        "history": [],
        "current_run": {
            "miner": {
                "design_name": "PlastronGlide Hydrophobic Ski Base",
                "inspiration_source": "Collembola cuticle",
                "domain": "Fluid Dynamics",
                "physical_mechanism": "Mimicking springtail cuticle.",
                "parameters": [
                    {"name": "slip_length", "value": 0.00002},
                    {"name": "film_thickness", "value": 0.00001}
                ],
                "governing_equation": "d2u_dy2 = 0",
                "boundary_conditions": ["u(0) = slip_length * du_dy(0)", "u(1) = 1"],
                "independent_variables": ["y"],
                "dependent_variables": ["u"]
            },
            "auditor": {
                "audited_parameters_dict": {"slip_length": 0.00002, "film_thickness": 0.00001},
                "dimensionless_numbers_dict": {},
                "ui_metadata": {
                    "primary_metric": {"label": "Bionische Reibung"},
                    "reference_metric": {"label": "Referenz Reibung"},
                    "performance_gain": {"label": "Reibungsreduktion"}
                }
            },
            "simulator": {
                "performance_gain_pct": 66.67,
                "solver_method": "analytical"
            }
        },
        "is_mock": False
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
            print("SUCCESS: Live chat response:")
            print(json.dumps(chat_res, indent=2, ensure_ascii=False))
    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code} - {e.reason}")
        print(e.read().decode("utf-8"))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_live()
