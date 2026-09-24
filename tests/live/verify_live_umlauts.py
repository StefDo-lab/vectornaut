import urllib.request
import json
import sys

# Configure stdout to use utf-8 to prevent charmap errors on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def verify():
    url_chat = "http://127.0.0.1:8080/api/chat"
    headers = {"Content-Type": "application/json"}
    
    chat_payload = {
        "message": "Erkläre mir bitte kurz die Reibungsreduktion auf Deutsch.",
        "history": [],
        "current_run": None,
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
            reply = chat_res.get("reply", "")
            print("Server Reply:")
            print(reply)
            
            # Check for dollar signs
            dollar_count = reply.count("$")
            print(f"Number of dollar signs in reply: {dollar_count}")
            if dollar_count > 0:
                print("WARNING: Dollar signs found in response!")
            else:
                print("SUCCESS: No dollar signs found in response.")
                
            # Check for other replacement characters
            if "\uFFFD" in reply:
                print("WARNING: Replacement character \uFFFD found in response!")
                
    except Exception as e:
        print(f"Error calling local server: {e}")

if __name__ == "__main__":
    verify()
