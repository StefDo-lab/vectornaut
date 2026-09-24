import urllib.request
import urllib.error
import json

def test_chat_encoding():
    url = "http://127.0.0.1:8080/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "message": "Erkläre das bitte",
        "history": [],
        "current_run": None,
        "is_mock": True
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            print("Response Headers:")
            for k, v in response.getheaders():
                print(f"  {k}: {v}")
            raw_bytes = response.read()
            print("\nRaw bytes length:", len(raw_bytes))
            print("Raw bytes (first 300):", raw_bytes[:300])
            try:
                decoded_str = raw_bytes.decode("utf-8")
                print("\nDecoded as UTF-8 successfully!")
                parsed_json = json.loads(decoded_str)
                print("Parsed reply first 150 chars:", parsed_json["reply"][:150])
            except Exception as e:
                print("Failed to decode/parse:", e)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_chat_encoding()
