import urllib.request
import json

def test():
    url = "http://127.0.0.1:8080/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "message": "Erkläre mir bitte kurz die Reibungsreduktion auf Deutsch.",
        "history": [],
        "current_run": None,
        "is_mock": False
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            print("Raw response length:", len(raw))
            # Let's inspect the reply field
            data = json.loads(raw.decode("utf-8"))
            reply = data.get("reply", "")
            print("Reply content (repr):")
            print(repr(reply))
            print("Contains replacement char:", "\\ufffd" in reply)
            # Find any non-ascii characters and print their hex values
            for char in reply:
                if ord(char) > 127:
                    print(f"Non-ascii char: {char!r} (U+{ord(char):04X})")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test()
