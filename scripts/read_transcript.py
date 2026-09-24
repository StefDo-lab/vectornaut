import json
import sys

def read():
    sys.stdout.reconfigure(encoding='utf-8')
    path = r"C:\Users\Stefan\.gemini\antigravity\brain\19789891-1e71-48f9-b30f-144fa2378083\.system_generated\logs\transcript.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("type") == "USER_INPUT":
                print(f"Step {data.get('step_index')}: USER: {data.get('content')[:150]}")
            elif "exception" in str(data).lower() or "error" in str(data).lower():
                if data.get("type") in ["RUN_COMMAND", "VIEW_FILE", "REPLACE_FILE_CONTENT", "WRITE_TO_FILE"]:
                    content = data.get("content", "")
                    if "500" in content or "traceback" in content.lower():
                        print(f"Step {data.get('step_index')}: {data.get('type')} error content: {content[:300]}")

if __name__ == "__main__":
    read()
