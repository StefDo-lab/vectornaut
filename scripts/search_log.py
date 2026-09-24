import json

def main():
    path = r"C:\Users\Stefan\.gemini\antigravity\brain\19789891-1e71-48f9-b30f-144fa2378083\.system_generated\logs\transcript.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("step_index") == 2519:
                content = data.get("content") or data.get("reply") or ""
                with open("step_2519.txt", "w", encoding="utf-8") as out:
                    out.write(content)
                print("Written step_2519.txt")
                break

if __name__ == "__main__":
    main()
