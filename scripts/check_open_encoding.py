import os
import re

def search_open_calls():
    for root, dirs, files in os.walk("."):
        if ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    for i, line in enumerate(lines):
                        if "open(" in line:
                            # check if encoding is specified
                            if "encoding" not in line:
                                print(f"WARNING: open() without encoding in {path}:{i+1} -> {line.strip()}")
                except Exception as e:
                    print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    search_open_calls()
