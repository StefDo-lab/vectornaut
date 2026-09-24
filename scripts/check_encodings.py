import os
import sys

def check_file(path):
    print(f"Checking {path}:")
    try:
        with open(path, "rb") as f:
            bytes_data = f.read()
        
        # Try decoding as UTF-8
        try:
            bytes_data.decode("utf-8")
            print("  Decodes as UTF-8: SUCCESS")
        except UnicodeDecodeError as e:
            print(f"  Decodes as UTF-8: FAILED ({e})")
            
        # Try decoding as cp1252
        try:
            bytes_data.decode("cp1252")
            print("  Decodes as cp1252: SUCCESS")
        except UnicodeDecodeError as e:
            print(f"  Decodes as cp1252: FAILED ({e})")
            
        # Find non-ascii chars
        non_ascii = []
        for i, b in enumerate(bytes_data):
            if b > 127:
                non_ascii.append((i, b))
        print(f"  Number of non-ascii bytes: {len(non_ascii)}")
        if non_ascii:
            print("  First 10 non-ascii bytes and their surroundings:")
            for idx, b in non_ascii[:10]:
                start = max(0, idx - 5)
                end = min(len(bytes_data), idx + 6)
                snippet = bytes_data[start:end]
                print(f"    At index {idx}: byte {hex(b)}, surrounding bytes: {snippet}")
    except Exception as e:
        print(f"  Error checking file: {e}")

if __name__ == "__main__":
    check_file("web_server.py")
    check_file("static/app.js")
    check_file("static/index.html")
