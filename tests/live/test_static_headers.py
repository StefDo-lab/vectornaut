import urllib.request

def test_static():
    url = "http://127.0.0.1:8080/static/app.js"
    # Actually, uvicorn mounts the static directory at / (not /static) because of:
    # app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
    # So we should request http://127.0.0.1:8080/app.js and index.html
    
    for path in ["/index.html", "/app.js", "/style.css"]:
        url = f"http://127.0.0.1:8080{path}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req) as response:
                print(f"\nURL: {url}")
                print("Response Headers:")
                for k, v in response.getheaders():
                    print(f"  {k}: {v}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")

if __name__ == "__main__":
    test_static()
