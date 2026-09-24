# pytest is optional (the canonical runner is `python -m unittest discover -s tests -t .`).
# tests/live/ needs a running server and/or a Gemini key; tests/manual/ holds ad-hoc scripts.
collect_ignore_glob = ["tests/live/*", "tests/manual/*"]
