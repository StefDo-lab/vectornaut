# Recorded model answers

Each folder holds the answers to every model call of one pipeline run, written by hand (Claude acting as Gemini) for the findings in `docs/LIVE_PATH_FINDINGS.md`. `session.json` has the query and run settings.

Replay one run (copy it first, because the tool writes `requests/`, `data/` and results into the session folder):

```bash
cp -r benchmarks/llm_sessions/riblet_hose /tmp/riblet_hose
python -m vectornaut.llm_replay --session /tmp/riblet_hose --query "$(python -c "import json;print(json.load(open('/tmp/riblet_hose/session.json'))['query'])")" --epochs 200 --rounds 2
```

The answers were written against the prompts of commit `9349c97`; `transient_heat_shield` calls 06–08 were re-answered after the sandboxed test-script contract (the recorded test script used `subprocess`, which is now rejected). If a pipeline change alters the order or number of model calls, a replay stops at the first call without a matching answer and writes that request, so you can answer it and continue.
