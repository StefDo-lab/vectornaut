# Live Gemini explorer run: facade cooling (2026-09-27)

First live run with real model calls: every stage (explorer generator, formulator, auditor, synthesizer, critic) used `gemini-3.8-flash`, 5 rounds x 3, seed 0, `--opt-rounds 1`, explorer v5 (commit `88a49b2`, streaming enabled).

- 15 of 15 candidates evaluated, 0 transient errors; 29 min.
- Usage: 55 calls, 147,105 prompt, 79,910 output and 361,871 thinking tokens — about US$1.8 at the introductory price (US$0.75 / 3.75 per M input / output tokens).
- An earlier attempt without streaming (commit `a043b45`) lost 9 of 15 candidates to `502 Bad Gateway`: the cloud environment's proxy cuts requests silent for ~30 s, which high-thinking calls exceed. Streaming with thought summaries fixed it.

Observations: Gemini's second-best concept (hermetically sealed closed-cell tile under a dense glaze) independently matches the best concept of the Claude role-play run. Gemini's critic was markedly more lenient (objective gains 32–48 % vs 18–20 %, requirement coverage ~0.9 vs ~0.6), and several claims are optimistic (sub-ambient facade under peak sun, SR 0.95–0.97).

Files: `archive.json`, `archive_export.json`, `map_report.md`, `run_log.txt`, `best3_summary.md` (top 3 in the blind-comparison format).
