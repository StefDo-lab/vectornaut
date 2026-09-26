# Explorer trials

Role-play runs of the idea-space explorer (`docs/EXPLORER.md`) in which Claude answered every model call as Gemini via `python -m vectornaut.llm_replay --explorer`. Each folder holds the recorded answers (`responses/`) and the cumulative map report of that run.

Query for all trials: "Entwickle eine bionisch inspirierte Beschichtung für Schiffsrümpfe, die den Reibungswiderstand im Wasser senkt, ohne giftige Antifouling-Wirkstoffe auszukommen, und mehrere Jahre im Salzwasser hält."

| Folder | Explorer version | Setup |
|---|---|---|
| `2026-09-25_v1_guided` | commit `01d2263` | 3 rounds x 3, default strategies, `--opt-rounds 1` |
| `2026-09-25_v1_baseline` | commit `01d2263` | 1 round x 9, `--strategy-weights seed=1` (no map guidance) |

v1 answers only replay on the v1 code: later explorer versions changed scheduling and added the critic call. The findings from v1 led to the evidence tiers, materials critic, requirement coverage and diversity changes in commit `8dda8b0`.
