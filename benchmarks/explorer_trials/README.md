# Explorer trials

Role-play runs of the idea-space explorer (`docs/EXPLORER.md`) in which Claude answered every model call as Gemini via `python -m vectornaut.llm_replay --explorer`. Each folder holds the recorded answers (`responses/`) and the cumulative map report of that run.

Query for all trials: "Entwickle eine bionisch inspirierte Beschichtung für Schiffsrümpfe, die den Reibungswiderstand im Wasser senkt, ohne giftige Antifouling-Wirkstoffe auszukommen, und mehrere Jahre im Salzwasser hält."

| Folder | Explorer version | Setup |
|---|---|---|
| `2026-09-25_v1_guided` | commit `01d2263` | 3 rounds x 3, default strategies, `--opt-rounds 1` |
| `2026-09-25_v1_baseline` | commit `01d2263` | 1 round x 9, `--strategy-weights seed=1` (no map guidance) |
| `2026-09-26_v2_guided` | commit `8dda8b0` (+ critic) | 5 rounds x 3, default strategies, `--opt-rounds 1` |
| `2026-09-26_v2_baseline` | commit `8dda8b0` (+ critic) | 1 round x 15, `--strategy-weights seed=1` |
| `2026-09-26_v3_guided` | commit `ad72735` | 6 rounds x 3, default strategies, `--opt-rounds 1` (objective/baseline framing, two-number critic) |

v1 answers only replay on the v1 code: later explorer versions changed scheduling and added the critic call. The findings from v1 led to the evidence tiers, materials critic, requirement coverage and diversity changes in commit `8dda8b0`.

v2 findings: no concept reduced clean-hull drag by more than ~0–2 % (critic estimates); both runs drifted to fouling release, the real lever over multi-year service, which the clean-hull baseline scored as ~0. This led to the objective/baseline framing, two-number critic scoring and strategy changes that followed.

v3 findings and the independent review (`review_2026-09-26.md`) led to the v4 scoring (objective-dominated, min(simulated, critic), proxy flag, bio-origin preference, combine anchors, extrapolate slots only for qualifying trends, tiebreak). Re-scoring the v3 archive under v4 raised the Spearman correlation between elite score and critic objective gain from 0.70 to 0.92. v3 answers parse under v4 but the search orders differ, so they do not replay one-to-one.

## Second task: passive facade/roof cooling

Query: "Entwickle eine bionisch inspirierte, passive Beschichtung oder Fassadenstruktur für Gebäude in heißen Klimazonen, die an Sommertagen die Wärmelast im Innenraum deutlich senkt, ohne Strom oder Wasserzufuhr auskommt und mindestens 20 Jahre wetterbeständig bleibt."

| Folder | Explorer version | Setup |
|---|---|---|
| `2026-09-27_facade_guided` | commit `374c50f` | 5 rounds x 3, default strategies, `--opt-rounds 1` |
| `2026-09-27_facade_baseline` | commit `374c50f` | 1 round x 15, `--strategy-weights seed=1` |

Findings: the objective scale (2 %, tuned for hull drag) saturated for gains of 5–35 %, so ranking was decided by requirement-rating noise and insulation-in-disguise concepts ranked high; a symbolic SymPy solve hung ~20 min without timeout; generator infeasibility was stored only per exact cell.

These findings led to archive version 5 (`docs/EXPLORER.md`): an objective scale per archive (a stated `target_gain_pct` / 2, else 2 × the median critic objective gain, at least 2 %, re-scored after every round), must/nice requirements with a soft minimum and a soft gate, a 0.8 factor for a non-conventional baseline, the critic's `conventional_equivalent_gain_pct` subtracted from the objective gain, infeasibility scopes and pair checks in fill_gap/diversify, relevant mechanism classes from the function analysis, and a time budget for the symbolic solve (`VECTORNAUT_SYMBOLIC_TIMEOUT_S`, default 20 s) with an on-disk solver memo for replays. Re-scoring offline: the facade archives get scales of 20 % (guided) and 12 % (baseline); in the baseline run the cork cladding (bionic 0.15) falls from rank 1 to 6 and the Cyphochilus coat is first; the hull v3 archive keeps the 2 % scale. The facade answers predate the conventional-equivalent field, so the insulation-like concepts of the guided run keep their gains in that re-scoring.
