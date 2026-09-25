# Idea-Space Explorer

`vectornaut/explorer/` searches the space of ideas for one request systematically instead of asking the model to "be creative" again and again. It works for materials concepts (evaluated by the simulation pipeline) and for business ideas (evaluated by a transparent unit-economics model).

## The idea

Ideas that can work are not spread evenly over the space of all ideas. They cluster, the way matter clumps into galaxies instead of filling space evenly. If we know where the good clusters are, we can search there on purpose: fill the gaps inside a cluster, follow a cluster's direction beyond its known edge, or mix two distant clusters.

To make "where" concrete, every idea gets coordinates on a few fixed **descriptor axes** (for materials: mechanism class, length scale, inspiration origin, governing quantity). One value per axis gives a **cell**. The explorer keeps an **archive** with the best idea per cell (a quality-diversity archive, as in MAP-Elites) and grows it over rounds and runs.

Two maps matter, and the archive keeps both:

- **Where ideas are proposed** (`proposals` per cell). A language model's free proposals cluster around textbook ideas (shark skin, lotus leaf). This is the model's prior, not reality.
- **Where ideas work** (the elite score per cell).

The interesting targets are cells with few proposals next to high-scoring cells: places the prior neglects but the evidence points to. The report prints both maps side by side.

## One round

1. **Schedule** a batch of search orders from the archive (see strategies below). Each order names a target cell (full or partial), the reason, and context (the elites to mutate or combine, neighbour scores, the trend).
2. **Generate** one candidate per order in one model call (`get_model_name("explorer")`, schema `MaterialsCandidateBatch` / `BusinessCandidateBatch`). The prompt walks through function analysis → mechanism classes → analogue search → candidates, lists nearby archived titles and textbook solutions to avoid, and allows the answer "this target cell cannot contain a working concept" (`target_feasible: false` + reason).
3. **Check** each candidate deterministically: descriptors from the closed vocabulary (case and spacing are tolerated, unknown values reject the candidate, nothing is guessed), required fields, finite numbers, duplicate titles, profile sanity checks.
4. **Evaluate** the valid candidates and store every candidate in the archive, including rejected and failed ones (they count towards the proposal density). An elite is only replaced by a strictly higher score.
5. **Report**: `round_NNN.md`, the cumulative `map.md` and `archive_export.json`.

A candidate whose descriptors do not match its target is kept in the cell it really belongs to (`on_target: false`). An honest label is worth more than a hit.

## Strategies

All strategies are pure functions of the archive and a seeded random generator (`vectornaut/explorer/strategies.py`), so a run is reproducible for a given seed.

| Strategy | Target | Notes |
|---|---|---|
| `refine` | an elite's own cell | Mutates the elite; elites with high scores and few earlier refinements are preferred. The order names the weakest score component and the stated main risk. |
| `fill_gap` | an empty cell at distance 1 from elites | Ranked by best neighbour score, plus a bonus for several elite neighbours, minus a penalty for earlier attempts that landed or aimed there (low proposal density first). |
| `extrapolate` | one step beyond the explored edge of an ordinal axis | Fits a least-squares line of elite score vs. ordinal index, holding the other axes fixed ("slice") or taking the best score per value ("marginal", gives a partial target). Skipped when the slope is below `--min-slope`, when the score drops at the edge (the optimum is inside the explored range), at the end of the scale, or when the target is infeasible. |
| `combine` | a cell mixing the descriptors of two distant elites | Picks the pair with the largest descriptor distance (then the highest joint score) that was not combined before; prefers an empty mixed cell. |
| `explore` | a random cell nobody has proposed or targeted | Small-probability fallback (default weight 0.1). |
| `seed` | no target | Cold start: the model proposes freely. This samples where ideas are usually proposed. |

Distance: a differing nominal axis counts 1, an ordinal axis counts its number of steps. "Distance 1" is one nominal change or one ordinal step.

The scheduler splits the batch by weight (default `refine=0.25, fill_gap=0.3, extrapolate=0.2, combine=0.15, explore=0.1`) with a largest-remainder rule; leftover slots are drawn with the seeded generator. Slots a strategy cannot fill fall back to other strategies; `seed` always succeeds. No two orders in a batch target the same cell. Without any elites only `seed` and `explore` run.

**Infeasible cells.** If the generator says a target cell cannot contain a working concept, or the pipeline rejects a concept as physically infeasible, the cell (or partial pattern) is stored under `infeasible` with the reason. No strategy targets it again (except refining an existing elite). This is information about the map, not a failure.

## Profiles

### materials

| Axis | Kind | Values |
|---|---|---|
| `mechanism_class` | nominal | interfacial_slip, flow_redirection, trapped_gas_or_liquid, porous_transport, graded_stiffness, architected_lattice, radiative_control, phase_change, electrostatic_field, other |
| `length_scale` | ordinal | nm < sub_um < um < 10_um < 100_um < mm < cm_plus |
| `inspiration_origin` | nominal | plant, animal, microbe, geology, atmosphere_ocean, technology, other |
| `governing_quantity` | nominal | wall_shear, flow_rate, heat_flux, temperature, deflection, stress, field_strength, other |

**Evaluation**: every candidate becomes a `MinerConceptOutput` and runs through the normal pipeline (formulator → auditor → solver → validator → optimizer → synthesizer) through the `PipelineRunRequest.concept` hook with `max_concept_attempts=1`, so exactly this concept is judged and never swapped for a re-mined one. The raw pipeline result is saved under `explorer/<profile>/results/<entry>.json`.

**Score** (0–100): `100 * (0.4 * validity + 0.6 * gain_score)`

- `validity` = validator score × 1.0 (pass) / 0.8 (warn) / 0 (fail).
- `gain_score` = `1 - exp(-gain_pct / 20)` from the simulated `performance_gain_pct` (negative gains score 0).
- If the simulator has no usable gain (missing, not finite, or a `gain_basis`-like field saying `n/a`), the candidate's own back-of-envelope estimate (in %) is used at half weight and the entry is marked **"estimated, not simulated"**.
- Pipeline rejection (`status: "rejected"`) → the cell is marked infeasible. A failing stage (audit, solver, validator, optimizer) → status `failed`, no score.

### business

| Axis | Kind | Values |
|---|---|---|
| `customer_segment` | nominal | consumers, smb, enterprise, public_sector, developers, healthcare, industry, education |
| `revenue_model` | nominal | subscription, transaction_fee, marketplace, licensing, hardware_plus_service, advertising, usage_based |
| `market_scale` | ordinal | niche < regional < national < continental < global |
| `advantage_type` | nominal | cost, time, quality, access, compliance, sustainability |
| `capital_intensity` | ordinal | bootstrap < seed < series_a < heavy |

**Evaluation**: a deterministic model computes, from eight inputs the candidate must estimate (monthly revenue per customer, gross margin, CAC, monthly churn, addressable customers, reachable share after 3 years, fixed costs per year, upfront capex): gross profit per customer, lifetime (capped at 60 months), LTV, LTV/CAC, payback months, 3-year revenue and profit (linear ramp to N customers, acquisitions include replaced churn), return on spend, and break-even customers. The formulas are in the docstring of `vectornaut/explorer/profiles/business.py`.

**Score** (0–100): `100 * base * (1 - sanity_penalty) * (1 - critic_penalty)` with `base = 0.3*ltv_cac + 0.2*payback + 0.3*roi + 0.2*scale`, each component clamped to 0..1 (LTV/CAC 1→0, 5→1; payback 0→1, 36 months→0; ROI −100 %→0, +100 %→1; 3-year revenue 100 kEUR→0, 1 bn EUR→1).

**Sanity flags** (penalised and listed in the breakdown): churn ≥ 100 % or ≤ 0 or implausibly low, margin outside 0..1, margin > 95 % with hardware, CAC ≤ 0, reachable share > 30 % (also capped at 30 % for the computation), negative costs, market scale or capital intensity contradicting the numbers, LTV/CAC > 20. Missing or non-positive revenue/market size scores 0.

**Critic** (optional, `get_model_name("critic")`, one call per batch, skip with `--no-critic`): tries to refute each candidate and returns killer risks, a verdict (survives / weakened / refuted) and corrected inputs. The score uses the critic's inputs when it gives a full set, plus a penalty (weakened 0.15, refuted 0.4, +0.05 per killer risk up to 0.15, total at most 0.6). Both the original and the used inputs are stored.

**Be clear about what this is**: every number the business model uses is an estimate made by the language model itself, possibly corrected by another call of a language model. The score ranks those estimates consistently and makes the assumptions visible. It is a structured brainstorming aid, not a market validation.

## Running it

```bash
# Offline, deterministic (fake generator; materials uses the mock pipeline)
python -m vectornaut.explorer --profile materials --query "Reduce drag of a ship hull coating" --rounds 3 --batch 6 --mock
python -m vectornaut.explorer --profile business --query "Reduce food waste in cities" --rounds 3 --batch 6 --mock

# Live (needs GEMINI_API_KEY)
python -m vectornaut.explorer --profile business --query "..." --rounds 3 --batch 6 --seed 7 \
    --strategy-weights refine=0.2,fill_gap=0.4,extrapolate=0.2,combine=0.1,explore=0.1

# Answer every model call by hand (same pause/resume protocol as for the pipeline)
python -m vectornaut.llm_replay --explorer --session runs/explore1 --profile business \
    --query "..." --rounds 2 --batch 4
```

Options: `--seed`, `--strategy-weights`, `--out DIR` (reports; default `<archive folder>/reports`), `--archive NAME` (separate map under `explorer/<profile>/<NAME>/`), `--allow-new-query`, `--epochs` and `--opt-rounds` (materials pipeline budget per candidate; defaults 40 and 1), `--no-critic`, `--min-slope`, `--verbose`.

The archive lives at `VECTORNAUT_DATA_DIR/explorer/<profile>/archive.json` and grows across runs. It remembers its query: scores for different requests are not comparable, so a different query is refused unless you pass `--archive NAME` (separate map) or `--allow-new-query`. Model and thinking level are configured like the pipeline stages: `VECTORNAUT_MODEL_EXPLORER`, `VECTORNAUT_MODEL_CRITIC`, `VECTORNAUT_THINKING_EXPLORER`, `VECTORNAUT_THINKING_CRITIC` (default thinking: medium).

With `llm_replay --explorer`, `--rounds` counts explorer rounds. Each invocation rebuilds the archive from scratch in `<session>/data/explorer` and replays the recorded answers, so the search orders and prompts are the same on every rerun. A materials candidate needs the generator call plus the pipeline calls (formulator, auditor, optimizer if `--opt-rounds` > 1, synthesizer) per candidate, so keep `--batch` small when answering by hand.

## Files

| File | Content |
|---|---|
| `explorer/descriptors.py` | Axes, vocabulary validation, cell keys, distance, neighbours |
| `explorer/archive.py` | JSON archive (entries, elites, proposal density, targets, infeasible cells, round log), atomic writes |
| `explorer/strategies.py` | refine, fill_gap, extrapolate, combine, explore, seed; scheduler |
| `explorer/schemas.py` | Response schemas of the generator and critic calls |
| `explorer/generator.py` | Prompt, model call, candidate checks |
| `explorer/profiles/materials.py`, `business.py` | Axes, prompt text, sanity checks, evaluators, mock generators |
| `explorer/report.py` | Markdown map/round reports, JSON export |
| `explorer/run.py` | Loop and CLI (`python -m vectornaut.explorer`) |

Tests: `tests/test_explorer_strategies.py` (descriptors, archive, strategies, scheduler), `tests/test_explorer_business.py` (unit economics with hand-computed numbers, sanity flags, critic scoring, candidate checks), `tests/test_explorer_loop.py` (mock loops for both profiles, determinism, replay pause/resume, the pipeline concept hook). They run offline in about 10 seconds.

## Limitations

- **The generator's prior**: candidates come from a language model. Even with targeted orders, it may drift back to familiar ideas or label a familiar idea with the target cell. The `on_target` flag and the proposal-density map make this visible, but do not remove it.
- **Business scores are self-estimates** (see above). A high score means "consistent and plausible by the model's own estimates", nothing more.
- **Materials scores inherit the pipeline's limits**: simplified steady 1D/2D models, and the current gain computation is 0 or unavailable outside slip-flow problems (see `docs/LIVE_PATH_FINDINGS.md`). Until metric and baseline are fixed, many live materials scores will rest on validity plus the discounted estimate.
- **The axes are hand-chosen** and fixed per profile. They decide what counts as "different"; two ideas in one cell compete even if they differ in ways no axis captures. Changing the axes needs a new archive.
- **Descriptors are assigned by the model** from a closed vocabulary. Validation catches values outside the vocabulary, not wrong but allowed values.
- **Trends are fitted on few points** (two elites are enough). They are hints about where to look next, not statistics.
- **One infeasibility claim closes a cell.** A wrong "impossible" from the model hides a region until the entry is removed from the archive by hand.
- Mock mode checks the machinery (scheduling, archive, reports), not idea quality: its landscape is synthetic.
