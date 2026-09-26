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
2. **Generate** one candidate per order in one model call (`get_model_name("explorer")`, schema `MaterialsCandidateBatch` / `BusinessCandidateBatch`). The prompt walks through function analysis → mechanism classes → analogue search → candidates, lists nearby archived titles and textbook solutions to avoid, and allows the answer "this target cell cannot contain a working concept" (`target_feasible: false` + reason). Order contexts are shortened field by field, so the JSON in the prompt stays valid and ids, titles, scores and main risks survive (`compact_context`, limit 2000 characters per order).
3. **Request analysis**: step 1 of the answer also lists the request's **requirements** (short name + one-line criterion, e.g. `non_toxic_antifouling: keeps fouling off without releasing biocides`) and, for materials, the **objective and baseline statements** and the **relevant governing quantities**. The first non-empty requirement list, objective statement and baseline statement are stored on the archive (`request_analysis`) and stay fixed, so scores of later rounds remain comparable; later generator prompts, the critic and the pipeline (via the concept text) see them. Relevant values are validated against the vocabulary and accumulate.
4. **Check** each candidate deterministically: descriptors from the closed vocabulary (case and spacing are tolerated, unknown values reject the candidate, nothing is guessed), required fields, finite numbers, duplicate titles, profile sanity checks.
5. **Evaluate** the valid candidates (materials: pipeline per candidate, then one critic call per batch; business: unit economics plus one critic call) and store every candidate in the archive, including rejected and failed ones (they count towards the proposal density). An elite is replaced only by stronger evidence or, at equal evidence, by a strictly higher score (see evidence tiers below).
6. **Report**: `round_NNN.md`, the cumulative `map.md` and `archive_export.json`.

A candidate whose descriptors do not match its target is kept in the cell it really belongs to (`on_target: false`). An honest label is worth more than a hit.

## Strategies

All strategies are pure functions of the archive and a seeded random generator (`vectornaut/explorer/strategies.py`), so a run is reproducible for a given seed.

| Strategy | Target | Notes |
|---|---|---|
| `refine` | an elite's own cell | Mutates the elite; elites with high scores and few earlier refinements are preferred. The order names the weakest score component, the stated main risk, the critic's main objection and weakly covered requirements. |
| `fill_gap` | an empty cell at distance 1 from elites | Priority = source value + 0.05 per additional *axis* along which the gap has elite neighbours (or for being bracketed on both sides of an ordinal axis; several elites that differ from the gap on the same nominal axis earn nothing) + 0.25 × under-exploration of the changed value − 0.2 × log(1 + earlier attempts). The **source** is the neighbour with the best source value (see *source rotation*); gaps are picked one at a time and re-ranked after each pick. |
| `extrapolate` | one step beyond the explored edge of an ordinal axis | Fits a least-squares line of elite score vs. ordinal index, holding the other axes fixed ("slice") or taking the best score per value ("marginal", gives a partial target). Only elites of one evidence tier are fitted (simulated, or unranked for business); estimates and implausible gains never form a trend. Needs at least 3 points (`min_trend_points`) and r² ≥ 0.5 (`min_r2`). Skipped when there are too few points, the slope is below `--min-slope`, the score drops at the edge (the optimum is inside the explored range), the fit is poor, at the end of the scale, or when the target is infeasible. Every round report says why it did or did not fire (each fitted trend with status and reason, e.g. `poor_fit: r2 0.20 < 0.5`; stored in the round log under `strategy_notes.extrapolate`). |
| `combine` | a cell mixing the descriptors of two distant elites | Picks the pair with the largest descriptor distance (then the highest joint score) that was not combined before; prefers an empty mixed cell with the most under-explored values, **among compatible mixtures only**: the child's pair on the profile's compatibility axes (materials: `mechanism_class` × `length_scale`) produced an evaluated concept somewhere, or the child lies within distance 2 of both parents (`combine_max_parent_distance`). A pair that only appears in infeasible reports is never used. A parent pair without any compatible mixture is skipped. (The recorded failure: a mm-scale mechanism moved to nm.) Profiles without compatibility axes (business) skip this check. |
| `diversify` | the least-proposed value of the least diverse axis, one change away from an elite | Axes are taken in order of proposal concentration (share of the most common value); on relevance axes only relevant values count. The source elite is the closest one on an ordinal axis, then the best evidence tier and the best source value (rotation), so the top elite does not seed every order. |
| `explore` | a random cell nobody has proposed or targeted | Restricted to relevant values of the relevance axes (materials: `governing_quantity`, from the function analysis plus the values of elites; nothing known yet → no explore order, e.g. at cold start). Weight = Π 1 / (1 + proposals with that value) over the other axes × exp(−`explore_distance_decay` · (d − 1)), d = distance to the nearest elite (default decay 1), × 0.1 if the cell's compatibility pair was only ever reported infeasible. Cells farther than `explore_max_distance` (3) from every elite are skipped while nearer ones are left. The order names the nearest concept that worked. |
| `seed` | no target | Cold start: the model proposes freely. This samples where ideas are usually proposed. |

Distance: a differing nominal axis counts 1, an ordinal axis counts its number of steps. "Distance 1" is one nominal change or one ordinal step.

**Source rotation.** In the recorded guided run, fill_gap and diversify always started from the top elite, and `graded_stiffness/mm` took 67 % of all proposals. Now an elite's value as a source is `score / 100 − source_penalty · log(1 + uses)` (`source_penalty` 0.1), where *uses* counts how often it already served as a source: the first parent of every earlier entry (both parents of a combine entry; refinements are counted separately) plus the orders already scheduled in the same batch (also across fallback calls). Lower-ranked elites therefore get their turn.

**Origin changes.** A fill_gap or diversify order that changes an origin axis (materials: `inspiration_origin`) tells the generator that the mechanism must really come from a system of the new origin; the parent's physics with a new origin label is a *relabelled analogue*. The critic flags such candidates (`relabelled_analogue`) and their score is halved.

**Under-exploration** of an axis value = concentration(axis) / (1 + proposals with that value), where concentration is the share of all proposals held by the axis' most common value. In the recorded hull-coating run every proposal had `inspiration_origin=animal` (concentration 1), so an unproposed origin scores 1, while an unproposed mechanism class on a mixed axis scores about 0.5. This steers `fill_gap`, `combine` and `diversify` towards the axis the generator's prior neglects. **Relevance**: fill-gap targets never move a relevance axis to a value that is neither named by the function analysis nor held by an elite.

The scheduler splits the batch by weight (default `refine=0.2, fill_gap=0.25, extrapolate=0.15, combine=0.15, diversify=0.15, explore=0.1`) with a largest-remainder rule; leftover slots are drawn with the seeded generator. Slots a strategy cannot fill fall back to other strategies (`fill_gap, diversify, refine, extrapolate, combine, explore, seed`); `seed` always succeeds. No two orders in a batch target the same cell. Without any elites only `seed` and `explore` run.

**Infeasible cells.** If the generator says a target cell cannot contain a working concept, or the pipeline rejects a concept as physically infeasible, the cell (or partial pattern) is stored under `infeasible` with the reason. No strategy targets it again (except refining an existing elite). This is information about the map, not a failure.

## Profiles

### materials

| Axis | Kind | Values |
|---|---|---|
| `mechanism_class` | nominal | interfacial_slip, flow_redirection, trapped_gas_or_liquid, porous_transport, graded_stiffness, architected_lattice, radiative_control, phase_change, electrostatic_field, other |
| `length_scale` | ordinal | nm < sub_um < um < 10_um < 100_um < mm < cm_plus |
| `inspiration_origin` | nominal | plant, animal, microbe, geology, atmosphere_ocean, technology, other |
| `governing_quantity` | nominal | wall_shear, flow_rate, heat_flux, temperature, deflection, stress, fouling_adhesion, degradation_rate, field_strength, other |

`governing_quantity` is the quantity the simulation computes for *this* concept, not necessarily the request's objective: a foul-release coating is simulated on `fouling_adhesion` (adhesion or release stress of foulers, lower = easier release), a wear or depletion concept on `degradation_rate` (loss of function per time). Both values were added in archive version 3 (see *Archive versions* below).

**Objective and baseline (problem framing).** Both role-play runs of the hull-coating query found that no coating lowers the drag of a *clean* hull by more than about 0–2 %, and both drifted to fouling release, which is the real lever: fouling raises hull drag massively over months and years. With a "clean standard coating" baseline, fouling-control concepts scored about 0 on the main benefit by construction. Step 1 of the generator therefore states:

- `objective_statement`: the main benefit as it applies over the stated service life and conditions (e.g. "time-averaged hull friction drag over a 5-year docking interval, including the effect of fouling"), not the benefit of a freshly applied surface when the request asks for years of service;
- `baseline_statement`: the conventional solution in the same condition (e.g. "conventional biocide-free silicone foul-release coating after 12–24 months in service").

Both are stored once per archive (like the requirements), shown in every later generator prompt, given to the critic, and written into the concept text for the formulator and auditor. A deterministic check (`framing_warnings`) warns in prompt and report when the request mentions years, durability or fouling but the stored statements name no service condition.

**Candidates** state their **baseline** explicitly (`baseline` field): the map's baseline statement, never a parent, neighbour or sibling concept and never a fouled or untreated surface. A missing baseline is flagged (`baseline_unstated`), not rejected. `back_of_envelope.value` estimates the improvement of the *objective* in percent.

**Evaluation**: every candidate becomes a `MinerConceptOutput` and runs through the normal pipeline (formulator → auditor → solver → validator → optimizer → synthesizer) through the `PipelineRunRequest.concept` hook with `max_concept_attempts=1`, so exactly this concept is judged and never swapped for a re-mined one. `physical_mechanism` carries, clearly labelled, the summary, the comparison baseline, the map's objective and conventional baseline ("compare against this, never against a parent or sibling concept"), the candidate's estimate (value and formula), the main risk and the novelty statement, so formulator and auditor see the candidate's assumptions and simulate against the conventional solution. The raw pipeline result (plus the critic review and the score breakdown) is saved under `explorer/<profile>/results/<entry>.json`.

**Critic** (`get_model_name("critic")`, one call per batch after the pipeline, schema `MaterialsCriticBatch`, skip with `--no-critic`): sees the objective and baseline statements, and per candidate its concept, cell, inspiration, search order and parent concepts (title, cell, summary), stated baseline and estimate, the formulated equation and boundary conditions, the audited parameters, the simulated quantity (and whether the function analysis named it), the metric and simulated baseline, and the simulated gain (marked when it was rejected as implausible). It returns per candidate:

- `plausible_simulated_benefit_pct`: the real-world improvement of the candidate's *own* simulated quantity against the conventional baseline;
- `plausible_objective_gain_pct`: the candidate's contribution to the stated *objective* against the stated baseline;
- `simulated_quantity_relevant` (true/false/null), `relabelled_analogue` (+ reason), `baseline_conventional` (+ issue);
- `key_assumption_issues`, `killer_risks` and `requirement_coverage` (0..1 plus a one-line reason per stored requirement).

A failed critic call keeps the simulations and flags the entries `critic_failed`.

**Numbers used for scoring** (all stored in `score_breakdown`: `simulated_gain_pct`, `critic_simulated_benefit_pct`, `simulated_benefit_used_pct`, `critic_objective_gain_pct`, `estimated_gain_pct`, `objective_gain_pct`, with their sources):

- The simulated gain is *usable* only if it is reported, finite, not declared unavailable (`gain_basis` `none`/`n/a`), within ±500 % and the validator's `physics_performance_gain_sanity` check did not fail (read from `validation.checks`). Otherwise: flag `implausible_gain` (sanity failure, non-finite, beyond ±500 %) or `gain_unavailable`.
- It is *relevant* unless the critic says `simulated_quantity_relevant: false` or the function analysis named relevant quantities and this one is not among them (flag `simulated_quantity_irrelevant`; the number is shown but not scored). Without any relevance information it counts.
- Usable and relevant → tier **simulated**; the simulated benefit is checked against the critic's `plausible_simulated_benefit_pct`: if they **differ strongly**, the lower one is used and the entry is flagged `model_assumption_sensitive`. Two numbers differ strongly when |a − b| > max(1 percentage point, 50 % of max(|a|, |b|)); a sign difference counts once |a − b| > 1 pp. So −28.7 vs −5 is flagged, 0.02 vs 0 is not.
- Otherwise → tier **estimated**: the simulated part is 0.
- The objective gain is the critic's `plausible_objective_gain_pct`; without it the candidate's own estimate (flag `objective_gain_unchecked`); without both it is 0 (flag `objective_gain_missing`). It counts fully only in the simulated tier with a critic value; otherwise at half weight.

**Score** (0–100):

```
score            = 100 * gate * (0.35 * objective_score + 0.15 * simulated_score + 0.5 * requirement_coverage)
                   * relabel_factor
gate             = validator_score * (1 pass | 0.8 warn | 0 fail or missing)
objective_score  = (1 - exp(-objective_gain_pct / 10)) * d        (0 for gains <= 0)
                   d = 1 if a critic judged it next to a relevant simulation (tier simulated), else 0.5
simulated_score  = 1 - exp(-simulated_benefit_pct / 20)           (0 for gains <= 0; 0 in the estimated tier)
relabel_factor   = 0.5 if the critic calls the concept a relabelled analogue, else 1
estimated tier: score <= 50
requirement_coverage = mean critic rating over the stored requirements
                       (unrated requirements count 0.5; no critic: 0.5, flag requirements_unrated)
```

Defaults and why: requirement coverage keeps half the weight (the request's hard constraints: no biocides, years of service). The objective (0.35) outweighs the concept's own simulated quantity (0.15), because the simulated number is a steady 1D/2D model of one mechanism while the objective is what the request asks for; the simulated part keeps the simulation evidence in the ranking (a foul-release coating that halves release stress earns up to 15 points even when the critic estimates a small objective gain). Objective gains are typically single-digit percentages (scale 10 %), simulated benefits often tens of percent (scale 20 %). `score_breakdown` holds `formula` (generated from the constants in `vectornaut/explorer/profiles/materials.py`), `weights`, `components` (gate, objective_score, simulated_score, requirement_coverage), `objective_discount`, `relabel_factor` and `contributions` (points from objective, simulated benefit and requirements; they add up to the score unless the estimated cap applies).

**Evidence tiers**: `evidence_rank` 2 = simulated, 1 = estimated, 0 = estimated after an implausible simulation. Tier "simulated" is used only when the simulated number actually enters the score (usable and relevant). Within a cell an entry never replaces an elite of a higher rank, and an entry of a higher rank replaces a lower-ranked elite regardless of score; reports list elites by rank first. Trends use only simulated scores.

**Flags** (`score_breakdown.flags`): `implausible_gain`, `gain_unavailable`, `simulated_quantity_irrelevant`, `model_assumption_sensitive`, `objective_gain_unchecked`, `objective_gain_missing`, `relabelled_analogue`, `baseline_not_conventional` (the critic says the candidate's or the simulation's baseline is not the conventional one, e.g. a parent concept), `requirements_unrated`, `requirements_partially_rated`, `critic_missing`, `critic_failed`, `baseline_unstated`.

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

Options: `--seed`, `--strategy-weights`, `--out DIR` (reports; default `<archive folder>/reports`), `--archive NAME` (separate map under `explorer/<profile>/<NAME>/`), `--allow-new-query`, `--epochs` and `--opt-rounds` (materials pipeline budget per candidate; defaults 40 and 1), `--no-critic` (both profiles; materials: no critic check of the simulated benefit, the objective gain is the candidate's own estimate at half weight, requirements unrated), `--min-slope`, `--verbose`.

The archive lives at `VECTORNAUT_DATA_DIR/explorer/<profile>/archive.json` and grows across runs. It remembers its query: scores for different requests are not comparable, so a different query is refused unless you pass `--archive NAME` (separate map) or `--allow-new-query`. Model and thinking level are configured like the pipeline stages: `VECTORNAUT_MODEL_EXPLORER`, `VECTORNAUT_MODEL_CRITIC`, `VECTORNAUT_THINKING_EXPLORER`, `VECTORNAUT_THINKING_CRITIC` (default thinking: medium).

**Archive versions.** The archive file carries `version` (currently 3). An older archive whose axes (names and values) equal the profile's is migrated on load (`version` raised, `migrated_from` recorded); business archives of version 2 load this way. An archive built with another vocabulary is refused with a message naming the difference, because its cells and scores are not comparable: materials archives of version 2 lack the `governing_quantity` values `fouling_adhesion` and `degradation_rate` and were scored with the old one-gain formula. Start a new map with `--archive NAME`, or move the old folder away. A newer archive than the code is refused too.

With `llm_replay --explorer`, `--rounds` counts explorer rounds. Each invocation rebuilds the archive from scratch in `<session>/data/explorer` and replays the recorded answers, so the search orders and prompts are the same on every rerun. A materials round needs the generator call, the pipeline calls (formulator, auditor, optimizer if `--opt-rounds` > 1, synthesizer) per candidate and one `MaterialsCriticBatch` call at the end, so keep `--batch` small when answering by hand. Sessions recorded before the critic existed need `--no-critic` to replay. Sessions recorded before archive version 3 (e.g. the v2 hull-coating role-plays) do not replay: the generator and critic prompts and schemas changed (objective/baseline statements, two critic numbers).

## Files

| File | Content |
|---|---|
| `explorer/descriptors.py` | Axes, vocabulary validation, cell keys, distance, neighbours |
| `explorer/archive.py` | JSON archive (entries, elites ranked by evidence then score, proposal density, targets, infeasible cells, request analysis incl. objective and baseline, source counts, round log), versioning, atomic writes |
| `explorer/strategies.py` | refine, fill_gap, extrapolate, combine, diversify, explore, seed; source rotation, compatibility pairs, scheduler with diagnostics |
| `explorer/schemas.py` | Response schemas of the generator and critic calls |
| `explorer/generator.py` | Prompt, JSON-safe context shortening, model call, candidate checks, request analysis |
| `explorer/profiles/materials.py`, `business.py` | Axes, prompt text, sanity checks, evaluators and critics, mock generators and critics |
| `explorer/report.py` | Markdown map/round reports (objective and baseline; elites with tier, score split objective / simulated / requirements, requirement coverage and flags; requirement table; proposal density vs score; values never proposed; why extrapolation fired or not), JSON export |
| `explorer/run.py` | Loop and CLI (`python -m vectornaut.explorer`) |

Tests: `tests/test_explorer_strategies.py` (descriptors, archive incl. version migration and vocabulary refusal, source counts, evidence ranking, strategies incl. diversity bonus, relevance filter and trend rules, combine compatibility, explore adjacency, source rotation, origin-change instructions, extrapolation diagnostics, scheduler), `tests/test_explorer_business.py` (unit economics with hand-computed numbers, sanity flags, critic scoring, candidate checks), `tests/test_explorer_materials.py` (fouling vocabulary, gain sanity, tiers and ceiling, two-number critic scoring, quantity relevance, relabel penalty, non-conventional baseline flag, `differs_strongly` cases, requirement coverage, framing storage and warnings, the critic call and prompt, concept hand-over incl. objective/baseline, JSON-safe context shortening, request analysis), `tests/test_explorer_loop.py` (mock loops for both profiles, framing flowing into context, prompts and reports, determinism, old-archive refusal, replay pause/resume, the pipeline concept hook). They run offline in about 15 seconds.

The map report shows, besides the elites and the two-axis map: the stored objective and baseline statements (with framing warnings), the stored requirements and relevant values, a requirement-coverage table (critic ratings per elite), the proposal-density-vs-score view per axis, the values of every axis that were never proposed (with the share of the most common value), gap targets with their under-exploration term, trends with their status, and flag counts.

## Limitations

- **The generator's prior**: candidates come from a language model. Even with targeted orders, it may drift back to familiar ideas or label a familiar idea with the target cell. The `on_target` flag and the proposal-density map make this visible, but do not remove it.
- **Business scores are self-estimates** (see above). A high score means "consistent and plausible by the model's own estimates", nothing more.
- **Materials scores inherit the pipeline's limits**: simplified steady 1D/2D models whose gains can hinge on one free modelling choice (an "equivalent laminar gap" in a Couette model of turbulent drag, a chosen slip length). The critic's plausible simulated benefit caps such results, but it is itself a model estimate; `model_assumption_sensitive` marks where the two disagree.
- **The objective gain is a critic estimate**: for fouling-control concepts no simulation computes time-averaged drag; the objective part of the score is the critic's judgement of how a simulated release-stress benefit translates into it. The simulated part (15 %) keeps the simulation evidence, but a generous critic moves the ranking.
- **The framing is fixed once**: the first objective and baseline statements stay for the whole map. `framing_warnings` catches a clean-surface baseline for a multi-year request by keywords only; a wrong but plausible-sounding framing needs a new archive.
- **Relabelled analogues are detected by the critic**, not by rule: it sees the order's parents, but a subtle copy may pass.
- **Compatibility is judged on one pair of axes** (materials: mechanism × length scale). Other incompatible mixtures (e.g. an origin that has no such mechanism) still reach the generator, which may answer `target_feasible=false`.
- **Requirement coverage is a model rating**: the critic rates each requirement 0..1 from the description and the model, not from tests. Without a critic every requirement counts 0.5, so uncritiqued scores are comparable with each other but not with critiqued ones.
- **The axes are hand-chosen** and fixed per profile. They decide what counts as "different"; two ideas in one cell compete even if they differ in ways no axis captures. Changing the axes needs a new archive.
- **Descriptors are assigned by the model** from a closed vocabulary. Validation catches values outside the vocabulary, not wrong but allowed values.
- **Trends are fitted on few points** (three simulated elites are enough). They are hints about where to look next, not statistics.
- **One infeasibility claim closes a cell.** A wrong "impossible" from the model hides a region until the entry is removed from the archive by hand.
- **Relevance comes from the function analysis.** A governing quantity the generator did not name (and no elite holds) is never explored; a too narrow list narrows the search.
- Mock mode checks the machinery (scheduling, archive, reports), not idea quality: its landscape is synthetic.
