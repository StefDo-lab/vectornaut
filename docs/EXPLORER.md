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
3. **Request analysis**: step 1 of the answer also lists the request's **requirements** (short name + one-line criterion + priority, e.g. `non_toxic_antifouling [must]: keeps fouling off without releasing biocides`; `must` for what the request states explicitly, e.g. "ohne", "mindestens", "bionisch", `nice` for implied or desirable properties; default `must`) and, for materials, the **objective and baseline statements**, an optional numeric **target gain** (`target_gain_pct`, e.g. 20 for "> 20 %"), the **relevant governing quantities** and the **relevant mechanism classes** (`relevant_mechanism_classes`, optional). The first non-empty requirement list, objective statement, baseline statement and target gain are stored on the archive (`request_analysis`) and stay fixed, so scores of later rounds remain comparable; later generator prompts, the critic and the pipeline (via the concept text) see them. Relevant values are validated against the vocabulary and accumulate.
4. **Check** each candidate deterministically: descriptors from the closed vocabulary (case and spacing are tolerated, unknown values reject the candidate, nothing is guessed), required fields, finite numbers, duplicate titles, profile sanity checks.
5. **Evaluate** the valid candidates (materials: pipeline per candidate, then one critic call per batch; business: unit economics plus one critic call) and store every candidate in the archive, including rejected and failed ones (they count towards the proposal density). An elite is replaced only by stronger evidence or, at equal evidence, by a strictly higher score (see evidence tiers below).
6. **Report**: `round_NNN.md`, the cumulative `map.md` and `archive_export.json`.

A candidate whose descriptors do not match its target is kept in the cell it really belongs to (`on_target: false`). An honest label is worth more than a hit.

## Strategies

All strategies are pure functions of the archive and a seeded random generator (`vectornaut/explorer/strategies.py`), so a run is reproducible for a given seed.

| Strategy | Target | Notes |
|---|---|---|
| `refine` | an elite's own cell | Mutates the elite; elites with high scores and few earlier refinements are preferred. The order names the weakest score component, the stated main risk, the critic's main objection and weakly covered requirements. |
| `fill_gap` | an empty cell at distance 1 from elites | Priority = source value + 0.05 per additional *axis* along which the gap has elite neighbours (or for being bracketed on both sides of an ordinal axis; several elites that differ from the gap on the same nominal axis earn nothing) + 0.25 × under-exploration of the changed value − 0.2 × log(1 + earlier attempts). The **source** is the neighbour with the best source value (see *source rotation*); gaps are picked one at a time and re-ranked after each pick. Gaps with a non-preferred value (see *preferred values*), with a mechanism class that is not relevant (see *mechanism relevance*) or whose compatibility pair (materials: `mechanism_class` × `length_scale`) only appears in infeasible reports are skipped. The report's "promising under-explored cells" use the same list. |
| `extrapolate` | one step beyond the explored edge of an ordinal axis | Fits a least-squares line of elite score vs. ordinal index, holding the other axes fixed ("slice") or taking the best score per value ("marginal", gives a partial target). Marginal fits stay within one value of the profile's group axes (materials: one `mechanism_class`, so different mechanisms are never fitted together; business: no grouping). Only elites of one evidence tier are fitted (simulated, or unranked for business); estimates, implausible gains and proxies by construction never form a trend. Needs at least 3 points (`min_trend_points`) and r² ≥ 0.5 (`min_r2`). Skipped when there are too few points, the slope is below `--min-slope`, the score drops at the edge (the optimum is inside the explored range), the fit is poor, at the end of the scale, when the target is infeasible or uses a non-preferred value (`not_preferred`). **Slots**: the scheduler gives extrapolate slots only if a qualifying *slice* trend exists (same values on all other axes, ≥ 3 simulated points, r² ≥ 0.5, improving at the edge, next cell open); otherwise its weight goes to the other strategies (in the v3 role-play every planned slot fell back because no trend ever qualified). Every round report says why it did or did not fire (each fitted trend with status and reason, e.g. `poor_fit: r2 0.20 < 0.5`, and the slot decision; stored in the round log under `strategy_notes.extrapolate`, `slots`). |
| `combine` | a cell mixing the descriptors of two distant elites | Picks the pair with the largest descriptor distance (then the highest joint score) that was not combined before; prefers an empty mixed cell with the most under-explored values, **among compatible mixtures only**: the child's pair on the profile's compatibility axes (materials: `mechanism_class` × `length_scale`) produced an evaluated concept somewhere, or the child lies within distance 2 of both parents (`combine_max_parent_distance`). A pair that only appears in infeasible reports is never used. A parent pair without any compatible mixture is skipped. (The recorded failure: a mm-scale mechanism moved to nm.) Profiles without compatibility axes (business) skip this check. **Anchors** (materials: `governing_quantity`, then `mechanism_class`): the child takes the governing quantity from the *stronger* parent (evidence tier first, then score), so it never inherits the quantity of an estimated-tier parent when the other one is simulated; it keeps the weaker parent's mechanism only if that mechanism produced a simulated concept on this quantity somewhere, otherwise the mechanism comes from the stronger parent too (context `anchored`). (The v3 case: an estimated coral-mucus parent, 15.4, gave `degradation_rate` to a mixture with the simulated fish-mucus skin.) Mixtures with non-preferred values are skipped. |
| `diversify` | the least-proposed value of the least diverse axis, one change away from an elite | Axes are taken in order of proposal concentration (share of the most common value); on relevance axes (hard and soft) only relevant values count, on axes with preferences only preferred values, and targets with a non-preferred value, an irrelevant mechanism class or a compatibility pair only reported infeasible are skipped. The source elite is the closest one on an ordinal axis, then the best evidence tier and the best source value (rotation), so the top elite does not seed every order. |
| `explore` | a random cell nobody has proposed or targeted | Restricted to relevant values of the relevance axes (materials: `governing_quantity`, from the function analysis plus the values of elites; nothing known yet → no explore order, e.g. at cold start). Weight = Π 1 / (1 + proposals with that value) over the other axes × exp(−`explore_distance_decay` · (d − 1)), d = distance to the nearest elite (default decay 1), × 0.1 if the cell's compatibility pair was only ever reported infeasible, × 0.1 (`nonpreferred_explore_factor`) if the cell has a non-preferred value, × 0.1 (`soft_irrelevant_explore_factor`) if its mechanism class is not relevant (the order says so in both cases). Cells farther than `explore_max_distance` (3) from every elite are skipped while nearer ones are left. The order names the nearest concept that worked. |
| `seed` | no target | Cold start: the model proposes freely. This samples where ideas are usually proposed. |

Distance: a differing nominal axis counts 1, an ordinal axis counts its number of steps. "Distance 1" is one nominal change or one ordinal step.

**Source rotation.** In the recorded guided run, fill_gap and diversify always started from the top elite, and `graded_stiffness/mm` took 67 % of all proposals. Now an elite's value as a source is `score / 100 − source_penalty · log(1 + uses)` (`source_penalty` 0.1), where *uses* counts how often it already served as a source: the first parent of every earlier entry (both parents of a combine entry; refinements are counted separately) plus the orders already scheduled in the same batch (also across fallback calls). Lower-ranked elites therefore get their turn.

**Origin changes.** A fill_gap or diversify order that changes an origin axis (materials: `inspiration_origin`) tells the generator that the mechanism must really come from a system of the new origin; the parent's physics with a new origin label is a *relabelled analogue*. The critic flags such candidates (`relabelled_analogue`) and their score is halved.

**Preferred values** (materials: `inspiration_origin`). If the request or one of its stored requirements asks for a biological model (keywords: bionic, bionisch, Bionik, bio-inspired, biomimetic, "derived from a biological model", ...), only `plant`, `animal` and `microbe` are preferred (`request_analysis.preferred`, with the reason; derived deterministically by the profile each round and shown in the prompt and the map report). Unlike relevant values, elites do not widen them. fill_gap, diversify, combine and extrapolate never target another origin; explore reaches the others at 0.1 × weight; refine stays in an elite's own cell. In the v3 role-play the under-exploration bonus pushed diversify and fill_gap to `atmosphere_ocean`, `geology` and `technology`, whose concepts then failed the `bionic_mechanism` requirement (critic 0.0–0.1). Without such keywords there is no preference.

**Under-exploration** of an axis value = concentration(axis) / (1 + proposals with that value), where concentration is the share of all proposals held by the axis' most common value. In the recorded hull-coating run every proposal had `inspiration_origin=animal` (concentration 1), so an unproposed origin scores 1, while an unproposed mechanism class on a mixed axis scores about 0.5. This steers `fill_gap`, `combine` and `diversify` towards the axis the generator's prior neglects. **Relevance**: fill-gap targets never move a relevance axis to a value that is neither named by the function analysis nor held by an elite.

The scheduler splits the batch by weight (default `refine=0.2, fill_gap=0.25, extrapolate=0.15, combine=0.15, diversify=0.15, explore=0.1`) with a largest-remainder rule; leftover slots are drawn with the seeded generator. Slots a strategy cannot fill fall back to other strategies (`fill_gap, diversify, refine, extrapolate, combine, explore, seed`); `seed` always succeeds. No two orders in a batch target the same cell. Without any elites only `seed` and `explore` run.

**Mechanism relevance** (soft relevance axis, materials: `mechanism_class`). The function analysis may name the mechanism classes that can plausibly deliver the request's benefit (`relevant_mechanism_classes`). Relevant are these plus the mechanism classes of the top 5 elites (`SOFT_RELEVANCE_TOP_ELITES`; unlike governing quantities, a weak elite does not widen the set). fill_gap and diversify only target relevant mechanism classes; explore reaches the others at 0.1 × weight. While none is named (old archives, or the model gives none), there is no restriction. (The facade case: fill_gap ordered `electrostatic_field/sub_um/plant`, and the report listed interfacial slip, graded stiffness and phase change as promising for a heat-flux request.)

**Infeasible cells.** If the generator says a target cell cannot contain a working concept, or the pipeline rejects a concept as physically infeasible, the cell (or partial pattern) is stored under `infeasible` with the reason. No strategy targets it again (except refining an existing elite). This is information about the map, not a failure. The generator may add `infeasibility_scope`, the axes the impossibility depends on (e.g. `["mechanism_class", "length_scale"]`: no origin or quantity can make it work); then the pattern of these values is closed (at least two axes of the target; a single axis is ignored and only the target closed, `Archive.infeasibility_pattern`; the record keeps `scope` and a note). Independently, fill_gap and diversify now skip cells whose compatibility pair only appears in infeasible reports, as combine did before. (The facade case: after `flow_redirection/sub_um/plant/heat_flux` was reported impossible because sub-micron features sit in the viscous sublayer, diversify targeted `flow_redirection/sub_um/animal/heat_flux` in the next round and got the same answer.)

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

Both are stored once per archive (like the requirements), shown in every later generator prompt, given to the critic, and written into the concept text for the formulator and auditor. An optional `target_gain_pct` (the numeric improvement the request, the objective or a requirement names) is stored the same way and sets the objective scale (see *score*). A deterministic check (`framing_warnings`) warns in prompt and report when the request mentions years, durability or fouling but the stored statements name no service condition.

**Candidates** state their **baseline** explicitly (`baseline` field): the map's baseline statement, never a parent, neighbour or sibling concept and never a fouled or untreated surface. A missing baseline is flagged (`baseline_unstated`), not rejected. `back_of_envelope.value` estimates the improvement of the *objective* in percent.

**Evaluation**: every candidate becomes a `MinerConceptOutput` and runs through the normal pipeline (formulator → auditor → solver → validator → optimizer → synthesizer) through the `PipelineRunRequest.concept` hook with `max_concept_attempts=1`, so exactly this concept is judged and never swapped for a re-mined one. `physical_mechanism` carries, clearly labelled, the summary, the comparison baseline, the map's objective and conventional baseline ("compare against this, never against a parent or sibling concept"), the candidate's estimate (value and formula), the main risk and the novelty statement, so formulator and auditor see the candidate's assumptions and simulate against the conventional solution. The raw pipeline result (plus the critic review and the score breakdown) is saved under `explorer/<profile>/results/<entry>.json`.

**Critic** (`get_model_name("critic")`, one call per batch after the pipeline, schema `MaterialsCriticBatch`, skip with `--no-critic`): sees the objective and baseline statements, and per candidate its concept, cell, inspiration, search order and parent concepts (title, cell, summary), stated baseline and estimate, the formulated equation and boundary conditions, the audited parameters, the simulated quantity (and whether the function analysis named it), the metric and simulated baseline, and the simulated gain (marked when it was rejected as implausible). It returns per candidate:

- `plausible_simulated_benefit_pct`: the real-world improvement of the candidate's *own* simulated quantity against the conventional baseline;
- `plausible_objective_gain_pct`: the candidate's contribution to the stated *objective* against the stated baseline;
- `conventional_equivalent_gain_pct`: the objective gain a *conventional* measure achieving the same physical effect would give (e.g. an equal-R layer of standard insulation for a concept whose benefit is added thermal resistance: cork, aerogel, an air cavity; 0 if the mechanism has no conventional equivalent). Only the gain beyond it counts;
- `simulated_quantity_relevant` (true/false/null), `relabelled_analogue` (+ reason), `baseline_conventional` (+ issue);
- `proxy_by_construction` (+ `proxy_reason`): the simulated gain follows directly from an input or baseline choice, not from modelled physics (the v3 case: a leaching flux of an oil-free design against an oil-containing baseline gave 78–100 %; a gain equal to an assumed friction ratio or settlement factor);
- `key_assumption_issues`, `killer_risks` and `requirement_coverage` (0..1 plus a one-line reason per stored requirement; the prompt marks each requirement `[must]` or `[nice]`).

The prompt is domain-neutral (e.g. "what is released into the environment", not "into the sea"; the examples cover release stress, wall shear and heat flux).

A failed critic call keeps the simulations and flags the entries `critic_failed`.

**Numbers used for scoring** (all stored in `score_breakdown`: `simulated_gain_pct`, `critic_simulated_benefit_pct`, `simulated_benefit_used_pct`, `critic_objective_gain_pct`, `estimated_gain_pct`, `objective_gain_pct`, with their sources):

- The simulated gain is *usable* only if it is reported, finite, not declared unavailable (`gain_basis` `none`/`n/a`), within ±500 % and the validator's `physics_performance_gain_sanity` check did not fail (read from `validation.checks`). Otherwise: flag `implausible_gain` (sanity failure, non-finite, beyond ±500 %) or `gain_unavailable`.
- It is *relevant* unless the critic says `simulated_quantity_relevant: false` or the function analysis named relevant quantities and this one is not among them (flag `simulated_quantity_irrelevant`; the number is shown but not scored). Without any relevance information it counts.
- Usable and relevant → tier **simulated**; the simulated benefit used is **min(simulation, critic's `plausible_simulated_benefit_pct`)** whenever the critic gave a number (a steady 1D/2D model is never scored above the critic's real-world estimate; in the v3 run the old "lower one only if they differ strongly by > 50 %" rule let the higher simulated number count for 8 of 13 elites). If they **differ strongly** the entry is also flagged `model_assumption_sensitive`: |a − b| > max(1 percentage point, 50 % of max(|a|, |b|)); a sign difference counts once |a − b| > 1 pp. So −28.7 vs −5 is flagged, 0.02 vs 0 is not.
- A proxy by construction (critic flag) keeps the tier, but its simulated part is 0 (flag `proxy_by_construction`) and it forms no trends.
- Otherwise → tier **estimated**: the simulated part is 0.
- The objective gain is the critic's `plausible_objective_gain_pct` minus its `conventional_equivalent_gain_pct` (floored at 0; flag `mostly_conventional_effect` when the conventional equivalent covers at least half of the plausible gain; both numbers are stored: `critic_objective_gain_pct`, `conventional_equivalent_gain_pct`, and the net value in `objective_gain_pct`); without a conventional value the plausible gain counts as it is. Without a critic number: the candidate's own estimate (flag `objective_gain_unchecked`); without both it is 0 (flag `objective_gain_missing`). It counts fully only in the simulated tier with a critic value; otherwise at half weight.

**Score** (0–100):

```
score             = 100 * gate * (0.45 * objective_score + 0.1 * simulated_score + 0.45 * requirement_score)
                    * relabel_factor * baseline_factor * must_factor
gate              = validator_score * (1 pass | 0.8 warn | 0 fail or missing)
objective_score   = (1 - exp(-objective_gain_pct / S)) * d        (0 for gains <= 0)
                    objective_gain_pct = max(0, critic objective gain - conventional-equivalent gain)
                    d = 1 if a critic judged it next to a relevant simulation (tier simulated), else 0.5
S (objective scale, per archive, recomputed after every round, see below)
                  = target_gain_pct / 2                            if a target gain is stated (0.5..50 %)
                  = 2 * median of the positive net critic objective gains of the archive (>= 3 values; 2..50 %)
                  = 2 %                                             otherwise
simulated_score   = 1 - exp(-simulated_benefit_pct / 20)          (0 for gains <= 0; 0 in the estimated tier;
                    simulated_benefit_pct = min(simulated, critic) 0 for a proxy by construction)
requirement_score = 0.5 * mean(all ratings) + 0.5 * min(ratings of the must-requirements)
                    (unrated requirements count 0.5; no critic: 0.5, flag requirements_unrated; no must: mean)
must_factor       = 1 - 0.5 * max(0, (0.3 - min must rating) / 0.3)   (0.15 -> 0.75, 0 -> 0.5; flag must_requirement_unmet)
baseline_factor   = 0.8 if the critic says the baseline is not the conventional one (flag baseline_not_conventional)
relabel_factor    = 0.5 if the critic calls the concept a relabelled analogue, else 1
estimated tier: score <= 50
```

`requirement_coverage` in the breakdown and the reports stays the plain mean; `requirement_score`, `must_min_coverage` and `must_factor` are stored next to it. Two digits are kept for S (e.g. 22 %, 6.6 %).

**Objective scale (archive version 5).** A gain of S scores 1 − 1/e = 0.63 of the objective part. The fixed 2 % of version 4 was tuned for hull drag (critic objective gains 0.3–1.5 %). In the facade-cooling role-play the critic's gains were 3–35 % of the heat load, so every concept got ~45 of 45 objective points; the ranking fell to requirement-rating noise and insulation in disguise ranked high. S is therefore set per archive: from a stated target (reaching it scores 0.86), else from the archive's own gains (the typical gain scores 0.39, as 1 % did at 2 % for the hull; the lower bound 2 % keeps the hull archive, median 0.8 %, at its version-4 scale), else 2 %. It is recomputed when the runner starts and after every round (a stated target at once); when it changes, every evaluated entry is re-scored from its breakdown and the elites are rebuilt, so all scores in one map use the same S. The archive stores `request_analysis.objective_scale` (`pct`, `source`: target / archive / default, `detail`, `history`); the map report prints it under the objective and in the score note, and the round report says when it changed. Round logs keep the scores after the round's re-scoring; the entry `outcome` (new elite, improved) stays as decided at insertion.

**Must-requirements.** A plain mean let the objective and the other ratings average a violated hard requirement away: in the facade baseline run the cork-oak cladding (bionic rating 0.15, critic objective gain 35 %, baseline not conventional) ranked first for a "bionisch" request. Now the weakest must-requirement enters the requirement score and, below 0.3, scales the whole score down; a gain measured against a non-conventional baseline costs 20 %.

**Re-scoring under version 5** (offline, through the version-4 → 5 migration; the new critic field `conventional_equivalent_gain_pct` and the requirement priorities did not exist, so they count as absent: no conventional equivalent, every requirement `must`). "min must" = the weakest requirement rating; ranks are elite ranks ("–" = not an elite).

*Facade guided* (5 rounds × 3). Version 5 scale: 20 % (2 × median 10 % of 13 positive critic gains); with a stated target of 20 % (the generator's requirement text said "roughly > 20 %"; no field existed): 10 %.

| entry | concept | critic obj. % | bionic | min must | v4 score (rank) | v5 (rank) | v5, target 20 % (rank) |
|---|---|---|---|---|---|---|---|
| e00012 | Birch-periderm glazed multi-row cell coat | 18 | 0.45 | 0.45 | 81.7 (1) | 58.9 (1) | 69.7 (1) |
| e00001 | Beetle-scale inorganic scattering coat | 20 | 0.35 | 0.35 | 81.1 (2) | 58.3 (2) | 68.7 (2) |
| e00005 | Cork-oak bark cladding with white mineral skin | 12 | 0.30 | 0.30 | 79.2 (3) | 48.1 (3) | 59.2 (3) |
| e00002 | Desert-snail double-shell cladding | 8 | 0.30 | 0.30 | 74.9 (4) | 39.9 (4) | 49.8 (5) |
| e00003 | Birch-bark self-renewing white coat (proxy) | 12 | 0.50 | 0.35 | 68.3 (7) | 39.9 (5) | 51.0 (4) |
| e00004 | Polar-bear aerogel-fibre roof underlayer | 10 | 0.25 | 0.25 | 73.3 (5) | 37.7 (6) | 47.5 (6) |
| e00008 | Tussah-silk nanovoid glass-fibre shade screen | 5 | 0.50 | 0.30 | 70.0 (6) | 33.0 (7) | 40.8 (7) |
| e00014 | Nanowood white insulating facade panel | 5 | 0.30 | 0.15 | 64.3 (8) | 20.3 (8) | 26.1 (8) |
| e00007 | Cactus-spine pile facade shade | 3 | 0.55 | 0.15 | 59.2 (9) | 17.2 (9) | 21.2 (9) |
| e00010 | Antistatic oxide-network white topcoat (proxy) | 1 | 0.00 | 0.00 | 33.5 (10) | 5.0 (10) | 6.1 (10) |

Elite scores now spread over 5–59 points instead of 34–82 (59–82 without the proxy), and 1 % of objective gain is worth about 1.4 points around the median gain instead of less than 0.2 points above 10 %. Spearman(score, critic objective gain) over the elites: 0.87 (v4) → 0.94 (v5) / 0.97 (target). The order of the top 3 stays: the gains of the insulation-like concepts (cork, snail, aerogel) can only drop once the critic reports their conventional equivalent, which these answers predate.

*Facade baseline* (15 seeds). Version 5 scale: 12 % (2 × median 6 %); with a target of 20 %: 10 %.

| entry | concept | critic obj. % | bionic | min must | flags | v4 score (rank) | v5 (rank) | v5, target 20 % (rank) |
|---|---|---|---|---|---|---|---|---|
| e00001 | Cyphochilus-scale inorganic disordered-network topcoat | 17 | 0.55 | 0.50 | – | 82.7 (2) | 67.8 (1) | 70.5 (1) |
| e00003 | Diatom-frustule silicate cooling paint | 12 | 0.30 | 0.30 | – | 81.0 (3) | 56.2 (2) | 59.2 (2) |
| e00010 | Self-renewing chalking mineral topcoat | 11 | 0.30 | 0.30 | – | 77.3 (4) | 52.8 (3) | 55.8 (3) |
| e00004 | Desert-snail white shell tile over sealed air cavity | 10 | 0.45 | 0.45 | baseline | 77.0 (5) | 43.5 (4) | 45.9 (4) |
| e00006 | Small-leaf convective shading screen | 6 | 0.40 | 0.30 | – | 74.3 (6) | 42.5 (5) | 45.1 (5) |
| e00012 | Cork-oak bark closed-cell cladding | 35 | 0.15 | 0.15 | baseline | 83.3 (1) | 41.7 (6) | 42.3 (6) |
| e00005 | Saguaro-rib sky-facing facade corrugation | 5 | 0.35 | 0.30 | – | 67.7 (9) | 37.3 (7) | 39.6 (7) |
| e00002 | Silver-ant prism-fibre mat | 8 | 0.70 | 0.20 | – | 72.2 (7) | 35.8 (8) | 38.2 (8) |
| e00007 | Termite-mound ventilated rainscreen | 5 | 0.25 | 0.25 | – | 71.5 (8) | 34.7 (9) | 36.9 (9) |
| e00009 | Thorny-devil night-sorption evaporative skin | 3 | 0.30 | 0.15 | – | 53.4 (10) | 17.6 (10) | 18.9 (10) |
| e00008 | Camel-coat insulation with PCM store | 1 | 0.30 | 0.10 | – | 37.6 (11) | 10.8 (11) | 11.3 (11) |
| e00014 | Bacterial self-healing calcite render | 0.5 | 0.70 | 0.02 | proxy, baseline | 32.6 (12) | 5.8 (12) | 6.0 (12) |
| e00013 | Nacre-like platelet coating (stress) | 2 | 0.60 | 0.05 | irrelevant, baseline | 39.9 (13) | 8.1 (13) | 8.4 (13) |

The cork cladding (bionic 0.15, gain against an uninsulated wall) falls from 1 to 6 (must factor 0.75, baseline factor 0.8); the Cyphochilus coat, the concept the reviewer judged best, is first. Spearman with the critic objective gain drops from 0.99 to 0.89, by design: the one concept with the largest gain fails a must-requirement.

*Hull coating, v3 archive* (the version-4 column is the v4 re-scoring with the six proxies flagged, last column of the table below). Version 5 scale: 2 % (2 × median 0.8 % = 1.6 %, raised to the 2 % floor), so the objective term is unchanged.

| entry | concept | critic obj. % | bionic | min must | v4 score (rank) | v5 (rank) |
|---|---|---|---|---|---|---|
| e00007 | Palm-stem anisotropic release layer | 1.0 | 0.55 | 0.30 | 47.7 (3) | 43.6 (1) |
| e00001 | Pilot-whale graded soft skin | 1.0 | 0.50 | 0.30 | 48.0 (2) | 43.5 (2) |
| e00005 | Mucin bottle-brush hydration skin | 1.0 | 0.60 | 0.20 | 45.5 (4) | 33.0 (3) |
| e00014 | Oil-free supersoft palm-stem layer (proxy) | 0.8 | 0.45 | 0.30 | 35.1 (7) | 31.7 (4) |
| e00015 | Pack-ice tiled skin on a soft bed | 1.5 | 0.10 | 0.10 | 52.0 (1) | 29.9 (5) |
| e00017 | Cyprid-scale wrinkled soft skin (proxy) | 0.5 | 0.40 | 0.25 | 30.2 (10) | 23.6 (6) |
| e00004 | Oil-free bound hydration reservoir (proxy) | 0.7 | 0.35 | 0.20 | 33.2 (8) | 23.1 (7) |
| e00010 | Plant-cuticle graded interphase | 0.3 | 0.45 | 0.15 | 36.7 (6) | 21.7 (8) |
| e00012 | Cartilage hydration-lubricated topcoat (proxy) | 0.5 | 0.60 | 0.20 | 31.0 (9) | 20.8 (9) |
| e00016 | Wax-platelet cuticle skin (proxy) | 0.5 | 0.45 | 0.20 | 29.8 (11) | 20.3 (10) |
| e00008 | Pulsed-heater thermoresponsive release | 1.0 | 0.00 | 0.00 | 38.3 (5) | 15.8 (11) |
| e00013 | Inclined-fibre anisotropic compliant skin | 0.3 | 0.40 | 0.15 | 23.9 (13) | 13.8 (12) |
| e00018 | Alginate-capsule self-healing gel | −0.5 | 0.50 | 0.10 | 25.8 (12) | 12.7 (13) |
| e00009 | Smectite interlayer-water skin | −0.5 | 0.05 | 0.05 | 18.0 (14) | 5.9 (14) |
| e00003 | Coral-mucus sloughing skin (estimated) | −4.0 | 0.40 | 0.10 | 13.9 (15) | 4.9 (15) |

Every hull concept has its weakest rating (0.2–0.3) on the requirement that restates the objective (clean-hull friction), so all scores drop by a similar share; the ranking changes where a must-requirement is clearly failed: the geology-inspired pack-ice skin (bionic 0.10) falls from 1 to 5 (the glacier skin in the same cell now loses the tie to it), the pulsed heater (bionic 0.0) from 5 to 11. Spearman with the critic objective gain: 0.91 (v4) → 0.80 (v5), for the same reason as in the facade baseline.


Defaults and why (archive version 4, chosen on the v3 role-play of the hull-coating query). The critic's objective gains there were 0.3–1.5 % of time-averaged friction, its plausible simulated benefits 25–80 %. With the version-3 constants (objective scale 10 %, weights 0.35 / 0.15 / 0.5, the higher simulated number used unless it differed from the critic by more than 50 %) a 1 % objective gain was worth 3.3 points, while the simulated term saturated at 14–15 points above ~50 %; the ranking was therefore driven by the requirement ratings (0.42–0.50, a few points of rating noise) and by proxy gains. Now the objective scale is 2 % (0.5 / 1.0 / 1.5 % score 0.22 / 0.39 / 0.53, i.e. 10 / 18 / 24 points at weight 0.45), the simulated term keeps the simulation evidence but at most 10 points and never above the critic, and requirement coverage keeps the hard constraints (0.45: no biocides, years of service). `score_breakdown` holds `formula` (generated from the constants in `vectornaut/explorer/profiles/materials.py`), `weights`, `scales_pct`, `components` (gate, objective_score, simulated_score, requirement_coverage), `objective_discount`, `relabel_factor`, `contributions` (points from objective, simulated benefit and requirements; they add up to the score unless the estimated cap applies) and `tiebreak`.

**Re-scoring of the v3 archive** (offline, through the version-3 → 4 migration; "P3" additionally marks as proxies the six entries whose critic text says the gain follows from an input or baseline choice). Rank = elite rank; "–" = not an elite.

| entry | concept | critic obj. % | sim. % | critic sim. % | req. | v3 score (rank) | v4 (rank) | v4 + proxies (rank) |
|---|---|---|---|---|---|---|---|---|
| e00015 | Pack-ice tiled skin on a soft bed | 1.5 | 78.3 | 60 | 0.42 | 40.4 (–, not better) | 52.0 (1, tiebreak) | 52.0 (1) |
| e00001 | Pilot-whale graded soft skin | 1.0 | 49.9 | 30 | 0.50 | 42.1 (1) | 48.0 (2) | 48.0 (2) |
| e00007 | Palm-stem anisotropic release layer | 1.0 | 51.3 | 35 | 0.48 | 41.3 (2) | 47.7 (3) | 47.7 (3) |
| e00005 | Mucin bottle-brush hydration skin | 1.0 | 60.1 | 25 | 0.46 | 36.9 (10) | 45.5 (4) | 45.5 (4) |
| e00008 | Pulsed-heater thermoresponsive release | 1.0 | 49.5 | 25 | 0.30 | 32.1 (11) | 38.3 (9) | 38.3 (5) |
| e00010 | Plant-cuticle graded interphase | 0.3 | 55.3 | 35 | 0.49 | 39.7 (5) | 36.7 (11) | 36.7 (6) |
| e00014 | Oil-free supersoft palm-stem layer (leaching) | 0.8 | 77.8 | 60 | 0.45 | 39.9 (4) | 44.6 (5) | 35.1 (7) |
| e00004 | Oil-free bound hydration reservoir (leaching) | 0.7 | 99.2 | 80 | 0.44 | 39.3 (6) | 43.0 (6) | 33.2 (8) |
| e00012 | Cartilage hydration-lubricated topcoat (input ratio) | 0.5 | 95.0 | 40 | 0.47 | 38.0 (8) | 39.6 (7) | 31.0 (9) |
| e00017 | Cyprid-scale wrinkled soft skin (input factor) | 0.5 | 49.8 | 25 | 0.45 | 38.0 (9) | 37.3 (10) | 30.2 (10) |
| e00016 | Wax-platelet cuticle skin (leaching) | 0.5 | 84.1 | 60 | 0.44 | 38.6 (7) | 39.3 (8) | 29.8 (11) |
| e00018 | Alginate-capsule self-healing gel | −0.5 | 70.0 | 30 | 0.40 | 31.7 (12) | 25.8 (12) | 25.8 (12) |
| e00013 | Inclined-fibre anisotropic compliant skin | 0.3 | 1.1 | 0 | 0.39 | 20.6 (13) | 23.9 (13) | 23.9 (13) |
| e00009 | Smectite interlayer-water skin (by construction) | −0.5 | 100.0 | 60 | 0.40 | 20.0 (14) | 18.0 (14) | 18.0 (14) |
| e00003 | Coral-mucus sloughing skin (estimated) | −4.0 | – | – | 0.31 | 15.4 (15) | 13.9 (15) | 13.9 (15) |
| e00006 | Glacier soft-bed release | 1.5 | 79.3 | 55 | 0.42 | 40.8 (3) | 52.2 (–, tie lost) | 52.2 (–) |

Spearman correlation of elite score with the critic's objective gain: 0.70 (v3) → 0.92 (v4) / 0.91 (v4 + proxies); mean critic objective gain of the top 5: 0.92 % → 1.06 % / 1.10 %. The top 5 are now the compliance-release concepts (tiled skin on a soft bed, pilot-whale soft skin, palm fibres) and the mucin brush; the leaching proxies fall to ranks 7–11 once flagged, matching the reviewer's judgement of the run.

**Evidence tiers**: `evidence_rank` 2 = simulated, 1 = estimated, 0 = estimated after an implausible simulation. Tier "simulated" is used only when the simulated number actually enters the score (usable and relevant). Within a cell an entry never replaces an elite of a higher rank, and an entry of a higher rank replaces a lower-ranked elite regardless of score; reports list elites by rank first. Trends use only simulated scores.

**Ties** (`TIE_EPSILON` = 1 point, `vectornaut/explorer/archive.py`): at equal rank, two scores within 1 point are a tie decided by `score_breakdown.tiebreak` = [critic objective gain, −number of killer risks, simulated benefit used] (larger is better, compared in this order). A new entry that wins a tie is `improved` with `tiebreak: improved_on_tiebreak`; one that loses although its score is slightly higher is `not_better` with `kept_on_tiebreak`. Equal keys, or entries without a key (business, no critic), fall back to "strictly higher score". (The v3 case: the pack-ice refinement of the glacier soft bed was physically better but 0.4 points lower, "not_better".)

**Flags** (`score_breakdown.flags`): `implausible_gain`, `gain_unavailable`, `simulated_quantity_irrelevant`, `model_assumption_sensitive`, `proxy_by_construction`, `objective_gain_unchecked`, `objective_gain_missing`, `relabelled_analogue`, `baseline_not_conventional` (the critic says the candidate's or the simulation's baseline is not the conventional one, e.g. a parent concept; × 0.8), `mostly_conventional_effect` (a conventional measure with the same physical effect gives at least half of the plausible objective gain), `must_requirement_unmet` (a must-requirement rated below 0.3), `requirements_unrated`, `requirements_partially_rated`, `critic_missing`, `critic_failed`, `baseline_unstated`.

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

**Archive versions.** The archive file carries `version` (currently 5). An older archive whose axes (names and values) equal the profile's is migrated on load (`version` raised, `migrated_from` recorded, the profile's summary appended to `migrations`); business archives of version 2 and 3 load this way unchanged. Materials archives of version 3 are **re-scored** on load: every evaluated entry gets the version-4 score computed from its stored breakdown (tier, gate, relabel factor, usable simulated gain, both critic numbers, objective gain and discount, requirement coverage; `rescored_from` keeps the old score and formula; a proxy flag did not exist and counts as false), and the elites are rebuilt with the tie rule (`migrations[].elite_changes`). Round logs keep the scores as recorded. Materials archives of version 3 and 4 are **re-scored to version 5** on load the same way (`migrate_archive` → `rescore_archive`): the objective scale is derived from the archive's critic objective gains (no target was stored), stored requirements without a priority count as `must`, the conventional-equivalent gain counts as absent, and the baseline factor comes from the stored `baseline_not_conventional` flag or critic field. The hull-coating v3 archive keeps S = 2 % (median 0.8 %, table above). An archive built with another vocabulary is refused with a message naming the difference, because its cells and scores are not comparable: materials archives of version 2 lack the `governing_quantity` values `fouling_adhesion` and `degradation_rate` and were scored with the old one-gain formula. Start a new map with `--archive NAME`, or move the old folder away. A newer archive than the code is refused too.

With `llm_replay --explorer`, `--rounds` counts explorer rounds. Each invocation rebuilds the archive from scratch in `<session>/data/explorer` and replays the recorded answers, so the search orders and prompts are the same on every rerun. A materials round needs the generator call, the pipeline calls (formulator, auditor, optimizer if `--opt-rounds` > 1, synthesizer) per candidate and one `MaterialsCriticBatch` call at the end, so keep `--batch` small when answering by hand. Sessions recorded before the critic existed need `--no-critic` to replay. Sessions recorded before archive version 3 (e.g. the v2 hull-coating role-plays) do not replay: the generator and critic prompts and schemas changed (objective/baseline statements, two critic numbers). The v3 role-play answers parse with version 4 (the new critic fields have defaults), but the search orders differ (preferred origins, extrapolate slots, combine anchors), so they no longer match the orders of a rerun. The facade role-play answers (version 4) parse with version 5 (all new fields are optional), but the prompts and, once mechanism classes are named, the orders differ. Replays memoise analytical (SymPy) solves in `<session>/solver_memo` (`--no-solver-memo` turns it off; see `VECTORNAUT_SOLVER_MEMO_DIR` and `VECTORNAUT_SYMBOLIC_TIMEOUT_S` in `docs/OPERATIONS.md`): the facade run's cactus model took ~20 minutes in `sp.solve` before the symbolic solve got its time budget (default 20 s, then SciPy).

## Files

| File | Content |
|---|---|
| `explorer/descriptors.py` | Axes, vocabulary validation, cell keys, distance, neighbours |
| `explorer/archive.py` | JSON archive (entries, elites ranked by evidence then score with the tie rule, proposal density, targets, infeasible cells and patterns with their scope, request analysis incl. must/nice requirements, objective, baseline, target gain, objective scale, hard and soft relevant values and preferred values, source counts, round log), versioning and migration hook, atomic writes |
| `explorer/strategies.py` | refine, fill_gap, extrapolate, combine, diversify, explore, seed; source rotation, compatibility pairs, scheduler with diagnostics |
| `explorer/schemas.py` | Response schemas of the generator and critic calls |
| `explorer/generator.py` | Prompt, JSON-safe context shortening, model call, candidate checks, request analysis |
| `explorer/profiles/materials.py`, `business.py` | Axes, prompt text, sanity checks, evaluators and critics, mock generators and critics |
| `explorer/report.py` | Markdown map/round reports (objective and baseline; elites with tier, score split objective / simulated / requirements, requirement coverage and flags; requirement table; proposal density vs score; values never proposed; why extrapolation fired or not), JSON export |
| `explorer/run.py` | Loop and CLI (`python -m vectornaut.explorer`) |

Tests: `tests/test_explorer_facade.py` (version 5: objective scale sources, bounds and re-scoring, the facade saturation case, conventional-equivalent gain, must/nice requirements and the soft gate, the cork-vs-Cyphochilus case, baseline factor, re-scoring of version-4 breakdowns, generator fields, infeasibility scope and pair checks in fill_gap/diversify, mechanism relevance in fill_gap/diversify/explore and the report, scheduling-time wording, mock loops with an adaptive and a stated target scale), `tests/test_symbolic_timeout.py` (symbolic time budget with the recorded cactus model, SciPy fallback and notes in the dispatcher, PINN reference without a symbolic solve, the solver memo and its use in replays), `tests/test_explorer_strategies.py` (descriptors, archive incl. version migration and vocabulary refusal, source counts, evidence ranking, tie rule, preferred values in every strategy, combine anchors, extrapolate slots and grouped marginal trends, strategies incl. diversity bonus, relevance filter and trend rules, combine compatibility, explore adjacency, source rotation, origin-change instructions, extrapolation diagnostics, scheduler), `tests/test_explorer_business.py` (unit economics with hand-computed numbers, sanity flags, critic scoring, candidate checks), `tests/test_explorer_materials.py` (fouling vocabulary, gain sanity, tiers and ceiling, two-number critic scoring, quantity relevance, relabel penalty, non-conventional baseline flag, `differs_strongly` cases, the min(simulation, critic) rule, the v3 ranking cases, proxies by construction, bio keywords, version-3 re-scoring, requirement coverage, framing storage and warnings, the critic call and prompt, concept hand-over incl. objective/baseline, JSON-safe context shortening, request analysis), `tests/test_explorer_loop.py` (mock loops for both profiles, framing flowing into context, prompts and reports, determinism, old-archive refusal, replay pause/resume, the pipeline concept hook). They run offline in about 15 seconds.

The map report shows, besides the elites and the two-axis map: the stored objective and baseline statements (with framing warnings), the stored requirements and relevant values, a requirement-coverage table (critic ratings per elite), the proposal-density-vs-score view per axis, the values of every axis that were never proposed (with the share of the most common value), gap targets with their under-exploration term, trends with their status, and flag counts.

## Limitations

- **The generator's prior**: candidates come from a language model. Even with targeted orders, it may drift back to familiar ideas or label a familiar idea with the target cell. The `on_target` flag and the proposal-density map make this visible, but do not remove it.
- **Business scores are self-estimates** (see above). A high score means "consistent and plausible by the model's own estimates", nothing more.
- **Materials scores inherit the pipeline's limits**: simplified steady 1D/2D models whose gains can hinge on one free modelling choice (an "equivalent laminar gap" in a Couette model of turbulent drag, a chosen slip length). The critic's plausible simulated benefit caps such results, but it is itself a model estimate; `model_assumption_sensitive` marks where the two disagree.
- **The objective gain is a critic estimate**: for fouling-control concepts no simulation computes time-averaged drag; the objective part of the score is the critic's judgement of how a simulated release-stress benefit translates into it. The simulated part (10 %, never above the critic's number) keeps the simulation evidence, but a generous critic moves the ranking even more than before, because the objective now carries most of the weight.
- **The framing is fixed once**: the first objective and baseline statements stay for the whole map. `framing_warnings` catches a clean-surface baseline for a multi-year request by keywords only; a wrong but plausible-sounding framing needs a new archive.
- **Relabelled analogues are detected by the critic**, not by rule: it sees the order's parents, but a subtle copy may pass.
- **Compatibility is judged on one pair of axes** (materials: mechanism × length scale). Other incompatible mixtures (e.g. an origin that has no such mechanism) still reach the generator, which may answer `target_feasible=false`.
- **Requirement coverage is a model rating**: the critic rates each requirement 0..1 from the description and the model, not from tests. Without a critic every requirement counts 0.5, so uncritiqued scores are comparable with each other but not with critiqued ones.
- **The axes are hand-chosen** and fixed per profile. They decide what counts as "different"; two ideas in one cell compete even if they differ in ways no axis captures. Changing the axes needs a new archive.
- **Descriptors are assigned by the model** from a closed vocabulary. Validation catches values outside the vocabulary, not wrong but allowed values.
- **Preferences come from keywords**: "bionic / bio-inspired / biomimetic" in the request or a requirement restricts origins to plant, animal and microbe. A request that wants biology without these words gets no preference; the requirement rating still penalises non-biological concepts, but they are not kept out of the targeted strategies.
- **Proxies are flagged by the critic**, not by rule: a leaching flux against an oil-containing baseline or a gain equal to an input ratio loses its simulated points only if the critic recognises it.
- **Trends are fitted on few points** (three simulated elites are enough). They are hints about where to look next, not statistics.
- **One infeasibility claim closes a cell** or, with a scope, a whole mechanism × length-scale pattern. A wrong "impossible" from the model hides a region until the entry is removed from the archive by hand.
- **The objective scale follows the archive.** Without a stated target it is 2 × the median critic objective gain, so it moves as the map grows (all entries are re-scored, so a map stays internally comparable, but scores of two maps are not). A generous critic raises the scale for everyone; a stated target avoids that.
- **The conventional equivalent is a critic estimate.** "What would equal-R standard insulation give?" is judged by the model; the archives recorded before version 5 have no such number, so insulation-like concepts keep their full gain there.
- **Must or nice is the generator's call** (default must). With every requirement `must` (old archives), a requirement that restates the objective and is rated low for every concept (hull: clean-hull friction ~0.2–0.3) lowers all scores alike; the must factor then hardly changes the ranking, but the absolute scores drop.
- **Mechanism relevance comes from the function analysis.** A mechanism class it did not name is reached only through explore (at 0.1 × weight) or seeds until an elite with it enters the top 5.
- **Relevance comes from the function analysis.** A governing quantity the generator did not name (and no elite holds) is never explored; a too narrow list narrows the search.
- Mock mode checks the machinery (scheduling, archive, reports), not idea quality: its landscape is synthetic.
