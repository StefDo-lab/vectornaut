# Explorer map: materials

Query: "Entwickle eine bionisch inspirierte, passive Beschichtung oder Fassadenstruktur für Gebäude in heißen Klimazonen, die an Sommertagen die Wärmelast im Innenraum deutlich senkt, ohne Strom oder Wasserzufuhr auskommt und mindestens 20 Jahre wetterbeständig bleibt."  
Archive: `/tmp/claude-0/-home-user-vectornaut/6d94a234-0063-56c5-b9d5-7a9e3882a427/scratchpad/facade_baseline/session/data/explorer/materials/archive.json`  
Rounds so far: 1

> score = 100 * gate * (0.45 * objective_score + 0.1 * simulated_score + 0.45 * requirement_coverage) * relabel_factor; gate = validator_score * (1 pass / 0.8 warn / 0 fail); objective_score = (1 - exp(-objective_gain_pct / 2)) * d, d = 1 if a critic judged the objective next to a relevant simulation, else 0.5; simulated_score = 1 - exp(-simulated_benefit_pct / 20) if the simulated quantity is relevant, else 0, simulated_benefit_pct = min(simulated, critic) when the critic gave a number, 0 for a proxy by construction; relabel_factor = 0.5 for a relabelled analogue, else 1; estimated tier capped at 50. Tier 'simulated' only when a usable simulated gain on a relevant governing quantity enters the score (never above the critic's plausible simulated benefit; not at all for a proxy by construction); otherwise tier 'estimated' (the simulated number is shown but not scored). The objective gain is the critic's plausible contribution to the stated objective against the stated baseline. 'points obj / sim / req' splits the score into its three parts (before the cap). Elites are ranked by tier first; within a cell, scores within 1 point are a tie decided by the critic's objective gain, then fewer killer risks.

## Coverage

- Cells in the grid: 4900 (10 x 7 x 7 x 10)
- Cells with an elite: 13 (0.27 %)
- Cells with at least one proposal: 13
- Infeasible cells/patterns: 0
- Entries: 15 (evaluated: 15)
- Elite score range: 32.6 – 83.3
- Elites by evidence tier: estimated: 1, simulated: 12
- Flags on evaluated entries: baseline_not_conventional: 4, model_assumption_sensitive: 2, proxy_by_construction: 1, simulated_quantity_irrelevant: 1

## Requirements of the request

Objective and conventional baseline from the function analysis (round 1); fixed for this map.

- **Objective**: Time-averaged summer daytime heat gain into the conditioned interior through the opaque envelope (roof and sun-exposed walls), per m2 of envelope, averaged over a 20-year service life in a hot climate, including the loss of solar reflectance and emittance by soiling, UV ageing and weathering.
- **Baseline**: Conventional white acrylic/elastomeric cool facade/roof paint (solar reflectance ~0.80 new, ~0.65-0.70 after 2-3 years of soiling and ageing, thermal emittance ~0.90) on the same wall/roof construction with the region's typical insulation, maintained per normal practice.

Extracted by the generator's function analysis (round 1); fixed for this map.

- **indoor_heat_gain_reduction**: Clearly reduces time-averaged summer daytime heat gain through the envelope (target >=20% of opaque-envelope gain) versus an aged conventional white cool paint.
- **passive_no_power_no_water**: Works without electricity, pumps, fans or supplied water; only sun, sky, wind, ambient air and rain.
- **weather_durability_20y**: Retains >=80% of its cooling benefit for 20 years of UV, thermal cycling, dust, rain and wind with at most light cleaning.
- **bionic_mechanism**: The working mechanism is genuinely derived from a named biological model.
- **building_practicality**: Applicable to real walls/roofs at acceptable cost, weight, fire safety and without glare or condensation problems.

Relevant `governing_quantity` values (explore/fill-gap/diversify use only these): heat_flux, temperature, degradation_rate (function analysis) + stress (held by elites)

Preferred `inspiration_origin` values (fill-gap/diversify/combine/extrapolate use only these; explore reaches the others at a low weight): plant, animal, microbe (the request asks for a biological model ('bionisch'))

## Elites (best per cell)

| # | score | tier | points obj / sim / req | objective gain % | sim. benefit used % | simulated % | simulated quantity | req. coverage | flags | title | cell | strategy | round | entry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 83.3 | simulated | 45.0 / 8.6 / 29.7 | 35.0 | 40.0 | 44.5 | heat_flux | 0.66 | baseline_not_conventional | Cork-oak bark closed-cell cladding with white mineral skin | mechanism_class=architected_lattice, length_scale=10_um, inspiration_origin=plant, governing_quantity=heat_flux | seed | 1 | e00012 |
| 2 | 82.7 | simulated | 45.0 / 7.1 / 30.6 | 17.0 | 25.0 | 35.6 | heat_flux | 0.68 | – | Cyphochilus-scale inorganic disordered-network white topcoat | mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=animal, governing_quantity=heat_flux | seed | 1 | e00001 |
| 3 | 81.0 | simulated | 44.9 / 5.9 / 30.1 | 12.0 | 18.0 | 28.8 | heat_flux | 0.67 | – | Diatom-frustule silicate cooling paint | mechanism_class=radiative_control, length_scale=um, inspiration_origin=microbe, governing_quantity=heat_flux | seed | 1 | e00003 |
| 4 | 77.3 | simulated | 44.8 / 5.5 / 27.0 | 11.0 | 16.0 | 22.5 | degradation_rate | 0.60 | – | Self-renewing chalking mineral topcoat (cuticle-wax analogue) | mechanism_class=radiative_control, length_scale=10_um, inspiration_origin=plant, governing_quantity=degradation_rate | seed | 1 | e00010 |
| 5 | 77.0 | simulated | 44.7 / 5.3 / 27.0 | 10.0 | 15.0 | 24.4 | heat_flux | 0.60 | baseline_not_conventional | Desert-snail white shell tile over sealed air cavity | mechanism_class=trapped_gas_or_liquid, length_scale=mm, inspiration_origin=animal, governing_quantity=heat_flux | seed | 1 | e00004 |
| 6 | 74.3 | simulated | 42.8 / 4.5 / 27.0 | 6.0 | 12.0 | 15.6 | heat_flux | 0.60 | – | Small-leaf convective shading screen | mechanism_class=other, length_scale=cm_plus, inspiration_origin=plant, governing_quantity=heat_flux | seed | 1 | e00006 |
| 7 | 72.2 | simulated | 44.2 / 5.0 / 22.9 | 8.0 | 14.0 | 22.7 | heat_flux | 0.51 | – | Silver-ant prism-fibre reflective-emissive mat | mechanism_class=radiative_control, length_scale=um, inspiration_origin=animal, governing_quantity=heat_flux | seed | 1 | e00002 |
| 8 | 71.5 | simulated | 41.3 / 3.6 / 26.6 | 5.0 | 9.0 | 13.9 | heat_flux | 0.59 | – | Termite-mound wind-and-stack ventilated rainscreen | mechanism_class=flow_redirection, length_scale=cm_plus, inspiration_origin=animal, governing_quantity=heat_flux | seed | 1 | e00007 |
| 9 | 67.7 | simulated | 41.3 / 3.9 / 22.5 | 5.0 | 10.0 | 18.6 | heat_flux | 0.50 | – | Saguaro-rib asymmetric sky-facing facade corrugation | mechanism_class=radiative_control, length_scale=cm_plus, inspiration_origin=plant, governing_quantity=heat_flux | seed | 1 | e00005 |
| 10 | 53.4 | simulated | 35.0 / 1.8 / 16.6 | 3.0 | 4.0 | 6.3 | heat_flux | 0.37 | – | Thorny-devil night-sorption evaporative skin | mechanism_class=phase_change, length_scale=100_um, inspiration_origin=animal, governing_quantity=heat_flux | seed | 1 | e00009 |
| 11 | 37.6 | simulated | 17.7 / 1.0 / 18.9 | 1.0 | 2.0 | 5.3 | heat_flux | 0.42 | model_assumption_sensitive | Camel-coat insulation with inner PCM heat store | mechanism_class=phase_change, length_scale=mm, inspiration_origin=animal, governing_quantity=heat_flux | seed | 1 | e00008 |
| 12 | 32.6 | simulated | 10.0 / 0.0 / 22.7 | 0.5 | – | 60.0 | degradation_rate | 0.50 | model_assumption_sensitive, proxy_by_construction, baseline_not_conventional | Bacterial self-healing white calcite render | mechanism_class=other, length_scale=10_um, inspiration_origin=microbe, governing_quantity=degradation_rate | seed | 1 | e00014 |
| 13 | 39.9 | estimated | 14.2 / 0.0 / 25.6 | 2.0 | – | 57.4 | stress | 0.57 | simulated_quantity_irrelevant, baseline_not_conventional | Nacre-like platelet reflective coating for crack tolerance | mechanism_class=architected_lattice, length_scale=um, inspiration_origin=animal, governing_quantity=stress | seed | 1 | e00013 |

## Requirement coverage (critic ratings, elites)

| entry | title | indoor_heat_gain_reduction | passive_no_power_no_water | weather_durability_20y | bionic_mechanism | building_practicality | mean |
|---|---|---|---|---|---|---|---|
| e00012 | Cork-oak bark closed-cell cladding with white mineral skin | 0.90 | 1.00 | 0.70 | 0.15 | 0.55 | 0.66 |
| e00001 | Cyphochilus-scale inorganic disordered-network white topcoat | 0.65 | 1.00 | 0.50 | 0.55 | 0.70 | 0.68 |
| e00003 | Diatom-frustule silicate cooling paint | 0.50 | 1.00 | 0.70 | 0.30 | 0.85 | 0.67 |
| e00010 | Self-renewing chalking mineral topcoat (cuticle-wax analogue) | 0.50 | 1.00 | 0.50 | 0.30 | 0.70 | 0.60 |
| e00004 | Desert-snail white shell tile over sealed air cavity | 0.45 | 1.00 | 0.60 | 0.45 | 0.50 | 0.60 |
| e00006 | Small-leaf convective shading screen | 0.30 | 1.00 | 0.80 | 0.40 | 0.50 | 0.60 |
| e00002 | Silver-ant prism-fibre reflective-emissive mat | 0.35 | 1.00 | 0.20 | 0.70 | 0.30 | 0.51 |
| e00007 | Termite-mound wind-and-stack ventilated rainscreen | 0.25 | 1.00 | 0.85 | 0.25 | 0.60 | 0.59 |
| e00005 | Saguaro-rib asymmetric sky-facing facade corrugation | 0.30 | 1.00 | 0.35 | 0.35 | 0.50 | 0.50 |
| e00009 | Thorny-devil night-sorption evaporative skin | 0.15 | 0.80 | 0.20 | 0.30 | 0.40 | 0.37 |
| e00008 | Camel-coat insulation with inner PCM heat store | 0.10 | 0.60 | 0.60 | 0.30 | 0.50 | 0.42 |
| e00014 | Bacterial self-healing white calcite render | 0.02 | 0.90 | 0.40 | 0.70 | 0.50 | 0.50 |
| e00013 | Nacre-like platelet reflective coating for crack tolerance | 0.05 | 1.00 | 0.50 | 0.60 | 0.70 | 0.57 |

## Map: mechanism_class x length_scale

Rows: `mechanism_class`, columns: `length_scale`. Each cell: best elite score over all other axes (number of proposals that landed there). `·` = no elite yet, `x` = contains cells reported infeasible.

| mechanism_class \ length_scale | nm | sub_um | um | 10_um | 100_um | mm | cm_plus |
|---|---|---|---|---|---|---|---|
| interfacial_slip | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| flow_redirection | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | 71.5 (1) |
| trapped_gas_or_liquid | · (0) | · (0) | · (0) | · (0) | · (0) | 77.0 (1) | · (0) |
| porous_transport | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| graded_stiffness | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| architected_lattice | · (0) | · (0) | 39.9 (1) | 83.3 (1) | · (0) | · (0) | · (0) |
| radiative_control | · (0) | 82.7 (2) | 81.0 (2) | 77.3 (1) | · (0) | · (0) | 67.7 (2) |
| phase_change | · (0) | · (0) | · (0) | · (0) | 53.4 (1) | 37.6 (1) | · (0) |
| electrostatic_field | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| other | · (0) | · (0) | · (0) | 32.6 (1) | · (0) | · (0) | 74.3 (1) |

## Where ideas are proposed vs where they work

**mechanism_class** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| interfacial_slip | 0 | 0 % | 0 | – | – |
| flow_redirection | 1 | 7 % | 1 | 71.5 | 71.5 |
| trapped_gas_or_liquid | 1 | 7 % | 1 | 77.0 | 77.0 |
| porous_transport | 0 | 0 % | 0 | – | – |
| graded_stiffness | 0 | 0 % | 0 | – | – |
| architected_lattice | 2 | 13 % | 2 | 83.3 | 61.6 |
| radiative_control | 7 | 47 % | 5 | 82.7 | 76.2 |
| phase_change | 2 | 13 % | 2 | 53.4 | 45.5 |
| electrostatic_field | 0 | 0 % | 0 | – | – |
| other | 2 | 13 % | 2 | 74.3 | 53.5 |

**length_scale** (ordinal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| nm | 0 | 0 % | 0 | – | – |
| sub_um | 2 | 13 % | 1 | 82.7 | 82.7 |
| um | 3 | 20 % | 3 | 81.0 | 64.3 |
| 10_um | 3 | 20 % | 3 | 83.3 | 64.4 |
| 100_um | 1 | 7 % | 1 | 53.4 | 53.4 |
| mm | 2 | 13 % | 2 | 77.0 | 57.3 |
| cm_plus | 4 | 27 % | 3 | 74.3 | 71.2 |

**inspiration_origin** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| plant | 5 | 33 % | 4 | 83.3 | 75.7 |
| animal | 8 | 53 % | 7 | 82.7 | 62.0 |
| microbe | 2 | 13 % | 2 | 81.0 | 56.8 |
| geology | 0 | 0 % | 0 | – | – |
| atmosphere_ocean | 0 | 0 % | 0 | – | – |
| technology | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

**governing_quantity** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| wall_shear | 0 | 0 % | 0 | – | – |
| flow_rate | 0 | 0 % | 0 | – | – |
| heat_flux | 12 | 80 % | 10 | 83.3 | 70.1 |
| temperature | 0 | 0 % | 0 | – | – |
| deflection | 0 | 0 % | 0 | – | – |
| stress | 1 | 7 % | 1 | 39.9 | 39.9 |
| fouling_adhesion | 0 | 0 % | 0 | – | – |
| degradation_rate | 2 | 13 % | 2 | 77.3 | 55.0 |
| field_strength | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

## Values never proposed

- `mechanism_class` (top value holds 47 % of proposals): interfacial_slip, porous_transport, graded_stiffness, electrostatic_field
- `length_scale` (top value holds 27 % of proposals): nm
- `inspiration_origin` (top value holds 53 % of proposals): geology, atmosphere_ocean, technology, other
- `governing_quantity` (top value holds 80 % of proposals): temperature (not relevant to the request, not searched: wall_shear, flow_rate, deflection, fouling_adhesion, field_strength, other)

## Promising under-explored cells (next fill-gap targets)

| empty cell | priority | best neighbour | elite neighbours | proposals | under-explored | step from source (rotation-adjusted) |
|---|---|---|---|---|---|---|
| mechanism_class=architected_lattice, length_scale=10_um, inspiration_origin=plant, governing_quantity=temperature | 1.033 | 83.3 | 1 | 0 | 0.80 | governing_quantity: heat_flux -> temperature |
| mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=animal, governing_quantity=temperature | 1.027 | 82.7 | 1 | 0 | 0.80 | governing_quantity: heat_flux -> temperature |
| mechanism_class=radiative_control, length_scale=um, inspiration_origin=microbe, governing_quantity=temperature | 1.010 | 81.0 | 1 | 0 | 0.80 | governing_quantity: heat_flux -> temperature |
| mechanism_class=radiative_control, length_scale=10_um, inspiration_origin=plant, governing_quantity=temperature | 0.973 | 77.3 | 1 | 0 | 0.80 | governing_quantity: degradation_rate -> temperature |
| mechanism_class=trapped_gas_or_liquid, length_scale=mm, inspiration_origin=animal, governing_quantity=temperature | 0.970 | 77.0 | 1 | 0 | 0.80 | governing_quantity: heat_flux -> temperature |
| mechanism_class=architected_lattice, length_scale=10_um, inspiration_origin=plant, governing_quantity=degradation_rate | 0.950 | 83.3 | 2 | 0 | 0.27 | governing_quantity: heat_flux -> degradation_rate |
| mechanism_class=electrostatic_field, length_scale=10_um, inspiration_origin=plant, governing_quantity=heat_flux | 0.950 | 83.3 | 1 | 0 | 0.47 | mechanism_class: architected_lattice -> electrostatic_field |
| mechanism_class=graded_stiffness, length_scale=10_um, inspiration_origin=plant, governing_quantity=heat_flux | 0.950 | 83.3 | 1 | 0 | 0.47 | mechanism_class: architected_lattice -> graded_stiffness |

## Strategy yield (all rounds)

| strategy | orders | evaluated | new elite | improved | not better | failed | infeasible | invalid/rejected | missing | off target | elites now |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed | 15 | 15 | 13 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 13 |

## Ordinal trends (current)

| axis | mode | held fixed | points (value: score) | slope/step | r2 | next | status |
|---|---|---|---|---|---|---|---|
| length_scale | slice | mechanism_class=phase_change, inspiration_origin=animal, governing_quantity=heat_flux | 100_um: 53.4, mm: 37.6 | -15.86 | 1.00 | – | too_few_points |
| length_scale | slice | mechanism_class=radiative_control, inspiration_origin=animal, governing_quantity=heat_flux | sub_um: 82.7, um: 72.2 | -10.57 | 1.00 | – | too_few_points |
| length_scale | marginal | (marginal) | sub_um: 82.7, um: 81.0, 10_um: 83.3, 100_um: 53.4, mm: 77.0, cm_plus: 74.3 | -2.41 | 0.16 | – | poor_fit |

## Trends used for extrapolation

_No extrapolation orders yet._

## Infeasible cells

_None._
