# Explorer map: materials

Query: "Entwickle eine bionisch inspirierte, passive Beschichtung oder Fassadenstruktur für Gebäude in heißen Klimazonen, die an Sommertagen die Wärmelast im Innenraum deutlich senkt, ohne Strom oder Wasserzufuhr auskommt und mindestens 20 Jahre wetterbeständig bleibt."  
Archive: `/tmp/claude-0/-home-user-vectornaut/6d94a234-0063-56c5-b9d5-7a9e3882a427/scratchpad/facade_guided/session/data/explorer/materials/archive.json`  
Rounds so far: 5

> score = 100 * gate * (0.45 * objective_score + 0.1 * simulated_score + 0.45 * requirement_coverage) * relabel_factor; gate = validator_score * (1 pass / 0.8 warn / 0 fail); objective_score = (1 - exp(-objective_gain_pct / 2)) * d, d = 1 if a critic judged the objective next to a relevant simulation, else 0.5; simulated_score = 1 - exp(-simulated_benefit_pct / 20) if the simulated quantity is relevant, else 0, simulated_benefit_pct = min(simulated, critic) when the critic gave a number, 0 for a proxy by construction; relabel_factor = 0.5 for a relabelled analogue, else 1; estimated tier capped at 50. Tier 'simulated' only when a usable simulated gain on a relevant governing quantity enters the score (never above the critic's plausible simulated benefit; not at all for a proxy by construction); otherwise tier 'estimated' (the simulated number is shown but not scored). The objective gain is the critic's plausible contribution to the stated objective against the stated baseline. 'points obj / sim / req' splits the score into its three parts (before the cap). Elites are ranked by tier first; within a cell, scores within 1 point are a tie decided by the critic's objective gain, then fewer killer risks.

## Coverage

- Cells in the grid: 4900 (10 x 7 x 7 x 10)
- Cells with an elite: 10 (0.20 %)
- Cells with at least one proposal: 10
- Infeasible cells/patterns: 2
- Entries: 15 (evaluated: 13, infeasible: 2)
- Elite score range: 33.5 – 81.7
- Elites by evidence tier: simulated: 10
- Flags on evaluated entries: model_assumption_sensitive: 1, proxy_by_construction: 4

## Requirements of the request

Objective and conventional baseline from the function analysis (round 1); fixed for this map.

- **Objective**: Summer-day (24 h) heat flux into the conditioned interior through the sun-exposed roof and facades of a building in a hot climate, averaged over a 20-year service life, including UV ageing, weathering and soiling of the outer surface (not the value of a freshly applied clean surface).
- **Baseline**: Conventional white acrylic/elastomeric cool-roof or facade coating (initial solar reflectance ~0.85, thermal emittance ~0.90) in its in-service aged and soiled condition (3-year aged solar reflectance ~0.65-0.70, as in CRRC ageing data), repainted every ~10 years, on the same insulated wall or roof.

Extracted by the generator's function analysis (round 1); fixed for this map.

- **indoor_heat_gain_reduction**: Clearly lowers the summer-day heat flux into the interior (roughly >20 %) versus the aged conventional white cool coating on the same envelope.
- **fully_passive**: Works without electricity, pumps, fans or water supply; only sun, sky, wind and rain act on it.
- **weather_durability_20yr**: Keeps its optical and thermal function (and adhesion) for at least 20 years under hot-climate UV, thermal cycling, rain, dust/sand and soiling without major maintenance.
- **bionic_mechanism**: The working mechanism is actually derived from a biological model (plant, animal, microbe), not only labelled so.
- **building_practicality**: Applicable as coating or facade structure on real buildings at acceptable cost, weight and glare, without toxic release.

Relevant `governing_quantity` values (explore/fill-gap/diversify use only these): heat_flux, temperature, degradation_rate (function analysis)

Preferred `inspiration_origin` values (fill-gap/diversify/combine/extrapolate use only these; explore reaches the others at a low weight): plant, animal, microbe (the request asks for a biological model ('bionisch'))

## Elites (best per cell)

| # | score | tier | points obj / sim / req | objective gain % | sim. benefit used % | simulated % | simulated quantity | req. coverage | flags | title | cell | strategy | round | entry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 81.7 | simulated | 45.0 / 7.4 / 29.2 | 18.0 | 27.0 | 32.8 | heat_flux | 0.65 | – | Birch-periderm glazed multi-row cell coat | mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=plant, governing_quantity=heat_flux | refine | 4 | e00012 |
| 2 | 81.1 | simulated | 45.0 / 7.8 / 28.4 | 20.0 | 30.0 | 40.2 | heat_flux | 0.63 | – | Beetle-scale inorganic scattering coat | mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=animal, governing_quantity=heat_flux | seed | 1 | e00001 |
| 3 | 79.2 | simulated | 44.9 / 7.8 / 26.6 | 12.0 | 30.0 | 36.1 | heat_flux | 0.59 | – | Cork-oak bark cladding with white mineral skin | mechanism_class=trapped_gas_or_liquid, length_scale=cm_plus, inspiration_origin=plant, governing_quantity=heat_flux | combine | 2 | e00005 |
| 4 | 74.9 | simulated | 44.2 / 5.9 / 24.8 | 8.0 | 18.0 | 26.2 | heat_flux | 0.55 | – | Desert-snail double-shell cladding | mechanism_class=trapped_gas_or_liquid, length_scale=cm_plus, inspiration_origin=animal, governing_quantity=heat_flux | seed | 1 | e00002 |
| 5 | 73.3 | simulated | 44.7 / 7.0 / 21.6 | 10.0 | 24.0 | 28.7 | heat_flux | 0.48 | – | Polar-bear aerogel-fibre roof underlayer | mechanism_class=architected_lattice, length_scale=sub_um, inspiration_origin=animal, governing_quantity=heat_flux | fill_gap | 2 | e00004 |
| 6 | 70.0 | simulated | 41.3 / 3.9 / 24.8 | 5.0 | 10.0 | 13.6 | temperature | 0.55 | – | Tussah-silk nanovoid glass-fibre shade screen | mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=animal, governing_quantity=temperature | diversify | 3 | e00008 |
| 7 | 68.3 | simulated | 44.9 / 0.0 / 23.4 | 12.0 | – | 37.5 | degradation_rate | 0.52 | proxy_by_construction | Birch-bark self-renewing white coat | mechanism_class=radiative_control, length_scale=10_um, inspiration_origin=plant, governing_quantity=degradation_rate | seed | 1 | e00003 |
| 8 | 64.3 | simulated | 41.3 / 4.5 / 18.4 | 5.0 | 12.0 | 18.4 | heat_flux | 0.41 | – | Nanowood white insulating facade panel | mechanism_class=architected_lattice, length_scale=sub_um, inspiration_origin=plant, governing_quantity=heat_flux | combine | 5 | e00014 |
| 9 | 59.2 | simulated | 35.0 / 2.2 / 22.1 | 3.0 | 5.0 | 6.6 | temperature | 0.49 | – | Cactus-spine pile facade shade | mechanism_class=trapped_gas_or_liquid, length_scale=cm_plus, inspiration_origin=plant, governing_quantity=temperature | fill_gap | 3 | e00007 |
| 10 | 33.5 | simulated | 17.7 / 0.0 / 15.8 | 1.0 | – | 1.7 | degradation_rate | 0.35 | proxy_by_construction | Antistatic oxide-network white topcoat | mechanism_class=electrostatic_field, length_scale=nm, inspiration_origin=technology, governing_quantity=degradation_rate | fill_gap | 4 | e00010 |

## Requirement coverage (critic ratings, elites)

| entry | title | indoor_heat_gain_reduction | fully_passive | weather_durability_20yr | bionic_mechanism | building_practicality | mean |
|---|---|---|---|---|---|---|---|
| e00012 | Birch-periderm glazed multi-row cell coat | 0.60 | 1.00 | 0.70 | 0.45 | 0.50 | 0.65 |
| e00001 | Beetle-scale inorganic scattering coat | 0.65 | 1.00 | 0.45 | 0.35 | 0.70 | 0.63 |
| e00005 | Cork-oak bark cladding with white mineral skin | 0.50 | 1.00 | 0.65 | 0.30 | 0.50 | 0.59 |
| e00002 | Desert-snail double-shell cladding | 0.40 | 1.00 | 0.70 | 0.30 | 0.35 | 0.55 |
| e00004 | Polar-bear aerogel-fibre roof underlayer | 0.45 | 1.00 | 0.40 | 0.25 | 0.30 | 0.48 |
| e00008 | Tussah-silk nanovoid glass-fibre shade screen | 0.30 | 1.00 | 0.55 | 0.50 | 0.40 | 0.55 |
| e00003 | Birch-bark self-renewing white coat | 0.35 | 1.00 | 0.35 | 0.50 | 0.40 | 0.52 |
| e00014 | Nanowood white insulating facade panel | 0.35 | 1.00 | 0.15 | 0.30 | 0.25 | 0.41 |
| e00007 | Cactus-spine pile facade shade | 0.15 | 1.00 | 0.50 | 0.55 | 0.25 | 0.49 |
| e00010 | Antistatic oxide-network white topcoat | 0.05 | 1.00 | 0.30 | 0.00 | 0.40 | 0.35 |

## Map: mechanism_class x length_scale

Rows: `mechanism_class`, columns: `length_scale`. Each cell: best elite score over all other axes (number of proposals that landed there). `·` = no elite yet, `x` = contains cells reported infeasible.

| mechanism_class \ length_scale | nm | sub_um | um | 10_um | 100_um | mm | cm_plus |
|---|---|---|---|---|---|---|---|
| interfacial_slip | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| flow_redirection | · (0) | · (0) x | · (0) | · (0) | · (0) | · (0) | · (0) |
| trapped_gas_or_liquid | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | 79.2 (3) |
| porous_transport | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| graded_stiffness | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| architected_lattice | · (0) | 73.3 (2) | · (0) | · (0) | · (0) | · (0) | · (0) |
| radiative_control | · (0) | 81.7 (4) | · (0) | 68.3 (3) | · (0) | · (0) | · (0) |
| phase_change | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| electrostatic_field | 33.5 (1) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| other | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |

## Where ideas are proposed vs where they work

**mechanism_class** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| interfacial_slip | 0 | 0 % | 0 | – | – |
| flow_redirection | 0 | 0 % | 0 | – | – |
| trapped_gas_or_liquid | 3 | 23 % | 3 | 79.2 | 71.1 |
| porous_transport | 0 | 0 % | 0 | – | – |
| graded_stiffness | 0 | 0 % | 0 | – | – |
| architected_lattice | 2 | 15 % | 2 | 73.3 | 68.8 |
| radiative_control | 7 | 54 % | 4 | 81.7 | 75.3 |
| phase_change | 0 | 0 % | 0 | – | – |
| electrostatic_field | 1 | 8 % | 1 | 33.5 | 33.5 |
| other | 0 | 0 % | 0 | – | – |

**length_scale** (ordinal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| nm | 1 | 8 % | 1 | 33.5 | 33.5 |
| sub_um | 6 | 46 % | 5 | 81.7 | 74.1 |
| um | 0 | 0 % | 0 | – | – |
| 10_um | 3 | 23 % | 1 | 68.3 | 68.3 |
| 100_um | 0 | 0 % | 0 | – | – |
| mm | 0 | 0 % | 0 | – | – |
| cm_plus | 3 | 23 % | 3 | 79.2 | 71.1 |

**inspiration_origin** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| plant | 8 | 62 % | 5 | 81.7 | 70.5 |
| animal | 4 | 31 % | 4 | 81.1 | 74.8 |
| microbe | 0 | 0 % | 0 | – | – |
| geology | 0 | 0 % | 0 | – | – |
| atmosphere_ocean | 0 | 0 % | 0 | – | – |
| technology | 1 | 8 % | 1 | 33.5 | 33.5 |
| other | 0 | 0 % | 0 | – | – |

**governing_quantity** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| wall_shear | 0 | 0 % | 0 | – | – |
| flow_rate | 0 | 0 % | 0 | – | – |
| heat_flux | 7 | 54 % | 6 | 81.7 | 75.7 |
| temperature | 2 | 15 % | 2 | 70.0 | 64.6 |
| deflection | 0 | 0 % | 0 | – | – |
| stress | 0 | 0 % | 0 | – | – |
| fouling_adhesion | 0 | 0 % | 0 | – | – |
| degradation_rate | 4 | 31 % | 2 | 68.3 | 50.9 |
| field_strength | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

## Values never proposed

- `mechanism_class` (top value holds 54 % of proposals): interfacial_slip, flow_redirection, porous_transport, graded_stiffness, phase_change, other
- `length_scale` (top value holds 46 % of proposals): um, 100_um, mm
- `inspiration_origin` (top value holds 62 % of proposals): microbe, geology, atmosphere_ocean, other
- `governing_quantity` (top value holds 54 % of proposals): every relevant value proposed (not relevant to the request, not searched: wall_shear, flow_rate, deflection, stress, fouling_adhesion, field_strength, other)

## Promising under-explored cells (next fill-gap targets)

| empty cell | priority | best neighbour | elite neighbours | proposals | under-explored | step from source (rotation-adjusted) |
|---|---|---|---|---|---|---|
| mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=microbe, governing_quantity=heat_flux | 0.970 | 81.7 | 2 | 0 | 0.62 | inspiration_origin: plant -> microbe |
| mechanism_class=graded_stiffness, length_scale=sub_um, inspiration_origin=plant, governing_quantity=heat_flux | 0.951 | 81.7 | 2 | 0 | 0.54 | mechanism_class: radiative_control -> graded_stiffness |
| mechanism_class=interfacial_slip, length_scale=sub_um, inspiration_origin=plant, governing_quantity=heat_flux | 0.951 | 81.7 | 2 | 0 | 0.54 | mechanism_class: radiative_control -> interfacial_slip |
| mechanism_class=other, length_scale=sub_um, inspiration_origin=plant, governing_quantity=heat_flux | 0.951 | 81.7 | 2 | 0 | 0.54 | mechanism_class: radiative_control -> other |
| mechanism_class=phase_change, length_scale=sub_um, inspiration_origin=plant, governing_quantity=heat_flux | 0.951 | 81.7 | 2 | 0 | 0.54 | mechanism_class: radiative_control -> phase_change |
| mechanism_class=porous_transport, length_scale=sub_um, inspiration_origin=plant, governing_quantity=heat_flux | 0.951 | 81.7 | 2 | 0 | 0.54 | mechanism_class: radiative_control -> porous_transport |
| mechanism_class=radiative_control, length_scale=um, inspiration_origin=plant, governing_quantity=heat_flux | 0.932 | 81.7 | 1 | 0 | 0.46 | length_scale: sub_um -> um |
| mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=plant, governing_quantity=temperature | 0.911 | 81.7 | 2 | 0 | 0.18 | governing_quantity: heat_flux -> temperature |

## Strategy yield (all rounds)

| strategy | orders | evaluated | new elite | improved | not better | failed | infeasible | invalid/rejected | missing | off target | elites now |
|---|---|---|---|---|---|---|---|---|---|---|---|
| refine | 3 | 3 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 1 |
| fill_gap | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| combine | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| diversify | 3 | 1 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 |
| seed | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |

## Ordinal trends (current)

| axis | mode | held fixed | points (value: score) | slope/step | r2 | next | status |
|---|---|---|---|---|---|---|---|
| length_scale | marginal | (marginal) | sub_um: 81.7, cm_plus: 79.2 | -0.49 | 1.00 | – | too_few_points |

## Trends used for extrapolation

_No extrapolation orders yet._

## Infeasible cells

| cell / pattern | reason | reported by | round | times |
|---|---|---|---|---|
| mechanism_class=flow_redirection, length_scale=sub_um, inspiration_origin=animal, governing_quantity=heat_flux | As for the plant variant of this cell: sub-micron features lie deep inside the viscous sublayer of the wind boundary layer (roughness Reynolds number ~1e-2), where Stokes flow cannot be redirected, so no sub-micron flow-redirection structure of any origin can change convective heat transfer; flow-steering features must be mm to m in size. | generator | 5 | 1 |
| mechanism_class=flow_redirection, length_scale=sub_um, inspiration_origin=plant, governing_quantity=heat_flux | Sub-micron surface features sit deep in the viscous sublayer of the wind boundary layer on a building (roughness Reynolds number u_tau*k/nu ~ 1e-2 for k = 0.5 um), where the flow is Stokes flow that cannot be redirected, so no sub-micron flow-redirection structure can change convective heat transfer or heat flux; flow-steering facade features must be mm to m in size. | generator | 4 | 1 |
