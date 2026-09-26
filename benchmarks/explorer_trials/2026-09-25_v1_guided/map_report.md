# Explorer map: materials

Query: "Entwickle eine bionisch inspirierte Beschichtung für Schiffsrümpfe, die den Reibungswiderstand im Wasser senkt, ohne giftige Antifouling-Wirkstoffe auszukommen, und mehrere Jahre im Salzwasser hält."  
Archive: `/tmp/claude-0/-home-user-vectornaut/6d94a234-0063-56c5-b9d5-7a9e3882a427/scratchpad/explorer_guided/session/data/explorer/materials/archive.json`  
Rounds so far: 3

> Scores come from the simulation pipeline; entries marked 'estimated, not simulated' use the candidate's back-of-envelope estimate at half weight.

## Coverage

- Cells in the grid: 3920 (10 x 7 x 7 x 8)
- Cells with an elite: 7 (0.18 %)
- Cells with at least one proposal: 7
- Infeasible cells/patterns: 1
- Entries: 9 (evaluated: 8, infeasible: 1)
- Elite score range: 51.8 – 77.7

## Elites (best per cell)

| # | score | basis | title | cell | strategy | round | entry |
|---|---|---|---|---|---|---|---|
| 1 | 77.7 | simulated | Closed-cell deep lubricant grooves with floor-pore feed and low solid fraction | mechanism_class=trapped_gas_or_liquid, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | refine | 2 | e00006 |
| 2 | 77.2 | simulated | Lateral-line canal lattice: streamwise-preferential permeable skin | mechanism_class=architected_lattice, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | fill_gap | 2 | e00004 |
| 3 | 76.8 | simulated | Mucus-gland polymer release rows: Toms-effect drag reduction from wall pores | mechanism_class=other, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | fill_gap | 3 | e00008 |
| 4 | 55.8 | estimated, not simulated | Dolphin-sloughing pneumatic skin: periodically strained soft fouling-release membrane | mechanism_class=other, length_scale=mm, inspiration_origin=animal, governing_quantity=deflection | seed | 1 | e00003 |
| 5 | 55.8 | estimated, not simulated | Whale-skin cm-bladder panels that peel off juvenile barnacles | mechanism_class=other, length_scale=cm_plus, inspiration_origin=animal, governing_quantity=deflection | extrapolate | 3 | e00007 |
| 6 | 53.5 | estimated, not simulated | Sweat-gland micro-lids: pressure-pulsed elastomer caps that strain the surface and pump lubricant | mechanism_class=other, length_scale=10_um, inspiration_origin=animal, governing_quantity=deflection | combine | 2 | e00005 |
| 7 | 51.8 | estimated, not simulated | Bladder-refilled closed-cell lubricant skin: pneumatic strokes that shed slime and refill grooves | mechanism_class=trapped_gas_or_liquid, length_scale=mm, inspiration_origin=animal, governing_quantity=deflection | combine | 3 | e00009 |

## Map: mechanism_class x length_scale

Rows: `mechanism_class`, columns: `length_scale`. Each cell: best elite score over all other axes (number of proposals that landed there). `·` = no elite yet, `x` = contains cells reported infeasible.

| mechanism_class \ length_scale | nm | sub_um | um | 10_um | 100_um | mm | cm_plus |
|---|---|---|---|---|---|---|---|
| interfacial_slip | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| flow_redirection | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| trapped_gas_or_liquid | · (0) | · (0) x | · (0) | 77.7 (2) | · (0) | 51.8 (1) | · (0) |
| porous_transport | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| graded_stiffness | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| architected_lattice | · (0) | · (0) | · (0) | 77.2 (1) | · (0) | · (0) | · (0) |
| radiative_control | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| phase_change | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| electrostatic_field | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| other | · (0) | · (0) | · (0) | 76.8 (2) | · (0) | 55.8 (1) | 55.8 (1) |

## Where ideas are proposed vs where they work

**mechanism_class** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| interfacial_slip | 0 | 0 % | 0 | – | – |
| flow_redirection | 0 | 0 % | 0 | – | – |
| trapped_gas_or_liquid | 3 | 38 % | 2 | 77.7 | 64.8 |
| porous_transport | 0 | 0 % | 0 | – | – |
| graded_stiffness | 0 | 0 % | 0 | – | – |
| architected_lattice | 1 | 12 % | 1 | 77.2 | 77.2 |
| radiative_control | 0 | 0 % | 0 | – | – |
| phase_change | 0 | 0 % | 0 | – | – |
| electrostatic_field | 0 | 0 % | 0 | – | – |
| other | 4 | 50 % | 4 | 76.8 | 60.5 |

**length_scale** (ordinal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| nm | 0 | 0 % | 0 | – | – |
| sub_um | 0 | 0 % | 0 | – | – |
| um | 0 | 0 % | 0 | – | – |
| 10_um | 5 | 62 % | 4 | 77.7 | 71.3 |
| 100_um | 0 | 0 % | 0 | – | – |
| mm | 2 | 25 % | 2 | 55.8 | 53.8 |
| cm_plus | 1 | 12 % | 1 | 55.8 | 55.8 |

**inspiration_origin** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| plant | 0 | 0 % | 0 | – | – |
| animal | 8 | 100 % | 7 | 77.7 | 64.1 |
| microbe | 0 | 0 % | 0 | – | – |
| geology | 0 | 0 % | 0 | – | – |
| atmosphere_ocean | 0 | 0 % | 0 | – | – |
| technology | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

**governing_quantity** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| wall_shear | 4 | 50 % | 3 | 77.7 | 77.2 |
| flow_rate | 0 | 0 % | 0 | – | – |
| heat_flux | 0 | 0 % | 0 | – | – |
| temperature | 0 | 0 % | 0 | – | – |
| deflection | 4 | 50 % | 4 | 55.8 | 54.2 |
| stress | 0 | 0 % | 0 | – | – |
| field_strength | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

## Promising under-explored cells (next fill-gap targets)

| empty cell | priority | best neighbour | elite neighbours | proposals | step from best |
|---|---|---|---|---|---|
| mechanism_class=electrostatic_field, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | 0.877 | 77.7 | 3 | 0 | mechanism_class: trapped_gas_or_liquid -> electrostatic_field |
| mechanism_class=flow_redirection, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | 0.877 | 77.7 | 3 | 0 | mechanism_class: trapped_gas_or_liquid -> flow_redirection |
| mechanism_class=graded_stiffness, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | 0.877 | 77.7 | 3 | 0 | mechanism_class: trapped_gas_or_liquid -> graded_stiffness |
| mechanism_class=interfacial_slip, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | 0.877 | 77.7 | 3 | 0 | mechanism_class: trapped_gas_or_liquid -> interfacial_slip |
| mechanism_class=phase_change, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | 0.877 | 77.7 | 3 | 0 | mechanism_class: trapped_gas_or_liquid -> phase_change |
| mechanism_class=porous_transport, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | 0.877 | 77.7 | 3 | 0 | mechanism_class: trapped_gas_or_liquid -> porous_transport |
| mechanism_class=radiative_control, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | 0.877 | 77.7 | 3 | 0 | mechanism_class: trapped_gas_or_liquid -> radiative_control |
| mechanism_class=trapped_gas_or_liquid, length_scale=10_um, inspiration_origin=animal, governing_quantity=deflection | 0.827 | 77.7 | 2 | 0 | governing_quantity: wall_shear -> deflection |

## Strategy yield (all rounds)

| strategy | orders | evaluated | new elite | improved | not better | failed | infeasible | invalid/rejected | missing | off target | elites now |
|---|---|---|---|---|---|---|---|---|---|---|---|
| refine | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| fill_gap | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| extrapolate | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| combine | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| explore | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| seed | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

## Ordinal trends (current)

| axis | mode | held fixed | points (value: score) | slope/step | r2 | next | status |
|---|---|---|---|---|---|---|---|
| length_scale | slice | mechanism_class=other, inspiration_origin=animal, governing_quantity=deflection | 10_um: 53.5, mm: 55.8, cm_plus: 55.8 | +0.82 | 0.89 | – | flat |
| length_scale | marginal | (marginal) | 10_um: 77.7, mm: 55.8, cm_plus: 55.8 | -7.82 | 0.89 | um | proposed |

## Trends used for extrapolation

| round | order | axis | mode | step | slope/step | result | score |
|---|---|---|---|---|---|---|---|
| 3 | r003-01 | length_scale | slice | mm -> cm_plus | +1.15 | new_elite | 55.8 |

## Infeasible cells

| cell / pattern | reason | reported by | round | times |
|---|---|---|---|---|
| mechanism_class=trapped_gas_or_liquid, length_scale=sub_um, inspiration_origin=atmosphere_ocean, governing_quantity=temperature | At sea the hull surface temperature is pinned to the water temperature by the turbulent heat-transfer coefficient (h ~ 1e4 W/m2K), and a sub-um gas layer adds only R = 5e-7 m / 0.026 W/mK ~ 2e-5 m2K/W (a fifth of 1/h), so no passive sub-um trapped-gas coating can shift wall temperature meaningfully and an active one would need ~1e4 W/m2 per kelvin (about 100 MW per kelvin for a 1e4 m2 hull), which exceeds the ship's total propulsion power; temperature therefore cannot be the governing benefit of a drag-reducing hull coating in this cell. | generator | 1 | 1 |
