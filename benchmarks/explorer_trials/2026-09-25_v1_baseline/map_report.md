# Explorer map: materials

Query: "Entwickle eine bionisch inspirierte Beschichtung für Schiffsrümpfe, die den Reibungswiderstand im Wasser senkt, ohne giftige Antifouling-Wirkstoffe auszukommen, und mehrere Jahre im Salzwasser hält."  
Archive: `/tmp/claude-0/-home-user-vectornaut/6d94a234-0063-56c5-b9d5-7a9e3882a427/scratchpad/explorer_baseline/session/data/explorer/materials/archive.json`  
Rounds so far: 1

> Scores come from the simulation pipeline; entries marked 'estimated, not simulated' use the candidate's back-of-envelope estimate at half weight.

## Coverage

- Cells in the grid: 3920 (10 x 7 x 7 x 8)
- Cells with an elite: 7 (0.18 %)
- Cells with at least one proposal: 9
- Infeasible cells/patterns: 0
- Entries: 9 (evaluated: 7, failed: 2)
- Elite score range: 31.0 – 90.3

## Elites (best per cell)

| # | score | basis | title | cell | strategy | round | entry |
|---|---|---|---|---|---|---|---|
| 1 | 90.3 | simulated | Graded-modulus foul-release laminate (whale-skin analogue) | mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=animal, governing_quantity=stress | seed | 1 | e00004 |
| 2 | 63.5 | simulated | Air-fed Salvinia plastron with hydrophilic-tip pinning | mechanism_class=trapped_gas_or_liquid, length_scale=100_um, inspiration_origin=plant, governing_quantity=wall_shear | seed | 1 | e00001 |
| 3 | 55.1 | simulated | Nacre-architected glass-flake hard coat with in-water grooming | mechanism_class=other, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | seed | 1 | e00007 |
| 4 | 51.4 | simulated | Streamwise-permeable micro-channel skin (seagrass canopy analogue) | mechanism_class=porous_transport, length_scale=10_um, inspiration_origin=plant, governing_quantity=wall_shear | seed | 1 | e00002 |
| 5 | 49.9 | estimated, not simulated | Electric-fish-type high-frequency AC settlement deterrent | mechanism_class=electrostatic_field, length_scale=100_um, inspiration_origin=animal, governing_quantity=field_strength | seed | 1 | e00009 |
| 6 | 48.5 | simulated | Low-modulus silicone riblets with foul-release skin | mechanism_class=flow_redirection, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | seed | 1 | e00003 |
| 7 | 31.0 | simulated | Byssus-graded tie zone between epoxy and silicone | mechanism_class=graded_stiffness, length_scale=10_um, inspiration_origin=animal, governing_quantity=stress | seed | 1 | e00005 |

## Map: mechanism_class x length_scale

Rows: `mechanism_class`, columns: `length_scale`. Each cell: best elite score over all other axes (number of proposals that landed there). `·` = no elite yet, `x` = contains cells reported infeasible.

| mechanism_class \ length_scale | nm | sub_um | um | 10_um | 100_um | mm | cm_plus |
|---|---|---|---|---|---|---|---|
| interfacial_slip | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| flow_redirection | · (0) | · (0) | · (0) | 48.5 (1) | · (0) | · (0) | · (0) |
| trapped_gas_or_liquid | · (0) | · (0) | · (0) | · (0) | 63.5 (1) | · (0) | · (0) |
| porous_transport | · (0) | · (0) | · (0) | 51.4 (1) | · (0) | · (0) | · (0) |
| graded_stiffness | · (0) | · (0) | · (0) | 31.0 (1) | 90.3 (1) | · (1) | · (0) |
| architected_lattice | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| radiative_control | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| phase_change | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| electrostatic_field | · (0) | · (0) | · (0) | · (0) | 49.9 (1) | · (0) | · (0) |
| other | · (0) | · (0) | · (1) | 55.1 (1) | · (0) | · (0) | · (0) |

## Where ideas are proposed vs where they work

**mechanism_class** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| interfacial_slip | 0 | 0 % | 0 | – | – |
| flow_redirection | 1 | 11 % | 1 | 48.5 | 48.5 |
| trapped_gas_or_liquid | 1 | 11 % | 1 | 63.5 | 63.5 |
| porous_transport | 1 | 11 % | 1 | 51.4 | 51.4 |
| graded_stiffness | 3 | 33 % | 2 | 90.3 | 60.7 |
| architected_lattice | 0 | 0 % | 0 | – | – |
| radiative_control | 0 | 0 % | 0 | – | – |
| phase_change | 0 | 0 % | 0 | – | – |
| electrostatic_field | 1 | 11 % | 1 | 49.9 | 49.9 |
| other | 2 | 22 % | 1 | 55.1 | 55.1 |

**length_scale** (ordinal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| nm | 0 | 0 % | 0 | – | – |
| sub_um | 0 | 0 % | 0 | – | – |
| um | 1 | 11 % | 0 | – | – |
| 10_um | 4 | 44 % | 4 | 55.1 | 46.5 |
| 100_um | 3 | 33 % | 3 | 90.3 | 67.9 |
| mm | 1 | 11 % | 0 | – | – |
| cm_plus | 0 | 0 % | 0 | – | – |

**inspiration_origin** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| plant | 2 | 22 % | 2 | 63.5 | 57.4 |
| animal | 7 | 78 % | 5 | 90.3 | 55.0 |
| microbe | 0 | 0 % | 0 | – | – |
| geology | 0 | 0 % | 0 | – | – |
| atmosphere_ocean | 0 | 0 % | 0 | – | – |
| technology | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

**governing_quantity** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| wall_shear | 6 | 67 % | 4 | 63.5 | 54.6 |
| flow_rate | 0 | 0 % | 0 | – | – |
| heat_flux | 0 | 0 % | 0 | – | – |
| temperature | 0 | 0 % | 0 | – | – |
| deflection | 0 | 0 % | 0 | – | – |
| stress | 2 | 22 % | 2 | 90.3 | 60.7 |
| field_strength | 1 | 11 % | 1 | 49.9 | 49.9 |
| other | 0 | 0 % | 0 | – | – |

## Promising under-explored cells (next fill-gap targets)

| empty cell | priority | best neighbour | elite neighbours | proposals | step from best |
|---|---|---|---|---|---|
| mechanism_class=electrostatic_field, length_scale=100_um, inspiration_origin=animal, governing_quantity=stress | 0.953 | 90.3 | 2 | 0 | mechanism_class: graded_stiffness -> electrostatic_field |
| mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=animal, governing_quantity=field_strength | 0.953 | 90.3 | 2 | 0 | governing_quantity: stress -> field_strength |
| mechanism_class=architected_lattice, length_scale=100_um, inspiration_origin=animal, governing_quantity=stress | 0.903 | 90.3 | 1 | 0 | mechanism_class: graded_stiffness -> architected_lattice |
| mechanism_class=flow_redirection, length_scale=100_um, inspiration_origin=animal, governing_quantity=stress | 0.903 | 90.3 | 1 | 0 | mechanism_class: graded_stiffness -> flow_redirection |
| mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=animal, governing_quantity=deflection | 0.903 | 90.3 | 1 | 0 | governing_quantity: stress -> deflection |
| mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=animal, governing_quantity=flow_rate | 0.903 | 90.3 | 1 | 0 | governing_quantity: stress -> flow_rate |
| mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=animal, governing_quantity=heat_flux | 0.903 | 90.3 | 1 | 0 | governing_quantity: stress -> heat_flux |
| mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=animal, governing_quantity=other | 0.903 | 90.3 | 1 | 0 | governing_quantity: stress -> other |

## Strategy yield (all rounds)

| strategy | orders | evaluated | new elite | improved | not better | failed | infeasible | invalid/rejected | missing | off target | elites now |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed | 9 | 7 | 7 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 7 |

## Ordinal trends (current)

| axis | mode | held fixed | points (value: score) | slope/step | r2 | next | status |
|---|---|---|---|---|---|---|---|
| length_scale | slice | mechanism_class=graded_stiffness, inspiration_origin=animal, governing_quantity=stress | 10_um: 31.0, 100_um: 90.3 | +59.35 | 1.00 | mm | proposed |
| length_scale | marginal | (marginal) | 10_um: 55.1, 100_um: 90.3 | +35.27 | 1.00 | mm | proposed |

## Trends used for extrapolation

_No extrapolation orders yet._

## Infeasible cells

_None._
