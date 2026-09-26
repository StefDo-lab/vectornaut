# Explorer map: materials

Query: "Entwickle eine bionisch inspirierte Beschichtung für Schiffsrümpfe, die den Reibungswiderstand im Wasser senkt, ohne giftige Antifouling-Wirkstoffe auszukommen, und mehrere Jahre im Salzwasser hält."  
Archive: `/tmp/claude-0/-home-user-vectornaut/6d94a234-0063-56c5-b9d5-7a9e3882a427/scratchpad/explorer_v2_guided/session/data/explorer/materials/archive.json`  
Rounds so far: 5

> Score = 100 * gate * (0.5 * gain_score + 0.5 * requirement coverage). Tier 'simulated' uses the simulated gain (the lower of simulation and critic if they differ by more than 2x); tier 'estimated' (no usable simulated gain, e.g. flagged implausible_gain) uses the lower of the critic's and the candidate's estimate at half weight and is capped at 50. Elites are ranked by tier first.

## Coverage

- Cells in the grid: 3920 (10 x 7 x 7 x 8)
- Cells with an elite: 9 (0.23 %)
- Cells with at least one proposal: 9
- Infeasible cells/patterns: 3
- Entries: 15 (evaluated: 12, infeasible: 3)
- Elite score range: 4.6 – 27.7
- Elites by evidence tier: simulated: 9
- Flags on evaluated entries: gain_unavailable: 1, model_assumption_sensitive: 9

## Requirements of the request

Extracted by the generator's function analysis (round 1); fixed for this map.

- **low_friction_drag**: Reduces the mean wall shear stress / frictional resistance at cruising speed versus a clean, standard foul-release hull coating.
- **non_toxic_antifouling**: Prevents fouling settlement or makes it release easily without releasing biocides, heavy metals or other toxic substances into the sea.
- **multi_year_seawater_durability**: Keeps its function and adhesion for at least 3-5 years in seawater (hydrolysis, abrasion, UV at the waterline, cleaning).
- **low_consumption_operation**: Needs no continuous supply of material or energy that is large compared with the fuel it saves, and no frequent reapplication.
- **hull_scale_applicability**: Can be applied to large steel hull areas with dry-dock methods at a cost comparable to premium hull coatings.

Relevant `governing_quantity` values (explore/fill-gap/diversify use only these): wall_shear, stress (function analysis)

## Elites (best per cell)

| # | score | tier | req. coverage | gain used % | simulated % | critic % | flags | title | cell | strategy | round | entry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 27.7 | simulated | 0.53 | 0.5 | 66.7 | 0.5 | model_assumption_sensitive | Dolphin-skin graded-modulus foul-release elastomer | mechanism_class=graded_stiffness, length_scale=mm, inspiration_origin=animal, governing_quantity=wall_shear | seed | 1 | e00001 |
| 2 | 27.5 | simulated | 0.55 | 0.0 | 50.0 | 0.0 | model_assumption_sensitive | Broadcast-scree silicone-capped armour coating | mechanism_class=graded_stiffness, length_scale=mm, inspiration_origin=geology, governing_quantity=wall_shear | refine | 5 | e00015 |
| 3 | 26.0 | simulated | 0.52 | 0.0 | 84.2 | 0.0 | model_assumption_sensitive | Byssus-graded tie-layer against coating delamination | mechanism_class=graded_stiffness, length_scale=mm, inspiration_origin=animal, governing_quantity=stress | fill_gap | 2 | e00006 |
| 4 | 24.5 | simulated | 0.49 | 0.0 | 68.1 | 0.0 | model_assumption_sensitive | Nilas-crust over slush-gel damped skin | mechanism_class=graded_stiffness, length_scale=mm, inspiration_origin=atmosphere_ocean, governing_quantity=wall_shear | refine | 4 | e00011 |
| 5 | 22.5 | simulated | 0.45 | 0.0 | 45.0 | 0.0 | model_assumption_sensitive | Cuttlebone-septa membrane lattice skin | mechanism_class=architected_lattice, length_scale=mm, inspiration_origin=animal, governing_quantity=wall_shear | fill_gap | 4 | e00010 |
| 6 | 18.8 | simulated | 0.49 | 0.0 | 76.1 | 0.0 | model_assumption_sensitive | Pack-ice crust load spreading against brush damage | mechanism_class=graded_stiffness, length_scale=mm, inspiration_origin=atmosphere_ocean, governing_quantity=stress | diversify | 5 | e00013 |
| 7 | 14.9 | simulated | 0.39 | -94.4 | -94.4 | 0.0 | – | Cartilage-style hydration-lubricated release interphase | mechanism_class=interfacial_slip, length_scale=nm, inspiration_origin=animal, governing_quantity=stress | combine | 5 | e00014 |
| 8 | 13.0 | simulated | 0.26 | 0.0 | 0.2 | 0.0 | model_assumption_sensitive | Hagfish-mucus zwitterionic brush hydration skin | mechanism_class=interfacial_slip, length_scale=nm, inspiration_origin=animal, governing_quantity=wall_shear | seed | 1 | e00002 |
| 9 | 4.6 | simulated | 0.12 | -28.7 | -28.7 | -5.0 | – | Swordfish-gland micro-capillary oil-weeping skin | mechanism_class=porous_transport, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | seed | 1 | e00003 |

## Requirement coverage (critic ratings, elites)

| entry | title | low_friction_drag | non_toxic_antifouling | multi_year_seawater_durability | low_consumption_operation | hull_scale_applicability | mean |
|---|---|---|---|---|---|---|---|
| e00001 | Dolphin-skin graded-modulus foul-release elastomer | 0.15 | 0.70 | 0.45 | 0.85 | 0.50 | 0.53 |
| e00015 | Broadcast-scree silicone-capped armour coating | 0.10 | 0.65 | 0.55 | 0.90 | 0.55 | 0.55 |
| e00006 | Byssus-graded tie-layer against coating delamination | 0.00 | 0.60 | 0.50 | 0.90 | 0.60 | 0.52 |
| e00011 | Nilas-crust over slush-gel damped skin | 0.05 | 0.75 | 0.40 | 0.85 | 0.40 | 0.49 |
| e00010 | Cuttlebone-septa membrane lattice skin | 0.05 | 0.60 | 0.45 | 0.90 | 0.25 | 0.45 |
| e00013 | Pack-ice crust load spreading against brush damage | 0.05 | 0.60 | 0.55 | 0.85 | 0.40 | 0.49 |
| e00014 | Cartilage-style hydration-lubricated release interphase | 0.00 | 0.40 | 0.30 | 0.70 | 0.55 | 0.39 |
| e00002 | Hagfish-mucus zwitterionic brush hydration skin | 0.05 | 0.40 | 0.10 | 0.50 | 0.25 | 0.26 |
| e00003 | Swordfish-gland micro-capillary oil-weeping skin | 0.00 | 0.20 | 0.25 | 0.05 | 0.10 | 0.12 |

## Map: mechanism_class x length_scale

Rows: `mechanism_class`, columns: `length_scale`. Each cell: best elite score over all other axes (number of proposals that landed there). `·` = no elite yet, `x` = contains cells reported infeasible.

| mechanism_class \ length_scale | nm | sub_um | um | 10_um | 100_um | mm | cm_plus |
|---|---|---|---|---|---|---|---|
| interfacial_slip | 14.9 (2) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| flow_redirection | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| trapped_gas_or_liquid | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| porous_transport | · (0) | · (0) | · (0) | 4.6 (1) | · (0) | · (0) | · (0) |
| graded_stiffness | · (0) x | · (0) | · (0) | · (0) | · (0) | 27.7 (8) | · (0) |
| architected_lattice | · (0) | · (0) | · (0) | · (0) | · (0) | 22.5 (1) | · (0) |
| radiative_control | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| phase_change | · (0) | · (0) | · (0) | · (0) x | · (0) | · (0) | · (0) |
| electrostatic_field | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) x | · (0) |
| other | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |

## Where ideas are proposed vs where they work

**mechanism_class** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| interfacial_slip | 2 | 17 % | 2 | 14.9 | 14.0 |
| flow_redirection | 0 | 0 % | 0 | – | – |
| trapped_gas_or_liquid | 0 | 0 % | 0 | – | – |
| porous_transport | 1 | 8 % | 1 | 4.6 | 4.6 |
| graded_stiffness | 8 | 67 % | 5 | 27.7 | 24.9 |
| architected_lattice | 1 | 8 % | 1 | 22.5 | 22.5 |
| radiative_control | 0 | 0 % | 0 | – | – |
| phase_change | 0 | 0 % | 0 | – | – |
| electrostatic_field | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

**length_scale** (ordinal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| nm | 2 | 17 % | 2 | 14.9 | 14.0 |
| sub_um | 0 | 0 % | 0 | – | – |
| um | 0 | 0 % | 0 | – | – |
| 10_um | 1 | 8 % | 1 | 4.6 | 4.6 |
| 100_um | 0 | 0 % | 0 | – | – |
| mm | 9 | 75 % | 6 | 27.7 | 24.5 |
| cm_plus | 0 | 0 % | 0 | – | – |

**inspiration_origin** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| plant | 0 | 0 % | 0 | – | – |
| animal | 7 | 58 % | 6 | 27.7 | 18.1 |
| microbe | 0 | 0 % | 0 | – | – |
| geology | 2 | 17 % | 1 | 27.5 | 27.5 |
| atmosphere_ocean | 3 | 25 % | 2 | 24.5 | 21.6 |
| technology | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

**governing_quantity** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| wall_shear | 9 | 75 % | 6 | 27.7 | 20.0 |
| flow_rate | 0 | 0 % | 0 | – | – |
| heat_flux | 0 | 0 % | 0 | – | – |
| temperature | 0 | 0 % | 0 | – | – |
| deflection | 0 | 0 % | 0 | – | – |
| stress | 3 | 25 % | 3 | 26.0 | 19.9 |
| field_strength | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

## Values never proposed

- `mechanism_class` (top value holds 67 % of proposals): flow_redirection, trapped_gas_or_liquid, radiative_control, phase_change, electrostatic_field, other
- `length_scale` (top value holds 75 % of proposals): sub_um, um, 100_um, cm_plus
- `inspiration_origin` (top value holds 58 % of proposals): plant, microbe, technology, other
- `governing_quantity` (top value holds 75 % of proposals): every relevant value proposed (not relevant to the request, not searched: flow_rate, heat_flux, temperature, deflection, field_strength, other)

## Promising under-explored cells (next fill-gap targets)

| empty cell | priority | best neighbour | elite neighbours | proposals | under-explored | step from best |
|---|---|---|---|---|---|---|
| mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=animal, governing_quantity=wall_shear | 0.465 | 27.7 | 1 | 0 | 0.75 | length_scale: mm -> 100_um |
| mechanism_class=graded_stiffness, length_scale=cm_plus, inspiration_origin=animal, governing_quantity=wall_shear | 0.465 | 27.7 | 1 | 0 | 0.75 | length_scale: mm -> cm_plus |
| mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=geology, governing_quantity=wall_shear | 0.463 | 27.5 | 1 | 0 | 0.75 | length_scale: mm -> 100_um |
| mechanism_class=graded_stiffness, length_scale=cm_plus, inspiration_origin=geology, governing_quantity=wall_shear | 0.463 | 27.5 | 1 | 0 | 0.75 | length_scale: mm -> cm_plus |
| mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=animal, governing_quantity=stress | 0.448 | 26.0 | 1 | 0 | 0.75 | length_scale: mm -> 100_um |
| mechanism_class=graded_stiffness, length_scale=cm_plus, inspiration_origin=animal, governing_quantity=stress | 0.448 | 26.0 | 1 | 0 | 0.75 | length_scale: mm -> cm_plus |
| mechanism_class=flow_redirection, length_scale=mm, inspiration_origin=animal, governing_quantity=wall_shear | 0.444 | 27.7 | 2 | 0 | 0.67 | mechanism_class: graded_stiffness -> flow_redirection |
| mechanism_class=other, length_scale=mm, inspiration_origin=animal, governing_quantity=wall_shear | 0.444 | 27.7 | 2 | 0 | 0.67 | mechanism_class: graded_stiffness -> other |

## Strategy yield (all rounds)

| strategy | orders | evaluated | new elite | improved | not better | failed | infeasible | invalid/rejected | missing | off target | elites now |
|---|---|---|---|---|---|---|---|---|---|---|---|
| refine | 3 | 3 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 2 |
| fill_gap | 4 | 3 | 3 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 |
| combine | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| diversify | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| explore | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| seed | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |

## Ordinal trends (current)

| axis | mode | held fixed | points (value: score) | slope/step | r2 | next | status |
|---|---|---|---|---|---|---|---|
| length_scale | marginal | (marginal) | nm: 14.9, 10_um: 4.6, mm: 27.7 | +2.09 | 0.20 | – | poor_fit |

## Trends used for extrapolation

_No extrapolation orders yet._

## Infeasible cells

| cell / pattern | reason | reported by | round | times |
|---|---|---|---|---|
| mechanism_class=electrostatic_field, length_scale=mm, inspiration_origin=animal, governing_quantity=wall_shear | Seawater screens static fields within the Debye length (~0.4 nm), so a mm-scale electric field can only be sustained by an ohmic current (sigma ~ 5 S/m: ~50 W/m2 for 100 V/m over 1 mm, comparable with the entire friction power density of ~126 W/m2) that electrolyses seawater into chlorine/hypochlorite, violating both the energy balance and the non-toxic requirement; no electrostatic mechanism at this scale can lower wall shear. | generator | 4 | 1 |
| mechanism_class=graded_stiffness, length_scale=nm, inspiration_origin=animal, governing_quantity=wall_shear | A graded-stiffness layer thinner than 100 nm deforms by only ~0.02 nm under hull wall shear (tau ~ 20 Pa, G ~ 0.1 MPa, strain 2e-4), about five orders of magnitude below the viscous length (~5 um), so compliance at the nm scale cannot change wall shear; its only possible effect is on fouling adhesion, which belongs to the stress cell, not to wall_shear. | generator | 2 | 1 |
| mechanism_class=phase_change, length_scale=10_um, inspiration_origin=microbe, governing_quantity=wall_shear | A phase-change drag layer on a hull needs either a vapour film (film boiling needs ~1e5 W/m2, i.e. ~1 GW for 1e4 m2, about 100x the ship's propulsion power, violating the energy balance of the request) or spontaneous degassing, which cannot occur because hydrostatic pressure keeps seawater under-saturated at hull depth; microbial gas production from seawater organics is 3-4 orders of magnitude too slow to feed even a thin gas layer. | generator | 3 | 1 |
