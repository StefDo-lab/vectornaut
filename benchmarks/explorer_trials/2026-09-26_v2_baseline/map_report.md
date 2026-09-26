# Explorer map: materials

Query: "Entwickle eine bionisch inspirierte Beschichtung für Schiffsrümpfe, die den Reibungswiderstand im Wasser senkt, ohne giftige Antifouling-Wirkstoffe auszukommen, und mehrere Jahre im Salzwasser hält."  
Archive: `/tmp/claude-0/-home-user-vectornaut/6d94a234-0063-56c5-b9d5-7a9e3882a427/scratchpad/explorer_v2_baseline/session/data/explorer/materials/archive.json`  
Rounds so far: 1

> Score = 100 * gate * (0.5 * gain_score + 0.5 * requirement coverage). Tier 'simulated' uses the simulated gain (the lower of simulation and critic if they differ by more than 2x); tier 'estimated' (no usable simulated gain, e.g. flagged implausible_gain) uses the lower of the critic's and the candidate's estimate at half weight and is capped at 50. Elites are ranked by tier first.

## Coverage

- Cells in the grid: 3920 (10 x 7 x 7 x 8)
- Cells with an elite: 13 (0.33 %)
- Cells with at least one proposal: 14
- Infeasible cells/patterns: 0
- Entries: 15 (evaluated: 14, failed: 1)
- Elite score range: 16.2 – 33.1
- Elites by evidence tier: estimated: 2, simulated: 11
- Flags on evaluated entries: gain_unavailable: 2, model_assumption_sensitive: 12

## Requirements of the request

Extracted by the generator's function analysis (round 1); fixed for this map.

- **low_friction_drag**: Measurably lowers hull skin friction (wall shear) at service speed versus a clean state-of-the-art foul-release coating.
- **non_toxic_fouling_control**: Keeps fouling off or releasable without biocides and without releasing oils, microplastics or electrochemically generated oxidants in relevant amounts.
- **multi_year_seawater_durability**: Keeps adhesion, structure and function for at least 3-5 years in seawater, including cleaning and impacts.
- **low_resource_consumption**: Needs no continuous supply of consumables and only energy that is small against the propulsion saving.
- **bionic_inspiration**: Derived from a biological model whose principle is actually transferred.

Relevant `governing_quantity` values (explore/fill-gap/diversify use only these): wall_shear, stress (function analysis)

## Elites (best per cell)

| # | score | tier | req. coverage | gain used % | simulated % | critic % | flags | title | cell | strategy | round | entry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 33.1 | simulated | 0.48 | 4.0 | 17.5 | 4.0 | model_assumption_sensitive | Penguin-plumage micro-ribbed bottom coating that stabilises an injected air layer | mechanism_class=trapped_gas_or_liquid, length_scale=100_um, inspiration_origin=animal, governing_quantity=wall_shear | seed | 1 | e00002 |
| 2 | 30.6 | simulated | 0.54 | 1.5 | 4.5 | 1.5 | model_assumption_sensitive | Foul-release silicone riblets with flexing fouling-shedding crests | mechanism_class=flow_redirection, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | seed | 1 | e00001 |
| 3 | 27.5 | simulated | 0.55 | 0.0 | 0.1 | 0.0 | model_assumption_sensitive | Pilot-whale nanoridge skin with zwitterionic hydrogel infill | mechanism_class=interfacial_slip, length_scale=sub_um, inspiration_origin=animal, governing_quantity=wall_shear | seed | 1 | e00007 |
| 4 | 24.9 | simulated | 0.45 | 1.0 | 4.5 | 1.0 | model_assumption_sensitive | Xylem-like anisotropic permeable skin (streamwise-open, spanwise-sealed) | mechanism_class=porous_transport, length_scale=10_um, inspiration_origin=plant, governing_quantity=wall_shear | seed | 1 | e00005 |
| 5 | 24.1 | simulated | 0.41 | 1.5 | 4.5 | 1.5 | model_assumption_sensitive | Sinusoidal riblets emulating spanwise wall oscillation | mechanism_class=flow_redirection, length_scale=100_um, inspiration_origin=technology, governing_quantity=wall_shear | seed | 1 | e00011 |
| 6 | 22.7 | simulated | 0.43 | 0.5 | 3.9 | 0.5 | model_assumption_sensitive | Diving-bell-spider plastron fed through a gas-permeable membrane backing | mechanism_class=trapped_gas_or_liquid, length_scale=um, inspiration_origin=animal, governing_quantity=wall_shear | seed | 1 | e00003 |
| 7 | 22.5 | simulated | 0.45 | 0.0 | 1.0 | 0.0 | model_assumption_sensitive | Dolphin-skin graded compliant coating tuned to ship speed | mechanism_class=graded_stiffness, length_scale=mm, inspiration_origin=animal, governing_quantity=wall_shear | seed | 1 | e00006 |
| 8 | 20.0 | simulated | 0.40 | 0.0 | 0.4 | 0.0 | model_assumption_sensitive | Rice-leaf anisotropic superhydrophobic streamwise grooves | mechanism_class=trapped_gas_or_liquid, length_scale=10_um, inspiration_origin=plant, governing_quantity=wall_shear | seed | 1 | e00010 |
| 9 | 18.0 | simulated | 0.36 | 0.0 | 0.0 | 0.0 | model_assumption_sensitive | River-pebble polished glass-ceramic topcoat with liquid-like brushes | mechanism_class=interfacial_slip, length_scale=nm, inspiration_origin=geology, governing_quantity=wall_shear | seed | 1 | e00015 |
| 10 | 17.4 | simulated | 0.30 | 1.0 | 3.1 | 1.0 | model_assumption_sensitive | Swordfish-rostrum pore-fed lubricant surface | mechanism_class=trapped_gas_or_liquid, length_scale=10_um, inspiration_origin=animal, governing_quantity=wall_shear | seed | 1 | e00014 |
| 11 | 16.2 | simulated | 0.30 | 0.5 | 3.6 | 0.5 | model_assumption_sensitive | Cathodic-protection-coupled electrolytic plastron regeneration | mechanism_class=trapped_gas_or_liquid, length_scale=10_um, inspiration_origin=technology, governing_quantity=wall_shear | seed | 1 | e00004 |
| 12 | 28.0 | estimated | 0.56 | 0.0 | 0.0 | 0.0 | gain_unavailable | Graded-modulus foul-release coating on a mussel-DOPA tie layer | mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=animal, governing_quantity=stress | seed | 1 | e00013 |
| 13 | 20.5 | estimated | 0.41 | 0.0 | 0.0 | 0.0 | gain_unavailable | Kelp-epidermis lamellar self-shedding foul-release coating | mechanism_class=other, length_scale=10_um, inspiration_origin=plant, governing_quantity=stress | seed | 1 | e00008 |

## Requirement coverage (critic ratings, elites)

| entry | title | low_friction_drag | non_toxic_fouling_control | multi_year_seawater_durability | low_resource_consumption | bionic_inspiration | mean |
|---|---|---|---|---|---|---|---|
| e00002 | Penguin-plumage micro-ribbed bottom coating that stabilises an injected air layer | 0.70 | 0.40 | 0.55 | 0.40 | 0.35 | 0.48 |
| e00001 | Foul-release silicone riblets with flexing fouling-shedding crests | 0.30 | 0.45 | 0.30 | 0.90 | 0.75 | 0.54 |
| e00007 | Pilot-whale nanoridge skin with zwitterionic hydrogel infill | 0.05 | 0.60 | 0.35 | 1.00 | 0.75 | 0.55 |
| e00005 | Xylem-like anisotropic permeable skin (streamwise-open, spanwise-sealed) | 0.30 | 0.30 | 0.20 | 1.00 | 0.45 | 0.45 |
| e00011 | Sinusoidal riblets emulating spanwise wall oscillation | 0.30 | 0.40 | 0.30 | 0.90 | 0.15 | 0.41 |
| e00003 | Diving-bell-spider plastron fed through a gas-permeable membrane backing | 0.25 | 0.35 | 0.15 | 0.60 | 0.80 | 0.43 |
| e00006 | Dolphin-skin graded compliant coating tuned to ship speed | 0.05 | 0.50 | 0.20 | 0.90 | 0.60 | 0.45 |
| e00010 | Rice-leaf anisotropic superhydrophobic streamwise grooves | 0.05 | 0.20 | 0.10 | 1.00 | 0.65 | 0.40 |
| e00015 | River-pebble polished glass-ceramic topcoat with liquid-like brushes | 0.05 | 0.30 | 0.45 | 0.90 | 0.10 | 0.36 |
| e00014 | Swordfish-rostrum pore-fed lubricant surface | 0.30 | 0.15 | 0.30 | 0.15 | 0.60 | 0.30 |
| e00004 | Cathodic-protection-coupled electrolytic plastron regeneration | 0.25 | 0.10 | 0.20 | 0.65 | 0.30 | 0.30 |
| e00013 | Graded-modulus foul-release coating on a mussel-DOPA tie layer | 0.05 | 0.55 | 0.60 | 1.00 | 0.60 | 0.56 |
| e00008 | Kelp-epidermis lamellar self-shedding foul-release coating | 0.10 | 0.35 | 0.35 | 0.45 | 0.80 | 0.41 |

## Map: mechanism_class x length_scale

Rows: `mechanism_class`, columns: `length_scale`. Each cell: best elite score over all other axes (number of proposals that landed there). `·` = no elite yet, `x` = contains cells reported infeasible.

| mechanism_class \ length_scale | nm | sub_um | um | 10_um | 100_um | mm | cm_plus |
|---|---|---|---|---|---|---|---|
| interfacial_slip | 18.0 (1) | 27.5 (2) | · (0) | · (0) | · (0) | · (0) | · (0) |
| flow_redirection | · (0) | · (0) | · (0) | 30.6 (1) | 24.1 (1) | · (0) | · (0) |
| trapped_gas_or_liquid | · (0) | · (0) | 22.7 (1) | 20.0 (3) | 33.1 (1) | · (0) | · (0) |
| porous_transport | · (0) | · (0) | · (0) | 24.9 (1) | · (0) | · (0) | · (0) |
| graded_stiffness | · (0) | · (0) | · (0) | · (0) | 28.0 (1) | 22.5 (1) | · (0) |
| architected_lattice | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| radiative_control | · (0) | · (0) | · (0) | · (0) | · (1) | · (0) | · (0) |
| phase_change | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| electrostatic_field | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| other | · (0) | · (0) | · (0) | 20.5 (1) | · (0) | · (0) | · (0) |

## Where ideas are proposed vs where they work

**mechanism_class** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| interfacial_slip | 3 | 20 % | 2 | 27.5 | 22.8 |
| flow_redirection | 2 | 13 % | 2 | 30.6 | 27.4 |
| trapped_gas_or_liquid | 5 | 33 % | 5 | 33.1 | 21.9 |
| porous_transport | 1 | 7 % | 1 | 24.9 | 24.9 |
| graded_stiffness | 2 | 13 % | 2 | 28.0 | 25.2 |
| architected_lattice | 0 | 0 % | 0 | – | – |
| radiative_control | 1 | 7 % | 0 | – | – |
| phase_change | 0 | 0 % | 0 | – | – |
| electrostatic_field | 0 | 0 % | 0 | – | – |
| other | 1 | 7 % | 1 | 20.5 | 20.5 |

**length_scale** (ordinal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| nm | 1 | 7 % | 1 | 18.0 | 18.0 |
| sub_um | 2 | 13 % | 1 | 27.5 | 27.5 |
| um | 1 | 7 % | 1 | 22.7 | 22.7 |
| 10_um | 6 | 40 % | 6 | 30.6 | 21.6 |
| 100_um | 4 | 27 % | 3 | 33.1 | 28.4 |
| mm | 1 | 7 % | 1 | 22.5 | 22.5 |
| cm_plus | 0 | 0 % | 0 | – | – |

**inspiration_origin** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| plant | 3 | 20 % | 3 | 24.9 | 21.8 |
| animal | 9 | 60 % | 7 | 33.1 | 26.0 |
| microbe | 0 | 0 % | 0 | – | – |
| geology | 1 | 7 % | 1 | 18.0 | 18.0 |
| atmosphere_ocean | 0 | 0 % | 0 | – | – |
| technology | 2 | 13 % | 2 | 24.1 | 20.2 |
| other | 0 | 0 % | 0 | – | – |

**governing_quantity** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| wall_shear | 12 | 80 % | 11 | 33.1 | 23.4 |
| flow_rate | 0 | 0 % | 0 | – | – |
| heat_flux | 0 | 0 % | 0 | – | – |
| temperature | 0 | 0 % | 0 | – | – |
| deflection | 0 | 0 % | 0 | – | – |
| stress | 2 | 13 % | 2 | 28.0 | 24.2 |
| field_strength | 0 | 0 % | 0 | – | – |
| other | 1 | 7 % | 0 | – | – |

## Values never proposed

- `mechanism_class` (top value holds 33 % of proposals): architected_lattice, phase_change, electrostatic_field
- `length_scale` (top value holds 40 % of proposals): cm_plus
- `inspiration_origin` (top value holds 60 % of proposals): microbe, atmosphere_ocean, other
- `governing_quantity` (top value holds 80 % of proposals): every relevant value proposed (not relevant to the request, not searched: flow_rate, heat_flux, temperature, deflection, field_strength)

## Promising under-explored cells (next fill-gap targets)

| empty cell | priority | best neighbour | elite neighbours | proposals | under-explored | step from best |
|---|---|---|---|---|---|---|
| mechanism_class=trapped_gas_or_liquid, length_scale=100_um, inspiration_origin=atmosphere_ocean, governing_quantity=wall_shear | 0.481 | 33.1 | 1 | 0 | 0.60 | inspiration_origin: animal -> atmosphere_ocean |
| mechanism_class=trapped_gas_or_liquid, length_scale=100_um, inspiration_origin=microbe, governing_quantity=wall_shear | 0.481 | 33.1 | 1 | 0 | 0.60 | inspiration_origin: animal -> microbe |
| mechanism_class=trapped_gas_or_liquid, length_scale=100_um, inspiration_origin=other, governing_quantity=wall_shear | 0.481 | 33.1 | 1 | 0 | 0.60 | inspiration_origin: animal -> other |
| mechanism_class=trapped_gas_or_liquid, length_scale=100_um, inspiration_origin=technology, governing_quantity=wall_shear | 0.481 | 33.1 | 3 | 0 | 0.20 | inspiration_origin: animal -> technology |
| mechanism_class=flow_redirection, length_scale=100_um, inspiration_origin=animal, governing_quantity=wall_shear | 0.458 | 33.1 | 3 | 0 | 0.11 | mechanism_class: trapped_gas_or_liquid -> flow_redirection |
| mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=animal, governing_quantity=wall_shear | 0.458 | 33.1 | 3 | 0 | 0.11 | mechanism_class: trapped_gas_or_liquid -> graded_stiffness |
| mechanism_class=flow_redirection, length_scale=10_um, inspiration_origin=atmosphere_ocean, governing_quantity=wall_shear | 0.456 | 30.6 | 1 | 0 | 0.60 | inspiration_origin: animal -> atmosphere_ocean |
| mechanism_class=flow_redirection, length_scale=10_um, inspiration_origin=microbe, governing_quantity=wall_shear | 0.456 | 30.6 | 1 | 0 | 0.60 | inspiration_origin: animal -> microbe |

## Strategy yield (all rounds)

| strategy | orders | evaluated | new elite | improved | not better | failed | infeasible | invalid/rejected | missing | off target | elites now |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed | 15 | 14 | 13 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 13 |

## Ordinal trends (current)

| axis | mode | held fixed | points (value: score) | slope/step | r2 | next | status |
|---|---|---|---|---|---|---|---|
| length_scale | slice | mechanism_class=trapped_gas_or_liquid, inspiration_origin=animal, governing_quantity=wall_shear | um: 22.7, 10_um: 17.4, 100_um: 33.1 | +5.16 | 0.42 | – | poor_fit |
| length_scale | marginal | (marginal) | nm: 18.0, sub_um: 27.5, um: 22.7, 10_um: 30.6, 100_um: 33.1, mm: 22.5 | +1.34 | 0.20 | – | peaked |

## Trends used for extrapolation

_No extrapolation orders yet._

## Infeasible cells

_None._
