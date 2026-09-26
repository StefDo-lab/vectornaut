# Explorer map: materials

Query: "Entwickle eine bionisch inspirierte Beschichtung für Schiffsrümpfe, die den Reibungswiderstand im Wasser senkt, ohne giftige Antifouling-Wirkstoffe auszukommen, und mehrere Jahre im Salzwasser hält."  
Archive: `/tmp/claude-0/-home-user-vectornaut/6d94a234-0063-56c5-b9d5-7a9e3882a427/scratchpad/explorer_v3_guided/session/data/explorer/materials/archive.json`  
Rounds so far: 6

> score = 100 * gate * (0.35 * objective_score + 0.15 * simulated_score + 0.5 * requirement_coverage) * relabel_factor; gate = validator_score * (1 pass / 0.8 warn / 0 fail); objective_score = (1 - exp(-objective_gain_pct / 10)) * d, d = 1 if a critic judged the objective next to a relevant simulation, else 0.5; simulated_score = 1 - exp(-simulated_benefit_pct / 20) if the simulated quantity is relevant, else 0; relabel_factor = 0.5 for a relabelled analogue, else 1; estimated tier capped at 50. Tier 'simulated' only when a usable simulated gain on a relevant governing quantity enters the score (the lower of simulation and critic if they differ strongly); otherwise tier 'estimated' (the simulated number is shown but not scored). The objective gain is the critic's plausible contribution to the stated objective against the stated baseline. 'points obj / sim / req' splits the score into its three parts (before the cap). Elites are ranked by tier first.

## Coverage

- Cells in the grid: 4900 (10 x 7 x 7 x 10)
- Cells with an elite: 15 (0.31 %)
- Cells with at least one proposal: 15
- Infeasible cells/patterns: 0
- Entries: 18 (evaluated: 18)
- Elite score range: 15.4 – 42.1
- Elites by evidence tier: estimated: 2, simulated: 13
- Flags on evaluated entries: baseline_not_conventional: 1, gain_unavailable: 1, model_assumption_sensitive: 5, simulated_quantity_irrelevant: 1

## Requirements of the request

Objective and conventional baseline from the function analysis (round 1); fixed for this map.

- **Objective**: Time-averaged hull frictional resistance of a merchant ship over a 5-year docking interval in seawater (mixed operating profile around 12-16 kn with idle periods), including the drag added by slime and macrofouling and by coating degradation, not the friction of a freshly applied clean surface.
- **Baseline**: Conventional biocide-free silicone (PDMS/fluoropolymer-modified) foul-release coating averaged over the same 5-year interval, i.e. in its typical in-service state after 12-24 months with a light-to-medium slime layer and occasional macrofouling during idle periods.

Extracted by the generator's function analysis (round 1); fixed for this map.

- **friction_drag_reduction**: Lowers the time-averaged hull friction resistance over the service interval compared with the conventional biocide-free foul-release coating.
- **no_toxic_biocides**: Releases no biocides, heavy metals, or other toxic or persistent substances (including leaching oils or polymer fragments) into seawater.
- **multi_year_seawater_durability**: Keeps adhesion, integrity and function for at least 5 years in seawater, including abrasion, UV at the waterline, cathodic protection and hydrolysis.
- **fouling_control**: Limits slime and macrofouling (or allows its easy release at service speeds or by gentle cleaning) throughout the interval, including idle periods.
- **bionic_mechanism**: The working mechanism is genuinely derived from a biological model.
- **practical_application**: Can be applied to large steel hulls with shipyard processes at a cost comparable to premium coatings and without continuous energy or material supply.

Relevant `governing_quantity` values (explore/fill-gap/diversify use only these): wall_shear, stress, fouling_adhesion, degradation_rate (function analysis)

## Elites (best per cell)

| # | score | tier | points obj / sim / req | objective gain % | sim. benefit used % | simulated % | simulated quantity | req. coverage | flags | title | cell | strategy | round | entry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 42.1 | simulated | 3.3 / 13.8 / 25.0 | 1.0 | 49.9 | 49.9 | fouling_adhesion | 0.50 | – | Pilot-whale graded soft skin: thick ultra-soft silicone over stiff tie layer | mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=animal, governing_quantity=fouling_adhesion | seed | 1 | e00001 |
| 2 | 41.3 | simulated | 3.3 / 13.8 / 24.2 | 1.0 | 51.3 | 51.3 | fouling_adhesion | 0.48 | – | Palm-stem anisotropic release layer: normal-aligned polymer microfibres in ultra-soft silicone | mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=plant, governing_quantity=fouling_adhesion | diversify | 3 | e00007 |
| 3 | 40.8 | simulated | 4.9 / 14.7 / 21.2 | 1.5 | 79.3 | 79.3 | fouling_adhesion | 0.42 | – | Glacier soft-bed release: tough thin skin on an ultra-compliant decoupling layer | mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=atmosphere_ocean, governing_quantity=fouling_adhesion | fill_gap | 2 | e00006 |
| 4 | 39.9 | simulated | 2.7 / 14.7 / 22.5 | 0.8 | 77.8 | 77.8 | degradation_rate | 0.45 | – | Oil-free supersoft palm-stem layer: bottle-brush silicone matrix with embedded fibres and near-zero extractables | mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=plant, governing_quantity=degradation_rate | combine | 5 | e00014 |
| 5 | 39.7 | simulated | 1.0 / 14.1 / 24.6 | 0.3 | 55.3 | 55.3 | stress | 0.49 | – | Plant-cuticle graded interphase: compositionally graded tie layer that halves edge delamination stress | mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=plant, governing_quantity=stress | fill_gap | 4 | e00010 |
| 6 | 39.3 | simulated | 2.4 / 14.9 / 22.1 | 0.7 | 99.2 | 99.2 | degradation_rate | 0.44 | – | Oil-free bound hydration reservoir: slowly cleaving zwitterion tethers replace leaching silicone oil | mechanism_class=trapped_gas_or_liquid, length_scale=nm, inspiration_origin=animal, governing_quantity=degradation_rate | combine | 2 | e00004 |
| 7 | 38.6 | simulated | 1.7 / 14.8 / 22.1 | 0.5 | 84.1 | 84.1 | degradation_rate | 0.44 | – | Wax-platelet cuticle skin: tortuous-path barrier over an oil-free soft layer | mechanism_class=graded_stiffness, length_scale=10_um, inspiration_origin=plant, governing_quantity=degradation_rate | fill_gap | 6 | e00016 |
| 8 | 38.0 | simulated | 1.7 / 13.0 / 23.3 | 0.5 | 40.0 | 95.0 | stress | 0.47 | model_assumption_sensitive | Cartilage hydration-lubricated topcoat: phosphocholine boundary layer that makes foul-release coatings groomable | mechanism_class=trapped_gas_or_liquid, length_scale=nm, inspiration_origin=animal, governing_quantity=stress | fill_gap | 4 | e00012 |
| 9 | 38.0 | simulated | 1.7 / 13.8 / 22.5 | 0.5 | 49.8 | 49.8 | fouling_adhesion | 0.45 | – | Cyprid-disc-scale wrinkled soft skin: 30-50 um buckling wrinkles on a graded silicone | mechanism_class=graded_stiffness, length_scale=10_um, inspiration_origin=animal, governing_quantity=fouling_adhesion | diversify | 6 | e00017 |
| 10 | 36.9 | simulated | 3.3 / 10.7 / 22.9 | 1.0 | 25.0 | 60.1 | fouling_adhesion | 0.46 | model_assumption_sensitive | Mucin bottle-brush hydration skin with idle-period protection | mechanism_class=trapped_gas_or_liquid, length_scale=nm, inspiration_origin=animal, governing_quantity=fouling_adhesion | refine | 2 | e00005 |
| 11 | 32.1 | simulated | 3.3 / 13.7 / 15.0 | 1.0 | 49.5 | 49.5 | fouling_adhesion | 0.30 | – | Pulsed-heater thermoresponsive release: patch-wise heating of an LCST hydrogel brush | mechanism_class=phase_change, length_scale=sub_um, inspiration_origin=technology, governing_quantity=fouling_adhesion | explore | 3 | e00008 |
| 12 | 31.7 | simulated | 0.0 / 11.7 / 20.0 | -0.5 | 30.0 | 70.0 | stress | 0.40 | model_assumption_sensitive | Alginate-capsule self-healing gel: seawater-calcium crosslinked polyanion layer that absorbs contact shear | mechanism_class=trapped_gas_or_liquid, length_scale=nm, inspiration_origin=microbe, governing_quantity=stress | explore | 6 | e00018 |
| 13 | 20.6 | simulated | 1.0 / 0.0 / 19.6 | 0.3 | 0.0 | 1.1 | wall_shear | 0.39 | model_assumption_sensitive | Inclined-fibre anisotropic compliant skin: palm-stem fibre angle tuned to the viscous sublayer | mechanism_class=graded_stiffness, length_scale=100_um, inspiration_origin=plant, governing_quantity=wall_shear | diversify | 5 | e00013 |
| 14 | 20.0 | estimated | 0.0 / 0.0 / 20.0 | -0.5 | – | 100.0 | degradation_rate | 0.40 | simulated_quantity_irrelevant | Smectite interlayer-water skin: exfoliated clay platelets bound in a silicone surface | mechanism_class=trapped_gas_or_liquid, length_scale=nm, inspiration_origin=geology, governing_quantity=degradation_rate | fill_gap | 3 | e00009 |
| 15 | 15.4 | estimated | 0.0 / 0.0 / 15.4 | -4.0 | – | 0.0 | degradation_rate | 0.31 | gain_unavailable, baseline_not_conventional | Coral-mucus sloughing skin: slowly hydrolysing biocide-free renewal layer | mechanism_class=other, length_scale=100_um, inspiration_origin=animal, governing_quantity=degradation_rate | seed | 1 | e00003 |

## Requirement coverage (critic ratings, elites)

| entry | title | friction_drag_reduction | no_toxic_biocides | multi_year_seawater_durability | fouling_control | bionic_mechanism | practical_application | mean |
|---|---|---|---|---|---|---|---|---|
| e00001 | Pilot-whale graded soft skin: thick ultra-soft silicone over stiff tie layer | 0.30 | 0.80 | 0.35 | 0.45 | 0.50 | 0.60 | 0.50 |
| e00007 | Palm-stem anisotropic release layer: normal-aligned polymer microfibres in ultra-soft silicone | 0.30 | 0.70 | 0.45 | 0.45 | 0.55 | 0.45 | 0.48 |
| e00006 | Glacier soft-bed release: tough thin skin on an ultra-compliant decoupling layer | 0.35 | 0.70 | 0.40 | 0.45 | 0.10 | 0.55 | 0.42 |
| e00014 | Oil-free supersoft palm-stem layer: bottle-brush silicone matrix with embedded fibres and near-zero extractables | 0.30 | 0.80 | 0.35 | 0.45 | 0.45 | 0.35 | 0.45 |
| e00010 | Plant-cuticle graded interphase: compositionally graded tie layer that halves edge delamination stress | 0.15 | 0.90 | 0.55 | 0.35 | 0.45 | 0.55 | 0.49 |
| e00004 | Oil-free bound hydration reservoir: slowly cleaving zwitterion tethers replace leaching silicone oil | 0.20 | 0.85 | 0.35 | 0.35 | 0.35 | 0.55 | 0.44 |
| e00016 | Wax-platelet cuticle skin: tortuous-path barrier over an oil-free soft layer | 0.20 | 0.80 | 0.45 | 0.30 | 0.45 | 0.45 | 0.44 |
| e00012 | Cartilage hydration-lubricated topcoat: phosphocholine boundary layer that makes foul-release coatings groomable | 0.20 | 0.85 | 0.30 | 0.35 | 0.60 | 0.50 | 0.47 |
| e00017 | Cyprid-disc-scale wrinkled soft skin: 30-50 um buckling wrinkles on a graded silicone | 0.25 | 0.90 | 0.35 | 0.40 | 0.40 | 0.40 | 0.45 |
| e00005 | Mucin bottle-brush hydration skin with idle-period protection | 0.25 | 0.85 | 0.20 | 0.40 | 0.60 | 0.45 | 0.46 |
| e00008 | Pulsed-heater thermoresponsive release: patch-wise heating of an LCST hydrogel brush | 0.25 | 0.80 | 0.20 | 0.45 | 0.00 | 0.10 | 0.30 |
| e00018 | Alginate-capsule self-healing gel: seawater-calcium crosslinked polyanion layer that absorbs contact shear | 0.10 | 0.85 | 0.30 | 0.15 | 0.50 | 0.50 | 0.40 |
| e00013 | Inclined-fibre anisotropic compliant skin: palm-stem fibre angle tuned to the viscous sublayer | 0.15 | 0.75 | 0.35 | 0.35 | 0.40 | 0.35 | 0.39 |
| e00009 | Smectite interlayer-water skin: exfoliated clay platelets bound in a silicone surface | 0.10 | 0.90 | 0.50 | 0.15 | 0.05 | 0.70 | 0.40 |
| e00003 | Coral-mucus sloughing skin: slowly hydrolysing biocide-free renewal layer | 0.10 | 0.30 | 0.30 | 0.15 | 0.40 | 0.60 | 0.31 |

## Map: mechanism_class x length_scale

Rows: `mechanism_class`, columns: `length_scale`. Each cell: best elite score over all other axes (number of proposals that landed there). `·` = no elite yet, `x` = contains cells reported infeasible.

| mechanism_class \ length_scale | nm | sub_um | um | 10_um | 100_um | mm | cm_plus |
|---|---|---|---|---|---|---|---|
| interfacial_slip | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| flow_redirection | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| trapped_gas_or_liquid | 39.3 (6) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| porous_transport | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| graded_stiffness | · (0) | · (0) | · (0) | 38.6 (2) | 42.1 (7) | · (0) | · (0) |
| architected_lattice | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| radiative_control | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| phase_change | · (0) | 32.1 (2) | · (0) | · (0) | · (0) | · (0) | · (0) |
| electrostatic_field | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| other | · (0) | · (0) | · (0) | · (0) | 15.4 (1) | · (0) | · (0) |

## Where ideas are proposed vs where they work

**mechanism_class** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| interfacial_slip | 0 | 0 % | 0 | – | – |
| flow_redirection | 0 | 0 % | 0 | – | – |
| trapped_gas_or_liquid | 6 | 33 % | 5 | 39.3 | 33.2 |
| porous_transport | 0 | 0 % | 0 | – | – |
| graded_stiffness | 9 | 50 % | 8 | 42.1 | 37.6 |
| architected_lattice | 0 | 0 % | 0 | – | – |
| radiative_control | 0 | 0 % | 0 | – | – |
| phase_change | 2 | 11 % | 1 | 32.1 | 32.1 |
| electrostatic_field | 0 | 0 % | 0 | – | – |
| other | 1 | 6 % | 1 | 15.4 | 15.4 |

**length_scale** (ordinal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| nm | 6 | 33 % | 5 | 39.3 | 33.2 |
| sub_um | 2 | 11 % | 1 | 32.1 | 32.1 |
| um | 0 | 0 % | 0 | – | – |
| 10_um | 2 | 11 % | 2 | 38.6 | 38.3 |
| 100_um | 8 | 44 % | 7 | 42.1 | 34.3 |
| mm | 0 | 0 % | 0 | – | – |
| cm_plus | 0 | 0 % | 0 | – | – |

**inspiration_origin** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| plant | 5 | 28 % | 5 | 41.3 | 36.0 |
| animal | 7 | 39 % | 6 | 42.1 | 35.0 |
| microbe | 1 | 6 % | 1 | 31.7 | 31.7 |
| geology | 1 | 6 % | 1 | 20.0 | 20.0 |
| atmosphere_ocean | 2 | 11 % | 1 | 40.8 | 40.8 |
| technology | 2 | 11 % | 1 | 32.1 | 32.1 |
| other | 0 | 0 % | 0 | – | – |

**governing_quantity** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| wall_shear | 1 | 6 % | 1 | 20.6 | 20.6 |
| flow_rate | 0 | 0 % | 0 | – | – |
| heat_flux | 0 | 0 % | 0 | – | – |
| temperature | 0 | 0 % | 0 | – | – |
| deflection | 0 | 0 % | 0 | – | – |
| stress | 3 | 17 % | 3 | 39.7 | 36.4 |
| fouling_adhesion | 9 | 50 % | 6 | 42.1 | 38.5 |
| degradation_rate | 5 | 28 % | 5 | 39.9 | 30.6 |
| field_strength | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

## Values never proposed

- `mechanism_class` (top value holds 50 % of proposals): interfacial_slip, flow_redirection, porous_transport, architected_lattice, radiative_control, electrostatic_field
- `length_scale` (top value holds 44 % of proposals): um, mm, cm_plus
- `inspiration_origin` (top value holds 39 % of proposals): other
- `governing_quantity` (top value holds 50 % of proposals): every relevant value proposed (not relevant to the request, not searched: flow_rate, heat_flux, temperature, deflection, field_strength, other)

## Promising under-explored cells (next fill-gap targets)

| empty cell | priority | best neighbour | elite neighbours | proposals | under-explored | step from source (rotation-adjusted) |
|---|---|---|---|---|---|---|
| mechanism_class=graded_stiffness, length_scale=10_um, inspiration_origin=plant, governing_quantity=fouling_adhesion | 0.523 | 41.3 | 3 | 0 | 0.15 | governing_quantity: degradation_rate -> fouling_adhesion |
| mechanism_class=architected_lattice, length_scale=10_um, inspiration_origin=plant, governing_quantity=degradation_rate | 0.511 | 38.6 | 1 | 0 | 0.50 | mechanism_class: graded_stiffness -> architected_lattice |
| mechanism_class=electrostatic_field, length_scale=10_um, inspiration_origin=plant, governing_quantity=degradation_rate | 0.511 | 38.6 | 1 | 0 | 0.50 | mechanism_class: graded_stiffness -> electrostatic_field |
| mechanism_class=flow_redirection, length_scale=10_um, inspiration_origin=plant, governing_quantity=degradation_rate | 0.511 | 38.6 | 1 | 0 | 0.50 | mechanism_class: graded_stiffness -> flow_redirection |
| mechanism_class=interfacial_slip, length_scale=10_um, inspiration_origin=plant, governing_quantity=degradation_rate | 0.511 | 38.6 | 1 | 0 | 0.50 | mechanism_class: graded_stiffness -> interfacial_slip |
| mechanism_class=porous_transport, length_scale=10_um, inspiration_origin=plant, governing_quantity=degradation_rate | 0.511 | 38.6 | 1 | 0 | 0.50 | mechanism_class: graded_stiffness -> porous_transport |
| mechanism_class=radiative_control, length_scale=10_um, inspiration_origin=plant, governing_quantity=degradation_rate | 0.511 | 38.6 | 1 | 0 | 0.50 | mechanism_class: graded_stiffness -> radiative_control |
| mechanism_class=architected_lattice, length_scale=nm, inspiration_origin=animal, governing_quantity=stress | 0.505 | 38.0 | 1 | 0 | 0.50 | mechanism_class: trapped_gas_or_liquid -> architected_lattice |

## Strategy yield (all rounds)

| strategy | orders | evaluated | new elite | improved | not better | failed | infeasible | invalid/rejected | missing | off target | elites now |
|---|---|---|---|---|---|---|---|---|---|---|---|
| refine | 3 | 3 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 1 |
| fill_gap | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| combine | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| diversify | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| explore | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |
| seed | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

## Ordinal trends (current)

| axis | mode | held fixed | points (value: score) | slope/step | r2 | next | status |
|---|---|---|---|---|---|---|---|
| length_scale | slice | mechanism_class=graded_stiffness, inspiration_origin=animal, governing_quantity=fouling_adhesion | 10_um: 38.0, 100_um: 42.1 | +4.13 | 1.00 | – | too_few_points |
| length_scale | slice | mechanism_class=graded_stiffness, inspiration_origin=plant, governing_quantity=degradation_rate | 10_um: 38.6, 100_um: 39.9 | +1.32 | 1.00 | – | too_few_points |
| length_scale | marginal | (marginal) | nm: 39.3, sub_um: 32.1, 10_um: 38.6, 100_um: 42.1 | +1.20 | 0.27 | – | poor_fit |

## Trends used for extrapolation

_No extrapolation orders yet._

## Infeasible cells

_None._
