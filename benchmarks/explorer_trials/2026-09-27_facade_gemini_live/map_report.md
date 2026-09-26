# Explorer map: materials

Query: "Entwickle eine bionisch inspirierte, passive Beschichtung oder Fassadenstruktur für Gebäude in heißen Klimazonen, die an Sommertagen die Wärmelast im Innenraum deutlich senkt, ohne Strom oder Wasserzufuhr auskommt und mindestens 20 Jahre wetterbeständig bleibt."  
Archive: `/tmp/claude-0/-home-user-vectornaut/6d94a234-0063-56c5-b9d5-7a9e3882a427/scratchpad/gemini_facade2/explorer/materials/archive.json`  
Rounds so far: 5

> score = 100 * gate * (0.45 * objective_score + 0.1 * simulated_score + 0.45 * requirement_score) * relabel_factor * baseline_factor * must_factor; gate = validator_score * (1 pass / 0.8 warn / 0 fail); objective_score = (1 - exp(-objective_gain_pct / 18)) * d with the objective scale 18 % (target), objective_gain_pct = critic's plausible objective gain minus its conventional-equivalent gain (floored at 0), d = 1 if a critic judged the objective next to a relevant simulation, else 0.5; simulated_score = 1 - exp(-simulated_benefit_pct / 20) if the simulated quantity is relevant, else 0, simulated_benefit_pct = min(simulated, critic) when the critic gave a number, 0 for a proxy by construction; requirement_score = 0.5 * mean + 0.5 * min over must requirements; must_factor = 1 - 0.5 * max(0, (0.3 - min must) / 0.3); baseline_factor = 0.8 if the baseline is not the conventional one; relabel_factor = 0.5 for a relabelled analogue, else 1; estimated tier capped at 50. Objective scale 18 % (target: stated target gain 35 % x 0.5 (reaching the target scores 0.86)). Tier 'simulated' only when a usable simulated gain on a relevant governing quantity enters the score (never above the critic's plausible simulated benefit; not at all for a proxy by construction); otherwise tier 'estimated' (the simulated number is shown but not scored). The objective gain is the critic's plausible contribution to the stated objective against the stated baseline, minus what a conventional measure with the same physical effect would give. 'points obj / sim / req' splits the score into its three parts (before the cap). The scale is recomputed after every round and all entries are then re-scored, so scores are comparable within this map. Elites are ranked by tier first; within a cell, scores within 1 point are a tie decided by the critic's objective gain, then fewer killer risks.

## Coverage

- Cells in the grid: 4900 (10 x 7 x 7 x 10)
- Cells with an elite: 11 (0.22 %)
- Cells with at least one proposal: 11
- Infeasible cells/patterns: 0
- Entries: 15 (evaluated: 15)
- Elite score range: 17.9 – 86.7
- Elites by evidence tier: simulated: 11
- Flags on evaluated entries: model_assumption_sensitive: 4, mostly_conventional_effect: 14, proxy_by_construction: 1, relabelled_analogue: 2

## Requirements of the request

Objective and conventional baseline from the function analysis (round 1); fixed for this map.

- **Objective**: Time-averaged reduction in inward conductive peak heat flux through the building envelope over a 20-year service life under hot-climate summer insolation.
- **Baseline**: Conventional high-albedo exterior acrylic facade paint (initial solar reflectance 0.80, thermal emittance 0.88) aged under 10-20 years equivalent outdoor weathering.
- **Target gain**: 35 % (stated by the function analysis, round 1)
- **Objective scale**: 18 % (target: stated target gain 35 % x 0.5 (reaching the target scores 0.86)); a gain of this size scores 0.63 of the objective part. History: 2 % (default, round –), 18 % (target, round 1)

Extracted by the generator's function analysis (round 1); fixed for this map.

- **passive_operation** [must]: Functions entirely without electrical power, moving mechanical pumps, or water/fluid feed.
- **twenty_year_durability** [must]: Maintains physical integrity and thermo-optical properties under exterior UV and weathering for at least 20 years.
- **bionic_inspiration** [must]: Employs an operational principle or micro/macrostructure directly abstracted from a biological system.
- **substantial_heat_load_reduction** [must]: Significantly reduces peak conductive heat influx through the building envelope compared to conventional standard white facades.
- **building_facade_compatibility** [nice]: Can be applied directly as an exterior coating or mounted as an external facade cladding panel.

Relevant `governing_quantity` values (explore/fill-gap/diversify use only these): heat_flux, temperature, degradation_rate (function analysis)

Relevant `mechanism_class` values (fill-gap/diversify use only these, explore reaches the others at a low weight): porous_transport, architected_lattice, radiative_control (function analysis)

Preferred `inspiration_origin` values (fill-gap/diversify/combine/extrapolate use only these; explore reaches the others at a low weight): plant, animal, microbe (the request asks for a biological model ('bionisch'))

## Elites (best per cell)

| # | score | tier | points obj / sim / req | objective gain used % | critic obj. % | conv. equivalent % | sim. benefit used % | simulated % | simulated quantity | req. coverage | flags | title | cell | strategy | round | entry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 86.7 | simulated | 38.6 / 8.5 / 39.6 | 35.0 | 35.0 | 0.0 | 38.0 | 58.2 | temperature | 0.91 | – | Tillandsia-Trichome-Mimetic Sintered Calcite-Silica Lamellar Glaze | mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=plant, governing_quantity=temperature | combine | 5 | e00014 |
| 2 | 81.5 | simulated | 32.5 / 9.3 / 39.8 | 23.0 | 48.0 | 25.0 | 52.0 | 61.5 | heat_flux | 0.92 | mostly_conventional_effect | Hermetically Sealed Closed-Cell Cyphochilus-Mimetic Sintered Alumina Tile | mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=animal, governing_quantity=heat_flux | refine | 2 | e00006 |
| 3 | 65.9 | simulated | 19.2 / 8.3 / 38.5 | 10.0 | 32.0 | 22.0 | 35.0 | 78.1 | temperature | 0.89 | model_assumption_sensitive, mostly_conventional_effect | Sphincterochila-Mimetic Calcitic Lamellar Sintered Radiative Cladding | mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=animal, governing_quantity=temperature | fill_gap | 3 | e00007 |
| 4 | 57.1 | simulated | 9.0 / 8.3 / 39.8 | 4.0 | 32.0 | 28.0 | 35.0 | 37.8 | degradation_rate | 0.92 | mostly_conventional_effect | Cotyledon-Mimetic Shingled Zinc-Aluminate Spinel Vitrified Radiative Glaze | mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=plant, governing_quantity=degradation_rate | refine | 4 | e00012 |
| 5 | 51.8 | simulated | 10.9 / 7.8 / 33.1 | 5.0 | 25.0 | 20.0 | 30.0 | 54.7 | heat_flux | 0.82 | mostly_conventional_effect | Coscinodiscus-Frustule-Templated Sintered Silica Radiative Facade Panel | mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=microbe, governing_quantity=heat_flux | diversify | 3 | e00008 |
| 6 | 49.8 | simulated | 9.0 / 7.8 / 33.1 | 4.0 | 32.0 | 28.0 | 30.0 | 33.3 | heat_flux | 0.82 | mostly_conventional_effect | Macro-Porous Macro-Termitary Bioclimatic Ventilated Facade Panel | mechanism_class=architected_lattice, length_scale=cm_plus, inspiration_origin=animal, governing_quantity=heat_flux | combine | 2 | e00005 |
| 7 | 37.1 | simulated | 4.7 / 7.1 / 25.2 | 2.0 | 20.0 | 18.0 | 25.0 | 60.3 | heat_flux | 0.72 | model_assumption_sensitive, mostly_conventional_effect | Antheraea-Cocoon-Mimetic Sintered Mullite Microfiber Radiative Veil | mechanism_class=radiative_control, length_scale=10_um, inspiration_origin=animal, governing_quantity=heat_flux | diversify | 5 | e00013 |
| 8 | 33.1 | simulated | 0.0 / 0.0 / 33.1 | 0.0 | 18.0 | 25.0 | – | 36.0 | temperature | 0.82 | proxy_by_construction, mostly_conventional_effect | Sphincterochila-Epiphragm-Mimetic Sub-Micron Knudsen Gas Transport Cladding | mechanism_class=porous_transport, length_scale=sub_um, inspiration_origin=animal, governing_quantity=temperature | diversify | 4 | e00011 |
| 9 | 30.2 | simulated | 2.4 / 2.3 / 25.4 | 1.0 | 5.0 | 4.0 | 5.3 | 5.3 | heat_flux | 0.78 | mostly_conventional_effect | Moloch-Mimetic Sub-Micron Mesoporous Aluminosilicate Dew-Evaporative Tile | mechanism_class=porous_transport, length_scale=sub_um, inspiration_origin=animal, governing_quantity=heat_flux | fill_gap | 4 | e00010 |
| 10 | 28.3 | simulated | 9.6 / 4.1 / 14.6 | 10.0 | 32.0 | 22.0 | 35.0 | 49.4 | degradation_rate | 0.80 | mostly_conventional_effect, relabelled_analogue | Stenocara-Structured Vitrified Silica Glaze for Soot-Resistant Radiative Cooling | mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=animal, governing_quantity=degradation_rate | fill_gap | 2 | e00004 |
| 11 | 17.9 | simulated | 0.0 / 0.0 / 17.9 | 0.0 | 5.0 | 35.0 | -173.0 | -173.0 | heat_flux | 0.74 | model_assumption_sensitive, mostly_conventional_effect | Cactus-Ribbed Self-Shading Convective Ventilated Facade Shroud | mechanism_class=architected_lattice, length_scale=cm_plus, inspiration_origin=plant, governing_quantity=heat_flux | seed | 1 | e00003 |

## Requirement coverage (critic ratings, elites)

| entry | title | passive_operation [must] | twenty_year_durability [must] | bionic_inspiration [must] | substantial_heat_load_reduction [must] | building_facade_compatibility [nice] | mean | min must | req. score | must factor |
|---|---|---|---|---|---|---|---|---|---|---|
| e00014 | Tillandsia-Trichome-Mimetic Sintered Calcite-Silica Lamellar Glaze | 1.00 | 0.85 | 0.90 | 0.85 | 0.95 | 0.91 | 0.85 | 0.88 | 1.00 |
| e00006 | Hermetically Sealed Closed-Cell Cyphochilus-Mimetic Sintered Alumina Tile | 1.00 | 0.85 | 0.95 | 0.95 | 0.85 | 0.92 | 0.85 | 0.89 | 1.00 |
| e00007 | Sphincterochila-Mimetic Calcitic Lamellar Sintered Radiative Cladding | 1.00 | 0.82 | 0.88 | 0.85 | 0.90 | 0.89 | 0.82 | 0.85 | 1.00 |
| e00012 | Cotyledon-Mimetic Shingled Zinc-Aluminate Spinel Vitrified Radiative Glaze | 1.00 | 0.95 | 0.85 | 0.85 | 0.95 | 0.92 | 0.85 | 0.89 | 1.00 |
| e00008 | Coscinodiscus-Frustule-Templated Sintered Silica Radiative Facade Panel | 1.00 | 0.65 | 0.85 | 0.80 | 0.80 | 0.82 | 0.65 | 0.73 | 1.00 |
| e00005 | Macro-Porous Macro-Termitary Bioclimatic Ventilated Facade Panel | 1.00 | 0.65 | 0.90 | 0.80 | 0.75 | 0.82 | 0.65 | 0.73 | 1.00 |
| e00013 | Antheraea-Cocoon-Mimetic Sintered Mullite Microfiber Radiative Veil | 1.00 | 0.40 | 0.85 | 0.65 | 0.70 | 0.72 | 0.40 | 0.56 | 1.00 |
| e00011 | Sphincterochila-Epiphragm-Mimetic Sub-Micron Knudsen Gas Transport Cladding | 1.00 | 0.75 | 0.80 | 0.65 | 0.90 | 0.82 | 0.65 | 0.73 | 1.00 |
| e00010 | Moloch-Mimetic Sub-Micron Mesoporous Aluminosilicate Dew-Evaporative Tile | 1.00 | 0.80 | 0.90 | 0.35 | 0.85 | 0.78 | 0.35 | 0.56 | 1.00 |
| e00004 | Stenocara-Structured Vitrified Silica Glaze for Soot-Resistant Radiative Cooling | 1.00 | 0.85 | 0.50 | 0.85 | 0.80 | 0.80 | 0.50 | 0.65 | 1.00 |
| e00003 | Cactus-Ribbed Self-Shading Convective Ventilated Facade Shroud | 1.00 | 0.80 | 0.85 | 0.30 | 0.75 | 0.74 | 0.30 | 0.52 | 1.00 |

## Map: mechanism_class x length_scale

Rows: `mechanism_class`, columns: `length_scale`. Each cell: best elite score over all other axes (number of proposals that landed there). `·` = no elite yet, `x` = contains cells reported infeasible.

| mechanism_class \ length_scale | nm | sub_um | um | 10_um | 100_um | mm | cm_plus |
|---|---|---|---|---|---|---|---|
| interfacial_slip | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| flow_redirection | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| trapped_gas_or_liquid | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| porous_transport | · (0) | 33.1 (2) | · (0) | · (0) | · (0) | · (0) | · (0) |
| graded_stiffness | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| architected_lattice | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | 49.8 (2) |
| radiative_control | · (0) | 86.7 (10) | · (0) | 37.1 (1) | · (0) | · (0) | · (0) |
| phase_change | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| electrostatic_field | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |
| other | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) | · (0) |

## Where ideas are proposed vs where they work

**mechanism_class** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| interfacial_slip | 0 | 0 % | 0 | – | – |
| flow_redirection | 0 | 0 % | 0 | – | – |
| trapped_gas_or_liquid | 0 | 0 % | 0 | – | – |
| porous_transport | 2 | 13 % | 2 | 33.1 | 31.6 |
| graded_stiffness | 0 | 0 % | 0 | – | – |
| architected_lattice | 2 | 13 % | 2 | 49.8 | 33.8 |
| radiative_control | 11 | 73 % | 7 | 86.7 | 58.3 |
| phase_change | 0 | 0 % | 0 | – | – |
| electrostatic_field | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

**length_scale** (ordinal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| nm | 0 | 0 % | 0 | – | – |
| sub_um | 12 | 80 % | 8 | 86.7 | 54.3 |
| um | 0 | 0 % | 0 | – | – |
| 10_um | 1 | 7 % | 1 | 37.1 | 37.1 |
| 100_um | 0 | 0 % | 0 | – | – |
| mm | 0 | 0 % | 0 | – | – |
| cm_plus | 2 | 13 % | 2 | 49.8 | 33.8 |

**inspiration_origin** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| plant | 4 | 27 % | 3 | 86.7 | 53.9 |
| animal | 9 | 60 % | 7 | 81.5 | 46.6 |
| microbe | 2 | 13 % | 1 | 51.8 | 51.8 |
| geology | 0 | 0 % | 0 | – | – |
| atmosphere_ocean | 0 | 0 % | 0 | – | – |
| technology | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

**governing_quantity** (nominal)

| value | proposals | share | elites | best | mean elite |
|---|---|---|---|---|---|
| wall_shear | 0 | 0 % | 0 | – | – |
| flow_rate | 0 | 0 % | 0 | – | – |
| heat_flux | 9 | 60 % | 6 | 81.5 | 44.7 |
| temperature | 3 | 20 % | 3 | 86.7 | 61.9 |
| deflection | 0 | 0 % | 0 | – | – |
| stress | 0 | 0 % | 0 | – | – |
| fouling_adhesion | 0 | 0 % | 0 | – | – |
| degradation_rate | 3 | 20 % | 2 | 57.1 | 42.7 |
| field_strength | 0 | 0 % | 0 | – | – |
| other | 0 | 0 % | 0 | – | – |

## Values never proposed

- `mechanism_class` (top value holds 73 % of proposals): every relevant value proposed (not relevant to the request, only explore, at a low weight: interfacial_slip, flow_redirection, trapped_gas_or_liquid, graded_stiffness, phase_change, electrostatic_field, other)
- `length_scale` (top value holds 80 % of proposals): nm, um, 100_um, mm
- `inspiration_origin` (top value holds 60 % of proposals): geology, atmosphere_ocean, technology, other
- `governing_quantity` (top value holds 60 % of proposals): every relevant value proposed (not relevant to the request, not searched: wall_shear, flow_rate, deflection, stress, fouling_adhesion, field_strength, other)

## Promising under-explored cells (next fill-gap targets)

| empty cell | priority | best neighbour | elite neighbours | proposals | under-explored | step from source (rotation-adjusted) |
|---|---|---|---|---|---|---|
| mechanism_class=radiative_control, length_scale=nm, inspiration_origin=plant, governing_quantity=temperature | 1.067 | 86.7 | 1 | 0 | 0.80 | length_scale: sub_um -> nm |
| mechanism_class=radiative_control, length_scale=um, inspiration_origin=plant, governing_quantity=temperature | 1.067 | 86.7 | 1 | 0 | 0.80 | length_scale: sub_um -> um |
| mechanism_class=porous_transport, length_scale=sub_um, inspiration_origin=plant, governing_quantity=temperature | 0.978 | 86.7 | 2 | 0 | 0.24 | mechanism_class: radiative_control -> porous_transport |
| mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=microbe, governing_quantity=temperature | 0.967 | 86.7 | 3 | 0 | 0.20 | inspiration_origin: plant -> microbe |
| mechanism_class=radiative_control, length_scale=sub_um, inspiration_origin=plant, governing_quantity=heat_flux | 0.947 | 86.7 | 4 | 0 | 0.12 | governing_quantity: temperature -> heat_flux |
| mechanism_class=architected_lattice, length_scale=sub_um, inspiration_origin=plant, governing_quantity=temperature | 0.928 | 86.7 | 1 | 0 | 0.24 | mechanism_class: radiative_control -> architected_lattice |
| mechanism_class=radiative_control, length_scale=um, inspiration_origin=animal, governing_quantity=heat_flux | 0.904 | 81.5 | 2 | 0 | 0.80 | length_scale: sub_um -> um |
| mechanism_class=radiative_control, length_scale=nm, inspiration_origin=animal, governing_quantity=heat_flux | 0.854 | 81.5 | 1 | 0 | 0.80 | length_scale: sub_um -> nm |

## Strategy yield (all rounds)

| strategy | orders | evaluated | new elite | improved | not better | failed | infeasible | invalid/rejected | missing | off target | elites now |
|---|---|---|---|---|---|---|---|---|---|---|---|
| refine | 3 | 3 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 2 |
| fill_gap | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| combine | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| diversify | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| seed | 3 | 3 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

## Ordinal trends (current)

| axis | mode | held fixed | points (value: score) | slope/step | r2 | next | status |
|---|---|---|---|---|---|---|---|
| length_scale | slice | mechanism_class=radiative_control, inspiration_origin=animal, governing_quantity=heat_flux | sub_um: 81.5, 10_um: 37.1 | -22.24 | 1.00 | – | too_few_points |
| length_scale | marginal | (marginal) | sub_um: 86.7, 10_um: 37.1, cm_plus: 49.8 | -6.45 | 0.40 | – | poor_fit |

## Trends used for extrapolation

_No extrapolation orders yet._

## Infeasible cells

_None._
