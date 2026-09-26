# Blind review: facade9 (C1–C9)

Baseline is aged white acrylic paint (SR about 0.65–0.72). My checks use a steady-state surface balance and a daily-mean sol-air model (hot-dry climate, T_air mean 33 °C, T_in 25 °C, h_o 20 W/m²K, sky ΔR 100 W/m² for roofs and 40 W/m² for walls). The script is `chk.py` in this folder.
- Roof, U = 3, α 0.30 to 0.13–0.15: −32…−35 % daily gain.
- West wall, U = 3.4, α 0.30 to 0.13: about −18 %. With α 0.35 to 0.04: about −31 %.
- For a heavy uninsulated wall, most of the daily gain comes from the air–indoor temperature difference. Changing reflectance alone cannot remove more than about a third of it.

Scale: 1 = poor, 5 = excellent.

| # | Concept | Novelty | Plausibility | Practicality | Impact vs aged paint | Fit | **Overall** |
|---|---|---|---|---|---|---|---|
| C1 | Silicate/BaSO₄ super-white paint | 2 | 4 | 4 | 3 | 4 | **3.5** |
| C2 | Tillandsia calcite-silica lamellar glaze | 3 | 1 | 1 | 2 | 3 | **1.5** |
| C3 | Sealed closed-pore alumina tile | 2 | 2 | 2 | 3 | 4 | **2.5** |
| C4 | Beetle-scale porous silicate coat | 1 | 4 | 3 | 3 | 4 | **2.5** |
| C5 | Ventilated Janus louver screen | 2 | 4 | 3 | 5 | 4 | **4** |
| C6 | Desert-snail carbonate cladding | 3 | 1 | 1 | 3 | 3 | **1.5** |
| C7 | Sealed BaSO₄ glass micro-cells under glaze | 3 | 4 | 3 | 3.5 | 4 | **3.5** |
| C8 | Micro-louver IGU (compound eye) | 1 | 4 | 4 | 4 | 3 | **3.5** |
| C9 | UV-paced self-chalking topcoat | 2 | 2 | 2 | 2 | 3 | **2** |

## Per-candidate

**C1. Silicate/BaSO₄ super-white paint (3.5).** The prior art is BaSO₄ radiative-cooling paint (Li/Ruan 2021), commercial silicate paints, and the Cyphochilus and silver-ant literature (Burresi 2014; Shi 2015). The concept is honest and holds together. The decisive lever is that the inorganic binder does not soften and pick up dirt. My model reproduces the benefit numbers: roof about −29…−35 %, wall about −11…−18 %.
Overstated or wrong:
- UV is about 3–5 % of AM1.5 below 400 nm, not 5–7 %.
- An aged SR of 0.85–0.88 is optimistic for a film above the CPVC. Open porosity is itself a dirt trap, the same risk C4 names, and desert dust is only partly acknowledged.
- A 300–450 µm silicate film cracks easily (the authors acknowledge this).

**C2. Tillandsia calcite-silica lamellar glaze (1.5).** Calcite decarbonates at about 700–900 °C and reacts with silica to form CaO/wollastonite, so calcite platelets cannot survive vitrification. Interference in a 400/300 nm lamellar stack is spectrally narrow, not broadband.
Overstated or wrong:
- Tillandsia trichomes are dead cellulose cell shields, not mineralised.
- Silica (about 9 µm) and calcite (about 7 and 11.4 µm) both have reststrahlen bands, so ε = 0.96 across 8–13 µm is doubtful.
- The 33.6 vs 48.2 °C result is reproducible only for a horizontal surface under a dry sky (my calc: 33.9 vs 48.0 °C).
- On a vertical facade that sees half hot ground, the surface sits at about 40 °C, so "below ambient under peak sun" is false for the stated use.
- "−38 %" is a percentage of a °C value, which is meaningless.

**C3. Sealed closed-pore alumina tile (2.5).** This is nearly the published Cyphochilus-inspired cooling ceramic (Lin et al., Science 2023: porous alumina, SR 0.996, ε 0.965) with a glaze added. The glaze is a sensible idea for keeping reflectance under soiling.
Overstated or wrong:
- 58 % *closed* porosity at 280 nm is not attainable by sintering. Pores stay interconnected until porosity falls below roughly 10 %.
- The claimed wall-flux cut of 32 to 6.5 W/m² (−80 %) is too large. My sol-air model gives 31.5 to 21.9 W/m² (−31 %), because the air–indoor ΔT dominates.
- A dense silicate glaze has a 9 µm emissivity dip.

**C4. Beetle-scale porous silicate coat (2.5).** This is essentially C1 with no specification: no fillers, thickness or binder chemistry. Its benefit ranges are realistic and it names the correct main risk, fouling of open pores.
Overstated or wrong: nothing major. It is too thin to evaluate or fund as it stands.

**C5. Ventilated Janus louver screen (4).** External shading of windows is the largest lever in hot climates. The prior art is brise-soleil and ventilated rainscreens, and the white-top / low-e-underside split is a sound refinement.
- The geometry checks out: d:p = 1 gives a 45° cut-off, and 2.5:1 gives about 22°, lower with tilt.
- The underside radiative transfer checks out: 4.7–9.5 W/m² at ΔT 7 K and ε_eff 0.1–0.2.

Overstated or wrong:
- Louver ΔT of 5–9 K needs h ≈ 25–30 W/m²K. In calm air it is 15–30 K.
- Warm-air convection in the 100–200 mm cavity onto the wall is ignored.
- A daily effective SHGC of 0.10 is optimistic. Diffuse sky and ground light pass through, and the white tops inter-reflect off the specular undersides toward the glass. About 0.15–0.25 is more realistic.
- Lost daylight adds lighting load.
- A PVDF white SR of 0.8 is at the top of the range. 0.65–0.75 is typical.
- The camel-fur facts are correct (Schmidt-Nielsen).

**C6. Desert-snail carbonate cladding (1.5).** The biology is right: the Sphincterochila shell reflects about 90–95 % (Schmidt-Nielsen 1971). The material route fails.
Overstated or wrong:
- Aragonite converts to calcite at about 400–500 °C and CaCO₃ decomposes at about 800 °C, so "sintered aragonite/calcite lamellae" cannot be made conventionally.
- Thin calcite (marble) cladding is a known failure. Anisotropic thermal expansion causes hysteresis bowing and strength loss (Finlandia Hall, Amoco/Aon tower). Acid rain dissolves carbonate. Twenty years is not credible.
- "Alumina" appears in the emittance claim without being part of the material.
- The noon surface temperatures assume a horizontal surface at 900 W/m², not a cladding plate.

**C7. Sealed BaSO₄ glass micro-cells under a glaze (3.5).** It targets the real lever, which is reflectance retained over the building's life, not clean SR. It admits the index-contrast ceiling (SR ≈ 0.85) and the glaze emissivity dip. My model supports its roof figure of −25 % daily (−32 %).
Overstated or wrong:
- **Fibre cement cannot be fired at 750–850 °C.** Its cement hydrates and cellulose fibres decompose far below that, so the concept is limited to ceramic substrates.
- BaSO₄ can partly dissolve in or react with a silicate melt (sulphate fining), which lowers scattering.
- Birch whiteness comes mainly from betulin crystals in the phellem, not from air cells alone.
- The prior art is white glazed cool tiles and radiative-cooling glass coatings (Zhao et al., Science 2023).

**C8. Micro-louver IGU, compound eye (3.5).** This is MicroShade (Photosolar), which the authors credit. It adds little new, but its numbers are internally consistent: SHGC 0.12/0.27 matches 78/176 W/m² and 0.48/1.08 kWh. They also match published MicroShade data (LT about 0.45, low g at high sun).
Overstated or wrong:
- The bionic analogy is decorative.
- Bright louver tops are atypical because commercial versions are dark. They reduce cavity heating but cause upward glare.
- It covers windows only, so the building-level benefit scales with window-to-wall ratio. It meets the request mainly as a component.

**C9. UV-paced self-chalking topcoat (2).** Self-chalking paints and photocatalytic TiO₂ self-cleaning are decades old.
Overstated or wrong:
- A truly mineral binder does not photo-erode. UV-paced erosion needs a photocatalyst, usually anatase, plus an organic binder, and the photocatalyst absorbs UV and lowers SR.
- The chalk and dirt still need rain or wind to be removed, so it is not "more robust than rain-paced."
- The −15 % figure is unsupported.
- The 20-year erosion reserve is unquantified.

## Ranking
1. C5
2. C7
3. C1
4. C8
5. C4
6. C3
7. C9
8. C2
9. C6

C7, C1 and C8 are close. C2 and C6 are tied at the bottom.

## Fund first
**C5.** It is the only concept that attacks window solar gain, the dominant facade load in hot climates. Its physics is sound and it carries almost no materials risk. It is also the only candidate plausibly giving well over 30 % facade-load reduction versus aged paint. Fund it as a monitored demonstrator: measure the effective daily SHGC, cavity air temperatures and the daylight penalty.

If the funder wants a new coating material instead, fund C7, restricted to ceramic substrates, or C1.

## Near-duplicates
- **C1 ≈ C4:** porous inorganic silicate super-white paint. C4 is an underspecified version of C1.
- **C3 ≈ C7:** sealed optical voids under a dense glaze to keep reflectance under soiling. They differ only in pore scale and filler.
- **C2 ≈ C6:** sintered sub-micron carbonate lamellae. Both share the same fatal processing flaw.
- **C5 ~ C8:** related but distinct. Both are angle-selective louvers, external screen versus inside the IGU.

## Sources
- Lin et al., Science 382, 691 (2023): https://www.science.org/doi/10.1126/science.adi4725
- MicroShade: https://microshade.com/microshade/
- LBNL angular-selective windows: https://www.osti.gov/servlets/purl/1248925
