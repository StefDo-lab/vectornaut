# Blind review: facade concepts F1–F6

Baseline: aged white acrylic cool paint (SR about 0.65–0.70) on the same building. Scores run 1–5.

| # | Concept | Novelty | Plausibility | Practicality | Impact vs aged paint | Fit to request | **Overall** |
|---|---|---|---|---|---|---|---|
| F1 | Beetle-scale porous silicate coat | 1 | 4 | 3 | 3 | 3 | **3** |
| F2 | Ventilated Janus louver screen ("camel fur") | 2 | 4 | 3 | 5 | 4 | **4** |
| F3 | Sealed BaSO4/air micro-cells under a glaze | 3 | 3 | 2 | 3 | 4 | **3** |
| F4 | Micro-louver foil inside double glazing (IGU) | 1 | 5 | 4 | 4 | 3 | **3** |
| F5 | UV-paced photo-chalking self-renewing top layer | 2 | 2 | 2 | 2 | 2 | **2** |
| F6 | Silicate super-white above CPVC ("silver ant") | 1 | 4 | 4 | 3 | 4 | **3** |

## Justifications

- **F1**: This is almost exactly the Cyphochilus-inspired porous alumina "cooling ceramic" (Lin et al., *Science* 2023, SR 0.996, emittance 0.965), and BaSO4 and inorganic cooling paints are also well known. The numbers are credible: going from SR 0.70 to 0.85 halves absorptance, which gives about −25…−30 % daily roof gain in a simple sol-air model. But the concept's own main risk, open pores that trap dust, is the decisive one, and the concept does not solve it. The seal or closed top layer is only named as a test variant.
- **F2**: External, ventilated shading that stops the sun before it reaches the glass or wall is the strongest proven lever in hot climates, so the impact is large and real. Brise-soleil and ventilated rainscreens are old prior art. The novel part is the Janus finish: a reflective top with a low-emissivity back, so the heat the louver absorbs is not radiated onto the wall. The heat balance is sound (about 200 W/m² absorbed, h ≈ 25 W/m²K on both faces, ΔT ≈ 8 K, 4–8 W/m² reaching the wall through a low-e back). The claimed SHGC of about 0.10 is too optimistic (see errors). Cost, wind loads and loss of daylight and view are real, but these are known engineering problems.
- **F3**: Moving the scattering volume under a sealed, glass-hard surface is the right answer to the soiling problem. It is the most original idea here, building on radiative-cooling glass (Zhao et al., *Science* 2023) and glazed white tiles. The concept is honest about its limits: low index contrast (BaSO4 n ≈ 1.64 against glass ≈ 1.5) caps SR near 0.85, and silicate glass has a reststrahlen emissivity dip near 9 µm. But the substrate and firing plan contains a hard error, and the path is new panels only, which limits how much of the building stock it can reach.
- **F4**: This is a description of existing products (MicroShade reaches g < 0.10 at visible transmittance > 0.40, and Okasolar is similar), and the concept says so, so novelty is 1. The physics and numbers match those products. Sealed in the IGU, the louvers last as long as the unit. It only works on windows, it is weak against low east and west sun, and the bio-inspiration is loose. Against the paint baseline its gains come on top, because paint does nothing for glass.
- **F5**: The idea of keeping SR by renewing the surface is interesting, but the mechanism does not hold up. A purely mineral matrix does not photo-chalk. Chalking needs an organic binder plus a photoactive pigment such as anatase TiO2, and that pigment absorbs UV and lowers SR. Removing chalk and dirt still needs rain or wind, so UV pacing only sets how fast the binder breaks down, not how fast the surface cleans. That is weakest exactly in arid hot climates. Chalking "self-cleaning" whites and photocatalytic facades (e.g. Pilkington Activ, TX Active) are old prior art.
- **F6**: The chemistry is sound and mature. Silicate paints have more than a century of track record on mineral substrates. BaSO4 is UV-transparent, and the concept sets a realistic aged SR target of 0.85–0.88. Its benefit claims are the most honest of the coatings: −29 % on an uninsulated roof, only about −11 % daily on walls, and it admits the coating does not go below ambient in daytime once aged. Novelty is low (BaSO4 cooling paints, cooling ceramic, commercial silicate cool paints). The open question is dust entering the voids of a film filled above CPVC, which is the same failure as F1.

## Ranking

1. **F2**: largest and most certain cut in heat load, and the only concept that treats glazing and walls together.
2. **F6**: mature, durable coating with honest numbers.
3. **F4**: proven, but already exists and covers windows only.
4. **F3**: most original answer to soiling, but has a fabrication error and needs new panels.
5. **F1**: dominated by F6 and does not solve its own main risk.
6. **F5**: mechanism does not work in dry climates.

## Fund first: F2

F2's effect on indoor heat load is several times larger than any coating's. My corrected estimate still cuts facade solar gain by about 50–60 % against aged white paint, and windows usually dominate facade load in hot climates. It uses aluminium and PVDF parts with 25–40 years of track record, and the proposed test costs under €1,000. The finish needs one change before the test (see below).

As a cheap parallel bet, I would run a single coupon study that merges F6 with F3's "sealed surface" idea. It would test a porous silicate or BaSO4 coat with and without a closed, fired or silicate top layer, and it answers the soiling question behind F1, F3 and F6.

## Errors in the physics claims

- **F2, SHGC ≈ 0.10 and wall gain −70 %**: These ignore light bouncing between the louvers. A diffuse white top (SR 0.8) sends part of the reflected beam toward the building, and the bare or low-e metal underside is also highly reflective to sunlight, so it relays that light inward like a light shelf.
  - A 2D Monte Carlo run (louver_mc.py in this folder) gives beam transmission of 0.29–0.33 for the 1:1 south louvers and 0.19–0.26 for the 2.5:1 east/west louvers.
  - Through a vertical opening, diffuse light passes at about 0.5–0.7.
  - A realistic effective SHGC is therefore about 0.15–0.3, not 0.10.
  - Fix: make the underside solar-absorbing but low-emissivity (a selective or dark low-e finish), or use specular top faces tilted to throw light back to the sky.
- **F2, loss of sky cooling**: The low-e back also blocks the wall's longwave loss to the night sky. This is a small effect, but it is not mentioned.
- **F3, firing fibre-cement**: Fibre-cement cannot be fired at 750–850 °C. The cellulose fibres burn at about 300 °C and portlandite (Ca(OH)₂) decomposes at about 450 °C, which destroys the panel. The route only works on ceramic bodies.
- **F3, SR numbers**: A life-mean SR of 0.84–0.87 contradicts the concept's own statement that clean SR "may top out near 0.85". The life mean cannot be higher than the clean value.
- **F3, BaSO4 in the glaze**: BaSO4 may partly dissolve into a molten glaze at the firing temperature, which would lower the scattering.
- **F3, bio-inspiration**: Birch bark is white mainly because of betulin crystals in the outer bark cells, not because of air-filled cork cells alone.
- **F5, chalking mechanism**: A "mineral" topcoat cannot photo-chalk. The photocatalyst needed for chalking absorbs UV and lowers SR. Renewal still needs rain to carry the chalk away, so the claim that UV pacing is "more robust than rain-paced" does not hold.
- **F6, UV share**: UV is about 4–5 % of global solar irradiance (AM1.5G) and about 3 % of direct beam, not 5–7 %. The effect on the conclusion is minor.
- **F6, acrylic at 70 °C**: Saying acrylic whites "get tacky" is an exaggeration. They soften above their glass transition temperature, which is the real cause of dirt pickup.
- **F6, "neither chalks"**: A silicate film filled above CPVC is under-bound. It can powder, wear by abrasion and take dust into its voids, so "neither chalks" is too strong.
- **F1, fouled case**: The ~−10 % roof benefit at a fouled SR of 0.78 looks slightly pessimistic. A simple sol-air model gives about −15 %. This is within modelling uncertainty and not a real error.
- **F4**: No material errors. The SHGC and visible transmittance are consistent with MicroShade data.

## Sources

- [Beetle-inspired cooling ceramic, CityU / Science 2023 (ScienceDaily)](https://www.sciencedaily.com/releases/2023/11/231110112508.htm)
- [Ceramic and glass radiative cooling coatings (ACerS)](https://ceramics.org/ceramic-tech-today/saving-the-planet-through-passive-cooling-new-ceramic-and-glass-radiative-coatings-offer-stability-and-scalability/)
- [Why paper birches are white: betulin (Northern Woodlands)](https://northernwoodlands.org/articles/article/why-are-paper-birches-so-white)
- [MicroShade performance (REHVA Journal)](https://www.rehva.eu/rehva-journal/chapter/microshadetm-provides-daylight-and-view-out-in-the-new-confederation-of-danish-industry-building-in-copenhagen)
- [MicroShade product page](https://microshade.com/microshade/)
