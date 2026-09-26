# Blind review: bio-inspired biocide-free hull coatings (H1–H6)

Reviewer basis: only `hull_blind.md`, plus web checks of prior art and quick calculations (`checks.py` in this folder).
Baseline for "impact": a modern biocide-free silicone/amphiphilic foul-release coating (FRC), about 150–250 µm, E ≈ 1 MPa, over several years of service.

## Score table (1–5)

| # | Concept | Novelty | Physical plausibility | Practicality | Multi-year impact vs FRC | Overall |
|---|---------|:-:|:-:|:-:|:-:|:-:|
| H1 | Thick soft "blubber" layer + tough amphiphilic skin | 2 | 4 | 3 | 3 | **4** |
| H2 | Segmented tough tiles on supersoft bed | 3 | 4 | 2 | 2 | **3** |
| H3 | Gas-fed re-entrant streamwise grooves (Salvinia + riblet) on flat bottom | 3 | 3 | 2 | 3 (high variance) | **3.5** |
| H4 | nm phosphocholine hydration-lubrication topcoat for grooming | 2 | 2 | 2 | 1 | **1.5** |
| H5 | Riblet foul-release film on parallel mid-body | 2 | 4 | 3 | 2 | **2.5** |
| H6 | Hull-normal fibres in soft silicone | 2 | 2 | 2 | 1 | **1.5** |

## Justifications

**H1 (overall 4).** The main lever is correct and supported by data. Removal stress of hard fouling scales roughly as √(G_c·E/t) (Kendall; Brady & Singer; Kim et al. 2008, Biofouling 24:4, which found removal stress set by (E/t)^0.5 for t = 210–770 µm; Wendt et al. 2006 measured lower release and better durability at 2 mm). The claimed 5–7× reduction checks out (5.0–5.8×). It saturates at about 4× for small barnacles, because the thin-layer scaling needs t ≪ barnacle radius. Novelty is limited: thickness and modulus are known levers, and the new part is putting a tough skin on top to protect a very thick soft layer. It is the lowest-risk candidate and the most sprayable. It addresses hard fouling during idle periods, which is a real failure mode of FRCs, but does little about slime.

**H2 (overall 3).** The mechanics are sound. A continuous stiff skin stiffens the bed over a length of about t_s·(E_s/E_b)^(1/3), and cutting the skin into tiles of about that size keeps the bed compliant while stopping cracks at the gaps. The approach is related to crack arrest by incisions (Chaudhury and co-workers) and to fish-scale or chiton armour. The benefit claim (0.5–1.5 %) is modest and credible. However, stencil or laser segmentation with 20 µm gaps over 10⁴ m² in a dry dock is not realistic today. The gaps become slime grooves and weak points, so the net multi-year gain over H1 is small.

**H3 (overall 3.5).** This has the highest upside and the highest risk. The physics numbers are internally consistent: Philip slip gives b = 41.7 µm; with Fukagata's spanwise correction ΔU⁺ ≈ 3.9–5.3, which means 18–24 % local drag reduction. The Laplace hold of 2γ/w is about 1.9 kPa. My mass-transfer estimate for gas make-up is about 7.5 kW, and bottom friction is about 3.2 MW. The main parts have prior art. AIRCOAT (EU H2020, 2018–21) built a self-adhesive Salvinia air-retaining FR foil. C.-J. Kim's group measured about 30 % drag reduction with re-entrant longitudinal micro-trenches under a motorboat at sea (Xu et al., PR Applied 2020; Flow 2024). Gas replenishment of plastrons is also known. The new element is using an air-lubrication manifold to feed segmented grooves, with riblet geometry as a fallback when flooded. Multi-year survival of 40 µm overhanging micro-features against surfactants, biofilm bridging in port, abrasion and cleaning is unproven. The fluorosilicone chemistry also carries PFAS regulatory risk in the EU.

**H4 (overall 1.5).** Pairing a coating with grooming is sensible, because slime is the main residual FRC penalty. But in-water grooming is established (Swain, Tribou), and zwitterionic MPC brushes are well-known antifouling and lubricating chemistry. A nm brush cannot survive three-body abrasion by sand or brush contact for years, and the write-up admits this. The coating's own benefit (~0.5 %) is negligible, and the concept is thin.

**H5 (overall 2.5).** The physics is textbook and correct: s⁺ = 14–20 at 12–15 kn, and ΔU⁺ 0.8–1 gives 4–5 % local drag reduction at U_b⁺ ≈ 37. Riblets with foul-release chemistry already exist (e.g. Benschop et al., Biofouling 2018; aviation riblet foils such as AeroSHARK). The authors state the decisive problem themselves: slime and silt fill the valleys, and the at-sea record is poor. Once clean-hull conditions are gone, the 1.5–2.5 % benefit disappears, so the multi-year gain is probably below 1 %. The settlement-deterrence citation (Berntsson et al. 2000, 30–45 µm features) is valid, but it applies to cyprids and not to diatom slime.

**H6 (overall 1.5).** The mechanics are questionable. Fibres oriented normal to the hull greatly stiffen the layer against normal and peel loads, and in thin layers peel release is controlled by exactly that compliance. Such fibres also do not stop a blade slicing between them. Hull-normal fibres are already commercial as exposed flocked nylon (Finsulate), which has a drag penalty. Embedding them removes Finsulate's antifouling mechanism and keeps the drawbacks. The ~0.5–1 % benefit has no basis.

## Ranking

1. **H1** – blubber layer (4)
2. **H3** – gas-fed grooved film (3.5; highest upside, highest risk)
3. **H2** – tiled skin on soft bed (3)
4. **H5** – riblet FR film (2.5)
5. **H4** – hydration-lubricated grooming topcoat (1.5)
6. **H6** – fibre-reinforced soft layer (1.5; ranked below H4 because its core mechanics work against it)

## Fund first: H1

- The physics lever is proven, and the modulus/thickness plaque matrix is cheap and finishes in months.
- It uses sprayable, existing silicone chemistry and needs no new ship systems.
- It addresses a real FRC failure mode, hard fouling during idle periods.
- Its outcome is also the gate for H2: if a continuous skin does not over-stiffen the bed, tiling is unnecessary.

H3 should run a parallel small Taylor–Couette programme. It has the larger prize, but its durability and fouling risks need a separate go/no-go test.

## Errors or overstatements in the physics claims

- **H1, flow force on barnacles.** The claim is that a 3–6 mm barnacle feels 4–8 kPa at 12–15 kn. A log-law estimate (u_τ ≈ 0.19 m/s, local velocity about 3.7–3.9 m/s at barnacle height, drag coefficient ~0.8) gives about 3–4 kPa. This only overlaps the bottom of the claimed 3–20 kPa release range, so hard fouling will come off at service speed only partly, as the most weakly attached fraction.
- **H1, fuel benefit.** 3–8 % averaged over the docking cycle is optimistic because slime is the dominant penalty on FRCs; 1–4 % is more defensible.
- **H1, material cost.** 1.5 mm is about 18 m³ of silicone per 12,000 m², 7–10× the volume of an FRC topcoat. "2–3× material cost" is only possible with a cheap bulk silicone and leaves out multi-coat application labour.
- **H1, surface waves.** "Shear-wave speed 8–9 m/s" is correct (8.0–9.8 m/s). It is a sufficient but conservative criterion; static-divergence onset is typically at U ≈ 2–3 c_t.
- **H2.** "Spreading factor 0.91 vs 0.57" is not derived and cannot be checked. It also writes the release law as √(2w/C) where it should be √(2w/(dC/dA)), a notation slip.
- **H3.** The numbers check out, but the fuel benefit "3–6 % beyond air lubrication" double-counts. Where air lubrication already forms a continuous air layer, the texture adds little. A realistic ideal is closer to 2–4 %.
- **H3, meniscus hold.** "Meniscus holds 2 kPa" assumes perfect pinning at the re-entrant caps. Contamination and wear lower this.
- **H3, pressure fluctuations.** p′ ≈ 0.1 kPa is the rms value; peaks are 3–5× higher, which still leaves margin.
- **H3, "in-port air layer blocks larval settlement."** This requires the trickle feed to be running continuously in port, and biofilm bridging (a risk the write-up itself lists) undermines it.
- **H5.** At 20 kn, s⁺ ≈ 24–26, which is at or beyond the break-even point for trapezoidal riblets. This is worse than "about 22" suggests.
- **H6.** The implied claim that hull-normal fibres keep shear compliance while giving cut resistance does not hold. Fibres stiffen normal and peel compliance, and cuts pass between fibres.

## Sources
- [Kim et al. 2008, Release of reattached barnacles vs (E/t)^0.5, Biofouling 24:4](https://www.tandfonline.com/doi/abs/10.1080/08927010802199945)
- [Wendt et al. 2006, Effect of coating thickness on barnacle critical removal stress, Biofouling 22:1](https://www.tandfonline.com/doi/abs/10.1080/08927010500499563)
- [AIRCOAT project (EU H2020)](https://aircoat.eu/drag-reduction/) and [Busch et al. 2019, Phil. Trans. R. Soc. A](https://royalsocietypublishing.org/rsta/article/377/2138/20180263/40898/Bionics-and-green-technology-in-maritime-shipping)
- [Xu et al. 2020, Superhydrophobic drag reduction in open water, PR Applied](https://link.aps.org/doi/10.1103/PhysRevApplied.13.034056); [motorboat micro-trench study, Flow](https://www.cambridge.org/core/journals/flow/article/drag-reduction-on-microtrench-and-micropost-superhydrophobic-surfaces-underneath-a-motorboat-on-the-sea/653871CBA7E6C3E6577AA38C0E64F4D0)
- [Benschop et al. 2018, Drag-reducing riblets with fouling-release properties, Biofouling](https://www.tandfonline.com/doi/full/10.1080/08927014.2018.1469747)
- [Finsulate flocked-fibre hull wrap](https://www.finsulate.com/en/)
