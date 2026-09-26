# Independent review: facade-cooling candidates (2026-09-27)

Reviewer: fresh Claude instance acting as building-physics / radiative-cooling expert, web search for prior art (no formal patent search), own 1D hourly heat-balance model (`review_facade_hb.py`, `review_facade_hb2.py`). Dossier: `review_facade_dossier.md`.

Key physics: each +0.01 solar reflectance (SR) cuts roof heat gain ~1.7–2.3 % vs an aged white paint (life-mean SR ~0.70); almost no candidate is sub-ambient at noon (these are cool surfaces, broadband ε ≈ 0.9 suffices); particles inside glass or binder (n ≈ 1.5) scatter 35–47× less than in air, so whiteness must come from air voids; raising SR 0.70 → 0.855 equals only ~3–40 mm of mineral wool depending on existing U.

| ID | Concept | Origin run | Novelty/Plaus./Pract./Impact | Verdict |
|---|---|---|---|---|
| A | Sealed glass micro-cells under a glaze (birch periderm) | guided only | 3/3/3/4 | Pursue, re-scoped: fired closed-porosity white glaze/enamel on porcelain or enamelled steel (not fibre-cement; BaSO4 is the wrong scatterer inside glass) |
| B | Cyphochilus-type porous inorganic coat | both | 1/3/3/3 | Known; soiling study only |
| C | Diatomite silicate paint | unguided only | 1/2/3/2 | Merge into B |
| D | Silverfish chaotic multilayer platelets | unguided only | 2/2/2/1 | Drop |
| E | Self-renewing chalking topcoat | both | 2/3/3/3 | Mineral sacrificial top layer over B, rainy climates only |
| F | Snail glazed tile + sealed cavity | both | 1/3/3/2 | Known; ventilate instead of seal |
| G | Cork cladding + white skin | both | 1/4/3/4 | Commercial product; it is insulation |
| H | Cactus-spine pile shade | guided only | 2/3/1/1 | Drop |
| I | Nanovoided glass-fibre shade screen | guided only | 3/3/2/1 | Drop |

Recommended combination: A's fired high-SR closed-porosity glaze on ventilated ceramic rainscreen panels (A + F), plus non-combustible insulation where walls are poorly insulated. First experiment: 4–6 foaming-frit formulations on coupons; SR (ASTM E903/C1549), emittance (ASTM C1371), ASTM D7897 accelerated soiling; go/no-go SR ≥ 0.85 after soiling; a few k€.

Overall: mostly recombination with bio-inspired labels; A is the only plausibly new, valuable element.

## What this says about the explorer

The single concept the reviewer recommends pursuing (A) came only from the guided run; the unguided baseline's unique proposals (C, D) were merged or dropped. Together with the hull trial (where both halves of the recommended combination came from guided runs) this is 2 of 2 tasks in favour of map guidance — still a small sample.
