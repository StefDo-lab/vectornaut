# Blind comparison: hull

Request: "Develop a bio-inspired coating for ship hulls that lowers frictional drag in water, works without toxic antifouling agents, and lasts several years in seawater."

Six candidate concepts, in random order. Judge each on substance, not on writing style or length.

## Candidate H1

"Blubber" coating: thick, soft layer that sheds fouling

**What:** 1–2 mm of soft, low-damping silicone (E ≈ 0.2–0.3 MPa) over the existing anticorrosive system and tie coat. On top, a 50–75 µm tough amphiphilic PEG/fluoro-silicone skin (E ≈ 1–2 MPa). Airless spray. Fender and tug zones keep a conventional coating.

**Inspiration:** Cetacean skin: a thin, smooth outer layer over thick, compliant tissue.

**Physics:**
- Barnacle release stress scales as √(G_c·E/t) (Kendall/Brady). Compared with a 200 µm, 1 MPa FRC it falls 5–7 times, from about 20–100 kPa to about 3–20 kPa.
- The flow at 12–15 kn pulls on a 3–6 mm barnacle with about 4–8 kPa, so hard fouling starts coming off at service speed, not only above about 20 kn.
- The skin bends over about 0.15 mm, much less than the barnacle base, so it does not stiffen the joint.
- Shear-wave speed ≈ 8–9 m/s is above ship speed, so there is no risk of drag-raising surface waves.

**Benefit versus FRC:** about 3–8 % fuel averaged over the docking cycle for slow-steaming or idle-prone ships. Slime is only partly addressed.

**Risks:** Cuts and dents. Barnacles adapting their bases. About 2–3 times the material cost. Tie-coat peeling.

**First experiment:** A matrix of plaques (E 0.1–1 MPa, t 0.2–2 mm) on a static raft for 3 months, then ASTM D5618-style shear removal of real barnacles. Follow with a 12 kn rotating drum to test whether fouling releases on its own.

## Candidate H2

**Tiled tough skin on an oil-free supersoft bed**
- What: 30–50 µm filled-silicone or silicone-polyurea tiles, 0.3–0.5 mm across with 20 µm gaps, on a 300–400 µm bottle-brush or low-sol PDMS bed (E ≈ 0.1 MPa), over tie coat and epoxy. Made by stencil spraying or laser/roller segmentation.
- Bionic inspiration: pilot-whale soft skin / pack-ice plates; fish-scale and chiton flexible armour.
- Why it could work: barnacle release stress scales as √(2w/C); the tiles keep the soft bed's compliance (spreading factor ~0.91 vs 0.57 for a continuous skin) while resisting cuts and arresting cracks at the gaps. Estimated 50–70 % lower barnacle release stress.
- Realistic benefit: ~0.5–1.5 % time-averaged friction vs a standard foul-release coating, mainly for ships with idle periods.
- Main risks: gaps as weak spots and slime grooves; tear strength of the bed; cost and process.
- First experiment: pseudo-barnacle and live-barnacle removal (ASTM D5618) on uniform FR vs continuous skin/bed vs tiles of 0.25/0.5/1/2 mm, plus knife-cut and fender-abrasion damage tests.

## Candidate H3

Shark–Salvinia grooved film holding an air layer (flat bottom)

**What:** A 0.3–0.5 mm adhesive-backed fluorosilicone-urea film. Streamwise ridges at 90 µm pitch, 12–15 µm wide and 40 µm tall, with 3–5 µm overhanging caps; gas fraction about 0.85. Cast roll-to-roll; soft elastomer can be pulled out of an undercut mould. Laminated in dry dock onto the flat bottom, aligned to CFD streamlines, grooves blocked off every 0.2–1 m. Air comes from a bow air-lubrication manifold, plus a trickle in port.

**Inspiration:** Salvinia and backswimmer (Notonecta) air-holding hairs, shark riblets, penguins releasing air from their plumage.

**Physics:**
- Groove slip length b ≈ (L/π)·ln sec(πφ/2) ≈ 42 µm (b⁺ ≈ 6), giving ΔU⁺ ≈ 4: an ideal local friction reduction of about 18–22 %.
- The meniscus holds about 2 kPa, against pressure fluctuations p′ ≈ 0.1 kPa.
- Replacing dissolved gas takes about 7 kW of compressor power for 12,000 m². The bottom's friction costs about 3.8 MW.
- If the grooves flood, they still act as riblets (s⁺ ≈ 13–18).

**Benefit versus FRC:** about 3–6 % fuel beyond air lubrication alone. The in-port air layer also blocks larval settlement, where FRCs fail. If the grooves flood, about 1–2 %.

**Risks:** Seawater surfactants stiffening the air–water interface and killing the slip. Biofilm bridging the ridges when the ship lies idle. Abrasion. Pressure gradients near the hull ends. Flat-bottomed ships only.

**First experiment:** A Taylor–Couette cell at 2 bar with natural seawater and u_τ ≈ 0.2 m/s, holding gas-fed film coupons. Measure torque and air-layer lifetime (optically), compared with the wetted and smooth film.

## Candidate H4

**Hydration-lubricated groomable topcoat**
- What: a nm-thin phosphocholine (zwitterionic) brush topcoat on a foul-release coating.
- Bionic inspiration: articular cartilage hydration lubrication.
- Why it could work: low-friction, non-damaging contact makes frequent gentle in-water grooming possible, which is the lever against slime (5–10 % penalty); the coating alone gives ~0.5 %.
- Main risks: the nm layer is worn away by the contacts it protects.
- First experiment: tribometer with nylon brushes in sandy artificial seawater for 1000 cycles, tracking friction and layer retention (XPS), then diatom removal by brushing.

## Candidate H5

Shark-skin riblet foul-release film (whole hull, fully passive)

**What:** Trapezoidal riblets (s = 90–100 µm, h ≈ 0.5·s) in an amphiphilic silicone foul-release film, made on the same line as concept 1. Laminated onto the parallel mid-body (about 70 % of wetted area) along CFD streamlines. The curved ends get sprayed FRC.

**Inspiration:** Shark placoid scales. The ridge size also matches textures that deter barnacle larvae.

**Physics:**
- s⁺ ≈ 12–17 at 12–15 kn (about 22 at 20 kn), inside the riblet-benefit range.
- Lab riblets give ΔU⁺ ≈ 0.8–1, which at ship Reynolds number (U⁺_b ≈ 37) is about 4–5 % less local friction.
- Lab studies found fewer barnacle settlements on ridges about 30–45 µm high.

**Benefit versus FRC:** about 1.5–2.5 % fuel after allowing for misalignment, coverage and the spread of s⁺, but only while clean. Best for fast, busy ships.

**Risks:** Diatom slime filling the low-shear valleys. Seams peeling. Damage from cleaning. A poor track record at sea, mostly because of fouling.

**First experiment:** Riblet and smooth coupons of the same chemistry on a rotating disk, run through a speed/idle cycle in harbour water for 3 months. Measure slime cover and the torque difference every month.

## Candidate H6

**Palm-stem-like fibre-reinforced soft layer**
- What: short fibres oriented normal to the hull embedded in a soft silicone layer.
- Bionic inspiration: palm stem vascular bundles.
- Why it could work: shear-compliant (good for release) but tougher against sharp contacts and indentation — a way around the "soft = fragile" trade-off.
- Realistic benefit: ~0.5–1 %.
- Main risks: fibre tips breaking through the skin; debonding.
- First experiment: cut and indentation tests plus pseudo-barnacle removal vs fibre angle and fraction.
