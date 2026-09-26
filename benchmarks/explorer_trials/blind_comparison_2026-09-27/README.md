# Blind comparison: explorer vs a careful direct answer

Question: does the explorer process find better solutions than a single, carefully reasoned direct answer from the same model?

Setup (both tasks): a fresh Claude instance received the request and was told to think as long as needed, self-critique and deliver its best 3 concepts (no web search, no repo access; `direct_*_best3.md`). The explorer's best 3 are the explorer runs' own final picks (hull: v3 guided run; facade: guided run), written in the same format. The six concepts per task were shuffled and anonymised (`*_blind.md`); a fresh reviewer per task scored them blind (web search allowed). `blind_key.json` maps labels to sources.

## Result

| Task | Reviewer ranking (source) | Winner |
|---|---|---|
| Hull coating | H1 direct > H3 direct > H2 explorer > H5 direct > H4 explorer > H6 explorer | direct |
| Facade cooling | F2 direct > F6 direct > F4 direct > F3 explorer > F1 explorer > F5 explorer | direct |

Novelty scores: the explorer's best concept tied (hull H2 = H3 = 3) or led (facade F3 = 3, highest) on novelty, but lost on practicality and expected impact.

## Interpretation

- Against a naive baseline (15 unguided proposals through the same evaluation), the explorer won on both tasks. Against a careful direct answer it lost on both. The earlier "2:0 for the map" therefore reflected a weak baseline.
- The direct answers reasoned at system level (facade: glazing dominates the heat load → external shading; hull: combine a soft release layer or air lubrication with existing ship systems), while the explorer stayed inside its framing and evaluator: 1D coating models, bio-labelled mechanism axes, and scores driven by proxy metrics and critic ratings.
- The explorer spends its budget on many shallow candidates plus evaluation; the direct answer spent it on deep reasoning about a few.
- Caveats: two tasks only; one reviewer per task; the explorer write-ups were summaries of the runs' outputs; all roles were played by the same model family.

## Implications for the explorer

1. Start each map with a deep system-level analysis (where does the load/loss actually come from?) and let it set the framing and relevant axes, instead of the request's wording alone.
2. Allow candidates outside the literal request scope when the analysis shows the lever is elsewhere (e.g. windows vs walls), flagged as scope extensions.
3. Use the evaluator as a filter against physically wrong ideas rather than as the ranking signal when its models are this simple.
4. Seed the map with a careful direct answer and use the explorer to push novelty around it.

## Three-way blind comparison, facade (9 candidates)

Adds the top 3 of the live Gemini explorer run (`../2026-09-27_facade_gemini_live`). "First experiment" sections were removed from all nine because the Gemini candidates have none. Fresh reviewer; key in `blind9_key.json`; review in `facade9_review.md`.

| Rank | Label | Source | Concept |
|---|---|---|---|
| 1 | C5 | direct (Claude) | ventilated louver screen |
| 2 | C7 | explorer (Claude role-play) | sealed glass micro-cells under a glaze |
| 3 | C1 | direct (Claude) | silicate super-white paint |
| 4 | C8 | direct (Claude) | micro-louver double glazing |
| 5 | C4 | explorer (Claude role-play) | beetle-scale porous silicate coat |
| 6 | C3 | explorer (Gemini live) | sealed closed-pore alumina tile |
| 7 | C9 | explorer (Claude role-play) | UV-paced self-chalking topcoat |
| 8 | C2 | explorer (Gemini live) | Tillandsia calcite-silica lamellar glaze |
| 9 | C6 | explorer (Gemini live) | desert-snail carbonate cladding |

Mean rank: direct 2.7, Claude explorer 4.7, Gemini explorer 7.7. The reviewer flagged C3/C7 as the same idea (sealed scattering voids under a glaze) — found independently by the Claude and Gemini explorer runs. Gemini's two carbonate concepts fail on basic processing chemistry (calcite/aragonite decompose or transform at firing temperatures), which Gemini's own critic did not catch; its critic was far more lenient than the Claude role-play critic.
