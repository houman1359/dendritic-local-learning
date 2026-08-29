# Workshop deck — 26-slide map and cut plans

Canonical workshop deck: `dendritic_credit_workshop_25.pdf` — 26 core
slides, no act dividers, followed by a Backup divider and A1–A7. The original
42-page workshop deck is preserved as an expanded technical source.

## Core sequence (29–30 minutes)

| Slides | Arc | Target |
|---:|---|---:|
| 1–6 | credit assignment → exact BP → biological locality → point-neuron bandwidth | 5:30 |
| 7–12 | dendritic state → conductance/shunting → eligibility × transported error → resolution ladder | 6:45 |
| 13–14 | stochastic credit operator → alignment × bandwidth phase boundary | 2:50 |
| 15–19 | neuron identity/ownership → credit-conflict task and boundary → ancestry budget → physical depth | 5:25 |
| 20–24 | sparse arbors → conditional shunting → measured null → alignment rescue → animal coordinates | 5:50 |
| 25–26 | populated phase plane → resource ledger, predictions, close | 3:05 |

Frame index: 1 title · 2 credit assignment · 3 exact BP · 4 neuronal gap ·
5 local-learning taxonomy · 6 point-neuron coordinate · 7 dendritic preview ·
8 conductance/shunting · 9 local eligibility · 10 transported field ·
11 exact factorization · 12 resolution ladder · 13 operator utility ·
14 theory boundary · 15 neuron identity/ownership · 16 credit-conflict task ·
17 path-demand crossover · 18 ancestry budget · 19 physical depth ·
20 reconstructed arbors · 21 focal shunting · 22 measured-response null ·
23 alignment rescue · 24 signed animal coordinates · 25 evidence phase plane ·
26 resource ledger.

## 15-minute cut (15 slides)

Keep 1, 2, 3, 5, 6, 8, 11, 13, 15, 16, 17, 21, 23, 25, 26.

This preserves the conceptual chain and the controlled task that makes a path
address beneficial. State the ancestry interior optimum verbally on slide 25.
Mention the measured functional null while transitioning from slide 21 to 23.
This cut loses the full hierarchy sweep, physical-depth existence proof,
wiring economics plot, and external animal result.

## 20-minute cut (20 slides)

Keep 1–3, 5–11, 13–17, 21–23, 25–26; skim 5 and 15. This is the preferred
short technical talk because it preserves the derivation and the
path-demand crossover plus the null-to-rescue arc. Add slide 20 for a
neuroanatomy-heavy room, replacing slide 15
if time is fixed.

## Backup pages

The Backup divider is PDF page 27. A1–A7 are pages 28–34:

| Backup | Topic |
|---|---|
| A1 | adjoint and path-product derivation |
| A2 | derivation/exactness caveat for \(U(M)\) |
| A3 | factorial controls and matched point implementation |
| A4 | validity exclusions and exact/BP agreement audit |
| A5 | reliability gain and same-span conditioning |
| A6 | fresh raw-additive CIFAR-10 feedback ladder |
| A7 | references |

## Asset and production policy

The Canvas exports under `canvas_assets/` are **design references only**.
They informed the white-ground layout, three-column logic, large typographic
claims, dark-blue footer rhythm, and the progression from network to neuron
to branch. They are raster images with generated text/equations and are not
safe as scientific or publication masters.

High-value design references from the newest set are:

- `canvas_assets/dendritic_workshop_slides/slide_11.png` — three-part
  task/required-credit/results composition. Its original generated equations
  and values were not reused; core slides 16–17 use a native/vector schematic
  and the confirmatory credit-conflict source data.
- `slide_12.png` — hierarchy/budget/result grid, used only as the layout
  reference for core slide 18.
- `slide_13.png` — matched-task/matched-resource/result structure for core
  slide 19.
- `slide_14.png` — morphology + model + capture organization for core slide
  20.
- `slide_15.png` — perturbation/control/result sequence for core slide 21;
  its unconditional language was replaced by the passive-null and
  high-conductance boundary.
- `slide_16.png` and `slide_17.png` — measured-null → controlled-rescue visual
  pair, rebuilt as core slides 22–23.
- `slide_18.png` — phase-plane composition, rebuilt with the background
  labeled theory guided and markers labeled measured/simulated on core slide
  25.
- `slide_19.png` — resource-ledger composition, simplified for core slide 26.
- `slide_20.png` — contact sheet only; omit from any deck.

The earlier files named `ChatGPT Image Aug 27, 2026, 10_37_*.png` and
`11_05_*.png` are useful references for the introductory hierarchy and
equation-card geometry, but not for content reuse. `extra/extra_38.pptx`
contains full-slide raster images rather than editable scientific objects.

The publication-ready sources in this deck are native/vector:
`credit_tree_lib.tex` for tree schematics, `eq_style.tex` for equations,
TikZ inside `dendritic_credit_workshop_25_core.tex`, and the vector data crops
in `pdf_assets/*.pdf`. `build_path_necessity_talk_assets.py` rebuilds the two
slide-specific derivatives of Supplementary Fig. S29 with projection-scale
type and the symbol \(\chi\). Keep geometry and semantic colors synchronized
through these sources; never trace equations or data from a Canvas PNG.
