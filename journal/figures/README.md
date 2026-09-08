# Publication figures

Article: *Dendritic morphology as a dictionary for local credit assignment*.

The Article contains nine main figures. Its Supplementary Information contains 35 figures organized by scientific question. Only the assets included by the two manuscript sources belong to the publication sequence; other vector files are reproducible rendering inputs.

## Main sequence

| Figure | Scientific question | Rendering source |
|---|---|---|
| 1 | Error source, spatial dictionary, gain, conditional noise filtering and the image reference | `scripts/credit_first_figures/build_restored_main.py` |
| 2 | Branch selection under contextual conflict | Authenticated native input; `scripts/build_main_figure_04.py` |
| 3 | Ancestry at matched bandwidth and its coefficient prediction | Authenticated native input; `scripts/credit_first_figures/build_ancestry.py` |
| 4 | Matched input spectra, extended learning and learned credit geometry | `scripts/credit_first_figures/build_restored_main.py` |
| 5 | Local inhibitory gating in a conductance tree | Authenticated native input; `scripts/conductance_local_gate/figure.py` |
| 6 | Task families, serial architecture and budget-dependent credit comparisons | `scripts/credit_first_figures/build_restored_main.py` |
| 7 | Anatomical columns, capture across budgets and delivery cost | `scripts/credit_first_figures/build_restored_main.py` |
| 8 | Ancestry-partition gain and its electrical-state dependence | Authenticated native input; `scripts/shunt_ancestry_gain/build_figure.py` |
| 9 | Measured ancestry alignment and conditional detection sensitivity | `scripts/credit_first_figures/build_restored_main.py` |

Paths are relative to the journal directory. All main assets are `main/figure_01.pdf` through `main/figure_09.pdf`. The four unchanged native inputs are pinned in `configs/figure_structure/retained_main_inputs.json`. The other five figures have per-panel definitions, plotted tables and input hashes in `figures/provenance/structure_restoration_20260908/`.

## Supplementary sequence

The contents and main-figure guide in `supplementary/supplementary.tex` locate the exact rules and theory, image tasks, branch/ancestry controls, interaction and Boolean tasks, conductance learning, physical depth, anatomical dictionaries, shunting, measured responses and structure selection. Captions distinguish primary experiments, subsequent analyses of the same observations, and independent cohorts.

Canonical supplementary assets have semantic filenames under `supplementary/curated/`. `scripts/supplement_consolidation/specification.py` declares the selected panels and captions. `configs/supplement_consolidation/manifest.json` records their source identities, original panel regions and final placements. The immutable vector inputs, source captions and numerical-source associations are authenticated by the registries in `scripts/supplement_consolidation/`. Complete scientific outcomes remain in Source Data even when a repeated graphical view is omitted.

## Rebuilding and checking

From the journal directory:

```bash
python scripts/rebuild_final_publication_figures.py
python scripts/credit_first_figures/verify_restored_main.py
make combined
```

The figure build uses frozen numerical tables and authenticated vector panels; it runs no training or experimental selection. Generated main display summaries are copied into `source_data/curated_publication/`. The explicit publication provenance is in `configs/credit_first_provenance/panel_sources.json` and `source_data/provenance_manifest.tsv`.

Source and Overleaf bundles include only the current manuscript inputs and displayed assets. The software package also retains the authenticated inputs needed to rebuild them. A source panel's original filename does not identify a current supplementary number: use the current caption, semantic label and manifest.

Inspect final compiled figures at manuscript scale, including lettering, type sizes, axis and legend clipping, intervals and caption agreement. Current page counts must come from the final PDFs.
