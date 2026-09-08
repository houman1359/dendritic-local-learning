# Publication figures

Article: *Dendritic morphology as a dictionary for local credit assignment*.

The Article contains nine main figures. Its Supplementary Information contains 35 figures organized by scientific question. Only the assets included by the two manuscript sources belong to the publication sequence; other vector files are reproducible rendering inputs.

## Main sequence

| Figure | Scientific question | Rendering source |
|---|---|---|
| 1 | Error source, spatial dictionary, gain, conditional noise filtering, image learning and matching-coordinate capture | `scripts/credit_first_figures/build_restored_main.py` |
| 2 | Branch selection under contextual conflict | Authenticated native input; `scripts/build_main_figure_04.py` |
| 3 | Ancestry at matched bandwidth, the coefficient prediction and a separate coefficient-learning test | `scripts/credit_first_figures/build_ancestry_focused.py` |
| 4 | Matched input spectra, extended learning and learned credit geometry | `scripts/credit_first_figures/build_restored_main.py` |
| 5 | Hard and continuous local inhibitory gating in a conductance tree | `scripts/conductance_local_gate/build_focused_main.py` |
| 6 | Task families, serial architecture and budget-dependent credit comparisons | `scripts/credit_first_figures/build_restored_main.py` |
| 7 | Anatomical columns, capture, cell heterogeneity and delivery cost | `scripts/credit_first_figures/build_restored_main.py` |
| 8 | Ancestry-partition gain and its electrical-state dependence | `scripts/shunt_ancestry_gain/build_focused_main.py` |
| 9 | Measured ancestry alignment and conditional detection sensitivity | `scripts/credit_first_figures/build_restored_main.py` |

Paths are relative to the journal directory. All main assets are `main/figure_01.pdf` through `main/figure_09.pdf`. Figure 2 is pinned in `configs/figure_structure/retained_main_inputs.json`. The other eight figures have per-panel definitions, plotted tables and input hashes in `figures/provenance/structure_restoration_20260908/` or `figures/provenance/credit_clarity_20260908/`. The latter builders preserve the original authenticated study renderers and their outputs. Actual rendering libraries and font hashes are recorded in `figures/provenance/publication_render_environment.json`.

Figure 1F uses activation-error capture from the same fresh exact-rule cohort as
the learning comparison; the older voltage-coordinate diagnostics remain
separate in SI. Figure 1G contains the separate image controls. Figure 3D uses a
separate coefficient-learning cohort and explicitly distinguishes primary noisy
soft coefficients from exploratory hard readout and noiseless calibration.
The matched-versus-rewired comparison remains in Supplementary Section S3 and
Source Data. Figure 6D identifies retained stopped states; the cell points in
Figure 7D use the same residual-energy coordinates as its interval.

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
