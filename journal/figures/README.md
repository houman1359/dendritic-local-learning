# Publication figures

Article: *Dendritic morphology as a dictionary for local credit assignment*.

The Article contains ten numbered main figures; Figures 5 and 6 separate mechanism and population/rescue. Its Supplementary Information contains 36 figures organized by scientific question. Only the assets included by the two manuscript sources belong to the publication sequence; other vector files are reproducible rendering inputs.

## Main sequence

| Figure | Scientific question | Rendering source |
|---|---|---|
| 1 | Error source, spatial dictionary, gain, conditional noise filtering, image learning and matching-coordinate capture | `scripts/credit_first_figures/build_framework.py` |
| 2 | Branch selection under contextual conflict | Authenticated native input; `scripts/build_main_figure_04.py` |
| 3 | Ancestry at matched bandwidth, the coefficient prediction and a separate coefficient-learning test | `scripts/credit_first_figures/build_ancestry.py` |
| 4 | Matched input spectra, noise controls, extended learning and learned credit geometry | `scripts/credit_first_figures/build_restored_main.py --figures 4 6` |
| 5 | Local inhibitory selection in a seven-compartment neuron | `scripts/conductance_local_gate/figure.py` |
| 6 | Population assignment controls, nonlinear failure and parent-sensitivity rescue | `scripts/review_completion/population_figure.py` |
| 7 | Task families, serial architecture and budget-dependent credit comparisons | `scripts/credit_first_figures/build_restored_main.py --figures 4 6` |
| 8 | Anatomical columns, capture, cell heterogeneity and delivery cost | `scripts/credit_first_figures/build_anatomy.py` |
| 9 | Ancestry-partition gain and its electrical-state dependence | `scripts/shunt_ancestry_gain/build_focused_main.py` |
| 10 | Measured ancestry alignment and conditional detection sensitivity | `scripts/credit_first_figures/build_measured.py` |

`scripts/rebuild_final_publication_figures.py` runs the whole sequence and holds this mapping.

Paths are relative to the journal directory. Main assets are `main/figure_01.pdf` through `main/figure_10.pdf`. Figure 2 is pinned in `configs/figure_structure/retained_main_inputs.json`. The other nine figures have per-panel definitions, plotted tables and input hashes in `figures/provenance/structure_restoration_20260908/` or `figures/provenance/credit_clarity_20260908/`. The latter builders preserve the original authenticated study renderers and their outputs. Actual rendering libraries and font hashes are recorded in `figures/provenance/publication_render_environment.json`.

Figure 1G uses activation-error capture from the same fresh exact-rule cohort as
the learning comparison; the older voltage-coordinate diagnostics are
Supplementary Figure S8. Figure 1F is the exact-path-minus-per-neuron
comparison. Figure 3F uses a
separate coefficient-learning cohort and explicitly distinguishes primary noisy
soft coefficients from exploratory hard readout and noiseless calibration.
The matched-versus-rewired comparison remains in Supplementary Section S3 and
Source Data. Figure 7E–F identify retained stopped states; Figure 8G distinguishes residual-energy contrasts from total-energy contrasts.

## Supplementary sequence

The contents and main-figure guide in `supplementary/supplementary.tex` locate the exact rules and theory, image tasks, branch/ancestry controls, interaction and Boolean tasks, conductance learning, physical depth, anatomical dictionaries, shunting, measured responses and structure selection. Captions distinguish primary experiments, subsequent analyses of the same observations, and independent cohorts.

Canonical supplementary assets have semantic filenames under `supplementary/curated/`. `scripts/supplement_consolidation/specification.py` declares the selected panels and captions. `configs/supplement_consolidation/manifest.json` records their source identities, original panel regions and final placements. The immutable vector inputs, source captions and numerical-source associations are authenticated by the registries in `scripts/supplement_consolidation/`. Complete scientific outcomes remain in Source Data even when a repeated graphical view is omitted.

## Rebuilding and checking

From the journal directory:

```bash
python scripts/rebuild_final_publication_figures.py
python scripts/audit_letter_alignment.py --strict
python scripts/audit_row_separation.py --strict
make combined
```

The figure build uses frozen numerical tables and authenticated vector panels; it runs no training or experimental selection. Generated main display summaries are copied into `source_data/curated_publication/`. The explicit publication provenance is in `configs/credit_first_provenance/panel_sources.json` and `source_data/provenance_manifest.tsv`.

Source and Overleaf bundles include only the current manuscript inputs and displayed assets. The software package also retains the authenticated inputs needed to rebuild them. A source panel's original filename does not identify a current supplementary number: use the current caption, semantic label and manifest.

Inspect final compiled figures at manuscript scale, including lettering, type sizes, axis and legend clipping, intervals and caption agreement. Current page counts must come from the final PDFs.

## Production specification

The ten main assets are `main/figure_01.pdf` through `main/figure_10.pdf`. The complete allowlist, including all
36 supplementary sheets, is `MAIN_FIGURES` / `SUPPLEMENTARY_FIGURES` in
`scripts/build_submission_bundle.py`. Other vector files are rendering inputs.

Current canvases have a common 518.4 pt authoring width and variable heights.
Use the current PDF metadata and layout audits for dimensions and typography;
older measurements of a uniform page height no longer describe this set.
The authoring width is 182.9 mm, requiring a 1.6% reduction to the journal's
180 mm production width. Each main legend is audited separately.

`figures/provenance/publication_render_environment.json` records rendering
libraries and font hashes. Full provenance and Source Data must be refreshed
before producing a release bundle; existing archives may predate this revision.
