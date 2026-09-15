# Supplementary Information: source map

`supplementary.tex` is the only canonical Supplementary Information master.
It compiles the notation guide, ten scientific sections, 36 supplementary
figures, reference tables and bibliography. The current manuscript includes
this supplement in `../main_with_supplementary.pdf`; the standalone output is
`supplementary.pdf`.

## Active sources and generated files

The master's recursive `\input` closure defines what is published, not the
set of TeX or PDF files present on disk. `../scripts/tex_sources.py` resolves
that closure using the same compilation root as LaTeX.

- `curated/si_notation.tex` is the notation guide. It is included before
  section S1 and Supplementary Figure S1, so symbols are available before
  the detailed derivations and results.
- `curated/si_01_exact.tex` through `curated/si_10_selection_statistics.tex`
  contain the maintained scientific text. `curated/si_tables.tex` includes
  `curated/si_tables_retained.tex` for the retained reference tables.
- `curated/si_*_figures.tex` contain generated figure inclusions and captions.
  Their source is `../scripts/supplement_consolidation/specification.py`,
  including its caption additions. Edit that specification rather than
  making a caption change only in a generated fragment.
- `../scripts/supplement_consolidation/build.py` generates the curated figure
  assets and figure fragments. It also writes the manifest, captions,
  reference map, source-panel bounds and audit report under
  `../configs/supplement_consolidation/`. It does not generate the scientific
  text or notation guide.
- `../figures/supplementary/curated/*.pdf` are the included figure sheets.
  `../configs/supplement_consolidation/manifest.json` maps their current
  S1–S36 numbers to the source assets and panels. Older filenames elsewhere
  under `../figures/supplementary/` are source or retained assets; their
  numbers need not match the current supplement. Numerical inputs remain
  under `../source_data/` and the builders named in the manifest.

As checked on 2026-09-14, the manifest contains 26 whole-source sheets and
10 panel compositions: eight combine multiple source assets, while S5 and
S26 reflow panels from one source each. All sheet and panel paste scales are
1.0. Full-scale placement does not mean that the composed sheets have no
crops or placement geometry.

From the journal directory, `make supplement` compiles the standalone
Supplementary Information; `make combined` builds the main paper and the
combined PDF. Figure regeneration is a separate upstream production step.

## Inactive top-level TeX fragments

The following 19 files are retained reference fragments, not alternate
masters and not part of the active compilation closure. Their bytes and
paths are preserved for source history and any upstream references. For
current scientific text, use the active `curated/` sources above; do not
assume that an inactive fragment is synchronized with the manuscript.

- `anatomy_commonmode_methods.tex`
- `boolean_morphology_methods.tex`
- `conductance_credit_controls_figures.tex`
- `conductance_credit_demand_methods.tex`
- `conductance_expanded_rates_figure.tex`
- `conductance_expanded_rates_methods.tex`
- `credit_first_bridge_methods.tex`
- `image_ladder_controls_figure.tex`
- `image_ladder_controls_methods.tex`
- `morphology_calibration_methods.tex`
- `morphology_conductance_methods.tex`
- `morphology_credit_methods.tex`
- `morphology_end_to_end_methods.tex`
- `morphology_followup_methods.tex`
- `noise_task_identity.tex`
- `physical_depth_budget_methods.tex`
- `review_followups.tex`
- `shunt_normalized_dose_figure.tex`
- `shunt_weak_channel_figure.tex`

`../tests/test_supplement_source_structure.py` checks this inventory against
the recursive TeX closure and guards the notation guide's placement and
unique label. No retained fragment needs to be deleted to build the paper.
