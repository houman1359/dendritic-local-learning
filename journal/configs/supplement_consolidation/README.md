# Supplementary figure sources

The 35 supplementary figures follow ten scientific sections. `manifest.json` records every selected source panel, exact vector crop, placement, source hash, numerical data path and native label or legend repair. `reference_map.json` maps all 56 original supplementary figure identifiers and their retained panels to current semantic labels. `publication_curation_map.json` maps scientific sections and tables. An omitted printed panel remains available through its original source asset and complete scientific data; omissions do not remove outcomes.

Rebuild from the journal directory with `python -B scripts/supplement_consolidation/build.py`. The builder uses the frozen asset, caption and provenance registries in that script directory. It checks original PDF identities and requires the original numerical-source tables named in the explicit source registry. Original figures remain immutable inputs. Vector panels, including original raster data layers where present, retain their scientific content. New panel letters and a few shared labels/keys are native text and vectors.

This is an editorial assembly, not a new scientific experiment. Derivations, negative controls, failed selectors, small effects and finite-budget limitations are located in the scientific notes and the full source index.
