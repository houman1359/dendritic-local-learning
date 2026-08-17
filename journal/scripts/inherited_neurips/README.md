# NeurIPS figure-generator snapshots

The three Python files in this directory are byte-identical snapshots from
`drafts/dendritic-local-learning` commit `8b8ae0c`, the source tree used for
the final NeurIPS/arXiv figure set:

- `generate_neurips_figures.py` generates the submitted gradient, competence,
  rule/feedback and regular-tree control panels;
- `generate_revision_figures.py` orchestrates the submitted mechanistic and
  supplementary figures;
- `generate_theory_diagnostics_figures.py` generates the submitted path-gain
  and mechanism panels.

They are preserved as immutable lineage records. Production journal figures
import reusable panel functions from these snapshots only when the estimand
and source data are unchanged. Journal-only or statistically revised panels
use the byte-identical `../neurips_style.py` visual system and explicitly
record that they are updated plots rather than exact inherited panels.

The submission Figure 1 source is separately preserved byte-for-byte as
`../figure1_neurips_components.py`, because its three panel functions are
directly imported into the expanded journal Figure 1.
