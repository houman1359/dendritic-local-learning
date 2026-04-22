# Dendritic Local Learning Draft

This directory holds the current submission-facing assets for the LocalCA / dendritic local learning paper.

The draft is now organized around one shared manuscript body plus two thin venue wrappers:

- `local_credit_assignment_body.tex`: shared paper text
- `local_credit_assignment.tex`: NeurIPS wrapper
- `local_credit_assignment_arxiv.tex`: arXiv wrapper
- `local_credit_assignment_checklist.tex`: NeurIPS checklist helper

## Canonical Submission Assets

### Main figures used by the paper

- `figures/fig1_model_and_credit.pdf`
- `figures/fig3_gradient_fidelity.pdf`
- `figures/fig5_mechanistic_evidence.pdf`
- `figures/fig2_competence_regime.pdf`
- `figures/fig_s8_morphology_ie_regime.pdf`
- `figures/fig_cifar10_mechanism_extension.pdf`
- `figures/fig6_cue_routing.pdf`

### Appendix figures used by the paper

- `figures/fig_s1_calibration.pdf`
- `figures/fig_s2_gradient_extended.pdf`
- `figures/fig_s3_5f_sensitivity.pdf`
- `figures/fig_s4_verification.pdf`
- `figures/fig_additional_stress_tests.pdf`
- `figures/fig_s_low_bandwidth.pdf`
- `figures/fig_s5_fa_dfa.pdf`
- `figures/fig_additive_norm_control.pdf`
- `figures/fig_s_soma_extension.pdf`
- `figures/fig_s_cue_routing_soma.pdf`
- `figures/fig_s_cifar10_soma_extension.pdf`
- `figures/fig_s_bm_policy.pdf`
- `figures/fig_s_ablation.pdf`
- `figures/fig_weight_distributions.pdf`
- `figures/fig_s_weight_dist_by_depth.pdf`
- `figures/fig_s_weight_dist_soma_comparison.pdf`

### Canonical result summaries feeding the current draft

- Corrected mechanism stack:
  - `analysis/theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix_summary/`
  - `analysis/path_transport_upper_bound_nonnegativeinput_fix_5seed/`
  - `analysis/rank_bridge_nonnegativeinput_fix/`
- Competence panel:
  - `figures/data/competence_summary_20260422.csv`
- Corrected CIFAR mechanism extension:
  - `analysis/cifar10_compactei_depth4_decoderfix_mechanism_5seed/`
- Pinned appendix additions:
  - `analysis/soma_extension_summary_20260417/`
  - `analysis/cue_routing_soma_summary_20260417/`
  - `analysis/cifar10_depth4_soma_summary_20260417/`
  - `analysis/component_ablation_summary_20260417/`
  - `analysis/weight_dist_by_depth_summary_20260417/`
  - `analysis/weight_dist_soma_summary_20260417/`
  - `analysis/bm_classification_summary_20260415/`

## Directory Guide

- `configs/`: paper-facing config snapshots and sweep manifests
- `scripts/`: figure generation, summary building, and manuscript helpers
- `figures/`: current paper figures plus local figure data
- `analysis/`: local result summaries and pinned paper inputs
- `archive/`: local archive for superseded draft assets; ignored by git
- `local_sweep_runs/`: local sweep bundles and outputs; ignored by git
- `outputs/`, `logs/`, `slurm_jobs/`: local execution artifacts; ignored by git

## Archive Policy

When a figure, note, or analysis artifact is no longer part of the current paper flow but is still worth keeping for reference, move it into `archive/` rather than deleting it. The `archive/` directory is ignored by git, so it is safe to use for local cleanup without losing history that matters to the submission.

Current rule of thumb:

- keep in place anything referenced by `local_credit_assignment_body.tex`
- keep in place anything consumed by the active figure-generation scripts
- move superseded or exploratory leftovers to `archive/`

## Rebuild Commands

Rebuild the current figures:

```bash
cd drafts/dendritic-local-learning
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_figure1_schematic.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_theory_diagnostics_figures.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_neurips_figures.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_revision_figures.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_cue_routing_figures.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_morphology_ie_regime_figure.py
```

Compile the two wrappers:

```bash
cd drafts/dendritic-local-learning
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment.tex
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment.tex
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment_arxiv.tex
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment_arxiv.tex
```

Refresh the newer appendix summaries from completed local runs:

```bash
cd drafts/dendritic-local-learning
PYTHONPATH=../../src:$PYTHONPATH python scripts/summarize_neurips_new_sweeps.py
```

## Historical Naming Note

Some older sweep manifests and on-disk run names still contain historical `_bp_` labels in LocalCA experiment names. In the current manuscript and summaries, the cleaned terminology is:

- `learned_local_bm`: LocalCA updates `(b,m)` through the local rule
- `quantile_bm`: `(b,m)` are refreshed by the quantile rule during training
- `backprop_bm`: optimizer/backprop updates `(b,m)` in the standard-training baseline

The current manuscript uses the explicit labels above even when older on-disk run names preserve the historical strings for reproducibility.
