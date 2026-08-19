# ARCHIVED CONFERENCE SNAPSHOT — Dendritic Local Learning

This directory is frozen provenance for the submitted NeurIPS manuscript. It
is not the current paper and should not be edited to change scientific claims,
figures or submission files. The standalone canonical article is
`../journal/main.tex`; its complete reading copy is
`../journal/main_with_supplementary.pdf`.

This folder contains the submission-facing draft for the LocalCA dendritic local-learning paper.
The manuscript uses one shared body and two thin wrappers:

- `local_credit_assignment_body.tex`: shared manuscript text, references, and appendix.
- `local_credit_assignment.tex`: anonymous NeurIPS 2026 submission wrapper.
- `local_credit_assignment_arxiv.tex`: arXiv/preprint wrapper.
- `local_credit_assignment_checklist.tex`: NeurIPS checklist, included only by the NeurIPS wrapper.

## NeurIPS Submission Checks

The NeurIPS wrapper intentionally uses the current style file without `final` or `preprint` options:

```tex
\usepackage{neurips_2026}
```

The current PDF layout is:

- Main content ends before references.
- References currently start on page 10; final page-limit trimming is intentionally deferred.
- Supplementary Results start after references.
- The NeurIPS checklist follows the appendix in the same PDF.
- Page size is letter.
- The PDF has no undefined references or citations in a two-pass compile.

The appendix begins directly after the bibliography so the final reference page does not carry avoidable blank space; appendix figures are kept near their source with local float placement.

## Main Figures

The current main-paper figures are:

- `figures/fig1_model_and_credit.pdf`: model, path gains, and broadcast taxonomy.
- `figures/fig2_gradient_fidelity.pdf`: exact factorization, final gradient-fidelity diagnostics, and the layer-soma factorial check.
- `figures/fig3_mechanistic_evidence.pdf`: path-gain dispersion, scope-corrected dendritic-feedback fidelity, inhibition intervention, and oracle learning.
- `figures/fig4_competence_regime.pdf`: matched-capacity performance, inhibition dose-response, morphology regime, mechanism controls, and the fifteen-seed neuron-wise-feedback intervention.
- `figures/fig5_rule_feedback_controls.pdf`: corrected, source-backed rule, error-source, exact-transport, and feedback-construction controls.

Key appendix figures cover verification and seed checks, extended gradient
diagnostics, CIFAR-10 mechanism stress tests, morphology maps,
feedback-alignment baselines, input-mode probes, and the fuller cue-routing
boundary analysis.

## Rebuild Commands

From this folder:

```bash
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_figure1_schematic.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_theory_diagnostics_figures.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_neurips_figures.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_alignment_norm_dynamics_figure.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_cifar_sweep_comparison_figures.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_cue_routing_figures.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_morphology_ie_regime_figure.py
PYTHONPATH=../../../src:$PYTHONPATH python scripts/generate_inhibition_causality_figure.py
```

Compile the NeurIPS wrapper:

```bash
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment.tex
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment.tex
```

Compile the arXiv wrapper:

```bash
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment_arxiv.tex
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment_arxiv.tex
```

Refresh appendix summaries from completed local runs:

```bash
PYTHONPATH=../../../src:$PYTHONPATH python scripts/summarize_neurips_new_sweeps.py
```

## Markdown Policy

Only this `README.md` should be tracked in this folder. Dated review notes, status reports, and local analysis narratives should stay untracked or be moved to `archive/`, which is ignored by git.

## Cleanup Policy

Keep files in place if they are referenced by `local_credit_assignment_body.tex` or consumed by active figure-generation scripts. Move superseded draft notes, review documents, and exploratory artifacts to `archive/` rather than leaving them beside the submission wrappers.
