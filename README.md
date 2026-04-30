# Dendritic Local Learning Draft

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
- References start on page 10, so the paper uses the allowed nine content pages.
- Supplementary Results start after references.
- The NeurIPS checklist follows the appendix in the same PDF.
- Page size is letter.
- The PDF has no undefined references or citations in a two-pass compile.

The appendix boundary is guarded with `\clearpage` after the bibliography and `\suppressfloats[t]` on the first appendix page so appendix figures do not float into references.

## Main Figures

The current main-paper figures are:

- `figures/fig1_model_and_credit.pdf`: model, path gains, and broadcast taxonomy.
- `figures/fig3_gradient_fidelity.pdf`: exact factorization and gradient-fidelity diagnostics.
- `figures/fig5_mechanistic_evidence.pdf`: path-gain/error-field compressibility, inhibition intervention, broadcast fidelity, and oracle learning.
- `figures/fig2_competence_regime.pdf`: capacity-calibrated competence, inhibition regime, morphology summary, and 3F/4F/5F ablation.

Key appendix figures include the local-rule and feedback-design controls, CIFAR-10 stress tests, morphology maps, cue-routing diagnostics, soma-on extensions, activation audits, and weight-statistics analyses.

## Rebuild Commands

From this folder:

```bash
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_figure1_schematic.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_theory_diagnostics_figures.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_neurips_figures.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_revision_figures.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_cue_routing_figures.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_morphology_ie_regime_figure.py
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

Refresh newer appendix summaries from completed local runs:

```bash
PYTHONPATH=../../src:$PYTHONPATH python scripts/summarize_neurips_new_sweeps.py
```

## Markdown Policy

Only this `README.md` should be tracked in this folder. Dated review notes, status reports, and local analysis narratives should stay untracked or be moved to `archive/`, which is ignored by git.

## Cleanup Policy

Keep files in place if they are referenced by `local_credit_assignment_body.tex` or consumed by active figure-generation scripts. Move superseded draft notes, review documents, and exploratory artifacts to `archive/` rather than leaving them beside the submission wrappers.
