# Reproducibility Notes

These notes give a compact map from the manuscript to the repository contents.
They are intentionally protocol-level for the current preprint. A final public
release will add a pinned environment, immutable tag, end-to-end launch scripts,
and expected summary hashes.

Run commands from the repository root:

```bash
cd drafts/dendritic-local-learning
```

## What Is Included

- Core dendritic model and LocalCA implementation.
- Training and diagnostic scripts for the reported experiment families.
- Representative configuration files for the main architectures and controls.
- Figure-generation scripts for the main and supplementary figures.
- Tests covering key dataset and implementation utilities.

## Main Experiment Families

- Exact-gradient reconstruction and gradient-fidelity diagnostics.
- Layer-soma factorial diagnostic separating soma-error reuse from within-tree
  path transport.
- Path-gain, exact-error rank, and implemented-feedback fidelity diagnostics.
- Post-training inhibitory-conductance interventions.
- Transported-error oracle and feedback-construction controls.
- Matched-capacity MNIST, Fashion-MNIST, and figure-ground MNIST performance.
- Supplementary stress tests: morphology, feedback noise, CIFAR-10 compact
  controls, FA/DFA, cue routing, and low-rank feedback.

## Figure Regeneration

The following commands regenerate the manuscript figures from the summaries
currently present in the repository:

```bash
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_figure1_schematic.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_theory_diagnostics_figures.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_neurips_figures.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_alignment_norm_dynamics_figure.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_cifar_sweep_comparison_figures.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_morphology_ie_regime_figure.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_inhibition_causality_figure.py
PYTHONPATH=../../src:$PYTHONPATH python scripts/generate_cue_routing_figures.py
```

These are figure/summarization entry points, not yet a frozen from-scratch
reproduction artifact. The final release will provide a single-script path from
raw training launches to checked summaries.

## Environment

The repository currently includes dependency specifications at:

- `requirements.txt`
- `pyproject.toml`
- `setup/environment.yml`

The accepted-version release will identify the exact environment file and commit
used for the public artifact.

## Build Commands

```bash
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment.tex
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment.tex
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment_arxiv.tex
pdflatex -interaction=nonstopmode -halt-on-error local_credit_assignment_arxiv.tex
```

After building, scan logs for unresolved citations, references, overfull boxes,
and fatal errors:

```bash
rg "Undefined|Warning: Citation|Warning: Reference|Overfull|Error|Fatal" *.log
```
