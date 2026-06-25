# Reproducibility Notes

This file records the analysis sources and commands behind the manuscript figures.
It is intentionally separate from the paper so the PDF can stay focused on the
scientific claims.

Run commands from this directory:

```bash
cd drafts/dendritic-local-learning
```

## Figure Generation

- Main figures and most summary tables:

```bash
python scripts/generate_neurips_figures.py
```

- Theory, path-gain, intervention, rank, and oracle diagnostic figures:

```bash
python scripts/generate_theory_diagnostics_figures.py
```

- Layer-soma factorial diagnostic used in Figure 2D:

```bash
python scripts/measure_layer_soma_factorial.py
python scripts/generate_neurips_figures.py
```

- Path-transport oracle summaries:

```bash
python scripts/summarize_path_transport_sweep.py
python scripts/generate_theory_diagnostics_figures.py
```

- Matched-capacity performance summary:

```bash
python scripts/summarize_neurips_claim_sweeps.py
python scripts/generate_neurips_figures.py
```

## Primary Analysis Sources

The manuscript uses the following source families for current figures and tables:

- Gradient and inhibition-dose diagnostics:
  `analysis/gradient_fidelity_vs_ie_nonnegativeinput_fix/`
- Theory diagnostics:
  `analysis/theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix_summary/`
- Layer-soma factorial diagnostic:
  `analysis/layer_soma_factorial_input_mode1_direct_i_pathgain_3f/`
- Exact-error rank diagnostics:
  `analysis/error_rank_selected_20260427/`
- Inhibitory-conductance interventions:
  `analysis/inhibition_causality_selected_20260427/`
- Path-transport oracle:
  `analysis/path_transport_upper_bound_nonnegativeinput_fix_5seed/`
- Noise-resilience feedback construction:
  `analysis/rank_bridge_nonnegativeinput_fix/`
- Matched-capacity performance:
  `figures/data/competence_summary_20260422.csv`

Older summaries can remain in the repository for comparison, but they should not
be mixed into the current figures unless the corresponding selection rule,
input regime, and seed set are explicitly rechecked.

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
