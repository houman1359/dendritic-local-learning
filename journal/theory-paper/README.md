# Focused theory companion

Working title: **Credit covariance and routing in dendritic trees**

This is a concrete companion-paper option, not a second manuscript that is
ready for simultaneous submission. It isolates the exact factorization,
route-subspace definition, covariance objective, descent guarantee,
fixed-state shunting proposition, and interference identity.

Build from this directory with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Editorial boundary

The working draft currently reuses empirical figures from the comprehensive
manuscript so that the theory can be evaluated in context. Before either paper
is submitted, the authors must choose one of two clean routes:

1. submit one integrated article and keep this file as an internal theory
   note; or
2. divide claims, figures, source tables, and text into genuinely distinct
   papers, cross-cite the companion, and disclose the relationship to both
   editors.

Do not submit the two current drafts independently: their empirical figures
and parts of the derivation overlap substantially.

## Suitable scope

If separated, this manuscript fits a theoretical-neuroscience or neural-
computation venue better than a general physics letter. Plausible homes are
Neural Computation, Physical Review Research, PLOS Computational Biology, or a
theory-focused article in Network Neuroscience. The comprehensive manuscript
should retain the MICrONS, animal-data, and biological validation story.
