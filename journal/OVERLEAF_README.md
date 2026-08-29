# Overleaf build instructions

When Overleaf is synchronized to the complete GitHub repository, select the
root-level `main.tex`; it inputs this directory's canonical `main.tex` while
preserving the visible `journal/`, `neurips/`, and `presentations/` structure.
The instructions below describe the smaller standalone ZIP fallback.

The canonical standalone Article source is `main.tex`. Use pdfLaTeX and
`references.bib`. Supplementary Information is in
`supplementary/supplementary.tex`; `main_with_supplementary.pdf` is the complete
reading copy.

The Article compiles exactly nine main assets:
`figures/main/figure_01.pdf` through `figure_09.pdf`. Supplementary assets are
in `figures/supplementary/` (S1--S29). The exact content map is
`figures/README.md`. Do not select the modular compositor inputs or files in
`figures/generated/`.

From the journal directory:

```bash
make combined
make overleaf-bundle
```

Upload `submission/Overleaf_Project.zip` as a new project, select `main.tex` as
the main document and use pdfLaTeX. Replacing an older Overleaf project in full
is safer than mixing this package with historical `fig*.pdf` files.
