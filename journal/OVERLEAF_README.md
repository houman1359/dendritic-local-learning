# Overleaf build instructions

Article: *Dendritic morphology as a dictionary for local credit assignment*.

For a repository-synchronized Overleaf project, select the root-level `main.tex`, which inputs the canonical journal Article. For the standalone ZIP, select its enclosed `main.tex`, use pdfLaTeX and the included `references.bib`. Build the Supplementary Information from `supplementary/supplementary.tex`. The combined reading copy is `main_with_supplementary.pdf`.

The Article uses eight main assets, `figures/main/figure_01.pdf` through `figure_08.pdf`, and 47 supplementary figures. The sequence runs from the credit framework and image tasks through branch selection, ancestry, interaction order, physical depth, anatomy, shunting and measured responses. The complete display map is `figures/README.md`.

Keep every local TeX input with its parent document, including the credit bridge, anatomy, historical noise-task identity and weak-channel figure fragments. The package builder resolves actual TeX inputs rather than relying on a fixed old list. S45 contains image diagnostics, S46 update utility and S47 the weak-channel local-linearization check. Historical panel suffixes in some supplementary filenames remain unchanged; current captions define their panels.

From the journal directory:

```bash
make combined
make overleaf-bundle
```

Rebuild after all source and figure changes are complete. Use `submission/Overleaf_Project.zip` for a new standalone project, or replace an older project in full to avoid mixing current sources with superseded assets. The package contains the scientific manuscript and required build materials; internal revision logs are excluded.

Page counts must be read from the final compiled PDFs. Verify the main and supplementary previews after upload, including all eight main and 47 supplementary displays. A local successful build does not submit the paper.
