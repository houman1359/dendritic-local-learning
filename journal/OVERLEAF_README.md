# Overleaf build instructions

Article: *Dendritic morphology as a dictionary for local credit assignment*.

For the standalone project, select `main.tex` and use pdfLaTeX with the included `references.bib`. Compile the Supplementary Information from `supplementary/supplementary.tex`. The combined reading copy is `main_with_supplementary.pdf`.

The Article uses ten main figures. The Supplementary Information uses 36 figures and nineteen tables, organized by scientific question. Its contents and main-figure guide locate the supporting methods and comparisons. `figures/README.md` describes the display sequence and reproducible rendering inputs.

Keep the `supplementary/curated/` methods, table and figure fragments with their master document. The package contains only the TeX inputs reachable from the current manuscript and its displayed figure assets. Superseded manuscript fragments and unused displays are excluded.

From the journal directory:

```bash
make combined
make overleaf-bundle
```

Use `submission/Overleaf_Project.zip` for a standalone project. Replace an older project in full to avoid mixing different manuscript versions. In a repository-synchronized project, the repository's root-level `main.tex` inputs the canonical journal Article.

Verify both compiled previews after upload. Page counts are recorded from the final PDFs; building or uploading the project does not submit the paper.

For GitHub synchronization, pull from `houman1359/dendritic-local-learning`
after the local changes have been pushed. Select the repository-root `main.tex`
for the journal Article, not a NeurIPS wrapper. The combined reading PDF is made
locally by `make combined`; compiling `main.tex` alone does not append the SI.
The presentation uses Git LFS only on the archive branch and is deliberately
absent from the synchronized branch. See `REPOSITORY_LAYOUT.md` for the full
evidence snapshot and local-only files. Do not synchronize the archive branch
to Overleaf.
