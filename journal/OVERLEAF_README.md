# Overleaf build instructions

The active manuscript is `main.tex`. Use pdfLaTeX; the bibliography is
`references.bib`. The separate Supplementary Information source is
`supplementary/supplementary.tex` and the combined reading copy is
`main_with_supplementary.pdf`.

The manuscript compiles eight numbered main figures. Some large figures span
more than one displayed asset through LaTeX's `ContinuedFloat`; their canonical
filenames explicitly give the manuscript figure number and panel range.

- Main-text assets: `figures/main/`
- Supplementary assets: `figures/supplementary/`
- Exact panel map: `figures/README.md`

Do not upload or select files from `figures/generated/`: these are internal
script outputs with historical descriptive names. The generated submission
bundle already excludes them and contains only the canonical assets.

From the journal project directory:

```bash
make combined          # compile Article + SI reading copy
make overleaf-bundle   # rebuild submission/Overleaf_Project.zip
```

Upload `Overleaf_Project.zip` as a new Overleaf project, or replace every file
in an existing project with its contents. Select `main.tex` as the main
document and pdfLaTeX as the compiler. Do not merge it with the older flat
`fig*.pdf` assets: those filenames are deliberately absent from this package.
