# Manuscript branch and full-evidence snapshot

The canonical journal source is `journal/main.tex`; repository-root `main.tex`
is its Overleaf entry point. The frozen earlier paper remains in `neurips/`.
The September 17 cleanup changes file tracking, not scientific text, numerical
results or publication artwork.

## What is tracked on main

- Article and SI TeX inputs, bibliography, and all 9 main and 36 SI figures.
- Earlier NeurIPS paper sources and figures.
- Source code, tests, small protocols, editorial documents and build scripts.
- Compact `source_data/curated_publication/` tables and their manifest.

## What remains local and archived

Bulk simulation outputs, detailed provenance inventories, YAML execution records,
internal reviews, superseded panel sheets, compiled reading PDFs and
`journal/LLR_n.pptx` were removed from the Git index, not from disk.
They are recoverable from the complete pre-cleanup tracked snapshot:

- Branch: `archive/pre-overleaf-cleanup-20260917`.
- Commit: `fbe379aef67a59a0827396f2b6b636dc1dabe0cd`.
- [Browse the preserved snapshot](https://github.com/houman1359/dendritic-local-learning/tree/archive/pre-overleaf-cleanup-20260917).
- [Presentation](https://github.com/houman1359/dendritic-local-learning/blob/archive/pre-overleaf-cleanup-20260917/journal/LLR_n.pptx)
  (Git LFS, 187,222,233 bytes; full download requires Git LFS).

The archive preserves previously tracked content, not every untracked local run.
It is not a newly validated software release. Existing external data and access
requirements still apply. Use a separate new checkout/worktree of the archive
for recovery or complete evidence packaging; do not overwrite the working paper
or force-add archived data into `main`. Git history was not rewritten, so the
repository's historical object storage remains large even though the current
tracked tree is much smaller.

## Build boundaries

A fresh `main` checkout can compile the Article and SI from the tracked figure
PDFs with pdfLaTeX and BibTeX (`make -C journal combined`). The repository-root
wrapper builds only the Article, not the combined reading copy. The compact
tables are not sufficient to regenerate every figure or scientific audit.
Those workflows and release packaging need the full frozen evidence and local
execution records. The existing research checkout retains those inputs.

Overleaf does not support Git LFS. It recommends fewer than 100 MB for GitHub
sync and limits projects to 2,000 files; editable-text limits apply separately.
This branch retains research code, so the minimal source bundle remains an
alternative if the editor's text-size limit prevents importing the full branch.
See [Overleaf limits](https://docs.overleaf.com/getting-started/free-and-premium-plans/plan-limits)
and [GitHub synchronization](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization).
An existing project may need a fresh import if its synchronization cannot apply
the large cleanup diff. Preserve any Overleaf-only edits before replacing it.
