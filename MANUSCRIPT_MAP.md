# Manuscript map

## Current journal article

Edit `journal/main.tex`. Its title, abstract, Results, Methods, declarations
and main-figure captions are the authoritative journal text.

Edit `journal/supplementary/supplementary.tex` for Supplementary Information.
The canonical reading copy is `journal/main_with_supplementary.pdf`, which
concatenates the compiled main Article and Supplementary Information.

Upload only `journal/submission/Overleaf_Project.zip` to Overleaf. It is built
from an explicit allow-list and contains only the TeX/BibTeX sources and the
canonical figure PDFs required to compile the paper. Source Data and software
are separate submission archives; they do not belong in the Overleaf editor.

## Frozen NeurIPS paper

The NeurIPS manuscript is not part of the journal Overleaf package. It remains
under `neurips/`:

- `local_credit_assignment_body.tex`: shared conference manuscript body;
- `local_credit_assignment.tex`: anonymous NeurIPS wrapper;
- `local_credit_assignment_arxiv.tex`: arXiv wrapper;
- `local_credit_assignment.pdf` and `local_credit_assignment_arxiv.pdf`:
  compiled reading copies.

The journal extends this work; it does not overwrite these files.

## Noncanonical material

Generated staging directories, raw training runs, superseded figures and old
submission ZIPs are ignored and kept outside the active working tree. The
former `journal/theory-paper/` was a theory-only writing experiment, not a
submission manuscript; it is retained in Git history and the local archive.

`journal/figures/generated/` is a temporary build directory. LaTeX references
only `journal/figures/main/` and `journal/figures/supplementary/`.
