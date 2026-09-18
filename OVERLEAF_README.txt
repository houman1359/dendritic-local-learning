CANONICAL PAPER AND OVERLEAF SYNC

Repository: https://github.com/houman1359/dendritic-local-learning
Overleaf-synchronized branch: main
Compiler: pdfLaTeX

Select main.tex at the repository root for the journal article.
Select supplementary.tex at the repository root for its Supplementary
Information. These wrappers use the canonical sources in journal/; they
are not alternative manuscripts. The supplement includes all 36 figures
and the new DendriNet methods and tables.

neurips/ retains the earlier manuscript and its figures for comparison.
It is not the current journal submission.

Only compilation inputs are tracked on main. Source data, figure generators,
experiment code, configurations, tests, provenance, submission forms,
presentations, previews, and compiled reading PDFs remain local and ignored.
The complete tracked research state immediately before this cleanup is
preserved on archive/research-before-overleaf-cleanup-20260918. Older evidence
already omitted from main remains on archive/pre-overleaf-cleanup-20260917.
Do not switch the active research checkout to these branches: inspect them
with git show or use a separate clone, to protect local ignored files.
Future research-code changes need an explicit research-branch commit;
they will not be included automatically when committing this paper branch.

After pulling from GitHub in Overleaf, check the main-document setting and
recompile from scratch. GitHub synchronization is not automatic. If Overleaf
offers a conflict, preserve its changes; do not force-push or replace them.

Fresh-clone builds with latexmk:
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex

The existing research checkout also retains its local journal/Makefile and
generators, including make combined for the combined reading PDF. Those
research/release commands require the local research inputs, not just a
fresh clone of this compilation-only branch.

Overleaf limits editable text to 7 MB per project and 2 MB per file, and the
project file count to 2,000. It does not support Git LFS or submodules:
https://docs.overleaf.com/getting-started/free-and-premium-plans/plan-limits
https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization
