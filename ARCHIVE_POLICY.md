# Archive policy

1. `journal/` is the only active manuscript. The full repository may be shown
   in Overleaf for provenance and comparison. Root `main.tex` is only an
   Overleaf entry point; `journal/main.tex` remains the sole scientific main
   document.
2. `neurips/` is frozen. It may be read for provenance but must not receive
   scientific edits intended for the current paper.
3. A result inherited from conference-era work belongs in the journal article
   only if it is valid, load-bearing and documented in the evidence ledger.
4. Publication-facing figures use the journal plotting system. Exact historical
   assets may be retained as audited supplementary provenance, but they must be
   labelled as such and must not create competing panel maps.
5. The legacy `local-learning-journal` and `dendritic-credit-routing`
   repositories are not current paper locations. Material migrated from them is
   tracked in `journal/reproducibility/ORIGIN_MANIFEST.tsv`.
6. Superseded drafts, generated PDFs and bulky result tables are recovered from
   Git or external raw archives; they are not stored beside the canonical TeX
   source. The complete snapshot immediately before the Overleaf-size cleanup
   is `archive/pre-overleaf-prune-20260820`.
