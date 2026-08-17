# Dendritic local-learning papers

## Canonical manuscript map

- **Current journal source:** `journal/main.tex`
- **Current complete reading copy:** `journal/main_with_supplementary.pdf`
- **Current Overleaf upload:** `journal/submission/Overleaf_Project.zip`
- **Frozen NeurIPS source:** `neurips/local_credit_assignment_body.tex`, with
  `local_credit_assignment.tex` and `local_credit_assignment_arxiv.tex` as
  wrappers.

No other TeX file or submission bundle is an active paper. See
`MANUSCRIPT_MAP.md` for the complete edit/build/archive rules.

This repository keeps the conference and journal versions together as
separate working projects:

- `neurips/` contains the submitted NeurIPS manuscript, canonical figures and
  reproducibility scripts; scratch and rebuttal artifacts remain in Git history
  and the recoverable local archive.
- `journal/` contains the active journal extension, including its manuscript,
  figure generators, canonical figures, compact source data and submission
  checks.

Work on the extension from `journal/`. Its inherited NeurIPS panels, source
tables, generator snapshots, and shared plotting style are frozen and audited,
so the two versions can be compared without silently redrawing the conference
figures.

The repository originated with the NeurIPS project. The journal directory is
now tracked directly here so a clone contains both complete paper projects and
their comparison/audit material. The journal's earlier independent history is
retained in the legacy `local-learning-journal` repository. Historical staging
trees, raw runs and superseded manuscript variants stay outside the active
working tree; Git history remains the permanent archive.

## Common commands

```bash
cd journal
make figures
make paper
make supplement
make reproducibility
make audit
```

The historical NeurIPS package remains under `neurips/`; see
`neurips/README.md` and `neurips/REPRODUCIBILITY.md` for its original workflow.
