# Dendritic local learning

## Canonical manuscript map

- **Canonical scientific source:** `journal/main.tex`
- **Canonical complete reading copy:** `journal/main_with_supplementary.pdf`
- **Canonical Overleaf upload:** `journal/submission/Overleaf_Project.zip`
- **Archived NeurIPS snapshot:** `neurips/local_credit_assignment_body.tex`, with
  `local_credit_assignment.tex` and `local_credit_assignment_arxiv.tex` as
  wrappers.

No other TeX file or submission bundle is an active paper. The journal article
is a standalone account that includes the load-bearing conference-era theory,
experiments and controls; it is not written as a sequel. See
`ASSET_AND_PROVENANCE_MAP.md` and `ARCHIVE_POLICY.md` for the edit, build and
archive rules.

This repository keeps the current article and its conference snapshot together
for provenance:

- `neurips/` is a frozen record of the submitted NeurIPS manuscript, figures and
  reproducibility scripts. It must not be used as a source of current claims.
- `journal/` contains the active standalone article, including its manuscript,
  figure generators, canonical figures, compact source data and submission
  checks.

Work only from `journal/`. Conference-era results used by the article are
integrated into the journal narrative and regenerated in the journal figure
system where possible. Frozen source tables, generator snapshots and hashes
remain available solely to audit provenance.

The repository originated with the NeurIPS project. The journal directory is
now the single scientific authority. The earlier independent
`local-learning-journal` repository and the `dendritic-credit-routing` working
repository are historical development records, not competing manuscripts.
Historical staging trees, raw runs and superseded variants stay outside the
active working tree; Git history remains the permanent archive.

## Common commands

```bash
cd journal
make figures
make paper
make supplement
make reproducibility
make audit
```

The archived NeurIPS package remains under `neurips/`; see
`neurips/README.md` and `neurips/REPRODUCIBILITY.md` for its original workflow.
