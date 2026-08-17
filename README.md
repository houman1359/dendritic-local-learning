# Dendritic local-learning papers

This repository keeps the conference and journal versions together as
separate working projects:

- `neurips/` contains the submitted NeurIPS project and its rebuttal history.
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
retained in the legacy `local-learning-journal` repository.

## Common commands

```bash
cd journal
make figures
make paper
make supplement
make theory
make reproducibility
make audit
```

The historical NeurIPS package remains under `neurips/`; see
`neurips/README.md` and `neurips/REPRODUCIBILITY.md` for its original workflow.
