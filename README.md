# Dendritic local-learning papers

This directory keeps the conference and journal versions together while
preserving them as separate working projects:

- `neurips/` contains the submitted NeurIPS project and its rebuttal history.
- `journal/` contains the active journal extension and its independent Git
  history.

Work on the extension from `journal/`. Its inherited NeurIPS panels, source
tables, generator snapshots, and shared plotting style are frozen and audited,
so the two versions can be compared without silently redrawing the conference
figures.

The outer Git repository is the original NeurIPS repository. The journal
directory remains an embedded, independent Git repository with its existing
`local-learning-journal` remote. No histories were merged by this filesystem
reorganization.

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
