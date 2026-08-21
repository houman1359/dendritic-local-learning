# Asset and provenance map

## Canonical article

Edit `journal/main.tex`. Its title, abstract, Results, Methods, declarations
and main-figure captions are the authoritative scientific text.

Edit `journal/supplementary/supplementary.tex` for Supplementary Information.
The canonical reading copy is `journal/main_with_supplementary.pdf`, which
concatenates the Article and Supplementary Information.

Upload only `journal/submission/Overleaf_Project.zip` to Overleaf. It is built
from an explicit allow-list and contains only the TeX/BibTeX sources and figure
PDFs needed to compile the current paper. Source Data and software are separate
submission archives.

## Archived conference snapshot

`neurips/` is retained only to document the submitted conference version:

- `local_credit_assignment_body.tex`: conference manuscript body;
- `local_credit_assignment.tex`: anonymous NeurIPS wrapper;
- `local_credit_assignment_arxiv.tex`: arXiv wrapper;
- the associated PDFs, source tables, figures and scripts.

The journal article assumes no prior paper. All load-bearing conference-era
results are stated, interpreted and cited directly in `journal/main.tex` or its
Supplementary Information. Do not edit `neurips/` to change the current paper.

## Figure authority

LaTeX may reference only `journal/figures/main/` and
`journal/figures/supplementary/`. Descriptive outputs in
`journal/figures/generated/` are build artifacts. Conference-era source tables
and generator snapshots under `journal/source_data/inherited_neurips/` and
`journal/scripts/inherited_neurips/` establish lineage; they are not a second
publication-facing figure set.

### Canonical claim and figure map

| Display | Scientific role | Principal lineage |
|---|---|---|
| Fig. 1 | Coordinate $\rightarrow$ address $\rightarrow$ gain framework and eligibility $\times$ transported-error factorization | Conference-stage theory, redrawn with the reusable workshop schematic library |
| Fig. 2 | Scalar, neuron-identity, ownership and exact-transport feedback ladder; second-dataset replication | Conference baseline plus prospective identity/ownership and Fashion-MNIST cohorts |
| Fig. 3 | Two-stream credit reversal and the bandwidth-dependent subtree-address factorial | Journal trained-routing experiments |
| Fig. 4 | Credit-operator signal--noise theory, spectral/depth crossovers and predictive validation | Journal theory and 50-seed/2,700-fit phase programme |
| Fig. 5 | Fixed-resource physical depth, grouped-star and literal grouped-point controls, local-credit decomposition | Journal physical-depth, alignment-dose and second-hierarchy programme |
| Fig. 6 | Reconstructed-arbor route capacity, topology controls and capture per feedback wiring | Mature analyses migrated from the routing workspace and extended here |
| Fig. 7 | Passive and active focal shunting, factor freeze and electrotonic boundary | Migrated focal perturbation analysis plus journal calibration and active-channel controls |
| Fig. 8 | Measured-response null, imposed-alignment rescue, complete-tree boundary, animal coordinates and final phase plane | Migrated functional join plus journal controlled and external-data boundary tests |

Supplementary Figs. S1--S3 are frozen, byte-audited conference-stage displays.
S4--S17 preserve full theory, sensitivity and biological controls. S18--S22
contain the dense physical-depth, identity/ownership, morphology, focal and
measured-response diagnostics that were consolidated out of the eight focused
main figures; no scientific panel was silently discarded.

### Routing-workspace migration status

| Component | Canonical status |
|---|---|
| MICrONS morphology reconstruction | Migrated to `journal/code/` and `journal/source_data/` |
| Route-capacity analysis | Migrated; focused in Fig. 6 and detailed in Fig. S20 |
| Focal-shunting analysis | Migrated and extended; Fig. 7 and Fig. S21 |
| Functional MICrONS join | Migrated and extended; Fig. 8 and Fig. S22 |
| Task-derived learning | Migrated and superseded by complete-tree and controlled-alignment versions |
| Gradient-covariance placement | Exploratory; excluded from the current Article |
| DeepST covariance rewiring | Deferred |
| Credit-gated structural plasticity | Deferred future direction |
| Old TMCR manuscript | Archived framing; not an active manuscript |

File-level origins and copied-source hashes are recorded in
`journal/reproducibility/origin_manifest.tsv` and
`journal/source_data/provenance_manifest.tsv`.

## Noncanonical material

Generated staging directories, raw training runs, superseded figures and old
submission ZIPs are ignored or retained outside the active working tree. The
former `journal/theory-paper/`, the legacy `local-learning-journal` repository
and the separate `dendritic-credit-routing` repository are historical or
exploratory records. Git history is the durable archive.
