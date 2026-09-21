# Independent-animal cortical anatomy contract

Frozen: 20 August 2026, after identifying the public release and before opening
mesh-derived routing outcomes or synapse counts for selected cells.

## Dataset and independence

The replication uses the MICrONS phase-1 layer-2/3 ("Pinky") v185 release,
Zenodo record 3710459. The release is a 250 x 140 x 90 micrometre volume from
visual cortex of a P36 male mouse. It predates and is biologically independent
of the single mouse in the minnie65 cubic-millimetre cohort used in the main
anatomical analysis. The release reports proofread dendrites and axons for the
excitatory neurons with somata in the volume, fixed meshes, and dense putative
synapses.

## Outcome-independent cohort

The eligible pool is every row labelled excitatory (`cell_type=e`) in the
published soma-valence table. Positions are parsed from `pt_position`; the
legacy convenience columns named `soma_*_nm` are not used because the archived
CSV duplicates coordinates in those columns.

Eligible cells are sorted by the second volume coordinate and divided into 12
equal-count strata. Within each stratum, the cell closest to the global median
of the first and third coordinates is selected, with root ID as the tie-break.
This yields a 12-cell sample spanning the volume while reducing lateral/boundary
truncation. Selection uses no mesh topology, synapse count, route-capture, or
electrical endpoint.

## Replication endpoints

The primary analysis reuses the same morphology-aware route dictionary and
equal-rank controls as the minnie65 analysis:

1. morphology-aware ancestry routes minus random anatomical routes;
2. morphology-aware routes minus depth-only bins;
3. morphology-aware routes minus shuffled ancestry;
4. capture per nonzero route coefficient at fixed channel count;
5. structural kernel rank and dendritic path-length summaries.

Incoming synapses are typed only when the presynaptic root is labelled
excitatory or inhibitory in the same published soma-valence table. Untyped
inputs are retained in mapping-QC counts but excluded from E/I conductance
fields. Analyses that require a minimum typed-input count use an outcome-blind
QC gate fixed here: at least 20 mapped typed incoming synapses, including at
least 3 inhibitory inputs. Every selected cell and exclusion reason remains in
the manifest.

## Inferential unit and claim boundary

Cells are nested in one independent animal; the animal, not the 12 cells, is
the biological replication unit. Cell-bootstrap intervals describe within-
volume stability and are not treated as population-level animal inference.
Agreement in sign with minnie65 supports cross-animal reproducibility of the
structural availability result. A null or reversal is reported as such. This
analysis does not test morphology-specific learning or in-vivo use of the
routes.

## Outcome-blind execution clarifications

Before any route-capture outcome was computed, the public archive exposed two
storage details that required deterministic handling. AppleDouble entries
named `._<root>.h5` are ignored in favor of exact `<root>.h5` basenames. Each
real mesh contains detached segmentation fragments, so preprocessing retains
the largest connected component after adding the archived `link_edges` and
chooses the soma-nearest root within that component. Opposite-surface ray
distance remains the frozen caliber definition; its implementation uses the
Embree-backed trimesh intersector (`embreex==2.17.7.post7`) because the triangle
backend has prohibitive memory scaling. Cells run in isolated subprocesses.
These corrections were made from archive structure and resource diagnostics
before any cohort endpoint was available.

## Immutable public inputs

- Release page: https://www.microns-explorer.org/phase1
- Zenodo record: https://doi.org/10.5281/zenodo.3710459
- `soma_valence_v185.csv`
- `pni_synapses_v185.csv`
- `layer23_v185.tar.gz` (fixed per-cell meshes)

Raw downloads remain in ignored `journal/data/pinky_v185/`; checksums,
selection manifest, analysis-ready derivatives, and summaries are archived
under `journal/source_data/pinky_v185_replication/`.
