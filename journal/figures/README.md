# Figure directory

The manuscript has one unambiguous publication-facing figure set.

- `main/figure_01.pdf` through `main/figure_09.pdf` are the only assets compiled
  as main Figures 1--9. Each number has one PDF and one continuous panel
  sequence.
- `supplementary/figure_S01_*.pdf` through
  `supplementary/figure_S28_*.pdf` are the Supplementary Figures.
- `generated/` contains reproducible internal outputs whose descriptive names
  are not manuscript figure numbers.
- Older modular blocks in `main/` are compositor inputs retained for provenance;
  neither LaTeX nor the Overleaf bundle compiles them directly.
- `components/` and `archive_superseded/` contain source or superseded assets.

Run `make canonical-figures` after regenerating plots. It first synchronizes
vector source blocks and then assembles the nine final PDFs. Raster previews
are local build products and are not canonical assets.

## Main figure map

| Figure | Compiled asset | Content |
|---|---|---|
| 1 | `figure_01.pdf` | Coordinates, dendritic addresses, gain and eligibility-by-transport factorization |
| 2 | `figure_02.pdf` | Neuron identity, ownership, two-stream routing and Fashion-MNIST replication; exactness audits are in the SI |
| 3 | `figure_03.pdf` | Trained subtree-address factorial across feedback bandwidth |
| 4 | `figure_04.pdf` | Credit-operator signal--noise theory, quantitative validation and alignment-by-bandwidth synthesis |
| 5 | `figure_05.pdf` | Task-aligned physical depth and decisive point/dendrite controls at H2--H3 |
| 6 | `figure_06.pdf` | H4 depth saturation plus the fixed-depth task-family and alignment boundary |
| 7 | `figure_07.pdf` | Reconstructed-arbor route capacity, wiring efficiency and independent-animal direction |
| 8 | `figure_08.pdf` | Focal shunting with the electrotonic boundary emphasized and active-conductance sensitivity |
| 9 | `figure_09.pdf` | Measured-response nulls, imposed-alignment rescue, animal test and complete-tree boundary |

Supplementary Figures S18--S28 retain the expanded physical-depth controls,
prospective identity/ownership diagnostics, full morphology diagnostics,
focal-shunting controls, measured-response diagnostics and the trained
partition-residual reconstruction and adaptive conductance-reliability test,
followed by the irregular-tree multiscale analysis, immutable-source H2/H3
replication, independent-animal Pinky routing analysis and physical-depth
credit-coordinate diagnostics. No
evidence-bearing panel removed from the main narrative is discarded.
