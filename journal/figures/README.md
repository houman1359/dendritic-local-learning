# Figure directory

The manuscript has one unambiguous publication-facing figure set.

- `main/figure_01.pdf` through `main/figure_08.pdf` are the only assets compiled
  as main Figures 1--8. Each number has one PDF and one continuous panel
  sequence.
- `supplementary/figure_S01_*.pdf` through
  `supplementary/figure_S24_*.pdf` are the Supplementary Figures.
- `generated/` contains reproducible internal outputs whose descriptive names
  are not manuscript figure numbers.
- Older modular blocks in `main/` are compositor inputs retained for provenance;
  neither LaTeX nor the Overleaf bundle compiles them directly.
- `components/` and `archive_superseded/` contain source or superseded assets.

Run `make canonical-figures` after regenerating plots. It first synchronizes
vector source blocks and then assembles the eight final PDFs. Raster previews
are local build products and are not canonical assets.

## Main figure map

| Figure | Compiled asset | Content |
|---|---|---|
| 1 | `figure_01.pdf` | Coordinates, dendritic addresses, gain and eligibility-by-transport factorization |
| 2 | `figure_02.pdf` | Neuron identity, ownership, two-stream routing and Fashion-MNIST replication |
| 3 | `figure_03.pdf` | Trained subtree-address factorial across feedback bandwidth |
| 4 | `figure_04.pdf` | Credit-operator signal--noise theory and quantitative validation |
| 5 | `figure_05.pdf` | Task-aligned physical depth and point/dendrite controls |
| 6 | `figure_06.pdf` | Reconstructed-arbor route capacity and capture per unit wiring |
| 7 | `figure_07.pdf` | Focal shunting, electrotonic boundary and active-conductance sensitivity |
| 8 | `figure_08.pdf` | Measured-response nulls, imposed-alignment rescue, animal test and phase-plane synthesis |

Supplementary Figures S18--S24 retain the expanded physical-depth controls,
prospective identity/ownership diagnostics, full morphology diagnostics,
focal-shunting controls, measured-response diagnostics and the trained
partition-residual reconstruction and adaptive conductance-reliability test,
respectively. No
evidence-bearing panel removed from the main narrative is discarded.
