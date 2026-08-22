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

## Final panel sequence

This is the panel lettering used by `main.tex`; modular source letters are not
publication letters.

- **Figure 1:** A point/tree comparison; B network layer; C coordinate,
  address and gain; D eligibility times transported error; E evidence path.
- **Figure 2:** A MNIST feedback; B gradient alignment; C identity gain across
  depth; D ownership; E credit reversal; F Fashion-MNIST ladder; G paired
  bottleneck contrasts. The coordinate/address distinction is defined once in
  Figure 1 rather than repeated as a banner.
- **Figure 3:** A compact route-resolution key; B learning
  across bandwidth; C best-control contrast; D task--topology alignment;
  E representation match; F capture and learning.
- **Figure 4:** A credit-operator utility schematic; B spectral alignment;
  C predictive utility; D route-resolution crossover; E projection boundary;
  F reliability gains; G alignment-by-bandwidth synthesis.
- **Figure 5:** A matched physical-depth inventory; B nested divisive task;
  C backpropagation depth test; D primary contrasts; E LocalCA transport;
  F divisive control; G architecture controls; H serial composition.
- **Figure 6:** A aligned H4; B reversed H4; C H4 contrasts; D optimum across
  task depth; E BP task-family boundary; F LocalCA boundary; G
  architecture-by-alignment interaction.
- **Figure 7:** A mapped reconstruction; B ancestry addresses; C reciprocal-
  cable field; D sparse capacity; E eight-channel efficiency;
  F capture per wire; G independent-animal direction.
- **Figure 8:** A matched focal-shunt design; B relation selectivity; C passive
  dose response; D adjoint transport; E electrotonic boundary; F active dose
  response; G cellwise contrast.
- **Figure 9:** A structure--function null; B task-field capture; C held-out
  learning; D imposed alignment; E signed six-animal contrast; F full-tree
  learning; G anatomy boundary.

Supplementary Figures S18--S28 retain the expanded physical-depth controls,
prospective identity/ownership diagnostics, full morphology diagnostics,
focal-shunting controls, measured-response diagnostics and the trained
partition-residual reconstruction and adaptive conductance-reliability test,
followed by the irregular-tree multiscale analysis, immutable-source H2/H3
replication, independent-animal Pinky routing analysis and physical-depth
credit-coordinate diagnostics. No
evidence-bearing panel removed from the main narrative is discarded.
