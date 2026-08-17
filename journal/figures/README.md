# Figure directory

This directory has one publication-facing organization.

- `main/` contains the canonical panel blocks compiled as main Figures 1--8.
- `supplementary/` contains exactly Supplementary Figures S1--S17.
- `generated/` contains internal outputs from the analysis/figure scripts. Their
  historical descriptive names are not manuscript figure numbers.
- `components/` contains source components assembled into a larger figure.
- `archive_superseded/` contains completed displays that are not compiled.

Run `make canonical-figures` after rebuilding any figure. The canonical
filename records the number and panel range shown in the compiled PDF; for
example, `main/figure_06_panels_A-J.pdf` is Figure 6A--J. LaTeX and the Overleaf
bundle refer only to `main/` and `supplementary/`.

## Main figure map

| Figure | Canonical asset(s) | Content |
|---|---|---|
| 1 | `figure_01_panels_A-E` | Framework and evidence roadmap |
| 2 | `figure_02_panels_A-F`, `G-O`, `P-Q` | Feedback coordinates, prospective controls, Fashion-MNIST replication |
| 3 | `figure_03_panels_A-F` | Trained subtree-address factorial |
| 4 | `figure_04_panels_A-I` | Credit-operator phase theory |
| 5 | `figure_05_panels_A-F`, `G-L`, `M-O`, `P-U` | Physical depth and point/dendrite controls |
| 6 | `figure_06_panels_A-J`, `K-L` | MICrONS route capacity and wiring economy |
| 7 | `figure_07_panels_A-I`, `J-M` | Focal shunting, electrotonic boundary and active-channel extension |
| 8 | `figure_08_panels_A-H`, `I-J`, `K-N`, `O` | Measured-response nulls, imposed-alignment rescue and phase-plane synthesis |

The supplementary canonical filenames similarly encode S-number and panel
range. Supplementary Figure S17 contains the external animal-coordinate
consistency test.
