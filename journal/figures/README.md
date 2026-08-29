# Figure directory

The manuscript has one unambiguous publication-facing figure set.

- `main/figure_01.pdf` through `main/figure_09.pdf` are the only assets compiled
  as main Figures 1--9. Each number has one PDF and one continuous panel
  sequence.
- The exact Supplementary Figure assets are the 29 files referenced by
  `supplementary/supplementary.tex`; the submission and Overleaf builders verify
  this list before packaging it.
- `generated/` contains reproducible internal outputs whose descriptive names
  are not manuscript figure numbers.
- Wider-span blocks in `main/` and `supplementary/` are compositor inputs
  retained for reproducibility. They are not manuscript figures, and neither
  LaTeX nor the Overleaf bundle includes them directly.
- `components/` and `archive_superseded/` contain source or superseded assets.

Run `make canonical-figures` after regenerating plots. It first synchronizes
vector source blocks and then assembles the nine final PDFs. Raster previews
are local build products and are not canonical assets.

## Main figure map

| Figure | Compiled asset | Content |
|---|---|---|
| 1 | `figure_01.pdf` | Point-to-dendrite gradient factorization and the coordinate--address--gain framework |
| 2 | `figure_02.pdf` | Standard-task feedback ladder: scalar, neuron-specific and compartment-resolved learning signals |
| 3 | `figure_03.pdf` | Credit-operator signal--noise theory, quantitative validation and alignment-by-bandwidth synthesis |
| 4 | `figure_04.pdf` | Context-gated branch conflict and the predicted boundary at which branch-specific feedback is required |
| 5 | `figure_05.pdf` | Hierarchical partial addressing, feedback bandwidth and matched subtree/topology controls |
| 6 | `figure_06.pdf` | Unified serial physical-depth boundary across task family, hierarchy and sensor alignment |
| 7 | `figure_07.pdf` | Reconstructed-arbor route capacity, field capture, wiring efficiency and second-animal direction |
| 8 | `figure_08.pdf` | Focal shunting and the conductance-state boundary for spatially selective route gain |
| 9 | `figure_09.pdf` | Measured-response pipeline and null, imposed-alignment rescue, animal contrast and evidence boundary |

## Final panel sequence

This is the panel lettering used by `main.tex`; the native builder number and
compiled figure number are identical.

- **Figure 1:** A point/tree comparison; B network layer; C coordinate,
  address and gain; D eligibility times transported error; E Results roadmap.
- **Figure 2:** A task and the three feedback resolutions; B MNIST ladder;
  C Fashion-MNIST ladder; D gradient alignment; E signal-to-arbor assignment
  versus within-tree address; F paired bottleneck contrasts.
- **Figure 3:** A credit-operator utility schematic; B spectral alignment;
  C route-resolution crossover; D projection boundary; E reliability gains;
  F predictive utility; G alignment-by-bandwidth synthesis.
- **Figure 4:** A compatible/conflicting task schematic; B forward selection
  versus backward credit; C analytic shared-mode boundary; D trained
  transition; E conflict-dependent benefit; F implementation controls.
- **Figure 5:** A eight-context hierarchy; B feedback-bandwidth ladder;
  C route-assignment control; D topology and basis controls; E learning across
  bandwidth; F paired route effects; G matched-versus-rewired topology.
- **Figure 6:** A nested, flat and local-ratio task families; B architecture
  controls; C H=3 quantitative boundary; D H=4 aligned/reversed factorial;
  E hierarchy-by-physical-depth saturation; F backpropagation and G local-rule
  alignment dose responses.
- **Figure 7:** A mapped reconstruction; B subtree projection and field-capture
  definition; C reciprocal-cable field; D model-derived field; E eight-channel
  efficiency; F wiring-normalized capture; G independent-animal direction.
- **Figure 8:** A matched focal-shunt design; B relation selectivity; C passive
  dose response; D exact factor-freeze test; E active dose response;
  F electrotonic boundary; G cellwise contrast.
- **Figure 9:** A measured-response pipeline; B complete-tree learning;
  C standardized topology effects; D imposed-alignment design; E controlled
  rescue; F signed six-animal contrast with the signed/common-mode energy
  partition inset; G evidence boundary.
Supplementary Figures S18--S29 retain the expanded physical-depth controls,
prospective feedback/assignment diagnostics, full morphology diagnostics,
focal-shunting controls, measured-response diagnostics and the trained
partition-residual reconstruction and adaptive conductance-reliability test,
followed by the irregular-tree multiscale analysis, immutable-source H2/H3
replication, independent-animal Pinky routing analysis, physical-depth
credit-coordinate diagnostics and the complete path-demand task and boundary.
No
evidence-bearing panel removed from the main narrative is discarded.
