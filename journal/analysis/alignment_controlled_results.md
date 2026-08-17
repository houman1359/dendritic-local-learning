# Alignment-controlled learning on reconstructed dendritic routes

## Outcome

The frozen experiment completed on 8 reconstructed MICrONS cells, with 40 nested Monte Carlo streams and 256 exact quadratic-gradient fields per stream.

When 100% of field energy lay in the eight-channel morphology-selected subspace, that dictionary captured 1.000 of exact-gradient energy and supported 1.000 of the norm-matched full-gradient one-step improvement.
When the field was rotated to 0% route alignment, morphology capture was 0.000; the same fixed anatomy no longer provided a useful direction.
At 40% alignment, morphology-selected routes captured 0.400 and supported 0.615 of the norm-matched one-step improvement. Their capture advantage over the three controls ranged from 0.203 to 0.228 and was positive in all 8 reconstructed cells.
Across the three control families, the median within-cell Spearman correlation between initial capture and 20-step loss reduction was 0.990 (cell range 0.969 to 0.996).

The endpoint cell-level capture contrasts were:

- `alignment_0`:
  - morphology minus random paths: -0.135; cell-bootstrap 95% CI [-0.190, -0.086]; positive in 0/8 cells; exact Wilcoxon p=0.0078.
  - morphology minus depth bins: -0.153; cell-bootstrap 95% CI [-0.228, -0.091]; positive in 0/8 cells; exact Wilcoxon p=0.0078.
  - morphology minus ancestry-shuffled paths: -0.152; cell-bootstrap 95% CI [-0.218, -0.095]; positive in 0/8 cells; exact Wilcoxon p=0.0078.
- `alignment_1`:
  - morphology minus random paths: +0.710; cell-bootstrap 95% CI [+0.620, +0.789]; positive in 8/8 cells; exact Wilcoxon p=0.0078.
  - morphology minus depth bins: +0.753; cell-bootstrap 95% CI [+0.688, +0.815]; positive in 8/8 cells; exact Wilcoxon p=0.0078.
  - morphology minus ancestry-shuffled paths: +0.799; cell-bootstrap 95% CI [+0.728, +0.864]; positive in 8/8 cells; exact Wilcoxon p=0.0078.

## Interpretation

This is a causal computational positive control for the paper's conditional claim. With anatomy, channel budget, target energy, and curvature held fixed, rotating exact credit into or out of the anatomical routing span changes both recoverable credit and loss reduction. It complements, but does not override, the measured-response cohort in which morphology was not reliably better than ancestry shuffling.

## Frozen design

For each cell, the conductance-weighted ancestry kernel was reconstructed as `K[n,k] = 1{k is an ancestor of n} * g_I[k]/g_total[k]`. Eight routes were selected before gradient generation by inhibitory fraction times descendant excitatory synaptic-area fraction. Segment coordinates were weighted by the square root of summed mapped excitatory synaptic area.

Let `Q` be an orthonormal basis for those eight weighted route vectors. For each exact gradient, independent unit vectors `u_parallel` in `span(Q)` and `u_orthogonal` in its orthogonal complement were drawn once. At alignment `a`, the tested gradient was `sqrt(a) u_parallel + sqrt(1-a) u_orthogonal`. It therefore had unit energy and exactly fraction `a` of its energy in the morphology-selected span. Directions were paired across all six alignment levels.

The control dictionaries used eight random anatomical paths, eight path-distance quantile bins, or independent row permutations of each morphology-selected route. The last control preserves channel values and nonzero count while breaking ancestry. Random and shuffled dictionaries were redrawn within each nested stream; the same realization received every alignment condition in that stream.

Field capture was the squared norm of the orthogonal projection divided by exact-gradient energy. Optimization used a positive diagonal quadratic curvature spectrum from 0.5 to 1.5, independently permuted across coordinates and fixed across methods and alignments. One-step comparisons matched update norm at 0.1 times exact-gradient norm. Iterative projected descent used learning rate 0.25 for 20 steps. Both were normalized by progress from the corresponding full-gradient update.

Intervals resampled cells and then one nested stream per sampled cell. Exact paired Wilcoxon tests used the eight cell means. Monte Carlo streams and fields were not treated as biological replicates.

## Scope and limitations

- Alignment is imposed by construction; the experiment establishes conditional sufficiency, not that MICrONS anatomy learned or represents these gradients.
- Projection coefficients use the exact gradient and are oracle upper bounds for each feedback family.
- The quadratic objective has fixed anatomy-independent curvature and does not represent a sensory benchmark.
- The eight reconstructed cells are the biological replication units; Monte Carlo streams are nested sensitivity samples.

## Reproduction

```bash
python scripts/run_alignment_controlled_learning.py
```

Machine-readable results are in `source_data/alignment_controlled/`.
