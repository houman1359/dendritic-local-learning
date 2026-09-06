# Ancestry dictionaries beyond a common broadcast

Adding the same budgeted broadcast to every dictionary repairs the missing-common-mode comparison. Ancestry routes capture additional spatial variation beyond random routes, depth bins, row-shuffled routes, and degree/depth-preserving surrogate trees in the original cells and in both follow-up cohorts. The advantage over the surrogate tree is smaller than the advantage over depth bins or row shuffles: coarse ancestry explains substantial capacity, with an additional contribution from the original branch relations.

This is a computational capacity result on reconstructed anatomy. The target is the reciprocal passive-cable response to modeled focal shunts, not a measured learning signal or a task-dependent credit covariance. It supports the paper's dictionary narrative without establishing endogenous use or an optimal morphology.

## Design and chronology

The protocol and input hashes were frozen at 2026-09-06T12:35:10.759954+00:00, before the new reciprocal-cable endpoints were evaluated in v661 or Pinky. The original eight-cell augmentation was post hoc development. Both follow-up cohorts had previously been used for route-generated positive controls; these are frozen follow-up analyses on existing cohorts, not newly sampled animals or externally preregistered experiments.

All datasets are cached analysis-ready segment tables; no download, training, or alteration of earlier numerical sources was required. The 47 v661 cells exclude the eight original cells by stable nucleus ID, but come from the same MICrONS mouse. Pinky is the separately documented second animal. Its inherited QC retains 10 of 12 selected cells; 8 have enough excitatory and inhibitory-bearing sites for the fixed K=8 comparison. The other two QC-passing cells remain in their feasible lower-budget results. No cohorts are pooled.

## What is compared

Every K-channel dictionary contains one all-site constant and K−1 spatial patterns. The primary budget is K=8; K=1, 2, 4, and 16 are descriptive when site counts permit them. Ancestry patterns use the existing morphological leverage order, without inspecting the response. Random routes sample inhibitory sites; row shuffles preserve every column's nonzero count; surrogate trees preserve node depths and parent out-degrees while reassigning parent-child relationships. The depth dictionary is reparameterized as a constant plus K−1 of the existing K bins, preserving exactly the original depth-bin span. The oracle is constrained to contain the same constant and uses K−1 singular vectors of the remaining response.

For response matrix H and diagonal site weights W, the weighted response is Y=W^(1/2)H. H contains the finite-difference change in the logarithm of the absolute excitatory synaptic gradient for each focal shunt, with soma voltage re-clamped after perturbation. W is proportional to excitatory synapse size. The normalized weighted constant q defines the common component qqᵀY and the spatial residual R=(I−qqᵀ)Y. Total capture is ||P_D Y||²_F/||Y||²_F; residual capture is ||P_D R||²_F/||R||²_F, where P_D projects onto the weighted dictionary span and the squared Frobenius norm sums squared entries. Thus total capture = broadcast capture + (1−broadcast capture) × residual capture. Both quantities are squared-energy fractions, not one minus an unsquared residual norm.

## Eight-channel outcomes

Each table entry is the mean across cells after averaging 200 randomized control dictionaries within each cell. The constrained oracle is a ceiling, not a learned biological implementation.

| Cohort | Dictionary | Total capture | Residual capture | Mean rank | Nonzero density |
|---|---|---:|---:|---:|---:|
| Original eight (post hoc development) | common + ancestry | 0.5854 | 0.4082 | 8.000 | 17.25% |
| Original eight (post hoc development) | common + random routes | 0.4968 | 0.2783 | 7.814 | 19.71% |
| Original eight (post hoc development) | common + depth bins | 0.4446 | 0.1966 | 8.000 | 23.30% |
| Original eight (post hoc development) | common + shuffled routes | 0.4265 | 0.1726 | 7.797 | 17.25% |
| Original eight (post hoc development) | common + surrogate ancestry | 0.5414 | 0.3453 | 7.994 | 23.97% |
| Original eight (post hoc development) | common-constrained SVD | 0.6930 | 0.5687 | 8.000 | 100.00% |
| Disjoint v661 cells (same mouse) | common + ancestry | 0.6191 | 0.5267 | 7.702 | 18.75% |
| Disjoint v661 cells (same mouse) | common + random routes | 0.4040 | 0.2550 | 5.663 | 17.79% |
| Disjoint v661 cells (same mouse) | common + depth bins | 0.5115 | 0.3916 | 8.000 | 23.19% |
| Disjoint v661 cells (same mouse) | common + shuffled routes | 0.4358 | 0.2961 | 7.648 | 18.75% |
| Disjoint v661 cells (same mouse) | common + surrogate ancestry | 0.5782 | 0.4759 | 7.616 | 20.80% |
| Disjoint v661 cells (same mouse) | common-constrained SVD | 0.7422 | 0.6807 | 8.000 | 100.00% |
| Pinky (second mouse) | common + ancestry | 0.9809 | 0.9699 | 6.000 | 22.94% |
| Pinky (second mouse) | common + random routes | 0.7949 | 0.6959 | 4.202 | 19.04% |
| Pinky (second mouse) | common + depth bins | 0.7003 | 0.5412 | 8.000 | 22.61% |
| Pinky (second mouse) | common + shuffled routes | 0.6911 | 0.5301 | 5.912 | 22.94% |
| Pinky (second mouse) | common + surrogate ancestry | 0.8804 | 0.8259 | 5.986 | 24.92% |
| Pinky (second mouse) | common-constrained SVD | 0.9968 | 0.9951 | 8.000 | 100.00% |

Equal K is not equal wiring or actual rank. The strongest directly cost-matched contrast is ancestry versus row shuffling: those dictionaries have identical nonzero counts in every realization, although their ranks can differ slightly. Depth has greater or equal rank than selected ancestry at K=8. Random routes often waste columns on zero or redundant supports, especially in sparsely typed Pinky anatomy. Consequently the random-route gap should not be presented as a rank-matched effect.

The original broadcast captures 0.3138 of total energy, while the leading unconstrained mode captures about 0.326: the approximately 96% ratio concerns the leading-mode ceiling, not total-field capture. The new comparison asks how much spatial variation remains explainable after that common mode is supplied equally.

## Paired ancestry contrasts

Intervals are 95% cell-bootstrap intervals. Cells are nested within one animal per cohort, so these quantify within-cohort consistency and do not establish an animal-population effect. Holm adjustments cover four controls separately for each endpoint and cohort at K=8. Other budgets are descriptive.

| Cohort | Control | Total-capture gain (pp), 95% CI | Residual-capture gain (pp), 95% CI | Cells positive |
|---|---|---:|---:|---:|
| Original eight (post hoc development) | common + random routes | 8.86 [7.53, 10.42] | 12.99 [11.16, 14.97] | 8/8 |
| Original eight (post hoc development) | common + depth bins | 14.08 [11.75, 17.03] | 21.16 [16.59, 26.44] | 8/8 |
| Original eight (post hoc development) | common + shuffled routes | 15.90 [14.27, 18.03] | 23.56 [20.37, 27.03] | 8/8 |
| Original eight (post hoc development) | common + surrogate ancestry | 4.40 [2.02, 6.72] | 6.30 [3.07, 9.39] | 7/8 |
| Disjoint v661 cells (same mouse) | common + random routes | 21.50 [18.25, 24.98] | 27.17 [22.86, 31.79] | 46/47 |
| Disjoint v661 cells (same mouse) | common + depth bins | 10.76 [7.69, 13.46] | 13.51 [9.86, 16.84] | 45/47 |
| Disjoint v661 cells (same mouse) | common + shuffled routes | 18.32 [15.52, 21.16] | 23.06 [19.46, 26.60] | 46/47 |
| Disjoint v661 cells (same mouse) | common + surrogate ancestry | 4.08 [1.99, 6.18] | 5.08 [2.44, 7.68] | 38/47 |
| Pinky (second mouse) | common + random routes | 18.61 [10.08, 28.94] | 27.40 [16.14, 39.81] | 8/8 |
| Pinky (second mouse) | common + depth bins | 28.07 [20.13, 36.75] | 42.87 [30.84, 56.51] | 8/8 |
| Pinky (second mouse) | common + shuffled routes | 28.98 [23.76, 34.84] | 43.99 [37.32, 51.12] | 8/8 |
| Pinky (second mouse) | common + surrogate ancestry | 10.05 [4.89, 16.99] | 14.40 [7.76, 22.47] | 8/8 |

Pinky's near-saturated capture should be interpreted in light of its small directly typed spatial domain: the eight K=8 cells have only 9–13 excitatory-bearing segments, and ancestry dictionaries have mean rank 6. These values should not be pooled with the denser original or v661 cohorts. The sign of the ordering reproduces across two animals; its magnitude is strongly influenced by the sampled anatomy and typed-input coverage.

## Figure captions

**Capacity figure.** Every dictionary receives an identical constant broadcast within the stated total channel budget. A–C, total weighted squared-energy capture in the original eight cells, 47 disjoint v661 cells from the same mouse, and eligible Pinky cells from a second mouse. D–F, capture of the weighted spatial residual after removing that broadcast, in the same cohort order. Curves are cell means; random, shuffled and surrogate controls average 200 draws per cell. At K=8, cohort sizes are 8, 47 and 8. At K=16 the v661 cohort has 46 eligible cells; Pinky has none. Pinky includes 10 inherited-QC cells at K=1, 2 and 4. Colors and markers identify dictionary families throughout; the SVD oracle contains the same broadcast and is not a learned encoder.

**Controls and cost figure.** A, paired ancestry advantages in spatial-residual capture at K=8, with 95% cell-bootstrap intervals, against each spatial control. B, mean nonzero wiring density of the realized K-column dictionaries. Cohort colors and markers agree between panels. A constant column already uses 12.5% of dense K=8 wiring. Ancestry and row-shuffled routes have identical nonzero counts. All intervals describe cell-level consistency within a cohort. Numerical ranks and individual-cell results are supplied separately.

## Reproduction and validation

From the journal directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/anatomy_commonmode/run.py --freeze-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/anatomy_commonmode/run.py --cohort original8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/anatomy_commonmode/run.py --cohort v661
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/anatomy_commonmode/run.py --cohort pinky
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/anatomy_commonmode/build_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/anatomy_commonmode/validate_and_report.py
python -m pytest scripts/anatomy_commonmode/test_metrics.py -q
```

Saved-outcome validation passed for 65 operators and 149,331 dictionaries, including 148,200 randomized realizations. Checks include frozen input hashes, all original ancestry-capture values against the earlier tables, weighted common/residual energy identities, constrained-oracle ceilings, depth-span equivalence, realized rank bounds, full coverage, exact ancestry/shuffle nonzero matching, and reconstruction of cell summaries from saved individual control draws. Five separate numerical unit tests pass.

The raw response operators, per-draw tables, inclusion records, cell summaries, cohort summaries, paired intervals, frozen protocol and two vector figure PDFs are all retained under this directory.
