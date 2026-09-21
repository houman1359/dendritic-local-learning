# Fixed-depth task-family by alignment contract

Frozen: 20 August 2026, before any seed 10500--10509 outcome from this
factorial was observed.

## Question and fixed architecture

This experiment asks whether the advantage of serial dendritic composition is
specific to a task whose nuisance factors match that composition, and whether
the same boundary remains visible with restricted local credit. It does not
vary physical depth. Every model has the frozen D3 morphology `[2,1,2]`, the
same 4/2/2 fine/coarse/proximal contact inventory, 64 somata, 16 excitatory and
12 inhibitory contacts per branch, child conductance 16, and the selected
signal/noise operating point from the earlier confirmatory experiment.

The architecture contrast is serial dendritic composition versus the literal
grouped-point emulation (`cross_level_mode=parallel_readout`) with the same
branches, contacts, candidates, parameters, and input tensors.

## Task families

The three prospectively fixed families alter how nuisance measurements compose,
not the label balance, input dimension, signal support, gain marginals, or
sample count.

1. **Nested-factor task:** the distal signal is multiplied by three nested
   nuisance factors and the three inhibitory blocks separately measure those
   factors (`factorized_sensors`, `hierarchical`). This is the positive-control
   family matched to serial divisive cancellation.
2. **Flat-factor task:** the same three-factor product and per-block marginal
   log-gain variance are retained, but nested spatial covariance is replaced by
   unrelated flat groups (`factorized_sensors`, `flat`).
3. **Local-ratio task:** excitation and inhibition at each level share the
   cumulative nuisance visible at that level (`paired_cumulative`,
   `hierarchical`); the class signal remains excitatory and distal-only. This
   favors local ratio readout rather than serial factor cancellation.

For each family, sensor alignment is swept at alpha = 0, 0.5, and 1 while
preserving the inhibitory sensor's marginal variance. Ten paired training seeds
10500--10509 are used at every cell.

## Credit rules and factorial size

Each family-by-alignment-by-architecture cell is trained with (i) exact
backpropagation and (ii) LocalCA with path transport. The full factorial is
3 task families x 3 alignment doses x 2 architectures x 2 credit rules x 10
paired seeds = 360 fits. LocalCA learning-rate settings are inherited without
retuning from the earlier confirmatory cohort.

All result directories are pinned to the `kempner_project_b` allocation on
`holylfs06`. External experiment tracking is disabled: the generated configs
set `use_wandb: false`, and the frozen task runtime uses a local no-op adapter
without importing or contacting W&B.

## Primary estimands

All contrasts are computed within seed.

1. Serial minus grouped-point test accuracy within each task family and
   alignment dose, separately for BP and LocalCA.
2. The alignment interaction: [serial minus grouped point at alpha=1] minus
   the same contrast at alpha=0.
3. The task-specificity interaction: the alignment interaction in the
   nested-factor family minus its counterpart in each mismatch family.
4. The credit interaction: each architecture-by-alignment contrast under
   LocalCA minus the corresponding BP contrast.

Secondary endpoints are test AUC, train--test gaps, intermediate-alpha
curvature, and final-loss/finite-check diagnostics.

## Gates and claim boundary

- all 360 fits must finish with finite metrics and no LocalCA fallback;
- paired cells must match in seed, tensors, contacts, candidate slots,
  trainable parameters, and stopping policy;
- a positive interaction claim requires a seed-bootstrap 95% interval that
  excludes zero and at least eight of ten seed signs in the claimed direction;
- family-level scans are reported in full, with no selective pooling;
- alpha=0 and the two mismatch families remain reportable null/negative
  controls even if they weaken the narrative.

A positive nested-family interaction supports a conditional statement: serial
dendritic composition helps when task covariance and sensor alignment match its
factorization. It does not establish a generic dendritic advantage. Persistence
under LocalCA shows compatibility with the restricted credit rule, not that the
same fixed feedback is implemented biologically.
