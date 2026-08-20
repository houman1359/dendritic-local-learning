# Review implementation matrix

Last audited: 12 August 2026

This matrix separates mathematical implementation, reduced rate-based
validation and end-to-end validation in the original dendritic simulation
stack. A suggestion is not called complete merely because a related toy model
exists.

## Complete and incorporated

| Review item | Evidence | Claim boundary |
|---|---|---|
| General stochastic credit-operator descent theorem | Supplementary Proposition 1; `code/theory/credit_phase.py`; Monte Carlo tests | Fixed operator and unbiased noise at the analyzed state. |
| Point/shared-coefficient partition residual | Supplementary partition proposition; executable weighted residual test | Defines address bias; not yet measured across original training checkpoints. |
| Spectral/Ky--Fan capacity theorem and alignment crossover | Supplementary Ky--Fan proposition; `code/theory/operator_spectral.py`; `source_data/credit_phase_theory/spectral_*` | Dense eigenspace is an oracle; crossover is relative to sampled random routes. |
| Address versus coefficient error decomposition | Supplementary exact Pythagorean decomposition; numerical tests | Does not specify how a biological circuit estimates route coefficients. |
| Static route-gain span invariance | Supplementary proposition; same-span projector gates | Applies to fixed nonzero column gains, not state-dependent row gains or silencing. |
| Fixed-step branch reliability theory | General $a_b^*=\min\{1,S_b/[c(S_b+N_b)]\}$ theorem and tests | Signal/noise energies remain oracle quantities. |
| Corrected positive-conductance reliability experiment | 50 fresh seeds, 2,250 rows in `source_data/positive_conductance_reliability_step_consistent/` | Single-compartment rate reduction with fixed shunts and oracle state clamp. |
| Adaptive local reliability estimation | 50 fresh seeds, 1,050 outcomes in `source_data/adaptive_conductance_reliability/`; global, shuffled, no-shunt, initial-oracle and point-gate controls | Recovers relative placement value but is worse than no shunt; still assumes paired probes, step fraction and exact state clamp. |
| Independent point/additive controls and physical finite difference | Separate update paths; canary and confirmatory gates; tests | Numerical equivalence shows point emulation, not dendrite exclusivity. |
| Same-span noisy coefficient learning | 50 seeds, 4,800 rows in `source_data/same_span_coefficient_learning/` | Rate-based linear field dynamics; Gram preconditioner is an oracle control. |
| Exact finite-time low-data bias--variance theorem | `code/theory/coefficient_learning.py`; Eq. S finite-risk; Monte Carlo tests | Explains the observed falsification/crossover; not a general nonlinear sample-complexity theorem. |
| Passive nonnegative adjoint and focal inverse-norm bound | M-matrix proposition and inverse-operator corollary; executable tests | Passive reciprocal model; spatial selectivity remains Green's-function dependent. |
| Idealized depth transport penalty | Lognormal path-gain cosine theorem and test | Ideal independent gains, not a fit to trained tree checkpoints. |
| Production checkpoint transport audit | 30 aligned-shunting checkpoints; exact-path LocalCA versus autograd and shared-soma gradients on identical held-out batches | Exact path transport is numerically verified; the shared-gradient depth trend is a post-training diagnostic, not a preregistered performance endpoint. |
| Exact-resource nonlinear physical-depth bridge | 270 valid fresh-seed production-model fits; aligned, independent-sensor, shuffled-trial, reversed-placement, additive and LocalCA controls | Fixes $H=3$, one 4/2/2 sensor inventory and one calibrated operating point; it is not the full $D\times H\times K\times\rho$ factorial or a dendrite-exclusive expressivity result. |
| Literal grouped-point and independent H2 replication | 220 frozen fits; direct-to-soma grouped-point emulation, serial BP, shared/path LocalCA, alignment reversal and exact-resource audits | Extends the calibrated synthetic task family to H2 and H3; it is not a second dataset or a full $D\times H\times K\times\rho$ factorial. |
| Fixed-state hierarchy-depth phase | 50 seeds in `source_data/credit_phase_theory/` | Routed linear quadratic depth, not nonlinear forward dendritic depth. |
| Trained route-bandwidth/address factorial | 2,700 fits in `source_data/trained_subtree_address_full_factorial/` | Algebraically matched representations share implementations; supports an address-resource claim. |
| Partition residual in the trained route factorial | Hash-gated deterministic reconstruction of all 2,700 fits at initialization and training; Supplementary Fig. S23 | Address capture explains bandwidth and ownership failures but not matched-capacity support differences; association is post hoc. |
| Irregular-tree wavelet scale analysis | Frozen weighted tree-Haar analysis on the 47-cell disjoint cohort and original eight cells; Supplementary Fig. S25 | Coarse route energy exceeds isotropic noise, but the ancestry-specific excess over column permutation does not replicate; modeled structural fields are not measured credit. |

## Partially implemented; stronger analysis remains compatible

| Review item | Current evidence | Required completion |
|---|---|---|
| Noise hierarchy | Coefficient noise, broadcast noise and several legacy ladders | One common trained design crossing input, label, feedback, branch, multiplicative and context noise. |
| Reactivation/NMDA factors | Legacy reactivation controls and active steady-state sensitivity | A prospective rate-based state-reactivation factorial coupled to local learning. |
| Context gating | Two-context switch and static overlap analyses | Latent-context local gating without an oracle context label. |
| Continual learning | Shallow two-context switch | Trained 8--32-task sequence with retention, interference and route reuse endpoints. |

## Compatible high-priority experiments not yet complete

1. **Full nonlinear physical-depth factorial:** extend the completed H2/H3
   bridge by crossing more forward depths $D$, route budget $K$ and
   alignment/noise $\rho$. Literal grouped-point hierarchies, exact-resource
   reversal, raw additive BP and two LocalCA transports are now complete; a
   BP-plus-projection arm and broader task families remain.
2. **Cable selectivity by reliability dose:** cross shunt location, dose,
   electrotonic background and branch SNR during learning, rather than only
   fixed-state sensitivity or a single-compartment clamp.
3. **Latent context and reactivation:** infer context locally from branch
   state, then test whether reactivation/NMDA-like nonlinearities preserve
   address separation under switching.
4. **Long-horizon continual learning:** 8, 16 and 32 tasks with branch reuse,
   interference, forgetting and recovery endpoints.

## Outside the present rate-based claim unless scope is deliberately expanded

- Spiking-time credit, spike-timing plasticity and kinetic channel learning.
- Cell-specific biophysical fits beyond the existing steady-state sensitivity
  ensemble.
- Claims of endogenous branch teaching signals in vivo without new
  simultaneous teaching-signal and branch-plasticity measurements.

These items are scientifically valuable, but implementing them as small
rate-based surrogates would not satisfy the corresponding biological claim.
