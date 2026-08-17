# Nonlinear physical-depth code--theory audit

Audited: 12 August 2026

## Scope

This audit asks whether the new physical-depth experiment implements the
rate-based theory in the same code path as the paper's regular-tree models.
It does.  The experiment is not the reduced quadratic credit-phase simulator:
it trains the registered `hierarchical_gain_load` task through the production
`PopulationNetwork`/`DendriNet` branch dynamics and the production LocalCA
gradient writer.

## Equation-to-code map

| Theoretical object | Production implementation | Audit conclusion |
|---|---|---|
| Positive conductance voltage, $V=(E+C)/(1+E+I+G+\epsilon)$ | `forward_branch_dynamics` in `branch_dynamics.py` | The shunting numerator contains excitatory and child currents; the denominator contains leak, excitation, inhibition and child conductance. All task inputs and learned conductances are nonnegative. |
| Local conductance eligibility, $xR^{\rm tot}(E_{\rm rev}-V)$ | LocalCA signal and TopK-gradient helpers in `local_learning_signals.py` and `local_learning_topk_mixin.py` | The three-factor rule uses the same voltage, total resistance, driving force and realized sparse input as the forward model. |
| Exact directed-tree transport | `_precompute_path_transport_errors` in `local_learning_broadcast_transport_mixin.py`, followed by `_compute_local_voltage_error` in `local_learning_post_factors_mixin.py` | The implementation stores activation-space error. It first transports $e_c^a=e_p^a f'_p(V_p)R_p^{\rm tot}g_{c\to p}$, then multiplies locally by $f'_c(V_c)$. The product is exactly the manuscript's voltage-space recursion $\delta_c^V=\delta_p^V R_p^{\rm tot}g_{c\to p}f'_c(V_c)$. Shape mismatch emits a fallback warning; the confirmatory artifact gate requires zero fallback mentions. |
| Neuron-indexed approximation | `per_soma_shared` in the LocalCA broadcast mixin | One decoder-derived soma coordinate is repeated across that soma's descendants without inserting an oracle branch label. |
| Nested positive-rate task | `create_hierarchical_gain_load_dataset` in `synthetic_datasets.py` | Class signal is placed in the distal excitatory block and multiplied by fine, coarse and global positive lognormal factors. Three inhibitory blocks measure the corresponding factors. Zero-alignment and trial-shuffle controls preserve factor marginals while removing informative pairing. |
| Physical hierarchy | `branch_factors` and `_plan_dendrinet_branch_layout` in `dendrinet.py` | Factors are ordered soma-to-distal. D1 `[8]`, D2 `[2,3]` and D3 `[2,1,2]` create one, two and three nonsomatic stages while retaining eight nonsomatic branch units per soma. |
| Nonlinear composition | Production shunting division at every physical stage | Reactivation is identity, so the only intermediate nonlinearity is the conductance denominator. The experiment tests divisive hierarchical composition, not NMDA/plateau or threshold composition. |

## Exact resource matching

Every D1--D3 configuration has:

- 64 somata and eight nonsomatic branch units per soma;
- the same 4/2/2 fine/coarse/proximal sensor inventory;
- 16 excitatory and 12 inhibitory contacts per branch;
- 14,336 active input synapses and 21,760 candidate synapse slots;
- 66,178 trainable parameters;
- 2,944 persistent state scalars per sample.

Thus physical stage count changes without changing branch-unit, active-contact,
candidate-slot, parameter or persistent-state budgets. The D2 and D3 trees use
nonuniform branching, including one unary D3 stage, specifically to preserve
this resource equality.

The structured-mask implementation was also checked rather than inferred from
the YAML labels. `inventory_feature_blocks` partitions the ordered 4/2/2
inventory over the available nonsomatic stages: D1 receives all three tiers at
one stage, D2 receives fine+coarse distally and global proximally, and D3
receives one tier at each of its distal, middle and proximal stages. The
placement control reverses the fine and global feature ranges while leaving
their 21/22/21 widths, counts, stage sizes and every input tensor unchanged.
An earlier three-cycle changed candidate mask slots and is retained as an
invalidated artifact-only execution.

## Frozen controls

The fresh-seed confirmatory design contains 270 fits:

1. aligned shunting BP at D1--D3;
2. zero-alignment, trial-shuffled sensor and reversed tree-placement controls;
3. a raw additive aligned mechanism control;
4. aligned and rewired-tree LocalCA with shared-soma and exact path transport.

The inference is seed paired. A positive aligned D3--D1 contrast establishes
forward capacity only. Task--tree specificity requires an aligned-minus-control
depth interaction. A LocalCA claim additionally requires its own depth and
rewiring interaction. The raw additive arm tests the divisive mechanism, not
the impossibility of a point implementation.

## Dedicated diagnostics

The generic compartment-statistics renderer was not used because it expects
legacy dense single-cell weight keys and fails after successful population
training. `diagnose_nonlinear_physical_depth.py` instead reloads the actual
population checkpoints and records, on one fixed held-out batch per seed:

- branch voltage, total conductance and input resistance;
- reactivation derivative;
- exact production path gain and log-gain dispersion;
- autograd conductance gradients;
- shared-soma and exact-path LocalCA conductance gradients on the identical
  state and batch;
- gradient cosine and norm ratio against autograd.

This read-only diagnostic was run only after the aligned training cohort had
completed and cannot change any frozen outcome.

All 30 aligned-shunting BP checkpoints passed the replay artifact gates. Exact
path transport reproduced the selected autograd conductance-gradient direction
at every checkpoint (minimum cosine 0.9999999999994). Mean shared-soma cosine
fell from 0.9457 at D1 to 0.7990 at D2 and 0.7674 at D3. The paired D3-minus-D1
change was -0.1784 (95% seed-bootstrap interval -0.2082 to -0.1490; 0/10
positive pairs). This post-training diagnostic validates the implemented path
recursion and shows that a single neuronal coordinate becomes a poorer proxy
for the branch-resolved gradient as physical depth increases; it is not a
separately preregistered performance endpoint.

## What this experiment does not implement

The reviewer proposed a much larger $D\times H\times K\times\rho$ program.
The present confirmatory bridge fixes task hierarchy at $H=3$, uses the model's
fixed 4/2/2 sensor/address inventory, and selects one transparent accessible
operating point after reporting all failed pilots. It does not yet include:

- a full hierarchy-depth or route-budget phase grid;
- backpropagation followed by the same route projection;
- nonlinear reactivation/NMDA/plateau factors;
- latent-context gates generated locally by branch state;
- learned shunt reliability;
- 8--32-task continual-learning sequences;
- a grouped-point hierarchy trained in this same 270-fit cohort.

A grouped point model supplied with the same nested intermediate state can
emulate the computation. The supported claim is therefore conditional and
resource based: matched physical depth can reuse nested branch-local divisive
computations and remain trainable under local credit transport. In the final
270-fit cohort, aligned-minus-reversed D3--D1 interactions were 31.28 points
for BP, 17.72 points for shared-soma LocalCA and 28.68 points for exact-path
LocalCA, all positive in 10/10 paired seeds. It is not an absolute separation
from all point or multilayer networks.
