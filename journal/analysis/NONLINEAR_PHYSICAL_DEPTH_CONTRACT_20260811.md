# Nonlinear physical-depth experiment contract

Frozen: 11 August 2026, before any journal-specific pilot outcome.

## Question

Can physical dendritic depth become useful for local learning when the forward
task contains nested, branch-local gain factors, rather than the flat MNIST and
noise tasks used for the fixed-contact negative control?

This experiment uses the repository's registered positive-rate
`hierarchical_gain_load` generator and the live `StatefulDendriNet`/
`DendriticBranchLayer` implementation.  It does not use the reduced linear
credit-phase model.  Every excitatory and inhibitory input is strictly
nonnegative, and the shunting denominator is the production branch equation.

## Pilot selection and separation from the gain--load project

The sibling gain--load project previously tested the same task family with
backpropagation, but its full six-cell shunting/additive hierarchy pilot was
blocked because no common clean operating point existed across mechanisms and
morphologies.  Those blocked outcomes are not evidence for this paper.

The journal canary therefore makes a narrower, prespecified shunting-only
test.  The clean calibration archive selected no global six-cell setting, but
at signal delta 0.24 and initial child conductance 16 all three shunting
morphologies were learnable (clean two-seed accuracies 0.7428, 0.7906 and
0.7519).  Those already observed calibration values select the operating
point; all learning comparisons below use fresh seeds.  Additive and point
controls require their own matched-operating-point contract and cannot be
silently pooled with this canary.

## Exact-resource physical morphologies

Each neuron has eight non-somatic branch units and the same 4/2/2 inventory of
fine, coarse and proximal E/I sensors:

| Label | Branch factors | Physical non-somatic depth |
|---|---:|---:|
| D1 | `[8]` | 1 |
| D2 | `[2,3]` | 2 |
| D3 | `[2,1,2]` | 3 |

All cells use 64 somata, 16 excitatory and 12 inhibitory contacts per branch,
no somatic synapses, `mechanism_neutral` axial initialization, softplus-positive
weights and passive identity reactivation.  Shunting itself supplies the
nonlinear divisive forward composition.  The candidate-score count, realized
branch inventory, active feedforward contacts and trainable parameter count
must be identical across D1--D3 before outcomes are interpreted.

## Task regimes

The hierarchy has three latent gain factors.  The canary crosses:

1. `aligned`: hierarchical gain, matched local sensors, alpha=1, correct 4/2/2
   tree placement;
2. `zero_alignment`: identical gain marginals but alpha=0;
3. `sensor_shuffled`: alpha=1 but the inhibitory sensor is trial-shuffled;
4. `rewired_tree`: alpha=1 and matched trials, but the three sensor blocks are
   cyclically reassigned to fine/coarse/proximal branch inventory tiers.

The first comparison asks whether depth is useful under a depth-relevant task.
The other three determine whether any benefit requires task--tree alignment
and locally available sensor information.

## Learning methods

- exact stochastic backpropagation (`standard`);
- three-factor LocalCA with one soma coordinate shared through each tree
  (`per_soma_shared`);
- three-factor LocalCA with exact production path transport
  (`path_transport`).

Path transport is the local-rule transport upper bound, not a claim of a
biological mechanism.  Backpropagation remains the same-model optimization
reference.  A later projected-BP control is required before attributing a
regularization advantage to the local rule rather than the route constraint.

## Seeds and gates

The canary uses two fresh seeds, 10100 and 10101.  It is diagnostic and cannot
enter the manuscript as confirmatory evidence.  Before a ten-seed
confirmatory run is generated, all of the following must hold:

1. every generated configuration validates and completes without NaN or
   fallback transport;
2. all rates and conductance components are finite and nonnegative;
3. D1--D3 have identical active input contacts and branch inventory;
4. every aligned BP cell is above 0.60 and below 0.98 held-out accuracy, so the
   comparison is neither at chance nor at ceiling;
5. both LocalCA modes produce descent in every aligned depth in at least one of
   two seeds;
6. the rewired condition changes only inventory-to-feature assignment, not
   parameter count, contact count, data tensors or marginal input statistics.

Failure of a gate is reported as a negative or inconclusive result.  The
operating point, task strength, optimizer and controls will not be retuned
after reading the canary and then described as confirmatory.

## Prespecified outcomes

Primary descriptive endpoint: held-out classification accuracy paired by seed.

Primary depth contrast in the aligned regime: the best of D2/D3 minus D1,
reported separately for BP, shared LocalCA and exact-transport LocalCA.  This
is a positive-depth screen, not a multiplicity-controlled confirmatory test.

Mechanistic endpoints: depth-wise branch voltage range, total conductance,
activation derivative, exact-versus-local update cosine, path-gain dispersion,
and the aligned-minus-rewired contrast.  A positive interpretation requires a
depth benefit in `aligned` that is reduced by zero alignment, sensor shuffling
or tree rewiring.  If depth benefits BP and exact transport equally, it is a
forward-computation effect.  If it appears only under routed LocalCA, it is a
credit-structure effect and requires the projected-BP control.

## Claim boundary

Even a positive result would show only that production rate-based dendritic
depth can be useful on a task with matched nested gain structure.  A
parameter-matched point or grouped-point network can emulate the computation
if it is explicitly supplied with the same hierarchy.  The scientific claim
is about physical reuse of nested branch-local computation and credit routes,
not an impossibility theorem for point networks.
