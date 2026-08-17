# Confirmatory nonlinear physical-depth contract

Frozen: 12 August 2026, after the exploratory boundary passed and before any
seed 10200--10209 outcome was observed.

## Selected operating point

The selected production positive-rate task uses signal delta 0.80, matched
per-factor train/test hierarchical log-gain SD 0.25, child conductance 16,
identity reactivation, mechanism-neutral initialization, and the original
divisive shunting equation.
The selection path and every failed exploratory cell remain reported.  The
confirmatory replication uses ten disjoint paired seeds 10200--10209.

## Exact-resource morphologies

D1 `[8]`, D2 `[2,3]`, and D3 `[2,1,2]` each have eight non-somatic branch
units, a 4/2/2 fine/coarse/proximal sensor inventory, 16 excitatory and 12
inhibitory contacts per branch, 64 somata and 66,178 trainable parameters.

## Frozen cohorts

1. Shunting backpropagation: all three depths in aligned, zero-alignment,
   trial-shuffled sensor and reversed-placement tree regimes (120 fits).
2. Raw additive backpropagation: all three depths in the aligned regime, with
   identical resource and mechanism-neutral initialization (30 fits).
3. Shunting LocalCA: all three depths, shared-soma and exact path transport,
   in aligned and rewired-tree regimes (120 fits).

Total: 270 fits.  No optimizer, signal, coupling, route or epoch setting may
be changed after outcomes are observed.

## Primary estimands

All accuracy contrasts are paired by seed.

1. BP physical-depth effect: aligned shunting D3 minus D1.
2. Alignment interaction: the aligned D3-minus-D1 effect minus the same effect
   under zero alignment.
3. Sensor interaction: aligned depth effect minus shuffled-sensor depth effect.
4. Tree-placement interaction: aligned depth effect minus rewired-tree depth
   effect.
5. Local-rule depth effects: aligned D3 minus D1, separately for shared-soma
   and path transport.
6. Local tree-placement interactions for the two transport modes.

Secondary endpoints are D2 contrasts, test AUC, train--test gaps, and aligned
shunting-minus-additive differences by depth.  Additive is a mechanism control,
not a point-neuron impossibility test.

## Gates

- all 270 configurations must complete with finite outcomes, no NaN and no
  LocalCA fallback transport;
- exact parameter/contact/inventory equality must hold within each comparison;
- at least nine of ten aligned BP seeds at every depth must have test accuracy
  above 0.55 and condition means must remain below 0.98;
- any positive claim requires the paired seed-bootstrap 95% interval for the
  corresponding contrast to exclude zero and at least eight of ten seed signs
  to agree;
- topology specificity requires the aligned-minus-control interaction, not a
  positive aligned depth effect alone.

The exploratory coupling interaction is reported descriptively and not pooled
with this confirmatory cohort.  The confirmatory study tests a selected
operating point, not a universal monotone law of depth.

## Claim boundary

A positive outcome supports the claim that, in the live rate-based shunting
implementation, nested physical depth can reuse a matched hierarchy to improve
forward learning and can remain usable under local credit transport.  If raw
additive depth shows the same effect, the benefit is hierarchical computation
rather than shunting-specific.  A parameter-matched grouped-point network can
emulate the hierarchy if explicitly given the same nested state; no result here
proves an absolute dendrite-specific expressivity advantage.

## Artifact-only amendment

The first cyclic tree-placement execution was invalidated before manuscript
inference because its reassignment mapped the 22-coordinate middle block onto
the four-branch tier, increasing candidate mask slots from 21,760 to 22,016.
Trainable parameters and active contacts were unchanged, but this violated the
frozen exact-resource gate. The complete invalid execution remains archived.
The replacement reverses the 21/22/21 feature-block order, which changes
fine-versus-global physical placement while preserving every block width,
inventory count, active contact, candidate slot, trainable parameter and state
scalar. It uses the same paired seeds and otherwise unchanged configuration.
This correction is determined solely by the failed resource gate, not by the
observed performance values.
